"""
oemof energy system model (rebuild) — successor of ``spice_ev/oemof.py``.

This file is rebuilt step by step for understanding. STEP 1 only contains the
skeleton: configuration + the ``EnergySystemModel`` interface and its ``run()``
pipeline with stubbed (not yet implemented) stages. The actual model logic is
added in later steps.

Architecture (baseline; may evolve in Step 2)
---------------------------------------------
The topology is derived from the spice_ev scenario and built CONDITIONALLY,
depending on which components exist (see ``_create_components``).

Buses
- bus_home            home AC bus (always): grid supply, household load, wallboxes.
- bus_pv              PV bus (only if PV generation exists): -> home via
                      converter_pv_to_home, grid feed-in via excess.
- bus_battery         DC bus of the stationary home battery (only if present),
                      coupled to bus_home via link_home_battery.
- bus_mobility_<vid>  one DC bus per vehicle (BEV storage + wallbox).

Components
- grid_supply / household_demand   always at bus_home.
- pv / excess_electricity          only at bus_pv (only if PV).
- home_battery + link_home_battery only if a stationary battery exists.
- per vehicle: wallbox_charge_<vid> (home->mobility), wallbox_discharge_<vid>
  (mobility->home, V2H, only with v2g), bev_battery_<vid> (driving demand as
  fixed_losses_absolute).

Inputs (__init__)
-----------------
In the production path all six are built by ``OemofSolve.build_oemof_inputs``
(spice_ev.strategies.oemof_solve) from the spice_ev scenario; standalone runs use
``SystemConfig.from_cfg_file`` / synthetic data (see ``main``).
- config (SystemConfig)  efficiencies/costs/solver + default/fallback values.
- timeseries_df          columns PV_kW/Load_kW on time_index.
- time_index             the shared grid (= spice_ev steps).
- vehicle_params         per vehicle: capacity/SOC/v2g + at_home/consumption series.
- grid_power             grid connection limit (kW); fallback grid_supply_power_kW.
- battery_params         per stationary home battery (capacity/power/SOC/efficiency).

Pipeline
--------
``run()`` orchestrates: load/validate data -> create time index -> build energy
system -> build components -> optimize -> solve -> extract results -> save.
``get_wallbox_schedule()`` returns the per-vehicle wallbox power (charge/V2H) back
to the spice_ev simulation.
"""

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from oemof.solph import EnergySystem, buses, components as cmp, flows

# Added in later steps when the stages need them:
#   Step 3:      from oemof.solph import Model, processing
#   Step 3:      from pyomo.opt import SolverStatus, TerminationCondition


###########################################################################
# Configuration
###########################################################################
# We collect all settings in one typed object, SystemConfig. It can be filled
# two ways:
#   - from a flat dict  -> comes from the OemofSolve strategy (oemof_config)
#   - from a cfg file   -> for standalone runs
@dataclass
class SystemConfig:
    """Configuration parameters for the energy system (one typed settings object).

    Baseline field set (mirrors the cleaned ``spice_ev/oemof.py``); will be
    trimmed/extended in Step 2 once the model topology is fixed. Each field is
    ``name: type = default`` — the type is documentation, Python does not enforce it.
    """

    # Time parameters
    start_date: str = "2025-01-01"
    periods: int = 96  # 15-minute steps (96 = 1 day for debug)
    freq: str = "15min"

    # Feature toggles (force a component off even if its data is present)
    enable_pv: bool = True
    enable_pv_to_home: bool = True  # build the PV->home converter (else PV only feeds in)
    enable_battery: bool = True
    enable_vehicles: bool = True
    enable_grid_feedin: bool = True  # allow home/BEV surplus to be exported to the grid

    # System parameters
    grid_supply_power_kW: float = 30.0

    # Converter
    converter_pv_to_home_power_kW: float = 10.0
    converter_pv_to_home_efficiency: float = 1.0
    converter_home_to_battery_efficiency: float = 0.96
    converter_battery_to_home_efficiency: float = 0.96
    converter_pv_to_home_variable_costs: float = 0.0

    # Stationary battery storage
    battery_capacity_kWh: float = 10.2
    battery_min_soc: float = 0.1
    battery_max_soc: float = 1.0
    battery_initial_soc: float = 0.5
    battery_efficiency: float = 1.0
    battery_max_power_kW: float = 10.0

    # BEV (default/fallback; overridden per vehicle from the master data)
    bev_capacity_kWh: float = 77.0
    bev_min_soc: float = 0.2
    bev_max_soc: float = 0.95
    bev_initial_soc: float = 0.95
    bev_discharge_limit: float = 0.5  # min SOC for V2H/V2G discharge (spice_ev VehicleType default)

    # Wallbox
    wallbox_power_kW: float = 11.0
    wallbox_efficiency_charge: float = 0.97
    wallbox_efficiency_discharge: float = 0.97
    enable_v2h: bool = True

    # Costs (ct/kWh)
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 35.0
    grid_feedin_tariff: float = -8.0  # negative = revenue

    # Solver
    solver: str = "cbc"
    solver_verbose: bool = False
    debug: bool = True
    solver_threads: int = 8
    solver_ratio_gap: float = 0.01

    # Result storage
    should_dump_results: bool = True
    dump_filename: str = "dump"

    @classmethod
    def from_options(cls, options: Optional[Dict[str, Any]] = None) -> "SystemConfig":
        """Build a SystemConfig from a flat ``{field: value}`` dict.

        This is the interface used by ``OemofSolve`` (it passes ``oemof_config``).
        Keys may carry an ``oemof_`` prefix; unknown keys are skipped (warning).
        Values are expected to already have the right Python type.
        """
        config = cls()                                  # start with all defaults
        valid_fields = {f.name for f in fields(cls)}    # set of allowed field names
        for key, value in (options or {}).items():
            name = key.removeprefix("oemof_")           # "oemof_solver" -> "solver"
            if name in valid_fields:
                setattr(config, name, value)            # config.<name> = value
            else:
                logging.warning("Unknown oemof parameter ignored: %s", key)
        return config

    @classmethod
    def from_cfg_file(cls, path) -> "SystemConfig":
        """Build a SystemConfig from a flat ``key = value`` cfg file (standalone).

        One parameter per line; ``#`` comments and blank lines are skipped. Values
        are parsed via ``json.loads`` (so ``96`` -> int, ``true`` -> bool, ``1.5``
        -> float); non-JSON tokens like ``cbc`` / ``15min`` stay strings. Then it
        delegates to ``from_options`` (no duplicated logic).
        """
        options: Dict[str, Any] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, raw = line.partition("=")
            if not sep:
                continue  # not a key=value line
            try:
                value = json.loads(raw.strip())
            except ValueError:
                value = raw.strip()
            options[key.strip()] = value
        return cls.from_options(options)


def _as_array(values, n):
    """Coerce a scalar/series/array to a float array of length n (pad with 0 / truncate).

    A scalar is broadcast to all n steps (e.g. default at_home=1.0 -> always home).
    """
    if np.isscalar(values):
        return np.full(n, float(values))
    arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr[:n]


###########################################################################
# Main class
###########################################################################
class EnergySystemModel:
    """Models and optimizes the energy system (PV + home battery + N BEVs).

    STEP 1: only the interface and the ``run()`` pipeline are present; the stages
    raise ``NotImplementedError`` and are filled in later steps.
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        timeseries_df: Optional[pd.DataFrame] = None,
        time_index: Optional[pd.DatetimeIndex] = None,
        vehicle_params: Optional[Dict[str, Dict[str, Any]]] = None,
        grid_power: Optional[float] = None,
        battery_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Store the inputs. See the module docstring for where each comes from."""
        # --- Inputs ---
        self.config = config or SystemConfig()
        self._timeseries_df_input = timeseries_df
        self._time_index_input = time_index
        self.vehicle_params: Dict[str, Dict[str, Any]] = dict(vehicle_params or {})
        self.grid_power = grid_power
        self.battery_params: Dict[str, Dict[str, Any]] = dict(battery_params or {})

        # --- Outputs, populated by the pipeline (Steps 2-4) ---
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.es = None              # oemof EnergySystem (Step 2)
        self.model = None           # oemof Model (Step 3)
        self.df_timeseries: Optional[pd.DataFrame] = None
        self._results_main = None   # processing.results(model) (Step 4)
        self._b_home = None         # bus_home reference (Step 2)
        self._vehicle_nodes: Dict[str, Dict[str, Any]] = {}  # per-vehicle nodes (Step 2)

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Run the full pipeline: load -> build -> solve -> extract -> save."""
        self._load_data()
        self._create_time_index()
        self._create_energy_system()
        self._create_components()
        self._optimize()
        self._solve()
        self._extract_results()
        self._save_results()

    # ------------------------------------------------------------------
    # Stages (stubs — implemented in later steps)
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Validate the provided timeseries DataFrame into ``self.df_timeseries``."""
        raise NotImplementedError("Step 2: _load_data")

    def _create_time_index(self) -> None:
        """Set ``self.time_index``: adopt the provided one, else build from config."""
        raise NotImplementedError("Step 2: _create_time_index")

    def _create_energy_system(self) -> None:
        """Create the oemof ``EnergySystem`` on ``self.time_index`` (the backbone)."""
        raise NotImplementedError("Step 2: _create_energy_system")

    def _create_components(self) -> None:
        """Build buses + components conditionally (the orchestrator)."""
        raise NotImplementedError("Step 2: _create_components")

    def _optimize(self) -> None:
        """Build the oemof ``Model`` from the energy system."""
        raise NotImplementedError("Step 3: _optimize")

    def _solve(self) -> None:
        """Solve the optimization problem and check the solver status."""
        raise NotImplementedError("Step 3: _solve")

    def _extract_results(self) -> None:
        """
        Extract the oemof results into ``self._results_main``.

        - Flows (Ladestationen)
        - Storage levels (Batterien + BEVs)
        - Costs ()

        """
        raise NotImplementedError("Step 4: _extract_results")

    def _save_results(self) -> None:
        """Optionally dump the results to disk."""
        raise NotImplementedError("Step 4: _save_results")

    def get_wallbox_schedule(self) -> Dict[str, pd.DataFrame]:
        """Return the per-vehicle wallbox schedule (charge/discharge/net per step)."""
        raise NotImplementedError("Step 4: get_wallbox_schedule")


###########################################################################
# Standalone entry point
###########################################################################
def main() -> None:
    """Standalone smoke test (synthetic inputs). Implemented in Step 4 once the
    pipeline is complete."""
    raise NotImplementedError("Step 4: main")


if __name__ == "__main__":
    main()

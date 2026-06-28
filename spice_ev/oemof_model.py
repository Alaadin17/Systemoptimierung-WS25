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
        """Validate the provided timeseries DataFrame into ``self.df_timeseries``.

        Requires a DataFrame (there is no CSV path). Clips PV to >= 0 and fills
        NaN with 0 — a light safety net; the data should already be clean.
        """
        if self._timeseries_df_input is None:
            raise ValueError("timeseries_df is required")
        df = self._timeseries_df_input.copy()
        if "PV_kW" in df.columns:
            df["PV_kW"] = df["PV_kW"].clip(lower=0)
        self.df_timeseries = df.fillna(0)

    def _create_time_index(self) -> None:
        """Set ``self.time_index``: adopt the provided one, else build from config."""
        if self._time_index_input is not None:
            self.time_index = pd.DatetimeIndex(self._time_index_input)
        else:
            self.time_index = pd.date_range(
                start=self.config.start_date,
                periods=self.config.periods,
                freq=self.config.freq,
            )
        self.config.periods = len(self.time_index)

    def _create_energy_system(self) -> None:
        """Create the oemof ``EnergySystem`` on ``self.time_index`` (the backbone)."""
        self.es = EnergySystem(timeindex=self.time_index, infer_last_interval=True)

    # --- existence helpers: decide what gets built ---
    def _has_pv(self) -> bool:
        """True if PV should be modelled: enabled in config AND PV_kW data present."""
        if not self.config.enable_pv:
            return False
        df = self.df_timeseries
        return df is not None and "PV_kW" in df.columns and float(df["PV_kW"].sum()) > 0.0

    def _has_battery(self) -> bool:
        """True if a stationary home battery should be modelled."""
        return self.config.enable_battery and bool(self.battery_params)

    def _has_vehicles(self) -> bool:
        """True if vehicles should be modelled."""
        return self.config.enable_vehicles and bool(self.vehicle_params)

    def _create_components(self) -> None:
        """Build buses + components conditionally (the orchestrator).

        Always: bus_home + grid_supply + household_demand (2b).
        Conditional: home battery per battery_params entry (2c); bus_pv + pv/excess if
        PV (2d); one mobility part per vehicle_params entry (2e); a grid_feedin sink at
        bus_home if enable_grid_feedin (lets home/BEV surplus reach the grid).
        """
        # --- always: home bus + grid supply + household load ---
        b_home = buses.Bus(label="bus_home")
        self.es.add(b_home)
        self._b_home = b_home
        self._add_grid_and_demand(b_home)

        # --- PV side: bus_pv + pv + excess [+ converter] (2d) ---
        if self._has_pv():
            self._add_pv(b_home)

        # --- one home battery per entry in battery_params (2c) ---
        if self._has_battery():
            for bid, bp in self.battery_params.items():
                self._add_battery(bid, bp, b_home)

        # --- one mobility part per entry in vehicle_params (2e) ---
        if self._has_vehicles():
            for vid, params in self.vehicle_params.items():
                self._add_vehicle(vid, params, b_home)

        # --- grid feed-in at bus_home: lets home/BEV surplus reach the grid (2e) ---
        if self.config.enable_grid_feedin:
            self.es.add(cmp.Sink(
                label="grid_feedin",
                inputs={b_home: flows.Flow(variable_costs=self.config.grid_feedin_tariff)},
            ))

    # ------------------------------------------------------------------
    # Component builders (called by _create_components)
    # ------------------------------------------------------------------
    def _add_grid_and_demand(self, b_home) -> None:
        """[2b] Grid supply (Source) + household load (Sink) at bus_home.

        - grid_supply: draws from the grid, capped at ``grid_power`` kW (falls back to
          ``config.grid_supply_power_kW``) and priced with ``grid_variable_costs`` so
          the optimizer avoids unnecessary grid draw. Draw only — no export here
          (grid feed-in comes later, only via PV ``excess``).
        - household_demand: a FIXED load (``Load_kW``) that must be served every step
          (missing column -> 0).
        """
        periods = self.config.periods
        grid_power_kW = (self.grid_power if self.grid_power is not None
                         else self.config.grid_supply_power_kW)
        if "Load_kW" in self.df_timeseries.columns:
            load = self.df_timeseries["Load_kW"].iloc[:periods]
        else:
            load = pd.Series(0.0, index=range(periods))

        self.es.add(cmp.Source(
            label="grid_supply",
            outputs={b_home: flows.Flow(
                nominal_value=grid_power_kW,
                variable_costs=self.config.grid_variable_costs,
            )},
        ))
        self.es.add(cmp.Sink(
            label="household_demand",
            inputs={b_home: flows.Flow(fix=load, nominal_value=1)},
        ))

    def _add_pv(self, b_home) -> None:
        """[2d] PV side: bus_pv + pv source + grid feed-in (excess) [+ converter].

        - pv: fixed generation (``PV_kW``) feeding bus_pv.
        - excess_electricity: grid feed-in sink, priced with ``grid_feedin_tariff``
          (negative = revenue).
        - converter_pv_to_home (bus_pv -> bus_home): only if ``config.enable_pv_to_home``;
          efficiency from ``converter_pv_to_home_efficiency``, capped at
          ``converter_pv_to_home_power_kW``. Without it PV can only be exported.
        """
        periods = self.config.periods
        pv_series = self.df_timeseries["PV_kW"].iloc[:periods]

        b_pv = buses.Bus(label="bus_pv")
        self.es.add(b_pv)

        self.es.add(cmp.Source(
            label="pv",
            outputs={b_pv: flows.Flow(
                fix=pv_series, nominal_value=1,
                variable_costs=self.config.pv_variable_costs)},
        ))
        # grid feed-in only from PV (no path from battery/vehicle to the grid)
        self.es.add(cmp.Sink(
            label="excess_electricity",
            inputs={b_pv: flows.Flow(variable_costs=self.config.grid_feedin_tariff)},
        ))
        if self.config.enable_pv_to_home:
            self.es.add(cmp.Converter(
                label="converter_pv_to_home",
                inputs={b_pv: flows.Flow(
                    nominal_value=self.config.converter_pv_to_home_power_kW,
                    variable_costs=self.config.converter_pv_to_home_variable_costs)},
                outputs={b_home: flows.Flow()},
                conversion_factors={b_home: self.config.converter_pv_to_home_efficiency},
            ))

    def _add_battery(self, bid, bp, b_home) -> None:
        """[2c] One home battery <bid>: bus_battery_<bid> + storage + DC/AC link.

        Variant A (losses in the link): the scenario efficiency is split evenly over
        both link directions (sqrt(eff) each, so charge*discharge = eff); the storage
        itself is lossless (loss_rate=0, conversion factors 1.0) so losses are not
        counted twice. Sizing comes from ``bp`` (one battery_params entry) with config
        fallbacks. Called once per entry -> supports multiple batteries.
        """
        capacity = float(bp.get("capacity_kWh", self.config.battery_capacity_kWh))
        power = float(bp.get("power_kW", self.config.battery_max_power_kW))
        discharge_power = float(bp.get("discharge_power_kW", power))
        init = float(bp.get("initial_soc", self.config.battery_initial_soc))
        init = min(max(init, self.config.battery_min_soc), self.config.battery_max_soc)
        efficiency = float(bp.get("efficiency", self.config.battery_efficiency))
        eff_dir = efficiency ** 0.5  # half the losses per link direction

        b_battery = buses.Bus(label=f"bus_battery_{bid}")
        self.es.add(b_battery)

        # ideal DC/AC link bus_home <-> bus_battery; the losses live here
        self.es.add(cmp.Link(
            label=f"link_home_battery_{bid}",
            inputs={
                b_battery: flows.Flow(nominal_value=discharge_power),  # discharge
                b_home: flows.Flow(nominal_value=power),               # charge
            },
            outputs={
                b_home: flows.Flow(nominal_value=discharge_power),
                b_battery: flows.Flow(nominal_value=power),
            },
            conversion_factors={
                (b_battery, b_home): eff_dir,   # discharge into the home
                (b_home, b_battery): eff_dir,   # charge from the home
            },
        ))
        # storage itself lossless (losses are in the link above)
        self.es.add(cmp.GenericStorage(
            label=f"home_battery_{bid}",
            inputs={b_battery: flows.Flow(nominal_value=power)},
            outputs={b_battery: flows.Flow(nominal_value=discharge_power)},
            nominal_storage_capacity=capacity,
            min_storage_level=self.config.battery_min_soc,
            max_storage_level=self.config.battery_max_soc,
            initial_storage_level=init,
            inflow_conversion_factor=1.0,
            outflow_conversion_factor=1.0,
            loss_rate=0.0,
            balanced=False,
        ))

    def _loss_factor(self) -> float:
        """kWh/step -> kW factor: oemof multiplies fixed_losses_absolute by the step
        duration (hours), so divide the per-step kWh by the step hours."""
        if len(self.time_index) > 1:
            step_hours = (self.time_index[1] - self.time_index[0]) / pd.Timedelta(hours=1)
        else:
            step_hours = 0.25
        return 1.0 / step_hours

    def _add_vehicle(self, vid, params, b_home) -> None:
        """[2e] One vehicle <vid>: bus_mobility + wallbox_charge [+ wallbox_discharge] + BEV.

        Mode follows spice_ev (binary per-vehicle ``v2g``) plus the global
        ``enable_grid_feedin`` switch:
        - charge_only : v2g False -> only wallbox_charge (bus_home -> bus_mobility).
        - V2H         : v2g True  -> + wallbox_discharge (bus_mobility -> bus_home).
        - V2G         : V2H + enable_grid_feedin -> surplus reaches the grid (grid_feedin
                        sink at bus_home). ``config.enable_v2h`` is the global master off.

        Refinements:
        - at_home: the wallbox (charge AND discharge) only works while the vehicle is home
          (``max = at_home``; 0 -> blocked).
        - consumption: driving demand as ``fixed_losses_absolute`` (kWh/step -> kW via
          ``_loss_factor``), drawn from the BEV storage even while away.
        - discharge_limit (Option A): for V2H/V2G-capable vehicles the SOC floor is
          raised to ``max(min_soc, discharge_limit)`` (spice_ev param, default 0.5), so
          the controllable discharge stays above it in the LP. Note this also reserves
          that SOC against driving (a bit more conservative than spice_ev, where
          discharge_limit only caps the runtime unload).
        """
        periods = self.config.periods
        at_home = _as_array(params.get("at_home", 1.0), periods)
        consumption = _as_array(params.get("consumption", 0.0), periods)
        loss_factor = self._loss_factor()

        capacity = float(params.get("capacity_kWh", self.config.bev_capacity_kWh))
        min_soc = float(params.get("min_soc", self.config.bev_min_soc))
        max_soc = float(params.get("max_soc", self.config.bev_max_soc))
        v2g = bool(params.get("v2g", False))
        wallbox_power = float(params.get("wallbox_power_kW", self.config.wallbox_power_kW))

        # discharge possible only if v2g and globally enabled; then raise the SOC floor
        # to discharge_limit so V2H/V2G cannot drain below it (Option A).
        can_discharge = self.config.enable_v2h and v2g
        discharge_limit = float(params.get("discharge_limit", self.config.bev_discharge_limit))
        storage_min = max(min_soc, discharge_limit) if can_discharge else min_soc
        init_soc = min(max(float(params.get("initial_soc", self.config.bev_initial_soc)),
                           storage_min), max_soc)

        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)

        # charge: bus_home -> bus_mobility (only while at home: max = at_home)
        self.es.add(cmp.Converter(
            label=f"wallbox_charge_{vid}",
            inputs={b_home: flows.Flow()},
            outputs={b_mobility: flows.Flow(max=at_home, nominal_value=wallbox_power)},
            conversion_factors={b_mobility: self.config.wallbox_efficiency_charge},
        ))

        # discharge (V2H): bus_mobility -> bus_home (only if can_discharge, and at home)
        if can_discharge:
            self.es.add(cmp.Converter(
                label=f"wallbox_discharge_{vid}",
                inputs={b_mobility: flows.Flow()},
                outputs={b_home: flows.Flow(max=at_home, nominal_value=wallbox_power)},
                conversion_factors={b_home: self.config.wallbox_efficiency_discharge},
            ))

        # BEV battery at bus_mobility; driving demand = fixed absolute losses
        self.es.add(cmp.GenericStorage(
            label=f"bev_battery_{vid}",
            inputs={b_mobility: flows.Flow()},
            outputs={b_mobility: flows.Flow()},
            nominal_storage_capacity=capacity,
            min_storage_level=storage_min,
            max_storage_level=max_soc,
            initial_storage_level=init_soc,
            loss_rate=0.0,
            fixed_losses_absolute=consumption * loss_factor,
            balanced=False,
        ))

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

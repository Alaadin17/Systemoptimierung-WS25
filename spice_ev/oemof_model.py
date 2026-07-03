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
from oemof.solph import EnergySystem, Model, buses, components as cmp, flows

# Added in later steps when the stages need them:
#   Step 3 (_solve):   import warnings; from pyomo.opt import SolverStatus, TerminationCondition
#   Step 4 (_extract): from oemof.solph import processing


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
    export_graph: bool = False  # render the built topology as SVG (oemof.network.graph -> Graphviz)

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
        grid_connectors: Optional[Dict[str, Dict[str, Any]]] = None,
        battery_params: Optional[Dict[str, Dict[str, Any]]] = None,
        charging_stations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Store the inputs. See the module docstring for where each comes from."""
        # --- Inputs ---
        self.config = config or SystemConfig()
        self._timeseries_df_input = timeseries_df
        self._time_index_input = time_index
        self.vehicle_params: Dict[str, Dict[str, Any]] = dict(vehicle_params or {})
        self.grid_power = grid_power
        # {gcid: {"max_power"}} — one grid source/sink per connector at bus_home
        self.grid_connectors: Dict[str, Dict[str, Any]] = dict(grid_connectors or {})
        self.battery_params: Dict[str, Dict[str, Any]] = dict(battery_params or {})
        # {csid: {"max_power", "parent"}} — one wallbox per CS, linked to all vehicles
        self.charging_stations: Dict[str, Dict[str, Any]] = dict(charging_stations or {})

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
        if self.config.export_graph:
            self._export_graph()
        self._optimize()
        self._solve()
        self._extract_results()
        self._save_results()

    # ------------------------------------------------------------------
    # Stages (stubs — implemented in later steps)
    # ------------------------------------------------------------------
    def _load_data(self) -> None:
        """Keep the optional legacy timeseries DataFrame (no longer required).

        Load and PV are now provided PER grid connector
        (``grid_connectors[gcid]['load'/'pv']``), so a global timeseries_df is optional.
        """
        self.df_timeseries = (self._timeseries_df_input.copy()
                              if self._timeseries_df_input is not None else None)

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

    def _create_components(self) -> None:
        """Per-GC topology (with pruning) — the orchestrator.

        1. Build a bus + BEV storage per vehicle.
        2. A charging station (wallbox) is kept only if >=1 vehicle plugs into it, and
           is connected only to the vehicles that actually use it (masked by
           ``connected_cs``).
        3. A grid connector is 'active' if it has a used wallbox OR load OR PV OR a
           battery. Each active GC gets its own bus ``Home_1``, ``Home_2``, ... with its
           own grid_supply source, grid_feedin sink, household_demand, PV and batteries.
        """
        periods = self.config.periods

        # 1) one bus + BEV per vehicle
        for vid, params in self.vehicle_params.items():
            self._add_vehicle(vid, params)

        # 2) which vehicles use each charging station (>=1 step) -> keep only used CS
        cs_users = {}
        for csid in self.charging_stations:
            users = [vid for vid, node in self._vehicle_nodes.items()
                     if csid in {c for c in node["connected_cs"] if c is not None}]
            if users:
                cs_users[csid] = users

        def gc_used_cs(gcid):
            return [c for c in cs_users if self.charging_stations[c].get("parent") == gcid]

        def gc_has_battery(gcid):
            return any(bp.get("parent") == gcid for bp in self.battery_params.values())

        def nonzero(arr):
            return arr is not None and float(np.sum(_as_array(arr, periods))) > 0.0

        # 3) active grid connectors -> named Home_1, Home_2, ...
        self._gc_bus = {}
        n = 0
        for gcid, gc in self.grid_connectors.items():
            has_pv = self.config.enable_pv and nonzero(gc.get("pv"))
            has_bat = self.config.enable_battery and gc_has_battery(gcid)
            if not (gc_used_cs(gcid) or nonzero(gc.get("load")) or has_pv or has_bat):
                continue  # GC carries nothing -> exclude it entirely
            n += 1
            name = f"Home_{n}"
            b = buses.Bus(label=name)
            self.es.add(b)
            self._gc_bus[gcid] = b
            self._add_grid_connector(gcid, gc, b, name, cs_users)

    # ------------------------------------------------------------------
    # Component builders (called by _create_components)
    # ------------------------------------------------------------------
    def _add_grid_connector(self, gcid, gc, b, name, cs_users) -> None:
        """Build one active grid connector ``name`` and everything on its bus ``b``:
        grid_supply source, grid_feedin sink, household_demand, PV, batteries and the
        wallboxes of the charging stations that belong to this GC (only used ones)."""
        periods = self.config.periods

        # grid supply source
        self.es.add(cmp.Source(
            label=f"grid_supply_{name}",
            outputs={b: flows.Flow(
                nominal_value=float(gc.get("max_power", self.config.grid_supply_power_kW)),
                variable_costs=self.config.grid_variable_costs)},
        ))
        # grid feed-in sink (export)
        if self.config.enable_grid_feedin:
            self.es.add(cmp.Sink(
                label=f"grid_feedin_{name}",
                inputs={b: flows.Flow(variable_costs=self.config.grid_feedin_tariff)},
            ))
        # household load (fixed) if this GC has one
        load = gc.get("load")
        if load is not None and float(np.sum(_as_array(load, periods))) > 0.0:
            self.es.add(cmp.Sink(
                label=f"household_demand_{name}",
                inputs={b: flows.Flow(fix=_as_array(load, periods), nominal_value=1)},
            ))
        # PV on this GC
        pv = gc.get("pv")
        if self.config.enable_pv and pv is not None and float(np.sum(_as_array(pv, periods))) > 0.0:
            self._add_pv(b, name, _as_array(pv, periods))
        # stationary batteries whose parent is this GC
        if self.config.enable_battery:
            for bid, bp in self.battery_params.items():
                if bp.get("parent") == gcid:
                    self._add_battery(bid, bp, b)
        # wallboxes of this GC's used charging stations
        for csid, users in cs_users.items():
            if self.charging_stations[csid].get("parent") == gcid:
                self._add_wallbox(csid, users, b)

    def _add_pv(self, gc_bus, name, pv_series) -> None:
        """[2d] PV on grid connector ``name``: bus_pv_<name> + pv source + excess sink
        [+ converter_pv_to_home to this GC's bus]."""
        b_pv = buses.Bus(label=f"bus_pv_{name}")
        self.es.add(b_pv)
        self.es.add(cmp.Source(
            label=f"pv_{name}",
            outputs={b_pv: flows.Flow(fix=pv_series, nominal_value=1,
                                      variable_costs=self.config.pv_variable_costs)},
        ))
        self.es.add(cmp.Sink(
            label=f"excess_{name}",
            inputs={b_pv: flows.Flow(variable_costs=self.config.grid_feedin_tariff)},
        ))
        if self.config.enable_pv_to_home:
            self.es.add(cmp.Converter(
                label=f"converter_pv_to_home_{name}",
                inputs={b_pv: flows.Flow(
                    nominal_value=self.config.converter_pv_to_home_power_kW,
                    variable_costs=self.config.converter_pv_to_home_variable_costs)},
                outputs={gc_bus: flows.Flow()},
                conversion_factors={gc_bus: self.config.converter_pv_to_home_efficiency},
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

    def _add_vehicle(self, vid, params) -> None:
        """[2e] One vehicle <vid>: bus_mobility + bev_battery.

        The wallboxes are NOT built here — they are added per charging station in
        ``_add_wallbox`` (only to the vehicles that actually use the station).

        - consumption: driving demand as ``fixed_losses_absolute`` (kWh/step -> kW via
          ``_loss_factor``), drawn from the BEV storage even while away.
        - discharge_limit (Option A): for V2H/V2G-capable vehicles the SOC floor is
          raised to ``max(min_soc, discharge_limit)``; otherwise ``min_soc``.

        Node refs + flags (bus, can_discharge, per-step connected_cs) are stored in
        ``self._vehicle_nodes[vid]`` so the wallbox builder can connect every CS to it.
        """
        periods = self.config.periods
        consumption = _as_array(params.get("consumption", 0.0), periods)
        loss_factor = self._loss_factor()

        capacity = float(params.get("capacity_kWh", self.config.bev_capacity_kWh))
        min_soc = float(params.get("min_soc", self.config.bev_min_soc))
        max_soc = float(params.get("max_soc", self.config.bev_max_soc))
        v2g = bool(params.get("v2g", False))

        # discharge possible only if v2g and globally enabled; then raise the SOC floor
        # to discharge_limit so V2H/V2G cannot drain below it (Option A).
        can_discharge = self.config.enable_v2h and v2g
        discharge_limit = float(params.get("discharge_limit", self.config.bev_discharge_limit))
        storage_min = max(min_soc, discharge_limit) if can_discharge else min_soc
        init_soc = min(max(float(params.get("initial_soc", self.config.bev_initial_soc)),
                           storage_min), max_soc)

        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)

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

        self._vehicle_nodes[vid] = {
            "bus": b_mobility,
            "can_discharge": can_discharge,
            "connected_cs": list(params.get("connected_cs", [])),
        }

    def _add_wallbox(self, csid, users, gc_bus) -> None:
        """[2e] One wallbox (charging station ``csid``) on ``gc_bus``, connected to the
        vehicles that actually use it.

        For each using vehicle a masked Converter gc_bus -> bus_mobility_<vid> is built;
        its ``max`` is 1 exactly in the steps where that vehicle is plugged into THIS
        station (``connected_cs == csid``), else 0. Power = the station's ``max_power``.
        For v2g vehicles (and ``enable_v2h``) a matching discharge converter (V2H/V2G)
        bus_mobility -> gc_bus is added.
        """
        periods = self.config.periods
        power = float(self.charging_stations[csid].get("max_power", self.config.wallbox_power_kW))
        for vid in users:
            node = self._vehicle_nodes[vid]
            mask = self._cs_mask(node["connected_cs"], csid, periods)
            b_mob = node["bus"]
            self.es.add(cmp.Converter(
                label=f"wallbox_charge_{csid}_{vid}",
                inputs={gc_bus: flows.Flow()},
                outputs={b_mob: flows.Flow(max=mask, nominal_value=power)},
                conversion_factors={b_mob: self.config.wallbox_efficiency_charge},
            ))
            if node["can_discharge"]:
                self.es.add(cmp.Converter(
                    label=f"wallbox_discharge_{csid}_{vid}",
                    inputs={b_mob: flows.Flow()},
                    outputs={gc_bus: flows.Flow(max=mask, nominal_value=power)},
                    conversion_factors={gc_bus: self.config.wallbox_efficiency_discharge},
                ))

    @staticmethod
    def _cs_mask(connected_cs, csid, n) -> np.ndarray:
        """0/1 array of length n: 1 where the vehicle is plugged into ``csid`` that step."""
        conn = list(connected_cs) if connected_cs is not None else []
        return np.array([1.0 if (i < len(conn) and conn[i] == csid) else 0.0
                         for i in range(n)], dtype=float)

    def _optimize(self) -> None:
        """Build the oemof ``Model`` (the LP) from the energy system.

        ``Model(self.es)`` turns every bus balance, flow limit (nominal_value/max) and
        storage equation into constraints plus the objective (sum of variable_costs).
        If ``config.debug`` is set, also write the model as a readable ``.lp`` file
        (constraints + objective) into ``results/`` for inspection.
        """
        self.model = Model(self.es)
        if self.config.debug:
            lp_path = Path("results") / f"{self.config.dump_filename}_debug.lp"
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

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

    def _export_graph(self) -> None:
        """Render the built energy system as an SVG topology graph (when ``export_graph``).

        Uses ``oemof.network.graph.create_nx_graph`` -> DOT -> Graphviz ``dot -Tsvg`` and
        writes ``results/<dump_filename>_graph.svg`` (+ .dot), coloured by node type. If
        Graphviz ``dot`` is not on the PATH, only the .dot file is written.
        """
        import shutil
        import subprocess
        from oemof.network.graph import create_nx_graph

        style = {
            "bus": ("box", "#dbeafe", "#1e40af"), "source": ("ellipse", "#dcfce7", "#15803d"),
            "sink": ("ellipse", "#fee2e2", "#b91c1c"), "converter": ("box", "#ffffff", "#475569"),
            "link": ("box", "#eef2ff", "#4338ca"), "storage": ("cylinder", "#fef9c3", "#a16207"),
            "other": ("box", "#f1f5f9", "#64748b"),
        }

        def kind(n):
            if isinstance(n, buses.Bus):
                return "bus"
            if isinstance(n, cmp.GenericStorage):
                return "storage"
            if isinstance(n, cmp.Link):
                return "link"
            if isinstance(n, cmp.Converter):
                return "converter"
            if isinstance(n, cmp.Source):
                return "source"
            if isinstance(n, cmp.Sink):
                return "sink"
            return "other"

        typ = {str(n): kind(n) for n in self.es.nodes}
        g = create_nx_graph(self.es)
        lines = ["digraph oemof_model {", "  rankdir=LR;",
                 '  node [style=filled, fontname="Segoe UI", fontsize=10];',
                 '  edge [color="#64748b", arrowsize=0.7];']
        for n in g.nodes():
            shape, fill, border = style[typ.get(n, "other")]
            lines.append(f'  "{n}" [shape={shape}, fillcolor="{fill}", color="{border}"];')
        for u, v in g.edges():
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")

        base = Path("results") / f"{self.config.dump_filename}_graph"
        base.parent.mkdir(parents=True, exist_ok=True)
        dot_path = base.with_suffix(".dot")
        dot_path.write_text("\n".join(lines), encoding="utf-8")

        dot_exe = shutil.which("dot")
        if dot_exe:
            subprocess.run([dot_exe, "-Tsvg", str(dot_path), "-o", str(base.with_suffix(".svg"))],
                           check=True)
            logging.info("oemof graph written: %s", base.with_suffix(".svg"))
        else:
            logging.warning("Graphviz 'dot' not on PATH -> only %s written", dot_path)


###########################################################################
# Standalone entry point
###########################################################################
def main() -> None:
    """Standalone smoke test (synthetic inputs). Implemented in Step 4 once the
    pipeline is complete."""
    raise NotImplementedError("Step 4: main")


if __name__ == "__main__":
    main()

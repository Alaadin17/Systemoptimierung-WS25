"""
oemof energy system - the LP behind the spice_ev strategy ``oemof_solve``.

Author: Alaa Alsleman, GitHub: Alaadin17

Builds an oemof.solph energy system from a spice_ev scenario, solves it ONCE over the whole
horizon and hands the schedule back to spice_ev.

Topology - one bus per grid connector
-------------------------------------
Everything is built CONDITIONALLY: a grid connector exists only if it carries something (a
used charging station, a load, PV or a battery), a wallbox only if a vehicle actually uses
it. See ``_create_components``. Labels as they appear in the dumps, for Home_1:

    Home_1                          AC bus per active grid connector; {gcid: bus} in _gc_bus
      grid_supply_Home_1            purchase from the grid
      grid_feedin_Home_1            export from the house bus (only with enable_grid_feedin)
      household_demand_Home_1       fixed load
      bus_pv_Home_1                 DC side of the PV: pv_Home_1 (source), excess_Home_1
                                    (feed-in sink), converter_pv_to_home_Home_1 (inverter,
                                    only with enable_pv_to_home)
      link_home_battery_<bid> <-> bus_battery_<bid> <-> home_battery_<bid>
      wallbox_charge_<csid>_<vid> -> bus_mobility_<vid> -> bev_battery_<vid>
      wallbox_discharge_<csid>_<vid>  only for v2g vehicles with enable_v2h

Two decisions keep the schedule congruent with spice_ev:
- The charging/discharging loss sits IN THE STORAGE (inflow/outflow_conversion_factor),
  not in the link and not in the wallbox - exactly like spice_ev's ``Battery``. Round trip
  eff^2.
- Wallboxes and links are LOSSLESS, they only limit the power. A loss there would count
  twice and let the planned SOC drift away from the simulated one.

The PV feeds the house bus through its inverter, everything else goes to the grid via
``excess_<n>``. From the house bus on, a PV kWh is indistinguishable from a grid kWh, so
there is no attribution "this much PV went into the car". Named direct branches with a PV
charging bonus once provided one; they were removed because they never changed the
schedule.

Inputs (__init__)
-----------------
``OemofSolve.build_oemof_inputs`` builds them from the spice_ev scenario (the tests build
them directly): config (SystemConfig), time_index (= the spice_ev steps), grid_connectors
(load, PV, max_power, optional price series and installed kWp), charging_stations,
vehicle_params, battery_params.

Flow
----
``run()``: time grid -> energy system -> components -> [graph] -> LP -> solve -> extract
-> CSV. ``get_plan()`` / ``get_wallbox_schedule()`` return the result.
"""

import logging
import time
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from oemof.solph import EnergySystem, Model, buses, components as cmp, flows
from oemof.solph import processing
from pyomo.opt import SolverStatus, TerminationCondition


###########################################################################
# Configuration
###########################################################################
# Every setting lives in one typed object, SystemConfig. It is filled from the flat
# dict of oemof_* keys the OemofSolve strategy receives (from_options); nothing else
# feeds it.
@dataclass
class SystemConfig:
    """Every setting of the model in one place.

    Filled from the ``oemof_`` keys of simulate.cfg (``from_options``). Whatever the
    scenario brings per component - max_power, capacity, SOC, efficiency - beats these
    values; the fields here are the fallback for what the strategy does not pass.
    """

    # Time parameters
    start_date: str = "2025-01-01"
    periods: int = 96  # 15-minute steps (96 = 1 day for debug)
    freq: str = "15min"

    # Switches for paths the scenario cannot express. What it CAN express - whether a
    # grid connector has PV, whether a battery exists, whether there are vehicles - is not
    # switched twice: PV and storages exist exactly when the scenario brings them.
    enable_pv_to_home: bool = True    # without the inverter PV can ONLY feed in -
    #                                   the scenario has no way to say that
    enable_grid_feedin: bool = True   # allow export from the house bus (battery/V2G)

    # System parameters
    grid_supply_power_kW: float = 30.0

    # Converter. The battery LINK is lossless on purpose (conversion factors 1.0) — the
    # charging/discharging loss lives in the storage, exactly like spice_ev's Battery, see
    # _add_battery. There are therefore no per-direction link efficiencies to configure.
    converter_pv_to_home_power_kW: float = 10.0
    converter_pv_to_home_efficiency: float = 1.0
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
    # The BEV's own charging/discharging loss is not configured here: the strategy hands
    # over each vehicle's Battery.efficiency, modelled AT THE STORAGE like spice_ev does.

    # Wallbox
    wallbox_power_kW: float = 11.0
    # spice_ev has NO wallbox loss: the charging station only limits the power, the loss
    # happens inside the battery. Keep these at 1.0 to match it (>1.0 would double-count).
    wallbox_efficiency_charge: float = 1.0
    wallbox_efficiency_discharge: float = 1.0
    enable_v2h: bool = True

    # Tiny anti-degeneracy cost (ct/kWh) on storage charging and V2H feed-back. Without it
    # the LP may cycle energy pointlessly (storage out -> in, or wallbox charge+discharge in
    # the same step) because that changes the objective by exactly zero — the SOC series then
    # drifts to its floor for no reason. 0.001 is far below any real price and does not alter
    # genuine decisions; it only makes useless cycling strictly worse than doing nothing.
    storage_cycle_penalty: float = 0.001

    # Forbids a storage to charge and discharge in the SAME time step. Costs one BINARY
    # variable per storage and step: the LP becomes a MILP and solve time rises sharply.
    # Off by default and kept for experiments only - why it once mattered is told in
    # _add_no_simultaneous_constraints.
    forbid_simultaneous_storage: bool = False

    # --- Prices (ct/kWh) -------------------------------------------------------------
    # grid_variable_costs is the purchase price when the scenario brings none: the
    # COMPLETE price as it stands on the bill - no price sheet, no markup, no capacity
    # charge. If the scenario carries price signals (include_price_csv in generate.cfg),
    # that series applies INSTEAD, per step, exactly as spice_ev's own strategies see it
    # (see OemofSolve._grid_price_series); grid_variable_costs is then without effect.
    # grid_feedin_tariff is always fixed and applies to BOTH export paths (PV surplus and
    # export from the house bus). Negative = revenue, 0 = no remuneration.
    #
    # NOTE - spice_ev's own cost calculation goes its own way: simulate.py evaluates
    # costs.py after the simulation with the price sheet (grid fees, levies, taxes,
    # capacity charge). The EUR/a in results.json therefore come from other prices than
    # the schedule did - both numbers are right, they are just not the same calculation.
    # The full price semantics are documented in examples/configs/simulate_with_oemof.cfg.
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 35.0
    grid_feedin_tariff: float = 0.0   # negative = revenue; 0 = feeding in earns nothing

    # --- consumer type ---------------------------------------------------------------
    # A price CSV holds the EXCHANGE price. What a customer actually pays is that price
    # plus grid fee, levies, concession fee and electricity tax, and for a household plus
    # VAT on the sum. consumer_type picks those components from CONSUMER_TYPES; the two
    # fields below override them when a study needs its own numbers.
    #
    # The markup applies ONLY to a price series the scenario brings along
    # (include_price_csv). Without one, grid_variable_costs applies, and that is already a
    # complete retail price - adding a markup there would count the same components twice.
    consumer_type: str = "household"
    grid_price_markup_ct_kWh: Optional[float] = None   # None = from consumer_type
    grid_price_vat: Optional[float] = None             # None = from consumer_type

    # Solver
    solver: str = "cbc"
    solver_verbose: bool = False
    debug: bool = True
    solver_threads: int = 8
    solver_ratio_gap: float = 0.01

    # Result storage
    should_dump_results: bool = True
    output_dir: str = "results"  # directory for the LP dump, result dump and graph
    dump_filename: str = "dump"
    export_graph: bool = False  # render the built topology as SVG (oemof.network.graph -> Graphviz)

    @classmethod
    def from_options(cls, options: Optional[Dict[str, Any]] = None) -> "SystemConfig":
        """From the strategy's cfg dict; unknown keys warn, wrong types are coerced.

        A leading ``oemof_`` is stripped - spice_ev's util.py already does that on the
        live path, so this only matters for direct calls (tests)."""
        config = cls()                                  # start with all defaults
        typ = {f.name: f.type for f in fields(cls)}     # field name -> declared type
        for key, value in (options or {}).items():
            name = key.removeprefix("oemof_")           # "oemof_solver" -> "solver"
            if name not in typ:
                logging.warning("Unknown oemof parameter ignored: %s", key)
                continue
            setattr(config, name, _coerce(name, value, typ[name]))
        return config

    def consumer_tariff(self):
        """(markup in ct/kWh, VAT factor) for this consumer type; overrides win.

        An unknown ``consumer_type`` warns and falls back to the household values rather
        than raising - a typo in a cfg should not abort a run that is otherwise fine.
        """
        if self.consumer_type not in CONSUMER_TYPES:
            logging.warning("Unknown oemof_consumer_type %r - using 'household'. Known: %s",
                            self.consumer_type, ", ".join(sorted(CONSUMER_TYPES)))
        markup, vat = CONSUMER_TYPES.get(self.consumer_type, CONSUMER_TYPES["household"])
        if self.grid_price_markup_ct_kWh is not None:
            markup = float(self.grid_price_markup_ct_kWh)
        if self.grid_price_vat is not None:
            vat = float(self.grid_price_vat)
        return markup, vat


# What a consumer pays ON TOP of the exchange price: (markup ct/kWh, VAT factor).
# Both numbers are the sum of the components in examples/data/price_sheet.json,
# default_grid_operator, so they can be checked against it:
#
#                              household (SLP)   commercial (RLM, MV)
#   grid_fee commodity_charge        7.48              3.49
#   levies (sum of five)             1.237             1.237
#   concession_fee                   1.32              1.32
#   tax_on_electricity               2.05              2.05
#                              ---------------   --------------------
#                                   12.09 ct           8.10 ct
#   value_added_tax                  19 %              0 % (reclaimable)
#
# power_procurement (7.70 ct) is deliberately NOT part of the markup - that IS the energy,
# and the energy comes from the exchange series.
CONSUMER_TYPES = {
    "household":  (12.09, 0.19),
    "commercial": (8.10, 0.00),
}


def _as_array(values, n):
    """Scalar/list/Series -> float array of length n.

    A scalar is stretched, a series that is too long is cut. CAUTION: one that is too short
    is silently padded with zeros - a PV profile that is too short then means "no sun".
    """
    if np.isscalar(values):
        return np.full(n, float(values))
    arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr[:n]


def _nonzero(values, n):
    """Does this series carry any energy at all? Decides whether a node is built.

    Must say the same thing at every call site: otherwise a grid connector could be admitted
    for a load that then gets no sink. The sum (instead of ``.any()``) is deliberate - a
    series that cancels to 0 carries nothing.
    """
    return values is not None and float(np.sum(_as_array(values, n))) > 0.0


_WAHR = {"true", "yes", "on", "1"}
_FALSCH = {"false", "no", "off", "0"}


def _coerce(name, value, typ):
    """Bring a cfg value to the declared field type.

    The cfg is read with ``json.loads``, and JSON only knows lowercase ``true``/``false``.
    ``oemof_enable_v2h = False`` therefore stays the STRING "False" - which is truthy in
    Python, so the switch would silently be ON. That has happened once.
    """
    ziel = str(typ)
    if "bool" in ziel:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in _WAHR:
            return True
        if text in _FALSCH:
            return False
        logging.warning("oemof_%s: '%s' is not a boolean (expected true/false)",
                        name, value)
        return bool(value)
    if "float" in ziel or "int" in ziel:
        try:
            zahl = float(value)
        except (TypeError, ValueError):
            logging.warning("oemof_%s: '%s' is not a number - value taken over as is",
                            name, value)
            return value
        return int(zahl) if "int" in ziel and "float" not in ziel else zahl
    return value


def _vid_from_bus(label) -> str:
    """Vehicle id from its bus label ``bus_mobility_<vid>``.

    Going through the bus instead of the wallbox label is necessary because
    ``wallbox_charge_<csid>_<vid>`` cannot be split unambiguously - both parts may contain
    underscores.
    """
    label = str(label)
    prefix = "bus_mobility_"
    return label[len(prefix):] if label.startswith(prefix) else label


###########################################################################
# Main class
###########################################################################
class EnergySystemModel:
    """Builds the energy system, solves it once over the whole horizon and extracts the
    schedule. Topology: see the module docstring; order of stages: see ``run()``."""

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        time_index: Optional[pd.DatetimeIndex] = None,
        vehicle_params: Optional[Dict[str, Dict[str, Any]]] = None,
        grid_connectors: Optional[Dict[str, Dict[str, Any]]] = None,
        battery_params: Optional[Dict[str, Dict[str, Any]]] = None,
        charging_stations: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """Store the inputs. See the module docstring for where each comes from."""
        # --- Inputs ---
        self.config = config or SystemConfig()
        self._time_index_input = time_index
        self.vehicle_params: Dict[str, Dict[str, Any]] = dict(vehicle_params or {})
        # {gcid: {"load", "pv", "max_power", "price_ct_kWh", "pv_power_kW"}} - the last three
        # optional; each ACTIVE GC becomes one Home_<n> bus
        self.grid_connectors: Dict[str, Dict[str, Any]] = dict(grid_connectors or {})
        self.battery_params: Dict[str, Dict[str, Any]] = dict(battery_params or {})
        # {csid: {"max_power", "parent"}} — one wallbox per CS, but only for the vehicles
        # that actually plug into it (unused stations are pruned)
        self.charging_stations: Dict[str, Dict[str, Any]] = dict(charging_stations or {})

        # --- Outputs, populated by the pipeline ---
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.es = None              # oemof EnergySystem (_create_energy_system)
        self.model = None           # oemof Model (_optimize)
        self._results_main = None   # processing.results(model) (_extract_results)
        self._wallbox_schedule: Optional[Dict[str, pd.DataFrame]] = None  # per vehicle
        self._battery_schedule: Optional[Dict[str, pd.DataFrame]] = None  # per stationary battery
        self._grid_schedule: Optional[Dict[str, pd.DataFrame]] = None     # per GC (verification)
        self._summary_df: Optional[pd.DataFrame] = None  # per-GC grid/PV/load + SOCs + wallboxes
        self._costs: Optional[Dict[str, float]] = None   # objective value
        self._gc_bus: Dict[str, Any] = {}   # {gcid: Home_<n> bus} (_create_components)
        self._vehicle_nodes: Dict[str, Dict[str, Any]] = {}  # per-vehicle nodes (bus/flags/cs)
        # storages that can charge AND discharge — the candidates for the binary
        # "not both at once" constraint (see _add_no_simultaneous_constraints)
        self._storage_pairs: list = []

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------
    def run(self) -> None:
        """The whole pipeline: time grid -> topology -> [graph] -> LP -> solve -> extract -> CSV."""
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
    # Pipeline stages (in the order run() calls them)
    # ------------------------------------------------------------------
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
        """The topology per grid connector - this is where it is decided what gets built.

        First one bus and one storage per vehicle, then the charging stations (only the used
        ones, and only linked to the vehicles that really use them), then per active grid
        connector one bus ``Home_<n>`` with everything attached to it. Active means: a used
        wallbox OR a load OR PV OR a battery - otherwise the connector is not built at all.
        """
        periods = self.config.periods
        # clear before the first node - the vehicles register themselves right here
        self._storage_pairs = []

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

        # 3) active grid connectors -> named Home_1, Home_2, ...
        self._gc_bus = {}
        n = 0
        for gcid, gc in self.grid_connectors.items():
            has_pv = _nonzero(gc.get("pv"), periods)
            has_bat = gc_has_battery(gcid)
            if not (gc_used_cs(gcid) or _nonzero(gc.get("load"), periods)
                    or has_pv or has_bat):
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

        # Supply source. If the strategy hands over a price series from the scenario
        # (include_price_csv) it applies per step - otherwise the fixed cfg value.
        preis = gc.get("price_ct_kWh")
        supply = cmp.Source(
            label=f"grid_supply_{name}",
            outputs={b: flows.Flow(
                nominal_value=float(gc.get("max_power", self.config.grid_supply_power_kW)),
                variable_costs=(_as_array(preis, periods) if preis is not None
                                else self.config.grid_variable_costs))},
        )
        self.es.add(supply)
        # Feed-in: the same fixed value for both paths - the PV surplus and the export
        # from the house bus (battery/V2G). All that matters is that it is not remunerated
        # POSITIVELY while purchase is cheaper: the LP would then buy power and feed it back
        # in the same step at a profit.
        feedin_tariff = float(self.config.grid_feedin_tariff)
        # grid feed-in sink (export from the home bus; battery/V2G)
        if self.config.enable_grid_feedin:
            self.es.add(cmp.Sink(
                label=f"grid_feedin_{name}",
                inputs={b: flows.Flow(variable_costs=feedin_tariff)},
            ))
        # household load (fixed) if this GC has one
        load = gc.get("load")
        if _nonzero(load, periods):
            self.es.add(cmp.Sink(
                label=f"household_demand_{name}",
                inputs={b: flows.Flow(fix=_as_array(load, periods), nominal_value=1)},
            ))
        # PV on this GC; converter limit = installed plant size when the scenario has one
        pv = gc.get("pv")
        if _nonzero(pv, periods):
            self._add_pv(
                b, name, _as_array(pv, periods), feedin_tariff,
                float(gc.get("pv_power_kW", self.config.converter_pv_to_home_power_kW)))
        # stationary batteries whose parent is this GC
        for bid, bp in self.battery_params.items():
            if bp.get("parent") == gcid:
                self._add_battery(bid, bp, b)
        # wallboxes of this GC's used charging stations
        for csid, users in cs_users.items():
            if self.charging_stations[csid].get("parent") == gcid:
                self._add_wallbox(csid, users, b)

    def _add_pv(self, gc_bus, name, pv_series, feedin_tariff, converter_power_kW) -> None:
        """PV at one grid connector: PV bus, source, surplus sink and - only with
        ``enable_pv_to_home`` - the inverter.

        The inverter carries ``nominal_value = converter_power_kW`` on its DC input and feeds
        the house bus. Whatever is not needed there leaves through ``excess_<name>`` as
        feed-in. From the house bus on, a PV kWh is indistinguishable from a grid kWh, so
        there is no attribution "this much PV went into the car".
        """
        b_pv = buses.Bus(label=f"bus_pv_{name}")
        self.es.add(b_pv)
        self.es.add(cmp.Source(
            label=f"pv_{name}",
            outputs={b_pv: flows.Flow(fix=pv_series, nominal_value=1,
                                      variable_costs=self.config.pv_variable_costs)},
        ))
        self.es.add(cmp.Sink(
            label=f"excess_{name}",
            inputs={b_pv: flows.Flow(variable_costs=feedin_tariff)},
        ))
        if not self.config.enable_pv_to_home:
            return
        self.es.add(cmp.Converter(
            label=f"converter_pv_to_home_{name}",
            inputs={b_pv: flows.Flow(
                nominal_value=converter_power_kW,
                variable_costs=self.config.converter_pv_to_home_variable_costs)},
            outputs={gc_bus: flows.Flow()},
            conversion_factors={gc_bus: self.config.converter_pv_to_home_efficiency},
        ))

    def _add_battery(self, bid, bp, b_home) -> None:
        """One stationary battery: bus, storage, AC/DC link. Called once per entry.

        Reachable only through the link from the house bus - that link is the only way in.

        The loss sits IN THE STORAGE (inflow/outflow_conversion_factor), exactly like
        spice_ev's ``Battery``: eff when charging, eff when discharging, round trip eff^2.
        The link is lossless and only limits the power. An earlier variant with sqrt(eff)
        per link direction gave a round trip of only eff - the LP battery was ~5 % better
        than the simulated one, and planned discharges ran empty.
        """
        capacity = float(bp.get("capacity_kWh", self.config.battery_capacity_kWh))
        power = float(bp.get("power_kW", self.config.battery_max_power_kW))
        discharge_power = float(bp.get("discharge_power_kW", power))
        init = float(bp.get("initial_soc", self.config.battery_initial_soc))
        init = min(max(init, self.config.battery_min_soc), self.config.battery_max_soc)
        efficiency = float(bp.get("efficiency", self.config.battery_efficiency))

        # ONE bus carries both directions; the link from the house bus is the only way in.
        b_battery = buses.Bus(label=f"bus_battery_{bid}")
        self.es.add(b_battery)

        # lossless DC/AC link between this GC's bus and the battery — it only
        # limits the power; the losses live in the storage below (like spice_ev)
        link = cmp.Link(
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
                (b_battery, b_home): 1.0,   # discharge into the home
                (b_home, b_battery): 1.0,   # charge from the home
            },
        )
        self.es.add(link)
        # the storage carries the losses: eff on charge, eff on discharge (round trip eff²)
        storage = cmp.GenericStorage(
            label=f"home_battery_{bid}",
            # tiny penalty on BOTH directions kills free storage-cycling degeneracy
            inputs={b_battery: flows.Flow(nominal_value=power,
                                          variable_costs=self.config.storage_cycle_penalty)},
            outputs={b_battery: flows.Flow(nominal_value=discharge_power,
                                           variable_costs=self.config.storage_cycle_penalty)},
            nominal_storage_capacity=capacity,
            min_storage_level=self.config.battery_min_soc,
            max_storage_level=self.config.battery_max_soc,
            initial_storage_level=init,
            inflow_conversion_factor=efficiency,    # AC in  -> stored   (x eff)
            outflow_conversion_factor=efficiency,   # stored -> AC out   (x eff)
            loss_rate=0.0,
            balanced=False,
        )
        self.es.add(storage)
        # at the two storage flows "charge" and "discharge" are unambiguous - this is where
        # forbid_simultaneous_storage attaches, if it is used
        self._storage_pairs.append({
            "label": f"home_battery_{bid}", "storage": storage,
            "in_bus": b_battery, "out_bus": b_battery,
            "p_in": power, "p_out": discharge_power,
        })

    def _loss_factor(self) -> float:
        """kWh/step -> kW factor: oemof multiplies fixed_losses_absolute by the step
        duration (hours), so divide the per-step kWh by the step hours."""
        return 1.0 / self._step_hours()

    def _add_vehicle(self, vid, params) -> None:
        """One vehicle: mobility bus + storage. The wallboxes are built by ``_add_wallbox``.

        - consumption is the driving demand as ``fixed_losses_absolute`` - it is taken from
          the storage even while the car is away.
        - The SOC floor is ``min_soc``, for V2H/V2G-capable vehicles
          ``max(min_soc, discharge_limit)``.
        - min_soc_series raises that floor PER STEP to the ``desired_soc`` from the scenario,
          in the step before each departure - so the car is as full as spice_ev expects.
          Capped at ``max_soc``, otherwise the storage would be infeasible.

        Bus and flags go to ``self._vehicle_nodes[vid]`` so the wallboxes can find them.
        """
        periods = self.config.periods
        consumption = _as_array(params.get("consumption", 0.0), periods)
        loss_factor = self._loss_factor()

        capacity = float(params.get("capacity_kWh", self.config.bev_capacity_kWh))
        min_soc = float(params.get("min_soc", self.config.bev_min_soc))
        max_soc = float(params.get("max_soc", self.config.bev_max_soc))
        v2g = bool(params.get("v2g", False))

        # discharge possible only if v2g and globally enabled; then raise the SOC floor
        # to discharge_limit so V2H/V2G cannot drain below it.
        can_discharge = self.config.enable_v2h and v2g
        # V2G discharges with the discharge_curve that spice_ev derives from the charging
        # curve times v2g_power_factor - typically half. Without this limit the LP plans
        # with the full station power and the planned SOC drifts away from the simulated one.
        p_entladen = params.get("discharge_power_kW")
        p_entladen = float(p_entladen) if p_entladen else None
        discharge_limit = float(params.get("discharge_limit", self.config.bev_discharge_limit))
        floor = max(min_soc, discharge_limit) if can_discharge else min_soc

        # per-step floor from the scenario (desired_soc at departures), else the constant one
        series = params.get("min_soc_series")
        if series is None:
            storage_min = floor
            first_min = floor
        else:
            arr = np.minimum(np.maximum(_as_array(series, periods), floor), max_soc)
            # oemof indexes storage_content over periods+1 points (incl. the end state)
            storage_min = np.concatenate([arr, arr[-1:]])
            first_min = float(arr[0])

        init_soc = min(max(float(params.get("initial_soc", self.config.bev_initial_soc)),
                           first_min), max_soc)

        # Charging/discharging loss lives IN THE BATTERY, exactly like spice_ev
        # (Battery.efficiency). The wallbox itself is lossless there, it only limits power.
        # 0.95 is spice_ev's own Battery default (battery.py); only the tests omit the key
        efficiency = float(params.get("efficiency", 0.95))

        # ONE bus carries both directions; the wallbox is the only way into the vehicle.
        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)

        # BEV battery at bus_mobility; driving demand = fixed absolute losses
        # (fixed_losses_absolute is NOT scaled by the conversion factors — the trip energy
        # leaves the storage directly, just like spice_ev subtracts soc_delta.)
        bev = cmp.GenericStorage(
            label=f"bev_battery_{vid}",
            # tiny penalty kills the free storage-cycling degeneracy (see SystemConfig)
            inputs={b_mobility: flows.Flow(
                variable_costs=self.config.storage_cycle_penalty)},
            # Non-v2g vehicles get NO storage outflow (nominal 0): energy only leaves by
            # driving (fixed_losses). An open, unbounded outflow would let the LP burn
            # stored energy via in+out cycling (x0.95² loss) — at NEGATIVE grid prices
            # that becomes a money pump (destroy, then get paid to re-buy).
            outputs={b_mobility: flows.Flow(
                nominal_value=None if can_discharge else 0,
                variable_costs=self.config.storage_cycle_penalty)},
            nominal_storage_capacity=capacity,
            min_storage_level=storage_min,
            max_storage_level=max_soc,
            initial_storage_level=init_soc,
            inflow_conversion_factor=efficiency,    # AC in  -> stored  (x eff)
            outflow_conversion_factor=efficiency,   # stored -> V2H out (x eff)
            loss_rate=0.0,
            fixed_losses_absolute=consumption * loss_factor,
            balanced=False,
        )
        self.es.add(bev)

        if can_discharge:
            # Only a V2H vehicle can charge and discharge at once. Its storage inflow has no
            # nominal_value of its own (the wallbox limits it), so the big-M comes from the
            # strongest station this vehicle ever plugs into.
            # no "or []": connected_cs is a numpy array, whose truth value raises
            angeschlossen = params.get("connected_cs")
            used = {c for c in ([] if angeschlossen is None else list(angeschlossen)) if c}
            p_in = max((float(self.charging_stations.get(c, {}).get(
                "max_power", self.config.wallbox_power_kW)) for c in used),
                default=self.config.wallbox_power_kW)
            self._storage_pairs.append({
                "label": f"bev_battery_{vid}", "storage": bev,
                "in_bus": b_mobility, "out_bus": b_mobility,
                "p_in": p_in, "p_out": min(p_in, p_entladen or p_in),
            })

        self._vehicle_nodes[vid] = {
            "bus": b_mobility,      # charging arrives here, V2H leaves here
            "can_discharge": can_discharge,
            "p_discharge": p_entladen,   # None = only the station power limits
            "connected_cs": list(params.get("connected_cs", [])),
        }

    def _add_wallbox(self, csid, users, gc_bus) -> None:
        """One charging station, linked to the vehicles that really use it.

        One masked converter per vehicle: ``max`` is 1 exactly in the steps in which the car
        is plugged into THIS station, 0 otherwise. For v2g vehicles with ``enable_v2h`` also
        the way back into the house.

        The wallbox is LOSSLESS and only limits the power - as in spice_ev. The
        charging/discharging loss sits in the vehicle battery; a loss here as well would
        count twice and let the planned SOC drift away from the simulated one.
        """
        periods = self.config.periods
        power = float(self.charging_stations[csid].get("max_power", self.config.wallbox_power_kW))
        for vid in users:
            node = self._vehicle_nodes[vid]
            mask = self._cs_mask(node["connected_cs"], csid, periods)
            b_mob = node["bus"]
            # Limit the AC side (-> wallbox): that is the power spice_ev commands and clamps
            # against cs.max_power. Limiting the DC output instead would let the AC draw
            # reach max_power/efficiency and exceed the station's rating.
            wb_charge = cmp.Converter(
                label=f"wallbox_charge_{csid}_{vid}",
                inputs={gc_bus: flows.Flow(max=mask, nominal_value=power)},
                outputs={b_mob: flows.Flow()},
                conversion_factors={b_mob: self.config.wallbox_efficiency_charge},
            )
            self.es.add(wb_charge)
            if node["can_discharge"]:
                # The discharge power is NOT the charging power: spice_ev limits it to the
                # discharge_curve (charging curve times v2g_power_factor). Allowing the full
                # station power here plans more feed-back than the simulation can deliver -
                # the SOC then diverges.
                p_ab = min(power, node["p_discharge"]) if node["p_discharge"] else power
                wb_discharge = cmp.Converter(
                    label=f"wallbox_discharge_{csid}_{vid}",
                    inputs={b_mob: flows.Flow()},
                    # the tiny penalty also prevents simultaneous charge+discharge (a free
                    # cycle through the two lossless wallbox converters)
                    outputs={gc_bus: flows.Flow(
                        max=mask, nominal_value=p_ab,
                        variable_costs=self.config.storage_cycle_penalty)},
                    conversion_factors={gc_bus: self.config.wallbox_efficiency_discharge},
                )
                self.es.add(wb_discharge)

    @staticmethod
    def _cs_mask(connected_cs, csid, n) -> np.ndarray:
        """0/1 array of length n: 1 where the vehicle is plugged into ``csid`` that step."""
        conn = list(connected_cs) if connected_cs is not None else []
        return np.array([1.0 if (i < len(conn) and conn[i] == csid) else 0.0
                         for i in range(n)], dtype=float)

    def _optimize(self) -> None:
        """Build the LP from the energy system: bus balances, power limits, storage
        equations and the objective. With ``debug`` a readable ``.lp`` file is written."""
        self.model = Model(self.es)
        self._add_no_simultaneous_constraints()   # optional, turns the LP into a MILP
        if self.config.debug:
            lp_path = Path(self.config.output_dir) / f"{self.config.dump_filename}_debug.lp"
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    def _add_no_simultaneous_constraints(self) -> None:
        """Forbid every storage to charge AND discharge in the same step.

        "Either A or B" is not a linear statement; it needs one binary variable per storage
        and step:

            inflow[t]   <=  P_charge    * y[t]
            outflow[t]  <=  P_discharge * (1 - y[t])          y[t] in {0, 1}

        Big-M is the power limit that exists anyway - the tighter, the faster the MILP.
        Attached at the STORAGE flows, because every path converges there.

        History: this was needed against a PV charging bonus that made it profitable to
        route PV *through* a storage into the house - each flow was allowed on its own, the
        detour only cost the round trip, and the bonus was collected in full. That bonus and
        its branches are gone, and with them the incentive: storage_cycle_penalty already
        makes cycling more expensive than doing nothing. Off by default; the switch stays
        for experiments.

        Price: the LP becomes a MILP. One binary per storage and step - 5856 for a
        two-month horizon - and solve time rises by orders of magnitude.
        """
        if not self.config.forbid_simultaneous_storage or not self._storage_pairs:
            return
        import pyomo.environ as po

        steps = list(self.model.TIMESTEPS)
        rows, m = {}, self.model
        for pair in self._storage_pairs:
            zufluss = (pair["in_bus"], pair["storage"])
            abfluss = (pair["storage"], pair["out_bus"])
            for t in steps:
                # only build where both directions exist at all
                if (zufluss[0], zufluss[1], t) in m.flow and (abfluss[0], abfluss[1], t) in m.flow:
                    rows[(pair["label"], t)] = (zufluss, abfluss,
                                                float(pair["p_in"]), float(pair["p_out"]))
        if not rows:
            return

        idx = list(rows)
        m.speicher_modus = po.Var(idx, domain=po.Binary)

        def _laden(model, label, t):
            zu, _, p_in, _ = rows[(label, t)]
            return model.flow[zu[0], zu[1], t] <= p_in * model.speicher_modus[label, t]

        def _entladen(model, label, t):
            _, ab, _, p_out = rows[(label, t)]
            return model.flow[ab[0], ab[1], t] <= p_out * (1 - model.speicher_modus[label, t])

        m.speicher_nur_laden = po.Constraint(idx, rule=_laden)
        m.speicher_nur_entladen = po.Constraint(idx, rule=_entladen)
        logging.info("Simultaneous charge/discharge forbidden: %d binary variables, %d rows "
                     "(%s)", len(idx), 2 * len(idx),
                     ", ".join(sorted({lbl for lbl, _ in idx})))

    def _solve(self) -> None:
        """Solve the LP and check optimality.

        A non-optimal result raises ``RuntimeError`` instead of silently carrying on with
        nonsense. With ``debug`` the solver shows its console and one line with duration,
        status and objective goes to the log.
        """
        solver_options = {}
        if self.config.solver == "cbc":
            solver_options = {"threads": self.config.solver_threads,
                              "ratioGap": self.config.solver_ratio_gap}
        # debug turns the solver console on so a run can be watched live
        tee = self.config.solver_verbose or self.config.debug

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            results = self.model.solve(
                solver=self.config.solver,
                solve_kwargs={"tee": tee},
                cmdline_options=solver_options,
            )
        solve_seconds = time.perf_counter() - t0

        status = results.solver.status
        termination = results.solver.termination_condition
        if status != SolverStatus.ok or termination != TerminationCondition.optimal:
            raise RuntimeError(
                "oemof solve did not reach an optimal solution "
                f"(status={status}, termination={termination}, "
                f"message={getattr(results.solver, 'message', 'n/a')})"
            )

        if self.config.debug:
            logging.info(
                "oemof solved in %.3fs | status=%s | termination=%s | objective=%.4f",
                solve_seconds, status, termination, self.model.objective(),
            )

    # ------------------------------------------------------------------
    # Result extraction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fit(arr, n) -> np.ndarray:
        """Coerce a result sequence to length n (truncate, or pad with the last value)."""
        arr = np.asarray(arr, dtype=float)
        if len(arr) >= n:
            return arr[:n]
        pad = arr[-1] if len(arr) else 0.0
        return np.concatenate([arr, np.full(n - len(arr), pad)])

    def _flow(self, res, source, target, n) -> np.ndarray:
        """Per-step flow (kW) between two nodes, 0 if that edge does not exist."""
        data = res.get((source, target))
        if data is None:
            return np.zeros(n)
        return self._fit(data["sequences"]["flow"].to_numpy(), n)

    def _storage_soc_end(self, res, node, n, capacity) -> np.ndarray:
        """The SOC every storage reaches at the END of each step.

        oemof indexes ``storage_content`` over n+1 points in time: t is the level at the
        START of step t. Here the ends count (1..n) - exactly the value ``step()`` steers
        the simulated battery to with ``Battery.load(target_soc=...)``.
        """
        data = res.get((node, None))
        if data is None or not capacity:
            return np.full(n, np.nan)
        seq = data["sequences"]
        col = "storage_content" if "storage_content" in seq.columns else seq.columns[0]
        arr = np.asarray(seq[col].to_numpy(), dtype=float)
        return self._fit(arr[1:] if len(arr) > n else arr, n) / float(capacity)

    def _storage_content(self, res, node, n) -> np.ndarray:
        """Per-step storage content (kWh) of a GenericStorage node (NaN if absent)."""
        data = res.get((node, None))
        if data is None:
            return np.full(n, np.nan)
        seq = data["sequences"]
        col = "storage_content" if "storage_content" in seq.columns else seq.columns[0]
        return self._fit(seq[col].to_numpy(), n)

    def _extract_results(self) -> None:
        """Translate the solved model into schedule, summary and cost.

        All powers are AC at the grid-connector bus - that is what spice_ev applies. Built
        here: ``_wallbox_schedule`` per vehicle (charge/discharge/net kW, soc_kWh, soc_end,
        consumption), ``_battery_schedule`` per stationary battery (charge/discharge kW,
        soc_end), ``_grid_schedule`` per grid connector (supply/feedin kW, verification
        only), ``_summary_df`` with one column family per node type (grid_supply_*,
        grid_feedin_*, pv_*, pv_feedin_*, pv_selfuse_*, household_demand_*,
        home_battery_<bid>_soc_kWh, wallbox_charge/discharge_<vid>,
        battery_charge/discharge_<bid>, grid_price_ct_*) and ``_costs``.
        """
        res = processing.results(self.model)
        self._results_main = res
        n = len(self.time_index)
        idx = self.time_index

        # --- per-vehicle wallbox schedule (AC at the GC bus) ---
        charge = {vid: np.zeros(n) for vid in self.vehicle_params}
        discharge = {vid: np.zeros(n) for vid in self.vehicle_params}
        for node in self.es.nodes:
            if not isinstance(node, cmp.Converter):
                continue
            if node.label.startswith("wallbox_charge_"):
                # this flow is the TOTAL AC power of the station - exactly what step()
                # books at the grid connector
                gc_bus = list(node.inputs)[0]
                vid = _vid_from_bus(list(node.outputs)[0].label)
                charge[vid] = charge[vid] + self._flow(res, gc_bus, node, n)
            elif node.label.startswith("wallbox_discharge_"):
                b_mob = list(node.inputs)[0]
                gc_bus = list(node.outputs)[0]         # converter -> gc_bus (AC fed back)
                vid = _vid_from_bus(b_mob.label)
                discharge[vid] = discharge[vid] + self._flow(res, node, gc_bus, n)

        schedule = {}
        for vid in self.vehicle_params:
            bev = self._node("bev_battery_" + vid)
            capacity = float(self.vehicle_params[vid].get(
                "capacity_kWh", self.config.bev_capacity_kWh))
            soc = self._storage_content(res, bev, n) if bev is not None else np.full(n, np.nan)
            soc_end = (self._storage_soc_end(res, bev, n, capacity) if bev is not None
                       else np.full(n, np.nan))
            consumption = _as_array(self.vehicle_params[vid].get("consumption", 0.0), n)
            schedule[vid] = pd.DataFrame({
                "charge_kW": charge[vid],
                "discharge_kW": discharge[vid],
                "net_kW": charge[vid] - discharge[vid],
                "soc_kWh": soc,
                # SOC (0..1) the plan reaches at the END of the step — this is what step()
                # steers the simulated battery to, see OemofSolve.step().
                "soc_end": soc_end,
                "consumption_kWh": consumption,   # driving demand per step (model input)
            }, index=idx)
        self._wallbox_schedule = schedule

        # --- summary per GC: grid purchase/feed-in, PV, PV feed-in, household load and
        #     the SOC of every battery ---
        summary: Dict[str, np.ndarray] = {}
        for node in self.es.nodes:
            lbl = node.label
            if lbl.startswith("grid_supply_"):
                summary[lbl] = self._flow(res, node, list(node.outputs)[0], n)
            elif lbl.startswith("grid_feedin_"):
                summary[lbl] = self._flow(res, list(node.inputs)[0], node, n)
            elif lbl.startswith("pv_") and isinstance(node, cmp.Source):
                summary[lbl] = self._flow(res, node, list(node.outputs)[0], n)
            elif lbl.startswith("excess_"):        # PV exported to the grid (PV feed-in)
                summary[f"pv_feedin_{lbl[len('excess_'):]}"] = self._flow(
                    res, list(node.inputs)[0], node, n)
            elif lbl.startswith("converter_pv_to_home_"):
                # the whole PV self-use: everything that is not exported passes through this
                # inverter, so pv - pv_feedin == pv_selfuse. WHERE the kWh goes afterwards
                # is no longer distinguishable at the house bus.
                summary[f"pv_selfuse_{lbl[len('converter_pv_to_home_'):]}"] = \
                    self._flow(res, node, list(node.outputs)[0], n)
            elif lbl.startswith("household_demand_"):
                summary[lbl] = self._flow(res, list(node.inputs)[0], node, n)
            elif lbl.startswith("home_battery_"):
                summary[f"{lbl}_soc_kWh"] = self._storage_content(res, node, n)
        # per-vehicle wallbox AC power (charge / V2H discharge)
        for vid in self.vehicle_params:
            summary[f"wallbox_charge_{vid}"] = charge[vid]
            summary[f"wallbox_discharge_{vid}"] = discharge[vid]

        # The price the objective really used - a constant or the step function from the
        # price CSV, depending on the scenario. Written out either way so plots and checks
        # have one source and need not look up the cfg.
        for gcid, bus in self._gc_bus.items():
            p = self.grid_connectors.get(gcid, {}).get("price_ct_kWh")
            summary[f"grid_price_ct_{bus.label}"] = (
                _as_array(p, n) if p is not None
                else np.full(n, float(self.config.grid_variable_costs)))

        # --- per-battery plan: AC power at the GC bus (the link flows) ---
        # charge   = flow Home_<n> -> link (AC drawn to charge the battery)
        # discharge= flow link -> Home_<n> (AC fed back into the house)
        battery_schedule = {}
        for node in self.es.nodes:
            if isinstance(node, cmp.Link) and node.label.startswith("link_home_battery_"):
                bid = node.label[len("link_home_battery_"):]
                home_bus = next(b for b in node.inputs if str(b.label).startswith("Home_"))
                storage = self._node("home_battery_" + bid)
                capacity = float(self.battery_params.get(bid, {}).get(
                    "capacity_kWh", self.config.battery_capacity_kWh))
                battery_schedule[bid] = pd.DataFrame({
                    # the link from the house bus is the only way into the battery
                    "charge_kW": self._flow(res, home_bus, node, n),
                    "discharge_kW": self._flow(res, node, home_bus, n),
                    # SOC (0..1) at the END of the step — the target step() steers to
                    "soc_end": (self._storage_soc_end(res, storage, n, capacity)
                                if storage is not None else np.full(n, np.nan)),
                }, index=idx)
        self._battery_schedule = battery_schedule
        # ... also into the summary, so the plan-vs-actual check can run from the CSVs
        for bid, frame in battery_schedule.items():
            summary[f"battery_charge_{bid}"] = frame["charge_kW"].to_numpy()
            summary[f"battery_discharge_{bid}"] = frame["discharge_kW"].to_numpy()
        self._summary_df = pd.DataFrame(summary, index=idx)

        # --- per-GC grid plan (for verification of the applied plan, keyed by gcid) ---
        grid_schedule = {}
        for gcid, bus in self._gc_bus.items():
            name = bus.label
            grid_schedule[gcid] = pd.DataFrame({
                "supply_kW": self._flow(res, self._node(f"grid_supply_{name}"), bus, n),
                "feedin_kW": self._flow(res, bus, self._node(f"grid_feedin_{name}"), n),
            }, index=idx)
        self._grid_schedule = grid_schedule

        # --- scalar results ---
        # The objective in ct: purchase, feed-in and the tiny storage_cycle_penalty
        # tie-breaker, nothing else. The EUR/a in
        # results.json are computed separately anyway, after the simulation from the
        # physical timeseries.
        step_hours = self._step_hours()
        self._costs = {
            "objective": float(self.model.objective()),
            "consumer_type": self.config.consumer_type,
            "periods": n,
            "step_hours": step_hours,
            # share of a year the simulated horizon covers - needed to pro-rate any annual
            # tariff onto this run
            "fraction_year": n * step_hours / 8760.0,
        }
        # Peak and energy per grid connector, for a tariff calculation done LATER and
        # outside the model. The model itself prices energy only: no capacity charge enters
        # the objective, deliberately - an annual amount would dominate a one-week schedule.
        for bus in self._gc_bus.values():
            spalte = f"grid_supply_{bus.label}"
            if spalte not in self._summary_df.columns:
                continue
            reihe = self._summary_df[spalte]
            self._costs[f"grid_peak_kW_{bus.label}"] = float(reihe.max())
            self._costs[f"grid_energy_kWh_{bus.label}"] = float(reihe.sum() * step_hours)

    def _step_hours(self) -> float:
        """Length of one time step in hours (0.25 for a 15-minute grid)."""
        if self.time_index is not None and len(self.time_index) > 1:
            return (self.time_index[1] - self.time_index[0]) / pd.Timedelta(hours=1)
        return 0.25

    def _save_results(self) -> None:
        """Write the schedule, summary and scalar results as CSV into ``config.output_dir``.

        ``<stem>_costs.csv`` is one row: the objective plus what a later tariff calculation
        needs - the consumer type, the horizon length, and the grid peak and energy per
        connector."""
        if not self.config.should_dump_results:
            return
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = self.config.dump_filename
        for vid, df in (self._wallbox_schedule or {}).items():
            df.to_csv(out / f"{stem}_wallbox_{vid}.csv")
        if self._summary_df is not None:
            self._summary_df.to_csv(out / f"{stem}_summary.csv")
        if self._costs is not None:
            pd.DataFrame([self._costs]).to_csv(out / f"{stem}_costs.csv", index=False)

    def _node(self, label):
        """Look up a built node by its label (None if it does not exist)."""
        for node in self.es.nodes:
            if node.label == label:
                return node
        return None

    def get_wallbox_schedule(self) -> Dict[str, pd.DataFrame]:
        """The schedule per vehicle. ``charge_kW``/``discharge_kW``/``soc_end`` are the
        contract with ``OemofSolve.commands_from_oemof``, the rest is extra."""
        if self._wallbox_schedule is None:
            raise RuntimeError("get_wallbox_schedule() called before _extract_results()")
        return self._wallbox_schedule

    def get_plan(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """The whole schedule, grouped by component type. Row k is simulation step k, so
        the strategy can look it up directly.

        - ``vehicles``  per vehicle, applied by ``step()``
        - ``batteries`` per stationary battery, applied as well
        - ``grid``      planned grid exchange - NOT applied, kept for verification
        """
        if self._wallbox_schedule is None:
            raise RuntimeError("get_plan() called before _extract_results()")
        return {
            "vehicles": self._wallbox_schedule,
            "batteries": self._battery_schedule or {},
            "grid": self._grid_schedule or {},
        }

    def _export_graph(self) -> None:
        """Draw the topology as SVG (only with ``export_graph``), coloured by node type.
        Without Graphviz on the PATH only the .dot file is written."""
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

        base = Path(self.config.output_dir) / f"{self.config.dump_filename}_graph"
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

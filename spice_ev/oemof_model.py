"""
oemof energy system model — the LP behind the spice_ev strategy ``oemof_solve``.

Autor: Alaa Alsleman, GitHub: Alaadin17

Builds an oemof.solph energy system from a spice_ev scenario, solves it once over the
full horizon and hands the resulting per-vehicle charging plan back to spice_ev.

Architecture (one bus per grid connector)
-----------------------------------------
The topology is derived from the spice_ev scenario and built CONDITIONALLY, with
pruning: only grid connectors that actually carry something are built, and only
wallboxes that a vehicle really uses (see ``_create_components``).

Buses
- Home_1, Home_2, ... one AC bus per ACTIVE grid connector. A GC counts as active if it
                      has a used charging station, a load, PV or a battery; otherwise it
                      is skipped entirely. Mapping {gcid: bus} in ``self._gc_bus``.
- bus_pv_<name>       PV bus of that GC (only if it has PV): -> Home_<n> via
                      converter_pv_to_home_<name>, surplus via excess_<name>.
- bus_battery_<bid>   DC bus per stationary battery, coupled to its parent GC bus via
                      link_home_battery_<bid>.
- bus_mobility_<vid>  one DC bus per vehicle (BEV storage + wallboxes).

Components
- grid_supply_<name>            on every active GC bus; grid_feedin_<name> only with
                                ``enable_grid_feedin``.
- household_demand_<name>       only if that GC has a non-zero load.
- pv_<name> / excess_<name>     at bus_pv_<name> (only if that GC has PV);
                                converter_pv_to_home_<name> only with ``enable_pv_to_home``.
- home_battery_<bid> + link_home_battery_<bid>  per stationary battery. Like spice_ev,
                                the loss sits IN THE STORAGE (eff on charge, eff on
                                discharge -> round trip eff²); the link is lossless and
                                only limits the power.
- wallbox_charge_<csid>_<vid>   per charging station AND using vehicle, masked to the steps
                                that vehicle is plugged into THIS station.
                                wallbox_discharge_<csid>_<vid> only for v2g vehicles (with
                                ``enable_v2h``). Wallboxes are LOSSLESS: they only limit
                                the power, exactly like spice_ev.
- bev_battery_<vid>             per vehicle. The charging/discharging loss sits in the
                                STORAGE (inflow/outflow_conversion_factor = spice_ev's
                                ``Battery.efficiency``), the driving demand is applied as
                                fixed_losses_absolute, and ``min_soc_series`` provides a
                                per-step SOC floor (desired_soc at the departure steps).

Inputs (__init__)
-----------------
In the production path these are built by ``OemofSolve.build_oemof_inputs``
(spice_ev.strategies.oemof_solve) from the spice_ev scenario.
- config (SystemConfig)  efficiencies/costs/solver + default/fallback values.
- time_index             the shared time grid (= the spice_ev steps).
- grid_connectors        per GC: its own load/pv series + optional max_power.
- charging_stations      per CS: max_power + parent GC.
- vehicle_params         per vehicle: capacity/SOC/v2g/efficiency plus the consumption,
                         connected_cs and min_soc_series series.
- battery_params         per stationary battery (capacity/power/SOC/efficiency + parent).
- timeseries_df          legacy/optional: only kept in ``self.df_timeseries`` and not read
                         by the model — load and PV arrive per GC via ``grid_connectors``.
- grid_power             legacy/optional: not read; the limit comes from each GC's own
                         ``max_power`` (fallback ``config.grid_supply_power_kW``).

Pipeline
--------
``run()`` orchestrates: load data -> create time index -> build energy system -> build
components -> [export graph] -> optimize -> solve -> extract results -> save.
``get_wallbox_schedule()`` returns the per-vehicle plan (charge/discharge/net/soc/
consumption per step) back to the spice_ev simulation.
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass, field, fields
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
# We collect all settings in one typed object, SystemConfig. It can be filled
# two ways:
#   - from a flat dict  -> comes from the OemofSolve strategy (oemof_config)
#   - from a cfg file   -> for standalone runs
@dataclass
class SystemConfig:
    """Configuration parameters for the energy system (one typed settings object).

    Each field is ``name: type = default`` — the type is documentation, Python does not
    enforce it. Filled either from a flat dict (``from_options``, used by the strategy:
    the ``oemof_`` prefix of the cfg keys is stripped) or from a cfg file
    (``from_cfg_file``). Values that the scenario provides per component (GC max_power,
    charging station power, vehicle capacity/SOC/efficiency) override these — the fields
    here are the defaults/fallbacks.
    """

    # Time parameters
    start_date: str = "2025-01-01"
    periods: int = 96  # 15-minute steps (96 = 1 day for debug)
    freq: str = "15min"

    # Feature toggles (force a component off even if its data is present)
    enable_pv: bool = True
    enable_pv_to_home: bool = True  # build the PV->home converter (else PV only feeds in)
    enable_battery: bool = True
    enable_grid_feedin: bool = True  # allow home/BEV surplus to be exported to the grid
    # (there is deliberately no enable_vehicles: _create_components builds a bus + BEV
    #  storage for every entry in vehicle_params, so the switch would have been a lie)

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
    # Charging/discharging loss of the BEV itself (spice_ev: Battery.efficiency, default 0.95).
    # Modelled AT THE STORAGE (inflow/outflow_conversion_factor) exactly like spice_ev does.
    bev_efficiency: float = 0.95

    # Wallbox
    wallbox_power_kW: float = 11.0
    # spice_ev has NO wallbox loss: the charging station only limits the power, the loss
    # happens inside the battery. Keep these at 1.0 to match it (>1.0 would double-count).
    wallbox_efficiency_charge: float = 1.0
    wallbox_efficiency_discharge: float = 1.0
    enable_v2h: bool = True
    # Bei KONSTANTEM Preis ist es dem LP egal, wann es vor der Abfahrt laedt - es gibt dann
    # unendlich viele gleich teure Loesungen und der Solver greift willkuerlich eine heraus
    # (im Plot sieht das nach zufaelligen Ladebloecken mitten in der Nacht aus). Mit diesem
    # Schalter bekommt frueheres Laden einen winzigen Aufschlag, sodass bei Gleichstand
    # moeglichst SPAET geladen wird - also nah an der Abfahrt und damit naeher an der
    # Morgen-PV. Der Betrag ist so klein, dass er echte Preisunterschiede nie ueberstimmt.
    prefer_late_charging: bool = False
    late_charging_penalty: float = 0.001   # ct/kWh im ersten Zeitschritt, fallend auf 0

    # Configures SPICE_EV behaviour (not the oemof model): spice_ev's clamp_power() drops any
    # charging power below cs.min_power / vehicle_type.min_charging_power to ZERO. The LP does
    # not know that rule and happily plans such small powers (e.g. to use a little PV surplus),
    # so that energy silently vanishes and the simulated SOC drifts below the planned one.
    # True -> set both limits to 0 so every planned power is applied. False -> leave as is.
    ignore_min_charging_power: bool = False

    # Tiny anti-degeneracy cost (ct/kWh) on storage charging and V2H feed-back. Without it
    # the LP may cycle energy pointlessly (storage out -> in, or wallbox charge+discharge in
    # the same step) because that changes the objective by exactly zero — the SOC series then
    # drifts to its floor for no reason. 0.001 is far below any real price and does not alter
    # genuine decisions; it only makes useless cycling strictly worse than doing nothing.
    storage_cycle_penalty: float = 0.001

    # --- PV-Eigenverbrauchsanreiz: eigene PV->Speicher-Zweige --------------------------
    # Am Hausbus sind PV-kWh und Netz-kWh nicht mehr unterscheidbar - ein Bonus auf
    # "Home_n -> wallbox_charge" wuerde also auch Netzstrom belohnen und jede Kennzahl
    # "so viel PV ging ins Auto" waere wertlos. Deshalb bekommt die PV hinter dem
    # Wechselrichter einen eigenen AC-Bus (bus_pvac_<name>), von dem je ein BENANNTER Zweig
    # direkt zur Wallbox-Klemme und in die Hausbatterie fuehrt. Weil bus_pv_<name> nur von
    # der fix=-PV-Quelle gespeist wird und keine Kante zurueckfuehrt, kann dort keine
    # Netz-kWh den Bonus einsammeln - der Anreiz ist strukturell nicht manipulierbar und
    # die Flusssumme ist eine echte, berichtbare Groesse.
    # Default AUS: ohne den Schalter ist das Modell unveraendert.
    pv_direct_to_storage: bool = False
    # Bonus in ct/kWh (positiv angeben, landet als NEGATIVE variable_costs im Modell).
    # Die Aufteilung Auto<->Hausbatterie haengt an der DIFFERENZ der beiden Werte, die
    # Entscheidung Speichern<->Einspeisen am MAXIMUM. Im Beispielszenario:
    #   0.01 / 0.01 -> reine MESSUNG: Fahrplan praktisch unveraendert, aber die Zuordnung
    #                  wird eindeutig (ohne Bonus ist sie entartet und CBC wuerfelt).
    #   6.0  / 0.0  -> anstupsen, exportneutral (<= Einspeiseverguetung 6.24 ct/kWh).
    #   8.0  / 0.0  -> kippt die Aufteilung (Differenz > 7.5 ct/kWh), kostet aber echtes Geld.
    # NUR POSITIVE Boni wirken: ein Malus macht den Direktpfad einfach ungenutzt, die PV
    # fliesst dann ueber conv_pvac_to_home -> Home_n -> link und umgeht ihn.
    pv_charge_bonus_vehicle_ct_kWh: float = 0.0
    pv_charge_bonus_battery_ct_kWh: float = 0.0
    # Verbietet einem Speicher, im SELBEN Zeitschritt zu laden und zu entladen. Ohne das
    # kann ein Bonus das LP dazu bringen, PV durch den Speicher ins Haus zu leiten statt
    # direkt - physikalisch erlaubt, aber sinnlos, und die Kennzahl pv_direct_* meldet dann
    # mehr als wirklich gespeichert wird. Kostet je Speicher und Zeitschritt eine
    # BINAERVARIABLE: aus dem LP wird ein MILP, die Loesezeit steigt deutlich. Nur
    # einschalten, wenn ein Bonus oberhalb von (1-eff^2)*Einspeiseverguetung noetig ist -
    # darunter tritt das Kreisen ohnehin nicht auf.
    forbid_simultaneous_storage: bool = False

    # Costs (ct/kWh)
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 35.0
    grid_feedin_tariff: float = -8.0  # negative = revenue
    # Wer bestimmt die Einspeiseverguetung: das Preisblatt oder die cfg?
    # True (Default, bisheriges Verhalten): liegt ein Preisblatt vor, gewinnt dessen
    # feed-in_remuneration.PV — bei 10 kWp sind das -6.24 ct/kWh, und grid_feedin_tariff
    # oben ist dann nur noch der Rueckfallwert fuer GCs ohne Blattwert. Das ueberrascht:
    # oemof_grid_feedin_tariff = 0.0 in der cfg bleibt dabei wirkungslos.
    # False: das Preisblatt wird fuer die Einspeisung ignoriert und grid_feedin_tariff gilt
    # fuer BEIDE Pfade (PV-Excess und Hausbus-Export). So laesst sich "Einspeisung ohne
    # Verguetung" tatsaechlich rechnen. Der Retail-Aufschlag auf den BEZUGSpreis bleibt
    # davon unberuehrt — das steuert use_retail_markup.
    feedin_tariff_from_price_sheet: bool = True
    # Retail markup on the grid price, mirroring the spice_ev cost calculation: adds the
    # grid fee (by fee_type), all levies, the concession fee and the electricity tax from
    # the price sheet to the spot price series, then applies VAT on top (the feed-in
    # remuneration stays net, exactly like costs.py). Only active when the strategy has a
    # price sheet AND the scenario provides price signals.
    use_retail_markup: bool = False
    # Same name and values as spice_ev's cost calculation (simulate.py/costs.py):
    # "SLP" = household standard load profile (flat grid fee) or "RLM" = metered
    # commercial customers (grid fee by the GC's voltage_level; like costs.py's edge
    # condition, the <2500 h/a utilization bracket is used).
    fee_type: str = "SLP"
    # Path to spice_ev's price sheet (same file the cost calculation uses). Source of the
    # PV feed-in remuneration and the retail markup components. Passed as an oemof_* key so
    # that spice_ev's own scripts stay untouched; relative to the working directory.
    cost_parameters_file: str = ""

    # Solver
    solver: str = "cbc"
    solver_verbose: bool = False
    debug: bool = True
    solver_threads: int = 8
    solver_ratio_gap: float = 0.01

    # Three-day debug mode. These switches steer the NOTEBOOK (example_1, Schritt 6), not
    # the model or the strategy: a spice_ev scenario is one continuous horizon, so the
    # notebook orchestrates three independent single-day generate+simulate runs. PV/load
    # are sliced BY INDEX from the yearly profile (day d -> rows d*96..(d+1)*96).
    debug_three_days: bool = False
    debug_day_starts: list = field(default_factory=lambda: [15, 180, 300])
    # Tage je Debug-Fenster; ausgewertet wird immer der LETZTE. Mit 2 hat der Speicher
    # einen Folgetag, fuer den sich Laden lohnt - sonst waere gespeicherte Energie um
    # Mitternacht wertlos (End-of-Horizon-Effekt) und die Batterie bliebe fast leer.
    debug_window_days: int = 2
    # Start-SOC des Fahrzeugs in den Debug-Tagen (die Hausbatterie startet immer leer).
    # Bewusst unter desired_soc, damit das Fahrzeug bis zur Abfahrt laden MUSS - sonst
    # zeigt der Debug-Tag keinen einzigen Ladevorgang.
    debug_initial_soc: float = 0.3

    # Result storage
    should_dump_results: bool = True
    output_dir: str = "results"  # directory for the LP dump, result dump and graph
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


def _vid_from_bus(label) -> str:
    """Vehicle id from one of its bus labels.

    A vehicle normally has a single ``bus_mobility_<vid>``; with the PV direct branches and
    V2H it also gets a pure inflow bus ``bus_mobin_<vid>``, which is where wallbox charging
    then arrives. Both must map back to the same vehicle.
    """
    label = str(label)
    for prefix in ("bus_mobin_", "bus_mobility_"):
        if label.startswith(prefix):
            return label[len(prefix):]
    return label


###########################################################################
# Main class
###########################################################################
class EnergySystemModel:
    """Models and optimizes the energy system (grid connectors + PV + batteries + N BEVs).

    Builds the oemof topology from the given inputs, solves it once over the full horizon
    and extracts the per-vehicle charging plan. See the module docstring for the topology
    and ``run()`` for the pipeline order.
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
        self.grid_power = grid_power  # legacy, not read (limit comes from each GC)
        # {gcid: {"load", "pv", "max_power"(optional)}} — each ACTIVE GC becomes one Home_<n> bus
        self.grid_connectors: Dict[str, Dict[str, Any]] = dict(grid_connectors or {})
        self.battery_params: Dict[str, Dict[str, Any]] = dict(battery_params or {})
        # {csid: {"max_power", "parent"}} — one wallbox per CS, but only for the vehicles
        # that actually plug into it (unused stations are pruned)
        self.charging_stations: Dict[str, Dict[str, Any]] = dict(charging_stations or {})

        # --- Outputs, populated by the pipeline ---
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.es = None              # oemof EnergySystem (_create_energy_system)
        self.model = None           # oemof Model (_optimize)
        self.df_timeseries: Optional[pd.DataFrame] = None   # legacy copy, not used by the model
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
    # Pipeline stages (in the order run() calls them)
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
        # vor dem ersten Knoten leeren — die Fahrzeuge tragen sich gleich hier ein
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

        # grid supply source — price per GC: time-varying from the scenario's grid
        # operator signals when available, else the constant config fallback
        price = gc.get("price_ct_kWh")
        supply_costs = (_as_array(price, periods) if price is not None
                        else self.config.grid_variable_costs)
        self.es.add(cmp.Source(
            label=f"grid_supply_{name}",
            outputs={b: flows.Flow(
                nominal_value=float(gc.get("max_power", self.config.grid_supply_power_kW)),
                variable_costs=supply_costs)},
        ))
        # feed-in tariffs per GC (negative = revenue), from the price sheet via the
        # strategy when available, else the config fallback. IMPORTANT: they differ!
        # PV export (excess sink) earns the PV remuneration; export from the HOME bus
        # (battery/V2G) earns 0 by the sheet — paying the PV tariff there would let the
        # LP buy cheap grid power and "feed it in" simultaneously for riskless profit.
        # feedin_tariff_from_price_sheet = False ignoriert die Blattwerte und laesst
        # config.grid_feedin_tariff fuer BEIDE Pfade gelten (siehe SystemConfig).
        if self.config.feedin_tariff_from_price_sheet:
            feedin_tariff = float(gc.get("feedin_tariff_ct_kWh", self.config.grid_feedin_tariff))
            homebus_tariff = float(gc.get("homebus_feedin_tariff_ct_kWh",
                                          self.config.grid_feedin_tariff))
        else:
            feedin_tariff = homebus_tariff = float(self.config.grid_feedin_tariff)
        # Wie gross darf der Bonus sein? Viel kleiner als man denkt.
        # Er macht es lohnend, PV DURCH den Speicher ins Haus zu leiten statt direkt: dabei
        # gehen (1 - eff^2) der kWh verloren, die sonst eingespeist worden waere. Rentabel
        # wird das ab
        #     bonus > (1 - eff^2) * |Einspeiseverguetung|
        # also z. B. 0.0975 * 6.24 = 0.61 ct/kWh -- weit UNTER der Verguetung selbst.
        # Darueber zirkuliert das LP Energie (laden und entladen im selben Schritt), der
        # Fahrplan wird real teurer, und - schlimmer - die gemeldete PV->Speicher-Menge
        # wird deutlich groesser als das, was tatsaechlich gespeichert bleibt. Die Kennzahl
        # misst dann nicht mehr "PV gespeichert", sondern "PV hat den Speicherbus beruehrt".
        # Der Bonus taugt deshalb als Tie-Breaker fuer eine EINDEUTIGE Zuordnung, nicht als
        # Steuerungsinstrument. Gegengeprueft in example_1: 0.5 ct/kWh = kein einziger
        # Kreislaufschritt, 1.0 ct/kWh = 2478 Schritte und +403 ct echte Mehrkosten.
        # Betroffen ist nur ein Speicher, der auch WIEDER ins Haus abgeben kann: die
        # Hausbatterie immer, das Fahrzeug nur mit V2H. Ohne Rueckweg gibt es keinen
        # Kreislauf, dort ist der Bonus unkritisch (in example_1 nachgemessen: 3 ct auf dem
        # Auto-Zweig ohne V2H -> kein einziger Kreislaufschritt).
        eff = float(self.config.battery_efficiency)
        cycle_bound = (1.0 - eff ** 2) * abs(feedin_tariff)
        riskant = {"Hausbatterie": float(self.config.pv_charge_bonus_battery_ct_kWh)}
        if self.config.enable_v2h:
            riskant["Fahrzeug (V2H aktiv)"] = float(self.config.pv_charge_bonus_vehicle_ct_kWh)
        if self.config.pv_direct_to_storage:
            for wer, bonus in riskant.items():
                if bonus > cycle_bound:
                    logging.warning(
                        "PV-Ladebonus %s %.3f ct/kWh liegt ueber der Kreislauf-Schwelle "
                        "%.3f ct/kWh ((1-eff^2)*Einspeiseverguetung, %s): das LP kann Energie "
                        "durch den Speicher zirkulieren, um ihn einzusammeln. pv_direct_* "
                        "meldet dann MEHR als wirklich gespeichert wird, und "
                        "objective_ohne_bonus steigt.", wer, bonus, cycle_bound, name)
        # grid feed-in sink (export from the home bus; battery/V2G)
        if self.config.enable_grid_feedin:
            self.es.add(cmp.Sink(
                label=f"grid_feedin_{name}",
                inputs={b: flows.Flow(variable_costs=homebus_tariff)},
            ))
        # household load (fixed) if this GC has one
        load = gc.get("load")
        if load is not None and float(np.sum(_as_array(load, periods))) > 0.0:
            self.es.add(cmp.Sink(
                label=f"household_demand_{name}",
                inputs={b: flows.Flow(fix=_as_array(load, periods), nominal_value=1)},
            ))
        # PV on this GC; converter limit = installed plant size when the scenario has one.
        # Built FIRST so the batteries and wallboxes below can branch off its AC bus.
        b_pvac = None
        pv = gc.get("pv")
        if self.config.enable_pv and pv is not None and float(np.sum(_as_array(pv, periods))) > 0.0:
            b_pvac = self._add_pv(
                b, name, _as_array(pv, periods), feedin_tariff,
                float(gc.get("pv_power_kW", self.config.converter_pv_to_home_power_kW)))
        # stationary batteries whose parent is this GC
        if self.config.enable_battery:
            for bid, bp in self.battery_params.items():
                if bp.get("parent") == gcid:
                    self._add_battery(bid, bp, b, b_pvac)
        # wallboxes of this GC's used charging stations
        for csid, users in cs_users.items():
            if self.charging_stations[csid].get("parent") == gcid:
                self._add_wallbox(csid, users, b, b_pvac)

    def _add_pv(self, gc_bus, name, pv_series, feedin_tariff, converter_power_kW):
        """[2d] PV on grid connector ``name``: bus_pv_<name> + pv source + excess sink
        [+ converter_pv_to_home to this GC's bus].

        ``feedin_tariff`` (negative = revenue) and ``converter_power_kW`` come per GC from
        the scenario/price sheet when available (see _add_grid_connector), else from config.

        With ``pv_direct_to_storage`` the converter no longer feeds the house bus directly
        but an AC-side PV bus ``bus_pvac_<name>``, from which named branches run to the house,
        to the wallboxes and to the batteries. Everything therefore stays BEHIND the single
        converter that carries ``nominal_value = converter_power_kW``, so the inverter rating
        limits the SUM of all PV destinations — no extra constraint needed. (Not academic:
        the yearly profile of example_1 peaks at 9.77 kW against a 10 kW rating.)

        Returns the bus the PV branches must start from (``bus_pvac_<name>``), or None when
        the direct paths are off or PV cannot reach the house at all.
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
            return None
        # the converter that carries the inverter rating; its target is the house bus, or
        # the PV-AC bus when the direct branches are built
        target = gc_bus
        b_pvac = None
        if self.config.pv_direct_to_storage:
            b_pvac = buses.Bus(label=f"bus_pvac_{name}")
            self.es.add(b_pvac)
            target = b_pvac
        self.es.add(cmp.Converter(
            label=f"converter_pv_to_home_{name}",
            inputs={b_pv: flows.Flow(
                nominal_value=converter_power_kW,
                variable_costs=self.config.converter_pv_to_home_variable_costs)},
            outputs={target: flows.Flow()},
            conversion_factors={target: self.config.converter_pv_to_home_efficiency},
        ))
        if b_pvac is not None:
            # the plain "PV serves the household" branch — the efficiency is already spent
            # above, so this one is a lossless, free pass-through
            self.es.add(cmp.Converter(
                label=f"conv_pvac_to_home_{name}",
                inputs={b_pvac: flows.Flow()},
                outputs={gc_bus: flows.Flow()},
                conversion_factors={gc_bus: 1.0},
            ))
        return b_pvac

    def _add_battery(self, bid, bp, b_home, b_pvac=None) -> None:
        """[2c] One home battery <bid>: bus_battery_<bid> + storage + DC/AC link.

        The charging/discharging loss sits IN THE STORAGE (inflow/outflow_conversion_factor
        = the battery's efficiency), exactly like spice_ev's ``Battery``: it loses eff on
        load AND eff on unload, i.e. a round trip of eff². The link is lossless and only
        limits the power. (This replaces the earlier 'Variant A' with sqrt(eff) per link
        direction — that gave a round trip of eff and made the LP battery ~5 % more
        efficient than the simulated one, so planned discharges ran empty.)
        Sizing comes from ``bp`` (one battery_params entry) with config fallbacks. Called
        once per entry -> supports multiple batteries.
        """
        capacity = float(bp.get("capacity_kWh", self.config.battery_capacity_kWh))
        power = float(bp.get("power_kW", self.config.battery_max_power_kW))
        discharge_power = float(bp.get("discharge_power_kW", power))
        init = float(bp.get("initial_soc", self.config.battery_initial_soc))
        init = min(max(init, self.config.battery_min_soc), self.config.battery_max_soc)
        efficiency = float(bp.get("efficiency", self.config.battery_efficiency))

        # Normally ONE bus carries both directions. With the PV direct branches the storage
        # gets a pure INFLOW side (bus_batin) and a pure OUTFLOW side (bus_batout): a PV kWh
        # arriving on a shared bus could otherwise walk straight back out through the link
        # into the house — collecting the storage bonus WITHOUT ever being stored. Splitting
        # makes that pass-through impossible instead of merely expensive, because the only
        # exit from bus_batin is the storage itself.
        if self.config.pv_direct_to_storage:
            b_bat_in = buses.Bus(label=f"bus_batin_{bid}")
            b_bat_out = buses.Bus(label=f"bus_batout_{bid}")
            self.es.add(b_bat_in, b_bat_out)
        else:
            b_bat_in = b_bat_out = buses.Bus(label=f"bus_battery_{bid}")
            self.es.add(b_bat_in)

        # lossless DC/AC link between this GC's bus and the battery — it only
        # limits the power; the losses live in the storage below (like spice_ev)
        link = cmp.Link(
            label=f"link_home_battery_{bid}",
            inputs={
                b_bat_out: flows.Flow(nominal_value=discharge_power),  # discharge
                b_home: flows.Flow(nominal_value=power),               # charge
            },
            outputs={
                b_home: flows.Flow(nominal_value=discharge_power),
                b_bat_in: flows.Flow(nominal_value=power),
            },
            conversion_factors={
                (b_bat_out, b_home): 1.0,   # discharge into the home
                (b_home, b_bat_in): 1.0,    # charge from the home
            },
        )
        self.es.add(link)
        # the storage carries the losses: eff on charge, eff on discharge (round trip eff²)
        storage = cmp.GenericStorage(
            label=f"home_battery_{bid}",
            # tiny penalty on BOTH directions kills free storage-cycling degeneracy.
            # The inflow cap is ALSO the joint power limit for the AC and the PV-direct
            # path, because both arrive on bus_batin and this is its only exit — that is
            # what keeps the plan executable for spice_ev's Battery.load(target_soc=...).
            inputs={b_bat_in: flows.Flow(nominal_value=power,
                                         variable_costs=self.config.storage_cycle_penalty)},
            outputs={b_bat_out: flows.Flow(nominal_value=discharge_power,
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
        # the two storage flows are where "charging" and "discharging" are unambiguous:
        # every path — AC via the link and PV direct — passes through them
        self._storage_pairs.append({
            "label": f"home_battery_{bid}", "storage": storage,
            "in_bus": b_bat_in, "out_bus": b_bat_out,
            "p_in": power, "p_out": discharge_power,
        })

        # PV straight into the battery, bypassing the house bus. The bonus (negative cost)
        # sits here and nowhere else: bus_pv is fed only by the fix= PV source and has no
        # edge leading back into it, so no grid kWh can ever collect it.
        if b_pvac is not None:
            self.es.add(cmp.Converter(
                label=f"conv_pv_to_battery_{bid}",
                inputs={b_pvac: flows.Flow(
                    nominal_value=power,
                    variable_costs=-float(self.config.pv_charge_bonus_battery_ct_kWh))},
                outputs={b_bat_in: flows.Flow()},
                conversion_factors={b_bat_in: 1.0},
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
        - min_soc_series (optional): a PER-STEP SOC floor taken from the spice_ev scenario
          (``desired_soc`` at the steps the vehicle departs). It makes the plan charge the
          car up to the SOC spice_ev expects before each trip, instead of riding the
          global ``min_soc`` floor. Combined with the constant floor via elementwise max
          and capped at ``max_soc`` so the storage stays feasible.

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
        efficiency = float(params.get("efficiency", self.config.bev_efficiency))

        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)

        # Same split as the home battery, but only needed when there IS a way back into the
        # house: with V2H the chain wallbox_charge -> bus_mobility -> wallbox_discharge is
        # lossless, so a bonused PV kWh could pass straight through without being stored.
        # Without V2H the storage inflow is the bus's only consumer anyway — no extra bus.
        if self.config.pv_direct_to_storage and can_discharge:
            b_mob_in = buses.Bus(label=f"bus_mobin_{vid}")
            self.es.add(b_mob_in)
        else:
            b_mob_in = b_mobility

        # BEV battery at bus_mobility; driving demand = fixed absolute losses
        # (fixed_losses_absolute is NOT scaled by the conversion factors — the trip energy
        # leaves the storage directly, just like spice_ev subtracts soc_delta.)
        bev = cmp.GenericStorage(
            label=f"bev_battery_{vid}",
            # tiny penalty kills the free storage-cycling degeneracy (see SystemConfig)
            inputs={b_mob_in: flows.Flow(
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
            inflow_conversion_factor=efficiency,    # AC in  -> stored  (x0.95)
            outflow_conversion_factor=efficiency,   # stored -> V2H out (x0.95)
            loss_rate=0.0,
            fixed_losses_absolute=consumption * loss_factor,
            balanced=False,
        )
        self.es.add(bev)

        if can_discharge:
            # Only a V2H vehicle can charge and discharge at once. Its storage inflow has no
            # nominal_value of its own (the wallbox limits it), so the big-M comes from the
            # strongest station this vehicle ever plugs into.
            used = {c for c in params.get("connected_cs", []) or [] if c}
            p_in = max((float(self.charging_stations.get(c, {}).get(
                "max_power", self.config.wallbox_power_kW)) for c in used),
                default=self.config.wallbox_power_kW)
            self._storage_pairs.append({
                "label": f"bev_battery_{vid}", "storage": bev,
                "in_bus": b_mob_in, "out_bus": b_mobility,
                "p_in": p_in, "p_out": p_in,
            })

        self._vehicle_nodes[vid] = {
            "bus": b_mobility,      # V2H feed-back leaves from here
            "bus_in": b_mob_in,     # wallbox charging arrives here (same bus unless split)
            "can_discharge": can_discharge,
            "connected_cs": list(params.get("connected_cs", [])),
        }

    def _add_wallbox(self, csid, users, gc_bus, b_pvac=None) -> None:
        """[2e] One wallbox (charging station ``csid``) on ``gc_bus``, connected to the
        vehicles that actually use it.

        For each using vehicle a masked Converter gc_bus -> bus_mobility_<vid> is built;
        its ``max`` is 1 exactly in the steps where that vehicle is plugged into THIS
        station (``connected_cs == csid``), else 0. Power = the station's ``max_power``.
        For v2g vehicles (and ``enable_v2h``) a matching discharge converter (V2H/V2G)
        bus_mobility -> gc_bus is added.

        The wallbox is LOSSLESS by default (``wallbox_efficiency_* = 1.0``), exactly like
        spice_ev: the charging station only limits the power; the charging/discharging loss
        happens inside the BEV battery (see ``_add_vehicle``: inflow/outflow_conversion_factor).
        Putting a loss here as well would double-count it and make the planned SOC drift
        away from the simulated one.
        """
        periods = self.config.periods
        power = float(self.charging_stations[csid].get("max_power", self.config.wallbox_power_kW))
        # Optionaler, mit der Zeit fallender Mini-Aufschlag: loest die Degeneriertheit bei
        # konstantem Preis auf, indem spaeteres Laden minimal guenstiger ist (siehe Config).
        late_costs = 0
        if self.config.prefer_late_charging and periods > 1:
            late_costs = (self.config.late_charging_penalty
                          * (1.0 - np.arange(periods) / (periods - 1)))
        for vid in users:
            node = self._vehicle_nodes[vid]
            mask = self._cs_mask(node["connected_cs"], csid, periods)
            b_mob = node["bus"]          # V2H source
            b_mob_in = node["bus_in"]    # charging target (differs only when split)
            # With the PV direct branches the wallbox is fed from its own AC terminal
            # bus_wbin_<csid>_<vid>, where the grid path and the PV path meet. The station
            # rating stays on wallbox_charge, the terminal's ONLY exit — so the bus balance
            # limits the SUM of both paths, which is what step() needs: it steers the SOC
            # with Battery.load(max_power=cs.max_power), and anything the plan asks beyond
            # that would silently be clipped and the simulated SOC would fall behind.
            src_bus, charge_costs = gc_bus, late_costs
            if b_pvac is not None:
                src_bus, charge_costs = buses.Bus(label=f"bus_wbin_{csid}_{vid}"), 0
                self.es.add(src_bus)
                self.es.add(cmp.Converter(      # the ordinary path out of the house bus
                    label=f"conv_home_to_wallbox_{csid}_{vid}",
                    inputs={gc_bus: flows.Flow(max=mask, nominal_value=power,
                                               variable_costs=late_costs)},
                    outputs={src_bus: flows.Flow()},
                    conversion_factors={src_bus: 1.0},
                ))
                self.es.add(cmp.Converter(      # PV straight into the car (carries the bonus)
                    label=f"conv_pv_to_wallbox_{csid}_{vid}",
                    inputs={b_pvac: flows.Flow(
                        max=mask, nominal_value=power,
                        variable_costs=-float(self.config.pv_charge_bonus_vehicle_ct_kWh))},
                    outputs={src_bus: flows.Flow()},
                    conversion_factors={src_bus: 1.0},
                ))
            # Limit the AC side (-> wallbox): that is the power spice_ev commands and clamps
            # against cs.max_power. Limiting the DC output instead would let the AC draw
            # reach max_power/efficiency and exceed the station's rating.
            wb_charge = cmp.Converter(
                label=f"wallbox_charge_{csid}_{vid}",
                inputs={src_bus: flows.Flow(max=mask, nominal_value=power,
                                            variable_costs=charge_costs)},
                outputs={b_mob_in: flows.Flow()},
                conversion_factors={b_mob_in: self.config.wallbox_efficiency_charge},
            )
            self.es.add(wb_charge)
            if node["can_discharge"]:
                wb_discharge = cmp.Converter(
                    label=f"wallbox_discharge_{csid}_{vid}",
                    inputs={b_mob: flows.Flow()},
                    # the tiny penalty also prevents simultaneous charge+discharge (a free
                    # cycle through the two lossless wallbox converters)
                    outputs={gc_bus: flows.Flow(
                        max=mask, nominal_value=power,
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
        """Build the oemof ``Model`` (the LP) from the energy system.

        ``Model(self.es)`` turns every bus balance, flow limit (nominal_value/max) and
        storage equation into constraints plus the objective (sum of variable_costs).
        If ``config.debug`` is set, also write the model as a readable ``.lp`` file
        (constraints + objective) into ``config.output_dir`` for inspection.
        """
        self.model = Model(self.es)
        self._add_no_simultaneous_constraints()   # optional, macht aus dem LP ein MILP
        if self.config.debug:
            lp_path = Path(self.config.output_dir) / f"{self.config.dump_filename}_debug.lp"
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    def _add_no_simultaneous_constraints(self) -> None:
        """Verbiete jedem Speicher, im selben Zeitschritt zu laden UND zu entladen.

        Warum ueberhaupt: mit einem PV-Ladebonus kann es sich lohnen, PV *durch* den Speicher
        ins Haus zu leiten statt direkt. Jeder Fluss fuer sich ist erlaubt, zusammen sind sie
        physikalisch sinnlos - es geht nur der Round-Trip-Wirkungsgrad verloren, waehrend der
        Bonus voll kassiert wird. In einem reinen LP laesst sich das nicht ausdruecken:
        "entweder A oder B" ist keine lineare Aussage. Es braucht je Speicher und Zeitschritt
        eine Binaervariable y und die klassische Big-M-Formulierung

            zufluss[t]  <=  P_laden    * y[t]
            abfluss[t]  <=  P_entladen * (1 - y[t])          y[t] in {0, 1}

        y = 1 erlaubt nur Laden, y = 0 nur Entladen. Als Big-M dient jeweils die ohnehin
        vorhandene Leistungsgrenze des Flusses, damit die Formulierung so eng wie moeglich
        bleibt (lose Big-Ms machen die LP-Relaxierung schwach und das MILP langsam).

        Angesetzt wird an den beiden STORAGE-Fluessen, nicht an den Link- oder
        Wallbox-Fluessen: dort laufen alle Wege zusammen (AC ueber den Link *und* der
        PV-Direktzweig), es gibt also keinen Pfad daran vorbei.

        Preis: aus dem LP wird ein MILP mit einer Binaervariablen je Speicher und Schritt.
        Bei 5856 Schritten und einer Hausbatterie sind das 5856 Binaerariablen - die
        Loesezeit steigt um Groessenordnungen. Deshalb Default AUS: unterhalb von
        (1-eff^2)*Einspeiseverguetung tritt das Kreisen ohnehin nicht auf, und dort ist die
        Nebenbedingung reiner Ballast.
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
                # nur bauen, wo beide Richtungen ueberhaupt existieren
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
        logging.info("Gleichzeitiges Laden/Entladen verboten: %d Binaervariablen, %d Zeilen "
                     "(%s)", len(idx), 2 * len(idx),
                     ", ".join(sorted({lbl for lbl, _ in idx})))

    def _solve(self) -> None:
        """Solve the LP with the configured solver and verify optimality.

        Debug mode (``config.debug``) makes testing fast to inspect: the solver console
        output is shown (``tee``) and a one-line summary — solve time, status,
        termination and objective — is logged. A non-optimal result raises
        ``RuntimeError`` instead of silently continuing with garbage.
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
        """SOC (fraction) each storage reaches at the END of every step.

        oemof indexes ``storage_content`` over periods+1 TIMEPOINTS: index t is the content
        at the START of step t, t+1 at its END. ``_storage_content`` returns the starts
        (0..n-1); this returns the ends (1..n) — the value ``step()`` steers the simulated
        battery to with ``Battery.load(target_soc=...)``.
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
        """Read the solved model into a per-vehicle schedule, a summary and the cost.

        - self._wallbox_schedule: {vid: DataFrame[charge_kW, discharge_kW, net_kW, soc_kWh,
          consumption_kWh]} — charge/discharge are AC power at the vehicle's grid-connector
          bus (what spice_ev applies), consumption_kWh is the per-step driving demand (input).
        - self._summary_df: per-GC grid supply/feed-in, PV, PV feed-in and household demand,
          every battery's SOC, plus per-vehicle wallbox charge/discharge (AC at the GC bus).
        - self._costs: {"objective": <solver objective>}.
        """
        res = processing.results(self.model)
        self._results_main = res
        n = len(self.time_index)
        idx = self.time_index

        # --- per-vehicle wallbox schedule (AC at the GC bus) ---
        charge = {vid: np.zeros(n) for vid in self.vehicle_params}
        discharge = {vid: np.zeros(n) for vid in self.vehicle_params}
        # PV that reaches a storage directly, bypassing the house bus (empty unless the
        # direct branches are built). Keyed like the schedules, so it can be reported and
        # added to the battery plan below.
        pv_to_vehicle = {vid: np.zeros(n) for vid in self.vehicle_params}
        pv_to_battery: Dict[str, np.ndarray] = {}
        wbin_vid: Dict[str, str] = {}     # wallbox terminal bus label -> vid
        pv_wallbox_nodes = []             # resolved after the loop, once wbin_vid is filled
        for node in self.es.nodes:
            if not isinstance(node, cmp.Converter):
                continue
            if node.label.startswith("conv_pv_to_wallbox_"):
                # the label is csid_vid and cannot be split unambiguously — resolve the
                # vehicle through the terminal bus instead
                pv_wallbox_nodes.append(node)
                continue
            if node.label.startswith("conv_pv_to_battery_"):
                bid = node.label[len("conv_pv_to_battery_"):]
                pv_to_battery[bid] = self._flow(res, list(node.inputs)[0], node, n)
                continue
            if node.label.startswith("wallbox_charge_"):
                # source is the GC bus, or the wallbox terminal when the PV branches exist —
                # either way this flow is the TOTAL AC power the station delivers, which is
                # exactly what step() books at the grid connector
                src_bus = list(node.inputs)[0]
                vid = _vid_from_bus(list(node.outputs)[0].label)
                charge[vid] = charge[vid] + self._flow(res, src_bus, node, n)
                wbin_vid[src_bus.label] = vid
            elif node.label.startswith("wallbox_discharge_"):
                b_mob = list(node.inputs)[0]
                gc_bus = list(node.outputs)[0]         # converter -> gc_bus (AC fed back)
                vid = _vid_from_bus(b_mob.label)
                discharge[vid] = discharge[vid] + self._flow(res, node, gc_bus, n)

        for node in pv_wallbox_nodes:
            vid = wbin_vid.get(list(node.outputs)[0].label)
            if vid in pv_to_vehicle:
                pv_to_vehicle[vid] = pv_to_vehicle[vid] + self._flow(
                    res, list(node.inputs)[0], node, n)

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

        # --- per-GC summary: grid supply/feed-in, PV, PV feed-in, household demand + battery SOC ---
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
                summary[f"pv_feedin_{lbl[len('excess_'):]}"] = self._flow(res, list(node.inputs)[0], node, n)
            elif lbl.startswith("converter_pv_to_home_"):
                # TOTAL PV self-consumption: without the direct branches this converter feeds
                # the house bus, with them it feeds bus_pvac and therefore still carries
                # everything that is not exported. Keeps pv - pv_feedin == pv_selfuse valid.
                summary[f"pv_selfuse_{lbl[len('converter_pv_to_home_'):]}"] = \
                    self._flow(res, node, list(node.outputs)[0], n)
            elif lbl.startswith("conv_pvac_to_home_"):      # the household share alone
                summary[f"pv_to_home_{lbl[len('conv_pvac_to_home_'):]}"] = \
                    self._flow(res, node, list(node.outputs)[0], n)
            elif lbl.startswith("household_demand_"):
                summary[lbl] = self._flow(res, list(node.inputs)[0], node, n)
            elif lbl.startswith("home_battery_"):
                summary[f"{lbl}_soc_kWh"] = self._storage_content(res, node, n)
        # per-vehicle wallbox AC power (charge = grid path + PV path / V2H discharge)
        for vid in self.vehicle_params:
            summary[f"wallbox_charge_{vid}"] = charge[vid]
            summary[f"wallbox_discharge_{vid}"] = discharge[vid]
        # the PV that went into a storage directly — the whole point of the direct branches.
        # Only reported when they exist, so existing dumps keep their exact column set.
        if self.config.pv_direct_to_storage:
            for vid, series in pv_to_vehicle.items():
                summary[f"pv_direct_wallbox_{vid}"] = series
            for bid, series in pv_to_battery.items():
                summary[f"pv_direct_battery_{bid}"] = series

        # the price series ACTUALLY used in the objective, per GC (incl. retail markup and
        # negative-price clipping) — the single source of truth for plots/verification
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
                    # charging arrives from the house bus AND, when built, straight from PV;
                    # both must be counted or the plan silently under-reports the power
                    "charge_kW": (self._flow(res, home_bus, node, n)
                                  + pv_to_battery.get(bid, 0.0)),
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

        # --- cost ---
        # The PV charging bonus is an ARTIFICIAL cost: it buys a higher PV share by leaving
        # the cost optimum on purpose. Because it sits on exactly two measurable flows it can
        # be subtracted again EXACTLY, so the energy cost of the chosen schedule stays
        # readable. (spice_ev's own cost calculation never sees the bonus at all — it runs
        # after the simulation on the physical GC timeseries, so its EUR/a are always real
        # money.) The extra keys only appear with the feature on, so old dumps stay identical.
        objective = float(self.model.objective())
        self._costs = {"objective": objective}
        if self.config.pv_direct_to_storage:
            step_hours = 1.0 / self._loss_factor()
            bonus_ct = -(
                float(self.config.pv_charge_bonus_vehicle_ct_kWh)
                * float(sum(np.sum(s) for s in pv_to_vehicle.values()))
                + float(self.config.pv_charge_bonus_battery_ct_kWh)
                * float(sum(np.sum(s) for s in pv_to_battery.values()))
            ) * step_hours
            self._costs["pv_bonus_ct"] = bonus_ct          # contribution to the objective (<= 0)
            self._costs["objective_ohne_bonus"] = objective - bonus_ct

    def _save_results(self) -> None:
        """Write the schedule, summary and cost as CSV into ``config.output_dir``."""
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
        """Return the per-vehicle wallbox schedule (charge/discharge/net/soc per step).

        The columns ``charge_kW`` and ``discharge_kW`` are the contract consumed by
        ``OemofSolve.commands_from_oemof``; ``net_kW``, ``soc_kWh`` and ``consumption_kWh``
        are extras.
        """
        if self._wallbox_schedule is None:
            raise RuntimeError("get_wallbox_schedule() called before _extract_results()")
        return self._wallbox_schedule

    def get_plan(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Return the full optimized plan, grouped by component type.

        Row ``k`` of every DataFrame corresponds exactly to spice_ev simulation step ``k``
        (the time index is built that way), so the strategy can look values up by step.

        - ``"vehicles"``:  {vid: DataFrame[charge_kW, discharge_kW, net_kW, soc_kWh,
          consumption_kWh]} — AC power at the vehicle's GC bus (= get_wallbox_schedule()).
        - ``"batteries"``: {bid: DataFrame[charge_kW, discharge_kW]} — AC power of the
          stationary battery's link at its GC bus; to be APPLIED by the strategy.
        - ``"grid"``:      {gcid: DataFrame[supply_kW, feedin_kW]} — planned exchange per
          grid connector; NOT applied, serves to verify the executed plan.
        """
        if self._wallbox_schedule is None:
            raise RuntimeError("get_plan() called before _extract_results()")
        return {
            "vehicles": self._wallbox_schedule,
            "batteries": self._battery_schedule or {},
            "grid": self._grid_schedule or {},
        }

    def _export_graph(self) -> None:
        """Render the built energy system as an SVG topology graph (when ``export_graph``).

        Uses ``oemof.network.graph.create_nx_graph`` -> DOT -> Graphviz ``dot -Tsvg`` and
        writes ``<output_dir>/<dump_filename>_graph.svg`` (+ .dot), coloured by node type.
        If Graphviz ``dot`` is not on the PATH, only the .dot file is written.
        """
        import shutil
        import subprocess
        from oemof.network.graph import create_nx_graph

        style = {
            "bus": ("box", "#dbeafe", "#1e40af"), "source": ("ellipse", "#dcfce7", "#15803d"),
            "sink": ("ellipse", "#fee2e2", "#b91c1c"), "converter": ("box", "#ffffff", "#475569"),
            "link": ("box", "#eef2ff", "#4338ca"), "storage": ("cylinder", "#fef9c3", "#a16207"),
            "other": ("box", "#f1f5f9", "#64748b"),
            # the bonus-carrying PV branches stand out — they are the ones to check
            "pv_direct": ("box", "#fef3c7", "#b45309"),
        }

        def kind(n):
            if str(n.label).startswith(("conv_pv_to_wallbox_", "conv_pv_to_battery_")):
                return "pv_direct"
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


###########################################################################
# Standalone entry point
###########################################################################
def main() -> None:
    """Standalone smoke test with synthetic inputs — intentionally not implemented.

    The model is driven by ``OemofSolve`` (spice_ev.strategies.oemof_solve), which builds
    all inputs from a spice_ev scenario. For a runnable end-to-end example see the notebook
    ``systemoptimierung/examples/example_1/test_run.ipynb``.
    """
    raise NotImplementedError(
        "No standalone entry point: run the model through the spice_ev strategy "
        "'oemof_solve' (see systemoptimierung/examples/example_1/test_run.ipynb)."
    )


if __name__ == "__main__":
    main()

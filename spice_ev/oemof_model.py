"""
oemof-Energiesystem - das LP hinter der spice_ev-Strategie ``oemof_solve``.

Autor: Alaa Alsleman, GitHub: Alaadin17

Baut aus einem spice_ev-Szenario ein oemof.solph-Energiesystem, loest es EINMAL ueber den
ganzen Horizont und gibt den Fahrplan an spice_ev zurueck.

Topologie - ein Bus je Netzanschluss
------------------------------------
Alles wird BEDINGT gebaut: ein Netzanschluss entsteht nur, wenn er etwas traegt (genutzte
Ladestation, Last, PV oder Batterie), eine Wallbox nur, wenn ein Fahrzeug sie wirklich
nutzt. Siehe ``_create_components``.

    Home_<n>            AC-Bus je aktivem Netzanschluss   {gcid: bus} in self._gc_bus
      grid_supply_<n>     Netzbezug            grid_feedin_<n>   Export (enable_grid_feedin)
      household_demand    feste Last           bus_pv_<n>        PV-Bus mit pv_/excess_
      link_home_battery_<bid> <-> bus_battery_<bid> <-> home_battery_<bid>
      wallbox_charge_<csid>_<vid> -> bus_mobility_<vid> -> bev_battery_<vid>
      wallbox_discharge_<csid>_<vid>  nur fuer v2g-Fahrzeuge mit enable_v2h

Zwei Entscheidungen, die den Fahrplan mit spice_ev deckungsgleich halten:
- Der Lade-/Entladeverlust sitzt IM SPEICHER (inflow/outflow_conversion_factor), nicht im
  Link und nicht in der Wallbox - genau wie in spice_evs ``Battery``. Round-Trip eff^2.
- Wallboxen und Links sind VERLUSTFREI, sie begrenzen nur die Leistung. Ein Verlust dort
  wuerde doppelt zaehlen und den geplanten SOC vom simulierten wegdriften lassen.

Die PV speist ueber ihren Wechselrichter auf den Hausbus, alles Uebrige geht ueber
``excess_<n>`` in die Einspeisung. Ab dem Hausbus ist eine PV-kWh nicht mehr von einer
Netz-kWh zu unterscheiden - eine Zuordnung "so viel PV ging ins Auto" gibt es deshalb
nicht. Benannte Direktzweige samt PV-Ladebonus gab es einmal; sie sind entfernt, weil sie
den Fahrplan nicht veraendert haben und ihre Kennzahl ohne den Bonus entartet war.

Eingaben (__init__)
-------------------
Im Produktivpfad baut ``OemofSolve.build_oemof_inputs`` sie aus dem spice_ev-Szenario:
config (SystemConfig), time_index (= die spice_ev-Schritte), grid_connectors (Last, PV,
max_power, optional Preisreihe), charging_stations, vehicle_params, battery_params.

Ablauf
------
``run()``: Zeitraster -> Energiesystem -> Komponenten -> [Graph] -> LP -> loesen ->
auslesen -> CSV. ``get_plan()`` / ``get_wallbox_schedule()`` liefern das Ergebnis zurueck.
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
# We collect all settings in one typed object, SystemConfig. It can be filled
# two ways:
#   - from a flat dict  -> comes from the OemofSolve strategy (oemof_config)
#   - from a cfg file   -> for standalone runs
@dataclass
class SystemConfig:
    """Alle Einstellungen des Modells an einer Stelle.

    Gefuellt aus den ``oemof_``-Schluesseln der simulate.cfg (``from_options``). Was das
    Szenario je Komponente mitbringt - max_power, Kapazitaet, SOC, Wirkungsgrad - schlaegt
    diese Werte; die Felder hier sind der Rueckfall.
    """

    # Time parameters
    start_date: str = "2025-01-01"
    periods: int = 96  # 15-minute steps (96 = 1 day for debug)
    freq: str = "15min"

    # Schalter fuer Wege, die das Szenario nicht ausdruecken kann. Was es ausdruecken KANN -
    # ob ein Netzanschluss PV hat, ob eine Batterie existiert, ob es Fahrzeuge gibt - wird
    # nicht doppelt geschaltet: PV und Speicher entstehen genau dann, wenn das Szenario sie
    # mitbringt. (Frueher gab es dafuer enable_pv und enable_battery; sie standen immer auf
    # true und haben die Szenario-Information nur wiederholt.)
    enable_pv_to_home: bool = True    # ohne den Wechselrichter kann PV NUR einspeisen -
    #                                   das ist im Szenario nicht darstellbar
    enable_grid_feedin: bool = True   # Export vom Hausbus (Batterie/V2G) erlauben

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

    # Tiny anti-degeneracy cost (ct/kWh) on storage charging and V2H feed-back. Without it
    # the LP may cycle energy pointlessly (storage out -> in, or wallbox charge+discharge in
    # the same step) because that changes the objective by exactly zero — the SOC series then
    # drifts to its floor for no reason. 0.001 is far below any real price and does not alter
    # genuine decisions; it only makes useless cycling strictly worse than doing nothing.
    storage_cycle_penalty: float = 0.001

    # Verbietet einem Speicher, im SELBEN Zeitschritt zu laden und zu entladen. Kostet je
    # Speicher und Zeitschritt eine BINAERVARIABLE: aus dem LP wird ein MILP, die Loesezeit
    # steigt deutlich.
    #
    # Gebraucht wurde das gegen den PV-Ladebonus, der es lohnend machen konnte, PV DURCH
    # einen Speicher ins Haus zu leiten. Diesen Bonus gibt es nicht mehr, und ohne ihn ist
    # das Kreisen schon durch storage_cycle_penalty teurer als Nichtstun. Der Schalter
    # bleibt fuer eigene Experimente - im Normalbetrieb braucht ihn niemand.
    forbid_simultaneous_storage: bool = False

    # --- Preise (ct/kWh) -------------------------------------------------------------
    # grid_variable_costs ist der Bezugspreis, wenn das Szenario keinen mitbringt: der
    # KOMPLETTE Preis, so wie er auf der Stromrechnung steht - kein Boersenpreis, auf den
    # noch etwas addiert wird. Ein Preisblatt wird nicht gelesen, es gibt keinen
    # Tarif-Aufschlag (Netzentgelt, Umlagen, Konzessionsabgabe, Stromsteuer, MwSt) und
    # keinen Leistungspreis.
    #
    # Bringt das Szenario Preissignale mit - ``include_price_csv`` in der generate.cfg macht
    # aus jeder CSV-Zeile ein GridOperatorSignal -, gilt STATTDESSEN diese Zeitreihe, je
    # Zeitschritt, in genau der Form, die auch spice_evs eigene Strategien sehen (siehe
    # OemofSolve._grid_price_series). grid_variable_costs ist dann wirkungslos.
    #
    # grid_feedin_tariff ist immer fest und gilt fuer BEIDE Exportwege (PV-Ueberschuss und
    # Export vom Hausbus). Negativ = Erloes, 0 = keine Verguetung.
    #
    # WICHTIG - die spice_ev-Kostenrechnung geht ihren eigenen Weg: simulate.py wertet nach
    # der Simulation costs.py aus, und das rechnet mit dem Preisblatt (Netzentgelt, Umlagen,
    # Steuern, Leistungspreis). Die EUR/a in results.json entstehen also aus anderen Preisen
    # als der Fahrplan - sie beantworten "was haette das gekostet", der Fahrplan beantwortet
    # "was ist bei diesem Preis sinnvoll". Beide Zahlen sind fuer sich richtig, nur nicht
    # dieselbe Rechnung.
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 35.0
    grid_feedin_tariff: float = 0.0   # negativ = Erloes; 0 = Einspeisung bringt nichts

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
        """Aus dem cfg-Dict der Strategie. ``oemof_`` faellt weg, Unbekanntes warnt."""
        config = cls()                                  # start with all defaults
        typ = {f.name: f.type for f in fields(cls)}     # Feldname -> deklarierter Typ
        for key, value in (options or {}).items():
            name = key.removeprefix("oemof_")           # "oemof_solver" -> "solver"
            if name not in typ:
                logging.warning("Unknown oemof parameter ignored: %s", key)
                continue
            setattr(config, name, _coerce(name, value, typ[name]))
        return config


def _as_array(values, n):
    """Skalar/Liste/Series -> float-Reihe der Laenge n.

    Ein Skalar wird gestreckt, eine zu lange Reihe abgeschnitten. ACHTUNG: eine zu kurze
    wird still mit Nullen aufgefuellt - ein zu kurzes PV-Profil heisst dann "keine Sonne".
    """
    if np.isscalar(values):
        return np.full(n, float(values))
    arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr[:n]


_WAHR = {"true", "yes", "on", "1"}
_FALSCH = {"false", "no", "off", "0"}


def _coerce(name, value, typ):
    """Einen cfg-Wert auf den deklarierten Feldtyp bringen.

    Die cfg wird per ``json.loads`` gelesen, und JSON kennt nur kleines ``true``/``false``.
    ``oemof_enable_v2h = False`` bleibt deshalb der STRING "False" - und der ist in Python
    wahr, der Schalter waere still AN. Genau das ist schon einmal passiert.
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
        logging.warning("oemof_%s: '%s' ist kein Wahrheitswert (erwartet true/false)",
                        name, value)
        return bool(value)
    if "float" in ziel or "int" in ziel:
        try:
            zahl = float(value)
        except (TypeError, ValueError):
            logging.warning("oemof_%s: '%s' ist keine Zahl - Wert wird uebernommen wie er ist",
                            name, value)
            return value
        return int(zahl) if "int" in ziel and "float" not in ziel else zahl
    return value


def _vid_from_bus(label) -> str:
    """Fahrzeug-ID aus seinem Busnamen ``bus_mobility_<vid>``.

    Der Umweg ueber den Bus statt ueber das Wallbox-Label ist noetig, weil
    ``wallbox_charge_<csid>_<vid>`` nicht eindeutig zerlegbar ist - beide Teile duerfen
    Unterstriche enthalten.
    """
    label = str(label)
    prefix = "bus_mobility_"
    return label[len(prefix):] if label.startswith(prefix) else label


###########################################################################
# Main class
###########################################################################
class EnergySystemModel:
    """Baut das Energiesystem, loest es einmal ueber den ganzen Horizont und liest den
    Fahrplan aus. Topologie siehe Modulkopf, Reihenfolge siehe ``run()``."""

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
        """Die ganze Pipeline: Zeitraster -> Topologie -> LP -> loesen -> auslesen -> CSV."""
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
        """Die Topologie je Netzanschluss - hier faellt die Entscheidung, was gebaut wird.

        Erst je Fahrzeug ein Bus und ein Speicher, dann die Ladestationen (nur genutzte, und
        nur zu den Fahrzeugen, die sie wirklich nutzen), dann je aktivem Netzanschluss ein
        Bus ``Home_<n>`` mit allem, was daran haengt. Aktiv heisst: genutzte Wallbox ODER
        Last ODER PV ODER Batterie - sonst entsteht der Anschluss gar nicht.
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
            has_pv = nonzero(gc.get("pv"))
            has_bat = gc_has_battery(gcid)
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

        # Bezugsquelle. Liefert die Strategie eine Preisreihe aus dem Szenario
        # (include_price_csv), gilt die je Zeitschritt - sonst der feste cfg-Wert.
        preis = gc.get("price_ct_kWh")
        supply = cmp.Source(
            label=f"grid_supply_{name}",
            outputs={b: flows.Flow(
                nominal_value=float(gc.get("max_power", self.config.grid_supply_power_kW)),
                variable_costs=(_as_array(preis, periods) if preis is not None
                                else self.config.grid_variable_costs))},
        )
        self.es.add(supply)
        # Einspeisung: derselbe feste Wert fuer beide Wege - den PV-Ueberschuss und den
        # Export vom Hausbus (Batterie/V2G). Wichtig ist nur, dass er nicht POSITIV verguetet
        # wird, waehrend Bezug billiger ist: sonst kauft das LP Strom und speist ihn im selben
        # Schritt gewinnbringend wieder ein.
        feedin_tariff = homebus_tariff = float(self.config.grid_feedin_tariff)
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
        # PV on this GC; converter limit = installed plant size when the scenario has one
        pv = gc.get("pv")
        if pv is not None and float(np.sum(_as_array(pv, periods))) > 0.0:
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
        """PV an einem Netzanschluss: PV-Bus, Quelle, Ueberschuss-Senke, Wechselrichter.

        Der Wechselrichter traegt ``nominal_value = converter_power_kW`` und speist auf den
        Hausbus. Was dort nicht gebraucht wird, geht ueber ``excess_<name>`` in die
        Einspeisung. Ab dem Hausbus ist eine PV-kWh nicht mehr von einer Netz-kWh zu
        unterscheiden - eine Zuordnung "so viel PV ging ins Auto" gibt es deshalb nicht.
        Dafuer existierten einmal benannte Direktzweige; sie sind entfernt, weil sie den
        Fahrplan nicht veraendert haben und ihre Kennzahl ohne einen Bonus, der sie
        eindeutig macht, entartet war.
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
        """Eine Hausbatterie: Bus, Speicher, AC/DC-Link. Je Eintrag einmal aufgerufen.

        Erreichbar nur ueber den Link vom Hausbus - einen PV-Direktzweig gibt es hier
        nicht mehr. Er haette die PV zwar zurechenbar gemacht, aber am Fahrplan nichts
        geaendert und im Gegenzug den Kreislauf ermoeglicht: PV in die Batterie, Bonus
        kassieren, im selben Schritt wieder ins Haus entladen.

        Der Verlust sitzt IM SPEICHER (inflow/outflow_conversion_factor), genau wie in
        spice_evs ``Battery``: eff beim Laden, eff beim Entladen, Round-Trip eff^2. Der Link
        ist verlustfrei und begrenzt nur die Leistung. Die frueher hier verwendete Variante
        mit sqrt(eff) je Linkrichtung ergab einen Round-Trip von nur eff - die LP-Batterie
        war damit ~5 % besser als die simulierte, und geplante Entladungen liefen leer.
        """
        capacity = float(bp.get("capacity_kWh", self.config.battery_capacity_kWh))
        power = float(bp.get("power_kW", self.config.battery_max_power_kW))
        discharge_power = float(bp.get("discharge_power_kW", power))
        init = float(bp.get("initial_soc", self.config.battery_initial_soc))
        init = min(max(init, self.config.battery_min_soc), self.config.battery_max_soc)
        efficiency = float(bp.get("efficiency", self.config.battery_efficiency))

        # EIN Bus traegt beide Richtungen. Eine Trennung in Zu- und Abflussseite braucht es
        # nur, wenn ein zweiter Weg IN den Speicher fuehrt, der sich sonst am Speicher vorbei
        # durch den Link zurueck ins Haus stehlen koennte. Den gab es mit dem PV-Direktzweig
        # zur Batterie; seit der weg ist, ist der Link der einzige Zugang.
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
            # tiny penalty on BOTH directions kills free storage-cycling degeneracy
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
        # an den beiden Speicherfluessen ist "laden" und "entladen" eindeutig - hier setzt
        # forbid_simultaneous_storage an, falls es gebraucht wird
        self._storage_pairs.append({
            "label": f"home_battery_{bid}", "storage": storage,
            "in_bus": b_bat_in, "out_bus": b_bat_out,
            "p_in": power, "p_out": discharge_power,
        })

    def _loss_factor(self) -> float:
        """kWh/step -> kW factor: oemof multiplies fixed_losses_absolute by the step
        duration (hours), so divide the per-step kWh by the step hours."""
        if len(self.time_index) > 1:
            step_hours = (self.time_index[1] - self.time_index[0]) / pd.Timedelta(hours=1)
        else:
            step_hours = 0.25
        return 1.0 / step_hours

    def _add_vehicle(self, vid, params) -> None:
        """Ein Fahrzeug: Mobilitaetsbus + Speicher. Die Wallboxen baut ``_add_wallbox``.

        - consumption ist der Fahrbedarf als ``fixed_losses_absolute`` - er wird dem Speicher
          auch abgezogen, waehrend das Auto weg ist.
        - Der SOC-Boden ist ``min_soc``, bei V2H/V2G-faehigen Fahrzeugen
          ``max(min_soc, discharge_limit)``.
        - min_soc_series hebt diesen Boden SCHRITTWEISE auf den ``desired_soc`` aus dem
          Szenario, jeweils im Schritt vor der Abfahrt - damit das Auto so voll ist, wie
          spice_ev es erwartet. Gedeckelt auf ``max_soc``, sonst waere der Speicher unloesbar.

        Busse und Flags landen in ``self._vehicle_nodes[vid]``, damit die Wallboxen sie finden.
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
        # V2G entlaedt mit der discharge_curve, die spice_ev aus der Ladekurve mal
        # v2g_power_factor bildet - typisch die Haelfte. Ohne diese Grenze plant das LP mit
        # der vollen Stationsleistung und der geplante SOC laeuft vom simulierten weg.
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
        efficiency = float(params.get("efficiency", self.config.bev_efficiency))

        # EIN Bus traegt beide Richtungen. Eine Trennung in Zu- und Abflussseite gab es, als
        # ein bonusberechtigter PV-Direktzweig hier ankam und sonst durch den Speicher
        # hindurch ins Haus haette rutschen koennen. Ohne diesen Zweig fuehrt jeder Weg ins
        # Fahrzeug ueber die Wallbox, und die Kosten machen den Durchgang unattraktiv.
        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)
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
            # kein "or []": connected_cs ist ein numpy-Array, dessen Wahrheitswert wirft
            angeschlossen = params.get("connected_cs")
            used = {c for c in ([] if angeschlossen is None else list(angeschlossen)) if c}
            p_in = max((float(self.charging_stations.get(c, {}).get(
                "max_power", self.config.wallbox_power_kW)) for c in used),
                default=self.config.wallbox_power_kW)
            self._storage_pairs.append({
                "label": f"bev_battery_{vid}", "storage": bev,
                "in_bus": b_mob_in, "out_bus": b_mobility,
                "p_in": p_in, "p_out": min(p_in, p_entladen or p_in),
            })

        self._vehicle_nodes[vid] = {
            "bus": b_mobility,      # Laden kommt hier an, V2H geht hier weg
            "can_discharge": can_discharge,
            "p_discharge": p_entladen,   # None = nur die Stationsleistung begrenzt
            "connected_cs": list(params.get("connected_cs", [])),
        }

    def _add_wallbox(self, csid, users, gc_bus) -> None:
        """Eine Ladestation, verbunden mit den Fahrzeugen, die sie wirklich nutzen.

        Je Fahrzeug ein maskierter Converter: ``max`` ist 1 genau in den Schritten, in denen
        das Auto an DIESER Station steckt, sonst 0. Fuer v2g-Fahrzeuge mit ``enable_v2h``
        zusaetzlich der Rueckweg ins Haus.

        Die Wallbox ist VERLUSTFREI und begrenzt nur die Leistung - wie in spice_ev. Der
        Lade-/Entladeverlust steckt in der Fahrzeugbatterie; ein Verlust auch hier wuerde
        doppelt zaehlen und den geplanten SOC vom simulierten wegdriften lassen.
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
            b_mob = node["bus"]
            # Limit the AC side (-> wallbox): that is the power spice_ev commands and clamps
            # against cs.max_power. Limiting the DC output instead would let the AC draw
            # reach max_power/efficiency and exceed the station's rating.
            wb_charge = cmp.Converter(
                label=f"wallbox_charge_{csid}_{vid}",
                inputs={gc_bus: flows.Flow(max=mask, nominal_value=power,
                                           variable_costs=late_costs)},
                outputs={b_mob: flows.Flow()},
                conversion_factors={b_mob: self.config.wallbox_efficiency_charge},
            )
            self.es.add(wb_charge)
            if node["can_discharge"]:
                # Die Entladeleistung ist NICHT die Ladeleistung: spice_ev begrenzt sie auf
                # die discharge_curve (Ladekurve mal v2g_power_factor). Wer hier die volle
                # Stationsleistung zulaesst, plant mehr Rueckspeisung, als die Simulation
                # liefern kann - der SOC laeuft dann auseinander.
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
        """Aus dem Energiesystem das LP bauen: Busbilanzen, Leistungsgrenzen,
        Speichergleichungen und die Zielfunktion. Mit ``debug`` faellt eine lesbare
        ``.lp``-Datei ab."""
        self.model = Model(self.es)
        self._add_no_simultaneous_constraints()   # optional, macht aus dem LP ein MILP
        if self.config.debug:
            lp_path = Path(self.config.output_dir) / f"{self.config.dump_filename}_debug.lp"
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    def _add_no_simultaneous_constraints(self) -> None:
        """Verbiete jedem Speicher, im selben Schritt zu laden UND zu entladen.

        "Entweder A oder B" ist keine lineare Aussage, es braucht je Speicher und Schritt
        eine Binaervariable:

            zufluss[t]  <=  P_laden    * y[t]
            abfluss[t]  <=  P_entladen * (1 - y[t])          y[t] in {0, 1}

        Big-M ist jeweils die ohnehin vorhandene Leistungsgrenze - je enger, desto schneller
        das MILP. Angesetzt an den STORAGE-Fluessen, weil dort jeder Weg zusammenlaeuft.

        Gebraucht wurde das gegen den PV-Ladebonus: er machte es lohnend, PV *durch* den
        Speicher ins Haus zu leiten, weil jeder Fluss fuer sich erlaubt war und der Umweg
        nur den Round-Trip kostete, waehrend der Bonus voll kassiert wurde. Den Bonus gibt
        es nicht mehr, damit auch den Anreiz nicht - storage_cycle_penalty macht das Kreisen
        ohnehin teurer als Nichtstun. Default AUS; der Schalter bleibt fuer Experimente.

        Preis: aus dem LP wird ein MILP. 5856 Schritte sind 5856 Binaervariablen je Speicher,
        die Loesezeit steigt um Groessenordnungen.
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
        """LP loesen und die Optimalitaet pruefen.

        Ein nicht-optimales Ergebnis wirft ``RuntimeError``, statt still mit Unsinn
        weiterzurechnen. Mit ``debug`` zeigt der Solver seine Konsole und eine Zeile mit
        Dauer, Status und Zielwert geht ins Log.
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
        """Der SOC, den jeder Speicher am ENDE jedes Schritts erreicht.

        oemof indiziert ``storage_content`` ueber n+1 Zeitpunkte: t ist der Stand zu BEGINN
        von Schritt t. Hier zaehlen die Enden (1..n) - genau der Wert, auf den ``step()``
        die simulierte Batterie mit ``Battery.load(target_soc=...)`` fahren laesst.
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
        """Das geloeste Modell in Fahrplan, Zusammenfassung und Kosten uebersetzen.

        Alle Leistungen sind AC am Netzanschlussbus - das ist, was spice_ev anwendet.
        Es entstehen ``_wallbox_schedule`` je Fahrzeug, ``_summary_df`` je Netzanschluss
        (Bezug, Einspeisung, PV, Last, SOC) und ``_costs``.
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
                # dieser Fluss ist die GESAMTE AC-Leistung der Station - genau das, was
                # step() am Netzanschluss verbucht
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

        # --- Zusammenfassung je GC: Netzbezug/-einspeisung, PV, PV-Einspeisung,
        #     Haushaltslast und der SOC jeder Batterie ---
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
                # die gesamte PV-Eigennutzung: alles, was nicht exportiert wird, laeuft ueber
                # diesen Wechselrichter. Damit gilt pv - pv_feedin == pv_selfuse. WOHIN die
                # kWh danach geht, ist am Hausbus nicht mehr unterscheidbar.
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

        # Der Preis, mit dem die Zielfunktion wirklich gerechnet hat - je nach Szenario eine
        # Konstante oder die Stufenfunktion aus der Preis-CSV. So oder so mitgeschrieben,
        # damit Plots und Pruefungen eine Quelle haben und nicht die cfg nachschlagen muessen.
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
                    # der Link vom Hausbus ist der einzige Zugang zur Batterie
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

        # --- Kosten ---
        # Der Zielwert in ct. Er enthaelt nur echte Preise: Bezug, Einspeisung und die
        # winzigen Tie-Breaker (storage_cycle_penalty, late_charging_penalty). Kuenstliche
        # Anreize gibt es keine mehr - der frueher hier verrechnete PV-Ladebonus ist samt
        # seiner Zweige entfernt. Die EUR/a in results.json entstehen ohnehin getrennt
        # davon, nach der Simulation aus der physikalischen Zeitreihe.
        self._costs = {"objective": float(self.model.objective())}

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
        """Der Fahrplan je Fahrzeug. ``charge_kW``/``discharge_kW`` sind der Vertrag mit
        ``OemofSolve.commands_from_oemof``, der Rest ist Zugabe."""
        if self._wallbox_schedule is None:
            raise RuntimeError("get_wallbox_schedule() called before _extract_results()")
        return self._wallbox_schedule

    def get_plan(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Der ganze Fahrplan, nach Komponententyp gruppiert. Zeile k ist Simulations-
        schritt k, die Strategie kann also direkt nachschlagen.

        - ``vehicles``  je Fahrzeug, wird von ``step()`` angewendet
        - ``batteries`` je Hausbatterie, wird ebenfalls angewendet
        - ``grid``      geplanter Netzaustausch - NICHT angewendet, dient der Kontrolle
        """
        if self._wallbox_schedule is None:
            raise RuntimeError("get_plan() called before _extract_results()")
        return {
            "vehicles": self._wallbox_schedule,
            "batteries": self._battery_schedule or {},
            "grid": self._grid_schedule or {},
        }

    def _export_graph(self) -> None:
        """Die Topologie als SVG zeichnen (nur mit ``export_graph``), nach Knotentyp
        eingefaerbt. Ohne Graphviz auf dem PATH entsteht nur die .dot-Datei."""
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

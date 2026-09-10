"""Tests for the per-grid-connector oemof model (spice_ev/oemof_model.py).

Autor: Alaa Alsleman, GitHub: Alaadin17

Die Tests ohne Solver bauen nur das EnergySystem (oder rufen einen Helfer) und pruefen
Zuschnitt und Verdrahtung je Netzanschluss, die Zuordnung von Last und PV, die Preisquelle
und die Typumwandlung der cfg-Werte. Die uebrigen brauchen CBC (ohne ihn werden sie
uebersprungen) und rechnen wirklich: Debug-Modus, SOC-Boden aus ``desired_soc``,
PV-Direktzweige, Speicher-Bustrennung und ein vollstaendiger ``run()`` samt Dumps.
"""
import dataclasses
import logging
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spice_ev.oemof_model import EnergySystemModel, SystemConfig
from spice_ev.strategies.oemof_solve import OemofSolve


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build_es(m):
    """Run the four build stages (no solver) and return the set of node labels."""
    m._create_time_index()
    m._create_energy_system()
    m._create_components()
    return {n.label for n in m.es.nodes}


def _nodes(m):
    return {n.label: n for n in m.es.nodes}


# Diese Tests loesen wirklich - ohne Solver haben sie nichts zu sagen.
requires_cbc = pytest.mark.skipif(shutil.which("cbc") is None,
                                  reason="CBC solver not installed")


def _out_flow(node):
    return list(node.outputs.values())[0]


def _seq(flow, n):
    """Read a Flow's fixed profile as a plain float list of length n."""
    return [float(flow.fix[t]) for t in range(n)]


# ---------------------------------------------------------------------------
# Test 1 — per-GC topology & pruning (no solve)
# ---------------------------------------------------------------------------
def test_per_gc_topology_and_pruning():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False),
        time_index=idx,
        grid_connectors={
            "GC1": {"max_power": 30.0, "load": [1, 1, 1, 1], "pv": [0, 3, 3, 0]},
            "GC2": {"max_power": 50.0},           # only CS2 is used -> active
            "GC3": {"max_power": 20.0},           # nothing at all -> pruned
        },
        charging_stations={
            "CS1": {"max_power": 11.0, "parent": "GC1"},
            "CS2": {"max_power": 22.0, "parent": "GC2"},
            "CS3": {"max_power": 11.0, "parent": "GC1"},   # never used -> no wallbox
        },
        battery_params={"BAT1": {"capacity_kWh": 10.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={
            "v1": {"capacity_kWh": 50.0, "v2g": True,
                   "connected_cs": ["CS1", "CS1", "CS1", None], "consumption": [0, 0, 0, 0]},
            "v2": {"capacity_kWh": 40.0, "v2g": False,
                   "connected_cs": ["CS2", "CS2", "CS2", "CS2"], "consumption": [0, 0, 0, 0]},
            "v3": {"capacity_kWh": 40.0, "v2g": False,
                   "connected_cs": [None, None, None, None], "consumption": [0, 0, 0, 0]},
        },
    )
    labels = _build_es(m)
    nodes = _nodes(m)

    # GC pruning + naming: GC1 -> Home_1, GC2 -> Home_2, GC3 dropped
    assert "Home_1" in labels and "Home_2" in labels and "Home_3" not in labels
    assert {"grid_supply_Home_1", "grid_supply_Home_2"} <= labels
    assert {"grid_feedin_Home_1", "grid_feedin_Home_2"} <= labels

    # load / PV only where the GC actually has them
    assert "household_demand_Home_1" in labels and "household_demand_Home_2" not in labels
    assert "pv_Home_1" in labels and "pv_Home_2" not in labels

    # stationary battery placed on its parent GC (GC1)
    assert "home_battery_BAT1" in labels and "link_home_battery_BAT1" in labels

    # wallbox pruning: used CS only, connected only to using vehicles
    assert {"wallbox_charge_CS1_v1", "wallbox_charge_CS2_v2"} <= labels
    assert "wallbox_discharge_CS1_v1" in labels          # v1 is v2g -> V2H
    assert "wallbox_discharge_CS2_v2" not in labels       # v2 is not v2g
    assert not any(lbl.startswith("wallbox_charge_CS3_") for lbl in labels)   # CS3 unused
    assert not any(lbl.startswith("wallbox_") and lbl.endswith("_v3")
                   for lbl in labels)                   # v3 unused
    assert "bus_mobility_v3" in labels                    # ... but every vehicle keeps a bus

    # grid source power = GC.max_power
    assert _out_flow(nodes["grid_supply_Home_1"]).nominal_value == 30.0
    assert _out_flow(nodes["grid_supply_Home_2"]).nominal_value == 50.0

    # each wallbox hangs on its CS's parent GC bus
    assert list(nodes["wallbox_charge_CS1_v1"].inputs.keys())[0].label == "Home_1"
    assert list(nodes["wallbox_charge_CS2_v2"].inputs.keys())[0].label == "Home_2"

    # non-v2g vehicles have NO storage outflow (blocks the energy-burning money pump at
    # negative prices); v2g vehicles keep an open outflow for V2H
    assert _out_flow(nodes["bev_battery_v2"]).nominal_value == 0
    assert _out_flow(nodes["bev_battery_v1"]).nominal_value is None


# ---------------------------------------------------------------------------
# Test 2 — per-GC load/PV grouped from spice_ev events (no solve)
# ---------------------------------------------------------------------------
def _ev(values, start, gcid):
    """A minimal EnergyValuesList stand-in (15-min steps) for one grid connector."""
    return SimpleNamespace(values=values, step_duration_s=900, start_time=start,
                           factor=1, grid_connector_id=gcid)


def test_per_gc_load_pv_from_events():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    start = idx[0]

    strat = OemofSolve.__new__(OemofSolve)   # bypass __init__, we only need two attrs
    strat.events = SimpleNamespace(
        fixed_load_lists={"L1": _ev([1, 1, 1, 1], start, "GC1"),
                          "L2": _ev([2, 2, 2, 2], start, "GC2")},
        local_generation_lists={"PV1": _ev([0, 3, 3, 0], start, "GC1")},
    )
    strat.world_state = SimpleNamespace(
        grid_connectors={"GC1": SimpleNamespace(max_power=30.0),
                         "GC2": SimpleNamespace(max_power=50.0)},
    )

    gcs = strat._grid_connectors(idx, _roh())

    # grouped strictly by grid_connector_id
    assert list(gcs["GC1"]["load"]) == [1, 1, 1, 1]
    assert list(gcs["GC1"]["pv"]) == [0, 3, 3, 0]
    assert list(gcs["GC2"]["load"]) == [2, 2, 2, 2]
    assert list(gcs["GC2"]["pv"]) == [0, 0, 0, 0]          # GC2 has no PV
    assert gcs["GC1"]["max_power"] == 30.0 and gcs["GC2"]["max_power"] == 50.0

    # feed the per-GC result into the model -> lands on the matching Home_N bus
    m = EnergySystemModel(config=SystemConfig(debug=False), time_index=idx,
                          grid_connectors=gcs)
    labels = _build_es(m)
    nodes = _nodes(m)

    assert _seq(list(nodes["household_demand_Home_1"].inputs.values())[0], 4) == [1, 1, 1, 1]
    assert _seq(list(nodes["household_demand_Home_2"].inputs.values())[0], 4) == [2, 2, 2, 2]
    assert _seq(_out_flow(nodes["pv_Home_1"]), 4) == [0, 3, 3, 0]
    assert "pv_Home_2" not in labels


# ---------------------------------------------------------------------------
# Test 3 — solve a tiny feasible model with CBC (debug mode on)
# ---------------------------------------------------------------------------
@requires_cbc
def test_solve_small_model(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)   # keep the debug LP dump inside the tmp dir
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=True, enable_grid_feedin=False, output_dir="lp_out"),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1]}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0,
                               "connected_cs": ["CS1"] * 4, "consumption": [0, 0, 0, 0]}},
    )
    _build_es(m)
    m._optimize()
    with caplog.at_level(logging.INFO):
        m._solve()             # raises RuntimeError if not optimal

    assert math.isfinite(m.model.objective())
    assert (tmp_path / "lp_out" / "dump_debug.lp").exists()    # LP dump in config.output_dir
    assert "oemof solved" in caplog.text                        # _solve debug log line


# ---------------------------------------------------------------------------
# Test 2a2 — feste Preise: die cfg ist die einzige Quelle
# ---------------------------------------------------------------------------
def test_prices_come_from_the_config_or_the_scenario_and_nowhere_else():
    """Bezugspreis: Szenario, sonst cfg. Einspeiseverguetung: immer cfg.

    Frueher konnte ein Preis aus drei Quellen kommen - den grid_operator_signals des
    Szenarios, dem Preisblatt (Netzentgelt, Umlagen, Konzession, Stromsteuer, MwSt) und der
    cfg -, die sich gegenseitig ueberschrieben haben. Genau daraus sind zwei stille Fehler
    entstanden: das LP kalkulierte einen anderen Tarif als abgerechnet wurde, und
    ``oemof_grid_feedin_tariff = 0`` blieb wirkungslos, solange ein Preisblatt konfiguriert
    war. Das Preisblatt ist jetzt ganz raus.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False, grid_variable_costs=35.0, grid_feedin_tariff=0.0),
        time_index=idx,
        # Verguetungs-Keys, wie das Preisblatt sie frueher geliefert hat: sie muessen
        # wirkungslos sein, sonst haette sich die alte Quelle nur versteckt
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1], "pv": [0, 3, 3, 0],
                                 "feedin_tariff_ct_kWh": -6.24,
                                 "homebus_feedin_tariff_ct_kWh": -1.5,
                                 "pv_power_kW": 7.5}},
    )
    _build_es(m)
    nodes = _nodes(m)
    supply = _out_flow(nodes["grid_supply_Home_1"])
    assert [float(supply.variable_costs[t]) for t in range(4)] == [35.0] * 4
    # beide Exportwege bekommen denselben cfg-Wert
    assert float(list(nodes["grid_feedin_Home_1"].inputs.values())[0].variable_costs[0]) == 0.0
    assert float(list(nodes["excess_Home_1"].inputs.values())[0].variable_costs[0]) == 0.0
    # die Anlagengroesse ist KEIN Preis und kommt weiterhin aus dem Szenario
    assert list(nodes["converter_pv_to_home_Home_1"].inputs.values())[0].nominal_value == 7.5
    # ... und die Preis-Felder existieren nicht mehr
    felder = {f.name for f in dataclasses.fields(SystemConfig)}
    assert not felder & {"tariff", "fee_type", "use_retail_markup", "cost_parameters_file",
                         "grid_price_from_scenario", "feedin_tariff_from_price_sheet"}


def _roh(consumer_type="household"):
    """Config whose markup is switched off - isolates the unit and the step function."""
    return SystemConfig(debug=False, consumer_type=consumer_type,
                        grid_price_markup_ct_kWh=0.0, grid_price_vat=0.0)


def test_strategy_hands_the_model_physics_and_the_scenario_price():
    """Last, PV, Anschlussleistung, kWp - und der Bezugspreis, wenn das Szenario einen hat.

    Der Aufschlag ist hier auf 0 gestellt, damit die EINHEIT allein gepruefte Sache bleibt:
    spice_ev fuehrt gc.cost in ct/kWh, also kommt der CSV-Wert unveraendert an. Was der
    Verbrauchertyp daraufschlaegt, prueft test_consumer_type_marks_up_the_exchange_price.
    Die Einspeiseverguetung bleibt in jedem Fall fest.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(
        fixed_load_lists={"L1": _ev([1, 1, 1, 1], idx[0], "GC1")},
        local_generation_lists={"PV1": _ev([0, 3, 3, 0], idx[0], "GC1")},
        grid_operator_signals=[SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                                               cost={"type": "fixed", "value": 30.0})],
    )
    strat.world_state = SimpleNamespace(
        grid_connectors={"GC1": SimpleNamespace(max_power=30.0)},
        photovoltaics={"PV1": SimpleNamespace(parent="GC1", nominal_power=10.0)},
    )
    info = strat._grid_connectors(idx, _roh())["GC1"]
    assert set(info) == {"load", "pv", "max_power", "pv_power_kW", "price_ct_kWh"}
    assert info["pv_power_kW"] == 10.0
    assert list(info["price_ct_kWh"]) == [30.0] * 4        # ct/kWh, unveraendert
    # ohne Preissignale bleibt der Schluessel weg -> das Modell nimmt die cfg
    strat.events.grid_operator_signals = []
    assert "price_ct_kWh" not in strat._grid_connectors(idx, _roh())["GC1"]
    # die Preisblatt-Methoden gibt es nicht mehr - kein toter Pfad, ueber den etwas zurueckkommt
    for name in ("_retail_markup_ct", "_feedin_tariff_ct", "_homebus_feedin_tariff_ct",
                 "tariff"):
        assert not hasattr(OemofSolve, name), name


def test_scenario_price_signals_become_a_step_function():
    """Die Events werden zu genau der Stufenfunktion, die spice_ev auch vor sich hat.

    Ein Signal gilt ab seiner ``start_time`` bis zum naechsten - so wertet spice_ev
    ``gc.cost`` in jedem Schritt aus, und so muss es beim LP ankommen. Geprueft werden die
    drei Faelle, die in der Praxis schiefgehen: die Einheit (ct/kWh - spice_ev teilt in
    scenario.py und costs.py durch 100, und generate.py nennt die Spalte per Default
    "price [ct/kWh]"), fremde Netzanschluesse, und negative Preise.
    """
    idx = pd.date_range("2025-01-01", periods=6, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(grid_operator_signals=[
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                        cost={"type": "fixed", "value": 30.0}),   # ct/kWh, nicht EUR!
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[2],
                        cost={"type": "fixed", "value": 5.0}),
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[4],
                        cost={"type": "fixed", "value": -4.0}),   # negativ -> gekappt
        SimpleNamespace(grid_connector_id="GC2", start_time=idx[0],
                        cost={"type": "fixed", "value": 99.0}),   # anderer GC -> ignoriert
    ])
    reihe = strat._grid_price_series("GC1", idx, _roh())
    assert list(reihe) == [30.0, 30.0, 5.0, 5.0, 0.0, 0.0]
    assert strat._grid_price_series("GC3", idx, _roh()) is None   # keine Signale -> cfg-Wert

    # ... und die Reihe landet als variable_costs im Modell, Schritt fuer Schritt
    m = EnergySystemModel(
        config=SystemConfig(debug=False, grid_variable_costs=35.0),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 6,
                                 "price_ct_kWh": reihe}},
    )
    _build_es(m)
    kosten = _out_flow(_nodes(m)["grid_supply_Home_1"]).variable_costs
    assert [float(kosten[t]) for t in range(6)] == [30.0, 30.0, 5.0, 5.0, 0.0, 0.0]


def test_consumer_type_marks_up_the_exchange_price():
    """Aus dem Boersenpreis wird ein Endkundenpreis: (boerse + aufschlag) * (1 + mwst).

    Die CSV eines Szenarios traegt den BOERSENpreis. Was ein Kunde zahlt, ist der plus
    Netzentgelt, Umlagen, Konzessionsabgabe und Stromsteuer - beim Haushalt zusaetzlich
    Mehrwertsteuer auf die Summe. Genau diese Zerlegung steckt in CONSUMER_TYPES; hier wird
    geprueft, dass sie ankommt und dass die Overrides sie schlagen.
    """
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(grid_operator_signals=[
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                        cost={"type": "fixed", "value": 10.0}),   # ct/kWh Boerse
    ])

    haushalt = strat._grid_price_series("GC1", idx, SystemConfig(consumer_type="household"))
    gewerbe = strat._grid_price_series("GC1", idx, SystemConfig(consumer_type="commercial"))
    # 12.09 netto + 19 % MwSt  /  8.10 netto, Vorsteuerabzug
    assert haushalt[0] == pytest.approx((10.0 + 12.09) * 1.19)
    assert gewerbe[0] == pytest.approx(10.0 + 8.10)
    assert haushalt[0] > gewerbe[0] > 10.0                  # beide teurer als die Boerse

    # Overrides schlagen den Verbrauchertyp
    eigen = strat._grid_price_series("GC1", idx, SystemConfig(
        consumer_type="household", grid_price_markup_ct_kWh=5.0, grid_price_vat=0.0))
    assert eigen[0] == pytest.approx(15.0)

    # ein Aufschlag hebt negative Boersenpreise ueber null - die Kappung greift dann nicht
    strat.events.grid_operator_signals[0].cost["value"] = -4.0
    assert strat._grid_price_series(
        "GC1", idx, SystemConfig(consumer_type="household"))[0] == pytest.approx(
            (-4.0 + 12.09) * 1.19)
    # ohne Aufschlag bleibt sie als Schutz aktiv
    assert strat._grid_price_series("GC1", idx, _roh())[0] == 0.0


def test_unknown_consumer_type_warns_and_falls_back(caplog):
    """Ein Tippfehler im cfg soll den Lauf nicht abbrechen, aber auffallen."""
    with caplog.at_level(logging.WARNING):
        markup, vat = SystemConfig(consumer_type="Haushalt").consumer_tariff()
    assert (markup, vat) == SystemConfig(consumer_type="household").consumer_tariff()
    assert "Haushalt" in caplog.text and "household" in caplog.text
    # der Schluessel kommt aus der cfg an
    assert SystemConfig.from_options(
        {"oemof_consumer_type": "commercial"}).consumer_tariff() == (8.10, 0.00)


def test_flat_fallback_price_is_never_marked_up():
    """Ohne Szenario-Kurve gilt grid_variable_costs - und das ist schon ein Endkundenpreis.

    Ein Aufschlag darauf wuerde Netzentgelt, Umlagen und Steuern doppelt zaehlen. Der
    Aufschlag darf also ausschliesslich auf die Szenario-Kurve wirken.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False, consumer_type="household",
                            grid_variable_costs=35.0),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 4}},   # kein price_ct_kWh
    )
    _build_es(m)
    # oemof normalisiert variable_costs zur Reihe je Schritt - alle Schritte 35, nicht 41.6
    kosten = _out_flow(_nodes(m)["grid_supply_Home_1"]).variable_costs
    assert [float(kosten[t]) for t in range(4)] == [35.0] * 4


@requires_cbc
def test_scalar_results_carry_the_grid_peak(tmp_path, monkeypatch):
    """dump_costs.csv traegt die Lastspitze - damit ein Tarif SPAETER gerechnet werden kann.

    Das Modell selbst bewertet keine Leistung: kein Leistungspreis in der Zielfunktion,
    bewusst, sonst dominierte ein Jahresbetrag einen Wochenfahrplan. Es gibt statt dessen
    die Spitze, die Energie und die Horizontlaenge heraus.
    """
    monkeypatch.chdir(tmp_path)
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False, should_dump_results=True, output_dir="out",
                            consumer_type="commercial"),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 9, 1, 1, 1, 1, 1]}},
    )
    m.run()

    k = m._costs
    reihe = m._summary_df["grid_supply_Home_1"]
    assert k["grid_peak_kW_Home_1"] == pytest.approx(float(reihe.max()))
    assert k["grid_energy_kWh_Home_1"] == pytest.approx(float(reihe.sum()) * 0.25)
    assert k["consumer_type"] == "commercial"
    assert k["periods"] == 8 and k["step_hours"] == pytest.approx(0.25)
    assert k["fraction_year"] == pytest.approx(8 * 0.25 / 8760.0)
    # die Spitze ist die Lastspitze des Zeitraums, nicht der Mittelwert
    assert k["grid_peak_kW_Home_1"] > k["grid_energy_kWh_Home_1"] / (8 * 0.25)
    # kein Euro-Betrag im Modell - der Tarif ist eine nachgelagerte Entscheidung
    assert not any("eur" in name.lower() or "capacity" in name.lower() for name in k)

    dump = pd.read_csv(tmp_path / "out" / "dump_costs.csv")
    assert dump.loc[0, "grid_peak_kW_Home_1"] == pytest.approx(k["grid_peak_kW_Home_1"])


def test_from_options_coerces_cfg_types():
    """A cfg value must land on the field's declared type, not as a raw string.

    The cfg is read with json.loads, which only knows lowercase true/false. Written with a
    capital F, "False" stays a STRING — and a non-empty string is truthy, so the switch
    would silently be ON while the cfg says False. This actually happened once.
    """
    c = SystemConfig.from_options({"oemof_enable_v2h": "False",
                                   "oemof_enable_grid_feedin": "FALSE",
                                   "oemof_forbid_simultaneous_storage": "yes",
                                   "oemof_grid_variable_costs": "22.5",
                                   "oemof_solver_threads": "4",
                                   "oemof_solver": "cbc"})
    assert c.enable_v2h is False and c.enable_grid_feedin is False
    assert c.forbid_simultaneous_storage is True
    assert c.grid_variable_costs == 22.5 and isinstance(c.grid_variable_costs, float)
    assert c.solver_threads == 4 and isinstance(c.solver_threads, int)
    assert c.solver == "cbc"          # Strings bleiben unangetastet
    # echte JSON-Werte gehen unveraendert durch
    assert SystemConfig.from_options({"oemof_enable_v2h": False}).enable_v2h is False


# ---------------------------------------------------------------------------
# Test 2c — step() skeleton: reads the right plan values, all guards intact
# ---------------------------------------------------------------------------
class _FakeBattery:
    """Minimal Battery stand-in for the SOC-driven step(): moves to the requested SOC.

    Mirrors spice_ev's relation ``soc_delta * capacity = avg_power * dt`` (efficiency 1.0
    here, so the arithmetic in the assertions stays readable — the real efficiency handling
    lives in spice_ev.Battery and is covered by the end-to-end runs). Doubles as a
    stationary battery: those live directly in world_state.batteries and carry their parent
    grid connector as an attribute.
    """

    CAPACITY = 40.0

    def __init__(self, parent=None, soc=0.5):
        self.parent = parent
        self.soc = soc
        self.calls = []

    def _power(self, interval, delta_soc):
        hours = interval.total_seconds() / 3600
        return abs(delta_soc) * self.CAPACITY / hours

    def load(self, interval, max_power=None, target_soc=None, target_power=None):
        self.calls.append(("load", max_power, target_soc))
        power = self._power(interval, target_soc - self.soc)
        self.soc = target_soc
        return {"avg_power": power}

    def unload(self, interval, max_power=None, target_soc=None, target_power=None):
        self.calls.append(("unload", max_power, target_soc))
        power = self._power(interval, self.soc - target_soc)
        self.soc = target_soc
        return {"avg_power": power}


class _FakeGC:
    """GridConnector stand-in: books loads like the real one (ledger + running total)."""

    def __init__(self):
        self.current_loads = {}

    def add_load(self, key, value):
        self.current_loads[key] = self.current_loads.get(key, 0.0) + value
        return self.current_loads[key]


def _fake_step_strategy():
    """OemofSolve with a faked world and plan — enough to exercise step() alone.

    Plan tuples are ``(charge_kW, discharge_kW, soc_end)``; step() applies ONLY soc_end,
    the powers are carried along for reporting.
    """
    from datetime import timedelta
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = None
    strat._solved = True
    strat._oemof_step = 0
    strat.current_time = "t0"
    strat.interval = timedelta(minutes=15)
    strat.EPS = 1e-5          # instance attribute of Strategy.__init__, bypassed here
    vtype = SimpleNamespace(min_charging_power=0.0, v2g=False)
    vtype_v2g = SimpleNamespace(min_charging_power=0.0, v2g=True, discharge_limit=0.5)
    strat.world_state = SimpleNamespace(
        vehicles={
            "v1": SimpleNamespace(connected_charging_station="CS1",
                                  battery=_FakeBattery(soc=0.50), vehicle_type=vtype),
            "v2": SimpleNamespace(connected_charging_station=None,     # away/driving
                                  battery=_FakeBattery(), vehicle_type=vtype),
            "v3": SimpleNamespace(connected_charging_station="CS3",    # no plan entry
                                  battery=_FakeBattery(), vehicle_type=vtype),
            "v4": SimpleNamespace(connected_charging_station="CS4",    # v2g: discharges
                                  battery=_FakeBattery(soc=0.60), vehicle_type=vtype_v2g),
        },
        charging_stations={
            "CS1": SimpleNamespace(parent="GC1", current_power=99.0,   # 99 -> must be reset
                                   max_power=22.0, min_power=0.0),
            "CS3": SimpleNamespace(parent="GC1", current_power=99.0,
                                   max_power=22.0, min_power=0.0),
            "CS4": SimpleNamespace(parent="GC1", current_power=99.0,
                                   max_power=22.0, min_power=0.0),
        },
        batteries={
            "BAT1": _FakeBattery(parent="GC1", soc=0.40),
            "BAT2": _FakeBattery(parent="GC_gone"),                    # its GC was pruned
        },
        grid_connectors={"GC1": _FakeGC()},
    )
    #                     charge_kW, discharge_kW, soc_end
    strat._plan = {
        "vehicles": {"v1": [(8.0, 0.0, 0.55), (0.0, 0.0, 0.55)],   # charge, then hold
                     "v2": [(1.0, 0.0, 0.60), (1.0, 0.0, 0.70)],   # away -> ignored
                     "v4": [(0.0, 8.0, 0.55), (0.0, 0.0, 0.55)]},  # V2G, then hold
        "batteries": {"BAT1": [(0.0, 8.0, 0.35), (8.0, 0.0, 0.40)],
                      "BAT2": [(3.0, 0.0, 0.60), (3.0, 0.0, 0.70)]},
        "grid": {"GC1": [(10.0, 0.0), (0.0, 0.0)]},
    }
    return strat


def test_step_skeleton_reads_plan_with_guards():
    strat = _fake_step_strategy()
    res = strat.step()

    assert res["current_time"] == "t0"                        # contract with scenario.py
    assert strat._oemof_step == 1                             # counter advanced

    # v1 + v4 applied; v2 (away) and v3 (no plan entry) untouched
    assert strat.world_state.vehicles["v1"].battery.calls
    assert strat.world_state.vehicles["v4"].battery.calls
    assert strat.world_state.vehicles["v2"].battery.calls == []
    assert strat.world_state.vehicles["v3"].battery.calls == []
    # BAT1 applied; BAT2 skipped because its grid connector does not exist
    assert strat.world_state.batteries["BAT1"].calls
    assert strat.world_state.batteries["BAT2"].calls == []

    # second step: the plan holds every SOC -> nothing more to do, no extra calls
    before = {k: len(v.battery.calls) for k, v in strat.world_state.vehicles.items()}
    strat.step()
    assert {k: len(v.battery.calls) for k, v in strat.world_state.vehicles.items()} == before

    # third call is past the horizon: nothing read, return still well-formed
    res = strat.step()
    assert res["commands"] == {}
    assert strat._oemof_step == 3


def test_step_applies_vehicle_charging():
    """The vehicle is steered to the SOC the plan reaches at the END of the step."""
    strat = _fake_step_strategy()
    res = strat.step()

    v1 = strat.world_state.vehicles["v1"]
    cs1 = strat.world_state.charging_stations["CS1"]
    gc = strat.world_state.grid_connectors["GC1"]

    # target_soc from the plan, max_power = the station rating (never target_power)
    assert v1.battery.calls == [("load", 22.0, 0.55)]
    assert v1.battery.soc == pytest.approx(0.55)
    # 0.05 * 40 kWh over 15 min = 8 kW, booked at the GC and reported in the commands
    assert gc.current_loads["CS1"] == pytest.approx(8.0)
    assert res["commands"]["CS1"] == pytest.approx(8.0)
    # charging station keeps track of its own load (after the per-step reset from 99)
    assert cs1.current_power == pytest.approx(8.0)

    # v2 is away, v3 has no plan: their batteries were never touched
    assert strat.world_state.vehicles["v2"].battery.calls == []
    assert strat.world_state.vehicles["v3"].battery.calls == []


def test_step_applies_v2g_discharge():
    """V2G feed-back discharges down to the planned SOC — the target IS the floor."""
    strat = _fake_step_strategy()
    res = strat.step()

    v4 = strat.world_state.vehicles["v4"]
    cs4 = strat.world_state.charging_stations["CS4"]
    gc = strat.world_state.grid_connectors["GC1"]

    assert v4.battery.calls == [("unload", 22.0, 0.55)]
    assert v4.battery.soc == pytest.approx(0.55)
    # booked as NEGATIVE load (feed-back) at the grid connector and in the commands
    assert gc.current_loads["CS4"] == pytest.approx(-8.0)
    assert res["commands"]["CS4"] == pytest.approx(-8.0)
    assert cs4.current_power == pytest.approx(-8.0)

    # second step plans the same SOC -> no further battery interaction
    strat.step()
    assert len(v4.battery.calls) == 1


def test_step_does_not_discharge_non_v2g_vehicles():
    """A falling SOC target must never discharge a vehicle that is not V2G capable."""
    strat = _fake_step_strategy()
    strat._plan["vehicles"]["v1"] = [(0.0, 8.0, 0.45), (0.0, 0.0, 0.45)]
    strat.step()
    assert strat.world_state.vehicles["v1"].battery.calls == []
    assert "CS1" not in strat.world_state.grid_connectors["GC1"].current_loads


def test_step_applies_stationary_battery_plan():
    """The stationary battery follows the planned SOC and books at its GC."""
    strat = _fake_step_strategy()
    gc = strat.world_state.grid_connectors["GC1"]
    bat1 = strat.world_state.batteries["BAT1"]
    bat2 = strat.world_state.batteries["BAT2"]

    strat.step()   # plan step 0: SOC 0.40 -> 0.35, i.e. feed the house
    assert bat1.calls == [("unload", None, 0.35)]
    assert gc.current_loads["BAT1"] == pytest.approx(-8.0)   # negative = feeds the house

    strat.step()   # plan step 1: SOC 0.35 -> 0.40, i.e. charge again
    assert bat1.calls[-1] == ("load", None, 0.40)
    assert gc.current_loads["BAT1"] == pytest.approx(0.0)

    # BAT2's grid connector was pruned in the LP: never touched
    assert bat2.calls == []


def test_step_ignores_missing_soc_target():
    """A NaN target (no storage in the results) must be skipped, not crash."""
    strat = _fake_step_strategy()
    strat._plan["vehicles"]["v1"] = [(8.0, 0.0, float("nan"))]
    strat.step()
    assert strat.world_state.vehicles["v1"].battery.calls == []


# ---------------------------------------------------------------------------
# Test 3b — per-step SOC floor from the scenario (desired_soc before departure)
# ---------------------------------------------------------------------------
@requires_cbc
def test_min_soc_series_forces_desired_soc_before_departure():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    # plugged in for steps 0-3, drives 4-7; spice_ev wants desired_soc=0.8 when it leaves
    floor = np.array([0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2])
    m = EnergySystemModel(
        config=SystemConfig(debug=False, should_dump_results=False, enable_grid_feedin=False),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 8}},
        charging_stations={"CS1": {"max_power": 22.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0, "initial_soc": 0.5, "min_soc": 0.2,
                               "min_soc_series": floor,
                               "connected_cs": ["CS1"] * 4 + [None] * 4,
                               "consumption": [0, 0, 0, 0, 4, 4, 4, 4]}},
    )
    _build_es(m)
    m._optimize()
    m._solve()
    m._extract_results()
    df = m.get_wallbox_schedule()["v1"]

    # the plan must charge the car to desired_soc by the departure step
    assert df["soc_kWh"].iloc[3] / 50.0 >= 0.8 - 1e-6
    # the AC command must never exceed the charging station rating (limit sits on the AC side)
    assert df["charge_kW"].max() <= 22.0 + 1e-6


# ---------------------------------------------------------------------------
# Test 4 — full run() extracts a per-vehicle schedule and dumps CSVs (CBC)
# ---------------------------------------------------------------------------
@requires_cbc
def test_full_run_extracts_schedule(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False, should_dump_results=True, output_dir="out"),
        time_index=idx,
        # car starts at 50% (25 kWh), drives steps 4-7 (16 kWh) -> must charge to stay >= min
        grid_connectors={"GC1": {"max_power": 30.0,
                                 "load": [1] * 8, "pv": [0, 0, 2, 4, 4, 2, 0, 0]}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 10.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0, "initial_soc": 0.5, "min_soc": 0.2,
                               "connected_cs": ["CS1", "CS1", "CS1", "CS1", None, None, None, None],
                               "consumption": [0, 0, 0, 0, 4, 4, 4, 4]}},
    )
    m.run()   # full pipeline incl. _extract_results + _save_results

    sched = m.get_wallbox_schedule()
    assert set(sched) == {"v1"}
    df = sched["v1"]
    assert list(df.columns) == ["charge_kW", "discharge_kW", "net_kW", "soc_kWh",
                                "soc_end", "consumption_kWh"]
    assert len(df) == 8
    assert df["charge_kW"].sum() > 0                              # must charge to survive the drive
    assert (df["net_kW"] == df["charge_kW"] - df["discharge_kW"]).all()
    assert df["soc_kWh"].notna().all()                            # BEV SOC extracted
    assert df["consumption_kWh"].sum() > 0                        # driving demand present

    # enriched per-GC summary now carries the household demand + wallbox columns
    assert "household_demand_Home_1" in m._summary_df.columns
    assert "wallbox_charge_v1" in m._summary_df.columns
    # the objective's actual price series is dumped per GC (constant fallback here)
    assert "grid_price_ct_Home_1" in m._summary_df.columns
    # PV self-consumption column: generation - feed-in must equal the converter flow
    assert "pv_selfuse_Home_1" in m._summary_df.columns
    s = m._summary_df
    assert np.allclose(s["pv_Home_1"] - s["pv_feedin_Home_1"], s["pv_selfuse_Home_1"], atol=1e-6)

    # get_plan(): the full per-component plan for the strategy's step()
    plan = m.get_plan()
    assert set(plan) == {"vehicles", "batteries", "grid"}
    assert set(plan["vehicles"]) == {"v1"} and set(plan["batteries"]) == {"BAT1"}
    assert list(plan["batteries"]["BAT1"].columns) == ["charge_kW", "discharge_kW", "soc_end"]
    assert list(plan["grid"]["GC1"].columns) == ["supply_kW", "feedin_kW"]
    assert all(len(frame) == 8 for group in plan.values() for frame in group.values())
    # the grid plan must agree with the summary (same flows, different keying)
    assert (plan["grid"]["GC1"]["supply_kW"].to_numpy()
            == m._summary_df["grid_supply_Home_1"].to_numpy()).all()

    # strategy side: commands_from_oemof turns the frames into per-step tuple lists
    strat = OemofSolve.__new__(OemofSolve)
    step_lists = strat.commands_from_oemof(plan)
    assert set(step_lists) == {"vehicles", "batteries", "grid"}
    assert len(step_lists["vehicles"]["v1"]) == 8
    assert step_lists["vehicles"]["v1"][0] == (float(df["charge_kW"].iloc[0]),
                                               float(df["discharge_kW"].iloc[0]),
                                               float(df["soc_end"].iloc[0]))
    # soc_end is the SOC at the END of the step = the start SOC of the next one
    assert df["soc_end"].iloc[0] * 50.0 == pytest.approx(df["soc_kWh"].iloc[1])

    # CSVs land in config.output_dir
    assert (tmp_path / "out" / "dump_wallbox_v1.csv").exists()
    assert (tmp_path / "out" / "dump_summary.csv").exists()
    assert (tmp_path / "out" / "dump_costs.csv").exists()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test 5 — PV, Speicher und Wallbox ohne Direktzweige
# ---------------------------------------------------------------------------
def _pv_scenario(config, pv_power_kW=10.0, pv=(0, 0, 20, 20, 20, 20, 0, 0), v2g=False):
    """PV-Ueberschuss bei angestecktem Auto - die Lage, um die es geht.

    Die Haushaltslast muss gross genug sein, dass das Netz wirklich gebraucht wird: mit
    kleiner Last traegt die 20-kWh-Batterie alles umsonst, und das LP exportiert jede kWh.
    """
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    return EnergySystemModel(
        config=config,
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 40.0, "load": [8] * 8, "pv": list(pv),
                                 "pv_power_kW": pv_power_kW}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 20.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 80.0, "initial_soc": 0.2, "min_soc": 0.1,
                               "v2g": v2g, "discharge_limit": 0.1,
                               "connected_cs": ["CS1"] * 8, "consumption": [0] * 8}},
    )


def _flow_between(m, src_label, dst_label, n=8):
    """Solved flow between two nodes, looked up by label."""
    nodes = _nodes(m)
    seq = m._results_main[(nodes[src_label], nodes[dst_label])]["sequences"]["flow"]
    return np.asarray(seq.to_numpy()[:n], dtype=float)


@pytest.mark.parametrize("v2h", [False, True])
def test_pv_reaches_everything_through_the_house_bus(v2h):
    """Ein Weg fuer die PV: Wechselrichter -> Hausbus. Von dort wie jede andere kWh.

    Frueher gab es benannte Direktzweige zu Wallbox und Batterie, dazu getrennte Zu- und
    Abflussbusse an den Speichern. Sie existierten allein, damit ein PV-Ladebonus daran
    haengen konnte. Ohne diesen Bonus waren sie eine Kennzahl ohne Aussage - der Solver
    waehlte willkuerlich zwischen zwei gleich teuren Wegen zum selben Ziel.
    """
    m = _pv_scenario(SystemConfig(debug=False, enable_v2h=v2h), v2g=v2h)
    labels = _build_es(m)

    # kein Direktzweig, keine Klemme, keine Bustrennung
    assert not any(lbl.startswith(("bus_pvac", "conv_pvac_to_home", "conv_pv_to_",
                                   "bus_wbin", "conv_home_to_wallbox", "bus_batin",
                                   "bus_batout", "bus_mobin")) for lbl in labels)
    # je Speicher genau ein Bus
    assert "bus_battery_BAT1" in labels and "bus_mobility_v1" in labels

    n = _nodes(m)
    # der Wechselrichter speist direkt ins Haus und traegt die Anlagengrenze
    pv_conv = n["converter_pv_to_home_Home_1"]
    assert list(pv_conv.inputs.values())[0].nominal_value == 10.0
    assert list(pv_conv.outputs)[0].label == "Home_1"
    # die Wallbox haengt am Hausbus, der Speicher an seinem einen Bus
    assert list(n["wallbox_charge_CS1_v1"].inputs)[0].label == "Home_1"
    assert list(n["wallbox_charge_CS1_v1"].outputs)[0].label == "bus_mobility_v1"
    for lbl in ("home_battery_BAT1", "bev_battery_v1"):
        speicher = n[lbl]
        assert list(speicher.inputs)[0].label == list(speicher.outputs)[0].label
    # V2H nur mit v2g UND Schalter
    assert ("wallbox_discharge_CS1_v1" in labels) is v2h


def test_no_artificial_incentives_are_left():
    """Die Zielfunktion enthaelt nur echte Preise - kein Bonus mehr, kein Restschluessel."""
    felder = {f.name for f in dataclasses.fields(SystemConfig)}
    assert not felder & {"pv_direct_to_storage", "pv_charge_bonus_vehicle_ct_kWh",
                         "pv_charge_bonus_battery_ct_kWh"}


@requires_cbc
def test_inverter_and_station_ratings_hold():
    """Zwei Grenzen, die der Fahrplan einhalten MUSS, damit step() ihn ausfuehren kann.

    Die Wallbox: step() steuert mit ``Battery.load(max_power=cs.max_power)``. Ein Plan, der
    mehr verlangt, wuerde stillschweigend gekappt und der simulierte SOC bliebe zurueck.
    Der Wechselrichter: er begrenzt, wie viel PV ueberhaupt ins Haus kommt.
    """
    m = _pv_scenario(SystemConfig(debug=False, should_dump_results=False),
                     pv_power_kW=6.0, pv=(30,) * 8)
    m.run()
    laden = m.get_wallbox_schedule()["v1"]["charge_kW"].to_numpy()
    assert np.all(laden <= 11.0 + 1e-6)
    eigen = m._summary_df["pv_selfuse_Home_1"].to_numpy()
    assert np.all(eigen <= 6.0 + 1e-6)
    assert eigen.max() == pytest.approx(6.0, abs=1e-6)      # und sie greift wirklich


@requires_cbc
def test_pv_splits_into_selfuse_and_export():
    """Die Erzeugung geht vollstaendig in Eigenverbrauch oder Einspeisung - nichts geht weg."""
    m = _pv_scenario(SystemConfig(debug=False, should_dump_results=False))
    m.run()
    s = m._summary_df
    assert np.allclose(s["pv_Home_1"], s["pv_selfuse_Home_1"] + s["pv_feedin_Home_1"],
                       atol=1e-6)
    assert s["pv_selfuse_Home_1"].sum() > 0        # sonst ginge die Zeile leer durch
    # eine Zuordnung "so viel PV ging ins Auto" gibt es bewusst nicht mehr
    assert not any(c.startswith("pv_direct") for c in s.columns)


@requires_cbc
def test_battery_is_reachable_only_through_the_link():
    """Die Batterie hat genau einen Zugang - den Link vom Hausbus.

    Deshalb braucht sie keine getrennten Busse: was hineingeht, kann nur ueber denselben
    Link wieder heraus, und der geplante charge_kW ist genau dieser eine Fluss.
    """
    m = _pv_scenario(SystemConfig(debug=False, should_dump_results=False))
    m.run()
    speicher = _nodes(m)["home_battery_BAT1"]
    assert {b.label for b in speicher.inputs} == {"bus_battery_BAT1"}
    link_in = _flow_between(m, "Home_1", "link_home_battery_BAT1")
    storage_in = _flow_between(m, "bus_battery_BAT1", "home_battery_BAT1")
    assert np.allclose(link_in, storage_in, atol=1e-6)
    assert np.all(storage_in <= 5.0 + 1e-6)                   # the battery's power limit
    assert np.allclose(m.get_plan()["batteries"]["BAT1"]["charge_kW"], link_in, atol=1e-6)


def test_v2g_discharge_is_capped_by_the_vehicles_own_curve():
    """V2G entlaedt NICHT mit der Ladeleistung der Station.

    spice_ev bildet die Entladekurve aus der Ladekurve mal ``v2g_power_factor`` (Default
    0.5) und ``Battery.unload`` begrenzt darauf. Plant das Modell mit der vollen
    Stationsleistung, liefert die Simulation weniger und der geplante SOC laeuft weg - in
    03_household_v2g waren das 5.27 kW je Schritt, bis die Grenze durchgereicht wurde.
    """
    m = _pv_scenario(SystemConfig(debug=False, enable_v2h=True), v2g=True)
    m.vehicle_params["v1"]["discharge_power_kW"] = 4.0      # Station kann 11
    _build_es(m)
    n = _nodes(m)
    ab = list(n["wallbox_discharge_CS1_v1"].outputs.values())[0]
    assert ab.nominal_value == 4.0
    # ohne den Wert bleibt es bei der Stationsleistung
    m2 = _pv_scenario(SystemConfig(debug=False, enable_v2h=True), v2g=True)
    _build_es(m2)
    ab2 = list(_nodes(m2)["wallbox_discharge_CS1_v1"].outputs.values())[0]
    assert ab2.nominal_value == 11.0


@requires_cbc
def test_forbid_simultaneous_storage_creates_binaries():
    """Der Schalter macht aus dem LP ein MILP - eine Binaervariable je Speicher und Schritt.

    Gebraucht wurde er gegen das Kreisen, das ein PV-Ladebonus ausloesen konnte. Den Bonus
    gibt es nicht mehr; der Schalter bleibt fuer eigene Experimente und steht auf aus.
    """
    assert SystemConfig().forbid_simultaneous_storage is False

    aus = _pv_scenario(SystemConfig(debug=False, enable_v2h=True), v2g=True)
    _build_es(aus)
    aus._optimize()
    assert not hasattr(aus.model, "speicher_modus")
    # die Registry ist trotzdem gefuellt - Einschalten braucht keinen Neuaufbau
    assert {p["label"] for p in aus._storage_pairs} == {"home_battery_BAT1", "bev_battery_v1"}

    an = _pv_scenario(SystemConfig(debug=False, enable_v2h=True,
                                   forbid_simultaneous_storage=True), v2g=True)
    _build_es(an)
    an._optimize()
    assert hasattr(an.model, "speicher_modus")

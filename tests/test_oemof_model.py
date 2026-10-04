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
import warnings
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
            # jeder aktive Anschluss braucht eine Preisreihe - es gibt keinen Rueckfall
            "GC1": {"max_power": 30.0, "load": [1, 1, 1, 1], "pv": [0, 3, 3, 0],
                    "price_ct_kWh": [30.0] * 4},
            "GC2": {"max_power": 50.0, "price_ct_kWh": [30.0] * 4},   # nur CS2 -> aktiv
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
    # there is no export path from the house bus - only the PV surplus leaves, and it
    # leaves through excess_ on the PV bus
    assert not any(lbl.startswith("grid_feedin_") for lbl in labels)

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
        # je Anschluss ein Preissignal - ohne eines lehnt die Strategie ab
        grid_operator_signals=[
            SimpleNamespace(grid_connector_id=g, start_time=start,
                            cost={"type": "fixed", "value": 30.0}) for g in ("GC1", "GC2")],
    )
    strat.world_state = SimpleNamespace(
        grid_connectors={"GC1": SimpleNamespace(max_power=30.0),
                         "GC2": SimpleNamespace(max_power=50.0)},
    )

    gcs = strat._grid_connectors(idx)

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


def test_a_load_is_judged_by_absolute_values_and_never_negative():
    """Whether a load gets a node is decided on absolute values, not on the sum.

    With the sum, a load of 3 and -3 kW (or a negative one throughout) was dropped silently -
    here even the whole grid connector, since it carried nothing else. A negative entry the
    model cannot take at all (a fixed flow is >= 0); it used to end as "infeasible" without a
    reason, now it stops with one.
    """
    from spice_ev.oemof_model import _nonzero
    assert _nonzero([3, -3, 0, 0], 4) and _nonzero([-2, -2, -2, -2], 4)
    assert not _nonzero([0, 0, 0, 0], 4) and not _nonzero(None, 4)

    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    for load in ([3, -3, 0, 0], [-2, -2, -2, -2], [3, -1, 0, 0]):
        m = EnergySystemModel(config=SystemConfig(debug=False), time_index=idx,
                              grid_connectors={"GC1": {"load": load, "price_ct_kWh": [30] * 4}})
        with pytest.raises(ValueError, match="load of grid connector 'GC1' is negative"):
            _build_es(m)
    m = EnergySystemModel(config=SystemConfig(debug=False), time_index=idx,
                          grid_connectors={"GC1": {"load": [1] * 4, "pv": [0, float("nan"), 0, 0],
                                                   "price_ct_kWh": [30] * 4}})
    with pytest.raises(ValueError, match="pv of grid connector 'GC1' has missing values"):
        _build_es(m)


def test_load_factor_is_applied_as_spice_ev_applies_it():
    """The factor of an EnergyValuesList scales it exactly as in spice_ev - 0 included.

    The model used to read ``getattr(ev_list, "factor", 1) or 1``, which turned a factor
    of 0 into 1: a load switched off in the scenario still reached the LP in full. Built
    from spice_ev's own class, so the defaults it guarantees are the real ones.
    """
    from pathlib import Path
    from spice_ev.events import EnergyValuesList, FixedLoad

    idx = pd.date_range("2023-04-01", periods=4, freq="15min", tz="Europe/Berlin")
    strat = OemofSolve.__new__(OemofSolve)
    for factor in (1, 0.5, 0):
        ev_list = EnergyValuesList({"start_time": "2023-04-01T00:00:00+02:00",
                                    "step_duration_s": 900, "grid_connector_id": "GC1",
                                    "values": [2, 2, 2, 2], "factor": factor}, Path("."))
        in_spice_ev = [e.value for e in ev_list.get_events("L1", FixedLoad)][:4]
        assert strat._sample_event_list(ev_list, idx).tolist() == in_spice_ev
        assert in_spice_ev == [2.0 * factor] * 4

    # without a factor in the scenario spice_ev sets 1 - nothing to fall back on here
    plain = EnergyValuesList({"start_time": "2023-04-01T00:00:00+02:00",
                              "step_duration_s": 900, "grid_connector_id": "GC1",
                              "values": [2, 2, 2, 2]}, Path("."))
    assert strat._sample_event_list(plain, idx).tolist() == [2.0] * 4


# ---------------------------------------------------------------------------
# Test 3 — solve a tiny feasible model with CBC (debug mode on)
# ---------------------------------------------------------------------------
@requires_cbc
def test_solve_small_model(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)   # keep the debug LP dump inside the tmp dir
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=True, output_dir="lp_out"),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1],
                                 "price_ct_kWh": [30.0] * 4}},
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
# Test 2a2 — der Preis kommt aus dem Szenario und sonst nirgendwo her
# ---------------------------------------------------------------------------
def test_prices_come_from_the_scenario_and_nowhere_else():
    """Bezugspreis: allein die Preiszeitreihe des Szenarios. Einspeiseverguetung: cfg.

    Das LP bewertet Energie mit derselben Reihe, die spice_evs eigene Strategien in
    gc.cost lesen - eine zweite Preisquelle gibt es nicht.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False, grid_feedin_tariff=0.0),
        time_index=idx,
        # fremde Verguetungs-Keys am Netzanschluss muessen wirkungslos bleiben
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1], "pv": [0, 3, 3, 0],
                                 "price_ct_kWh": [30.0] * 4,
                                 "feedin_tariff_ct_kWh": -6.24,
                                 "homebus_feedin_tariff_ct_kWh": -1.5}},
    )
    _build_es(m)
    nodes = _nodes(m)
    supply = _out_flow(nodes["grid_supply_Home_1"])
    assert [float(supply.variable_costs[t]) for t in range(4)] == [30.0] * 4
    # der einzige Exportweg bekommt den cfg-Wert
    assert float(list(nodes["excess_Home_1"].inputs.values())[0].variable_costs[0]) == 0.0
    # die cfg kennt nur diese Preis- und Kostenfelder - keins davon bewertet den Bezug
    felder = {f.name for f in dataclasses.fields(SystemConfig)}
    assert {f for f in felder if any(s in f for s in ("price", "tariff", "cost", "fee"))} == {
        "grid_feedin_tariff", "pv_variable_costs", "converter_pv_to_home_variable_costs"}


def test_strategy_hands_the_model_physics_and_the_scenario_price():
    """Last, PV, Anschlussleistung - und der Bezugspreis des Szenarios. Kein kWp-Wert.

    Geprueft wird vor allem die EINHEIT: spice_ev fuehrt gc.cost in ct/kWh, also kommt der
    CSV-Wert unveraendert an. Die Einspeiseverguetung bleibt in jedem Fall fest.
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
    info = strat._grid_connectors(idx)["GC1"]
    # die Anlagengroesse geht nicht ans Modell - die PV-Reihe ist schon die Erzeugung
    assert set(info) == {"load", "pv", "max_power", "price_ct_kWh"}
    assert list(info["price_ct_kWh"]) == [30.0] * 4        # ct/kWh, unveraendert


def test_a_scenario_without_prices_is_refused():
    """Ohne Preisreihe bricht der Lauf ab, statt mit einer erfundenen Zahl zu rechnen.

    Es gilt allein die Preiszeitreihe. Ein Szenario ohne ``include_price_csv`` kann
    nicht optimiert werden, und das soll man merken - eine Vorgabe wuerde eine Frage
    beantworten, die niemand gestellt hat. Geprueft werden beide Ebenen: die Strategie
    (sie nennt den Netzanschluss und den Weg) und das Modell selbst.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(fixed_load_lists={}, local_generation_lists={},
                                   grid_operator_signals=[])
    strat.world_state = SimpleNamespace(
        grid_connectors={"GC1": SimpleNamespace(max_power=30.0)}, photovoltaics={})
    assert strat._grid_price_series("GC1", idx) is None
    with pytest.raises(ValueError, match="include_price_csv"):
        strat._grid_connectors(idx)

    # und das Modell laesst sich auch nicht direkt ohne Preisreihe bauen
    m = EnergySystemModel(
        config=SystemConfig(debug=False), time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 4}},   # kein price_ct_kWh
    )
    with pytest.raises(ValueError, match="no price series"):
        _build_es(m)


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
    reihe = strat._grid_price_series("GC1", idx)
    assert list(reihe) == [30.0, 30.0, 5.0, 5.0, 0.0, 0.0]
    assert strat._grid_price_series("GC3", idx) is None            # keine Signale

    # ... und die Reihe landet als variable_costs im Modell, Schritt fuer Schritt
    m = EnergySystemModel(
        config=SystemConfig(debug=False),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 6,
                                 "price_ct_kWh": reihe}},
    )
    _build_es(m)
    kosten = _out_flow(_nodes(m)["grid_supply_Home_1"]).variable_costs
    assert [float(kosten[t]) for t in range(6)] == [30.0, 30.0, 5.0, 5.0, 0.0, 0.0]


def test_negative_prices_are_clipped_to_zero():
    """Die EINZIGE Abweichung vom Preis, den spice_evs Strategien sehen.

    spice_ev gibt einen negativen Preis unveraendert an greedy, balanced und
    balanced_market weiter. Das LP darf ihn nicht sehen: ein lineares Programm kann nicht
    daran gehindert werden, Energie loszuwerden - laden und entladen im selben Schritt
    verbrennt sie ueber den Wirkungsgrad. Wird man fuer den Bezug BEZAHLT, ist das eine
    Geldpumpe. Ein reales Haus kann das nicht, und spice_ev simuliert es auch nicht.

    Die Kappung ist damit eine erklaerte Modellentscheidung und keine Preisbildung. Auf
    einer Reihe mit positivem Minimum greift sie nie.
    """
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(grid_operator_signals=[
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                        cost={"type": "fixed", "value": -4.0}),
    ])
    assert strat._grid_price_series("GC1", idx)[0] == 0.0
    # ein positiver Wert wird nicht angetastet - die Kappung ist keine Preisbildung
    strat.events.grid_operator_signals[0].cost["value"] = 10.0
    assert strat._grid_price_series("GC1", idx)[0] == pytest.approx(10.0)


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
        config=SystemConfig(debug=False, should_dump_results=True, output_dir="out"),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 9, 1, 1, 1, 1, 1],
                                 "price_ct_kWh": [30.0] * 8}},
    )
    m.run()

    k = m._costs
    reihe = m._summary_df["grid_supply_Home_1"]
    assert k["grid_peak_kW_Home_1"] == pytest.approx(float(reihe.max()))
    assert k["grid_energy_kWh_Home_1"] == pytest.approx(float(reihe.sum()) * 0.25)
    assert k["periods"] == 8 and k["step_hours"] == pytest.approx(0.25)
    assert k["fraction_year"] == pytest.approx(8 * 0.25 / 8760.0)
    # die Spitze ist die Lastspitze des Zeitraums, nicht der Mittelwert
    assert k["grid_peak_kW_Home_1"] > k["grid_energy_kWh_Home_1"] / (8 * 0.25)
    # genau diese Werte - kein Euro-Betrag im Modell, der Tarif ist nachgelagert
    assert set(k) == {"objective", "periods", "step_hours", "fraction_year",
                      "grid_peak_kW_Home_1", "grid_energy_kWh_Home_1"}

    dump = pd.read_csv(tmp_path / "out" / "dump_costs.csv")
    assert dump.loc[0, "grid_peak_kW_Home_1"] == pytest.approx(k["grid_peak_kW_Home_1"])


def test_from_options_coerces_cfg_types():
    """A cfg value must land on the field's declared type, not as a raw string.

    The cfg is read with json.loads, which only knows lowercase true/false. Written with a
    capital F, "False" stays a STRING — and a non-empty string is truthy, so the switch
    would silently be ON while the cfg says False. This actually happened once.
    """
    c = SystemConfig.from_options({"oemof_enable_v2h": "False",
                                   "oemof_enable_pv_to_home": "FALSE",
                                   "oemof_grid_feedin_tariff": "-6.24",
                                   "oemof_storage_cycle_penalty": "0.002",
                                   "oemof_solver_threads": "4",
                                   "oemof_solver": "cbc"})
    assert c.enable_v2h is False and c.enable_pv_to_home is False
    assert c.grid_feedin_tariff == -6.24 and isinstance(c.grid_feedin_tariff, float)
    assert c.storage_cycle_penalty == 0.002
    assert c.solver_threads == 4 and isinstance(c.solver_threads, int)
    assert c.solver == "cbc"          # Strings bleiben unangetastet
    # echte JSON-Werte gehen unveraendert durch
    assert SystemConfig.from_options({"oemof_enable_v2h": False}).enable_v2h is False


def test_fallbacks_are_spice_evs_defaults():
    """Where spice_ev has a default, the model's fallback is that default.

    Read from spice_ev's own classes, built with only their required keys. The model used
    to fall back to a battery band of 0.1..1, a lossless battery starting half full and
    vehicles charged to 0.95 - none of which spice_ev does.
    """
    from spice_ev import components
    bat = components.StationaryBattery({"parent": "GC1", "charging_curve": [[0, 5], [1, 5]]})
    vt = components.VehicleType({"name": "t", "capacity": 50,
                                 "charging_curve": [[0, 11], [1, 11]]})
    veh = components.Vehicle({"vehicle_type": "t"}, {"t": vt})

    c = SystemConfig()
    assert (c.battery_min_soc, c.battery_max_soc) == (0.0, 1.0)   # charges to 1, empties to 0
    assert c.battery_initial_soc == bat.soc
    assert c.battery_efficiency == bat.efficiency
    assert c.bev_max_soc == 1.0
    assert c.bev_initial_soc == veh.battery.soc
    assert c.bev_discharge_limit == vt.discharge_limit


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
        self.capacity = self.CAPACITY
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
    # the LP got the batteries' own capacity: no rescaling of the planned SOC
    strat._lp_battery_capacity = {"BAT1": _FakeBattery.CAPACITY, "BAT2": _FakeBattery.CAPACITY}
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

    # only target_soc from the plan - no max_power, no target_power: the LP holds the rating
    assert v1.battery.calls == [("load", None, 0.55)]
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

    assert v4.battery.calls == [("unload", None, 0.55)]
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
    floor = np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0])
    m = EnergySystemModel(
        config=SystemConfig(debug=False, should_dump_results=False),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 8,
                                 "price_ct_kWh": [30.0] * 8}},
        charging_stations={"CS1": {"max_power": 22.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0, "initial_soc": 0.5,
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
# Test 3c — a vehicle that is on the road when the scenario starts (no solver)
# ---------------------------------------------------------------------------
def test_a_vehicle_on_the_road_at_scenario_start():
    """A first event that is an ARRIVAL means the vehicle starts the scenario driving.

    generate_from_csv creates every vehicle like that: not connected, first event an
    arrival carrying the soc_delta of the trip under way. spice_ev subtracts that
    soc_delta at the arrival, so the model has to see the same energy - it used to be
    dropped, because the trip had no departure to pair with.
    """
    def t(clock):
        return pd.Timestamp(f"2023-04-01 {clock}", tz="Europe/Berlin")

    def ev(vid, kind, clock, **update):
        return SimpleNamespace(vehicle_id=vid, event_type=kind, start_time=t(clock),
                               signal_time=t(clock), update=update)

    def vehicle(cs):
        return SimpleNamespace(connected_charging_station=cs, desired_soc=0.8,
                               battery=SimpleNamespace(capacity=50.0))

    start, stop = t("00:00"), pd.Timestamp("2023-04-02 00:00", tz="Europe/Berlin")
    interval = pd.Timedelta(minutes=15)
    events = [
        # a: on the road at the start, arrives 06:10, makes a second trip later
        ev("a", "arrival", "06:10", connected_charging_station="CS_a", soc_delta=-0.30),
        ev("a", "departure", "08:47"),
        ev("a", "arrival", "17:05", connected_charging_station="CS_a", soc_delta=-0.20),
        # b: the ordinary case - plugged in at the start, leaves first
        ev("b", "departure", "07:00"),
        ev("b", "arrival", "16:00", connected_charging_station="CS_b", soc_delta=-0.10),
    ]
    vehicles = {"a": vehicle(None), "b": vehicle("CS_b"), "c": vehicle("CS_c")}  # c: no events

    strat = OemofSolve.__new__(OemofSolve)
    segments = strat._build_state_segments(events, start, stop, vehicles)
    # first ROW per vehicle (groupby().first() would skip the None of an unplugged start)
    first = segments.drop_duplicates("vehicle_id").set_index("vehicle_id")
    assert first.loc["a", "state"] == "driving"
    # isna, not "is None": pandas 3 stores a missing string as NaN
    assert pd.isna(first.loc["a", "connected_charging_station"])
    assert first.loc["b", "state"] == "parked"
    assert first.loc["b", "connected_charging_station"] == "CS_b"
    assert first.loc["c", "state"] == "parked"

    trips = strat._build_trip_df(events, vehicles, start)
    assert trips.loc[trips.vehicle_id == "a", "departure_time"].tolist() == [start, t("08:47")]

    segments = strat._map_trips_to_state_segments(
        strat._group_trips_by_vehicle(trips), segments)
    per_vehicle, _ = strat._map_segments_to_timeseries(
        segments, strat._build_time_index(start, stop, interval), interval)

    def booked(vid):
        e = per_vehicle[vid]["energy_kwh"].fillna(0).astype(float)
        return {f"{ts:%H:%M}": kwh for ts, kwh in e[e > 0].items()}

    # the trip under way lands in the last step before the arrival, like every other trip
    assert booked("a") == {"06:00": pytest.approx(15.0), "17:00": pytest.approx(10.0)}
    assert booked("b") == {"15:45": pytest.approx(5.0)}
    assert booked("c") == {}


# ---------------------------------------------------------------------------
# Test 3d — "not plugged in" reaches the model as None, whatever pandas does (no solver)
# ---------------------------------------------------------------------------
def test_unplugged_steps_reach_the_model_as_none():
    """While a vehicle is away its charging station is None - not NaN.

    pandas >= 3 infers a string dtype for the segment table, where a missing value is NaN.
    ``NaN is not None`` is true, so every step counted as plugged in, no departure was
    found and the desired_soc floor before a trip silently vanished: the examples ran
    through and came out wrong (02 at 37 ct instead of 687). Under pandas 2 this test
    passes either way; it guards the pandas 3 path.
    """
    def t(clock):
        return pd.Timestamp(f"2023-04-01 {clock}", tz="Europe/Berlin")

    start, stop = t("00:00"), pd.Timestamp("2023-04-02 00:00", tz="Europe/Berlin")
    interval = pd.Timedelta(minutes=15)
    events = [
        SimpleNamespace(vehicle_id="v1", event_type="departure", start_time=t("08:47"),
                        update={}),
        SimpleNamespace(vehicle_id="v1", event_type="arrival", start_time=t("17:05"),
                        update={"connected_charging_station": "CS1", "desired_soc": 0.8}),
    ]
    vehicles = {"v1": SimpleNamespace(connected_charging_station="CS1", desired_soc=0.8)}

    strat = OemofSolve.__new__(OemofSolve)
    segments = strat._map_trips_to_state_segments(
        {}, strat._build_state_segments(events, start, stop, vehicles))
    ts = strat._map_segments_to_timeseries(
        segments, strat._build_time_index(start, stop, interval), interval)[0]["v1"]

    away = ts["state"].eq("driving").to_numpy()
    station = ts["connected_charging_station"].tolist()
    assert away.sum() == 33                                   # 09:00 .. 17:00
    assert all(c is None for c, a in zip(station, away) if a)
    assert all(c == "CS1" for c, a in zip(station, away) if not a)

    # exactly one departure step - 08:45, the last one plugged in - carries desired_soc;
    # every other step has no floor at all, as in spice_ev
    floor = OemofSolve._min_soc_series(ts)
    assert [f"{ts.index[i]:%H:%M}" for i in np.flatnonzero(floor > 0)] == ["08:45"]
    assert floor[35] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 3d2 — V2H is allowed per standing period, decided from the scenario
# ---------------------------------------------------------------------------
@requires_cbc
def test_v2h_is_decided_per_standing_period():
    """The example of chapter 4.6.8: E-Golf 50 kWh, discharge_limit 0.3, desired_soc 0.8.

    Mon 00:00-07:00 plugged in (starts with 0.6), trip -0.2, Mon 17:00-Tue 07:00 plugged
    in, long trip -0.6, Tue 20:00-Wed 07:00 plugged in. Expected arrival: 0.6, 0.6 and
    0.8 - 0.6 = 0.2. So V2H is on in the first two standing periods and off in the third.
    spice_ev lets the car arrive with 0.2 and simply not discharge; the earlier rule (floor
    0.3 in every step) made the LP charge it to 0.9 before the long trip instead.
    """
    def t(day, clock):
        return pd.Timestamp(f"2023-04-0{day} {clock}", tz="Europe/Berlin")

    def ev(kind, day, clock, **update):
        return SimpleNamespace(vehicle_id="v1", event_type=kind, start_time=t(day, clock),
                               signal_time=t(day, clock), update=update)

    start, stop, interval = t(3, "00:00"), t(5, "08:00"), pd.Timedelta(minutes=15)
    events = [ev("departure", 3, "07:00"),
              ev("arrival", 3, "17:00", connected_charging_station="CS1", soc_delta=-0.2,
                 desired_soc=0.8),
              ev("departure", 4, "07:00"),
              ev("arrival", 4, "20:00", connected_charging_station="CS1", soc_delta=-0.6,
                 desired_soc=0.8),
              ev("departure", 5, "07:00")]
    vehicles = {"v1": SimpleNamespace(connected_charging_station="CS1", desired_soc=0.8,
                                      battery=SimpleNamespace(capacity=50.0))}

    strat = OemofSolve.__new__(OemofSolve)
    segments = strat._build_state_segments(events, start, stop, vehicles)
    trips = strat._build_trip_df(events, vehicles, start)
    segments = strat._map_trips_to_state_segments(strat._group_trips_by_vehicle(trips),
                                                  segments)
    idx = strat._build_time_index(start, stop, interval)
    ts = strat._map_segments_to_timeseries(segments, idx, interval)[0]["v1"]
    mask = OemofSolve._v2h_mask(ts, 0.6, 50.0, 0.3)

    def step(day, clock):
        return int((t(day, clock) - start) / interval)

    assert mask[step(3, "00:00"):step(3, "07:00")].all()          # standing period 1: on
    assert mask[step(3, "17:00"):step(4, "07:00")].all()          # standing period 2: on
    assert not mask[step(4, "20:00"):step(5, "07:00")].any()      # standing period 3: off
    assert not mask[step(3, "07:00"):step(3, "17:00")].any()      # never while driving

    # expensive evenings make feeding the house worthwhile - where it is allowed
    price = np.full(len(idx), 10.0)
    for a, b in ((step(3, "18:00"), step(3, "22:00")), (step(4, "21:00"), step(4, "23:00"))):
        price[a:b] = 40.0

    def solve(v2h_mask):
        m = EnergySystemModel(
            config=SystemConfig(debug=False, should_dump_results=False, enable_v2h=True),
            time_index=idx,
            grid_connectors={"GC1": {"max_power": 30.0, "load": [2.0] * len(idx),
                                     "price_ct_kWh": price}},
            charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
            vehicle_params={"v1": {
                "capacity_kWh": 50.0, "initial_soc": 0.6, "v2g": True, "discharge_limit": 0.3,
                "consumption": ts["energy_kwh"].fillna(0).to_numpy(dtype=float),
                "connected_cs": ts["connected_charging_station"].to_numpy(),
                "min_soc_series": OemofSolve._min_soc_series(ts), "v2h_mask": v2h_mask}},
        )
        m.run()
        return m.get_wallbox_schedule()["v1"]

    plan = solve(mask)
    soc, ab = plan["soc_end"].to_numpy(), plan["discharge_kW"].to_numpy()
    assert soc[step(4, "06:45")] == pytest.approx(0.8, abs=1e-6)  # leaves with desired_soc
    assert soc[step(4, "19:45")] == pytest.approx(0.2, abs=1e-6)  # ... and arrives with 0.2
    assert ab[step(4, "20:00"):step(5, "07:00")].max() <= 1e-9    # no V2H below the limit
    assert ab[step(3, "17:00"):step(4, "07:00")].max() > 0.1      # V2H where it is allowed
    assert soc[step(3, "17:00"):step(4, "07:00")].min() >= 0.3 - 1e-6

    # the earlier rule - floor 0.3 in every step - had to charge it to 0.9 instead
    old = solve(np.ones(len(idx)))["soc_end"].to_numpy()
    assert old[step(4, "06:45")] == pytest.approx(0.9, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3e — a segment without a time step warns, an inverted one stops (no solver)
# ---------------------------------------------------------------------------
def test_segments_without_a_time_step_warn_and_inverted_ones_stop():
    """A segment that no grid point falls into never reaches the model.

    An 8-minute trip between 08:47 and 08:55 is such a segment: its driving demand is
    lost while spice_ev still subtracts it at the arrival - that must not pass silently.
    A zero-length segment (two events at the same moment) loses nothing and stays quiet.
    A segment that ends before it starts cannot come from time-sorted events and stops.
    """
    def t(clock):
        return pd.Timestamp(f"2023-04-01 {clock}", tz="Europe/Berlin")

    stop = pd.Timestamp("2023-04-02 00:00", tz="Europe/Berlin")
    interval = pd.Timedelta(minutes=15)
    strat = OemofSolve.__new__(OemofSolve)
    idx = strat._build_time_index(t("00:00"), stop, interval)

    def table(*segments):
        return pd.DataFrame([
            {"vehicle_id": "v1", "start_time": a, "end_time": b, "state": state,
             "energy_kwh": energy, "desired_soc": 0.8,
             "connected_charging_station": None if state == "driving" else "CS1"}
            for a, b, state, energy in segments])

    short_trip = table((t("00:00"), t("08:47"), "parked", None),
                       (t("08:47"), t("08:55"), "driving", 3.0),
                       (t("08:55"), stop, "parked", None))
    with pytest.warns(UserWarning, match=r"driving demand of 3\.00 kWh is lost"):
        ts = strat._map_segments_to_timeseries(short_trip, idx, interval)[0]["v1"]
    assert ts["energy_kwh"].fillna(0).sum() == 0          # the demand never arrived

    zero_length = table((t("00:00"), t("08:47"), "parked", None),
                        (t("08:47"), t("08:47"), "driving", None),
                        (t("08:47"), stop, "parked", None))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # any warning fails the test
        strat._map_segments_to_timeseries(zero_length, idx, interval)

    inverted = table((t("00:00"), t("09:00"), "parked", None),
                     (t("09:00"), t("08:00"), "driving", 2.0),
                     (t("08:00"), stop, "parked", None))
    with pytest.raises(ValueError, match="starts after it ends"):
        strat._map_segments_to_timeseries(inverted, idx, interval)


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
        # car starts at 50% (25 kWh) and drives 32 kWh in steps 4-7 -> it must charge before
        # it leaves, or the SOC would go below 0
        grid_connectors={"GC1": {"max_power": 30.0, "price_ct_kWh": [30.0] * 8,
                                 "load": [1] * 8, "pv": [0, 0, 2, 4, 4, 2, 0, 0]}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 10.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0, "initial_soc": 0.5,
                               "connected_cs": ["CS1", "CS1", "CS1", "CS1", None, None, None, None],
                               "consumption": [0, 0, 0, 0, 8, 8, 8, 8]}},
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


@requires_cbc
def test_a_battery_without_capacity_is_sized_not_skipped(tmp_path, monkeypatch):
    """capacity -1 means "size unknown" in spice_ev: StationaryBattery makes it 2**64, the
    simulation runs it unlimited and the report sizes it from the highest stored energy.

    The strategy used to skip it (capacity > 1e9), so the LP planned without the battery
    while spice_ev had it. Now the LP gets a capacity that cannot bind (5 kW * 2 h = 10 kWh)
    and the simulation follows the plan - both on spice_ev's 2**64 scale.
    """
    from spice_ev import scenario
    monkeypatch.chdir(tmp_path)
    t0 = "2020-01-01T00:00:00+01:00"
    s = scenario.Scenario({
        "scenario": {"start_time": t0, "interval": 15, "n_intervals": 8},
        "components": {
            "grid_connectors": {"GC1": {"max_power": 50}},
            "batteries": {"BAT1": {"parent": "GC1", "charging_curve": [[0, 5], [1, 5]]}},
            # oemof_solve needs at least one trip; this one needs no charging
            "charging_stations": {"CS1": {"max_power": 11, "parent": "GC1"}},
            "vehicle_types": {"t": {"name": "t", "capacity": 50,
                                    "charging_curve": [[0, 11], [1, 11]]}},
            "vehicles": {"v1": {"vehicle_type": "t", "soc": 0.8, "desired_soc": 0.8,
                                "connected_charging_station": "CS1"}},
        },
        "events": {
            "vehicle_events": [
                {"signal_time": t0, "start_time": "2020-01-01T00:30:00+01:00",
                 "vehicle_id": "v1", "event_type": "departure",
                 "update": {"estimated_time_of_arrival": "2020-01-01T01:00:00+01:00"}},
                {"signal_time": t0, "start_time": "2020-01-01T01:00:00+01:00",
                 "vehicle_id": "v1", "event_type": "arrival",
                 "update": {"connected_charging_station": "CS1", "soc_delta": -0.02,
                            "estimated_time_of_departure": "2020-01-02T00:00:00+01:00",
                            "desired_soc": 0.5}}],
            # cheap first hour, expensive second hour
            "grid_operator_signals": [
                {"signal_time": t0, "start_time": t0, "grid_connector_id": "GC1",
                 "cost": {"type": "fixed", "value": 0.1}},
                {"signal_time": t0, "start_time": "2020-01-01T01:00:00+01:00",
                 "grid_connector_id": "GC1", "cost": {"type": "fixed", "value": 0.4}}],
            "fixed_load": {"house": {"start_time": t0, "step_duration_s": 900,
                                     "grid_connector_id": "GC1", "values": [4] * 8}},
        },
    })
    strats = []
    original = OemofSolve._ensure_solved

    def keep(self):
        strats.append(self)
        original(self)
    monkeypatch.setattr(OemofSolve, "_ensure_solved", keep)
    s.run("oemof_solve", {"oemof_config": {"should_dump_results": False}})

    strat = strats[0]
    assert strat.world_state.batteries["BAT1"].capacity == 2 ** 64
    assert strat._lp_battery_capacity == {"BAT1": pytest.approx(10.0)}

    planned = np.array([p[2] for p in strat._plan["batteries"]["BAT1"]]) * 10.0
    simulated = np.array(s.batteryLevels["BAT1"])    # energy at the START of each step
    assert planned.max() > 1.0                       # the LP stores in the cheap hour ...
    assert planned[-1] == pytest.approx(0.0, abs=1e-6)   # ... and uses it all up later
    assert np.allclose(simulated[1:], planned[:-1], atol=1e-6)   # spice_ev follows the plan
    # spice_ev's report sizes the battery by this value
    assert max(simulated) == pytest.approx(planned.max(), abs=1e-6)

    # a battery of unknown size with a soc > 0 would hold soc * 2**64 kWh
    strat.world_state.batteries["BAT1"].soc = 0.5
    with pytest.raises(ValueError, match="no capacity"):
        strat._battery_params(strat._model.config)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Test 5 — PV, Speicher und Wallbox ohne Direktzweige
# ---------------------------------------------------------------------------
def _pv_scenario(config, pv=(0, 0, 20, 20, 20, 20, 0, 0), v2g=False, load=8):
    """PV-Ueberschuss bei angestecktem Auto - die Lage, um die es geht.

    Die Haushaltslast muss gross genug sein, dass das Netz wirklich gebraucht wird: mit
    kleiner Last traegt die 20-kWh-Batterie alles umsonst, und das LP exportiert jede kWh.
    """
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    return EnergySystemModel(
        config=config,
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 40.0, "load": [load] * 8, "pv": list(pv),
                                 "price_ct_kWh": [30.0] * 8}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 20.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 80.0, "initial_soc": 0.2,
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
    # der Wechselrichter speist direkt ins Haus, ohne eigene Leistungsgrenze
    pv_conv = n["converter_pv_to_home_Home_1"]
    assert list(pv_conv.inputs.values())[0].nominal_value is None
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
def test_station_rating_holds_and_pv_is_not_capped():
    """Die Wallbox hat eine Grenze, die der Fahrplan einhalten MUSS - die PV hat keine.

    Die Wallbox: step() gibt spice_ev nur den Ziel-SOC, ohne max_power. Die Stationsleistung
    haelt allein der Plan ein - ein Plan, der mehr verlangt, liesse die Simulation die
    Station ueberschreiten.
    Die PV: ihre Zeitreihe ist schon die Leistung am Netzanschluss, und spice_ev bucht sie
    dort vollstaendig. Der Wechselrichter darf davon nichts zurueckhalten - bei 30 kW PV
    deckt sie die 15 kW Hauslast in jedem Schritt, das Netz liefert nichts. (Die fruehere
    Ersatzgrenze von 10 kW haette hier 5 kW aus dem Netz verlangt.)
    """
    m = _pv_scenario(SystemConfig(debug=False, should_dump_results=False),
                     pv=(30,) * 8, load=15)
    m.run()
    laden = m.get_wallbox_schedule()["v1"]["charge_kW"].to_numpy()
    assert np.all(laden <= 11.0 + 1e-6)
    eigen = m._summary_df["pv_selfuse_Home_1"].to_numpy()
    assert np.all(eigen >= 15.0 - 1e-6)
    assert m._summary_df["grid_supply_Home_1"].max() <= 1e-6


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

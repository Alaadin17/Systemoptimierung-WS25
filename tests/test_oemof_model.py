"""Tests for the per-grid-connector oemof model (spice_ev/oemof_model.py).

Autor: Alaa Alsleman, GitHub: Alaadin17

Four tests need no solver: they build the EnergySystem (or call a helper) and check the
per-GC pruning / naming / wiring, that load and PV are grouped per grid connector, and
that the min-charging-power override works. The remaining three need CBC (they are
skipped without it) and actually solve: the debug mode, the per-step SOC floor from
``desired_soc``, and a full ``run()`` including the result dumps.
"""
import dataclasses
import json
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
    m._load_data()
    m._create_time_index()
    m._create_energy_system()
    m._create_components()
    return {n.label for n in m.es.nodes}


def _nodes(m):
    return {n.label: n for n in m.es.nodes}


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
    assert not any(l.startswith("wallbox_charge_CS3_") for l in labels)   # CS3 unused
    assert not any(l.startswith("wallbox_") and l.endswith("_v3") for l in labels)  # v3 unused
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
    strat.cost_parameters_file = None        # no price sheet in this test
    strat.events = SimpleNamespace(
        fixed_load_lists={"L1": _ev([1, 1, 1, 1], start, "GC1"),
                          "L2": _ev([2, 2, 2, 2], start, "GC2")},
        local_generation_lists={"PV1": _ev([0, 3, 3, 0], start, "GC1")},
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


# ---------------------------------------------------------------------------
# Test 3 — solve a tiny feasible model with CBC (debug mode on)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
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
    m._load_data()
    m._create_time_index()
    m._create_energy_system()
    m._create_components()
    m._optimize()
    with caplog.at_level(logging.INFO):
        m._solve()             # raises RuntimeError if not optimal

    assert math.isfinite(m.model.objective())
    assert (tmp_path / "lp_out" / "dump_debug.lp").exists()    # LP dump in config.output_dir
    assert "oemof solved" in caplog.text                        # _solve debug log line


# ---------------------------------------------------------------------------
# Test 2a2 — per-GC price series, feed-in tariff and PV power reach the model
# ---------------------------------------------------------------------------
def test_per_gc_price_tariff_and_pv_power_from_scenario():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=SystemConfig(debug=False),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1],
                                 "pv": [0, 3, 3, 0],
                                 "price_ct_kWh": np.array([30.0, 5.0, 5.0, 30.0]),
                                 "feedin_tariff_ct_kWh": -6.24,
                                 "homebus_feedin_tariff_ct_kWh": 0.0,
                                 "pv_power_kW": 7.5}},
    )
    _build_es(m)
    nodes = _nodes(m)
    supply = _out_flow(nodes["grid_supply_Home_1"])
    assert [float(supply.variable_costs[t]) for t in range(4)] == [30.0, 5.0, 5.0, 30.0]
    # the two export paths have DIFFERENT tariffs: PV earns, home-bus export does not
    assert float(list(nodes["grid_feedin_Home_1"].inputs.values())[0].variable_costs[0]) == 0.0
    assert float(list(nodes["excess_Home_1"].inputs.values())[0].variable_costs[0]) == -6.24
    assert list(nodes["converter_pv_to_home_Home_1"].inputs.values())[0].nominal_value == 7.5


def test_feedin_tariff_from_price_sheet_switch():
    """feedin_tariff_from_price_sheet = False lets the cfg value win over the price sheet.

    Otherwise ``oemof_grid_feedin_tariff`` is silently ignored whenever a price sheet is
    configured — the surprise this switch exists to remove. The grid SUPPLY price is not
    affected either way (that is what the tariff steers).
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    gc = {"GC1": {"max_power": 30.0, "load": [1, 1, 1, 1], "pv": [0, 3, 3, 0],
                  "price_ct_kWh": np.array([30.0, 5.0, 5.0, 30.0]),
                  "feedin_tariff_ct_kWh": -6.24,          # from the price sheet
                  "homebus_feedin_tariff_ct_kWh": -1.5}}

    def tariffs(cfg):
        m = EnergySystemModel(config=cfg, time_index=idx,
                              grid_connectors={k: dict(v) for k, v in gc.items()})
        _build_es(m)
        n = _nodes(m)
        return (float(list(n["excess_Home_1"].inputs.values())[0].variable_costs[0]),
                float(list(n["grid_feedin_Home_1"].inputs.values())[0].variable_costs[0]),
                [float(_out_flow(n["grid_supply_Home_1"]).variable_costs[t]) for t in range(4)])

    # default: the sheet wins on BOTH paths, each with its own value
    assert tariffs(SystemConfig(debug=False, grid_feedin_tariff=0.0))[:2] == (-6.24, -1.5)
    # switched off: the cfg value wins on both paths
    assert tariffs(SystemConfig(debug=False, grid_feedin_tariff=0.0,
                                feedin_tariff_from_price_sheet=False))[:2] == (0.0, 0.0)
    # ... and it really is the cfg value, not a hard-coded zero
    assert tariffs(SystemConfig(debug=False, grid_feedin_tariff=-8.0,
                                feedin_tariff_from_price_sheet=False))[:2] == (-8.0, -8.0)
    # the purchase price is untouched by the switch
    assert tariffs(SystemConfig(debug=False, feedin_tariff_from_price_sheet=False))[2] == \
        [30.0, 5.0, 5.0, 30.0]


def test_tariff_selects_the_price_build_up(caplog):
    """RLM | SLP | fixed — one switch instead of fee_type plus use_retail_markup.

    RLM is the default because simulate.py bills every strategy outside
    greedy/balanced/distributed as RLM (its line 71), and oemof_solve is one of them. With
    the previous SLP default the LP priced 7.48 ct/kWh of grid fee and no capacity charge
    while the bill was written with 3.49 ct/kWh plus 41.06 EUR/(kW*a).
    """
    assert SystemConfig().tariff == "RLM"
    assert SystemConfig().include_capacity_charge is False
    # die alten Felder gibt es nicht mehr
    felder = {f.name for f in dataclasses.fields(SystemConfig)}
    assert "fee_type" not in felder and "use_retail_markup" not in felder

    strat = OemofSolve.__new__(OemofSolve)
    for wert, erwartet in (("RLM", "RLM"), ("SLP", "SLP"), ("fixed", "fixed"),
                           ("rlm", "RLM"), ("FIXED", "fixed")):
        strat._oemof_cfg = SystemConfig(tariff=wert)
        assert strat.tariff() == erwartet, wert
    # Unsinn faellt NICHT still auf einen Zweig, sondern warnt und nimmt den Default
    strat._oemof_cfg = SystemConfig(tariff="Haushalt")
    with caplog.at_level(logging.WARNING):
        assert strat.tariff() == "RLM"
    assert "Haushalt" in caplog.text and "unbekannt" in caplog.text


def test_from_options_coerces_cfg_types():
    """A cfg value must land on the field's declared type, not as a raw string.

    The cfg is read with json.loads, which only knows lowercase true/false. Written with a
    capital F, "False" stays a STRING — and a non-empty string is truthy, so the switch
    would silently be ON while the cfg says False. This actually happened once.
    """
    c = SystemConfig.from_options({"oemof_enable_v2h": "False",
                                   "oemof_grid_price_from_scenario": "FALSE",
                                   "oemof_pv_direct_to_storage": "yes",
                                   "oemof_grid_variable_costs": "22.5",
                                   "oemof_solver_threads": "4",
                                   "oemof_solver": "cbc"})
    assert c.enable_v2h is False and c.grid_price_from_scenario is False
    assert c.pv_direct_to_storage is True
    assert c.grid_variable_costs == 22.5 and isinstance(c.grid_variable_costs, float)
    assert c.solver_threads == 4 and isinstance(c.solver_threads, int)
    assert c.solver == "cbc"          # Strings bleiben unangetastet
    # echte JSON-Werte gehen unveraendert durch
    assert SystemConfig.from_options({"oemof_enable_v2h": False}).enable_v2h is False


def test_fixed_grid_price_overrides_the_scenario_signals():
    """grid_price_from_scenario = False turns grid_variable_costs into a FIXED price.

    It must return a constant series rather than None, so the retail markup still runs and
    a fixed run stays comparable to a variable one.
    """
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    strat = OemofSolve.__new__(OemofSolve)
    strat.events = SimpleNamespace(grid_operator_signals=[
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                        cost={"type": "fixed", "value": 0.30}),
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[2],
                        cost={"type": "fixed", "value": 0.05}),
    ])
    # Default: the scenario's signals win and the price varies
    strat._oemof_cfg = SystemConfig(debug=False)
    assert list(strat._grid_price_series("GC1", idx)) == [30.0, 30.0, 5.0, 5.0]
    # Switched off: one constant price from the config, signals ignored
    strat._oemof_cfg = SystemConfig(debug=False, grid_price_from_scenario=False,
                                    grid_variable_costs=22.5)
    assert list(strat._grid_price_series("GC1", idx)) == [22.5] * 4
    # ... and it is a series, not None — otherwise the retail markup would be skipped
    assert strat._grid_price_series("GC1", idx) is not None


def test_strategy_sources_price_and_feedin_from_spice_ev(tmp_path):
    strat = OemofSolve.__new__(OemofSolve)
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    strat.events = SimpleNamespace(grid_operator_signals=[
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[0],
                        cost={"type": "fixed", "value": 0.30}),      # EUR/kWh!
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[2],
                        cost={"type": "fixed", "value": 0.05}),
        SimpleNamespace(grid_connector_id="GC1", start_time=idx[3],
                        cost={"type": "fixed", "value": -0.04}),     # negativ -> gekappt
        SimpleNamespace(grid_connector_id="GC2", start_time=idx[0],
                        cost={"type": "fixed", "value": 0.99}),
    ])
    sheet = {"default_grid_operator": {"feed-in_remuneration": {
        "PV": {"kWp": [10, 40, 100], "remuneration": [6.24, 6.06, 4.74]}}}}
    sheet_path = tmp_path / "price_sheet.json"
    sheet_path.write_text(json.dumps(sheet), encoding="utf-8")
    strat.cost_parameters_file = str(sheet_path)
    strat._price_sheet = None
    strat.world_state = SimpleNamespace(
        grid_connectors={"GC1": SimpleNamespace(grid_operator="default_grid_operator")},
        photovoltaics={"PV1": SimpleNamespace(parent="GC1", nominal_power=10.0)},
    )

    # signals -> ct/kWh, piecewise constant, only the matching GC; negatives clipped to 0
    assert list(strat._grid_price_series("GC1", idx)) == [30.0, 30.0, 5.0, 0.0]
    assert strat._grid_price_series("GC3", idx) is None          # no signals -> fallback

    # tariff markup, mirroring spice_ev's costs.py (same SLP/RLM names and values):
    # commodity + levies + concession + electricity tax, plus the VAT rate
    strat._price_sheet = None   # reload with fee components
    sheet["default_grid_operator"].update({
        "grid_fee": {"SLP": {"commodity_charge_ct/kWh": {"net_price": 7.48}},
                     "RLM": {"<2500_h/a": {"commodity_charge_ct/kWh": {"MV": 3.49}}}},
        "levies": {"EEG_levy": 0, "chp_levy": 0.378, "individual_charge_levy": 0.437,
                   "offshore_levy": 0.419, "interruptible_loads_levy": 0.003},
        "concession_fee": {"charge": 1.32},
        "taxes": {"value_added_tax": 19, "tax_on_electricity": 2.05},
    })
    sheet_path.write_text(json.dumps(sheet), encoding="utf-8")
    strat.world_state.grid_connectors["GC1"].voltage_level = "MV"
    strat._oemof_cfg = SystemConfig(tariff="SLP")
    markup, vat = strat._retail_markup_ct("GC1")
    assert abs(markup - (7.48 + 1.237 + 1.32 + 2.05)) < 1e-9     # = 12.087 ct netto
    assert vat == 19
    strat._oemof_cfg = SystemConfig()                            # Default RLM
    markup_rlm, _ = strat._retail_markup_ct("GC1")               # RLM: je Spannungsebene
    assert abs(markup_rlm - (3.49 + 1.237 + 1.32 + 2.05)) < 1e-9
    strat._oemof_cfg = SystemConfig(tariff="fixed")              # fixed: gar kein Aufschlag
    assert strat._retail_markup_ct("GC1") is None

    # feed-in from the price sheet, staggered by installed kWp (negative = revenue)
    assert strat._feedin_tariff_ct("GC1") == -6.24               # 10 kWp -> first step
    strat.world_state.photovoltaics["PV1"].nominal_power = 50.0
    assert strat._feedin_tariff_ct("GC1") == -4.74               # 50 kWp -> <=100 step
    strat.world_state.photovoltaics["PV1"].parent = "GC_other"
    assert strat._feedin_tariff_ct("GC1") is None                # no PV here -> fallback


# ---------------------------------------------------------------------------
# Test 2b — switching spice_ev's minimum charging power off
# ---------------------------------------------------------------------------
def _fake_world(min_charging_power=0.2, cs_min_power=0.5):
    """world_state stand-in with one vehicle type shared by two vehicles."""
    vtype = SimpleNamespace(min_charging_power=min_charging_power)
    return SimpleNamespace(
        vehicles={"v1": SimpleNamespace(vehicle_type=vtype),
                  "v2": SimpleNamespace(vehicle_type=vtype)},      # same (shared) type object
        charging_stations={"CS1": SimpleNamespace(min_power=cs_min_power)},
    )


def test_ignore_min_charging_power_clears_both_limits():
    strat = OemofSolve.__new__(OemofSolve)          # bypass __init__
    strat.world_state = _fake_world()
    strat._apply_min_power_override(SystemConfig(ignore_min_charging_power=True))

    assert strat.world_state.vehicles["v1"].vehicle_type.min_charging_power == 0.0
    assert strat.world_state.vehicles["v2"].vehicle_type.min_charging_power == 0.0
    assert strat.world_state.charging_stations["CS1"].min_power == 0.0


def test_min_charging_power_untouched_by_default():
    strat = OemofSolve.__new__(OemofSolve)
    strat.world_state = _fake_world()
    strat._apply_min_power_override(SystemConfig())   # default: flag off

    assert strat.world_state.vehicles["v1"].vehicle_type.min_charging_power == 0.2
    assert strat.world_state.charging_stations["CS1"].min_power == 0.5


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
    strat._oemof_cfg = SystemConfig()
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
@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
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
    m._load_data()
    m._create_time_index()
    m._create_energy_system()
    m._create_components()
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
@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
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
# Test 5 — splitting the storage buses changes nothing on its own
# ---------------------------------------------------------------------------
def _split_scenario(config):
    """A small PV + battery + car scenario, built with the given config.

    Two things make the optimum UNIQUE, which any flow-level comparison needs:
    - ``homebus_feedin_tariff_ct_kWh = 0`` mirrors the real price sheet. On the config
      fallback (-8.0) exporting from the home bus PAYS, so with a cheap price window the LP
      finds a money pump — charge cheap, export at a profit — with countless optima.
    - the household load is big enough that the 10 kWh battery cannot carry it alone. With a
      small load the LP simply drains the battery, buys NOTHING, and the marginal cost is
      zero in every step — then the price series has no effect at all and every schedule ties.
    """
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    return EnergySystemModel(
        config=config,
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [6] * 8,
                                 "pv": [0, 0, 2, 4, 4, 2, 0, 0],
                                 # deliberately ALL DISTINCT: equal prices make charging in
                                 # either step a tie, and CBC then picks an arbitrary vertex
                                 "price_ct_kWh": np.array([41, 39, 5, 6, 42, 43, 44, 45.0]),
                                 "feedin_tariff_ct_kWh": -6.24,
                                 "homebus_feedin_tariff_ct_kWh": 0.0,
                                 "pv_power_kW": 10.0}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 10.0, "power_kW": 5.0, "parent": "GC1"}},
        # connected for the first half, then drives — so charging is a real need, not arbitrage.
        # 20 kWh of driving against 25 kWh start and a 10 kWh floor forces >= 5 kWh of charging.
        vehicle_params={"v1": {"capacity_kWh": 50.0, "initial_soc": 0.5, "min_soc": 0.2,
                               "v2g": True, "discharge_limit": 0.3,
                               "connected_cs": ["CS1"] * 4 + [None] * 4,
                               "consumption": [0, 0, 0, 0, 5, 5, 5, 5]}},
    )


@pytest.mark.parametrize("v2h", [False, True])
def test_storage_bus_split_builds_the_expected_nodes(v2h):
    """pv_direct_to_storage splits each storage into a pure inflow and outflow side.

    The vehicle only needs it when V2H exists — without a way back into the house the
    mobility bus has the storage as its only consumer and cannot be passed through.
    """
    off = _split_scenario(SystemConfig(debug=False, enable_v2h=v2h))
    on = _split_scenario(SystemConfig(debug=False, enable_v2h=v2h, pv_direct_to_storage=True))
    labels_off, labels_on = _build_es(off), _build_es(on)

    assert "bus_battery_BAT1" in labels_off
    assert {"bus_batin_BAT1", "bus_batout_BAT1"} & labels_off == set()
    assert {"bus_batin_BAT1", "bus_batout_BAT1"} <= labels_on
    assert "bus_battery_BAT1" not in labels_on
    # the vehicle bus is split only when there is a V2H path back into the house
    assert "bus_mobin_v1" not in labels_off
    assert ("bus_mobin_v1" in labels_on) is v2h

    # wiring: charging arrives on the inflow side, discharging leaves from the outflow side
    n = _nodes(on)
    storage = n["home_battery_BAT1"]
    assert list(storage.inputs)[0].label == "bus_batin_BAT1"
    assert list(storage.outputs)[0].label == "bus_batout_BAT1"
    assert list(n["wallbox_charge_CS1_v1"].outputs)[0].label == \
        ("bus_mobin_v1" if v2h else "bus_mobility_v1")
    if v2h:
        # V2H still feeds back from the OUTflow side, so no pass-through exists
        assert list(n["wallbox_discharge_CS1_v1"].inputs)[0].label == "bus_mobility_v1"
        assert list(n["bev_battery_v1"].inputs)[0].label == "bus_mobin_v1"
        assert list(n["bev_battery_v1"].outputs)[0].label == "bus_mobility_v1"


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
@pytest.mark.parametrize("v2h", [False, True])
def test_storage_bus_split_is_cost_neutral(v2h):
    """The split costs nothing and delivers the same physical schedule.

    Asserted on the NET power and the SOC, not on charge/discharge separately, and that is
    deliberate: the split also removes the degenerate cycles that the shared bus allows
    (``Home -> link -> bus_battery -> link -> Home``, and charge+discharge in the same step
    with V2H). Those cycles are lossless or nearly so, so they cost ~nothing and the solver
    may or may not include them — but they inflate charge AND discharge by the same amount
    and cancel in the net. Removing them is the point of the split, not a side effect.
    """
    off = _split_scenario(SystemConfig(debug=False, enable_v2h=v2h, should_dump_results=False))
    on = _split_scenario(SystemConfig(debug=False, enable_v2h=v2h, should_dump_results=False,
                                      pv_direct_to_storage=True))
    off.run()
    on.run()

    assert off._costs["objective"] == pytest.approx(on._costs["objective"], abs=1e-6)
    # what physically happens must be identical: net power and the SOC trajectory
    for col in ("net_kW", "soc_kWh", "soc_end"):
        assert np.allclose(off.get_wallbox_schedule()["v1"][col],
                           on.get_wallbox_schedule()["v1"][col], atol=1e-6), col
    bat_off, bat_on = (x.get_plan()["batteries"]["BAT1"] for x in (off, on))
    assert np.allclose(bat_off["charge_kW"] - bat_off["discharge_kW"],
                       bat_on["charge_kW"] - bat_on["discharge_kW"], atol=1e-6)
    assert np.allclose(bat_off["soc_end"], bat_on["soc_end"], atol=1e-6)
    assert np.allclose(off._summary_df["grid_supply_Home_1"],
                       on._summary_df["grid_supply_Home_1"], atol=1e-6)
    assert np.allclose(off._summary_df["pv_feedin_Home_1"],
                       on._summary_df["pv_feedin_Home_1"], atol=1e-6)

    # and the split really does forbid the pointless cycling: never both directions at once
    assert (np.minimum(bat_on["charge_kW"], bat_on["discharge_kW"]) < 1e-6).all()
    sched_on = on.get_wallbox_schedule()["v1"]
    assert (np.minimum(sched_on["charge_kW"], sched_on["discharge_kW"]) < 1e-6).all()


# ---------------------------------------------------------------------------
# Test 6 — the PV direct branches: wiring, limits, balance, incentive
# ---------------------------------------------------------------------------
def _pv_direct_config(**kw):
    return SystemConfig(debug=False, should_dump_results=False,
                        pv_direct_to_storage=True, **kw)


def test_pv_direct_branch_wiring():
    """The branches start at the PV-AC bus and end on the storages' inflow sides."""
    m = _split_scenario(_pv_direct_config())
    labels = _build_es(m)
    assert {"bus_pvac_Home_1", "conv_pvac_to_home_Home_1",
            "conv_pv_to_wallbox_CS1_v1", "conv_pv_to_battery_BAT1",
            "bus_wbin_CS1_v1"} <= labels
    n = _nodes(m)
    # the inverter rating sits on the ONE converter everything hangs behind
    pv_conv = n["converter_pv_to_home_Home_1"]
    assert list(pv_conv.inputs.values())[0].nominal_value == 10.0
    assert list(pv_conv.outputs)[0].label == "bus_pvac_Home_1"
    for lbl in ("conv_pvac_to_home_Home_1", "conv_pv_to_wallbox_CS1_v1",
                "conv_pv_to_battery_BAT1"):
        assert list(n[lbl].inputs)[0].label == "bus_pvac_Home_1"
    # both wallbox paths meet on the terminal, whose only exit carries the station rating
    assert list(n["conv_home_to_wallbox_CS1_v1"].outputs)[0].label == "bus_wbin_CS1_v1"
    assert list(n["conv_pv_to_wallbox_CS1_v1"].outputs)[0].label == "bus_wbin_CS1_v1"
    wb = n["wallbox_charge_CS1_v1"]
    assert list(wb.inputs)[0].label == "bus_wbin_CS1_v1"
    assert list(wb.inputs.values())[0].nominal_value == 11.0
    # PV into the battery lands on the inflow side, never on the shared/outflow bus
    assert list(n["conv_pv_to_battery_BAT1"].outputs)[0].label == "bus_batin_BAT1"


def test_pv_direct_branches_carry_the_bonus_as_negative_cost():
    m = _split_scenario(_pv_direct_config(pv_charge_bonus_vehicle_ct_kWh=8.0,
                                          pv_charge_bonus_battery_ct_kWh=2.0))
    _build_es(m)
    n = _nodes(m)
    assert float(list(n["conv_pv_to_wallbox_CS1_v1"].inputs.values())[0].variable_costs[0]) == -8.0
    assert float(list(n["conv_pv_to_battery_BAT1"].inputs.values())[0].variable_costs[0]) == -2.0
    # the plain household branch stays free, and the grid path keeps the late-charging ramp
    assert float(list(n["conv_pvac_to_home_Home_1"].inputs.values())[0].variable_costs[0]) == 0.0


def test_pv_direct_skipped_without_pv():
    """No PV on the GC -> no branches, and the wallbox keeps its direct house feed."""
    idx = pd.date_range("2025-01-01", periods=4, freq="15min")
    m = EnergySystemModel(
        config=_pv_direct_config(),
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 30.0, "load": [1] * 4}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 10.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 50.0, "connected_cs": ["CS1"] * 4}},
    )
    labels = _build_es(m)
    assert not any(lb.startswith(("bus_pvac_", "bus_wbin_", "conv_pv_to_")) for lb in labels)
    assert list(_nodes(m)["wallbox_charge_CS1_v1"].inputs)[0].label == "Home_1"


def _pv_rich_scenario(config, pv_power_kW=10.0, pv=(0, 0, 20, 20, 20, 20, 0, 0)):
    """PV surplus with the car plugged in the whole time — the situation the bonus targets.

    The household load must be big enough that the grid is actually needed: with a small
    load the 20 kWh battery carries everything for free, PV self-consumption then displaces
    nothing, and the LP exports every kWh no matter what the bonus says.
    """
    idx = pd.date_range("2025-01-01", periods=8, freq="15min")
    return EnergySystemModel(
        config=config,
        time_index=idx,
        grid_connectors={"GC1": {"max_power": 40.0, "load": [8] * 8, "pv": list(pv),
                                 "price_ct_kWh": np.full(8, 30.0),
                                 "feedin_tariff_ct_kWh": -6.24,
                                 "homebus_feedin_tariff_ct_kWh": 0.0,
                                 "pv_power_kW": pv_power_kW}},
        charging_stations={"CS1": {"max_power": 11.0, "parent": "GC1"}},
        battery_params={"BAT1": {"capacity_kWh": 20.0, "power_kW": 5.0, "parent": "GC1"}},
        vehicle_params={"v1": {"capacity_kWh": 80.0, "initial_soc": 0.2, "min_soc": 0.1,
                               "connected_cs": ["CS1"] * 8, "consumption": [0] * 8}},
    )


def _flow_between(m, src_label, dst_label, n=8):
    """Solved flow between two nodes, looked up by label."""
    nodes = _nodes(m)
    seq = m._results_main[(nodes[src_label], nodes[dst_label])]["sequences"]["flow"]
    return np.asarray(seq.to_numpy()[:n], dtype=float)


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_respects_the_station_rating():
    """Even with an absurd bonus the two paths together stay within cs.max_power.

    This is the limit step() depends on: it steers the SOC with
    ``Battery.load(max_power=cs.max_power)``, so a plan asking for more would silently be
    clipped and the simulated SOC would fall behind the planned one.
    """
    m = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_vehicle_ct_kWh=100.0),
                          pv_power_kW=30.0, pv=(30,) * 8)
    m.run()
    charge = m.get_wallbox_schedule()["v1"]["charge_kW"].to_numpy()
    assert charge.max() <= 11.0 + 1e-6
    assert charge.max() == pytest.approx(11.0, abs=1e-6)      # the bonus really pushes it there
    grid = _flow_between(m, "Home_1", "conv_home_to_wallbox_CS1_v1")
    pv_branch = _flow_between(m, "bus_pvac_Home_1", "conv_pv_to_wallbox_CS1_v1")
    assert np.all(grid + pv_branch <= 11.0 + 1e-6)
    assert np.allclose(grid + pv_branch, charge, atol=1e-6)


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_respects_the_inverter_rating():
    """All three PV destinations together stay within the inverter rating."""
    m = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_vehicle_ct_kWh=100.0,
                                            pv_charge_bonus_battery_ct_kWh=100.0),
                          pv_power_kW=6.0, pv=(30,) * 8)
    m.run()
    s = m._summary_df
    total = (s["pv_to_home_Home_1"] + s["pv_direct_wallbox_v1"]
             + s["pv_direct_battery_BAT1"]).to_numpy()
    assert np.all(total <= 6.0 + 1e-6)
    assert total.max() == pytest.approx(6.0, abs=1e-6)        # and it is actually binding


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_cannot_bypass_the_battery():
    """Everything that reaches the battery bus goes INTO the storage — no pass-through.

    Without the inflow/outflow split a bonused PV kWh could enter bus_battery, walk straight
    back out through the link into the house and collect the bonus without ever being
    stored. Here the storage inflow is the only exit, so the two arrivals must add up to it
    exactly. This assertion is the reason the split exists.
    """
    m = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_battery_ct_kWh=100.0))
    m.run()
    pv_in = _flow_between(m, "bus_pvac_Home_1", "conv_pv_to_battery_BAT1")
    link_in = _flow_between(m, "Home_1", "link_home_battery_BAT1")
    storage_in = _flow_between(m, "bus_batin_BAT1", "home_battery_BAT1")
    assert np.allclose(pv_in + link_in, storage_in, atol=1e-6)
    assert np.all(storage_in <= 5.0 + 1e-6)                   # the battery's power limit
    assert pv_in.sum() > 0                                    # the branch is actually used


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_energy_balance_and_reporting():
    """Generation splits exactly into export + the three self-use branches."""
    m = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_vehicle_ct_kWh=1.0,
                                            pv_charge_bonus_battery_ct_kWh=1.0))
    m.run()
    s = m._summary_df
    branches = (s["pv_to_home_Home_1"] + s["pv_direct_wallbox_v1"]
                + s["pv_direct_battery_BAT1"])
    assert np.allclose(s["pv_selfuse_Home_1"], branches, atol=1e-6)
    assert np.allclose(s["pv_Home_1"] - s["pv_feedin_Home_1"], s["pv_selfuse_Home_1"], atol=1e-6)
    # the battery plan must count the direct inflow too, or the CSV under-reports the power
    assert np.allclose(m.get_plan()["batteries"]["BAT1"]["charge_kW"],
                       s["battery_charge_BAT1"], atol=1e-6)
    assert np.allclose(s["battery_charge_BAT1"],
                       _flow_between(m, "bus_batin_BAT1", "home_battery_BAT1"), atol=1e-6)


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_bonus_moves_pv_into_the_car():
    """A vehicle bonus shifts PV from the house battery into the car — and costs money."""
    base = _pv_rich_scenario(_pv_direct_config())
    nudged = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_vehicle_ct_kWh=20.0))
    base.run()
    nudged.run()
    assert (nudged._summary_df["pv_direct_wallbox_v1"].sum()
            > base._summary_df["pv_direct_wallbox_v1"].sum())
    # the real energy cost gets WORSE — that is the price of leaving the cost optimum
    assert nudged._costs["objective_ohne_bonus"] > base._costs["objective_ohne_bonus"] - 1e-9
    # and the artificial part is exactly reconstructible from the reported flows
    step_h = 0.25
    expected = -20.0 * nudged._summary_df["pv_direct_wallbox_v1"].sum() * step_h
    assert nudged._costs["pv_bonus_ct"] == pytest.approx(expected, abs=1e-6)
    assert nudged._costs["objective_ohne_bonus"] == pytest.approx(
        nudged._costs["objective"] - nudged._costs["pv_bonus_ct"], abs=1e-9)
    assert base._costs["pv_bonus_ct"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_forbid_simultaneous_storage_stops_the_circulation():
    """The binary "charge XOR discharge" removes the circulation a large bonus induces.

    Without it a bonus above (1 - eff^2) * feed-in tariff makes it profitable to route PV
    *through* the storage into the house: every single flow is legal, together they only
    burn the round-trip efficiency while collecting the full bonus. That is an
    either/or statement, which a pure LP cannot express — hence one binary per storage and
    step. This test pins both halves: the circulation exists, and the switch removes it.
    """
    frei = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_battery_ct_kWh=100.0))
    fest = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_battery_ct_kWh=100.0,
                                               forbid_simultaneous_storage=True))
    frei.run()
    fest.run()
    b_frei = frei.get_plan()["batteries"]["BAT1"]
    b_fest = fest.get_plan()["batteries"]["BAT1"]
    assert np.minimum(b_frei["charge_kW"], b_frei["discharge_kW"]).max() > 1e-6
    assert np.minimum(b_fest["charge_kW"], b_fest["discharge_kW"]).max() < 1e-6
    # and the reported PV-into-storage now really is stored: no more than the SOC can hold
    soc = fest._summary_df["home_battery_BAT1_soc_kWh"].to_numpy()
    gespeichert = np.maximum(np.diff(soc, prepend=soc[0]), 0.0).sum()
    assert fest._summary_df["pv_direct_battery_BAT1"].sum() * 0.25 <= gespeichert / 0.95 + 1e-6


def test_forbid_simultaneous_storage_is_off_and_lp_stays_an_lp():
    """Default off — and with it off no binary variable is created at all."""
    assert SystemConfig().forbid_simultaneous_storage is False
    m = _split_scenario(_pv_direct_config())
    m._load_data()
    m._create_time_index()
    m._create_energy_system()
    m._create_components()
    m._optimize()
    assert not hasattr(m.model, "speicher_modus")
    # the registry is filled either way, so switching on needs no rebuild of the topology
    assert {p["label"] for p in m._storage_pairs} == {"home_battery_BAT1", "bev_battery_v1"}


@pytest.mark.skipif(shutil.which("cbc") is None, reason="CBC solver not installed")
def test_pv_direct_malus_has_no_effect():
    """Only POSITIVE bonuses act: a malus just leaves the direct branch unused.

    The branch is optional — with a penalty on it the LP simply routes the PV the long way
    round (conv_pvac_to_home -> Home_1 -> link) and reaches the same battery at the same
    cost. Worth pinning down, because it means one cannot make the car win by punishing the
    house battery; the car's bonus has to carry the whole difference.
    """
    neutral = _pv_rich_scenario(_pv_direct_config())
    bonus = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_battery_ct_kWh=1.0))
    malus = _pv_rich_scenario(_pv_direct_config(pv_charge_bonus_battery_ct_kWh=-5.0))
    for m in (neutral, bonus, malus):
        m.run()
    # a positive bonus DOES pull PV onto the direct branch — without this the test below
    # would pass vacuously
    assert bonus._summary_df["pv_direct_battery_BAT1"].sum() > 0
    # the malus leaves it completely unused and costs exactly nothing
    assert malus._summary_df["pv_direct_battery_BAT1"].sum() == pytest.approx(0.0, abs=1e-6)
    assert malus._costs["objective"] == pytest.approx(neutral._costs["objective"], abs=1e-6)

"""Tests for the per-grid-connector oemof model (spice_ev/oemof_model.py).

Two topology tests build the EnergySystem WITHOUT solving (fast, CBC-free) and check
the per-GC pruning / naming / wiring. A third test actually solves a tiny feasible
model with CBC and checks the debug mode (LP dump + log line + finite objective).
"""
import logging
import math
import shutil
from types import SimpleNamespace

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
        config=SystemConfig(debug=True, enable_grid_feedin=False),
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
    assert (tmp_path / "results" / "dump_debug.lp").exists()   # _optimize debug dump
    assert "oemof solved" in caplog.text                        # _solve debug log line

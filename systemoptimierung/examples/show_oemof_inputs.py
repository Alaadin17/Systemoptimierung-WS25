"""Show, stage by stage, how oemof_solve turns a spice_ev scenario into model inputs.

The strategy does this in two methods, and this script prints the result of every stage
as a table, so the data preparation can be read off real numbers instead of code:

    prepare_inputs()       events -> trips -> state segments -> timeseries on the time grid
    build_oemof_inputs()   the six dicts EnergySystemModel is built from

Nothing is solved here - no CBC, no LP. Run it on any scenario:

    python systemoptimierung/examples/show_oemof_inputs.py
    python systemoptimierung/examples/show_oemof_inputs.py 02_commercial_fleet
    python systemoptimierung/examples/show_oemof_inputs.py 03_household_v2g --vehicle golf_0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:            # runnable from anywhere, no install needed
    sys.path.insert(0, str(ROOT))

from spice_ev.scenario import Scenario                      # noqa: E402
from spice_ev.strategies.oemof_solve import OemofSolve      # noqa: E402


def titel(nr, text):
    print(f"\n{'=' * 78}\n{nr}  {text}\n{'=' * 78}")


def tabelle(df, zeilen=None):
    """Print a frame without the index, optionally only the first rows."""
    kopf = df if zeilen is None else df.head(zeilen)
    print(kopf.to_string(index=False))
    if zeilen is not None and len(df) > zeilen:
        print(f"... {len(df) - zeilen} weitere Zeilen, insgesamt {len(df)}")


def reihe(name, werte):
    """One line per array: length, range, first values - arrays are too long to print."""
    a = np.asarray(werte)
    try:
        z = a.astype(float)
        spanne = f"min {z.min():>9.4f}   mittel {z.mean():>9.4f}   max {z.max():>9.4f}"
    except (TypeError, ValueError):
        spanne = f"Werte: {sorted({str(x) for x in a})}"
    print(f"  {name:<20s} len {len(a):>4d}   {spanne}")


def lade(beispiel):
    pfad = Path(__file__).resolve().parent / beispiel / "scenario.json"
    if not pfad.exists():
        raise SystemExit(f"{pfad} gibt es nicht - waehle einen der Beispielordner")
    sc = Scenario(json.loads(pfad.read_text(encoding="utf-8")), pfad.parent)
    return sc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("beispiel", nargs="?", default="01_household_baseline")
    p.add_argument("--vehicle", default=None, help="which vehicle to show in detail")
    p.add_argument("--zeilen", type=int, default=6, help="rows per long table")
    args = p.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    sc = lade(args.beispiel)
    # No oemof_* key is read here: the prices come from the scenario's own signals, and
    # the rest of the defaults do not change any of the tables shown below.
    s = OemofSolve(sc.components, sc.start_time, events=sc.events,
                   interval=sc.interval, stop_time=sc.stop_time, oemof_config={})

    print(f"Beispiel: {args.beispiel}")
    print(f"Zeitraum: {sc.start_time} bis {sc.stop_time}   Intervall: {sc.interval}")

    # ---------------------------------------------------------------- prepare_inputs()
    f = s.prepare_inputs()
    ti = s.time_index
    fz = args.vehicle or f["vehicles"]["vehicle_id"].iloc[0]

    titel("1a", "vehicle_events - every scenario event as one row")
    tabelle(f["vehicle_events"], args.zeilen)

    titel("1b", "vehicles - master data, one row per vehicle")
    tabelle(f["vehicles"])

    titel("2", "trips - each departure paired with its arrival, soc_delta -> kWh")
    tabelle(f["trips"], args.zeilen)

    titel("3", "state_segments - the week cut into parked/driving, gapless")
    tabelle(f["state_segments"], args.zeilen)

    titel("4a", f"time_index - the regular grid: {len(ti)} steps of {sc.interval}")
    print(f"  von {ti[0]}  bis {ti[-1]}")

    titel("4b", f"per_vehicle_ts['{fz}'] - the segments sampled onto that grid")
    ts = f["per_vehicle_ts"][fz]
    print(ts.head(args.zeilen).to_string())

    titel("4c", "where a segment boundary falls INSIDE a step")
    # A departure at 08:47 does not sit on the grid: the step it falls into stays parked
    # (the vehicle can still charge in it), the next one is the first driving step.
    ab = f["trips"].query("vehicle_id == @fz")["departure_time"].iloc[0]
    lts = f["long_ts"]
    # NOTE: trips/state_segments/time_index carry the scenario's timezone, per_vehicle_ts
    # and long_ts do not (_map_segments_to_timeseries drops it). Harmless downstream,
    # because every series is read positionally - but a comparison needs the same kind.
    grenze = ab.tz_localize(None)
    fenster = lts[(lts.vehicle_id == fz)
                  & (lts.timestamp >= grenze - pd.Timedelta(minutes=45))
                  & (lts.timestamp <= grenze + pd.Timedelta(minutes=45))]
    print(f"  erste Abfahrt: {ab}")
    tabelle(fenster[["timestamp", "state", "is_driving", "is_parked", "energy_kwh",
                     "connected_charging_station"]])

    # ------------------------------------------------------------ build_oemof_inputs()
    oi = s.build_oemof_inputs()

    titel("5", "build_oemof_inputs() - what the model is built from")
    for k, v in oi.items():
        art = type(v).__name__
        inhalt = f"keys {list(v)}" if isinstance(v, dict) else (
            f"{len(v)} Zeitpunkte" if hasattr(v, "__len__") else "")
        print(f"  {k:<18s} {art:<15s} {inhalt}")

    titel("6", f"vehicle_params['{fz}'] - scalars, and three series over all steps")
    vp = oi["vehicle_params"][fz]
    for k, v in vp.items():
        if isinstance(v, (np.ndarray, list)):
            reihe(k, v)
        else:
            print(f"  {k:<20s} {v!r}")

    print("\n  min_soc_series is the only requirement the trips put on the LP:")
    ms = pd.Series(vp["min_soc_series"], index=ti)
    print(f"  it is {ms.min()} everywhere except in {int((ms > ms.min()).sum())} steps, "
          f"where it is {ms.max()} - the last parked step before each departure:")
    tabelle(pd.DataFrame({"Schritt": ms[ms > ms.min()].index,
                          "min_soc": ms[ms > ms.min()].values}))

    print("\n  consumption books each trip in ONE step, the one it arrives in:")
    verbrauch = pd.Series(vp["consumption"], index=ti)
    tabelle(pd.DataFrame({"Schritt": verbrauch[verbrauch > 0].index,
                          "kWh": verbrauch[verbrauch > 0].values.round(6)}))
    print(f"  Summe {verbrauch.sum():.6f} kWh = Summe der Trips "
          f"{f['trips'].query('vehicle_id == @fz')['energy_kwh'].sum():.6f} kWh")

    titel("7", "grid_connectors - load, PV and the scenario's price per step")
    for gcid, gc in oi["grid_connectors"].items():
        print(f"  {gcid}:")
        for k, v in gc.items():
            if isinstance(v, np.ndarray):
                reihe(k, v)
            else:
                print(f"  {k:<20s} {v!r}")

    titel("8", "battery_params and charging_stations")
    for k in ("battery_params", "charging_stations"):
        print(f"  {k}:")
        for name, d in oi[k].items():
            print(f"    {name}: {d}")


if __name__ == "__main__":
    main()

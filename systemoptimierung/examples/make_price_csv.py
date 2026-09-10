"""Write the shared exchange-price series into every example folder.

Why this exists: the three ``input_preis.csv`` used to be hand-made artefacts that nothing
in the repository could reproduce - and one of them carried a flat +20 ct markup baked into
every row, which made that example incomparable to the others. The exchange series now
lives in ONE file, ``input_preis_boerse.csv``, and this script copies it into each example.
The markup is a model parameter (``oemof_consumer_type``, see ``spice_ev/oemof_model.py``),
so the CSVs stay pure exchange prices.

Each example needs its own copy because ``generate.py`` stores the bare file name in
scenario.json and spice_ev resolves it relative to that file (``spice_ev/events.py:194``).

Run from anywhere:

    python systemoptimierung/examples/make_price_csv.py
"""
import csv
import pathlib
import sys

HIER = pathlib.Path(__file__).parent
QUELLE = HIER / "input_preis_boerse.csv"
ZIELE = ["01_household_baseline", "02_commercial_fleet", "03_household_v2g"]
SPALTE = "preis_ct_kwh"


def lese(pfad):
    """Read the source series; returns (rows, values). Fails loudly on a bad file."""
    with open(pfad, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{pfad} is empty")
    if SPALTE not in rows[0]:
        sys.exit(f"{pfad} has no column {SPALTE!r} (found {list(rows[0])})")
    werte = [float(r[SPALTE]) for r in rows]
    if min(werte) > 20.0:
        sys.exit(f"{pfad} looks like a retail series (min {min(werte):.2f} ct) - the source "
                 f"must be the pure exchange price; the markup is a model parameter")
    return rows, werte


def schreibe(pfad, rows):
    with open(pfad, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["time", SPALTE])
        w.writerows([[r["time"], r[SPALTE]] for r in rows])


def main():
    if not QUELLE.exists():
        sys.exit(f"missing source series: {QUELLE}")
    rows, werte = lese(QUELLE)
    print(f"{QUELLE.name}: {len(werte)} rows, {min(werte):.2f} .. {max(werte):.2f} ct/kWh, "
          f"mean {sum(werte) / len(werte):.2f}")
    for name in ZIELE:
        ziel = HIER / name / "input_preis.csv"
        if not ziel.parent.is_dir():
            print(f"  {name}: folder missing, skipped")
            continue
        schreibe(ziel, rows)
        print(f"  {name}/input_preis.csv written")


if __name__ == "__main__":
    main()

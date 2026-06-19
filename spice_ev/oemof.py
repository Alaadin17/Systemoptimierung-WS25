"""
oemof Energy System Model: PV + home battery + N electric vehicles.

The topology is derived from the spice_ev scenario and built CONDITIONALLY
depending on the available components (details: _create_components).

Buses
-----
- bus_home            home AC bus (always): grid supply, household load, wallboxes.
- bus_pv              PV bus (only if PV generation is present): feeds into the
                      home via converter_pv_to_home, feed-in via excess.
- bus_battery         DC bus of the stationary home battery (only if present in
                      the scenario), attached to bus_home via link_home_battery.
- bus_mobility_<vid>  one DC bus per vehicle (BEV storage + wallbox).

Components
----------
- grid_supply              grid supply at bus_home (nominal power = GC max_power).
- household_demand         household load (fixed timeseries) at bus_home.
- pv / excess_electricity  PV source resp. grid feed-in (only at bus_pv).
- home_battery             stationary storage (capacity/power from the scenario).
- per vehicle <vid>:
    wallbox_charge_<vid>     bus_home -> bus_mobility (only when at home),
                             nominal power = max_power of the charging station.
    wallbox_discharge_<vid>  bus_mobility -> bus_home (V2H, only with v2g).
    bev_battery_<vid>        BEV storage; driving demand as fixed_losses_absolute,
                             time-dependent minimum SOC (desired_soc before departure).

Properties
----------
- The BEV can NOT feed into the grid (excess only at bus_pv, BEV at bus_mobility).
- Grid and PV can charge the BEV; the BEV can supply the home via V2H.

Inputs (__init__): SystemConfig (efficiencies/costs/solver), timeseries_df
(PV_kW/Load_kW), time_index, vehicle_params (per vehicle), grid_power,
battery_params. In the production path fed by
spice_ev.strategies.oemof_solve.OemofSolve.

Result: get_wallbox_schedule() returns the AC wallbox power per vehicle and
time step (charging/V2H) back to the spice_ev simulation.
"""

###########################################################################
# Imports
###########################################################################
import logging
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple
import warnings


import numpy as np
import pandas as pd
from oemof.solph import (
    EnergySystem,
    Model,
    buses,
    components as cmp,
    flows,
    processing,
)
from oemof.tools import logger
from pyomo.opt import SolverStatus, TerminationCondition


###########################################################################
# Configuration (formerly spice_ev/oemof_config_loader.py)
###########################################################################
def _coerce_value(value: Any, target_type: Any) -> Any:
    """Cast an (possibly already typed) value to the field type of SystemConfig.

    Values from the cfg parser are usually already typed (json.loads). Strings
    like ``freq=15min`` or ``start_date=2025-01-01`` stay strings.
    """
    if target_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is list:
        if isinstance(value, list):
            return value
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if target_type is str:
        return str(value)
    return value


def _as_array(values, n):
    # Bring values (series/array/scalar) to length n: pad with 0 or truncate.
    arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    if len(arr) < n:
        arr = np.concatenate([arr, np.zeros(n - len(arr))])
    return arr[:n]


@dataclass
class SystemConfig:
    """Configuration parameters for the energy system."""

    # Time parameters
    start_date: str = "2025-01-01"
    periods: int = 96  # 15-minute steps (96 = 1 day for debug)
    freq: str = "15min"

    # Required (global) columns in the timeseries file
    required_columns: list = field(
        default_factory=lambda: ["PV_kW", "Load_kW", "BEV_at_home", "consumption"]
    )

    # System parameters
    grid_supply_power_kW: float = 30.0

    # Converter parameters
    converter_pv_to_home_power_kW: float = 10
    # (home battery converter power is derived from the scenario, see
    #  battery_params; therefore no converter_*_to_battery_power_kW needed anymore)

    # Converter efficiencies
    # (converter_pv_to_home uses a fixed 1.0 -> no dedicated field)
    converter_home_to_battery_efficiency: float = 0.96
    converter_battery_to_home_efficiency: float = 0.96

    # Variable costs for converters (cent/kWh)
    converter_pv_to_home_variable_costs: float = 0.0

    # Stationary battery storage
    battery_capacity_kWh: float = 10.2
    battery_min_soc: float = 0.1
    battery_max_soc: float = 1.0
    battery_initial_soc: float = 0.5
    battery_efficiency: float = 1.0
    battery_max_power_kW: float = 10.0

    # BEV parameters (default/fallback; overridden per vehicle from the master data)
    bev_capacity_kWh: float = 77.0
    bev_min_soc: float = 0.2
    bev_max_soc: float = 0.95
    bev_initial_soc: float = 0.95

    # Wallbox parameters
    wallbox_power_kW: float = 11.0
    wallbox_efficiency_charge: float = 0.97
    wallbox_efficiency_discharge: float = 0.97
    enable_v2h: bool = True

    # Costs
    pv_variable_costs: float = 0.0
    grid_variable_costs: float = 35.0  # grid supply (ct/kWh), fixed
    grid_feedin_tariff: float = -8.0  # feed-in (ct/kWh), negative = revenue, fixed

    # Solver
    solver: str = "cbc"
    solver_verbose: bool = False
    debug: bool = True
    solver_threads: int = 8
    solver_ratio_gap: float = 0.01

    # Result storage
    should_dump_results: bool = True
    dump_filename: str = "dump"

    # Logging
    log_filename: str = "oemof_case00.log"
    log_screen_level: int = logging.INFO
    log_file_level: int = logging.INFO

    @classmethod
    def from_options(cls, mapping: Optional[Mapping[str, Any]]) -> "SystemConfig":
        """Build a SystemConfig from a flat dict.

        Keys may be prefixed with ``oemof_`` (e.g.
        ``oemof_battery_capacity_kWh``) or already name the plain field.
        Unknown keys are ignored (warning).
        """
        config = cls()
        if not mapping:
            return config

        field_types = {f.name: f.type for f in fields(cls)}
        for raw_key, value in mapping.items():
            name = raw_key[len("oemof_"):] if raw_key.startswith("oemof_") else raw_key
            if name not in field_types:
                logging.warning("Unknown oemof parameter is ignored: %s", raw_key)
                continue
            setattr(config, name, _coerce_value(value, field_types[name]))
        return config


###########################################################################
# Helper functions
###########################################################################

def setup_logging(config: SystemConfig) -> None:
    """
    Configure the logging system

    Parameters:
    -----------
    config : SystemConfig
        Configuration object with logging parameters
    """
    logger.define_logging(
        logfile=config.log_filename,
        screen_level=config.log_screen_level,
        file_level=config.log_file_level,
    )


def load_timeseries(config: SystemConfig,
    input_file: Optional[Path] = None
) -> Tuple[pd.DataFrame, Path]:
    """
    Load the timeseries data from CSV

    Parameters:
    -----------
    input_file : Path, optional
        Path to the input file. If None, the default path is used.

    Returns:
    --------
    df_timeseries : pd.DataFrame
        DataFrame with all timeseries
    timeseries_path : Path
        Path to the loaded file
    """
    if input_file is None:
        script_dir = Path(__file__).resolve().parent
        input_file = script_dir / "Input_timeseries" / "input_timeseries.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Timeseries file not found: {input_file}")

    logging.info(f"Loading timeseries from: {input_file}")
    df_timeseries = pd.read_csv(input_file, delimiter=",")

    # Validate required columns
    required_columns = config.required_columns
    missing_cols = set(required_columns) - set(df_timeseries.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in the timeseries file: {missing_cols}")

    return df_timeseries, input_file


def validate_and_clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the timeseries data

    Parameters:
    -----------
    df : pd.DataFrame
        Raw data

    Returns:
    --------
    df_clean : pd.DataFrame
        Cleaned data
    """
    df_clean = df.copy()

    # PV values must be >= 0 (negative values -> 0)
    if "PV_kW" in df_clean:
        df_clean["PV_kW"] = df_clean["PV_kW"].clip(lower=0)

    # Check for NaN values (only existing columns)
    known_cols = [c for c in ("PV_kW", "Load_kW", "BEV_at_home", "consumption")
                  if c in df_clean]
    nan_counts = df_clean[known_cols].isna().sum()
    if nan_counts.any():
        logging.warning(f"NaN values found:\n{nan_counts[nan_counts > 0]}")
        df_clean = df_clean.fillna(0)

    # Validate BEV_at_home (should be binary) – only if globally present
    if "BEV_at_home" in df_clean and not df_clean["BEV_at_home"].isin([0, 1]).all():
        logging.warning("BEV_at_home contains non-binary values. Rounding to 0/1.")
        df_clean["BEV_at_home"] = df_clean["BEV_at_home"].round().astype(int)

    logging.info(f"Timeseries validated: {len(df_clean)} time steps")
    if "PV_kW" in df_clean:
        logging.info(f"  PV: {df_clean['PV_kW'].min():.2f} - {df_clean['PV_kW'].max():.2f} kW")
    if "Load_kW" in df_clean:
        logging.info(f"  Load: {df_clean['Load_kW'].min():.2f} - {df_clean['Load_kW'].max():.2f} kW")
    if "consumption" in df_clean:
        logging.info(f"  BEV consumption: {df_clean['consumption'].sum():.2f} kWh total")

    return df_clean


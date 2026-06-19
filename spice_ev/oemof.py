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


###########################################################################
# Main class
###########################################################################
class EnergySystemModel:
    """
    Models and optimizes an energy system with PV and battery storage (2-bus architecture)
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        timeseries_file: Optional[Path] = None,
        timeseries_df: Optional[pd.DataFrame] = None,
        time_index: Optional[pd.DatetimeIndex] = None,
        vehicle_params: Optional[Dict[str, Dict[str, Any]]] = None,
        grid_power: Optional[float] = None,
        battery_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the energy system model

        Parameters:
        -----------
        config : SystemConfig, optional
            Configuration. If None, the default is used.
        timeseries_file : Path, optional
            Path to the timeseries CSV. Only used when ``timeseries_df`` is None.
        timeseries_df : pd.DataFrame, optional
            Already prepared (global) timeseries with at least ``PV_kW``/``Load_kW``.
            Takes precedence over ``timeseries_file`` (no CSV detour).
        time_index : pd.DatetimeIndex, optional
            Shared time grid. If set, it is used directly
            (instead of being built from ``start_date``/``periods``/``freq``).
        vehicle_params : dict, optional
            Per vehicle: ``{capacity_kWh, min_soc, max_soc, initial_soc, v2g,
            at_home (series 0/1), consumption (series kWh/step),
            wallbox_power_kW (optional)}``. If empty, a default vehicle is built
            from ``config`` + the columns ``BEV_at_home``/``consumption``.
        grid_power : float, optional
            Grid connection power (kW). If None,
            ``config.grid_supply_power_kW`` is used. Usually derived from the
            scenario (``grid_connectors[gc].max_power``).
        """
        self.config = config or SystemConfig()
        self.timeseries_file = timeseries_file
        self._timeseries_df_input = timeseries_df
        self._time_index_input = time_index
        self.vehicle_params: Dict[str, Dict[str, Any]] = dict(vehicle_params or {})
        self.grid_power = grid_power
        # Stationary home batteries from the scenario: {battery_id: infos}
        # (empty dict = no battery)
        self.battery_params: Dict[str, Dict[str, Any]] = dict(battery_params or {})

        # System variables
        self.time_index: Optional[pd.DatetimeIndex] = None
        self.es: Optional[EnergySystem] = None
        self.model: Optional[Model] = None
        self.df_timeseries: Optional[pd.DataFrame] = None
        self._results_main = None  # processing.results(model) for the extraction
        # Node references per vehicle for the result extraction
        self._b_home = None
        self._vehicle_nodes: Dict[str, Dict[str, Any]] = {}

        # Setup
        setup_logging(self.config)
        logging.info("=" * 80)
        logging.info("Initializing Energy System Model (3-bus architecture)")
        logging.info("=" * 80)

    def run(self) -> None:
        """Run the full modeling and optimization"""
        try:
            self._load_data()
            self._create_time_index()
            self._create_energy_system()
            self._create_components()
            self._optimize()
            self._solve()
            self._extract_results()
            self._save_results()

            logging.info("=" * 80)
            logging.info("[SUCCESS] Simulation completed successfully")
            logging.info("=" * 80)

        except Exception as e:
            logging.error(f"❌ Error during the simulation: {e}")
            raise

    def _load_data(self) -> None:
        """Load and validate the input data"""
        if self._timeseries_df_input is not None:
            logging.info("Step 1: Use the provided DataFrame (no CSV)")
            self.df_timeseries = validate_and_clean_timeseries(self._timeseries_df_input)
        else:
            logging.info("Step 1: Load timeseries data from CSV")
            df_raw, _ = load_timeseries(self.config, self.timeseries_file)
            self.df_timeseries = validate_and_clean_timeseries(df_raw)

        # If a time grid is provided, it determines the number of periods
        if self._time_index_input is not None:
            self.config.periods = len(pd.DatetimeIndex(self._time_index_input))

        # Check the length of the timeseries
        if len(self.df_timeseries) < self.config.periods:
            logging.warning(
                f"Timeseries too short! Available: {len(self.df_timeseries)}, "
                f"Required: {self.config.periods}. Truncating simulation."
            )
            self.config.periods = len(self.df_timeseries)

    def _create_time_index(self) -> None:
        """Create the time index for the simulation"""
        logging.info("Step 2: Create time index")
        if self._time_index_input is not None:
            # Adopt the shared grid with spice_ev directly
            self.time_index = pd.DatetimeIndex(self._time_index_input)
            self.config.periods = len(self.time_index)
        else:
            self.time_index = pd.date_range(
                start=self.config.start_date,
                periods=self.config.periods,
                freq=self.config.freq
            )
        logging.info(
            f"  Time range: {self.time_index[0]} to {self.time_index[-1]}"
        )
        logging.info(f"  Number of time steps: {self.config.periods}")

    def _create_energy_system(self) -> None:
        """Create the oemof EnergySystem object"""
        logging.info("Step 3: Create Energy System")
        self.es = EnergySystem(
            timeindex=self.time_index,
            infer_last_interval=True
        )
        logging.info(f"  EnergySystem created with {len(self.time_index)} time steps")

    def _create_components(self) -> None:
        """Orchestrate the model construction based on the decision variables.

        - bus_home, grid supply and household load are always created.
        - PV (Bus_PV + plant + feed-in + converter) only if PV generation exists.
        - Stationary home battery (Bus_Battery + Link + storage) only if present.
        - Per vehicle a dedicated mobility part (Bus + wallbox + BEV storage).
        """
        logging.info("Step 4: Create components")
        periods = self.config.periods

        # Global timeseries (PV/load) + optional standalone default vehicle
        pv_timeseries, load_timeseries = self._extract_global_timeseries(periods)
        if not self.vehicle_params and "BEV_at_home" in self.df_timeseries:
            self._build_default_vehicle_params(periods)

        loss_factor = self._loss_factor()

        # ===== DECISION VARIABLES: what gets built? =====
        pv_present = float(pv_timeseries.sum()) > 0.0
        ev_present = bool(self.vehicle_params)
        battery_present = bool(self.battery_params)

        # ===== Always: Bus_Home + grid + household load =====
        b_home = buses.Bus(label="bus_home")
        self.es.add(b_home)
        self._b_home = b_home  # for the result extraction
        self._add_grid_and_demand(b_home, load_timeseries)

        # 1. PV present -> Bus_PV + plant + feed-in + converter
        if pv_present:
            self._add_pv(b_home, pv_timeseries)
        else:
            logging.info("  - No PV generation: no Bus_PV / no feed-in")

        # 3. Stationary battery/batteries -> per battery Bus_Battery + Link + storage
        if battery_present:
            for bid, bp in self.battery_params.items():
                self._add_battery(bid, bp, b_home)
        else:
            logging.info("  - No stationary battery in the scenario")

        # 2. EVs present -> per vehicle a mobility part
        if not ev_present:
            logging.warning("  - No vehicles defined: no mobility/BEV part")
        for vid, params in self.vehicle_params.items():
            self._add_vehicle(vid, params, b_home, loss_factor, periods)

        logging.info(f"  [OK] {len(self.es.nodes)} components created")

    # ------------------------------------------------------------------
    # Helper methods for _create_components
    # ------------------------------------------------------------------
    def _extract_global_timeseries(self, periods):
        """Global PV/load as series; if missing, fill with 0."""
        df = self.df_timeseries
        pv = df["PV_kW"].iloc[:periods] if "PV_kW" in df else pd.Series(0.0, index=range(periods))
        load = df["Load_kW"].iloc[:periods] if "Load_kW" in df else pd.Series(0.0, index=range(periods))
        return pv, load

    def _loss_factor(self):
        """kWh/step -> kW (oemof multiplies fixed_losses_absolute by the step duration)."""
        if len(self.time_index) > 1:
            step_hours = (self.time_index[1] - self.time_index[0]) / pd.Timedelta(hours=1)
        else:
            step_hours = 0.25
        return 1.0 / step_hours

    def _build_default_vehicle_params(self, periods):
        """Standalone fallback: a default BEV from config + global columns."""
        df = self.df_timeseries
        consumption = df.get("consumption", pd.Series(0.0, index=df.index))
        self.vehicle_params = {
            "bev": {
                "capacity_kWh": self.config.bev_capacity_kWh,
                "min_soc": self.config.bev_min_soc,
                "max_soc": self.config.bev_max_soc,
                "initial_soc": self.config.bev_initial_soc,
                "v2g": self.config.enable_v2h,
                "at_home": df["BEV_at_home"].iloc[:periods].to_numpy(),
                "consumption": consumption.iloc[:periods].to_numpy(),
            }
        }

    def _add_grid_and_demand(self, b_home, load_timeseries):
        """Grid supply + household load at bus_home (always)."""
        grid_power_kW = (self.grid_power if self.grid_power is not None
                         else self.config.grid_supply_power_kW)
        logging.info(f"  - Grid connection ({grid_power_kW} kW) + household load")
        self.es.add(cmp.Source(
            label="grid_supply",
            outputs={b_home: flows.Flow(
                variable_costs=self.config.grid_variable_costs,
                nominal_value=grid_power_kW)},
        ))
        self.es.add(cmp.Sink(
            label="household_demand",
            inputs={b_home: flows.Flow(fix=load_timeseries, nominal_value=1)},
        ))

    def _add_pv(self, b_home, pv_timeseries):
        """[PV] Bus_PV + PV plant + grid feed-in + converter PV->Home."""
        logging.info("  - [PV] Bus_PV + PV plant + feed-in + converter")
        b_pv = buses.Bus(label="bus_pv")
        self.es.add(b_pv)
        self.es.add(cmp.Source(
            label="pv",
            outputs={b_pv: flows.Flow(
                fix=pv_timeseries, nominal_value=1,
                variable_costs=self.config.pv_variable_costs)},
        ))
        # Grid feed-in ONLY from Bus_PV (BEV/home battery cannot feed in)
        self.es.add(cmp.Sink(
            label="excess_electricity",
            inputs={b_pv: flows.Flow(variable_costs=self.config.grid_feedin_tariff)},
        ))
        self.es.add(cmp.Converter(
            label="converter_pv_to_home",
            inputs={b_pv: flows.Flow(
                nominal_value=self.config.converter_pv_to_home_power_kW,
                variable_costs=self.config.converter_pv_to_home_variable_costs)},
            outputs={b_home: flows.Flow()},
            conversion_factors={b_home: 1.0},
        ))
        return b_pv

    def _add_battery(self, bid, bp, b_home):
        """[Battery] Bus_Battery + DC/AC link + home storage for one battery <bid>.

        Unique labels per battery (`*_<bid>`), sizing from `bp`
        (scenario). The efficiency comes from the scenario and is split evenly
        across both link directions (√efficiency per direction ⇒
        forth × back = efficiency). The storage itself is lossless (1.0),
        so that the losses are not counted twice.
        """
        capacity = float(bp.get("capacity_kWh", self.config.battery_capacity_kWh))
        power = float(bp.get("power_kW", self.config.battery_max_power_kW))
        discharge_power = float(bp.get("discharge_power_kW", power))
        init = float(bp.get("initial_soc", self.config.battery_initial_soc))
        init = min(max(init, self.config.battery_min_soc), self.config.battery_max_soc)
        efficiency = float(bp.get("efficiency", self.config.battery_efficiency))
        eff_dir = efficiency ** 0.5  # half the losses per direction

        logging.info(f"  - [Battery '{bid}'] {capacity:.1f} kWh, "
                     f"charge {power:.1f} / discharge {discharge_power:.1f} kW, "
                     f"η={efficiency:.2f} (→ {eff_dir:.3f} per direction)")

        b_battery = buses.Bus(label=f"bus_battery_{bid}")
        self.es.add(b_battery)
        # Ideal DC/AC converter between bus_home and bus_battery
        self.es.add(cmp.Link(
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
                (b_battery, b_home): eff_dir,   # discharge into the home
                (b_home, b_battery): eff_dir,   # charge from the home
            },
        ))
        self.es.add(cmp.GenericStorage(
            label=f"home_battery_{bid}",
            inputs={b_battery: flows.Flow(nominal_value=power)},
            outputs={b_battery: flows.Flow(nominal_value=discharge_power)},
            nominal_storage_capacity=capacity,
            min_storage_level=self.config.battery_min_soc,
            max_storage_level=self.config.battery_max_soc,
            initial_storage_level=init,
            inflow_conversion_factor=1.0,   # losses are in the link (see above)
            outflow_conversion_factor=1.0,
            loss_rate=0.0,
            balanced=False,
        ))
        return b_battery

    def _add_vehicle(self, vid, params, b_home, loss_factor, periods):
        """[EV] Bus_Mobility + wallbox(+V2H) + BEV storage for one vehicle."""
        at_home = _as_array(params.get("at_home", 1.0), periods)
        consumption = _as_array(params.get("consumption", 0.0), periods)
        capacity = float(params.get("capacity_kWh", self.config.bev_capacity_kWh))
        min_soc = float(params.get("min_soc", self.config.bev_min_soc))
        max_soc = float(params.get("max_soc", self.config.bev_max_soc))
        init_soc = min(max(float(params.get("initial_soc", self.config.bev_initial_soc)),
                           min_soc), max_soc)
        v2g = bool(params.get("v2g", self.config.enable_v2h))
        wallbox_power = float(params.get("wallbox_power_kW", self.config.wallbox_power_kW))

        # Time-dependent minimum SOC (desired_soc before departures); periods+1 support points
        min_series = params.get("min_soc_series")
        if min_series is not None:
            min_level = np.concatenate([_as_array(min_series, periods), [min_soc]])
        else:
            min_level = min_soc

        logging.info(f"  - [EV] '{vid}': {capacity:.1f} kWh, "
                     f"SOC [{min_soc:.2f}, {max_soc:.2f}], v2g={v2g}, "
                     f"wallbox {wallbox_power:.0f} kW")

        b_mobility = buses.Bus(label=f"bus_mobility_{vid}")
        self.es.add(b_mobility)

        # Wallbox charging: bus_home -> bus_mobility (only when at home)
        charge_node = cmp.Converter(
            label=f"wallbox_charge_{vid}",
            inputs={b_home: flows.Flow()},
            outputs={b_mobility: flows.Flow(max=at_home, nominal_value=wallbox_power)},
            conversion_factors={b_mobility: self.config.wallbox_efficiency_charge},
        )
        self.es.add(charge_node)

        # V2H discharging: bus_mobility -> bus_home (only when globally active AND v2g)
        discharge_node = None
        if self.config.enable_v2h and v2g:
            discharge_node = cmp.Converter(
                label=f"wallbox_discharge_{vid}",
                inputs={b_mobility: flows.Flow()},
                outputs={b_home: flows.Flow(max=at_home, nominal_value=wallbox_power)},
                conversion_factors={b_home: self.config.wallbox_efficiency_discharge},
            )
            self.es.add(discharge_node)

        # BEV battery (driving demand as fixed absolute losses)
        battery_node = cmp.GenericStorage(
            label=f"bev_battery_{vid}",
            inputs={b_mobility: flows.Flow()},
            outputs={b_mobility: flows.Flow()},
            nominal_storage_capacity=capacity,
            min_storage_level=min_level,
            max_storage_level=max_soc,
            initial_storage_level=init_soc,
            loss_rate=0.0,
            fixed_losses_absolute=consumption * loss_factor,
            balanced=False,
        )
        self.es.add(battery_node)

        self._vehicle_nodes[vid] = {
            "bus": b_mobility,
            "charge": charge_node,
            "discharge": discharge_node,
            "battery": battery_node,
        }

    def _optimize(self) -> None:
        """Create the optimization model"""
        logging.info("Step 5: Create optimization model")
        self.model = Model(self.es)

        # Optional: write LP file for debugging
        if self.config.debug:
            # Determine the save path
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parents[2]
            lp_path = project_root / "results" / "oemof-V2H-WS25" / "dumps" / "lp_files"/ f"{self.config.dump_filename}_debug.lp"
            logging.info(f"  Debug mode: writing LP file to {lp_path}")
            lp_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.write(str(lp_path), io_options={"symbolic_solver_labels": True})

    def _solve(self) -> None:
        """Solve the optimization model"""
        logging.info("Step 6: Solve optimization problem")
        logging.info(f"  Solver: {self.config.solver}")

        # Solver options
        solver_options = {}
        if self.config.solver == "cbc":
            solver_options = {
                "threads": self.config.solver_threads,
                "ratioGap": self.config.solver_ratio_gap,
            }

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)


            results = self.model.solve(
                solver=self.config.solver,
                solve_kwargs={"tee": self.config.solver_verbose},
                cmdline_options=solver_options

            )

            # Check solver status
            status = results.solver.status
            termination = results.solver.termination_condition

            if status != SolverStatus.ok or termination != TerminationCondition.optimal:
                error_msg = (
                    f"\n❌ Optimization failed!\n"
                    f"  Status: {status}\n"
                    f"  Termination: {termination}\n"
                    f"  Message: {results.solver.message}\n"
                )
                logging.error(error_msg)
                raise RuntimeError(error_msg)

            logging.info("  [OK] Optimization successful")
            logging.info(f"  Objective Value: {self.model.objective():.2f} €")

    def _extract_results(self) -> None:
        """Extract the results from the solved model"""
        logging.info("Step 7: Extract results")
        self._results_main = processing.results(self.model)
        self.es.results["main"] = self._results_main
        self.es.results["meta"] = processing.meta_results(self.model)
        logging.info("  [OK] Results extracted")

    @staticmethod
    def _edge_flow(results, src, dst, n) -> np.ndarray:
        """Read the flow sequence of the edge (src -> dst) from the oemof results."""
        entry = results.get((src, dst))
        if entry is None:
            return np.zeros(n)
        seq = entry.get("sequences")
        if seq is None or "flow" not in seq:
            return np.zeros(n)
        vals = np.asarray(seq["flow"].to_numpy(), dtype=float)
        if len(vals) < n:
            vals = np.concatenate([vals, np.zeros(n - len(vals))])
        return vals[:n]

    def get_wallbox_schedule(self) -> Dict[str, pd.DataFrame]:
        """Wallbox power (AC side at bus_home) per vehicle and time step.

        Returns:
            dict[vehicle_id] -> DataFrame(index=time_index) with columns:
            ``charge_kW`` (home -> wallbox, charging >= 0),
            ``discharge_kW`` (wallbox -> home, V2H >= 0),
            ``net_kW`` (= charge - discharge; > 0 = net draw at the GC).
        """
        if self._results_main is None:
            raise RuntimeError("No results available – run() must be called first.")
        results = self._results_main
        n = len(self.time_index)
        schedule: Dict[str, pd.DataFrame] = {}
        for vid, nodes in self._vehicle_nodes.items():
            charge = self._edge_flow(results, self._b_home, nodes["charge"], n)
            if nodes["discharge"] is not None:
                discharge = self._edge_flow(results, nodes["discharge"], self._b_home, n)
            else:
                discharge = np.zeros(n)
            schedule[vid] = pd.DataFrame(
                {
                    "charge_kW": charge,
                    "discharge_kW": discharge,
                    "net_kW": charge - discharge,
                },
                index=self.time_index,
            )
        return schedule

    def _save_results(self) -> None:
        """Save the results as a dump"""
        if not self.config.should_dump_results:
            logging.info("Step 8: Result storage skipped (disabled)")
            return

        logging.info("Step 8: Save results")

        # Determine the save path
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parents[2]
        dump_path = project_root / "results" / "oemof-V2H-WS25" / "dumps"
        dump_path.mkdir(parents=True, exist_ok=True)

        try:
            self.es.dump(
                dpath=str(dump_path),
                filename=self.config.dump_filename
            )
            logging.info(f"  [OK] Dump saved: {dump_path / self.config.dump_filename}")
        except Exception as e:
            logging.error(f"  ❌ Error while saving: {e}")
            raise


###########################################################################
# Main program (standalone smoke test with synthetic data)
###########################################################################
def main():
    """Standalone demo: 2 vehicles, 1 day (96 x 15min), synthetic PV/load.

    Serves as a smoke test of the model without spice_ev/CSV. In the production
    path ``EnergySystemModel`` is called from the strategy ``OemofSolve`` with
    ``timeseries_df``, ``time_index`` and ``vehicle_params``.
    """
    config = SystemConfig()
    config.periods = 96
    config.debug = False
    config.should_dump_results = False
    idx = pd.date_range(start=config.start_date, periods=config.periods, freq=config.freq)

    hours = idx.hour + idx.minute / 60.0
    pv = np.clip(np.sin((hours - 6.0) / 12.0 * np.pi), 0, None) * 8.0  # PV bell curve
    load = np.full(config.periods, 0.5)  # constant base load
    df = pd.DataFrame({"PV_kW": pv, "Load_kW": load}, index=idx)

    # Vehicle 1: away during the day (~08:00–14:00), 10 kWh trip on arrival
    at_home_1 = np.ones(config.periods)
    at_home_1[32:56] = 0
    cons_1 = np.zeros(config.periods)
    cons_1[55] = 10.0
    # Vehicle 2: at home continuously, no V2G
    vehicle_params = {
        "veh_1": {"capacity_kWh": 77.0, "min_soc": 0.2, "max_soc": 0.95,
                  "initial_soc": 0.5, "v2g": True,
                  "at_home": at_home_1, "consumption": cons_1},
        "veh_2": {"capacity_kWh": 58.0, "min_soc": 0.2, "max_soc": 0.9,
                  "initial_soc": 0.6, "v2g": False,
                  "at_home": np.ones(config.periods),
                  "consumption": np.zeros(config.periods)},
    }

    model = EnergySystemModel(
        config=config, timeseries_df=df, time_index=idx, vehicle_params=vehicle_params
    )
    model.run()
    for vid, sched in model.get_wallbox_schedule().items():
        print(f"{vid}: total charging = {sched['charge_kW'].sum():.1f} kWh-eq, "
              f"V2H total = {sched['discharge_kW'].sum():.1f} kWh-eq")


if __name__ == "__main__":
    main()

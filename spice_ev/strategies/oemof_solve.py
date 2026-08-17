'''
----------------- OemofSolve strategy -----------------------

Autor: Alaa Alsleman, GitHub: Alaadin17

Goal: derive the charging strategy from an oemof optimization instead of a heuristic.
The whole horizon is optimized ONCE (open loop) and the resulting plan is then applied
step by step by the spice_ev simulation.

Inputs (from the kwargs that scenario.py passes in):
 - self.events       (Events object: vehicle events, fixed_load / local_generation lists)
 - self.world_state  (Vehicles, Charging Stations, Grid Connectors, Batteries)
 - self.oemof_config (flat dict of the oemof_* keys from simulate.cfg, prefix stripped;
                      turned into a SystemConfig via SystemConfig.from_options)
 - self.interval     (datetime.timedelta of one simulation step)
 - self.stop_time    (end of the simulation)

Flow:
 step() -> _ensure_solved() -> prepare_inputs() -> build_oemof_inputs()
        -> run_oemof_model() [EnergySystemModel.run()] -> commands_from_oemof()
'''


import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from spice_ev import events
from spice_ev.strategy import Strategy
from spice_ev.util import get_cost


class OemofSolve(Strategy):
    """Charging strategy that follows a plan optimized with oemof.

    Prepares the inputs from the spice_ev scenario, builds and solves the oemof model once
    over the full horizon (``_ensure_solved``) and caches the resulting per-vehicle plan in
    ``self._schedule``. ``step()`` then applies that plan step by step.
    """

    def __init__(self, components, start_time, **kwargs):
        super().__init__(components, start_time, **kwargs)
        self.description = "oemof_solve"
        
        # Inputs from kwargs
        self.events = kwargs.get("events")
        self.cfg = kwargs.get("cfg")
        # Flat dict with oemof_* parameters (from simulate.cfg, prefix removed)
        self.oemof_config = kwargs.get("oemof_config", {}) or {}
        self.vehicles = self.world_state.vehicles
        self.interval = kwargs.get("interval")
        self.stop_time = kwargs.get("stop_time")
        self.start_time = start_time
        self._price_sheet = None   # lazy-loaded JSON (price sheet, see below)

        # Output containers, populated later by prepare_inputs()
        self.time_index = None
        self.input_frames = {}
        self._prepared = False

        # Open-loop state: the optimization runs ONCE, then the cached plan is applied.
        self._solved = False
        self._model = None
        # The full plan, grouped by component type — filled ONCE by commands_from_oemof():
        #   {"vehicles":  {vid:  [(charge_kW, discharge_kW, soc_end), ...]}, applied by step()
        #    "batteries": {bid:  [(charge_kW, discharge_kW, soc_end), ...]}, applied by step()
        #    "grid":      {gcid: [(supply_kW, feedin_kW),            ...]}}  verification only
        # One tuple per simulation step. step() applies soc_end (the SOC the optimization
        # reaches at the END of that step); the powers are for reporting and verification.
        self._plan: Dict[str, Dict[str, list]] = {}
        # Alias to self._plan["vehicles"], kept for existing references.
        self._schedule: Dict[str, list] = {}
        # Index of the CURRENT simulation step = position in the lists above; step() reads
        # self._plan[<type>][<id>][self._oemof_step] and increments it afterwards.
        self._oemof_step = 0

        # Optionally switch off spice_ev's minimum charging power (see the method's docstring).
        # Done here, so it is guaranteed to happen before the first simulation step. The
        # config object is kept — step() needs it for the V2G discharge floor.
        from spice_ev.oemof_model import SystemConfig
        self._oemof_cfg = SystemConfig.from_options(self.oemof_config)
        self._apply_min_power_override(self._oemof_cfg)

        # Path to spice_ev's price sheet (source of the PV feed-in remuneration and the
        # retail markup). It arrives as the oemof_cost_parameters_file cfg key, so that
        # spice_ev's own scripts need no modification; a kwargs value still wins if some
        # caller supplies one directly. Empty -> the config values stay the fallback.
        self.cost_parameters_file = (kwargs.get("cost_parameters_file")
                                     or self._oemof_cfg.cost_parameters_file or None)

    def _apply_min_power_override(self, config) -> None:
        """Optionally drop spice_ev's minimum charging power to zero.

        ``clamp_power`` (spice_ev/util.py) sets any charging power below ``cs.min_power`` or
        ``vehicle_type.min_charging_power`` to ZERO. Other strategies call it to turn a
        computed power into a command; the oemof model does not know that rule (it would
        need binary variables) and plans such small powers anyway — e.g. to use a little PV
        surplus. That energy then silently never reaches the battery.

        NOTE: since ``step()`` applies the planned SOC (``Battery.load(target_soc=...)``)
        instead of a commanded power, it does NOT call ``clamp_power`` — so on this path the
        flag currently has no effect. It is kept because it belongs to the scenario, not to
        the strategy: it stays correct if a power-based command path is ever reintroduced.

        With ``config.ignore_min_charging_power`` both limits are set to 0.
        ``min_charging_power`` is an optional VehicleType field defaulting to 0.0 anyway, so
        this is a regular value.

        Stationary batteries (``Battery.min_charging_power``) are deliberately NOT touched:
        they are not driven by the oemof charging plan.

        Args:
            config: SystemConfig carrying the ``ignore_min_charging_power`` flag.
        """
        if not getattr(config, "ignore_min_charging_power", False):
            return
        # VehicleType objects are shared between vehicles of the same type -> covers all types
        for vehicle in self.world_state.vehicles.values():
            vehicle.vehicle_type.min_charging_power = 0.0
        for cs in self.world_state.charging_stations.values():
            cs.min_power = 0.0


###########################################################################
########## 1) Preprocessing: raw data -> structured DataFrames #########
############################################################################

    def prepare_inputs(self) -> Dict[str, Any]:
        """Run the full preprocessing pipeline and store the result frames.

        Builds the trip table, the parked/driving state segments and finally the
        per-vehicle timeseries on the simulation time grid. Results are stored in
        ``self.input_frames`` and ``self.time_index``.

        Returns:
            Dict[str, Any]: the populated ``self.input_frames``.
        """
        if self.events is None or self.world_state is None:
            raise ValueError("OemofSolve requires events and world_state to prepare inputs")

        vehicle_events = self.events.vehicle_events
        vehicles = self.world_state.vehicles

        # 1) Master data / raw events as DataFrames
        df_vehicle_events = self._build_vehicle_events_df(vehicle_events)
        df_vehicles = self._build_world_state_vehicles_df(vehicles)

        # 2) Trips (departure/arrival pairs) per vehicle
        trip_df = self._build_trip_df(vehicle_events, vehicles)
        trip_df_by_vehicle = self._group_trips_by_vehicle(trip_df)

        # 3) State segments (parked/driving) + energy from trips
        state_segments_df = self._build_state_segments(
            vehicle_events, self.start_time, self.stop_time, vehicles)
        state_segments_df = self._map_trips_to_state_segments(
            trip_df_by_vehicle, state_segments_df)

        # 4) Time grid + mapping onto timeseries
        self.time_index = self._build_time_index(
            self.start_time, self.stop_time, self.interval)
        per_vehicle_ts, long_ts = self._map_segments_to_timeseries(
            state_segments_df, self.time_index, self.interval)

        self.input_frames = {
            "vehicle_events": df_vehicle_events,
            "vehicles": df_vehicles,
            "trips": trip_df,
            "state_segments": state_segments_df,
            "per_vehicle_ts": per_vehicle_ts,
            "long_ts": long_ts,
        }
        self._prepared = True
        return self.input_frames


    def _build_vehicle_events_df(self, vehicle_events) -> pd.DataFrame:
        """Build a vehicle_events dataframe.
        args:
            vehicle_events (List[VehicleEvent]): List of VehicleEvent objects.
        
        returns:
                pd.DataFrame: DataFrame with vehicle events and update fields.
                Dataframe with columns: 
                                        - signal_time, 
                                        - start_time, 
                                        - vehicle_id, 
                                        - event_type, 
                                        - update_*
        """

        rows = []
        for ev in vehicle_events:
            row = {
                "signal_time": ev.signal_time,
                "start_time": ev.start_time,
                "vehicle_id": ev.vehicle_id,
                "event_type": ev.event_type,
            }
            for k, v in ev.update.items():
                row[f"update_{k}"] = v
            rows.append(row)

        df_vehicle_events = (
            pd.DataFrame(rows)
            .sort_values(["start_time", "vehicle_id"])
            .reset_index(drop=True)
        )

        return df_vehicle_events


    def _build_world_state_vehicles_df(self, world_state_vehicles) -> pd.DataFrame:
        '''Build a DataFrame from a collection of WorldStateVehicles.
    
        args:
            world_state_vehicles (Dict[str, WorldStateVehicle]): Dictionary of WorldStateVehicle objects.

        returns:
            pd.DataFrame: DataFrame with the WorldStateVehicle data.
                DataFrame with columns: 
                                        - vehicle_id, 
                                        - vehicle_type, 
                                        - capacity_kwh, 
                                        - min_charging_power, 
                                        - v2g, 
                                        - discharge_limit, 
                                        - connected_charging_station, 
                                        - desired_soc, 
                                        - soc
        '''
        vehicle_rows = []
        for vid, v in world_state_vehicles.items():
            vt = v.vehicle_type
            vehicle_rows.append({
                "vehicle_id": vid,
                "vehicle_type": vt.name,
                "capacity_kwh": v.battery.capacity,
                "min_charging_power": vt.min_charging_power,
                "v2g": vt.v2g,
                "discharge_limit": vt.discharge_limit,
                "connected_charging_station": v.connected_charging_station,
                "desired_soc": v.desired_soc,
                "soc": v.battery.soc,
            })

        df_vehicles = (
            pd.DataFrame(vehicle_rows)
            .sort_values("vehicle_id")
            .reset_index(drop=True)
        )
        return df_vehicles


    def _build_trip_df(self, vehicle_events, vehicles) -> pd.DataFrame:
        """Build a trip table from departure/arrival events.

        Each trip pairs the last departure with the next arrival for a vehicle and
        computes energy_kwh from soc_delta * battery capacity.

        Args:
            vehicle_events: List of VehicleEvent objects.
            vehicles: Dict[vehicle_id, Vehicle] to access battery capacity.

        Returns:
            DataFrame with columns: 
                                    -vehicle_id, 
                                    -departure_time, 
                                    -arrival_time, 
                                    -soc_delta, 
                                    -energy_kwh.
        """
        # Build a trip table from departure/arrival events per vehicle.
        rows = []
        last_departure = {}
        ordered = sorted(
            [e for e in vehicle_events if e.event_type in ("departure", "arrival")],
            key=lambda e: (e.vehicle_id, e.start_time),
        )

        # ev: vehicle event, ordered by vehicle and time
        for ev in ordered:
            if ev.event_type == "departure":
                # Remember the latest departure time for this vehicle.
                last_departure[ev.vehicle_id] = ev.start_time
            elif ev.event_type == "arrival":
                # Pair arrival with the last departure and compute energy usage.
                dep_time = last_departure.get(ev.vehicle_id)
                soc_delta = ev.update.get("soc_delta")
                cap = vehicles[ev.vehicle_id].battery.capacity
                # soc_delta is negative for driving; convert to positive energy in kWh.
                energy_kwh = None if soc_delta is None else -soc_delta * cap
                rows.append({
                    "vehicle_id": ev.vehicle_id,
                    "departure_time": dep_time,
                    "arrival_time": ev.start_time,
                    "soc_delta": soc_delta,
                    "energy_kwh": energy_kwh,
                })
        return pd.DataFrame(rows)


    def _group_trips_by_vehicle(self, trip_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group trips by vehicle_id.

        Args:
            trip_df: DataFrame with trip data including vehicle_id.
        
        Returns:
            Dict mapping vehicle_id to its corresponding trip DataFrame.
        """
        trip_df_by_vehicle = {}
        for vid, group in trip_df.groupby("vehicle_id"):
            trip_df_by_vehicle[vid] = group.sort_values("departure_time").reset_index(drop=True)
        return trip_df_by_vehicle


    def _build_state_segments(self, vehicle_events, start_time, stop_time, vehicles=None) -> pd.DataFrame:
        """Build contiguous state segments (driving/parked) per vehicle.

        For each vehicle, departure and arrival events define state changes.
        The first segment starts at start_time; the last ends at stop_time.
        Each parked segment also carries the connected_charging_station it is plugged
        into (from the arrival event that starts it; the first segment uses the
        vehicle's initial connected_charging_station). Driving segments -> None.

        Args:
            vehicle_events: List of VehicleEvent objects.
            start_time: Scenario start time (datetime).
            stop_time: Scenario stop time (datetime).
            vehicles: Dict[vehicle_id, Vehicle] for the initial connected_charging_station.

        Returns:
            DataFrame with columns:
                                    -vehicle_id,
                                    -start_time,
                                    -end_time,
                                    -state,
                                    -connected_charging_station,
                                    -desired_soc (SOC the vehicle must reach before it
                                     leaves again; seeded from the vehicle and updated by
                                     every arrival event).
        """
        vehicles = vehicles or {}
        rows = []
        # include vehicles without any events (always parked) too
        vehicle_ids = sorted(set(ev.vehicle_id for ev in vehicle_events) | set(vehicles))
        for vid in vehicle_ids:
            v_events = [
                ev for ev in vehicle_events
                if ev.vehicle_id == vid and ev.event_type in ("departure", "arrival")
            ]
            v_events = sorted(v_events, key=lambda e: e.start_time)
            state = "parked"  # Assume parked at the start until we see a departure
            cs = getattr(vehicles.get(vid), "connected_charging_station", None)
            # SOC the vehicle must reach before it leaves again (spice_ev: desired_soc).
            # Set on the vehicle initially and updated by every arrival event.
            desired = getattr(vehicles.get(vid), "desired_soc", None)
            cursor = start_time
            for ev in v_events:
                rows.append({
                    "vehicle_id": vid,
                    "start_time": cursor,
                    "end_time": ev.start_time,
                    "state": state,
                    "connected_charging_station": cs,
                    "desired_soc": desired,
                })
                if ev.event_type == "departure":
                    state = "driving"
                    cs = None  # not plugged in while driving
                elif ev.event_type == "arrival":
                    state = "parked"
                    cs = ev.update.get("connected_charging_station")
                    desired = ev.update.get("desired_soc", desired)
                cursor = ev.start_time
            rows.append({
                "vehicle_id": vid,
                "start_time": cursor,
                "end_time": stop_time,
                "state": state,
                "connected_charging_station": cs,
                "desired_soc": desired,
            })
        return pd.DataFrame(rows)


    def _build_time_index(self, start_time, stop_time, interval) -> pd.DatetimeIndex:

        """Build a time index from start_time to stop_time with given interval.

        The grid starts at ``start_time`` (not start+interval), so that
        row ``k`` corresponds exactly to spice_ev step ``k`` (current_time =
        start+k*interval). Otherwise the charging commands fed back into
        spice_ev would be shifted by one time step.

        Args:
            start_time: Scenario start time (datetime).
            stop_time: Scenario stop time (datetime).
            interval: Length of one simulation step as a datetime.timedelta
                (spice_ev passes ``datetime.timedelta(minutes=scenario['interval'])``).
        Returns:
            DatetimeIndex format: DatetimeIndex(['2024-01-01 00:00:00', '2024-01-01 00:15:00', ...])
        """
        return pd.date_range(
            start=start_time, end=stop_time - pd.Timedelta(interval), freq=interval)


    def _map_trips_to_state_segments(self, trip_df_by_vehicle: Dict[str, pd.DataFrame], state_segments_df: pd.DataFrame) -> pd.DataFrame:
         
        '''
            Map trips to state segments.
            For each segment in state_segments_df, check whether there is an overlapping trip in trip_df_by_vehicle.
            If so, the trip energy in kWh is written into the segment's "energy_kwh" column. Otherwise "energy_kwh" stays None.

            Args:
                trip_df_by_vehicle: Dict[vehicle_id, DataFrame] with trips per vehicle (columns: departure_time, arrival_time, energy_kwh)
                state_segments_df: DataFrame from _build_state_segments, i.e. columns
                    [vehicle_id, start_time, end_time, state, connected_charging_station,
                     desired_soc]
            Returns:
                DataFrame with columns:
                                        -vehicle_id,
                                        -start_time,
                                        -end_time,
                                        -state,
                                        -energy_kwh,
                                        -connected_charging_station (passed through),
                                        -desired_soc (passed through)
        '''
        rows = []
        for _, segment in state_segments_df.iterrows():
            vid = segment["vehicle_id"]
            start = segment["start_time"]
            end = segment["end_time"]
            state = segment["state"]

            trip_df = trip_df_by_vehicle.get(vid)
            energy_kwh = None
            if trip_df is not None:
                for _, trip in trip_df.iterrows():
                    dep_time = trip["departure_time"]
                    arr_time = trip["arrival_time"]
                    if dep_time < end and arr_time > start:
                        energy_kwh = trip["energy_kwh"]
                        break

            rows.append({
                "vehicle_id": vid,
                "start_time": start,
                "end_time": end,
                "state": state,
                "energy_kwh": energy_kwh,
                "connected_charging_station": segment["connected_charging_station"],
                "desired_soc": segment["desired_soc"],
                })
        return pd.DataFrame(rows)


    def _map_segments_to_timeseries(
        self, state_segments_df: pd.DataFrame, time_index: pd.DatetimeIndex, interval
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
            
            """Build per-vehicle timeseries on a fixed time grid.

            Each row in state_segments_df represents a continuous segment [start_time, end_time)
            with a state and optional energy_kwh. A time bin belongs to the segment iff its own
            start lies in [start_time, end_time) — i.e. the assignment is by bin START, not by
            overlap. Returns both per-vehicle tables and a long-format table.

            Args:
                state_segments_df: DataFrame with columns [vehicle_id, start_time, end_time,
                    state, energy_kwh, connected_charging_station, desired_soc]; the last two
                    are mapped onto the grid as well.
                time_index: Iterable of timestamps defining bin starts.
                interval: Bin width as Timedelta or a pandas-compatible frequency string (e.g. "15min").

            Returns:
                Tuple (per_vehicle, long_df):
                    per_vehicle: dict[vehicle_id, DataFrame] indexed by time_index.
                    long_df: DataFrame with columns:
                                                    - timestamp, 
                                                    - vehicle_id, 
                                                    - state, 
                                                    - unterwegs, 
                                                    - zuhause, 
                                                    - energy_kwh
            """
            # Normalize the time index and interval to comparable types.
            time_index = pd.DatetimeIndex(time_index)
            if time_index.tz is not None:
                # Remove timezone to avoid tz-aware vs tz-naive comparisons.
                time_index = time_index.tz_localize(None)
            interval = pd.Timedelta(interval)

            per_vehicle = {}
            long_rows = []

            # Process each vehicle separately to create a per-vehicle timeseries.
            # vid: vehicle_id, segs: all segments for this vehicle
            # segs: DataFrame with columns [start_time, end_time, state, energy_kwh] for this vehicle
            # Iterator over pairs (key = value of the vehicle_id column, group_df = the rows with that vehicle_id)
            for vid, segs in state_segments_df.groupby("vehicle_id"):
                print(f"Mapping segments to timeseries for vehicle {vid} with {len(segs)} segments.")
                # Initialize per-vehicle dataframe with default values.
                df = pd.DataFrame(index=time_index)
                df["state"] = "parked"  # Default state; will be overwritten by segments
                df["energy_kwh"] = 0  # Default energy; will be overwritten by segments
                df["connected_charging_station"] = None  # Default: not plugged in
                df["desired_soc"] = np.nan  # SOC required before the next departure

                # Apply each segment to all overlapping time bins.
                # _: index of the segment (ignored), seg: the segment row with start_time, end_time, state, energy_kwh
                for _, seg in segs.iterrows():
                    start = pd.to_datetime(seg["start_time"])

                    end = pd.to_datetime(seg["end_time"])

                    if start.tzinfo is not None:
                        start = start.tz_localize(None)
                    if end.tzinfo is not None:
                        end = end.tz_localize(None)

                    # bin ts belongs to the segment iff start <= ts < end
                    left = time_index.searchsorted(start, side="left")
                    right = time_index.searchsorted(end, side="left")


                    if left < right:
                        ts_slice = time_index[left:right]
                        df.loc[ts_slice, "state"] = seg["state"]
                        df.loc[ts_slice, "energy_kwh"] = 0
                        df.loc[ts_slice[-1], "energy_kwh"] = seg["energy_kwh"]
                        df.loc[ts_slice, "connected_charging_station"] = seg["connected_charging_station"]
                        if seg["desired_soc"] is not None and not pd.isna(seg["desired_soc"]):
                            df.loc[ts_slice, "desired_soc"] = float(seg["desired_soc"])

                # Convenience boolean columns for quick filtering/plotting.
                df["unterwegs"] = df["state"].eq("driving")
                df["zuhause"] = df["state"].eq("parked")
                df["vehicle_id"] = vid

                # Store per-vehicle and also build a long-format table.
                per_vehicle[vid] = df
                long_rows.append(df.reset_index().rename(columns={"index": "timestamp"}))

            # Concatenate all vehicles into one long table (timestamp, vehicle_id, ...).
            if long_rows:
                long_df = pd.concat(long_rows, ignore_index=True)
                long_df = long_df[["timestamp", "vehicle_id", "state", "unterwegs", "zuhause", "energy_kwh", "connected_charging_station", "desired_soc"]]
            else:
                long_df = pd.DataFrame(
                    columns=["timestamp", "vehicle_id", "state", "unterwegs", "zuhause", "energy_kwh", "connected_charging_station", "desired_soc"]
                )
            print(f"Mapped segments to timeseries for {len(per_vehicle)} vehicles, resulting 1) dictionary with {len(per_vehicle)} vehicles with {len(per_vehicle)} dataframes with {len(per_vehicle[vid])} rows and 2) complete table with {len(long_df)} rows.")
            # Return both representations: per-vehicle dict and long-format table.
            return per_vehicle, long_df
        

    # ------------------------------------------------------------------
    # Bridge spice_ev -> oemof
    # ------------------------------------------------------------------
    def _sample_event_list(self, ev_list, time_index: pd.DatetimeIndex) -> pd.Series:
        """
        Sample an EnergyValuesList (step function) onto the time grid.

        Args:
                ev_list: EnergyValuesList with values and step_duration_s.
                time_index: target time index for the output (DatetimeIndex).

        Returns:
                pd.Series with index=time_index, values from ev_list (piecewise constant)
                and 0 outside the ev_list times.

        We need it when the scenario contains PV or Load
        (include_local_generation_csv / include_fixed_load_csv in the generate.cfg).

        Example:
                If the PV plant has 15min steps, but the simulation runs with 1h steps,
                we need to sample the PV values onto the 1h grid.

        Important:
                Direction matters.

                Coarse -> fine, for example 1h -> 15min:
                Forward fill is correct for power values because the value is held constant
                over the smaller time steps.

                Fine -> coarse, for example 15min -> 1h:
                Forward fill is not sufficient because it would only pick one value.
                Instead, the mean value over the target interval is used.

                Example:
                15min values [1, 2, 3, 4] sampled to 1h should become 2.5,
                not 1.0.
        """

        values = list(getattr(ev_list, "values", []) or [])
        if not values:
            return pd.Series(0.0, index=time_index)

        delta = pd.Timedelta(seconds=ev_list.step_duration_s)
        raw_index = pd.date_range(start=ev_list.start_time, periods=len(values), freq=delta)
        factor = getattr(ev_list, "factor", 1) or 1
        raw = pd.Series(np.asarray(values, dtype=float) * factor, index=raw_index)

        # Unify time zones (tz-naive) so that reindex works
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)

        target = time_index
        if target.tz is not None:
            target = target.tz_localize(None)

        # Determine target step size
        if len(target) >= 2:
            target_delta = target[1] - target[0]
        else:
            target_delta = delta

        raw_start = raw.index[0]
        raw_end = raw.index[-1] + delta

        # Case 1: coarse -> fine or same resolution
        # Example: source 1h, target 15min
        if delta >= target_delta:
            aligned = raw.reindex(target, method="ffill").fillna(0.0)

            # Set values outside the raw time range to 0
            valid_mask = (target >= raw_start) & (target < raw_end)
            aligned.loc[~valid_mask] = 0.0

        # Case 2: fine -> coarse
        # Example: source 15min, target 1h
        else:
            aligned_values = []

            for t_start in target:
                t_end = t_start + target_delta

                # Select raw values whose time interval lies inside the target interval
                mask = (raw.index >= t_start) & (raw.index < t_end)
                interval_values = raw.loc[mask]

                if not interval_values.empty:
                    aligned_values.append(float(interval_values.mean()))
                else:
                    aligned_values.append(0.0)

            aligned = pd.Series(aligned_values, index=target)

        aligned.index = time_index
        return aligned

    def _aggregate_event_lists(self, event_lists, time_index: pd.DatetimeIndex) -> pd.Series:
        """
        Sum several EnergyValuesLists (e.g. multiple PV plants) onto the grid.
        
        Args:
            event_lists: Dict[plant_id, EnergyValuesList] with values and step_duration_s.
            time_index: target time index for the output (DatetimeIndex).
        
        Returns:
            pd.Series with index=time_index, values = sum of all event_lists (piecewise constant) and 0 outside the event_list times.

        example: if the scenario contains multiple PV plants, we need to sum their outputs onto the simulation time grid.
        """
        total = pd.Series(0.0, index=time_index)
        for ev_list in (event_lists or {}).values():
            total = total.add(self._sample_event_list(ev_list, time_index), fill_value=0.0)
        return total

    def _grid_power(self) -> Optional[float]:
        """
        Sum of the grid connection powers (max_power) of the grid connectors.
        Args:
                world_state.grid_connectors: Dict[connector_id, GridConnector] with possible max_power attributes.
        Returns:
                - Float: the sum of max_power across all grid connectors, 
                - None: if no max_power is defined.
        """
        powers = [gc.max_power for gc in self.world_state.grid_connectors.values()
                  if getattr(gc, "max_power", None)]
        return float(sum(powers)) if powers else None

    def _grid_connectors(self, time_index) -> Dict[str, Dict[str, Any]]:
        """Per grid connector: max_power + its OWN household load and PV timeseries.

        The oemof model builds one bus (Home_N) + source + feed-in sink per active GC;
        each GC also carries its own load and PV, grouped by the events'
        ``grid_connector_id``. Returns {gcid: {"max_power"(optional), "load", "pv"}}.
        """
        fixed = getattr(self.events, "fixed_load_lists", {})
        gen = getattr(self.events, "local_generation_lists", {})
        result: Dict[str, Dict[str, Any]] = {}
        for gcid, gc in self.world_state.grid_connectors.items():
            info: Dict[str, Any] = {
                "load": self._aggregate_event_lists_for_gc(fixed, gcid, time_index),
                "pv": self._aggregate_event_lists_for_gc(gen, gcid, time_index),
            }
            mp = getattr(gc, "max_power", None)
            if mp:
                info["max_power"] = float(mp)
            # values sourced from the scenario / price sheet (config stays the fallback):
            price = self._grid_price_series(gcid, time_index)
            if price is not None:
                markup = self._retail_markup_ct(gcid)
                if markup is not None:
                    # retail price like the spice_ev cost calculation: net components
                    # summed, VAT on top of everything (feed-in stays net there too)
                    net_markup, vat_percent = markup
                    price = (price + net_markup) * (1.0 + vat_percent / 100.0)
                info["price_ct_kWh"] = price          # time-varying grid price
            tariff = self._feedin_tariff_ct(gcid)
            if tariff is not None:
                info["feedin_tariff_ct_kWh"] = tariff  # PV feed-in remuneration (negative)
            hb = self._homebus_feedin_tariff_ct(gcid)
            if hb is not None:
                info["homebus_feedin_tariff_ct_kWh"] = hb  # battery/V2G export (0 by sheet)
            kwp = self._pv_kwp(gcid)
            if kwp > 0:
                info["pv_power_kW"] = kwp              # PV plant size -> converter limit
            result[gcid] = info
        return result

    def _aggregate_event_lists_for_gc(self, event_lists, gcid, time_index):
        """Sum only the EnergyValuesLists whose grid_connector_id == gcid (as np array)."""
        total = pd.Series(0.0, index=time_index)
        for ev_list in (event_lists or {}).values():
            if getattr(ev_list, "grid_connector_id", None) == gcid:
                total = total.add(self._sample_event_list(ev_list, time_index), fill_value=0.0)
        return total.to_numpy()

    def _grid_price_series(self, gcid, time_index) -> Optional[np.ndarray]:
        """Time-varying grid price for one GC from the scenario's grid operator signals.

        spice_ev carries prices as GridOperatorSignal events (a ``cost`` dict per GC,
        evaluated with ``util.get_cost`` — the same mechanism every other strategy uses).
        The LP receives them as a per-step array in ct/kWh (signals are EUR/kWh -> x100),
        piecewise constant from each signal's ``start_time``. Returns None if the scenario
        has no priced signals for this GC (-> the config value stays as fallback).
        """
        # Fester Preis statt Szenario-Signalen: eine konstante Reihe zurueckgeben, damit
        # alles danach (Retail-Aufschlag, MwSt) unveraendert weiterlaeuft. Wuerden wir
        # stattdessen None liefern, griffe im Modell zwar auch grid_variable_costs - aber
        # OHNE Aufschlag, und fest und variabel waeren nicht mehr vergleichbar.
        cfg = getattr(self, "_oemof_cfg", None)
        if cfg is not None and not getattr(cfg, "grid_price_from_scenario", True):
            return np.full(len(time_index), float(cfg.grid_variable_costs))

        sigs = [s for s in getattr(self.events, "grid_operator_signals", []) or []
                if getattr(s, "grid_connector_id", None) == gcid
                and getattr(s, "cost", None)]
        if not sigs:
            return None
        pairs = []
        for s in sorted(sigs, key=lambda s: s.start_time):
            t = pd.Timestamp(s.start_time)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            pairs.append((t, float(get_cost(1, s.cost)) * 100.0))   # EUR/kWh -> ct/kWh
        target = time_index
        if target.tz is not None:
            target = target.tz_localize(None)
        starts = pd.DatetimeIndex([p[0] for p in pairs])
        values = np.array([p[1] for p in pairs])
        # NEGATIVE prices are clipped to 0 for the LP: a linear model cannot forbid
        # "disposing" of energy (storage in+out cycling burns it), so being PAID to buy
        # becomes a money pump — buy, burn, get paid to re-buy. Real households cannot
        # destroy energy for profit. At 0 ct the LP still charges everything useful, only
        # the destruction premium is gone. (Exact modelling would need binary variables.)
        values = np.maximum(values, 0.0)
        idx = np.searchsorted(starts, target, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)   # before the first signal: first value
        return values[idx]

    def tariff(self) -> str:
        """Der gewaehlte Tarif, normalisiert auf "RLM" / "SLP" / "fixed".

        Ein unbekannter Wert faellt NICHT still auf einen Zweig zurueck, sondern warnt und
        nimmt den Default - sonst rechnet man unbemerkt mit einem anderen Tarif, als in der
        cfg steht.
        """
        wert = str(getattr(getattr(self, "_oemof_cfg", None), "tariff", "RLM")).strip()
        for gueltig in ("RLM", "SLP", "fixed"):
            if wert.lower() == gueltig.lower():
                return gueltig
        logging.warning("oemof_tariff = '%s' ist unbekannt (RLM | SLP | fixed) - "
                        "es wird RLM gerechnet", wert)
        return "RLM"

    def _retail_markup_ct(self, gcid) -> Optional[Tuple[float, float]]:
        """Fixed per-kWh retail components + VAT rate from the price sheet.

        Mirrors spice_ev's cost calculation (costs.py) so both worlds price identically:
        grid fee commodity charge by tariff (SLP flat net price; RLM by the GC's
        voltage_level in the <2500 h/a bracket — the same edge-condition constant costs.py
        uses), plus all levies, the concession fee and the electricity tax. All values are
        NET; VAT is applied by the caller on (spot + markup), exactly like costs.py applies
        it to the total while leaving the feed-in remuneration untaxed.

        Returns (markup_net_ct_per_kWh, vat_percent), or None when the tariff is "fixed"
        (then grid_variable_costs IS the price), no price sheet is configured or the sheet
        lacks the entries (-> spot price only).
        """
        cfg = getattr(self, "_oemof_cfg", None)
        if cfg is None or self.tariff() == "fixed":
            return None
        if not self.cost_parameters_file:
            return None
        if self._price_sheet is None:
            with open(self.cost_parameters_file, encoding="utf-8") as f:
                self._price_sheet = json.load(f)
        gc = self.world_state.grid_connectors.get(gcid)
        operator = getattr(gc, "grid_operator", "default_grid_operator") or "default_grid_operator"
        try:
            sheet = self._price_sheet[operator]
            if self.tariff() == "SLP":
                commodity = float(sheet["grid_fee"]["SLP"]["commodity_charge_ct/kWh"]["net_price"])
            else:   # RLM: by voltage level, <2500 h/a utilization bracket
                voltage = getattr(gc, "voltage_level", None) or "MV"
                commodity = float(
                    sheet["grid_fee"]["RLM"]["<2500_h/a"]["commodity_charge_ct/kWh"][voltage])
            levies = sum(v for v in sheet["levies"].values() if isinstance(v, (int, float)))
            concession = float(sheet["concession_fee"]["charge"])
            electricity_tax = float(sheet["taxes"]["tax_on_electricity"])
            vat_percent = float(sheet["taxes"]["value_added_tax"])
        except (KeyError, TypeError):
            return None
        return commodity + float(levies) + concession + electricity_tax, vat_percent

    def _pv_kwp(self, gcid) -> float:
        """Installed PV nominal power (kWp) at one grid connector, summed over its plants."""
        return sum(float(pv.nominal_power)
                   for pv in getattr(self.world_state, "photovoltaics", {}).values()
                   if getattr(pv, "parent", None) == gcid)

    def _feedin_tariff_ct(self, gcid) -> Optional[float]:
        """PV feed-in tariff for one GC from spice_ev's price sheet (negative = revenue).

        Reads the SAME price sheet the spice_ev cost calculation uses
        (``cost_parameters_file`` in simulate.cfg): ``feed-in_remuneration.PV`` maps plant
        size steps (kWp) to a remuneration in ct/kWh. The step is chosen by the installed
        PV power at this GC. Returns None if no sheet is configured or the GC has no PV
        (-> the config value stays as fallback).
        """
        kwp = self._pv_kwp(gcid)
        if not self.cost_parameters_file or kwp <= 0:
            return None
        if self._price_sheet is None:
            with open(self.cost_parameters_file, encoding="utf-8") as f:
                self._price_sheet = json.load(f)
        operator = getattr(self.world_state.grid_connectors.get(gcid), "grid_operator",
                           "default_grid_operator") or "default_grid_operator"
        try:
            fee = self._price_sheet[operator]["feed-in_remuneration"]["PV"]
            steps, rems = fee["kWp"], fee["remuneration"]
        except (KeyError, TypeError):
            return None
        i = int(np.searchsorted(np.asarray(steps, dtype=float), kwp, side="left"))
        i = min(i, len(rems) - 1)
        return -float(rems[i])   # negative = revenue (model convention)

    def _homebus_feedin_tariff_ct(self, gcid) -> Optional[float]:
        """Remuneration for exports from the HOME bus (battery / V2G), from the price sheet.

        The sheet lists these separately from PV (``feed-in_remuneration.V2G`` and
        ``.battery`` — both 0 in the default sheet): re-exported or battery energy earns
        nothing, only PV does. Returns None without a sheet (-> config fallback).
        """
        if not self.cost_parameters_file:
            return None
        if self._price_sheet is None:
            with open(self.cost_parameters_file, encoding="utf-8") as f:
                self._price_sheet = json.load(f)
        operator = getattr(self.world_state.grid_connectors.get(gcid), "grid_operator",
                           "default_grid_operator") or "default_grid_operator"
        try:
            fee = self._price_sheet[operator]["feed-in_remuneration"]
            value = max(float(fee.get("V2G", 0.0)), float(fee.get("battery", 0.0)))
        except (KeyError, TypeError):
            return None
        return -value   # 0 in the default sheet -> exporting from the home bus earns nothing

    @staticmethod
    def _min_soc_series(ts, config) -> np.ndarray:
        """Per-step SOC floor for one vehicle, taken from the spice_ev scenario.

        spice_ev expects a vehicle to be charged to its ``desired_soc`` when it leaves
        (the trips are sized for that). The oemof model therefore gets a time-varying
        ``min_storage_level``: ``desired_soc`` at every DEPARTURE step (the last step the
        vehicle is still plugged in before it drives off), and the global
        ``config.bev_min_soc`` everywhere else — a constant desired_soc floor would be
        infeasible, since driving must be allowed to drain the battery below it.

        Args:
            ts: per-vehicle timeseries with columns connected_charging_station, desired_soc.
            config: SystemConfig (fallback floor ``bev_min_soc``).

        Returns:
            np.ndarray of length len(ts) with the SOC floor (0..1) per step.
        """
        base = float(config.bev_min_soc)
        connected = np.array([c is not None for c in ts["connected_charging_station"]])
        n = len(connected)
        floor = np.full(n, base, dtype=float)
        if n == 0:
            return floor

        desired = ts["desired_soc"].to_numpy(dtype=float)
        desired = np.where(np.isnan(desired), base, desired)

        # departure = plugged in now, gone in the next step
        departs = np.zeros(n, dtype=bool)
        departs[:-1] = connected[:-1] & ~connected[1:]
        floor[departs] = np.maximum(base, desired[departs])
        return floor

    def _battery_params(self, config) -> Dict[str, Dict[str, Any]]:
        """Read ALL stationary batteries from the scenario.

        Args:
            config: SystemConfig with fallback values (power/SOC/efficiency).

        Returns:
            Dict[battery_id, infos] – empty dict if there is no (valid) battery.
            infos per battery:
              capacity_kWh, power_kW (charging), discharge_power_kW (discharging),
              initial_soc, efficiency, min_power_kW, loss_rate (dict), parent (GC).
        """
        result: Dict[str, Dict[str, Any]] = {}
        for bid, bat in getattr(self.world_state, "batteries", {}).items():
            capacity = float(getattr(bat, "capacity", 0) or 0)
            # <=0 or unlimited (StationaryBattery sets 2**64) -> skip
            if capacity <= 0 or capacity > 1e9:
                continue
            try:
                power = float(bat.loading_curve.max_power)
            except Exception:
                power = config.battery_max_power_kW
            try:
                discharge_power = float(bat.unloading_curve.max_power)
            except Exception:
                discharge_power = power  # no dedicated discharge curve -> same as charging
            result[bid] = {
                "capacity_kWh": capacity,
                "power_kW": power,
                "discharge_power_kW": discharge_power,
                "initial_soc": float(getattr(bat, "soc", config.battery_initial_soc)),
                "efficiency": float(getattr(bat, "efficiency", config.battery_efficiency)),
                "min_power_kW": float(getattr(bat, "min_charging_power", 0.0) or 0.0),
                "loss_rate": dict(getattr(bat, "loss_rate", {}) or {}),
                "parent": getattr(bat, "parent", None),
            }
        return result

    def build_oemof_inputs(self) -> Dict[str, Any]:
        """Assemble every input EnergySystemModel needs, from the spice_ev scenario.

        Returns a dict with: config (SystemConfig from the oemof_* cfg keys), time_index,
        grid_connectors (per GC: max_power + its own load/pv), charging_stations (max_power
        + parent GC), vehicle_params (capacity/SOC/v2g/efficiency + consumption,
        connected_cs and min_soc_series), battery_params, plus the legacy timeseries_df and
        grid_power (kept for standalone use, not read by the model).
        """
        if not self._prepared:
            raise ValueError("Inputs must be prepared before building Oemof inputs")

        from spice_ev.oemof_model import SystemConfig

        config = SystemConfig.from_options(self.oemof_config)

        # Bring global PV/load from the events onto the time grid
        pv = self._aggregate_event_lists(
            getattr(self.events, "local_generation_lists", {}), self.time_index)
        load = self._aggregate_event_lists(
            getattr(self.events, "fixed_load_lists", {}), self.time_index)
        timeseries_df = pd.DataFrame(
            {"PV_kW": pv.to_numpy(), "Load_kW": load.to_numpy()}, index=self.time_index)

        # Vehicle master data (capacity/SOC/v2g/discharge_limit) per vehicle
        vehicles_df = self.input_frames["vehicles"].set_index("vehicle_id")
        per_vehicle_ts = self.input_frames["per_vehicle_ts"]

        vehicle_params: Dict[str, Dict[str, Any]] = {}
        for vid, ts in per_vehicle_ts.items():
            consumption = ts["energy_kwh"].fillna(0).astype(float).to_numpy()
            # Per-step charging station the vehicle is plugged into (None while driving).
            # The oemof model derives the wallbox availability/power from this.
            connected_cs = ts["connected_charging_station"].to_numpy()

            if vid in vehicles_df.index:
                row = vehicles_df.loc[vid]
                capacity = float(row["capacity_kwh"])
                init_soc = float(row["soc"])
                v2g = bool(row["v2g"])
                discharge_limit = (float(row["discharge_limit"])
                                   if not pd.isna(row.get("discharge_limit"))
                                   else config.bev_discharge_limit)
            else:
                capacity = config.bev_capacity_kWh
                init_soc = config.bev_initial_soc
                v2g = config.enable_v2h
                discharge_limit = config.bev_discharge_limit

            # spice_ev has NO wallbox loss: the charging station only limits the power and
            # the loss happens INSIDE the battery (Battery.efficiency, default 0.95). The
            # oemof model mirrors that (storage inflow/outflow_conversion_factor), so we
            # hand over the vehicle's real battery efficiency. Any mismatch here makes the
            # planned SOC drift away from the simulated one.
            veh = self.world_state.vehicles.get(vid)
            eff = float(getattr(getattr(veh, "battery", None), "efficiency", 0.95) or 0.95)

            vehicle_params[vid] = {
                "capacity_kWh": capacity,
                "min_soc": config.bev_min_soc,
                "max_soc": config.bev_max_soc,
                "initial_soc": init_soc,
                "v2g": v2g,
                "discharge_limit": discharge_limit,
                "consumption": consumption,
                "connected_cs": connected_cs,
                "min_soc_series": self._min_soc_series(ts, config),
                "efficiency": eff,   # spice_ev Battery.efficiency -> storage in/outflow
            }

        # Charging stations (one wallbox per CS in the oemof model): power + parent GC
        charging_stations = {
            csid: {"max_power": float(cs.max_power), "parent": getattr(cs, "parent", None)}
            for csid, cs in self.world_state.charging_stations.items()
        }

        return {
            "config": config,
            "timeseries_df": timeseries_df,
            "time_index": self.time_index,
            "vehicle_params": vehicle_params,
            "grid_power": self._grid_power(),
            "grid_connectors": self._grid_connectors(self.time_index),
            "battery_params": self._battery_params(config),
            "charging_stations": charging_stations,
        }

############################################################################
############################ Oemof Model ###################################
############################################################################

    def run_oemof_model(self, oemof_inputs: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Build the oemof model, solve it (full horizon) and return the full plan.

        The returned dict comes from ``EnergySystemModel.get_plan()`` and is grouped by
        component type: ``vehicles`` / ``batteries`` / ``grid`` (see there).
        """
        from spice_ev.oemof_model import EnergySystemModel

        model = EnergySystemModel(
            config=oemof_inputs["config"],
            timeseries_df=oemof_inputs["timeseries_df"],
            time_index=oemof_inputs["time_index"],
            vehicle_params=oemof_inputs["vehicle_params"],
            grid_power=oemof_inputs.get("grid_power"),
            grid_connectors=oemof_inputs.get("grid_connectors"),
            battery_params=oemof_inputs.get("battery_params"),
            charging_stations=oemof_inputs.get("charging_stations"),
        )
        model.run()
        self._model = model
        return model.get_plan()

    @staticmethod
    def _frames_to_step_lists(frames: Dict[str, pd.DataFrame], *cols: str
                              ) -> Dict[str, list]:
        """{id: DataFrame} -> {id: [(col_1, col_2, ...), ...]}, one tuple per step.

        A column that the frame does not carry becomes NaN, so callers can ask for the
        optional ``soc_end`` without every producer having to supply it.
        """
        out: Dict[str, list] = {}
        for key, df in (frames or {}).items():
            series = [df[c].astype(float).tolist() if c in df.columns
                      else [float("nan")] * len(df) for c in cols]
            out[key] = list(zip(*series))
        return out

    def commands_from_oemof(self, plan: Dict[str, Dict[str, pd.DataFrame]]
                            ) -> Dict[str, Dict[str, list]]:
        """Convert the plan DataFrames into position-indexed lists per component.

        Result: ``{"vehicles": {vid: [(charge_kW, discharge_kW, soc_end), ...]},
        "batteries": {bid: [(charge_kW, discharge_kW, soc_end), ...]},
        "grid": {gcid: [(supply_kW, feedin_kW), ...]}}`` — index = simulation step, so
        ``step()`` can read the values for the current step directly via
        ``self._plan[<type>][<id>][self._oemof_step]``. The grid entry is not applied,
        it exists to verify the executed plan against the optimized one.

        ``soc_end`` is the SOC (0..1) the optimization reaches at the END of that step and
        is what ``step()`` actually applies; the two powers are kept for reporting and for
        the plan-vs-actual comparison.
        """
        plan = plan or {}
        return {
            "vehicles": self._frames_to_step_lists(
                plan.get("vehicles"), "charge_kW", "discharge_kW", "soc_end"),
            "batteries": self._frames_to_step_lists(
                plan.get("batteries"), "charge_kW", "discharge_kW", "soc_end"),
            "grid": self._frames_to_step_lists(plan.get("grid"), "supply_kW", "feedin_kW"),
        }

    def _ensure_solved(self) -> None:
        """Solve the optimization exactly once and cache the charging plan."""
        if self._solved:
            return
        self.prepare_inputs()
        oemof_inputs = self.build_oemof_inputs()
        results = self.run_oemof_model(oemof_inputs)
        self._plan = self.commands_from_oemof(results)
        self._schedule = self._plan["vehicles"]   # alias, kept for existing references
        self._solved = True
        self._oemof_step = 0

    def step(self):
        """Apply the optimized plan for the current simulation step.

        The plan is applied via the **SOC**, not via a power: for every vehicle and every
        stationary battery the optimization knows the state of charge it wants at the END
        of this step (``soc_end`` in the plan tuple), and ``step()`` simply steers the
        simulated battery to exactly that value with ``Battery.load(target_soc=...)`` /
        ``Battery.unload(target_soc=...)``. spice_ev then works out the power itself and
        returns it as ``avg_power``, which is what gets booked at the grid connector.

        Why the SOC and not the power:
        - The SOC is the ONE state both models share. Steering it makes plan and simulation
          agree by construction, and a deviation cannot accumulate: the next step targets
          the plan's absolute SOC again, so the run self-corrects instead of drifting.
        - spice_ev applies its own limits inside ``Battery.load`` (loading curve, full
          battery). Handing it a target SOC lets it do that, whereas a commanded power
          could silently be cut — which is exactly how the simulated SOC used to fall
          behind the planned one.

        Design (decided):
        - PV and household load need NO handling here — the base class books their events
          directly into ``gc.current_loads`` at the right grid connector.
        - ``distribute_surplus_power()`` and ``update_batteries()`` are deliberately NOT
          called: the plan already decides surplus usage and battery behaviour, the
          heuristics would work against it.
        - Multi-entity scaling is implicit: every charging station / battery knows its
          ``parent`` grid connector, so bookings land at the right GC for any number of
          vehicles, batteries and GCs.
        """
        # On the first call, solve the full-horizon optimization once
        if not self._solved and self.events is not None:
            self._ensure_solved()

        # Reset the per-step power of every charging station (every strategy does this;
        # it keeps cs.current_power meaningful for the reports).
        for cs in self.world_state.charging_stations.values():
            cs.current_power = 0

        idx = self._oemof_step
        commands: Dict[str, Any] = {}

        # --- 1) vehicles: steer to the SOC the plan wants at the end of this step -----
        vehicle_plan = self._plan.get("vehicles", {})
        for vid in sorted(self.world_state.vehicles):
            vehicle = self.world_state.vehicles[vid]
            cs_id = vehicle.connected_charging_station
            if cs_id is None:
                continue          # away/driving: the base class already booked soc_delta
            cs = self.world_state.charging_stations.get(cs_id)
            plan = vehicle_plan.get(vid)
            if cs is None or plan is None or idx >= len(plan):
                continue          # unknown station / no plan entry / past the horizon
            target_soc = plan[idx][2]
            if target_soc is None or target_soc != target_soc:      # NaN -> no plan value
                continue
            gc = self.world_state.grid_connectors[cs.parent]
            soc_now = vehicle.battery.soc

            if target_soc > soc_now + self.EPS:
                # charge up to the planned SOC. max_power is the station rating: the plan
                # respects it anyway, so this only guarantees the command can never exceed
                # what the charging station can physically deliver.
                avg_power = vehicle.battery.load(
                    self.interval, target_soc=target_soc, max_power=cs.max_power)["avg_power"]
                commands[cs_id] = gc.add_load(cs_id, avg_power)
                cs.current_power += avg_power
            elif target_soc < soc_now - self.EPS and vehicle.vehicle_type.v2g:
                # V2H/V2G: discharge down to the planned SOC. The plan never goes below the
                # LP's own floor (max of bev_min_soc and discharge_limit), so the target IS
                # the floor — no separate safety net needed.
                avg_power = vehicle.battery.unload(
                    self.interval, target_soc=target_soc, max_power=cs.max_power)["avg_power"]
                commands[cs_id] = gc.add_load(cs_id, -avg_power)
                cs.current_power -= avg_power

        # --- 2) stationary batteries: same, steered to the planned SOC ---------------
        battery_plan = self._plan.get("batteries", {})
        for bid in sorted(self.world_state.batteries):
            battery = self.world_state.batteries[bid]
            plan = battery_plan.get(bid)
            gc = self.world_state.grid_connectors.get(battery.parent)
            if plan is None or gc is None or idx >= len(plan):
                continue          # battery's GC was pruned in the LP / past the horizon
            target_soc = plan[idx][2]
            if target_soc is None or target_soc != target_soc:
                continue
            soc_now = battery.soc

            if target_soc > soc_now + self.EPS:
                avg_power = battery.load(self.interval, target_soc=target_soc)["avg_power"]
                gc.add_load(bid, avg_power)
            elif target_soc < soc_now - self.EPS:
                avg_power = battery.unload(self.interval, target_soc=target_soc)["avg_power"]
                gc.add_load(bid, -avg_power)

        # --- 3) PV + household load: nothing to do (events already booked at the GC) --
        # --- 4) no distribute_surplus_power / update_batteries (plan replaces them) ---

        self._oemof_step += 1
        return {"current_time": self.current_time, "commands": commands}



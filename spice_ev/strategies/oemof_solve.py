'''
----------------- OemofSolve strategy -----------------------

Author: Alaa Alsleman, GitHub: Alaadin17

Goal: derive the charging strategy from an oemof optimization instead of a heuristic.
The whole horizon is optimized ONCE (open loop) and the resulting plan is then applied
step by step by the spice_ev simulation.

Inputs:
 - self.world_state  (Vehicles, Charging Stations, Grid Connectors, Batteries) - set by
                     the Strategy base class from the positional ``components`` argument
 - self.events       (Events object: vehicle events, fixed_load / local_generation lists;
                     kwarg passed in by scenario.py, like everything below)
 - self.oemof_config (flat dict of the oemof_* keys from simulate.cfg, prefix stripped;
                      turned into a SystemConfig via SystemConfig.from_options)
 - self.interval     (datetime.timedelta of one simulation step)
 - self.stop_time    (end of the simulation)

Flow:
 step() -> _ensure_solved() -> prepare_inputs() -> build_oemof_inputs()
        -> run_oemof_model() [EnergySystemModel.run()] -> commands_from_oemof()
'''


from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from spice_ev.strategy import Strategy
from spice_ev.util import get_cost


class OemofSolve(Strategy):
    """Charging strategy that follows a plan optimized with oemof.

    Prepares the inputs from the spice_ev scenario, builds and solves the oemof model once
    over the full horizon (``_ensure_solved``) and caches the resulting plan in
    ``self._plan``. ``step()`` then applies that plan step by step.
    """

    def __init__(self, components, start_time, **kwargs):
        super().__init__(components, start_time, **kwargs)
        self.description = "oemof_solve"

        # Inputs from kwargs
        self.events = kwargs.get("events")
        # Flat dict with oemof_* parameters (from simulate.cfg, prefix removed)
        self.oemof_config = kwargs.get("oemof_config", {}) or {}
        self.interval = kwargs.get("interval")
        self.stop_time = kwargs.get("stop_time")
        self.start_time = start_time

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
        # Index of the CURRENT simulation step = position in the lists above; step() reads
        # self._plan[<type>][<id>][self._oemof_step] and increments it afterwards.
        self._oemof_step = 0

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
            world_state_vehicles (Dict[str, WorldStateVehicle]): Dictionary of
                WorldStateVehicle objects.

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

    def _build_state_segments(self, vehicle_events, start_time, stop_time,
                              vehicles=None) -> pd.DataFrame:
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

    def _map_trips_to_state_segments(self, trip_df_by_vehicle: Dict[str, pd.DataFrame],
                                     state_segments_df: pd.DataFrame) -> pd.DataFrame:
        '''
            Map trips to state segments.
            For each segment in state_segments_df, check whether there is an overlapping
            trip in trip_df_by_vehicle. If so, the trip energy in kWh is written into the
            segment's "energy_kwh" column. Otherwise "energy_kwh" stays None.

            Args:
                trip_df_by_vehicle: Dict[vehicle_id, DataFrame] with trips per vehicle
                    (columns: departure_time, arrival_time, energy_kwh)
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
            interval: Bin width as Timedelta or a pandas-compatible frequency string
                (e.g. "15min").

        Returns:
            Tuple (per_vehicle, long_df):
                per_vehicle: dict[vehicle_id, DataFrame] indexed by time_index.
                long_df: DataFrame with columns timestamp, vehicle_id, state,
                is_driving, is_parked, energy_kwh, connected_charging_station,
                desired_soc.
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
        # segs: DataFrame with columns [start_time, end_time, state, energy_kwh] for this
        # vehicle. Iterator over pairs (key = value of the vehicle_id column, group_df =
        # the rows with that vehicle_id)
        for vid, segs in state_segments_df.groupby("vehicle_id"):
            # Initialize per-vehicle dataframe with default values.
            df = pd.DataFrame(index=time_index)
            df["state"] = "parked"  # Default state; will be overwritten by segments
            df["energy_kwh"] = 0  # Default energy; will be overwritten by segments
            df["connected_charging_station"] = None  # Default: not plugged in
            df["desired_soc"] = np.nan  # SOC required before the next departure

            # Apply each segment to all overlapping time bins.
            # _: index of the segment (ignored), seg: the segment row with start_time,
            # end_time, state, energy_kwh
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
                    df.loc[ts_slice, "connected_charging_station"] = (
                        seg["connected_charging_station"])
                    if seg["desired_soc"] is not None and not pd.isna(seg["desired_soc"]):
                        df.loc[ts_slice, "desired_soc"] = float(seg["desired_soc"])

            # Convenience boolean columns for quick filtering/plotting.
            df["is_driving"] = df["state"].eq("driving")
            df["is_parked"] = df["state"].eq("parked")
            df["vehicle_id"] = vid

            # Store per-vehicle and also build a long-format table.
            per_vehicle[vid] = df
            long_rows.append(df.reset_index().rename(columns={"index": "timestamp"}))

        # Concatenate all vehicles into one long table (timestamp, vehicle_id, ...).
        if long_rows:
            long_df = pd.concat(long_rows, ignore_index=True)
            long_df = long_df[["timestamp", "vehicle_id", "state", "is_driving", "is_parked",
                               "energy_kwh", "connected_charging_station", "desired_soc"]]
        else:
            long_df = pd.DataFrame(
                columns=["timestamp", "vehicle_id", "state", "is_driving", "is_parked",
                         "energy_kwh", "connected_charging_station", "desired_soc"]
            )
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

    def _grid_connectors(self, time_index) -> Dict[str, Dict[str, Any]]:
        """Per grid connector: max_power plus its OWN household load and PV timeseries.

        The oemof model builds one bus (Home_<n>) + source per active GC; each GC also
        carries its own load and PV, grouped by the events' ``grid_connector_id``.
        Returns {gcid: {"load", "pv", "max_power" (if the GC has one), "price_ct_kWh"
        (the scenario's price series, see ``_grid_price_series``), "pv_power_kW"
        (installed kWp, only when > 0)}}.

        A GC without price signals raises: the model prices energy with the scenario's
        own series and has no fallback.
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
            # Purchase price: the scenario's own series, the same one spice_ev's built-in
            # strategies read. The feed-in tariff is fixed (grid_feedin_tariff) and only
            # the PV surplus can earn it - the house bus has no export path.
            price = self._grid_price_series(gcid, time_index)
            if price is None:
                raise ValueError(
                    f"grid connector {gcid!r} carries no price signals. oemof_solve "
                    "prices energy with the scenario's own series and has no fallback - "
                    "add a price curve to the scenario (include_price_csv in "
                    "generate.cfg).")
            info["price_ct_kWh"] = price
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
        """The purchase price per time step, exactly as spice_ev carries it.

        spice_ev does not keep prices as a timeseries but as GridOperatorSignal EVENTS: one
        ``cost`` dict per event, valid from its ``start_time`` until the next signal.
        ``include_price_csv`` in generate.cfg turns EVERY CSV row into such an event. This
        method rebuilds the step function that spice_ev has in front of it as ``gc.cost``
        in every step - with the same ``util.get_cost`` that greedy, balanced and
        balanced_market use. Nothing is added: the LP is priced with the series the
        scenario brings, so its schedule and spice_ev's own timeseries.csv column
        "price [ct/kWh]" rest on the same numbers.

        Unit: ct/kWh. spice_ev keeps gc.cost in ct/kWh (scenario.py and costs.py both divide
        by 100 to get EUR, and generate.py names the column "price [ct/kWh]" by default), so
        the CSV is read unchanged - whoever writes EUR/kWh there is off by a factor of 100.

        Returns None when the scenario carries no signals for this GC. There is no
        fallback price; the caller raises.

        THE ONE deliberate difference to spice_ev: NEGATIVE prices are clipped to 0. An LP
        cannot be forbidden to get rid of energy - charging and discharging in the same
        step burns it through the efficiency. Being PAID to draw power then becomes a money
        pump: buy, destroy, collect again. A real house cannot do that and spice_ev does
        not simulate it either - only the LP would find it. spice_ev's own strategies do
        see the negative value, so this clip is the single remaining deviation between the
        price they read and the price the LP optimizes against. On a series whose minimum
        is above zero it never fires.
        """
        signals = [s for s in getattr(self.events, "grid_operator_signals", []) or []
                   if getattr(s, "grid_connector_id", None) == gcid
                   and getattr(s, "cost", None)]
        if not signals:
            return None
        pairs = []
        for s in sorted(signals, key=lambda s: s.start_time):
            t = pd.Timestamp(s.start_time)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            pairs.append((t, float(get_cost(1, s.cost))))          # ct/kWh, as in spice_ev
        target = time_index
        if target.tz is not None:
            target = target.tz_localize(None)
        starts = pd.DatetimeIndex([p[0] for p in pairs])
        exchange = np.array([p[1] for p in pairs])
        values = np.maximum(exchange, 0.0)
        idx = np.searchsorted(starts, target, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)   # before the first signal its value applies
        return values[idx]

    def _pv_kwp(self, gcid) -> float:
        """Installed PV nominal power (kWp) at one grid connector, summed over its plants."""
        return sum(float(pv.nominal_power)
                   for pv in getattr(self.world_state, "photovoltaics", {}).values()
                   if getattr(pv, "parent", None) == gcid)

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
            config: SystemConfig with fallback values (power/efficiency).

        Returns:
            Dict[battery_id, infos] – empty dict if there is no (valid) battery.
            infos per battery:
              capacity_kWh, power_kW (charging), discharge_power_kW (discharging),
              initial_soc, efficiency, parent (GC).

        Self-discharge is NOT handed over: spice_ev applies ``StationaryBattery.loss_rate``
        after every step (strategy.py: ``apply_battery_losses``), the LP assumes
        ``loss_rate=0.0``. As long as no scenario sets a loss rate this goes unnoticed;
        whoever sets one has to mirror it in the model.
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
                "initial_soc": float(bat.soc),  # StationaryBattery always has it (default 0.0)
                "efficiency": float(getattr(bat, "efficiency", config.battery_efficiency)),
                "parent": getattr(bat, "parent", None),
            }
        return result

    def build_oemof_inputs(self) -> Dict[str, Any]:
        """Assemble every input EnergySystemModel needs, from the spice_ev scenario.

        Returns a dict with: config (SystemConfig from the oemof_* cfg keys), time_index,
        grid_connectors (per GC: max_power, its own load/pv, the scenario's price series,
        the installed kWp when > 0), charging_stations (max_power +
        parent GC), vehicle_params (capacity/SOC/v2g/efficiency + consumption,
        connected_cs, min_soc_series and discharge_power_kW) and battery_params.
        """
        if not self._prepared:
            raise ValueError("Inputs must be prepared before building Oemof inputs")

        from spice_ev.oemof_model import SystemConfig

        config = SystemConfig.from_options(self.oemof_config)

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

            # V2G does NOT discharge at the charging power: spice_ev scales the charging
            # curve by vehicle_type.v2g_power_factor (default 0.5) into the discharge_curve,
            # and Battery.unload clamps to it. Without this value the LP plans up to the
            # full station power, the simulation delivers half, and the planned SOC drifts
            # away - measured in 03_household_v2g as 5.27 kW per step before the value was passed.
            discharge_power = getattr(
                getattr(getattr(veh, "battery", None), "unloading_curve", None),
                "max_power", None)

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
                "discharge_power_kW": (float(discharge_power)
                                       if discharge_power is not None else None),
            }

        # Charging stations (one wallbox per CS in the oemof model): power + parent GC
        charging_stations = {
            csid: {"max_power": float(cs.max_power), "parent": getattr(cs, "parent", None)}
            for csid, cs in self.world_state.charging_stations.items()
        }

        return {
            "config": config,
            "time_index": self.time_index,
            "vehicle_params": vehicle_params,
            "grid_connectors": self._grid_connectors(self.time_index),
            "battery_params": self._battery_params(config),
            "charging_stations": charging_stations,
        }

############################################################################
# ---------------------------- Oemof Model ----------------------------------
############################################################################

    def run_oemof_model(self, oemof_inputs: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Build the oemof model, solve it (full horizon) and return the full plan.

        The returned dict comes from ``EnergySystemModel.get_plan()`` and is grouped by
        component type: ``vehicles`` / ``batteries`` / ``grid`` (see there).
        """
        from spice_ev.oemof_model import EnergySystemModel

        model = EnergySystemModel(
            config=oemof_inputs["config"],
            time_index=oemof_inputs["time_index"],
            vehicle_params=oemof_inputs["vehicle_params"],
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

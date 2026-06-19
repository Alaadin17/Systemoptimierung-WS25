'''
----------------- OemofSolve strategies structure -----------------------
 
 Ziel: Optimierung der Ladestrategie mittels Oemof um die Simulationslaufzeit zu reduzieren.


 Eingaben:
 - self.events (Events-Objekt)
 - self.world_state (Vehicles, Charging Stations, Grid Connectors)
 - self.cfg (Konfiguration fuer spaetere Oemof-Anbindung)

'''


from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from spice_ev import events
from spice_ev.strategy import Strategy
from spice_ev.util import clamp_power


class OemofSolve(Strategy):
    """
    The goal is to prepare inputs (cfg + dataframes) for an Oemof model.
    Actual model creation/solving is intentionally left as placeholders.
    """

    def __init__(self, components, start_time, **kwargs):
        super().__init__(components, start_time, **kwargs)
        self.description = "oemof_solve"
        
        # Inputs from kwargs
        self.events = kwargs.get("events")
        self.cfg = kwargs.get("cfg")
        # Flaches Dict mit oemof_*-Parametern (aus simulate.cfg, Präfix entfernt)
        self.oemof_config = kwargs.get("oemof_config", {}) or {}
        self.vehicles = self.world_state.vehicles
        self.interval = kwargs.get("interval")
        self.stop_time = kwargs.get("stop_time")
        self.start_time = start_time

        # Output containers, populated later by prepare_inputs()
        self.time_index = None
        self.input_frames = {}
        self._prepared = False

        # Closed-loop-Zustand: einmalige Optimierung + gecachter Fahrplan
        self._solved = False
        self._model = None
        self._schedule: Dict[str, list] = {}  # vehicle_id -> Liste (charge_kW, discharge_kW)
        self._oemof_step = 0


###########################################################################
########## 1) Preprocessing: Rohdaten -> strukturierte DataFrames #########
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

        # 1) Stammdaten / Roh-Events als DataFrames
        df_vehicle_events = self._build_vehicle_events_df(vehicle_events)
        df_vehicles = self._build_world_state_vehicles_df(vehicles)

        # 2) Trips (departure/arrival-Paare) je Fahrzeug
        trip_df = self._build_trip_df(vehicle_events, vehicles)
        trip_df_by_vehicle = self._group_trips_by_vehicle(trip_df)

        # 3) State-Segmente (parked/driving) + Energie aus Trips
        state_segments_df = self._build_state_segments(
            vehicle_events, self.start_time, self.stop_time)
        state_segments_df = self._map_trips_to_state_segments(
            trip_df_by_vehicle, state_segments_df)

        # 4) Zeitraster + Mapping auf Zeitreihen
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
        '''Erstellt einen DataFrame aus einer Liste von WorldStateVehicles.
    
        args:
            world_state_vehicles (Dict[str, WorldStateVehicle]): Dictionary mit WorldStateVehicle-Objekten.

        returns:
            pd.DataFrame: DataFrame mit den Daten der WorldStateVehicles.
                DataFrame mit Spalten: 
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


    def _build_state_segments(self, vehicle_events, start_time, stop_time) -> pd.DataFrame:
        """Build contiguous state segments (driving/parked) per vehicle.

        For each vehicle, departure and arrival events define state changes.
        The first segment starts at start_time; the last ends at stop_time.

        Args:
            vehicle_events: List of VehicleEvent objects.
            start_time: Scenario start time (datetime).
            stop_time: Scenario stop time (datetime).

        Returns:
            DataFrame with columns: 
                                    -vehicle_id, 
                                    -start_time, 
                                    -end_time, 
                                    -state.
        """
        rows = []
        vehicle_ids = sorted({ev.vehicle_id for ev in vehicle_events})
        for vid in vehicle_ids:
            v_events = [
                ev for ev in vehicle_events
                if ev.vehicle_id == vid and ev.event_type in ("departure", "arrival")
            ]
            v_events = sorted(v_events, key=lambda e: e.start_time)
            state = "parked"  # Assume parked at the start until we see a departure
            cursor = start_time
            for ev in v_events:
                rows.append({
                    "vehicle_id": vid,
                    "start_time": cursor,
                    "end_time": ev.start_time,
                    "state": state,
                })
                if ev.event_type == "departure":
                    state = "driving"
                elif ev.event_type == "arrival":
                    state = "parked"
                cursor = ev.start_time
            rows.append({
                "vehicle_id": vid,
                "start_time": cursor,
                "end_time": stop_time,
                "state": state,
            })
        return pd.DataFrame(rows)


    def _build_time_index(self, start_time, stop_time, interval) -> pd.DatetimeIndex:

        """Build a time index from start_time to stop_time with given interval.

        Das Raster startet bei ``start_time`` (nicht start+interval), damit
        Zeile ``k`` exakt dem spice_ev-Schritt ``k`` (current_time =
        start+k*interval) entspricht. Andernfalls wären die zurückgespeisten
        Ladebefehle um einen Zeitschritt verschoben.

        Args:
            start_time: Scenario start time (datetime).
            stop_time: Scenario stop time (datetime).
            interval: Time interval (e.g., '1H' for hourly).
        Returns:
            DatetimeIndex format: DatetimeIndex(['2024-01-01 00:00:00', '2024-01-01 01:00:00', ...])
        """
        return pd.date_range(
            start=start_time, end=stop_time - pd.Timedelta(interval), freq=interval)


    def _map_trips_to_state_segments(self, trip_df_by_vehicle: Dict[str, pd.DataFrame], state_segments_df: pd.DataFrame) -> pd.DataFrame:
         
        '''
            Mappt Trips zu State-Segmente.
            Für jedes Segment in state_segments_df wird geprüft, ob es einen überlappenden Trip in trip_df_by_vehicle gibt.
            Wenn ja, wird die Energie des Trips in kWh in die Spalte "energy_kwh" des Segments eingetragen. Ansonsten bleibt "energy_kwh" None.

            Args:
                trip_df_by_vehicle: Dict[vehicle_id, DataFrame] mit Trips pro Fahrzeug (Spalten: departure_time, arrival_time, energy_kwh)
                state_segments_df: DataFrame mit Spalten [vehicle_id, start_time, end_time, state]
            Returns:
                DataFrame mit Spalten: 
                                        -vehicle_id, 
                                        -start_time, 
                                        -end_time, 
                                        -state, 
                                        -energy_kwh
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
                })
        return pd.DataFrame(rows)


    def _map_segments_to_timeseries(
        self, state_segments_df: pd.DataFrame, time_index: pd.DatetimeIndex, interval
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
            
            """Build per-vehicle timeseries on a fixed time grid.

            Each row in state_segments_df represents a continuous segment [start_time, end_time)
            with a state and optional energy_kwh. A time bin [ts, ts+interval) is marked active
            if it overlaps the segment. Returns both per-vehicle tables and a long-format table.

            Args:
                state_segments_df: DataFrame with columns [vehicle_id, start_time, end_time, state, energy_kwh].
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
            # Iterator aus Paaren (key=der wert der spalte vehicle_id, group_df=die zeilen mit diesem vehicle_id)
            for vid, segs in state_segments_df.groupby("vehicle_id"):
                print(f"Mapping segments to timeseries for vehicle {vid} with {len(segs)} segments.")
                # Initialize per-vehicle dataframe with default values.
                df = pd.DataFrame(index=time_index)
                df["state"] = "parked"  # Default state; will be overwritten by segments
                df["energy_kwh"] = 0  # Default energy; will be overwritten by segments

                # Apply each segment to all overlapping time bins.
                # _: index of the segment (ignored), seg: the segment row with start_time, end_time, state, energy_kwh
                for _, seg in segs.iterrows():
                    start = pd.to_datetime(seg["start_time"])

                    end = pd.to_datetime(seg["end_time"])

                    if start.tzinfo is not None:
                        start = start.tz_localize(None)
                    if end.tzinfo is not None:
                        end = end.tz_localize(None)

                    left = time_index.searchsorted(start, side="right")
                    right = time_index.searchsorted(end, side="left")


                    if left < right:
                        ts_slice = time_index[left:right]
                        df.loc[ts_slice, "state"] = seg["state"]
                        df.loc[ts_slice, "energy_kwh"] = 0
                        df.loc[ts_slice[-1], "energy_kwh"] = seg["energy_kwh"]

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
                long_df = long_df[["timestamp", "vehicle_id", "state", "unterwegs", "zuhause", "energy_kwh"]]
            else:
                long_df = pd.DataFrame(
                    columns=["timestamp", "vehicle_id", "state", "unterwegs", "zuhause", "energy_kwh"]
                )
            print(f"Mapped segments to timeseries for {len(per_vehicle)} vehicles, resulting 1) dictionary with {len(per_vehicle)} vehicles with {len(per_vehicle)} dataframes with {len(per_vehicle[vid])} rows and 2) complete table with {len(long_df)} rows.")
            # Return both representations: per-vehicle dict and long-format table.
            return per_vehicle, long_df
        

    # ------------------------------------------------------------------
    # Brücke spice_ev -> oemof
    # ------------------------------------------------------------------
    def _sample_event_list(self, ev_list, time_index: pd.DatetimeIndex) -> pd.Series:
        """Tastet eine EnergyValuesList (Step-Funktion) auf das Zeitraster ab.
        Args:            
                ev_list: EnergyValuesList mit Werten und step_duration_s.
                time_index: Ziel-Zeitindex für die Ausgabe (DatetimeIndex).
        
        Returns:
                pd.Series mit index=time_index, Werten aus ev_list (stepweise konstant) und 0 außerhalb der ev_list-Zeiten.

        We need it when there are in the Szenario PV or Load (include_local_generation_csv / include_fixed_load_csv in der generate.cfg). 
        """
        values = list(getattr(ev_list, "values", []) or [])
        if not values:
            return pd.Series(0.0, index=time_index)

        delta = pd.Timedelta(seconds=ev_list.step_duration_s)
        raw_index = pd.date_range(start=ev_list.start_time, periods=len(values), freq=delta)
        factor = getattr(ev_list, "factor", 1) or 1
        raw = pd.Series(np.asarray(values, dtype=float) * factor, index=raw_index)

        # Zeitzonen vereinheitlichen (tz-naiv), damit reindex funktioniert
        if raw.index.tz is not None:
            raw.index = raw.index.tz_localize(None)
        target = time_index
        if target.tz is not None:
            target = target.tz_localize(None)

        aligned = raw.reindex(target, method="ffill").fillna(0.0)
        aligned.index = time_index
        return aligned

    def _aggregate_event_lists(self, event_lists, time_index: pd.DatetimeIndex) -> pd.Series:
        """Summiert mehrere EnergyValuesLists (z.B. mehrere PV-Anlagen) auf das Raster."""
        total = pd.Series(0.0, index=time_index)
        for ev_list in (event_lists or {}).values():
            total = total.add(self._sample_event_list(ev_list, time_index), fill_value=0.0)
        return total

    def _vehicle_cs_map(self) -> Dict[str, str]:
        """Ordnet jedem Fahrzeug seine Ladestation zu (aus Events / Initialzustand).
        Args:
            vehicle_events: List[VehicleEvent] mit möglichen "connected_charging_station"-Updates.
            world_state.vehicles: Dict[vehicle_id, Vehicle] mit initial verbundenen Ladestationen.

        Returns:
            Dict[vehicle_id, charging_station_id] mit der zugeordneten Ladestation je Fahrzeug.
        """
        mapping: Dict[str, str] = {}
        for ev in getattr(self.events, "vehicle_events", []):
            cs = ev.update.get("connected_charging_station")
            if cs:
                mapping.setdefault(ev.vehicle_id, cs)
        # Fallback: initial verbundene CS aus dem world_state
        for vid, v in self.world_state.vehicles.items():
            if vid not in mapping and getattr(v, "connected_charging_station", None):
                mapping[vid] = v.connected_charging_station
        return mapping

    def _grid_power(self) -> Optional[float]:
        """
        Summe der Netzanschlussleistungen (max_power) der Grid-Connectors.
        Args:
                world_state.grid_connectors: Dict[connector_id, GridConnector] mit möglichen max_power-Attributen.
        Returns:
                - Float: mit der Summe der max_power aller Grid-Connectors, 
                - None: wenn keine max_power definiert ist.
        """
        powers = [gc.max_power for gc in self.world_state.grid_connectors.values()
                  if getattr(gc, "max_power", None)]
        return float(sum(powers)) if powers else None

    def _battery_params(self, config) -> Dict[str, Dict[str, Any]]:
        """Liest ALLE stationären Batterien aus dem Szenario.

        Args:
            config: SystemConfig mit Fallback-Werten (Leistung/SOC/Effizienz).

        Returns:
            Dict[battery_id, infos] – leeres Dict, wenn keine (valide) Batterie da ist.
            infos je Batterie:
              capacity_kWh, power_kW (laden), discharge_power_kW (entladen),
              initial_soc, efficiency, min_power_kW, loss_rate (dict), parent (GC).
        """
        result: Dict[str, Dict[str, Any]] = {}
        for bid, bat in getattr(self.world_state, "batteries", {}).items():
            capacity = float(getattr(bat, "capacity", 0) or 0)
            # <=0 oder unbegrenzt (StationaryBattery setzt 2**64) -> ueberspringen
            if capacity <= 0 or capacity > 1e9:
                continue
            try:
                power = float(bat.loading_curve.max_power)
            except Exception:
                power = config.battery_max_power_kW
            try:
                discharge_power = float(bat.unloading_curve.max_power)
            except Exception:
                discharge_power = power  # keine eigene Entladekurve -> wie Laden
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
        """Baut die oemof-Eingaben (config, Zeitreihe, Fahrzeugparameter)."""
        if not self._prepared:
            raise ValueError("Inputs must be prepared before building Oemof inputs")

        from spice_ev.oemof import SystemConfig

        config = SystemConfig.from_options(self.oemof_config)

        # Globale PV/Last aus den Events auf das Zeitraster bringen
        pv = self._aggregate_event_lists(
            getattr(self.events, "local_generation_lists", {}), self.time_index)
        load = self._aggregate_event_lists(
            getattr(self.events, "fixed_load_lists", {}), self.time_index)
        timeseries_df = pd.DataFrame(
            {"PV_kW": pv.to_numpy(), "Load_kW": load.to_numpy()}, index=self.time_index)

        # Fahrzeug-Stammdaten (Kapazität/SOC/v2g) je Fahrzeug
        vehicles_df = self.input_frames["vehicles"].set_index("vehicle_id")
        per_vehicle_ts = self.input_frames["per_vehicle_ts"]

        # Fahrzeug -> Ladestation, um die Wallbox-Leistung aus dem Szenario zu ziehen
        cs_map = self._vehicle_cs_map()
        charging_stations = self.world_state.charging_stations

        vehicle_params: Dict[str, Dict[str, Any]] = {}
        for vid, ts in per_vehicle_ts.items():
            at_home = ts["zuhause"].astype(float).to_numpy()
            consumption = ts["energy_kwh"].fillna(0).astype(float).to_numpy()

            if vid in vehicles_df.index:
                row = vehicles_df.loc[vid]
                capacity = float(row["capacity_kwh"])
                init_soc = float(row["soc"])
                v2g = bool(row["v2g"])
                desired_soc = float(row["desired_soc"]) if not pd.isna(
                    row.get("desired_soc")) else config.bev_max_soc
            else:
                capacity = config.bev_capacity_kWh
                init_soc = config.bev_initial_soc
                v2g = config.enable_v2h
                desired_soc = config.bev_max_soc

            # desired_soc in den zulässigen Bereich klemmen
            desired_soc = min(max(desired_soc, config.bev_min_soc), config.bev_max_soc)

            # Wallbox-Leistung aus der zugeordneten Ladestation (Fallback: config)
            cs = charging_stations.get(cs_map.get(vid))
            wallbox_power = (float(cs.max_power) if cs is not None
                             else config.wallbox_power_kW)

            # Zeitabhängiger Mindest-SOC: vor jeder Abfahrt (letzter Zuhause-Schritt
            # vor einer Fahrt) muss der BEV auf desired_soc geladen sein. So bleibt
            # genug Puffer, damit spice_ev (nichtlineares Batteriemodell) nicht
            # unter 0 / desired fällt.
            min_soc_series = np.full(len(at_home), config.bev_min_soc, dtype=float)
            for i in range(len(at_home) - 1):
                if at_home[i] >= 0.5 and at_home[i + 1] < 0.5:
                    min_soc_series[i] = desired_soc

            vehicle_params[vid] = {
                "capacity_kWh": capacity,
                "min_soc": config.bev_min_soc,
                "max_soc": config.bev_max_soc,
                "initial_soc": init_soc,
                "v2g": v2g,
                "at_home": at_home,
                "consumption": consumption,
                "min_soc_series": min_soc_series,
                "wallbox_power_kW": wallbox_power,
            }

        return {
            "config": config,
            "timeseries_df": timeseries_df,
            "time_index": self.time_index,
            "vehicle_params": vehicle_params,
            "grid_power": self._grid_power(),
            "battery_params": self._battery_params(config),
        }

    def run_oemof_model(self, oemof_inputs: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Erstellt das oemof-Modell, löst es (Ganzhorizont) und liefert den Fahrplan."""
        from spice_ev.oemof import EnergySystemModel

        model = EnergySystemModel(
            config=oemof_inputs["config"],
            timeseries_df=oemof_inputs["timeseries_df"],
            time_index=oemof_inputs["time_index"],
            vehicle_params=oemof_inputs["vehicle_params"],
            grid_power=oemof_inputs.get("grid_power"),
            battery_params=oemof_inputs.get("battery_params"),
        )
        model.run()
        self._model = model
        return model.get_wallbox_schedule()

    def commands_from_oemof(self, oemof_results: Dict[str, pd.DataFrame]) -> Dict[str, list]:
        """Wandelt die per-Fahrzeug-Fahrpläne in positionsindizierte Befehlslisten."""
        schedule: Dict[str, list] = {}
        for vid, df in (oemof_results or {}).items():
            charge = df["charge_kW"].to_numpy()
            discharge = df["discharge_kW"].to_numpy()
            schedule[vid] = list(zip(charge.tolist(), discharge.tolist()))
        return schedule

    def _ensure_solved(self) -> None:
        """Löst die Optimierung genau einmal und cached den Ladeplan."""
        if self._solved:
            return
        self.prepare_inputs()
        oemof_inputs = self.build_oemof_inputs()
        results = self.run_oemof_model(oemof_inputs)
        self._schedule = self.commands_from_oemof(results)
        self._solved = True
        self._oemof_step = 0

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

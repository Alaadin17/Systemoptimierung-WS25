# -*- coding: utf-8 -*-
# Copyright (c) 2025 by Alaa Alsleman
###############################################
'''
This Part sets up the environment to run SpiceEV scripts by adding the SpiceEV repository
to the system path.
'''
from pathlib import Path
import sys
import json
import random
from datetime import datetime, timedelta

# Pfad zum SpiceEV-Repo hier EINTRAGEN:
SPICE_EV_DIR = Path(r"C:\git\github\spice_ev").absolute()

if str(SPICE_EV_DIR) not in sys.path:
    sys.path.append(str(SPICE_EV_DIR))

###############################################
###############################################


class ScenarioGenerator:
    """
    Class to generate different types of scenarios for SpiceEV
    
    Methods:
        - generate_basic_scenario
        - generate_workplace_scenario
        - generate_fleet_scenario
        - generate_custom_scenario
        - generate_random_scenario

    Each method creates a scenario dictionary and saves it as a JSON file in its own folder
    under the 'scenarios' directory.

    Attributes:
        - output_folder: Path to the folder where scenarios will be saved
    """
    
    def __init__(self, output_folder="scenarios"):
        # Create output folder in the same directory as this script
        script_dir = Path(__file__).parent
        self.output_folder = script_dir / output_folder
        self.output_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist
        
        self.base_scenario = {
            "scenario": {
                "start_time": "2023-01-01 00:00:00",
                "interval": 15,
                "n_intervals": 96,
                "description": "Generated scenario"
            },
            "components": {
                "vehicle_types": {},
                "grid_connectors": {},
                "charging_stations": {},
                "vehicles": {}
            },
            "events": {}
        }

    def _create_base_scenario(self, start_time="2023-01-01 00:00:00", interval=15, n_intervals=96):
        """Create a base scenario with specified parameters"""
        scenario = self.base_scenario.copy()
        scenario["scenario"]["start_time"] = start_time
        scenario["scenario"]["interval"] = interval
        scenario["scenario"]["n_intervals"] = n_intervals
        return scenario

    def generate_basic_scenario(self, num_vehicles=1, start_time="2023-01-01 00:00:00", 
                               interval=15, n_intervals=96, scenario_name="basic_scenario"):
        """Generate a basic scenario with simple parameters"""
        scenario = self._create_base_scenario(start_time, interval, n_intervals)
        
        # Set scenario name
        scenario["scenario"]["description"] = f"Basic scenario with {num_vehicles} vehicles"
        
        # Add vehicle types
        scenario["components"]["vehicle_types"] = {
            "standard_ev": {
                "name": "standard_ev",
                "capacity": 50,
                "charging_curve": [[0, 50], [0.8, 50], [1.0, 10]],
                "min_charging_power": 0
            }
        }
        
        # Add grid connector
        scenario["components"]["grid_connectors"] = {
            "GC1": {
                "max_power": 100,
                "cost": {"type": "fixed", "value": 0.30},
                "voltage_level": "LV"
            }
        }
        
        # Add charging stations
        scenario["components"]["charging_stations"] = {
            f"CS{i}": {
                "max_power": 22,
                "parent": "GC1"
            } for i in range(1, num_vehicles + 1)
        }
        
        # Add vehicles with random arrival/departure times
        for i in range(1, num_vehicles + 1):
            arrival_hour = random.randint(7, 10)
            departure_hour = random.randint(16, 19)
            
            scenario["components"]["vehicles"][f"v{i}"] = {
                "vehicle_type": "standard_ev",
                "connected_charging_station": f"CS{i}",
                "estimated_time_of_arrival": f"{start_time[:10]} {arrival_hour:02d}:00:00",
                "estimated_time_of_departure": f"{start_time[:10]} {departure_hour:02d}:00:00",
                "desired_soc": 0.8,
                "soc_delta": random.uniform(0.2, 0.6)
            }
        
        self._save_scenario(scenario, scenario_name)
        return scenario

    def generate_workplace_scenario(self, num_vehicles=1, start_time="2023-01-01 00:00:00", 
                                   interval=15, n_intervals=96, scenario_name="workplace_scenario"):
        """Generate a workplace charging scenario"""
        scenario = self._create_base_scenario(start_time, interval, n_intervals)
        
        scenario["scenario"]["description"] = f"Workplace charging scenario with {num_vehicles} vehicles"
        
        # Vehicle types with different battery sizes
        scenario["components"]["vehicle_types"] = {
            "small_ev": {"name": "small_ev", "capacity": 30, "charging_curve": [[0, 30], [0.8, 30], [1.0, 8]]},
            "medium_ev": {"name": "medium_ev", "capacity": 50, "charging_curve": [[0, 50], [0.8, 50], [1.0, 12]]},
            "large_ev": {"name": "large_ev", "capacity": 75, "charging_curve": [[0, 75], [0.8, 75], [1.0, 18]]}
        }
        
        # Grid connection
        scenario["components"]["grid_connectors"] = {
            "workplace_grid": {
                "max_power": 200,
                "cost": {"type": "time_series", "csv_file": "workplace_tariff.csv"},
                "voltage_level": "MV"
            }
        }
        
        # Charging stations
        for i in range(1, num_vehicles + 1):
            scenario["components"]["charging_stations"][f"WP_CS{i}"] = {
                "max_power": 11,
                "parent": "workplace_grid"
            }
        
        # Vehicles arriving in morning, leaving in evening
        vehicle_types = ["small_ev", "medium_ev", "large_ev"]
        for i in range(1, num_vehicles + 1):
            arrival_time = f"{start_time[:10]} {random.randint(7, 9):02d}:{random.randint(0, 59):02d}:00"
            departure_time = f"{start_time[:10]} {random.randint(16, 18):02d}:{random.randint(0, 59):02d}:00"
            
            scenario["components"]["vehicles"][f"emp{i}"] = {
                "vehicle_type": random.choice(vehicle_types),
                "connected_charging_station": f"WP_CS{i}",
                "estimated_time_of_arrival": arrival_time,
                "estimated_time_of_departure": departure_time,
                "desired_soc": 0.8,
                "soc_delta": random.uniform(0.3, 0.7)
            }
        
        self._save_scenario(scenario, scenario_name)
        return scenario

    def generate_fleet_scenario(self, num_vehicles=1, start_time="2023-01-01 00:00:00", 
                               interval=15, n_intervals=672, scenario_name="fleet_scenario"):
        """Generate a commercial fleet scenario"""
        scenario = self._create_base_scenario(start_time, interval, n_intervals)
        
        scenario["scenario"]["description"] = f"Commercial fleet scenario with {num_vehicles} vehicles"
        
        # Heavy duty vehicles
        scenario["components"]["vehicle_types"] = {
            "delivery_van": {"name": "delivery_van", "capacity": 60, "charging_curve": [[0, 60], [0.8, 60], [1.0, 15]]},
            "truck": {"name": "truck", "capacity": 100, "charging_curve": [[0, 100], [0.8, 100], [1.0, 25]]}
        }
        
        # High-power grid connection
        scenario["components"]["grid_connectors"] = {
            "depot_grid": {
                "max_power": 500,
                "cost": {"type": "fixed", "value": 0.25},
                "voltage_level": "MV"
            }
        }
        
        # Fast charging stations
        for i in range(1, num_vehicles + 1):
            scenario["components"]["charging_stations"][f"DC{i}"] = {
                "max_power": 50,
                "parent": "depot_grid"
            }
        
        # Vehicles with shift patterns
        for i in range(1, num_vehicles + 1):
            vehicle_type = "delivery_van" if i <= 10 else "truck"
            shift_start = random.choice([6, 14, 22])  # 3 shifts
            
            scenario["components"]["vehicles"][f"fleet{i}"] = {
                "vehicle_type": vehicle_type,
                "connected_charging_station": f"DC{i}",
                "estimated_time_of_arrival": f"{start_time[:10]} {shift_start:02d}:00:00",
                "estimated_time_of_departure": f"{start_time[:10]} {(shift_start + 8) % 24:02d}:00:00",
                "desired_soc": 0.9,
                "soc_delta": random.uniform(0.4, 0.8)
            }
        
        self._save_scenario(scenario, scenario_name)
        return scenario

    def generate_custom_scenario(self, config_dict, start_time="2023-01-01 00:00:00", 
                                interval=15, n_intervals=96, scenario_name="custom_scenario"):
        """Generate a scenario based on custom configuration"""
        scenario = self._create_base_scenario(start_time, interval, n_intervals)
        
        # Update with custom config
        for key, value in config_dict.items():
            if key in scenario:
                scenario[key].update(value)
        
        self._save_scenario(scenario, scenario_name)
        return scenario

    def generate_random_scenario(self, num_vehicles=1, start_time="2023-01-01 00:00:00", 
                                interval=15, n_intervals=96, scenario_name="random_scenario"):
        """Generate a completely randomized scenario"""
        if num_vehicles is None:
            num_vehicles = random.randint(5, 30)
            
        scenario = self._create_base_scenario(start_time, interval, n_intervals)
        
        scenario["scenario"]["description"] = f"Random scenario with {num_vehicles} vehicles"
        
        # Random vehicle types
        scenario["components"]["vehicle_types"] = {
            f"type_{i}": {
                "name": f"type_{i}",
                "capacity": random.randint(20, 100),
                "charging_curve": [[0, random.randint(20, 100)], [0.8, random.randint(20, 100)], [1.0, random.randint(5, 25)]]
            } for i in range(1, 4)
        }
        
        # Random grid setup
        scenario["components"]["grid_connectors"] = {
            "random_grid": {
                "max_power": random.randint(100, 500),
                "cost": {"type": "fixed", "value": round(random.uniform(0.15, 0.45), 2)},
                "voltage_level": random.choice(["LV", "MV", "HV"])
            }
        }
        
        # Random charging stations and vehicles
        vehicle_types = list(scenario["components"]["vehicle_types"].keys())
        for i in range(1, num_vehicles + 1):
            scenario["components"]["charging_stations"][f"CS{i}"] = {
                "max_power": random.choice([11, 22, 50]),
                "parent": "random_grid"
            }
            
            scenario["components"]["vehicles"][f"v{i}"] = {
                "vehicle_type": random.choice(vehicle_types),
                "connected_charging_station": f"CS{i}",
                "estimated_time_of_arrival": f"{start_time[:10]} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
                "estimated_time_of_departure": f"{start_time[:10]} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00",
                "desired_soc": round(random.uniform(0.7, 1.0), 2),
                "soc_delta": round(random.uniform(0.1, 0.8), 2)
            }
        
        self._save_scenario(scenario, scenario_name)
        return scenario

    def _save_scenario(self, scenario, scenario_name):
        """Save scenario to JSON file in its own folder under scenarios/"""
        scenario_folder = self.output_folder / scenario_name
        scenario_folder.mkdir(exist_ok=True)
        
        filepath = scenario_folder / f"{scenario_name}.json"
        with open(filepath, 'w') as f:
            json.dump(scenario, f, indent=2)
        print(f"{scenario.get('scenario', {}).get('description', scenario_name)} saved to: {filepath}")


# Example usage functions
def main():
    """Example of how to use the scenario generator with configurable parameters"""
    
    # Configure your scenario parameters here
    start_time = "2023-01-01 00:00:00"  # <-- Hier kannst du das Datum ändern, z.B. "2024-06-15 00:00:00"
    interval = 15                        # Time interval in minutes
    n_intervals = 24*4                    # Number of intervals (192 * 15min = 48 hours)
    num_vehicles = 1                    # Number of vehicles
    
    # Create generator
    generator = ScenarioGenerator()
    
    print("Generating scenarios with custom parameters...")
    print(f"Start time: {start_time}")
    print(f"Interval: {interval} minutes")
    print(f"Number of intervals: {n_intervals}")
    print(f"Number of vehicles: {num_vehicles}")
    print("-" * 50)
    

    # Basic scenario with custom parameters
    generator.generate_basic_scenario(
        num_vehicles=num_vehicles,
        start_time=start_time,
        interval=interval,
        n_intervals=n_intervals,
        scenario_name="basic_scenario"
    )
    
    # # Workplace scenario
    # generator.generate_workplace_scenario(
    #     num_vehicles=num_vehicles,
    #     start_time=start_time,
    #     interval=interval,
    #     n_intervals=n_intervals,
    #     scenario_name="workplace_custom"
    # )
    
    # # Fleet scenario
    # generator.generate_fleet_scenario(
    #     num_vehicles=num_vehicles,
    #     start_time=start_time,
    #     interval=interval,
    #     n_intervals=672,  # One week for fleet
    #     scenario_name="fleet_custom"
    # )
    
    # # Random scenario
    # generator.generate_random_scenario(
    #     num_vehicles=num_vehicles,
    #     start_time=start_time,
    #     interval=interval,
    #     n_intervals=n_intervals,
    #     scenario_name="random_custom"
    # )
    
    print("All scenarios generated!")


if __name__ == "__main__":
    main()
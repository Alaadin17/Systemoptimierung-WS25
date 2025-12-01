# -*- coding: utf-8 -*-
# Copyright (c) 2025 by Alaa Alsleman
###############################################
'''
This Part sets up the environment to run SpiceEV scripts by adding the SpiceEV repository
to the system path.
'''
from pathlib import Path
import sys

# Pfad zum SpiceEV-Repo hier EINTRAGEN:
SPICE_EV_DIR = Path(r"C:\git\github\spice_ev").absolute()

if str(SPICE_EV_DIR) not in sys.path:
    sys.path.append(str(SPICE_EV_DIR))

# Prüfe ob der Pfad existiert
if not SPICE_EV_DIR.exists():
    print(f"Error: SpiceEV directory not found at {SPICE_EV_DIR}")
    print("Please update SPICE_EV_DIR variable with the correct path.")
    sys.exit(1)


###############################################
'''
This Part provides a function to simulate a SpiceEV scenario
using specified parameters.
'''
from argparse import Namespace
from simulate import simulate
from spice_ev.util import set_options_from_config
import os


def simulate_scenario(config_file: str):
    """
    Simulate a SpiceEV scenario using a config file.

    :param config_file: Pfad zur Config-Datei
    """
    # Save current directory
    original_dir = os.getcwd()
    
    # Change to SpiceEv directory so relative paths in config work
    config_path = Path(config_file).absolute()
    spiceev_dir = config_path.parent.parent  # Go to SpiceEv folder
    os.chdir(spiceev_dir)
    
    try:
        # Get relative path from SpiceEv folder to config file
        relative_config = config_path.relative_to(spiceev_dir)
        
        # Create minimal Namespace with config file path
        params = Namespace(config=str(relative_config))
        
        # Let SpiceEV parse the config file
        set_options_from_config(params, check=None, verbose=False)
        
        # Check if input file exists
        input_path = Path(params.input)
        if not input_path.exists():
            print(f"\nError: Scenario file not found: {input_path.absolute()}")
            print(f"\nPlease run 'generate_scenario.py' first to create scenarios.")
            print(f"Available scenarios:")
            scenarios_folder = spiceev_dir / "scenarios"
            if scenarios_folder.exists():
                for folder in scenarios_folder.iterdir():
                    if folder.is_dir() and folder.name != "price":
                        json_file = folder / f"{folder.name}.json"
                        if json_file.exists():
                            print(f"  - {folder.name}")
            return
        
        # Run simulation
        simulate(params)
    finally:
        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    project_root = Path(__file__).parent
    
    # Pfad zur Config-Datei
    config_file = project_root / "configs" / "simulate.cfg"
    
    print(f"Using config file: {config_file}")
    
    # Prüfe ob die Config-Datei existiert
    if not config_file.exists():
        print(f"Error: Config file not found at {config_file}")
        sys.exit(1)
    
    simulate_scenario(str(config_file))
    print("Simulation completed.")
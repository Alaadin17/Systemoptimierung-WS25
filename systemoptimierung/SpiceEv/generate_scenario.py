import subprocess
from pathlib import Path


def generate_scenario_statistics(mode=None, scenario_name=None, use_config=False, config_path=None):
    """
    Run the generate.py script to create a scenario.
    
    Args:
        mode (str): Generation mode - 'statistics', 'csv', or 'simbev'. Default: 'statistics'
        scenario_name (str): Name of the output scenario file (without .json extension). Default: 'scenario'
        use_config (bool): If True, use a config file instead of command line arguments. Default: False
        config_path (str): Path to the config file. Default: 'examples/configs/generate.cfg'
    
    The output will be saved to systemoptimierung/SpiceEv/scenarios/
    
    Returns:
        subprocess.CompletedProcess: The result of the command execution
    """
    # Get the root directory of the project (3 levels up from this file)
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]
    
    # Define the output directory
    output_dir = current_file.parent / "scenarios"
    
    # Create the scenarios directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Define output file path
    output_file = output_dir / f"{scenario_name}.json"
    
    # Build command based on whether config file is used
    if use_config:
        # Use config file
        if config_path is None:
            config_path = "examples/configs/generate.cfg"
        
        cmd = ["python", "generate.py", "--config", config_path]
        print(f"Running with config file: {config_path}")
    else:
        # Use command line arguments
        cmd = ["python", "generate.py", mode, "-o", str(output_file)]
    
    # Change to project root directory and run the command
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True
    )
    
    # Print output
    if result.stdout:
        print("Output:", result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    print(f"Return code: {result.returncode}")
    
    if result.returncode == 0:
        print(f"Scenario successfully generated at: {output_file}")
    else:
        print("Command failed!")
    
    return result


if __name__ == "__main__":
    generate_scenario_statistics(mode="statistics", scenario_name="scenario_statistics_1")
    generate_scenario_statistics(mode="statistics", scenario_name="scenario_statistics_2", use_config=True, config_path="generate.cfg")
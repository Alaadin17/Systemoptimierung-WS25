import subprocess
from pathlib import Path


def generate_scenario_statistics(mode=None, scenario_name=None):
    """
    Run the generate.py script to create a scenario.
    
    Args:
        mode (str): Generation mode - 'statistics', 'csv', or 'simbev'. Default: 'statistics'
        scenario_name (str): Name of the output scenario file (without .json extension). Default: 'scenario'
    
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
    
    # Change to project root directory and run the command
    result = subprocess.run(
        ["python", "generate.py", mode, "-o", str(output_file)],
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

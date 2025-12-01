import subprocess
from pathlib import Path


def generate_scenario_statistics():
    """
    Run the generate.py script with statistics mode to create a scenario.
    
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
    output_file = output_dir / "scenario.json"
    
    # Change to project root directory and run the command
    result = subprocess.run(
        ["python", "generate.py", "statistics", "-o", str(output_file)],
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
    generate_scenario_statistics()

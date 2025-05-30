import subprocess
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run a series of simulations for a single configuration.")

    # Configuration parameters for study.py
    parser.add_argument("--config_l", type=int, required=True, help="Hierarchy levels (parameter l for generate_S).")
    parser.add_argument("--config_n_bottom", type=int, required=True, help="Number of bottom-level series (parameter n for generate_S).")
    parser.add_argument("--config_n_obs", type=int, required=True, help="Number of observations for the series.")
    parser.add_argument("--num_config", type=int, required=True, help="Configuration number (identifier for this experiment setup).")

    # Model and simulation parameters for study.py
    parser.add_argument("--model_type", type=str, default="gam", choices=["gam", "lm"], help="Forecasting model type (gam or lm).")
    parser.add_argument("--noise_type", type=str, default="A", choices=["A", "B"], help="Noise type for data generation (A or B).")
    parser.add_argument("--alpha", type=float, default=0.1, help="Significance level for conformal prediction.")

    # Batch execution parameters for this script
    parser.add_argument("--start_index", type=int, default=0, help="Starting simulation index (inclusive).")
    parser.add_argument("--end_index", type=int, required=True, help="Ending simulation index (exclusive). Defines the total number of simulations in this batch.")

    # Paths
    parser.add_argument("--python_executable", type=str, default=sys.executable, help="Path to the Python interpreter.")
    parser.add_argument("--study_script_path", type=str, default="py_simulation/src/study.py", help="Path to the study.py script.")

    args = parser.parse_args()

    # Validate study_script_path
    study_script_file = Path(args.study_script_path)
    if not study_script_file.is_file():
        # Try resolving relative to the current script's directory if not found directly
        current_script_dir = Path(__file__).parent
        study_script_file = current_script_dir / args.study_script_path
        if not study_script_file.is_file():
            # If still not found, try relative to project root (assuming src/run_study_series.py)
            project_root_study_path = Path(__file__).parent.parent.parent / args.study_script_path
            if project_root_study_path.is_file():
                 study_script_file = project_root_study_path
            else:
                print(f"Error: study.py script not found at specified path '{args.study_script_path}' or common relative paths.", file=sys.stderr)
                sys.exit(1)

    # The num_simulations argument for study.py should reflect the total count in this batch
    # This is for context if study.py uses it for logging, e.g. "Simulation X out of Y"
    # If study.py doesn't use it, it's fine. The R script had it.
    # Here, end_index effectively defines the total number of simulations *if* start_index is 0.
    # If start_index is not 0, study.py's num_simulations might be more about the total planned for the config.
    # Let's pass args.end_index as the total number of simulations intended for this config.
    # This implies that if you run 0-50, then 50-100, num_simulations for study.py should be 100.
    # For simplicity here, we will assume end_index represents the total simulations for the purpose of study.py's argument.
    # If a more complex setup (e.g. multiple batches for one config) is needed, this might need adjustment.
    # The R script used 'num_simulations' for the total in the config, and 'simulation_index' for the current one.
    num_simulations_for_study_py = args.end_index

    print(f"Starting batch simulation for Config {args.num_config} from index {args.start_index} to {args.end_index - 1}.")
    print(f"Python executable: {args.python_executable}")
    print(f"Study script: {study_script_file.resolve()}")

    for sim_idx in range(args.start_index, args.end_index):
        command = [
            args.python_executable,
            str(study_script_file.resolve()),
            "--config_l", str(args.config_l),
            "--config_n_bottom", str(args.config_n_bottom),
            "--config_n_obs", str(args.config_n_obs),
            "--num_config", str(args.num_config),
            "--model_type", args.model_type,
            "--noise_type", args.noise_type,
            "--alpha", str(args.alpha),
            "--simulation_index", str(sim_idx),
            "--num_simulations", str(num_simulations_for_study_py) # study.py expects this
        ]

        print(f"\nRunning Config {args.num_config}, Simulation Index: {sim_idx}...")
        print(f"Command: {' '.join(command)}")

        try:
            # Set cwd to project root if paths in study.py are relative (e.g. ./Config_X)
            # Assuming this script (run_study_series.py) is in py_simulation/src/
            # Project root would be two levels up from this script's parent.
            project_root = Path(__file__).resolve().parent.parent.parent

            process = subprocess.run(command, check=True, text=True, cwd=project_root)
            # No capture_output=True, so study.py's stdout/stderr will print to console directly.

        except subprocess.CalledProcessError as e:
            print(f"Error running study.py for simulation index {sim_idx}, Config {args.num_config}.", file=sys.stderr)
            print(f"Command failed: {' '.join(e.cmd)}", file=sys.stderr)
            print(f"Return code: {e.returncode}", file=sys.stderr)
            if e.stdout:
                print(f"stdout:\n{e.stdout}", file=sys.stderr)
            if e.stderr:
                print(f"stderr:\n{e.stderr}", file=sys.stderr)
            # Decide if to continue with other simulations or exit
            # For now, let's print error and continue
            # To stop on first error, uncomment: sys.exit(1)
        except FileNotFoundError:
            print(f"Error: Python executable '{args.python_executable}' or study script '{study_script_file.resolve()}' not found.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while trying to run simulation index {sim_idx}: {e}", file=sys.stderr)
            # For now, continue to next simulation

    print(f"\nBatch simulation run completed for Config {args.num_config} from index {args.start_index} to {args.end_index - 1}.")

if __name__ == "__main__":
    main()

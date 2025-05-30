import subprocess
import sys
import argparse
from pathlib import Path
import numpy as np # For sum in m_total_series calculation

def calculate_m_total_series(l_val: int, n_bottom_val: int) -> int:
    """
    Calculates the total number of series (m) based on the logic in generate_S.
    m = 1 (grand total) + sum_{i=1 to l} (12^i) + n_bottom.
    """
    sum_agg_levels = sum([12**i for i in range(1, l_val + 1)])
    return 1 + sum_agg_levels + n_bottom_val

def main():
    parser = argparse.ArgumentParser(description="Main pipeline to run simulations and evaluations.")
    parser.add_argument("--skip-simulations", action="store_true", help="Skip the simulation phase (study.py runs).")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the evaluation phase (gather, plot, latex).")
    parser.add_argument("--model-type", type=str, default="gam", choices=["gam", "lm"],
                        help="Global model type for simulations if not specified per config.")
    # Add arguments to select specific config_nums to run, e.g. --configs 1 2 3
    parser.add_argument("--configs", type=int, nargs='+', default=None,
                        help="Optional list of config_num to run. If None, runs all defined configs.")


    args = parser.parse_args()

    # Define configurations
    # For local testing, n_obs and num_sim_indices are small.
    # m_total_series is calculated based on l and n_bottom.
    # model_type can be global or per-config. Here, use global from args.model_type.

    all_configs_data_template = [
        # Based on R's config_mat: l_vector = c(1,2,1,1,2,3), n_vector = c(12,12,144,24,144,144)
        # Simulating first two for brevity in testing
        {"l": 1, "n_bottom": 12, "n_obs": 500, "num_sim_indices": 2, "config_num": 1},
        {"l": 2, "n_bottom": 12, "n_obs": 500, "num_sim_indices": 2, "config_num": 2},
        # {"l": 1, "n_bottom": 144, "n_obs": 500, "num_sim_indices": 2, "config_num": 3},
        # {"l": 1, "n_bottom": 24, "n_obs": 500, "num_sim_indices": 2, "config_num": 4},
        # {"l": 2, "n_bottom": 144, "n_obs": 500, "num_sim_indices": 2, "config_num": 5},
        # {"l": 3, "n_bottom": 144, "n_obs": 500, "num_sim_indices": 2, "config_num": 6},
    ]

    # Calculate m_total_series for each config and add model_type
    configs_to_run = []
    for config_template in all_configs_data_template:
        if args.configs is None or config_template["config_num"] in args.configs:
            m_total = calculate_m_total_series(config_template["l"], config_template["n_bottom"])
            configs_to_run.append({**config_template, "m_total_series": m_total, "model_type": args.model_type})

    if not configs_to_run:
        print("No configurations selected to run. Exiting.")
        if args.configs:
             print(f"You specified --configs {args.configs}, but these are not in the defined templates or templates are empty.")
        sys.exit(0)


    python_executable = sys.executable
    project_root = Path(__file__).resolve().parent.parent.parent # py_simulation/src/main_pipeline.py -> project_root

    # Paths to scripts (relative to project_root assuming they are in py_simulation/src/)
    run_study_series_script_path = project_root / "py_simulation" / "src" / "run_study_series.py"
    gather_script_path = project_root / "py_simulation" / "src" / "gather.py"
    plotting_script_path = project_root / "py_simulation" / "src" / "plotting.py"
    latex_tables_script_path = project_root / "py_simulation" / "src" / "latex_tables.py"

    # --- Simulation Phase ---
    if not args.skip_simulations:
        print("PHASE: Running Simulations...")
        for config_params in configs_to_run:
            print(f"\nStarting simulations for Config {config_params['config_num']}...")
            cmd_run_series = [
                python_executable, str(run_study_series_script_path),
                "--config_l", str(config_params["l"]),
                "--config_n_bottom", str(config_params["n_bottom"]),
                "--config_n_obs", str(config_params["n_obs"]),
                "--num_config", str(config_params["config_num"]),
                "--model_type", config_params["model_type"],
                # Assuming run_study_series.py defaults noise_type and alpha if not specified
                "--start_index", "0",
                "--end_index", str(config_params["num_sim_indices"]),
                # study_script_path in run_study_series.py defaults to "py_simulation/src/study.py"
                # which will be resolved relative to project_root if run_study_series is run with cwd=project_root
            ]
            print(f"Executing: {' '.join(cmd_run_series)}")
            try:
                subprocess.run(cmd_run_series, check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error during simulation phase for Config {config_params['config_num']}: {e}", file=sys.stderr)
                # Decide whether to stop or continue with other configs
                # sys.exit(1) # Option to stop
                print("Continuing with next configuration if any.", file=sys.stderr)
            except FileNotFoundError:
                 print(f"Error: Script {run_study_series_script_path} not found. Ensure paths are correct.", file=sys.stderr)
                 sys.exit(1)

    else:
        print("Skipping simulation phase.")

    # --- Evaluation Phase ---
    if not args.skip_evaluation:
        print("\nPHASE: Running Evaluation (Gathering, Plotting)...")
        for config_params in configs_to_run:
            print(f"\nStarting evaluation for Config {config_params['config_num']}...")

            # Gather results
            print(f"Gathering results for Config {config_params['config_num']}...")
            cmd_gather = [
                python_executable, str(gather_script_path),
                "--num_config", str(config_params["config_num"]),
                "--num_simulations", str(config_params["num_sim_indices"])
            ]
            print(f"Executing: {' '.join(cmd_gather)}")
            try:
                subprocess.run(cmd_gather, check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error during gathering phase for Config {config_params['config_num']}: {e}", file=sys.stderr)
                print("Continuing with plotting and other configurations if possible.", file=sys.stderr)
            except FileNotFoundError:
                 print(f"Error: Script {gather_script_path} not found. Ensure paths are correct.", file=sys.stderr)
                 sys.exit(1) # Critical for next steps

            # Generate plots
            print(f"Generating plots for Config {config_params['config_num']}...")
            cmd_plot = [
                python_executable, str(plotting_script_path),
                "--num_config", str(config_params["config_num"]),
                "--m_total_series", str(config_params["m_total_series"])
            ]
            print(f"Executing: {' '.join(cmd_plot)}")
            try:
                subprocess.run(cmd_plot, check=True, cwd=project_root)
            except subprocess.CalledProcessError as e:
                print(f"Error during plotting phase for Config {config_params['config_num']}: {e}", file=sys.stderr)
                print("Continuing with other configurations if possible.", file=sys.stderr)
            except FileNotFoundError:
                 print(f"Error: Script {plotting_script_path} not found. Ensure paths are correct.", file=sys.stderr)
                 # Not exiting, as table generation might still be possible or desired.

        # Final LaTeX Table Generation (after all specified configs are processed)
        print("\nGenerating final LaTeX tables...")
        cmd_latex = [python_executable, str(latex_tables_script_path)]
        print(f"Executing: {' '.join(cmd_latex)}")
        try:
            subprocess.run(cmd_latex, check=True, cwd=project_root)
        except subprocess.CalledProcessError as e:
            print(f"Error during LaTeX table generation: {e}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Error: Script {latex_tables_script_path} not found. Ensure paths are correct.", file=sys.stderr)

    else:
        print("Skipping evaluation phase.")

    print("\nMain pipeline finished.")

if __name__ == "__main__":
    main()

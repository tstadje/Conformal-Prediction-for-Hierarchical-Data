import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Assuming functions.py is in the same directory or src is in PYTHONPATH
try:
    from functions import generate_S, simulation_multi
except ImportError:
    # If running script directly from src, functions.py might not be found as a module
    # This is a common issue with relative imports if structure is not package-like
    # Or if script is moved/called from a different working directory.
    # For robustness, one might add parent dir to sys.path if needed,
    # but for now, assume it's run in a way that `from functions import ...` works
    # (e.g. `python -m py_simulation.src.study` from repo root, if __init__.py are set up)
    # Or if PYTHONPATH includes the project root.
    # For this specific environment, direct import should work if run from py_simulation/src.
    print("Error: Could not import from functions.py. Ensure it's in the same directory or PYTHONPATH is set.", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run a single simulation for hierarchical time series.")
    parser.add_argument("--simulation_index", type=int, required=True, help="Index of the current simulation run.")
    parser.add_argument("--config_l", type=int, required=True, help="Hierarchy levels (parameter l for generate_S).")
    parser.add_argument("--config_n_bottom", type=int, required=True, help="Number of bottom-level series (parameter n for generate_S).")
    parser.add_argument("--config_n_obs", type=int, required=True, help="Number of observations for the series.")
    parser.add_argument("--num_simulations", type=int, required=True, help="Total number of simulations in the batch (for context/logging).")
    parser.add_argument("--config_num", type=int, required=True, help="Configuration number (used for naming output directory).")
    parser.add_argument("--noise_type", type=str, default="A", choices=["A", "B"], help="Noise type for data generation (A or B).")
    parser.add_argument("--alpha", type=float, default=0.1, help="Significance level for conformal prediction.")
    parser.add_argument("--model_type", type=str, default="gam", choices=["gam", "lm"], help="Forecasting model type (gam or lm).")

    args = parser.parse_args()

    print(f"Running simulation {args.simulation_index + 1}/{args.num_simulations} for Config {args.config_num}")
    print(f"Parameters: l={args.config_l}, n_bottom={args.config_n_bottom}, n_obs={args.config_n_obs}")
    print(f"Model type: {args.model_type}, Noise type: {args.noise_type}, Alpha: {args.alpha}")

    # Create output directory relative to the script's execution path.
    # If study.py is in py_simulation/src and run from py_simulation/src: ./Config_X
    # If run from repo root as `python py_simulation/src/study.py`: ./Config_X
    # This matches R's behavior.
    output_dir = Path(f"./Config_{args.config_num}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    # Generate S matrix
    # Note: R's generate_S uses 'n' for bottom-level series count and 'l' for levels.
    # S_input_matrix has shape (m_total_series, n_bottom_level_series)
    try:
        S_matrix = generate_S(n=args.config_n_bottom, l=args.config_l)
    except ValueError as e:
        print(f"Error generating S matrix: {e}", file=sys.stderr)
        sys.exit(1)

    m_total_series = S_matrix.shape[0]
    print(f"S_matrix generated with shape: {S_matrix.shape} (m_total_series={m_total_series}, n_bottom_level_series={args.config_n_bottom})")

    # Run the simulation
    # simulation_multi returns [long_data_df, joint_results_dict]
    try:
        results_simu_list = simulation_multi(
            S_input_matrix=S_matrix,
            n_total_obs=args.config_n_obs,
            model_type=args.model_type,
            noise_type=args.noise_type,
            alpha=args.alpha
        )
    except Exception as e:
        print(f"Error during simulation_multi execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    marginal_results_df = results_simu_list[0]
    joint_results_dict = results_simu_list[1]

    # Add simulation index to results
    marginal_results_df['simulation'] = args.simulation_index

    joint_results_df_list = []
    for method, df in joint_results_dict.items():
        temp_df = df.copy()
        # 'method' column should already be in df from metrics_proba_multi if conformal functions add it,
        # or from how metrics_proba_multi processes its inputs.
        # Our metrics_proba_multi's joint_conformal call returns a df that doesn't have 'method'
        # but metrics_proba_multi stores it in a dict.
        # The R code `do.call(rbind, results_simu[[2]])` implies method name might be from rownames or a column.
        # Let's ensure 'method' is a column in the final joint DataFrame.
        if 'method' not in temp_df.columns: # If joint_conformal output doesn't include it
             temp_df['method'] = method
        temp_df['simulation'] = args.simulation_index
        joint_results_df_list.append(temp_df)

    all_joint_results_df = pd.concat(joint_results_df_list, ignore_index=True)

    # Save results as pickle files (alternative to R's RDS)
    try:
        marginal_filename = output_dir / f"simulation_{args.simulation_index}.pkl"
        marginal_results_df.to_pickle(marginal_filename)
        print(f"Marginal results saved to: {marginal_filename}")

        joint_filename = output_dir / f"simulation_joint_{args.simulation_index}.pkl"
        all_joint_results_df.to_pickle(joint_filename)
        print(f"Joint results saved to: {joint_filename}")
    except Exception as e:
        print(f"Error saving results to pickle files: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Simulation {args.simulation_index + 1} for Config {args.config_num} completed successfully.")

if __name__ == "__main__":
    main()

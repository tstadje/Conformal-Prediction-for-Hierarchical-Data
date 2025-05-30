import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Gather simulation results from a Config_X directory.")
    parser.add_argument("--num_config", type=int, required=True,
                        help="Configuration number (specifies the Config_X directory to read from).")
    parser.add_argument("--num_simulations", type=int, required=True,
                        help="Total number of simulation files to look for (e.g., from simulation_0.pkl to simulation_{num_simulations-1}.pkl).")

    args = parser.parse_args()

    print(f"Gathering results for Config_{args.num_config} across {args.num_simulations} simulation runs.")

    input_dir = Path(f"./Config_{args.num_config}")

    if not input_dir.is_dir():
        print(f"Error: Input directory {input_dir} does not exist. Please run study.py for this config first.", file=sys.stderr)
        sys.exit(1)

    df_list_marginal = []
    df_list_joint = []

    # Loop from 0 to num_simulations - 1, as file names are expected to be 0-indexed
    for i in range(args.num_simulations):
        marginal_file = input_dir / f"simulation_{i}.pkl"
        joint_file = input_dir / f"simulation_joint_{i}.pkl"

        marginal_found = False
        if marginal_file.exists():
            try:
                marginal_df = pd.read_pickle(marginal_file)
                df_list_marginal.append(marginal_df)
                marginal_found = True
            except Exception as e:
                print(f"Warning: Could not read marginal file {marginal_file}. Error: {e}. Skipping.", file=sys.stderr)
        else:
            print(f"Warning: Marginal file {marginal_file} not found. Skipping.", file=sys.stderr)

        if joint_file.exists():
            try:
                joint_df = pd.read_pickle(joint_file)
                df_list_joint.append(joint_df)
                # If marginal was found but joint wasn't, or vice-versa, it might indicate an incomplete run.
                # For now, we collect what we can. If marginal_found is False here, it's already printed.
            except Exception as e:
                print(f"Warning: Could not read joint file {joint_file}. Error: {e}. Skipping.", file=sys.stderr)
        elif marginal_found: # Only print missing joint if marginal was there (to avoid double warnings for a totally missing sim index)
             print(f"Warning: Joint file {joint_file} not found, but marginal file was present. Skipping joint part for sim {i}.", file=sys.stderr)


    if not df_list_marginal:
        print(f"Error: No marginal simulation dataframes were successfully read from {input_dir}. Aggregated file will not be created.", file=sys.stderr)
        # Depending on desired behavior, could exit or just skip saving marginal.
        # For now, let's allow saving empty/partial results if some parts are missing.
    else:
        results_proba_marginal = pd.concat(df_list_marginal, ignore_index=True)
        marginal_output_file = input_dir / "Final_Output.pkl"
        try:
            results_proba_marginal.to_pickle(marginal_output_file)
            print(f"Aggregated marginal results saved to: {marginal_output_file}")
        except Exception as e:
            print(f"Error saving aggregated marginal results to {marginal_output_file}: {e}", file=sys.stderr)


    if not df_list_joint:
        print(f"Error: No joint simulation dataframes were successfully read from {input_dir}. Aggregated file will not be created.", file=sys.stderr)
    else:
        results_proba_joint = pd.concat(df_list_joint, ignore_index=True)
        joint_output_file = input_dir / "Final_Output_joint.pkl"
        try:
            results_proba_joint.to_pickle(joint_output_file)
            print(f"Aggregated joint results saved to: {joint_output_file}")
        except Exception as e:
            print(f"Error saving aggregated joint results to {joint_output_file}: {e}", file=sys.stderr)

    print(f"Data gathering for Config_{args.num_config} completed.")

if __name__ == "__main__":
    main()

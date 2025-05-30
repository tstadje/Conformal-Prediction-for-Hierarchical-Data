import pandas as pd
import numpy as np
from pathlib import Path
import sys

try:
    from functions import get_table, get_table_occurence
except ImportError:
    print("Error: Could not import from functions.py. Ensure it's in the same directory or PYTHONPATH is set.", file=sys.stderr)
    sys.exit(1)

def main():
    # Define method names as they appear in the 'method' column of results
    # These will become the index for rows in summary tables before transposing for get_table
    method_names = ["Direct", "OLS", "WLS", "Combi", "MinT"]
    num_simulations_for_se = None # Will be determined from loaded data

    # --- Marginal Statistics Processing ---
    print("Processing Marginal Statistics...")
    # Initialize DataFrames to store aggregated results (means and SEs)
    # These will have methods as index and Configs as columns
    results_coverage_mean = pd.DataFrame(index=method_names)
    results_coverage_se = pd.DataFrame(index=method_names)
    results_length_sqrt_mean = pd.DataFrame(index=method_names) # For sqrt(Length)
    results_length_sqrt_se = pd.DataFrame(index=method_names)   # For SE of sqrt(Length)
    # results_length_mean = pd.DataFrame(index=method_names) # For actual Length (squared in R code)
    # results_length_se = pd.DataFrame(index=method_names)   # For SE of actual Length

    # R script processes configs 1 through 6 for marginal stats
    # Configs map to different n_bottom and l values, defined in the shell script that calls study.R
    # We assume these configs (1-6) exist as directories Config_1, Config_2, ...
    # And that these config numbers correspond to columns C1, C2, ... in LaTeX tables
    processed_configs_marginal = 0
    for config_idx in range(1, 7): # Config_1 to Config_6
        config_dir_name = f"Config_{config_idx}"
        config_col_name = f"C{config_idx}" # Column name for the summary table
        input_file = Path(f"./{config_dir_name}/Final_Output.pkl")

        if not input_file.exists():
            print(f"Warning: Marginal results file {input_file} not found. Skipping for marginal stats.", file=sys.stderr)
            # Add NaN columns to keep table structure consistent if a config is missing
            results_coverage_mean[config_col_name] = np.nan
            results_coverage_se[config_col_name] = np.nan
            results_length_sqrt_mean[config_col_name] = np.nan
            results_length_sqrt_se[config_col_name] = np.nan
            continue

        try:
            df_marginal = pd.read_pickle(input_file)
            processed_configs_marginal +=1
        except Exception as e:
            print(f"Warning: Could not read {input_file}. Error: {e}. Skipping for marginal stats.", file=sys.stderr)
            results_coverage_mean[config_col_name] = np.nan
            results_coverage_se[config_col_name] = np.nan
            results_length_sqrt_mean[config_col_name] = np.nan
            results_length_sqrt_se[config_col_name] = np.nan
            continue

        if num_simulations_for_se is None and 'simulation' in df_marginal.columns:
            num_simulations_for_se = df_marginal['simulation'].nunique()
            if num_simulations_for_se == 0 : num_simulations_for_se = 1 # Avoid division by zero if somehow 0
            print(f"Determined num_simulations_for_se = {num_simulations_for_se} from the first loaded file.")


        # Calculate metrics per method for this config
        # Coverage
        coverage_stats = df_marginal[df_marginal['metric'] == 'Coverage'].groupby('method')['Value'].agg(
            Mean='mean',
            Std='std',
            N='count' # N should be num_simulations * num_series for coverage if not averaged per simulation first
        )
        # The R code calculates SE based on N_simulation.
        # The 'Value' in Final_Output.pkl for Coverage is already averaged per series for that simulation run.
        # So, N here should be number of simulations for that method.
        # If 'Value' is coverage for a single series, then N is num_simulations * num_series.
        # Assuming 'Value' for coverage metric is already per-series, averaged over test set.
        # The grouping by (simulation, method, Series) then averaging is done before this script.
        # The `Final_Output.pkl` has columns 'metric', 'method', 'Series', 'Value', 'simulation'.
        # So, we need to average over 'Series' for each simulation first, then average over simulations.

        # Recalculating N for SE based on number of simulations.
        # R code example: `SE_Coverage = qt(0.975,N_simulation-1)*Std_Coverage/sqrt(N_simulation)`
        # Using 1.96 for ~qt(0.975, large_df)
        n_sim_this_file = df_marginal['simulation'].nunique() if 'simulation' in df_marginal.columns else 1

        # Calculate Mean and SE for Coverage
        # Group by method and simulation, then average over series, then average over simulations
        avg_coverage_per_sim = df_marginal[df_marginal['metric'] == 'Coverage'].groupby(['method', 'simulation'])['Value'].mean()
        mean_coverage = avg_coverage_per_sim.groupby('method').mean()
        std_coverage = avg_coverage_per_sim.groupby('method').std(ddof=1)
        results_coverage_mean[config_col_name] = mean_coverage
        results_coverage_se[config_col_name] = 1.96 * std_coverage / np.sqrt(n_sim_this_file)


        # Length (sqrt of mean squared length from conformal(), then handle SE)
        # R code stores 'Length' as (q_upper - q_lower)^2.
        # For sqrt_Mean_Length table: sqrt(mean of these squared lengths)
        # For sqrt_SE_Length table: sqrt(SE of these squared lengths) - this is unusual. Typically SE of sqrt(length).

        df_length_values = df_marginal[df_marginal['metric'] == 'Length']['Value'] # These are squared lengths
        df_length_values_sqrt = np.sqrt(df_length_values) # Take sqrt for each individual length value

        # Add method and simulation columns back for grouping
        temp_df_sqrt_length = pd.DataFrame({
            'Value': df_length_values_sqrt,
            'method': df_marginal.loc[df_length_values_sqrt.index, 'method'],
            'simulation': df_marginal.loc[df_length_values_sqrt.index, 'simulation']
        })

        avg_sqrt_length_per_sim = temp_df_sqrt_length.groupby(['method', 'simulation'])['Value'].mean()
        mean_sqrt_length = avg_sqrt_length_per_sim.groupby('method').mean()
        std_sqrt_length = avg_sqrt_length_per_sim.groupby('method').std(ddof=1)

        results_length_sqrt_mean[config_col_name] = mean_sqrt_length
        results_length_sqrt_se[config_col_name] = 1.96 * std_sqrt_length / np.sqrt(n_sim_this_file)


    # Generate LaTeX tables for marginal stats
    if processed_configs_marginal > 0:
        with open("LaTeX_file.txt", "w") as f:
            f.write("% LaTeX Table for sqrt(Length) - Mean and SE\n")
            # Transpose because get_table expects configs in rows, methods in columns (like R xtable)
            # but our summary DFs have methods in rows, configs in columns.
            latex_sqrt_length_table = get_table(
                results_length_sqrt_mean.T,
                uncertainty_df=results_length_sqrt_se.T,
                bold=True,
                digits=2
            )
            f.write(latex_sqrt_length_table)
            f.write("\n\n")

            f.write("% LaTeX Table for Coverage - Mean and SE\n")
            latex_coverage_table = get_table(
                results_coverage_mean.T,
                uncertainty_df=results_coverage_se.T,
                bold=False, # R script had bold=FALSE for coverage
                digits=2
            )
            f.write(latex_coverage_table)
            f.write("\n")
        print("Marginal statistics LaTeX tables written to LaTeX_file.txt")
    else:
        print("No marginal data processed, LaTeX_file.txt not generated.")

    # --- Third table (get_table_occurence) from R script ---
    # This part depends on `results_length_individuals` which is not directly
    # generated by the `gather.py` script from `simulation_{i}.pkl`.
    # It seems to imply a different level of aggregation or specific data structure.
    # For now, this part is omitted as the input data source is unclear.
    print("\nSkipping 'occurrence' table generation as input 'results_length_individuals' source is not defined in gather.py.")


    # --- Joint Statistics Processing ---
    print("\nProcessing Joint Statistics...")
    results_joint_volume_mean = pd.DataFrame(index=method_names)
    results_joint_volume_se = pd.DataFrame(index=method_names)
    results_joint_coverage_mean = pd.DataFrame(index=method_names)
    results_joint_coverage_se = pd.DataFrame(index=method_names)

    # R script processes a subset of configs for joint stats, e.g., n=12 (Config_1) and specific l values (Config_3, Config_4)
    # Let's assume Config_1 and Config_2 for this example, adjust as needed.
    # The R code used `n_vector = c(12, 24, 36, 144, 144, 144)` and `l_vector = c(1, 1, 1, 1, 2, 3)`
    # Joint tables were for `n=12` and `l=1,2,3`.
    # Assuming Config_1 (n=12, l=1), Config_5 (n=144, l=2, map to C2 for table), Config_6 (n=144, l=3, map to C3 for table)
    # This mapping is based on the R script's comments for joint tables.
    # For simplicity, let's use Config_1, Config_2, Config_3 as placeholders for joint table columns.
    # This part needs to align with how configs are actually numbered and what they represent.
    # R code uses: C1 for (n=12,l=1), C2 for (n=12,l=2), C3 for (n=12,l=3) for joint.
    # This means we need to know which Config_X corresponds to (n=12, l=2) and (n=12, l=3).
    # Let's assume Config_1 is (n=12,l=1), Config_X1 is (n=12,l=2), Config_X2 is (n=12,l=3).
    # For this example, I'll use Config_1, Config_2, Config_3 as stand-ins.

    joint_config_mapping = {"C1": 1, "C2": 2, "C3": 3} # Example: Table Col C1 uses Config_1 data
    processed_configs_joint = 0

    for table_col_name, config_idx in joint_config_mapping.items():
        config_dir_name = f"Config_{config_idx}"
        input_file_joint = Path(f"./{config_dir_name}/Final_Output_joint.pkl")

        if not input_file_joint.exists():
            print(f"Warning: Joint results file {input_file_joint} not found. Skipping for joint stats.", file=sys.stderr)
            results_joint_volume_mean[table_col_name] = np.nan
            results_joint_volume_se[table_col_name] = np.nan
            results_joint_coverage_mean[table_col_name] = np.nan
            results_joint_coverage_se[table_col_name] = np.nan
            continue

        try:
            df_joint = pd.read_pickle(input_file_joint)
            processed_configs_joint +=1
        except Exception as e:
            print(f"Warning: Could not read {input_file_joint}. Error: {e}. Skipping for joint stats.", file=sys.stderr)
            results_joint_volume_mean[table_col_name] = np.nan
            results_joint_volume_se[table_col_name] = np.nan
            results_joint_coverage_mean[table_col_name] = np.nan
            results_joint_coverage_se[table_col_name] = np.nan
            continue

        n_sim_this_file_joint = df_joint['simulation'].nunique() if 'simulation' in df_joint.columns else 1

        # Volume
        mean_volume = df_joint[df_joint['metric'] == 'Volume'].groupby('method')['V'].mean()
        std_volume = df_joint[df_joint['metric'] == 'Volume'].groupby('method')['V'].std(ddof=1)
        results_joint_volume_mean[table_col_name] = mean_volume
        results_joint_volume_se[table_col_name] = 1.96 * std_volume / np.sqrt(n_sim_this_file_joint)

        # Joint Coverage
        mean_joint_coverage = df_joint[df_joint['metric'] == 'Coverage'].groupby('method')['V'].mean()
        std_joint_coverage = df_joint[df_joint['metric'] == 'Coverage'].groupby('method')['V'].std(ddof=1)
        results_joint_coverage_mean[table_col_name] = mean_joint_coverage
        results_joint_coverage_se[table_col_name] = 1.96 * std_joint_coverage / np.sqrt(n_sim_this_file_joint)

    if processed_configs_joint > 0:
        with open("LaTeX_joint_file.txt", "w") as f:
            f.write("% LaTeX Table for Joint Volume - Mean and SE\n")
            latex_joint_volume_table = get_table(
                results_joint_volume_mean.T,
                uncertainty_df=results_joint_volume_se.T,
                bold=True,
                digits=2
            )
            f.write(latex_joint_volume_table)
            f.write("\n\n")

            f.write("% LaTeX Table for Joint Coverage - Mean and SE\n")
            latex_joint_coverage_table = get_table(
                results_joint_coverage_mean.T,
                uncertainty_df=results_joint_coverage_se.T,
                bold=False,
                digits=2
            )
            f.write(latex_joint_coverage_table)
            f.write("\n")
        print("Joint statistics LaTeX tables written to LaTeX_joint_file.txt")
    else:
        print("No joint data processed, LaTeX_joint_file.txt not generated.")

    print("\nLaTeX table generation process finished.")

if __name__ == "__main__":
    main()

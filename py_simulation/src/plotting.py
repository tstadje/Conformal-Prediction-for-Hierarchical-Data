import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Generate Coverage vs. Length plots for simulation results.")
    parser.add_argument("--num_config", type=int, required=True,
                        help="Configuration number (specifies Config_X input directory and output subdirectory name).")
    parser.add_argument("--m_total_series", type=int, required=True,
                        help="Total number of series (m) in this configuration, for iterating V1 to Vm.")
    parser.add_argument("--alpha_value", type=float, default=0.1,
                        help="Alpha value for the reference line on the plot (1-alpha).")

    args = parser.parse_args()

    print(f"Generating plots for Config_{args.num_config}, m_total_series={args.m_total_series}")

    input_file = Path(f"./Config_{args.num_config}/Final_Output.pkl")
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found. Please run gather.py for this config first.", file=sys.stderr)
        sys.exit(1)

    try:
        df_results = pd.read_pickle(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Output directory structure: Plot/Config_X/
    plot_output_dir = Path(f"./Plot/Config_{args.num_config}")
    plot_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Plot output directory: {plot_output_dir.resolve()}")

    # Aesthetics for plotting (colors and markers)
    method_aesthetics = {
        "Direct": {"color": "black", "marker": "o"},
        "OLS": {"color": "blue", "marker": "s"},
        "WLS": {"color": "red", "marker": "^"},
        "Combi": {"color": "purple", "marker": "D"},
        "MinT": {"color": "green", "marker": "x"},
    }

    # Determine N_simulation for SE calculation
    # Assuming 'simulation' column exists and indicates distinct simulation runs
    if 'simulation' not in df_results.columns:
        print("Error: 'simulation' column not found in results. Cannot determine N for SE calculation.", file=sys.stderr)
        sys.exit(1)

    N_simulation = df_results['simulation'].nunique()
    if N_simulation == 0:
        print("Error: No simulation runs found in the data (N_simulation is 0).", file=sys.stderr)
        sys.exit(1)
    print(f"Number of simulations found: {N_simulation}")


    for j in range(1, args.m_total_series + 1):
        series_name = f"V{j}"
        print(f"Processing and plotting for series: {series_name}")

        df_series = df_results[df_results['Series'] == series_name] # R: variable == paste0("V", j)

        if df_series.empty:
            print(f"Warning: No data found for series {series_name}. Skipping plot.", file=sys.stderr)
            continue

        # Aggregate metrics (Coverage, Length) by method
        # 'Value' column contains the metric value for a given (simulation, method, series, metric)
        # We need to average these 'Value's over simulations for each (method, metric) for this series

        summary_stats_list = []
        for method_name in method_aesthetics.keys():
            df_method_series = df_series[df_series['method'] == method_name]
            if df_method_series.empty:
                continue

            # Coverage stats
            coverage_values = df_method_series[df_method_series['metric'] == 'Coverage']['Value']
            mean_coverage = coverage_values.mean()
            std_coverage = coverage_values.std(ddof=1)
            se_coverage = 1.96 * std_coverage / np.sqrt(N_simulation) if N_simulation > 0 else np.nan

            # Length stats (original 'Length' metric is squared length)
            length_sq_values = df_method_series[df_method_series['metric'] == 'Length']['Value']
            mean_length_sq = length_sq_values.mean()
            std_length_sq = length_sq_values.std(ddof=1)
            se_length_sq = 1.96 * std_length_sq / np.sqrt(N_simulation) if N_simulation > 0 else np.nan

            # Transform length: sqrt(Mean_Length_Sq) and sqrt(SE_Length_Sq) as per R script
            sqrt_mean_length = np.sqrt(mean_length_sq) if mean_length_sq >= 0 else np.nan
            # R code: `sqrt(SE_Length)`. If SE_Length_Sq can be negative (e.g. due to issues), handle sqrt.
            sqrt_se_length = np.sqrt(se_length_sq) if se_length_sq >= 0 else np.nan

            summary_stats_list.append({
                'method': method_name,
                'Mean_Coverage': mean_coverage,
                'SE_Coverage': se_coverage,
                'sqrt_Mean_Length': sqrt_mean_length,
                'sqrt_SE_Length': sqrt_se_length
            })

        if not summary_stats_list:
            print(f"Warning: No summary statistics generated for series {series_name}. Skipping plot.", file=sys.stderr)
            continue

        summary_df = pd.DataFrame(summary_stats_list)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 7))

        for idx, row in summary_df.iterrows():
            method = row['method']
            aesthetics = method_aesthetics.get(method, {"color": "gray", "marker": "."}) # Default if method not in dict

            ax.errorbar(
                x=row['sqrt_Mean_Length'],
                y=row['Mean_Coverage'],
                xerr=row['sqrt_SE_Length'],
                yerr=row['SE_Coverage'],
                label=method,
                fmt=aesthetics["marker"], # Marker only for the point
                color=aesthetics["color"],
                capsize=3, # Error bar caps
                linestyle='None' # No line connecting points for errorbar itself
            )

        # Vertical line for target coverage
        target_coverage_line = (1 - args.alpha_value) # R: `100*(1-alpha)` but coverage usually 0-1 scale
        ax.axvline(x=target_coverage_line, linestyle='--', color='gray', label=f'{100*(1-args.alpha_value):.0f}% Target Coverage (Reference)')
        # The R code places this line on X-axis if plotting Coverage vs Length.
        # Here, it's a vertical line at `target_coverage_line` if X is Coverage.
        # If X is Length, Y is Coverage, then it should be `ax.axhline(y=target_coverage_line, ...)`
        # The R code `geom_vline(xintercept = 100*(1-alpha))` means x-axis is coverage.
        # My interpretation: Y is Coverage, X is sqrt(Length). So it's axhline.

        # Correcting: R plot `aes(y=Mean_Coverage, x=sqrt_Mean_Length)`. `geom_vline(xintercept=...)` is on x-axis.
        # This means their x-axis is NOT length.
        # R code has `ggplot(aes(y=Mean_Coverage*100, x=sqrt_Mean_Length*100))`.
        # And `geom_vline(xintercept = 100*(1-alpha))`
        # This means they might have swapped axes or the vline is for a different metric.
        # Let's assume the plot is Coverage (Y) vs sqrt(Length) (X).
        # A common reference is for coverage to be at (1-alpha). So horizontal line.
        ax.axhline(y=target_coverage_line, linestyle='--', color='gray', label=f'{100*(1-args.alpha_value):.0f}% Target Coverage')


        ax.set_xlabel("sqrt(Interval Length)")
        ax.set_ylabel("Coverage")
        ax.set_title(f"Coverage vs. sqrt(Length) for Series {series_name} (Config {args.num_config})")

        # Set Y-axis limits for coverage (typically 0 to 1, or slightly more for error bars)
        ax.set_ylim(min(0, summary_df['Mean_Coverage'].min() - summary_df['SE_Coverage'].max() - 0.05),
                    max(1, summary_df['Mean_Coverage'].max() + summary_df['SE_Coverage'].max() + 0.05) if not summary_df['Mean_Coverage'].isnull().all() else 1.1)

        # Optional: Set X-axis limits based on data or R script's coord_cartesian if values are comparable
        # R: `coord_cartesian(xlim = c(0, 100), ylim = c(0,100))`
        # This implies their sqrt_Mean_Length and Mean_Coverage were scaled to 0-100.
        # Our values are 0-1 for coverage. sqrt_Mean_Length depends on data scale.
        # Let's keep x-axis auto-scaled for now unless specific limits are derived.

        ax.legend(title="Method")
        ax.grid(True, linestyle=':', alpha=0.7)

        plot_filename = plot_output_dir / f"Config_{args.num_config}_CovLength_{series_name}.pdf"
        try:
            plt.savefig(plot_filename, format="pdf", bbox_inches="tight")
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot {plot_filename}: {e}", file=sys.stderr)

        plt.close(fig) # Close the figure to free memory

    print(f"Plotting completed for Config_{args.num_config}.")

if __name__ == "__main__":
    main()

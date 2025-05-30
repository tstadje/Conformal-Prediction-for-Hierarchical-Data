import pytest
import subprocess
import sys
import pandas as pd
from pathlib import Path
import shutil
import os # For modifying main_pipeline.py content for test configuration

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MAIN_PIPELINE_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "main_pipeline.py"
PYTHON_EXECUTABLE = sys.executable

# Test-specific configuration numbers
TEST_CONFIG_NUM_MAIN_FULL = 998
TEST_CONFIG_NUM_MAIN_SIM_ONLY = 997
TEST_CONFIG_NUM_MAIN_EVAL_ONLY = 996

# Parameters for the test configurations that main_pipeline.py will run
# These need to be defined *inside* main_pipeline.py's config list for the test to work with --configs flag.
# For l=1, n_bottom=12: m = 1 (total) + 12^1 (agg level 1 groups) + 12 (bottom) = 25
M_TOTAL_SERIES_FOR_TEST_CONFIG = 25
TEST_L = 1
TEST_N_BOTTOM = 12
TEST_N_OBS_MAIN = 60 # Small for pipeline tests
TEST_NUM_SIM_INDICES_MAIN = 1 # Run only one simulation for speed

# Fixture to manage test directories and modify main_pipeline.py for test configs
@pytest.fixture(scope="module")
def setup_main_pipeline_tests():
    original_main_pipeline_content = None
    main_pipeline_file = MAIN_PIPELINE_SCRIPT_PATH

    # Define test configurations string to be inserted into main_pipeline.py
    # This ensures that when main_pipeline.py is called with --configs 998 etc., it knows what they are.
    test_configs_string_for_injection = f"""
    all_configs_data_template = [
        {{"l": {TEST_L}, "n_bottom": {TEST_N_BOTTOM}, "n_obs": {TEST_N_OBS_MAIN}, "num_sim_indices": {TEST_NUM_SIM_INDICES_MAIN}, "config_num": {TEST_CONFIG_NUM_MAIN_FULL}}},
        {{"l": {TEST_L}, "n_bottom": {TEST_N_BOTTOM}, "n_obs": {TEST_N_OBS_MAIN}, "num_sim_indices": {TEST_NUM_SIM_INDICES_MAIN}, "config_num": {TEST_CONFIG_NUM_MAIN_SIM_ONLY}}},
        {{"l": {TEST_L}, "n_bottom": {TEST_N_BOTTOM}, "n_obs": {TEST_N_OBS_MAIN}, "num_sim_indices": {TEST_NUM_SIM_INDICES_MAIN}, "config_num": {TEST_CONFIG_NUM_MAIN_EVAL_ONLY}}},
        # Add any other minimal configs needed by default if --configs is not used by a test (not current case)
    ]
"""
    try:
        # Read original content and backup
        original_main_pipeline_content = main_pipeline_file.read_text()

        # Modify main_pipeline.py to use these test configurations
        # This is done by finding the line `all_configs_data_template = [` and replacing it and subsequent lines defining the list.
        lines = original_main_pipeline_content.splitlines()
        start_replace_idx = -1
        end_replace_idx = -1
        for i, line in enumerate(lines):
            if "all_configs_data_template = [" in line:
                start_replace_idx = i
            if start_replace_idx != -1 and "]" in line and i > start_replace_idx : # end of list
                end_replace_idx = i
                break

        if start_replace_idx != -1 and end_replace_idx != -1:
            new_lines = lines[:start_replace_idx] + test_configs_string_for_injection.strip().splitlines() + lines[end_replace_idx+1:]
            main_pipeline_file.write_text("\n".join(new_lines))
            print(f"Temporarily modified {main_pipeline_file} with test configurations.")
        else:
            pytest.fail(f"Could not find 'all_configs_data_template' list in {main_pipeline_file} to modify for tests.")

    except Exception as e:
        pytest.fail(f"Failed to set up test configurations in {main_pipeline_file}: {e}")


    # Directory cleanup logic
    dirs_to_clean_before_and_after = [
        PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_FULL}",
        PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_SIM_ONLY}",
        PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_EVAL_ONLY}",
        PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_FULL}",
        PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_SIM_ONLY}", # Though skip-eval might not create
        PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_EVAL_ONLY}",
    ]

    for test_dir in dirs_to_clean_before_and_after:
        if test_dir.exists():
            shutil.rmtree(test_dir)

    yield # Tests run here

    # Restore original main_pipeline.py content
    if original_main_pipeline_content is not None:
        main_pipeline_file.write_text(original_main_pipeline_content)
        print(f"Restored original content of {main_pipeline_file}.")

    # Final cleanup of directories and files
    for test_dir in dirs_to_clean_before_and_after:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            # Clean parent Plot/ directory if it becomes empty
            if test_dir.parent.name == "Plot" and not any(test_dir.parent.iterdir()):
                try:
                    test_dir.parent.rmdir()
                except OSError: pass


    for tex_file_name in ["LaTeX_file.txt", "LaTeX_joint_file.txt"]:
        tex_file_path = PROJECT_ROOT / tex_file_name
        if tex_file_path.exists():
            tex_file_path.unlink()


class TestMainPipelineScript:
    def test_main_pipeline_full_run_minimal_config(self, setup_main_pipeline_tests):
        cmd = [
            PYTHON_EXECUTABLE, str(MAIN_PIPELINE_SCRIPT_PATH),
            "--configs", str(TEST_CONFIG_NUM_MAIN_FULL),
            "--model-type", "lm" # Use lm for faster testing
        ]

        print(f"Running command for TestMainPipelineScript (full run): {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"main_pipeline.py full run failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        config_dir = PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_FULL}"
        assert (config_dir / f"simulation_0.pkl").exists(), "Simulation output file not found."
        assert (config_dir / "Final_Output.pkl").exists(), "Aggregated marginal output not found."
        assert (config_dir / "Final_Output_joint.pkl").exists(), "Aggregated joint output not found."

        plot_dir = PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_FULL}"
        assert plot_dir.is_dir(), "Plot directory not created."
        # Expected PDFs = M_TOTAL_SERIES_FOR_TEST_CONFIG
        assert len(list(plot_dir.glob("*.pdf"))) == M_TOTAL_SERIES_FOR_TEST_CONFIG, "Incorrect number of plot files."

        assert (PROJECT_ROOT / "LaTeX_file.txt").exists(), "LaTeX_file.txt not created."
        assert (PROJECT_ROOT / "LaTeX_joint_file.txt").exists(), "LaTeX_joint_file.txt not created."

    def test_main_pipeline_skip_simulations(self, setup_main_pipeline_tests):
        eval_config_dir = PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_EVAL_ONLY}"
        eval_config_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal dummy data for gather.py and subsequent steps
        num_dummy_sims = TEST_NUM_SIM_INDICES_MAIN # Should match what main_pipeline expects for this config
        for i in range(num_dummy_sims):
            # Dummy marginal data: Needs 'metric', 'method', 'Series', 'Value', 'simulation'
            pd.DataFrame({
                'metric':['Coverage', 'Length'] * M_TOTAL_SERIES_FOR_TEST_CONFIG,
                'method':['Direct'] * 2 * M_TOTAL_SERIES_FOR_TEST_CONFIG,
                'Series':np.repeat([f'V{s+1}' for s in range(M_TOTAL_SERIES_FOR_TEST_CONFIG)], 2),
                'Value':[0.9, 1.0] * M_TOTAL_SERIES_FOR_TEST_CONFIG,
                'simulation':[i] * 2 * M_TOTAL_SERIES_FOR_TEST_CONFIG
            }).to_pickle(eval_config_dir / f"simulation_{i}.pkl")

            # Dummy joint data: Needs 'metric', 'method', 'V', 'simulation'
            pd.DataFrame({
                'metric':['Coverage', 'Volume', 'Quantile'],
                'method':['Direct'] * 3, # This column is added by metrics_proba_multi logic
                'V':[0.85, 10.0, 0.9],
                'simulation':[i] * 3
            }).to_pickle(eval_config_dir / f"simulation_joint_{i}.pkl")

        cmd = [
            PYTHON_EXECUTABLE, str(MAIN_PIPELINE_SCRIPT_PATH),
            "--configs", str(TEST_CONFIG_NUM_MAIN_EVAL_ONLY),
            "--skip-simulations" # This implies model-type is irrelevant for this run
        ]
        print(f"Running command for TestMainPipelineScript (skip simulations): {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"main_pipeline.py --skip-simulations failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        # Verify that new simulation files were NOT created (e.g., check for a sim index beyond dummy files)
        # This is tricky if num_sim_indices in main_pipeline's config for 996 is same as num_dummy_sims.
        # The check is that evaluation steps *did* run.
        assert (eval_config_dir / "Final_Output.pkl").exists(), "Evaluation (gather) did not run for marginal."
        assert (eval_config_dir / "Final_Output_joint.pkl").exists(), "Evaluation (gather) did not run for joint."
        assert (PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_EVAL_ONLY}").is_dir(), "Plotting did not run."
        # LaTeX files are global, so their existence means the step ran.
        assert (PROJECT_ROOT / "LaTeX_file.txt").exists(), "LaTeX_file.txt not created in skip-simulations run."

    def test_main_pipeline_skip_evaluation(self, setup_main_pipeline_tests):
        cmd = [
            PYTHON_EXECUTABLE, str(MAIN_PIPELINE_SCRIPT_PATH),
            "--configs", str(TEST_CONFIG_NUM_MAIN_SIM_ONLY),
            "--model-type", "lm",
            "--skip-evaluation"
        ]
        print(f"Running command for TestMainPipelineScript (skip evaluation): {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"main_pipeline.py --skip-evaluation failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        config_dir = PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM_MAIN_SIM_ONLY}"
        assert (config_dir / f"simulation_0.pkl").exists(), "Simulations did not run." # Checks for one sim file

        assert not (config_dir / "Final_Output.pkl").exists(), "Gathering should have been skipped."
        assert not (PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM_MAIN_SIM_ONLY}").exists(), "Plotting should have been skipped."
        assert not (PROJECT_ROOT / "LaTeX_file.txt").exists(), "LaTeX generation should have been skipped."
```

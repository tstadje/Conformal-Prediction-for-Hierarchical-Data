import pytest
import subprocess
import sys
import pandas as pd
from pathlib import Path
import shutil # For cleaning up directories

# Determine project root to construct paths to scripts
# Assuming this test file is in py_simulation/tests/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STUDY_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "study.py"
RUN_SERIES_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "run_study_series.py"
PYTHON_EXECUTABLE = sys.executable

# Define a test-specific configuration number to avoid conflicts
TEST_CONFIG_NUM = 999
TEST_CONFIG_DIR = PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM}"

# Minimal parameters for quick test runs
TEST_L = 1 # Using l=1 which is simpler and used in some actual configs
TEST_N_BOTTOM = 12
TEST_N_OBS = 60 # Small, but enough for train/valid/calib/test splits (0.4, 0.2, 0.2, 0.2)
                # 60*0.4=24 (train), 60*0.2=12 (valid), 12 (calib), 12 (test)
                # Needs to be large enough for model fitting and splits in simulation_multi

@pytest.fixture(scope="module")
def setup_teardown_test_config_dir():
    # Setup: ensure the test config directory is clean before tests in this module
    if TEST_CONFIG_DIR.exists():
        shutil.rmtree(TEST_CONFIG_DIR)
    # The scripts under test are expected to create this directory.

    yield # This is where the tests will run

    # Teardown: remove the test config directory after all tests in the module are done
    if TEST_CONFIG_DIR.exists():
        # print(f"Tearing down {TEST_CONFIG_DIR}") # Optional: for debugging test runs
        shutil.rmtree(TEST_CONFIG_DIR)

class TestStudyScript:
    def test_study_py_runs_and_creates_output(self, setup_teardown_test_config_dir):
        sim_idx = 0
        # num_simulations arg for study.py is for context, e.g. total runs in this config for logging
        num_simulations_total_for_study = 1
        cmd = [
            PYTHON_EXECUTABLE, str(STUDY_SCRIPT_PATH),
            "--config_l", str(TEST_L),
            "--config_n_bottom", str(TEST_N_BOTTOM),
            "--config_n_obs", str(TEST_N_OBS),
            "--num_config", str(TEST_CONFIG_NUM),
            "--simulation_index", str(sim_idx),
            "--num_simulations", str(num_simulations_total_for_study),
            "--model_type", "lm", # Use lm for faster test
            "--alpha", "0.1",
            "--noise_type", "A"
        ]

        print(f"Running command for TestStudyScript: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        assert result.returncode == 0, f"study.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert TEST_CONFIG_DIR.exists(), "Config directory was not created by study.py"

        expected_marginal_file = TEST_CONFIG_DIR / f"simulation_{sim_idx}.pkl"
        expected_joint_file = TEST_CONFIG_DIR / f"simulation_joint_{sim_idx}.pkl"

        assert expected_marginal_file.exists(), f"Marginal results pickle file not found: {expected_marginal_file}"
        assert expected_joint_file.exists(), f"Joint results pickle file not found: {expected_joint_file}"

        # Basic check of pickle content
        marginal_df = pd.read_pickle(expected_marginal_file)
        joint_df = pd.read_pickle(expected_joint_file)

        assert not marginal_df.empty
        assert "metric" in marginal_df.columns and "method" in marginal_df.columns
        assert "Series" in marginal_df.columns and "Value" in marginal_df.columns

        assert not joint_df.empty
        assert "metric" in joint_df.columns and "method" in joint_df.columns
        assert "V" in joint_df.columns # 'V' for value in joint results

class TestRunStudySeriesScript:
    def test_run_study_series_py_runs_multiple(self, setup_teardown_test_config_dir):
        # This test relies on the directory potentially existing from TestStudyScript
        # or being cleaned by the module-scoped fixture if run first.
        # If TestStudyScript ran first, Config_999/simulation_0.pkl might exist.
        # run_study_series.py will call study.py which overwrites files if they exist.
        start_idx = 0
        end_idx = 2 # Runs for sim_idx 0 and 1. Total 2 simulations.

        cmd = [
            PYTHON_EXECUTABLE, str(RUN_SERIES_SCRIPT_PATH),
            "--config_l", str(TEST_L),
            "--config_n_bottom", str(TEST_N_BOTTOM),
            "--config_n_obs", str(TEST_N_OBS),
            "--num_config", str(TEST_CONFIG_NUM),
            "--model_type", "lm",
            "--start_index", str(start_idx),
            "--end_index", str(end_idx)
            # alpha, noise_type use defaults in run_study_series.py if not passed to study.py
            # study_script_path and python_executable use defaults in run_study_series.py
        ]

        print(f"Running command for TestRunStudySeriesScript: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        assert result.returncode == 0, f"run_study_series.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        assert TEST_CONFIG_DIR.exists(), "Config directory was not created by run_study_series.py (or was removed)"

        for sim_idx_check in range(start_idx, end_idx):
            expected_marginal_file = TEST_CONFIG_DIR / f"simulation_{sim_idx_check}.pkl"
            expected_joint_file = TEST_CONFIG_DIR / f"simulation_joint_{sim_idx_check}.pkl"
            assert expected_marginal_file.exists(), f"Marginal file for sim_idx {sim_idx_check} not found"
            assert expected_joint_file.exists(), f"Joint file for sim_idx {sim_idx_check} not found"

            if sim_idx_check == start_idx: # Basic check for one of the generated files
                marginal_df = pd.read_pickle(expected_marginal_file)
                assert not marginal_df.empty
                joint_df = pd.read_pickle(expected_joint_file)
                assert not joint_df.empty

# Note on fixture scope:
# If TestStudyScript creates simulation_0.pkl and then TestRunStudySeriesScript runs for 0 and 1,
# simulation_0.pkl will be overwritten. This is acceptable for these tests as they primarily check
# for existence and basic readability, not specific content that might differ if not overwritten.
# The module-scoped fixture ensures Config_999 is clean before this module's tests start
# and is removed after they all finish.
```

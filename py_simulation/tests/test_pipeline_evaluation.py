import pytest
import subprocess
import sys
import pandas as pd
from pathlib import Path
import shutil

# Determine project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GATHER_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "gather.py"
PLOTTING_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "plotting.py"
LATEX_TABLES_SCRIPT_PATH = PROJECT_ROOT / "py_simulation" / "src" / "latex_tables.py"
PYTHON_EXECUTABLE = sys.executable

TEST_CONFIG_NUM = 999 # Must match what test_pipeline_simulation.py uses
TEST_CONFIG_DIR = PROJECT_ROOT / f"Config_{TEST_CONFIG_NUM}"
# Must match the number of simulation files created by test_run_study_series_py_runs_multiple
NUM_SIMULATIONS_IN_TEST_CONFIG = 2

# For plotting, m_total_series for TEST_CONFIG_NUM (l=1, n_bottom=12 for Config_999 in simulation tests)
# m = 1 (total) + 12^1 (agg level 1 groups) + 12 (bottom) = 1 + 12 + 12 = 25
M_TOTAL_SERIES_FOR_TEST_CONFIG = 25

@pytest.fixture(scope="module")
def ensure_simulation_output_exists_for_evaluation():
    """
    Checks if the output from test_pipeline_simulation.py (Config_999) exists.
    If not, skips these evaluation tests.
    This fixture doesn't create data itself to avoid complex duplication of simulation logic.
    It relies on the simulation tests having run successfully and their teardown
    (module-scoped) not having removed the data yet if run in the same session.
    If run in a separate session, this data would need to be manually generated or
    this fixture enhanced.
    """
    all_files_present = True
    if not TEST_CONFIG_DIR.is_dir():
        all_files_present = False
    else:
        for i in range(NUM_SIMULATIONS_IN_TEST_CONFIG):
            if not (TEST_CONFIG_DIR / f"simulation_{i}.pkl").exists():
                all_files_present = False
                break
            if not (TEST_CONFIG_DIR / f"simulation_joint_{i}.pkl").exists():
                all_files_present = False
                break

    if not all_files_present:
        pytest.skip(f"Required simulation output in {TEST_CONFIG_DIR} not found. "
                    "Run simulation tests (test_pipeline_simulation.py) first, "
                    "or ensure its module-scoped teardown fixture doesn't run before this module if in same session.")
    yield
    # No cleanup specific to this fixture; relies on test_pipeline_simulation.py's module cleanup
    # or a global cleanup strategy if these are part of a larger test suite run.

class TestGatherScript:
    def test_gather_py_runs_and_creates_output(self, ensure_simulation_output_exists_for_evaluation):
        cmd = [
            PYTHON_EXECUTABLE, str(GATHER_SCRIPT_PATH),
            "--num_config", str(TEST_CONFIG_NUM),
            "--num_simulations", str(NUM_SIMULATIONS_IN_TEST_CONFIG)
        ]
        print(f"Running command for TestGatherScript: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"gather.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        expected_final_marginal = TEST_CONFIG_DIR / "Final_Output.pkl"
        expected_final_joint = TEST_CONFIG_DIR / "Final_Output_joint.pkl"

        assert expected_final_marginal.exists(), f"Final_Output.pkl not found in {TEST_CONFIG_DIR}"
        assert expected_final_joint.exists(), f"Final_Output_joint.pkl not found in {TEST_CONFIG_DIR}"

        marginal_agg_df = pd.read_pickle(expected_final_marginal)
        joint_agg_df = pd.read_pickle(expected_final_joint)

        assert not marginal_agg_df.empty, "Aggregated marginal DataFrame is empty."
        # Expecting NUM_SIMULATIONS_IN_TEST_CONFIG * number_of_series * number_of_metrics_per_series
        # A simpler check: ensure 'simulation' column reflects the combined simulations.
        if 'simulation' in marginal_agg_df.columns:
             assert len(marginal_agg_df['simulation'].unique()) == NUM_SIMULATIONS_IN_TEST_CONFIG

        assert not joint_agg_df.empty, "Aggregated joint DataFrame is empty."
        if 'simulation' in joint_agg_df.columns:
            assert len(joint_agg_df['simulation'].unique()) == NUM_SIMULATIONS_IN_TEST_CONFIG

class TestPlottingScript:
    def test_plotting_py_runs_and_creates_plots(self, ensure_simulation_output_exists_for_evaluation):
        # Ensure Final_Output.pkl exists for plotting, by running gather if needed.
        final_output_file = TEST_CONFIG_DIR / "Final_Output.pkl"
        if not final_output_file.exists():
            print(f"{final_output_file} not found, running gather.py first for TestPlottingScript...")
            gather_cmd = [
                PYTHON_EXECUTABLE, str(GATHER_SCRIPT_PATH),
                "--num_config", str(TEST_CONFIG_NUM),
                "--num_simulations", str(NUM_SIMULATIONS_IN_TEST_CONFIG)
            ]
            subprocess.run(gather_cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
            assert final_output_file.exists(), f"gather.py failed to create {final_output_file} for plotting test."

        plot_config_output_dir = PROJECT_ROOT / "Plot" / f"Config_{TEST_CONFIG_NUM}"
        if plot_config_output_dir.exists(): # Clean up from previous partial runs if any
             shutil.rmtree(plot_config_output_dir)
        # plotting.py should create the directory

        cmd = [
            PYTHON_EXECUTABLE, str(PLOTTING_SCRIPT_PATH),
            "--num_config", str(TEST_CONFIG_NUM),
            "--m_total_series", str(M_TOTAL_SERIES_FOR_TEST_CONFIG)
        ]
        print(f"Running command for TestPlottingScript: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        assert result.returncode == 0, f"plotting.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        assert plot_config_output_dir.exists(), "Plot output directory not created by plotting.py"

        pdf_files = list(plot_config_output_dir.glob("*.pdf"))
        assert len(pdf_files) == M_TOTAL_SERIES_FOR_TEST_CONFIG, \
            f"Expected {M_TOTAL_SERIES_FOR_TEST_CONFIG} PDF plot files, found {len(pdf_files)}"

        # Teardown for this specific test's plot outputs
        if plot_config_output_dir.exists():
             shutil.rmtree(plot_config_output_dir)
        # Remove the parent Plot dir only if it's empty (or manage it globally)
        try:
            if plot_config_output_dir.parent.is_dir() and not any(plot_config_output_dir.parent.iterdir()):
                plot_config_output_dir.parent.rmdir()
        except OSError: # Might fail if other test created something there concurrently
            pass

class TestLatexTablesScript:
    def test_latex_tables_py_runs_and_creates_output(self, ensure_simulation_output_exists_for_evaluation):
        # latex_tables.py reads Final_Output.pkl and Final_Output_joint.pkl from various Config_X dirs.
        # The script is hardcoded to look for Config_1 through Config_6 (marginal) and a subset for joint.
        # To make this test meaningful for Config_999, we need to ensure its data is in a place
        # latex_tables.py will find, e.g., by temporarily copying Config_999 to Config_1.

        # Ensure Config_999/Final_Output*.pkl exist by running gather if needed.
        if not (TEST_CONFIG_DIR / "Final_Output.pkl").exists() or \
           not (TEST_CONFIG_DIR / "Final_Output_joint.pkl").exists():
            print(f"Final_Output files not found in {TEST_CONFIG_DIR}, running gather.py first for TestLatexTablesScript...")
            gather_cmd = [
                PYTHON_EXECUTABLE, str(GATHER_SCRIPT_PATH),
                "--num_config", str(TEST_CONFIG_NUM),
                "--num_simulations", str(NUM_SIMULATIONS_IN_TEST_CONFIG)
            ]
            subprocess.run(gather_cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

        temp_config_1_dir = PROJECT_ROOT / "Config_1"
        original_config_1_backup_path = None
        copied_for_test = False

        if temp_config_1_dir.exists():
            # Temporarily move existing Config_1 if it's there from actual runs
            original_config_1_backup_path = temp_config_1_dir.with_name("Config_1_backup_testing")
            if original_config_1_backup_path.exists(): # Clean up very old backup
                shutil.rmtree(original_config_1_backup_path)
            shutil.move(str(temp_config_1_dir), str(original_config_1_backup_path))
            print(f"Backed up existing {temp_config_1_dir} to {original_config_1_backup_path}")

        if TEST_CONFIG_DIR.exists(): # Copy test data to where latex_tables.py expects it
            shutil.copytree(TEST_CONFIG_DIR, temp_config_1_dir)
            copied_for_test = True
            print(f"Copied {TEST_CONFIG_DIR} to {temp_config_1_dir} for test.")
        else: # Should have been caught by ensure_simulation_output_exists_for_evaluation
            pytest.fail(f"Test setup error: {TEST_CONFIG_DIR} does not exist for latex_tables test.")

        latex_file_path = PROJECT_ROOT / "LaTeX_file.txt"
        latex_joint_file_path = PROJECT_ROOT / "LaTeX_joint_file.txt"
        if latex_file_path.exists(): latex_file_path.unlink()
        if latex_joint_file_path.exists(): latex_joint_file_path.unlink()

        cmd = [PYTHON_EXECUTABLE, str(LATEX_TABLES_SCRIPT_PATH)]
        print(f"Running command for TestLatexTablesScript: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)

        # latex_tables.py might print warnings if other Config_X dirs (2-6) are missing.
        # We mainly care that it doesn't crash and produces output files using Config_1 (our Config_999 copy).
        assert result.returncode == 0, f"latex_tables.py failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

        assert latex_file_path.exists(), "LaTeX_file.txt not created"
        assert latex_joint_file_path.exists(), "LaTeX_joint_file.txt not created"

        # Basic check for content (not empty)
        assert latex_file_path.read_text().strip() != "", "LaTeX_file.txt is empty"
        assert latex_joint_file_path.read_text().strip() != "", "LaTeX_joint_file.txt is empty"

        # Cleanup
        if latex_file_path.exists(): latex_file_path.unlink()
        if latex_joint_file_path.exists(): latex_joint_file_path.unlink()
        if copied_for_test and temp_config_1_dir.exists():
            shutil.rmtree(temp_config_1_dir)
            print(f"Removed temporary {temp_config_1_dir}")
        if original_config_1_backup_path and original_config_1_backup_path.exists():
            shutil.move(str(original_config_1_backup_path), str(temp_config_1_dir))
            print(f"Restored original {temp_config_1_dir} from backup.")
```

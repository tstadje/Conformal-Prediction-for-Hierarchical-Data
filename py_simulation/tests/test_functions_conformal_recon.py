import pytest
import numpy as np
import pandas as pd
from py_simulation.src.functions import (
    root_det, _calculate_G_matrix, conformal, joint_conformal,
    _get_conformal_quantile, get_table, get_table_occurence
)

class TestRootDet:
    def test_root_det_identity(self):
        assert np.isclose(root_det(np.eye(2)), 1.0)
        assert np.isclose(root_det(np.eye(3)), 1.0)

    def test_root_det_simple_matrix(self):
        A = np.array([[4, 0], [0, 9]], dtype=float) # det(A) = 36. m=2. (36)^(1/(2*2)) = 36^(1/4) = sqrt(6)
        assert np.isclose(root_det(A), np.sqrt(6))

    def test_root_det_singular_matrix(self):
        A_2x2_singular = np.array([[1, 1], [1, 1]], dtype=float) # det(A) = 0
        assert np.isclose(root_det(A_2x2_singular), 0.0)

        A_3x3_singular = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float) # Also singular
        assert np.isclose(root_det(A_3x3_singular), 0.0)

    def test_root_det_negative_determinant(self):
        A = np.array([[1, 2], [3, 1]], dtype=float) # det = 1 - 6 = -5
        # Current implementation of root_det returns np.nan if slogdet sign is -1
        assert np.isnan(root_det(A)), "root_det should be nan for negative determinant"

class TestCalculateGMatrix:
    def test_calculate_G_simple_case_OLS(self):
        S_matrix = np.array([[1, 1], [1, 0], [0, 1]], dtype=float) # m=3, n_bottom=2
        W_cov = np.eye(3) # OLS case
        # Expected G = (1/3) * [[1, 2, -1], [1, -1, 2]]
        G_expected = (1/3) * np.array([[1, 2, -1], [1, -1, 2]], dtype=float)

        G_calculated = _calculate_G_matrix(S_matrix, W_cov)
        assert G_calculated.shape == (2, 3) # n_bottom x m_total
        assert np.allclose(G_calculated, G_expected)

    def test_calculate_G_diag_W(self):
        S_matrix = np.array([[1,1],[1,0],[0,1]], dtype=float)
        W_cov = np.diag([1.0, 2.0, 3.0])
        # Expected G = [[1/3, 2/3, -1/3], [0.5, -0.5, 0.5]]
        G_expected = np.array([[1/3, 2/3, -1/3], [0.5, -0.5, 0.5]], dtype=float)

        G_calculated = _calculate_G_matrix(S_matrix, W_cov)
        assert G_calculated.shape == (2, 3)
        assert np.allclose(G_calculated, G_expected)

    def test_calculate_G_singular_W(self):
        # Test with a W that might cause issues if not handled (e.g. pseudo-inverse + regularization)
        S_matrix = np.array([[1, 1], [1, 0], [0, 1]], dtype=float)
        W_singular = np.array([[1,1,0],[1,1,0],[0,0,1]], dtype=float) # Singular W
        # Expect calculation to proceed due to ginv and regularization in _calculate_G_matrix
        G_calculated = _calculate_G_matrix(S_matrix, W_singular)
        assert G_calculated.shape == (2,3) # Check shape, specific values depend on ginv behavior

class TestGetConformalQuantile:
    def test_get_conformal_quantile_basic(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) # n_cal = 10
        n_cal = len(scores)
        # p = 0.9, (10+1)*0.9 = 9.9 -> k=10. scores[9] = 1.0
        assert np.isclose(_get_conformal_quantile(scores, 0.9, n_cal), 1.0)
        # p = 0.95, (10+1)*0.95 = 10.45 -> k=11 -> clamped to 10. scores[9] = 1.0
        assert np.isclose(_get_conformal_quantile(scores, 0.95, n_cal), 1.0)
        # p = 0.05, (10+1)*0.05 = 0.55 -> k=1. scores[0] = 0.1
        assert np.isclose(_get_conformal_quantile(scores, 0.05, n_cal), 0.1)

    def test_get_conformal_quantile_edge_cases(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        n_cal = len(scores)

        # p_target_coverage_for_tail = 0 => k_float = (5+1)*0 = 0. k_int = 0. k_eff = max(1,0) = 1. sorted_scores[0] = 0.1
        assert np.isclose(_get_conformal_quantile(scores, 0, n_cal), 0.1)
        # p_target_coverage_for_tail = 1 => k_float = (5+1)*1 = 6. k_int = 6. k_eff = min(max(1,6),5) = 5. sorted_scores[4] = 0.5
        assert np.isclose(_get_conformal_quantile(scores, 1, n_cal), 0.5)
        # p_target_coverage_for_tail = 0.5 => k_float = (5+1)*0.5 = 3. k_int = 3. k_eff = min(max(1,3),5) = 3. sorted_scores[2] = 0.3
        assert np.isclose(_get_conformal_quantile(scores, 0.5, n_cal), 0.3)

    def test_get_conformal_quantile_small_n_cal(self):
        small_scores = np.array([0.1])
        n_cal_small = len(small_scores)
        # p = 0.95 => k_float = (1+1)*0.95 = 1.9. k_int = 2. k_eff = min(max(1,2),1) = 1. sorted_scores[0] = 0.1
        assert np.isclose(_get_conformal_quantile(small_scores, 0.95, n_cal_small), 0.1)

        empty_scores = np.array([])
        # n_cal = 0 case in _get_conformal_quantile returns np.inf
        assert _get_conformal_quantile(empty_scores, 0.95, 0) == np.inf

@pytest.fixture
def sample_conformal_data():
    np.random.seed(42) # For reproducibility of test data
    n_cal, n_test, m_series = 100, 50, 2
    alpha = 0.1

    cal_data_dict = {}
    for i in range(1, m_series + 1):
        cal_data_dict[f'V{i}'] = np.random.rand(n_cal)
        cal_data_dict[f'V{i}_hat'] = cal_data_dict[f'V{i}'] + np.random.normal(0, 0.1, n_cal)
    cal_df = pd.DataFrame(cal_data_dict)

    test_data_dict = {}
    for i in range(1, m_series + 1):
        test_data_dict[f'V{i}'] = np.random.rand(n_test)
        test_data_dict[f'V{i}_hat'] = test_data_dict[f'V{i}'] + np.random.normal(0, 0.1, n_test)
    test_df = pd.DataFrame(test_data_dict)

    sigma_val_residuals = np.eye(m_series)
    return cal_df, test_df, m_series, alpha, sigma_val_residuals

class TestConformal:
    def test_conformal_output_structure(self, sample_conformal_data):
        cal_df, test_df, m, alpha, _ = sample_conformal_data
        n_bottom = m

        result_df = conformal(cal_df, test_df, n=n_bottom, m=m, method="hat", alpha=alpha)

        assert isinstance(result_df, pd.DataFrame)
        expected_cols = ['metric'] + [f'V{i+1}' for i in range(m)]
        assert list(result_df.columns) == expected_cols
        assert list(result_df['metric']) == ["Coverage", "Length", "Quantile"]
        for i in range(m):
            series_col = f'V{i+1}'
            assert result_df.loc[result_df['metric'] == 'Coverage', series_col].iloc[0] >= 0
            assert result_df.loc[result_df['metric'] == 'Coverage', series_col].iloc[0] <= 1
            assert result_df.loc[result_df['metric'] == 'Length', series_col].iloc[0] >= 0

    def test_conformal_perfect_calibration_scores(self, sample_conformal_data):
        cal_df_orig, test_df, m, alpha, _ = sample_conformal_data
        n_bottom = m

        cal_df_perfect = cal_df_orig.copy()
        for i in range(1, m + 1):
            cal_df_perfect[f'V{i}_hat_perfect'] = cal_df_perfect[f'V{i}']

        result_df = conformal(cal_df_perfect, test_df, n=n_bottom, m=m, method="hat_perfect", alpha=alpha)

        for i in range(m):
            series_col = f'V{i+1}'
            assert np.isclose(result_df.loc[result_df['metric'] == 'Length', series_col].iloc[0], 0.0, atol=1e-9)

    def test_conformal_with_nans_in_calibration(self, sample_conformal_data):
        cal_df_orig, test_df, m, alpha, _ = sample_conformal_data
        n_bottom = m
        cal_df_with_nans = cal_df_orig.copy()
        cal_df_with_nans.loc[0, 'V1_hat'] = np.nan # Introduce a NaN

        result_df = conformal(cal_df_with_nans, test_df, n=n_bottom, m=m, method="hat", alpha=alpha)
        assert not result_df.isnull().values.any(), "Should handle NaNs in calibration scores gracefully"


class TestJointConformal:
    def test_joint_conformal_output_structure(self, sample_conformal_data):
        cal_df, test_df, m, alpha, sigma_val_residuals = sample_conformal_data
        n_bottom = m

        result_df = joint_conformal(cal_df, sigma_val_residuals, test_df, n=n_bottom, m=m, method="hat", alpha=alpha)

        assert isinstance(result_df, pd.DataFrame)
        assert list(result_df.columns) == ['metric', 'V']

        metrics_series = result_df.set_index('metric')['V']
        assert 'Coverage' in metrics_series and metrics_series['Coverage'] >= 0 and metrics_series['Coverage'] <= 1
        assert 'Volume' in metrics_series and metrics_series['Volume'] >= 0
        assert 'Quantile' in metrics_series and np.isclose(metrics_series['Quantile'], 1 - alpha)

    def test_joint_conformal_perfect_calibration_scores(self, sample_conformal_data):
        cal_df_orig, test_df, m, alpha, sigma_val_residuals_orig = sample_conformal_data
        n_bottom = m

        cal_df_perfect = cal_df_orig.copy()
        for i in range(1, m + 1):
             cal_df_perfect[f'V{i}_hat_perfect'] = cal_df_perfect[f'V{i}']

        result_df = joint_conformal(cal_df_perfect, sigma_val_residuals_orig, test_df, n=n_bottom, m=m, method="hat_perfect", alpha=alpha)
        metrics_series = result_df.set_index('metric')['V']
        assert np.isclose(metrics_series['Volume'], 0.0, atol=1e-9)

    def test_joint_conformal_with_nans_in_calibration(self, sample_conformal_data):
        cal_df_orig, test_df, m, alpha, sigma_val_residuals = sample_conformal_data
        n_bottom = m
        cal_df_with_nans = cal_df_orig.copy()
        cal_df_with_nans.loc[0, 'V1_hat'] = np.nan # Ensures some residuals might be NaN

        result_df = joint_conformal(cal_df_with_nans, sigma_val_residuals, test_df, n=n_bottom, m=m, method="hat", alpha=alpha)
        # joint_conformal's cal_residuals are filtered for rows with NaNs.
        # If all rows become NaN, n_cal becomes 0, then coverage/volume become NaN.
        # If some rows remain, it should proceed.
        assert not result_df['V'].isnull().all(), "Joint conformal results should not be all NaN with some NaNs in cal data"


class TestTableGeneration:
    def test_get_table_empty_input(self):
        empty_df = pd.DataFrame()
        # Expect get_table to handle empty df, perhaps by producing an empty table structure or raising specific error
        # Current get_table will likely error on shape attribute or indexing.
        # For now, let's assume it should produce a minimal valid LaTeX table string.
        latex_str = get_table(empty_df, bold=False)
        assert "\\begin{tabular}" in latex_str
        assert "\\end{tabular}" in latex_str
        assert "\\toprule" in latex_str # Expect headers even if no data rows

    def test_get_table_with_nans(self):
        data_with_nans = pd.DataFrame({'M1': [1.0, np.nan], 'M2': [np.nan, 2.0]}, index=['C1', 'C2'])
        unc_with_nans = pd.DataFrame({'M1': [0.1, 0.2], 'M2': [0.3, np.nan]}, index=['C1', 'C2'])
        latex_str = get_table(data_with_nans, uncertainty_df=unc_with_nans, bold=True, digits=1)

        assert "nan" not in latex_str.lower(), "Raw 'nan' string should not appear in LaTeX output"
        assert "C1 & 1.0 $\\pm$ 0.1 & - $\\pm$ 0.3 \\\\" in latex_str # Check NaN replaced by "-"
        assert "C2 & - $\\pm$ 0.2 & 2.0 $\\pm$ - \\\\" in latex_str   # Check NaN replaced by "-"
        assert "\\begin{tabular}" in latex_str
        # Bolding should still apply to non-NaN minimums. M1's 1.0 is min in row C1.
        # M2's 2.0 is min in row C2.
        # The check for bolding on 1.0 (M1, C1) depends on whether "-" from M2 is considered smaller.
        # Current get_table idxmin will skip NaNs. So 1.0 (M1) vs - (M2) -> 1.0 is min.
        # For C2: - (M1) vs 2.0 (M2) -> 2.0 is min.
        assert "\\textbf{1.0 $\\pm$ 0.1}" in latex_str # C1, M1
        assert "\\textbf{2.0 $\\pm$ -}" in latex_str # C2, M2


    def test_get_table_occurence_empty_input(self):
        # Requires MultiIndex with 2 levels.
        empty_multi_idx = pd.MultiIndex.from_tuples([], names=['Config', 'Level'])
        empty_multi_df = pd.DataFrame(columns=['M1', 'M2'], index=empty_multi_idx)

        # The function should not raise ValueError for an empty DataFrame if the MultiIndex structure is valid.
        # It should produce a table with only headers.
        latex_str = get_table_occurence(empty_multi_df)
        assert "\\begin{tabular}" in latex_str
        assert "\\toprule" in latex_str
        assert "Config & Level & M1 & M2 \\\\" in latex_str # Header check
        assert "\\midrule" in latex_str # Midrule should be there
        assert "\\bottomrule" in latex_str
        # Check that no data rows are present beyond the header section
        lines = latex_str.splitlines()
        header_line_idx = -1
        for i, line in enumerate(lines):
            if "Config & Level & M1 & M2 \\\\" in line:
                header_line_idx = i
                break
        assert header_line_idx != -1
        # Expect only table structure lines after header, like midrule, bottomrule, end tabular
        assert not any(l.strip().startswith("C") or l.strip().startswith("& L") for l in lines[header_line_idx+2:-2])


    def test_get_table_occurence_with_nans(self):
        index_gto = pd.MultiIndex.from_tuples([("C1","L1"),("C1","L2")], names=("Config","Level"))
        data_gto_nans = pd.DataFrame({'M1': [10.0, np.nan], 'M2': [np.nan, 20.0]}, index=index_gto)
        latex_str = get_table_occurence(data_gto_nans, digits=1)

        assert "nan" not in latex_str.lower(), "Raw 'nan' string should not appear in LaTeX output"
        assert "C1 & L1 & \\textbf{10.0} & - \\\\" in latex_str # M1 is max, M2 is NaN ("-")
        assert " & L2 & - & \\textbf{20.0} \\\\" in latex_str   # M1 is NaN ("-"), M2 is max
        assert "\\begin{tabular}" in latex_str
        assert "textbf" in latex_str # Check that bolding logic still runs

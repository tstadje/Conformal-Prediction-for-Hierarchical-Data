import pytest
import numpy as np
import pandas as pd
from py_simulation.src.functions import prediction_fast

@pytest.fixture
def sample_forecasting_data():
    np.random.seed(123) # For reproducibility
    # n_obs_total = 50, train_ratio = 0.6 (30 train, 20 test/predict)
    # m_series = 2 (V1, V2), n_bottom_param = 2 (all are bottom for simplicity)
    n_obs_total = 50
    n_train = 30
    m_series_fixture = 2
    # In prediction_fast, 'n' is n_bottom_level_series
    n_bottom_fixture = 2 # implies all series are bottom level for easier formula handling in tests

    # Features X1, X2, X3
    X1 = np.random.rand(n_obs_total) * 10
    X2 = np.random.rand(n_obs_total) * 5
    X3 = np.random.rand(n_obs_total) * 2

    # Target series V1, V2 (simple linear relationship for testing LM)
    # V1 = 1*X1 + 0.5*X2 + noise
    # V2 = 0.8*X3 + noise
    V1 = 1.0 * X1 + 0.5 * X2 + np.random.normal(0, 0.1, n_obs_total)
    V2 = 0.8 * X3 + np.random.normal(0, 0.1, n_obs_total)

    full_df = pd.DataFrame({
        'V1': V1, 'V2': V2,
        'X1': X1, 'X2': X2, 'X3': X3
    })

    train_indices = list(np.arange(n_train)) # Convert to list as per prediction_fast type hint

    return full_df, train_indices, m_series_fixture, n_bottom_fixture

class TestPredictionFast:
    def test_prediction_fast_lm_output_shape_and_type(self, sample_forecasting_data):
        full_df, train_indices, m, n_bottom = sample_forecasting_data
        n_predict_expected = len(full_df) - len(train_indices)

        # `n` in prediction_fast is n_bottom_level_series
        forecasts = prediction_fast(full_df, model_type="lm", train_indices=train_indices, m=m, n=n_bottom)

        assert isinstance(forecasts, np.ndarray)
        assert forecasts.shape == (n_predict_expected, m)
        assert not np.all(np.isnan(forecasts)), "All LM forecasts are NaN, model fitting might have failed generally."
        # Check if any individual series is all NaN (can happen if a specific model fails)
        for i in range(m):
            assert not np.all(np.isnan(forecasts[:, i])), f"All LM forecasts for series V{i+1} are NaN."


    def test_prediction_fast_gam_output_shape_and_type(self, sample_forecasting_data):
        full_df, train_indices, m, n_bottom = sample_forecasting_data
        n_predict_expected = len(full_df) - len(train_indices)

        try:
            # `n` in prediction_fast is n_bottom_level_series
            forecasts = prediction_fast(full_df, model_type="gam", train_indices=train_indices, m=m, n=n_bottom)

            assert isinstance(forecasts, np.ndarray)
            assert forecasts.shape == (n_predict_expected, m)
            assert not np.all(np.isnan(forecasts)), "All GAM forecasts are NaN, model fitting might have failed generally."
            for i in range(m):
                assert not np.all(np.isnan(forecasts[:, i])), f"All GAM forecasts for series V{i+1} are NaN."

        except Exception as e:
            # PyGAM can be sensitive, especially with small datasets or certain data patterns.
            # If this test fails due to PyGAM errors, the fixture data or GAM parameters in prediction_fast might need adjustment for testing.
            pytest.fail(f"prediction_fast with GAM failed: {e}")

    def test_prediction_fast_lm_varying_series_formulas(self):
        # Test the logic where bottom series might use fewer predictors (the 80/20 rule)
        # and aggregated series use all predictors.
        np.random.seed(456) # Seed for this test's specific data
        n_obs_total = 60
        n_train = 40
        train_indices = list(np.arange(n_train))

        # m_total = 3. V1 is bottom, V2 and V3 are treated as "aggregated" by index logic.
        m_test = 3
        n_bottom_test = 1 # Only V1 (index 0, so series name V1) is "bottom" as per (i > m - n) logic.
                          # V2 (index 1) and V3 (index 2) are "aggregated".
                          # Series indices in loop: 1, 2, 3.
                          # V1 (i=1): is_bottom_level = (1 > 3 - 1) = (1 > 2) = False. This is wrong.
                          # R code: i <= (m-n) is agg. So i=1, i=2 are agg. i=3 is bottom.
                          # Python: is_bottom_level = (series_loop_idx > (m_total - n_bottom_actual))
                          # series_loop_idx goes from 1 to m_total.
                          # If m=3, n_bottom=1: m-n = 2.
                          #   Series V1 (idx 0, loop i=1): is_bottom = (1 > 2) = False. (Agg)
                          #   Series V2 (idx 1, loop i=2): is_bottom = (2 > 2) = False. (Agg)
                          #   Series V3 (idx 2, loop i=3): is_bottom = (3 > 2) = True.  (Bottom)
                          # So, V3 will be bottom, V1 & V2 will be aggregated.

        X1 = np.random.rand(n_obs_total) * 10
        X2 = np.random.rand(n_obs_total) * 5
        X3 = np.random.rand(n_obs_total) * 2
        V1 = 1.0 * X1 + 0.5 * X2 + np.random.normal(0, 0.1, n_obs_total) # Agg1
        V2 = 0.8 * X3 + np.random.normal(0, 0.1, n_obs_total)           # Agg2
        V3 = 0.5 * X1 + 0.3 * X2 + 0.2 * X3 + np.random.normal(0,0.1,n_obs_total) # Bottom

        full_df_test = pd.DataFrame({
            'V1': V1, 'V2': V2, 'V3': V3,
            'X1': X1, 'X2': X2, 'X3': X3
        })

        # To make the stochastic choice of predictors for V3 (bottom series) deterministic for this test:
        # We can't easily mock np.random.binomial inside prediction_fast from here without more complex setup.
        # Instead, we'll run it a few times and hope both paths (3-var and 2-var for V3) are covered,
        # or accept that this test mainly ensures it runs with the mixed logic.
        # For this unit test, we mostly care that it runs and produces the right shape.
        # The exact formula chosen for V3 is stochastic.

        forecasts = prediction_fast(full_df_test,
                                    model_type="lm",
                                    train_indices=train_indices,
                                    m=m_test,
                                    n=n_bottom_test) # n is n_bottom_level_series

        n_predict_expected = len(full_df_test) - len(train_indices)
        assert forecasts.shape == (n_predict_expected, m_test)
        assert not np.all(np.isnan(forecasts)), "All LM forecasts are NaN with mixed series types."
        for i in range(m_test):
            assert not np.all(np.isnan(forecasts[:, i])), f"All LM forecasts for series V{i+1} (mixed type) are NaN."

    def test_prediction_fast_empty_train_indices(self, sample_forecasting_data):
        full_df, _, m, n_bottom = sample_forecasting_data
        train_indices = [] # Empty training set

        # Expect model fitting to fail or behave poorly, possibly leading to NaNs or errors.
        # statsmodels OLS with empty data: `ValueError: shapes (0,3) and (3,) not aligned: 3 (dim 1) != 0 (dim 0)`
        # PyGAM with empty data: Likely similar errors.
        # The function's internal NaN handling `valid_train_idx.any()` should catch this.
        # If `valid_train_idx.any()` is false, it prints a warning and forecasts for that series are NaN.

        forecasts = prediction_fast(full_df, model_type="lm", train_indices=train_indices, m=m, n=n_bottom)
        n_predict_expected = len(full_df) - len(train_indices)
        assert forecasts.shape == (n_predict_expected, m)
        assert np.all(np.isnan(forecasts)), "Forecasts should be all NaN if training set is empty."

    def test_prediction_fast_all_train_indices(self, sample_forecasting_data):
        full_df, _, m, n_bottom = sample_forecasting_data
        train_indices = list(np.arange(len(full_df))) # All data used for training

        # Expect no test samples, so forecasts array should be empty (shape (0,m))
        forecasts = prediction_fast(full_df, model_type="lm", train_indices=train_indices, m=m, n=n_bottom)
        assert forecasts.shape == (0, m)

```

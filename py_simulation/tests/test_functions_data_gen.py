import pytest
import numpy as np
# import pandas as pd # Not strictly needed for these tests but good practice if extending

# Assuming pytest runs from project root, so py_simulation.src is discoverable
from py_simulation.src.functions import generate_S, generate_multi

class TestGenerateS:
    @pytest.mark.parametrize("n_bottom, l_levels, expected_rows, expected_cols, expected_agg_sum", [
        (12, 1, 1 + 12**1 + 12, 12, 1),  # 25 rows, agg sum = 1
        (12, 2, 1 + 12**1 + 12**2 + 12, 12, 2), # 169 rows, agg sum = 2
        (24, 1, 1 + 12**1 + 24, 24, 1) # n=24, l=1. 24 must be mult of 12^1.
    ])
    def test_generate_S_shape_and_structure(self, n_bottom, l_levels, expected_rows, expected_cols, expected_agg_sum):
        """Tests shape and key structural properties of S matrix for valid inputs."""
        # generate_S itself is deterministic, so no seed needed here unless sub-functions it calls are random.

        S = generate_S(n=n_bottom, l=l_levels)

        assert S.shape == (expected_rows, expected_cols), f"Shape mismatch for n={n_bottom}, l={l_levels}"
        assert np.all(S[0, :] == 1), "First row (Total) should be all ones"
        assert np.all(S[-n_bottom:, :] == np.eye(n_bottom)), "Last n_bottom rows should form an identity matrix"

        if l_levels > 0:
            agg_rows = S[1:-n_bottom, :] # Rows corresponding to aggregated series (excluding total and bottom levels)
            expected_num_agg_rows = sum(12**i for i in range(1, l_levels + 1))
            assert agg_rows.shape[0] == expected_num_agg_rows, "Number of aggregation rows mismatch"
            # Each bottom series must be included 'expected_agg_sum' times in the aggregation sums.
            # expected_agg_sum is effectively l_levels for this S generation logic.
            assert np.all(np.isclose(np.sum(agg_rows, axis=0), expected_agg_sum)), \
                f"Sum over aggregation rows for each column failed for n={n_bottom}, l={l_levels}. Expected sum {expected_agg_sum}."

    @pytest.mark.parametrize("n_invalid, l_levels", [
        (20, 1),    # 20 is not multiple of 12^1
        (140, 2),   # 140 is not multiple of 12^2=144
        (12, 3)     # 12 is not multiple of 12^2 or 12^3, which are checked for l=3
    ])
    def test_generate_S_n_divisibility_error(self, n_invalid, l_levels):
        """Tests that generate_S raises ValueError for invalid n relative to l."""
        # Regex matches "n must be a multiple of 12^i for i up to l" or similar from the code.
        with pytest.raises(ValueError, match="n must be a multiple of 12\\^(\\d+).*{0} is not a multiple of {1}".format(n_invalid, "12\\^\\d+").replace("\\", "\\\\") ):
             generate_S(n=n_invalid, l=l_levels)

    def test_generate_S_valid_multiples_pass(self):
        """Test cases that should pass the divisibility checks."""
        generate_S(n=12, l=1)
        generate_S(n=24, l=1) # 24 is multiple of 12^1
        generate_S(n=144, l=1) # 144 is multiple of 12^1
        generate_S(n=144, l=2) # 144 is multiple of 12^1 and 12^2
        generate_S(n=1728, l=3) # 1728 = 12^3, so multiple of 12^1, 12^2, 12^3

    def test_generate_S_l0_case(self):
        """Tests the base case where l=0 (no aggregation levels beyond total)."""
        n_bottom = 12
        l_levels = 0
        S = generate_S(n=n_bottom, l=l_levels)
        assert S.shape == (1 + n_bottom, n_bottom), "Shape mismatch for l=0"
        assert np.all(S[0, :] == 1), "First row (Total) should be all ones for l=0"
        assert np.all(S[1:, :] == np.eye(n_bottom)), "Rows after total should be identity for l=0"


class TestGenerateMulti:
    @pytest.fixture
    def sample_S_matrix_l1_n12(self):
        """Provides a standard S matrix for generate_multi tests."""
        return generate_S(n=12, l=1) # Shape will be (25, 12)

    @pytest.fixture
    def generate_multi_inputs(self, sample_S_matrix_l1_n12):
        """Sets up inputs for generate_multi, including a seed for its random parts."""
        np.random.seed(123) # Seed for data generation process *within* generate_multi
        n_obs = 50
        S_matrix = sample_S_matrix_l1_n12
        return S_matrix, n_obs

    def test_generate_multi_output_structure_and_types(self, generate_multi_inputs):
        S_matrix, n_obs = generate_multi_inputs
        results = generate_multi(S=S_matrix, n_obs=n_obs, noise_type="A")

        assert isinstance(results, dict)
        assert "Features" in results
        assert "Y_level" in results

        features = results["Features"]
        y_level = results["Y_level"]

        assert isinstance(features, np.ndarray)
        assert features.shape == (n_obs, 3), "Features shape mismatch (X1, X2, X3)"

        assert isinstance(y_level, np.ndarray)
        assert y_level.shape == (S_matrix.shape[0], n_obs), "Y_level shape mismatch"

    def test_generate_multi_features_distribution(self, generate_multi_inputs):
        """Checks if feature distributions are close to specified parameters."""
        S_matrix, _ = generate_multi_inputs # n_obs from fixture is 50, use larger for stats
        n_obs_large = 3000

        np.random.seed(1234) # Ensure this specific call to generate_multi is reproducible
        results = generate_multi(S=S_matrix, n_obs=n_obs_large, noise_type="A")
        features = results["Features"]

        # Parameters from Python's generate_multi:
        # X1: loc=10, scale=10
        # X2: loc=0, scale=1
        # X3: loc=10, scale=1
        assert np.isclose(np.mean(features[:, 0]), 10, atol=0.75), "X1 mean out of expected range" # Wider atol for higher std dev
        assert np.isclose(np.std(features[:, 0], ddof=1), 10, atol=0.75), "X1 std dev out of expected range"

        assert np.isclose(np.mean(features[:, 1]), 0, atol=0.1), "X2 mean out of expected range" # Tighter atol for smaller scale
        assert np.isclose(np.std(features[:, 1], ddof=1), 1, atol=0.1), "X2 std dev out of expected range"

        assert np.isclose(np.mean(features[:, 2]), 10, atol=0.1), "X3 mean out of expected range"
        assert np.isclose(np.std(features[:, 2], ddof=1), 1, atol=0.1), "X3 std dev out of expected range"

    def test_generate_multi_noise_type_B_not_implemented(self, generate_multi_inputs):
        S_matrix, n_obs = generate_multi_inputs
        with pytest.raises(NotImplementedError, match="Noise type 'B' .* not yet implemented"):
            generate_multi(S=S_matrix, n_obs=n_obs, noise_type="B")

    def test_y_level_coherence(self, generate_multi_inputs):
        """Tests if Y_level is coherent with S matrix and its own bottom levels for S(n=12, l=1)."""
        S_matrix, n_obs = generate_multi_inputs # S_matrix here is for n=12, l=1
        np.random.seed(12345) # Ensure this call is reproducible
        results = generate_multi(S=S_matrix, n_obs=n_obs, noise_type="A")
        y_level = results["Y_level"]

        n_bottom = S_matrix.shape[1] # Should be 12
        assert n_bottom == 12, "Test logic assumes n_bottom=12 for this S_matrix"
        assert S_matrix.shape == (25,12), "Test logic assumes S_matrix is 25x12 for n=12, l=1"

        # Last n_bottom rows of Y_level are considered the noisy bottom-level series outputs
        # This is because the last n_bottom rows of S_matrix form an identity matrix.
        y_noisy_bottom_level_transposed = y_level[-n_bottom:, :]

        # 1. Grand Total (first row of Y_level)
        # S_matrix[0,:] is all ones. So y_level[0,:] should be sum of all series in y_noisy_bottom_level_transposed.
        expected_grand_total = np.sum(y_noisy_bottom_level_transposed, axis=0)
        assert np.allclose(y_level[0, :], expected_grand_total), \
            "Grand total in Y_level does not match sum of its reconstructed bottom levels"

        # 2. Aggregation Levels for S(n=12, l=1)
        # For this specific S matrix (n=12, l=1):
        # S[0,:] = Total
        # S[1:1+n_bottom, :] = These are the first level aggregation rows.
        # For l=1, generate_S creates these rows such that S[1+k, k] = 1 and others are 0.
        # This means these "aggregation" series are direct pass-throughs of the bottom series.
        # y_level[1+k, :] should be equal to y_noisy_bottom_level_transposed[k, :]

        for k_idx in range(n_bottom):
            # y_level row index for these pass-through aggregation rows is 1 + k_idx
            # Corresponding bottom series in y_noisy_bottom_level_transposed is index k_idx
            # The S_matrix row S[1+k_idx,:] should be an elementary vector e_k (1 at k_idx, 0 elsewhere).
            assert np.isclose(S_matrix[1+k_idx, k_idx], 1.0), f"S_matrix[1+{k_idx}, {k_idx}] should be 1 for l=1 structure"
            assert np.isclose(np.sum(S_matrix[1+k_idx, :]), 1.0), f"Sum of S_matrix[1+{k_idx},:] should be 1"

            assert np.allclose(y_level[1 + k_idx, :], y_noisy_bottom_level_transposed[k_idx, :]), \
                f"Aggregation series {1+k_idx} in Y_level does not match corresponding bottom series for l=1 case."
```

This test suite covers:
-   `generate_S`:
    -   Correct shape and basic structural properties (total row, identity part) for (l=1, n=12) and (l=2, n=12).
    -   Correct sum of aggregation rows (each bottom series contributes to `l` aggregation series).
    -   Error handling for `n` not being a multiple of `12^l`.
    -   Correct behavior for `l=0` (base case: total + identity).
-   `generate_multi`:
    -   Output structure (dict with "Features", "Y_level") and types (NumPy arrays).
    -   Correct shapes of "Features" and "Y_level".
    -   Plausible means and standard deviations for generated features (X1, X2, X3) based on the current parameters in `functions.py`.
    -   `NotImplementedError` for `noise_type="B"`.
    -   Coherence of `Y_level` with the input `S` matrix (total row sum, and aggregation rows correctly summing their constituent bottom-level series from `Y_level` itself).

The `conftest.py` suggestion is good practice if path issues arise, but `pytest` often handles finding modules from the project root correctly if the `py_simulation` directory is structured as a package (with `__init__.py` files) and tests are run from the root. For now, I'll assume this works.

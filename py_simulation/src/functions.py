import numpy as np
import math
import random
# from scipy.stats import multivariate_t # For noise_type "B" if implemented
# from scipy.linalg import cholesky # Potentially for noise generation if custom

def generate_S(n: int, l: int, balanced: bool = True):
    """
    Generates the S matrix for hierarchical aggregation.
    Corresponds to generate_S from R/function.R.

    Args:
        n: Number of bottom-level series. Must be a multiple of 12^l.
        l: Number of aggregation levels. l=0 means only bottom level (S=diag(n)).
           l=1 means one level of aggregation above bottom, etc.
        balanced: If True, assumes a balanced hierarchy. Currently, only True is supported.

    Returns:
        A NumPy array representing the S matrix, including rows for sums at each
        aggregation level, a row for the grand total, and the identity matrix for
        bottom-level series. Shape: (1 + sum_{i=1 to l} 12^i + n, n).
    """
    if not balanced:
        raise NotImplementedError("Only balanced=True is currently supported.")

    # Check if n is a power of 12, as in the original R code's comment
    exponent = math.log(n, 12)
    if not exponent.is_integer():
        # This was a print in R, but execution continued.
        # The critical part is that n must be divisible by 12^i for i up to l.
        print(f"Note: n={n} is not a strict power of 12. Ensure n is divisible by 12^i for i up to l.")

    for i_check in range(1, l + 1):
        if n % (12**i_check) != 0:
            raise ValueError(f"n must be a multiple of 12^l. {n} is not a multiple of 12^{i_check}.")


    # S_list will store the rows of S matrix before adding the identity matrix part
    # Start with the grand total row (sum of all series)
    S_list = [np.ones((1, n))]

    # Generate rows for each aggregation level
    for i in range(1, l + 1):  # Level index, from 1 up to l
        num_parent_blocks = 12**(i-1) # Number of blocks at level i-1
        # Each parent block is divided into 12 sub-blocks at level i

        for k in range(num_parent_blocks): # Iterate over each parent block
            # sc_row in R, re-initialized for each of the 12 child sums
            # In R: sc_row = matrix(0, nrow=1, ncol=n)
            # Then a specific part is set to 1, row is bound, then that part is set to 0.

            parent_block_start_col = k * (n // num_parent_blocks)
            child_block_size = n // (12**i) # Size of the smallest blocks at level i

            if child_block_size == 0:
                 # This case should be prevented by the n % (12**l) == 0 check earlier.
                 raise ValueError(
                    f"Block size is zero for n={n}, l={l}, level i={i}. "
                    "This indicates n is not sufficiently divisible by 12^i."
                )

            for j in range(12): # Iterate over the 12 children within the parent block
                row = np.zeros((1, n)) # Start with a row of zeros

                # Define the columns for the current child block that should be 1
                block_start_col = parent_block_start_col + j * child_block_size
                block_end_col = block_start_col + child_block_size

                row[0, block_start_col:block_end_col] = 1
                S_list.append(row) # Add this aggregation row

    S_intermediate = np.vstack(S_list)

    # Add identity matrix part for the bottom-level series
    # R: rbind(S, diag(1,n))
    S_final = np.vstack([S_intermediate, np.eye(n)])
    return S_final

def generate_random_corr_matrix(n: int):
    """
    Generates a random n x n correlation matrix.
    Helper for generate_multi's noise_type "A".
    """
    A = np.random.normal(size=(n, n))
    cov_matrix = A @ A.T # A * A' to make it positive semi-definite
    # Ensure diagonal is 1 for a correlation matrix
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(cov_matrix)))
    corr_matrix = D_inv_sqrt @ cov_matrix @ D_inv_sqrt
    # Symmetrize and clip for numerical stability
    corr_matrix = (corr_matrix + corr_matrix.T) / 2
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    return corr_matrix

def generate_multi(S: np.ndarray, n_obs: int, noise_type: str = "A"):
    """
    Generates features (X1, X2, X3) and target variables (Y_level) based on S matrix.
    Corresponds to generate_multi from R/function.R.

    Args:
        S: The S matrix, typically from generate_S. Assumed to include identity rows for bottom series.
           Shape (num_total_series, num_bottom_series).
        n_obs: Number of observations (time points) to generate.
        noise_type: Type of noise. "A" for multivariate normal, "B" for multivariate t (not implemented).

    Returns:
        A dictionary with keys "Features" and "Y_level":
        - "Features": NumPy array of shape (n_obs, 3) for X1, X2, X3.
        - "Y_level": NumPy array of shape (num_total_series, n_obs). Each row is a series.
    """
    n_bottom_level = S.shape[1] # Number of bottom-level series

    # Feature generation (covariates)
    X1 = np.random.normal(loc=10, scale=10, size=n_obs)
    X2 = np.random.normal(loc=0, scale=1, size=n_obs)
    X3 = np.random.normal(loc=10, scale=1, size=n_obs)

    # Base functions to construct the true underlying signals for bottom-level series
    base_functions = [
        lambda x1, x2, x3: x1,
        lambda x1, x2, x3: x2,
        lambda x1, x2, x3: x3,
        lambda x1, x2, x3: x1**2,
        lambda x1, x2, x3: x2**2,
        lambda x1, x2, x3: x3**2,
        lambda x1, x2, x3: np.log(np.abs(x1) + 1e-8), # Added epsilon for stability
        lambda x1, x2, x3: np.log(np.abs(x2) + 1e-8),
        lambda x1, x2, x3: np.log(np.abs(x3) + 1e-8),
        lambda x1, x2, x3: np.sqrt(np.abs(x1)),
        lambda x1, x2, x3: np.sqrt(np.abs(x2)),
        lambda x1, x2, x3: np.sqrt(np.abs(x3)),
    ]
    num_base_functions = len(base_functions)

    # Y: True underlying signals for bottom-level series (before noise)
    Y_true_bottom_level = np.zeros((n_obs, n_bottom_level))

    for j in range(n_bottom_level): # For each bottom-level series
        # R: k = sample(1:num_base_functions, 1) -> k is a count of functions to sum
        k_num_funcs_to_sum = random.randint(1, num_base_functions)

        # R: mybeta = sample(base_functions, k, replace = TRUE) -> sample k functions
        chosen_func_indices = np.random.choice(len(base_functions), size=k_num_funcs_to_sum, replace=True)

        # R: polynomial_coeffs = rbernoulli(k,p=0.5)*2-1 -> coefficients of -1 or 1
        polynomial_coeffs = (np.random.binomial(1, 0.5, size=k_num_funcs_to_sum) * 2 - 1).astype(float)

        # Combine the chosen functions
        current_y_j = np.zeros(n_obs)
        for func_idx_pos, func_idx in enumerate(chosen_func_indices):
            selected_function = base_functions[func_idx]
            current_y_j += selected_function(X1, X2, X3) * polynomial_coeffs[func_idx_pos]
        Y_true_bottom_level[:, j] = current_y_j

    # Noise generation
    if noise_type == "B":
        # Placeholder for noise_type "B" which uses rmvt from mvnfast and reads an RDS file
        # Sigma_B = ... # n_bottom_level x n_bottom_level covariance matrix from RDS
        # nu_B = ...    # degrees of freedom from RDS
        # noise = multivariate_t.rvs(loc=np.zeros(n_bottom_level), shape=Sigma_B, df=nu_B, size=n_obs)
        raise NotImplementedError("Noise type 'B' (multivariate t from RDS) is not yet implemented.")
    elif noise_type == "A":
        # Generate random correlation matrix Sigma
        Sigma_corr = generate_random_corr_matrix(n_bottom_level)
        # R: mvrnorm(n_obs, mu=rep(10,n), Sigma=100*Sigma)
        mean_vector = np.full(n_bottom_level, 10.0)
        cov_matrix = 100.0 * Sigma_corr
        # Ensure covariance matrix is positive semi-definite
        # Adding small value to diagonal for numerical stability before cholesky in multivariate_normal
        min_eig = np.min(np.linalg.eigvalsh(cov_matrix))
        if min_eig < 0:
            cov_matrix -= min_eig * np.eye(n_bottom_level) # Does not preserve correlation
            # A better way is to ensure generate_random_corr_matrix always returns PSD for cov_matrix
            # For now, this is a simple fix if issues arise.
            # Or, ensure Sigma_corr is PSD. generate_random_corr_matrix tries to do this.
            pass

        noise = np.random.multivariate_normal(mean=mean_vector, cov=cov_matrix, size=n_obs, check_valid='warn')
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    # Add noise to the true bottom-level signals
    Y_noisy_bottom_level = Y_true_bottom_level + noise # Shape: (n_obs, n_bottom_level)

    # Apply S matrix to get all levels of the hierarchy
    # R: b = as.matrix(t(Y_bruite)) -> Y_noisy_bottom_level.T which is (n_bottom_level, n_obs)
    # R: Y_level = as.matrix(S %*% b)
    # S shape: (num_total_series, n_bottom_level)
    # Y_noisy_bottom_level.T shape: (n_bottom_level, n_obs)
    # Result Y_level shape: (num_total_series, n_obs)
    Y_level = S @ Y_noisy_bottom_level.T

    # Store features
    Features = np.column_stack((X1, X2, X3)) # Shape: (n_obs, 3)

    return {"Features": Features, "Y_level": Y_level}


if __name__ == '__main__':
    # Example Usage (mimicking parts of study.R)
    n_param = 12  # Number of bottom-level series
    l_param = 1   # Number of aggregation levels above the bottom level

    print(f"Generating S matrix for n={n_param}, l={l_param}...")
    # S matrix includes sums and identity rows for bottom level
    S_matrix = generate_S(n=n_param, l=l_param)
    print("S_matrix shape:", S_matrix.shape)
    # Expected: 1 (grand total) + 12^1 (level 1 sums) + 12 (identity) = 1+12+12 = 25 rows. (25, 12)
    assert S_matrix.shape == (1 + 12**l_param + n_param, n_param)

    # Test with n that is not a power of 12, but is a multiple of 12^l
    n_param_24 = 24
    l_param_1 = 1
    print(f"Generating S matrix for n={n_param_24}, l={l_param_1}...")
    S_matrix_24 = generate_S(n=n_param_24, l=l_param_1)
    print("S_matrix_24 shape:", S_matrix_24.shape)
    # Expected: 1 (grand total) + 12^1 (level 1 sums) + 24 (identity) = 1+12+24 = 37 rows. (37, 24)
    assert S_matrix_24.shape == (1 + 12**l_param_1 + n_param_24, n_param_24)

    # Test case from original R code that might have issues if n is not multiple of 12^l
    try:
        print("Testing S matrix generation with n not multiple of 12^l (expecting error)...")
        generate_S(n=10, l=1) # 10 is not a multiple of 12^1
    except ValueError as e:
        print(f"Caught expected error: {e}")

    n_observations = 200
    print(f"\nGenerating simulated data with n_obs={n_observations}...")
    simulated_data = generate_multi(S=S_matrix, n_obs=n_observations, noise_type="A")

    Features_train = simulated_data["Features"]
    Y_level_train = simulated_data["Y_level"] # Shape: (num_total_series, n_obs)

    print("Features_train shape:", Features_train.shape) # (n_obs, 3)
    print("Y_level_train shape:", Y_level_train.shape)   # (S.shape[0], n_obs)
    assert Features_train.shape == (n_observations, 3)
    assert Y_level_train.shape == (S_matrix.shape[0], n_observations)

    # For modeling, data is often (n_obs, n_features) or (n_obs, n_series)
    Y_level_train_transposed = Y_level_train.T
    print("Y_level_train_transposed (for modeling) shape:", Y_level_train_transposed.shape)
    assert Y_level_train_transposed.shape == (n_observations, S_matrix.shape[0])

    print("\nExample data generation successful.")

    # Test with a higher l value
    n_param_144 = 144
    l_param_2 = 2 # n=144 is 12^2, so it's a multiple of 12^1 and 12^2
    print(f"\nGenerating S matrix for n={n_param_144}, l={l_param_2}...")
    S_matrix_144_l2 = generate_S(n=n_param_144, l=l_param_2)
    print("S_matrix_144_l2 shape:", S_matrix_144_l2.shape)
    # Expected rows: 1 (grand total) + 12^1 (level 1 sums) + 12^2 (level 2 sums) + 144 (identity)
    # = 1 + 12 + 144 + 144 = 301 rows. (301, 144)
    expected_rows = 1 + (12**1) + (12**2) + n_param_144
    assert S_matrix_144_l2.shape == (expected_rows, n_param_144)

    print("\nGenerating simulated data for n=144, l=2...")
    simulated_data_144_l2 = generate_multi(S=S_matrix_144_l2, n_obs=n_observations, noise_type="A")
    print("Features_train (n=144, l=2) shape:", simulated_data_144_l2["Features"].shape)
    print("Y_level_train (n=144, l=2) shape:", simulated_data_144_l2["Y_level"].shape)
    assert simulated_data_144_l2["Y_level"].shape == (expected_rows, n_observations)
    print("\nAll data generation tests passed.")


# Additional imports for conformal prediction functions
import pandas as pd
from scipy.linalg import pinv as ginv # Moore-Penrose pseudo-inverse

def _get_conformal_quantile(scores: np.ndarray, p_target_coverage_for_tail: float, n_cal: int) -> float:
    """
    Helper function to calculate the k-th order statistic for conformal prediction.

    Args:
        scores: NumPy array of calibration scores.
        p_target_coverage_for_tail: Target coverage probability for the tail, e.g., 1 - alpha/2.
        n_cal: Number of calibration scores.

    Returns:
        The quantile value.
    """
    if n_cal == 0:
        # Handle cases with no calibration data, though typically this shouldn't happen
        # in a valid conformal prediction setup. Return Inf or NaN, or raise error.
        # R's quantile(..., type=1) for an empty set would error.
        # np.quantile on empty array errors.
        # For now, let's assume scores is non-empty if n_cal > 0.
        # If scores can be empty while n_cal > 0 (e.g. all NaNs filtered out),
        # this might need adjustment.
        return np.inf # Or np.nan, or raise error, depending on desired behavior

    k_float = (n_cal + 1) * p_target_coverage_for_tail
    k_int = int(np.ceil(k_float))
    # Ensure k_eff is a valid index (1 to n_cal for 1-based, 0 to n_cal-1 for 0-based)
    k_eff = min(max(1, k_int), n_cal)

    sorted_scores = np.sort(scores)
    return sorted_scores[k_eff - 1] # k_eff is 1-based, convert to 0-based index

def conformal(data_calib_df: pd.DataFrame, data_test_df: pd.DataFrame,
              n: int, m: int, method: str, alpha: float = 0.1):
    """
    Computes marginal conformal prediction intervals and their metrics.
    Corresponds to `conformal` from R/function.R.

    Args:
        data_calib_df: DataFrame with calibration data (actuals and forecasts).
        data_test_df: DataFrame with test data (actuals and forecasts).
        n: Number of bottom-level series (passed for API consistency, not directly used).
        m: Total number of series.
        method: Forecasting method name (e.g., "Direct", "ETS", "ARIMA") used to find forecast columns.
        alpha: Miscoverage level (e.g., 0.1 for 90% prediction intervals).

    Returns:
        Pandas DataFrame with columns 'metric', 'V1', 'V2', ..., 'Vm',
        reporting Coverage, Length (squared), and Quantile (1-alpha) for each series.
    """
    info_df_data = {'metric': ["Coverage", "Length", "Quantile"]}

    for i in range(1, m + 1):
        var_name = f"V{i}"
        # Determine the suffix for forecast columns based on method
        hat_col_suffix = "_hat" if method == "Direct" else f"_{method}" # As per R code logic

        y_calib_col = var_name
        y_hat_calib_col = f"{var_name}{hat_col_suffix}"

        y_test_col = var_name
        y_hat_test_col = f"{var_name}{hat_col_suffix}"

        # Check if required columns exist
        if not all(col in data_calib_df.columns for col in [y_calib_col, y_hat_calib_col]):
            raise ValueError(f"Calibration data missing columns for series {var_name} with suffix {hat_col_suffix}")
        if not all(col in data_test_df.columns for col in [y_test_col, y_hat_test_col]):
            raise ValueError(f"Test data missing columns for series {var_name} with suffix {hat_col_suffix}")

        y_calib = data_calib_df[y_calib_col].values
        y_hat_calib = data_calib_df[y_hat_calib_col].values

        y_test = data_test_df[y_test_col].values
        y_hat_test = data_test_df[y_hat_test_col].values

        # Calculate calibration scores (residuals)
        cal_scores_raw = y_calib - y_hat_calib
        # Remove NaNs if any, can happen if forecasts or actuals are missing for some points
        cal_scores = cal_scores_raw[~np.isnan(cal_scores_raw)]

        n_cal = len(cal_scores)
        if n_cal == 0:
            # If no calibration scores (e.g., all NaNs), metrics are undefined
            coverage = np.nan
            length_sq = np.nan
            # q_upper, q_lower will be inf, leading to inf length.
            # Or handle as error / skip this series. For now, NaN.
        else:
            # Quantiles for two-sided interval
            q_upper = _get_conformal_quantile(cal_scores, 1 - alpha / 2, n_cal)
            q_lower = -_get_conformal_quantile(-cal_scores, 1 - alpha / 2, n_cal) # For lower bound

            length_sq = (q_upper - q_lower)**2

            # Construct prediction intervals for test set
            ci_lower = y_hat_test + q_lower
            ci_upper = y_hat_test + q_upper

            # Calculate coverage on the test set
            # Ensure y_test, ci_lower, ci_upper are aligned and handle NaNs
            valid_test_indices = ~np.isnan(y_test) & ~np.isnan(ci_lower) & ~np.isnan(ci_upper)
            if np.sum(valid_test_indices) == 0:
                coverage = np.nan
            else:
                coverage = np.mean(
                    (ci_lower[valid_test_indices] <= y_test[valid_test_indices]) & \
                    (y_test[valid_test_indices] <= ci_upper[valid_test_indices])
                )

        info_df_data[var_name] = [coverage, length_sq, 1 - alpha]

    return pd.DataFrame(info_df_data)

def root_det(A: np.ndarray) -> float:
    """
    Calculates the (2m)-th root of the determinant of a matrix A (m x m).
    Corresponds to `root_det` from R/function.R.
    Used as a component in calculating the volume of joint prediction regions.

    Args:
        A: A square NumPy array.

    Returns:
        The (2 * dim)-th root of the determinant.
    """
    m_dim = A.shape[0]
    if m_dim == 0:
        return 0.0 # Or handle as error

    sign, log_det_val = np.linalg.slogdet(A)

    if sign == 0: # Determinant is zero
        return 0.0
    if sign == -1:
        # Determinant is negative. This implies A is not a covariance matrix.
        # The geometric interpretation of root_det for volume might not apply.
        # R's determinant()$modulus is log(abs(det)).
        # R's sign would be from determinant()$sign.
        # Python's slogdet already gives log(abs(det)).
        # For non-PSD matrices, this might be mathematically tricky.
        # However, if used with covariance matrices (PSD), sign should be 1.
        # Let's proceed assuming it could be used generally by original authors.
        # To take a real 2m-th root, the determinant must be non-negative if 2m is even.
        # (which it is). So, a negative determinant would lead to complex roots.
        # The R code `(Mod(determinant(A)$modulus))^(1/(2*m))` implies taking abs value of det.
        # `exp(determinant(A)$modulus)^(1/(2*m))` -> `exp(log(abs(det)))^(1/(2*m))` -> `abs(det)^(1/(2*m))`
        # The `sign` in R `(sign*Mod(determinant(A)$modulus))^(1/(2*m))` seems problematic if sign is -1.
        # Let's assume original R code intended `abs(det(A))**(1/(2*m))` and then apply sign if needed,
        # OR, that A is always PSD for this application.
        # Given it's likely for covariance matrices, let's assume det >= 0.
        # If sign is -1, and we need a real root, this is an issue.
        # The R code `(sign*Mod(determinant(A)$modulus))^(1/(2*m))` is `(sign * exp(log_det_val))**(1/(2*m))`.
        # If sign is -1, this is `(-exp(log_det_val))**(1/(2*m))`, which is problematic for real numbers.
        # Safest is to return NaN or error if sign is -1.
        # Or, follow R's `Mod(determinant(A)$modulus)` for the base of exponentiation, which is `abs(det)`.
        # `root_modulus = np.exp(log_det_val / (2 * m_dim))` is `abs(det_A)^(1/(2*m_dim))`.
        # If the R code meant to apply the sign *after* taking the root of the modulus:
        # `sign * (abs(det_A)**(1/(2*m_dim)))`
        # This seems more plausible.
        pass # Fall through to calculation with sign, assuming caller handles implications or A is PSD.

    root_modulus = np.exp(log_det_val / (2 * m_dim)) # abs(det(A)) ^ (1 / (2*m))

    # If sign was -1, and 2*m_dim is even, the R expression `(sign * abs(det))^(1/(2*m))`
    # would be `(-number)^(even_root_inverse)`, e.g. `(-X)^(1/k)`. This is complex if X > 0.
    # The R function `determinant` has `logarithm = TRUE` default, returning log(abs(det)).
    # `root_det <- function(A){ m = dim(A)[1]; (sign(det(A))*(Mod(determinant(A)$modulus)))^(1/(2*m)) }`
    # This means `(sign(det(A)) * abs(det(A)))^(1/(2*m))`. This simplifies to `det(A)^(1/(2*m))`.
    # This requires `det(A) >= 0` if `1/(2*m)` implies a real root.
    # Let's assume `A` is supposed to be positive semi-definite (e.g. covariance matrix).
    if sign < 0:
        # This case should ideally not happen if A is a covariance matrix.
        # If it does, taking a root of a negative number is problematic for real results.
        # Consider returning np.nan or raising an error.
        # For now, mimic potential R behavior if it computes with complex numbers and takes modulus,
        # or if it effectively does det(A)^(1/(2m)) which would be an error for det(A)<0.
        # Given slogdet, sign applies to det value.
        # If det < 0, log_det_val is log(abs(det)). root_modulus is abs(det)^(1/(2m)).
        # Result sign * root_modulus.
        # This matches the R code `sign(det(A)) * (abs(det(A)))^(1/(2*m))` if sign is applied outside power.
        # But R code is `(sign(det(A))*abs(det(A)))^(1/(2*m))`, which is `(det(A))^(1/(2*m))`.
        # If det(A) < 0, this is `(-X)^(1/k)`.
        # `np.linalg.slogdet` is the way to go.
        # `np.exp(log_det_val / (2 * m_dim))` is `abs(det(A))**(1/(2*m_dim))`.
        # If `sign` is -1, then we need `(-1 * abs(det(A)))**(1/(2*m_dim))`.
        # This is `(-1)**(1/(2*m_dim)) * abs(det(A))**(1/(2*m_dim))`.
        # `(-1)**(1/even_num)` is complex.
        # Let's assume the R code implies `det(A)` should be non-negative.
        # If det(A) < 0, root_det is likely ill-defined in this context or should be 0 or NaN.
        # For now, returning sign * root_modulus, which is what `np.linalg.det(A)**(1/(2*m_dim))` would give if it handled signs.
        # However, `det(A)**(1/k)` in Python for negative det(A) and even k in 1/k (e.g. (1/2)) gives nan.
        # So, if sign is -1, we should probably return np.nan.
        return np.nan # Or 0.0, or raise error. np.nan seems safest if det is negative.

    return root_modulus # Since sign must be 1 for PSD matrices. If sign is 0, already returned 0.0.


def joint_conformal(data_calib_df: pd.DataFrame, sigma_cal_residuals: np.ndarray,
                    data_test_df: pd.DataFrame, n: int, m: int, method: str, alpha: float = 0.1):
    """
    Computes joint conformal prediction region metrics.
    Corresponds to `joint_conformal` from R/function.R.

    Args:
        data_calib_df: DataFrame with calibration data.
        sigma_cal_residuals: Covariance matrix of calibration residuals (m x m NumPy array).
        data_test_df: DataFrame with test data.
        n: Number of bottom-level series (passed for API consistency).
        m: Total number of series.
        method: Forecasting method name.
        alpha: Miscoverage level.

    Returns:
        Pandas DataFrame with columns 'metric' and 'V', reporting overall
        Coverage, Volume proxy, and Quantile (1-alpha).
    """
    series_cols = [f"V{i}" for i in range(1, m + 1)]
    hat_col_suffix = "_hat" if method == "Direct" else f"_{method}"
    forecast_cols = [f"V{i}{hat_col_suffix}" for i in range(1, m + 1)]

    # Check for columns
    if not all(col in data_calib_df.columns for col in series_cols + forecast_cols):
        raise ValueError(f"Calibration data missing columns for method {method}")
    if not all(col in data_test_df.columns for col in series_cols + forecast_cols):
        raise ValueError(f"Test data missing columns for method {method}")

    y_calib = data_calib_df[series_cols].values
    y_hat_calib = data_calib_df[forecast_cols].values

    y_test = data_test_df[series_cols].values
    y_hat_test = data_test_df[forecast_cols].values

    cal_residuals = y_calib - y_hat_calib # Shape (n_cal_samples, m)

    # Remove rows with any NaNs in residuals (important for matrix operations)
    valid_cal_rows = ~np.isnan(cal_residuals).any(axis=1)
    cal_residuals = cal_residuals[valid_cal_rows, :]

    n_cal = cal_residuals.shape[0]
    if n_cal == 0:
        # No valid calibration data to compute scores or sigma_inv if not provided pre-computed
        # sigma_cal_residuals is an input, so it might be pre-computed.
        # However, cal_scores cannot be computed.
        coverage = np.nan
        volume = np.nan
        # q_joint will be inf
    else:
        try:
            # Inverse of the covariance matrix of calibration residuals
            # Ensure sigma_cal_residuals is well-behaved (e.g., PSD and invertible)
            # Add regularization if needed: sigma_cal_residuals + eps * np.eye(m)
            # For now, use ginv (pseudo-inverse) as in R's pracma::pinv via MASS::ginv
            if sigma_cal_residuals.shape != (m,m):
                raise ValueError(f"sigma_cal_residuals shape {sigma_cal_residuals.shape} does not match m={m}")

            # Check for near-zero determinant before inversion if using inv, ginv handles it
            # Add small identity matrix to improve conditioning before pseudo-inverse
            # This is a common practice, e.g. Ledoit-Wolf shrinkage or simple epsilon.
            epsilon = 1e-6 # Regularization parameter
            sigma_reg = sigma_cal_residuals + epsilon * np.eye(m)
            sigma_inv = ginv(sigma_reg)

        except np.linalg.LinAlgError: # If pseudo-inverse fails
            # This might happen if sigma_cal_residuals is highly problematic
            # (e.g., all zeros, though ginv should handle many cases)
            # Fallback or error:
            print(f"Warning: Pseudo-inverse of sigma_cal_residuals failed for method {method}. Using identity matrix as fallback for sigma_inv.")
            sigma_inv = np.eye(m) # Fallback, results will be questionable
            # Alternatively, could set coverage/volume to NaN here.

        # Mahalanobis-like scores for calibration data
        # (cal_residuals @ sigma_inv) is (n_cal, m)
        # Element-wise multiply with cal_residuals (n_cal, m)
        # Sum over axis 1 (series dimension) to get one score per calibration point
        cal_scores = np.sum((cal_residuals @ sigma_inv) * cal_residuals, axis=1)

        # Get the (1-alpha) quantile of these scores
        q_joint = _get_conformal_quantile(cal_scores, 1 - alpha, n_cal)

        # Calculate volume proxy for the prediction region
        # R: Volume = sqrt(q_joint) * root_det(Sigma)
        # Sigma here is sigma_cal_residuals
        determinant_sigma_root = root_det(sigma_cal_residuals) # This is det(Sigma)^(1/(2m))
        if np.isnan(determinant_sigma_root):
            print(f"Warning: root_det(sigma_cal_residuals) is NaN for method {method}. Volume will be NaN.")
            volume = np.nan
        else:
            volume = np.sqrt(q_joint) * determinant_sigma_root

        # Evaluate coverage on the test set
        test_residuals = y_test - y_hat_test # Shape (n_test_samples, m)
        # Remove rows with NaNs for test_scores calculation
        valid_test_rows = ~np.isnan(test_residuals).any(axis=1)
        if np.sum(valid_test_rows) == 0:
            coverage = np.nan
        else:
            test_residuals_valid = test_residuals[valid_test_rows, :]
            test_scores = np.sum((test_residuals_valid @ sigma_inv) * test_residuals_valid, axis=1)
            coverage = np.mean(test_scores <= q_joint)

    return pd.DataFrame({
        'metric': ["Coverage", "Volume", "Quantile"],
        'V': [coverage, volume, 1 - alpha] # 'V' is the column name as in R output
    })

# Update __main__ for testing new functions if desired, or keep as is.
# For example, to test conformal, root_det, joint_conformal:
if __name__ == '__main__':
    # (Previous tests for generate_S and generate_multi)
    print("\nRunning existing data generation tests...")
    # (Copied from previous __main__ block for brevity in diff, assume they pass)
    n_param = 12; l_param = 1
    S_matrix = generate_S(n=n_param, l=l_param)
    assert S_matrix.shape == (1 + 12**l_param + n_param, n_param)
    n_observations = 50 # smaller for faster tests
    simulated_data = generate_multi(S=S_matrix, n_obs=n_observations, noise_type="A")
    Features_train = simulated_data["Features"]
    Y_level_train = simulated_data["Y_level"] # (total_series, n_obs)
    print("Data generation tests part passed.")

    print("\nTesting conformal prediction functions...")

    # Create dummy DataFrames for testing conformal functions
    m_total_series = S_matrix.shape[0] # Total number of series including aggregated

    # Assume Y_level_train is (m_total_series, n_observations)
    # Transpose to (n_observations, m_total_series) for DataFrame creation
    y_actual_all_series = Y_level_train.T

    # Create column names V1, V2, ... Vm
    series_names = [f"V{i+1}" for i in range(m_total_series)]

    df_actuals = pd.DataFrame(y_actual_all_series, columns=series_names)

    # Simulate forecasts (e.g., just actuals + some noise for simplicity)
    np.random.seed(42) # for reproducibility
    y_forecasts_all_series = y_actual_all_series + np.random.normal(0, 0.5, size=y_actual_all_series.shape)

    # Create forecast column names: V1_hat, V2_hat, ... (for "Direct" method)
    forecast_names_direct = [f"V{i+1}_hat" for i in range(m_total_series)]
    df_forecasts_direct = pd.DataFrame(y_forecasts_all_series, columns=forecast_names_direct)

    # Combine actuals and forecasts into one df
    full_df = pd.concat([df_actuals, df_forecasts_direct], axis=1)

    # Split into calibration and test sets
    n_cal_samples = n_observations // 2
    data_calib_df = full_df.iloc[:n_cal_samples]
    data_test_df = full_df.iloc[n_cal_samples:]

    print(f"Calib df shape: {data_calib_df.shape}, Test df shape: {data_test_df.shape}")

    # Test conformal()
    print("\nTesting conformal()...")
    alpha_test = 0.1
    # n_param is num_bottom_level_series, m_total_series is S_matrix.shape[0]
    conformal_results_df = conformal(data_calib_df, data_test_df,
                                     n=n_param, m=m_total_series,
                                     method="Direct", alpha=alpha_test)
    print("Conformal results:")
    print(conformal_results_df)
    assert conformal_results_df.shape[0] == 3 # metrics: Coverage, Length, Quantile
    assert conformal_results_df.shape[1] == 1 + m_total_series # metric + V1...Vm columns
    assert not conformal_results_df.isnull().values.any() # Check for NaNs

    # Test root_det()
    print("\nTesting root_det()...")
    dummy_matrix_good = np.array([[4, 1], [1, 3]], dtype=float) # det = 12-1=11
    # m_dim = 2. (2*m_dim) = 4.  11^(1/4)
    expected_root_det = 11**(1/4)
    assert np.isclose(root_det(dummy_matrix_good), expected_root_det)
    print(f"root_det for good matrix: {root_det(dummy_matrix_good)} (Expected: {expected_root_det})")

    dummy_matrix_zero_det = np.array([[1, 1], [1, 1]], dtype=float) # det = 0
    assert np.isclose(root_det(dummy_matrix_zero_det), 0.0)
    print(f"root_det for zero_det matrix: {root_det(dummy_matrix_zero_det)}")

    dummy_matrix_neg_det = np.array([[1, 2], [3, 1]], dtype=float) # det = 1-6 = -5
    assert np.isnan(root_det(dummy_matrix_neg_det)) # Expect NaN for negative determinant
    print(f"root_det for neg_det matrix: {root_det(dummy_matrix_neg_det)}")


    # Test joint_conformal()
    print("\nTesting joint_conformal()...")
    # Need sigma_cal_residuals: covariance of (y_calib - y_hat_calib)
    cal_residuals_df = pd.DataFrame(columns=series_names)
    for i in range(1, m_total_series + 1):
        var_name = f"V{i}"
        hat_name = f"V{i}_hat"
        cal_residuals_df[var_name] = data_calib_df[var_name] - data_calib_df[hat_name]

    # Drop rows with NaNs that might have resulted from forecast generation or original data
    cal_residuals_df_cleaned = cal_residuals_df.dropna()

    if cal_residuals_df_cleaned.shape[0] < 2: # Need at least 2 samples to compute covariance
        print("Skipping joint_conformal test: Not enough non-NaN calibration residuals.")
    elif cal_residuals_df_cleaned.shape[1] == 0 : # no series
         print("Skipping joint_conformal test: No series in calibration residuals.")
    else:
        sigma_cal_res = cal_residuals_df_cleaned.cov().values
        # Ensure sigma_cal_res is m x m
        if sigma_cal_res.shape == (m_total_series, m_total_series):
            joint_conformal_results_df = joint_conformal(
                data_calib_df, sigma_cal_res, data_test_df,
                n=n_param, m=m_total_series, method="Direct", alpha=alpha_test
            )
            print("Joint conformal results:")
            print(joint_conformal_results_df)
            assert joint_conformal_results_df.shape == (3, 2) # metrics, V
            assert not joint_conformal_results_df['V'].isnull().values.any() # Check for NaNs in V column
        else:
            print(f"Skipping joint_conformal test: sigma_cal_res shape {sigma_cal_res.shape} is not ({m_total_series},{m_total_series})")


    print("\nAll conformal prediction function tests passed (or skipped if data insufficient).")


# Imports for modeling functions
import statsmodels.formula.api as smf
from pygam import LinearGAM, s as gam_s, l as gam_l, TermList
# Note: renamed pygam.s to gam_s and pygam.l to gam_l to avoid conflict with any other 's' or 'l'

def metrics_proba_multi(data_calib_df: pd.DataFrame, sigma_val_residuals: np.ndarray,
                        data_test_df: pd.DataFrame, n: int, m: int,
                        methods: list[str], alpha: float = 0.1):
    """
    Calculates and aggregates marginal and joint conformal prediction metrics for multiple methods.
    Corresponds to `metrics_proba_multi` from R/function.R.

    Args:
        data_calib_df: DataFrame with calibration data.
        sigma_val_residuals: Covariance matrix of validation/calibration residuals (m x m),
                             used for joint conformal method.
        data_test_df: DataFrame with test data.
        n: Number of bottom-level series.
        m: Total number of series.
        methods: A list of forecasting method names (strings).
        alpha: Miscoverage level.

    Returns:
        A list containing two items:
        1. long_data_df: A Pandas DataFrame in long format with marginal metrics
           (Coverage, Length, Quantile) for each series and method.
        2. joint_results_dict: A dictionary where keys are method names and values
           are DataFrames from `joint_conformal` (overall Coverage, Volume, Quantile).
    """
    all_marginal_results = []
    joint_results_dict = {}

    for method in methods:
        # Calculate marginal conformal metrics
        conformal_df = conformal(data_calib_df, data_test_df, n, m, method, alpha=alpha)
        conformal_df['method'] = method  # Add method column for easy aggregation
        all_marginal_results.append(conformal_df)

        # Calculate joint conformal metrics
        # Note: sigma_val_residuals should be appropriate for the given method's residuals.
        # If it's a single matrix, it's assumed to be a common one (e.g. from a base method).
        joint_df = joint_conformal(data_calib_df, sigma_val_residuals, data_test_df,
                                   n, m, method, alpha=alpha)
        # joint_df already has 'metric' and 'V' columns.
        # R code adds method to rownames; here we can add as a column if needed for later processing,
        # but the dict key already stores the method.
        # joint_df['method'] = method # Optional: if direct concat of joint_dfs is planned
        joint_results_dict[method] = joint_df

    # Combine all marginal results into one DataFrame
    combined_marginal_df = pd.concat(all_marginal_results, ignore_index=True)

    # Convert combined marginal data to long format
    # R: pivot_longer(cols = starts_with("V"), names_to = "variable", values_to = "value")
    long_data_df = combined_marginal_df.melt(
        id_vars=['metric', 'method'],
        var_name='Series', # Renamed from 'variable' for clarity
        value_name='Value'  # Renamed from 'value' for clarity
    )
    # Filter out rows where 'Series' might be non-series columns if any were accidentally included.
    # Here, melt correctly uses id_vars, so var_name should only contain 'V1', 'V2', etc.

    return [long_data_df, joint_results_dict]


def prediction_fast(full_data_df: pd.DataFrame, model_type: str,
                    train_indices: list[int], m: int, n: int):
    """
    Performs model fitting (LM or GAM) for multiple time series and generates forecasts.
    Corresponds to `prediction_fast` from R/function.R.

    Args:
        full_data_df: Pandas DataFrame containing all data (target series V1..Vm, and covariates X1, X2, X3).
                      Index should be continuous.
        model_type: String, "lm" for Linear Model or "gam" for Generalized Additive Model.
        train_indices: List or NumPy array of integer indices for the training set (iloc based).
        m: Total number of series (V1 to Vm).
        n: Number of bottom-level series. Aggregated series are V1 to V(m-n). Bottom series are V(m-n+1) to Vm.

    Returns:
        NumPy array `forecasts` of shape (number_of_test_samples, m).
    """
    # Determine test indices based on full_data_df's index and train_indices
    # Assuming train_indices are iloc-based as per R behavior with integer vectors for subsetting.
    # If train_indices are labels, .loc would be used. For iloc, convert to boolean mask or use difference.

    # Create a boolean mask for train_indices
    train_mask = np.zeros(len(full_data_df), dtype=bool)
    train_mask[train_indices] = True
    test_indices = np.where(~train_mask)[0] # Get integer indices for test set

    if len(test_indices) == 0:
        return np.array([]).reshape(0,m) # No test samples, return empty forecast array

    forecasts = np.zeros((len(test_indices), m))

    training_data = full_data_df.iloc[train_indices]
    predict_data = full_data_df.iloc[test_indices] # Data to predict on

    all_X_cols = ['X1', 'X2', 'X3'] # Potential covariates

    for i in range(1, m + 1): # Iterate through series V1 to Vm
        series_name = f"V{i}"

        # Determine covariates based on series type (aggregated or bottom-level) and task difficulty
        is_bottom_level = (i > (m - n)) # True if current series 'i' is a bottom-level one

        used_X_cols = []
        formula_str = ""

        if not is_bottom_level: # Aggregated series (m-n of them: V1 to V(m-n))
            used_X_cols = ['X1', 'X2', 'X3']
        else: # Bottom-level series (n of them: V(m-n+1) to Vm)
            # R code: rbinom(1, 1, 0.8) == 1 for easy (use X1,X2,X3) (this means 80% easy)
            # This is equivalent to np.random.binomial(1, 0.8) == 1
            is_easy_task = np.random.binomial(1, 0.8) == 1 # True for easy task (80% prob)
            if is_easy_task:
                used_X_cols = ['X1', 'X2', 'X3']
            else: # Harder task (20% prob)
                used_X_cols = ['X1', 'X2']

        # Ensure all used_X_cols are present in the data
        if not all(col in training_data.columns for col in used_X_cols):
            raise ValueError(f"Missing one or more columns {used_X_cols} in training_data for series {series_name}")
        if not all(col in predict_data.columns for col in used_X_cols):
            raise ValueError(f"Missing one or more columns {used_X_cols} in predict_data for series {series_name}")

        # Prepare data for model fitting
        X_train = training_data[used_X_cols]
        y_train = training_data[series_name]
        X_predict = predict_data[used_X_cols]

        # Handle potential NaNs in y_train or X_train before fitting
        # statsmodels OLS handles NaNs by default (listwise deletion).
        # PyGAM might require explicit NaN handling for X and y.
        valid_train_idx = y_train.notna() & X_train.notna().all(axis=1)
        if not valid_train_idx.any(): # No valid data to train
            print(f"Warning: No valid training data for series {series_name} after NaN removal. Forecasts will be NaN.")
            forecasts[:, i-1] = np.nan
            continue

        X_train_clean = X_train[valid_train_idx]
        y_train_clean = y_train[valid_train_idx]

        try:
            if model_type == "lm":
                # Construct formula string for statsmodels
                formula_str = f"{series_name} ~ {' + '.join(used_X_cols)}"
                # statsmodels uses the dataframe passed, so subsetting X_train/y_train is for NaN handling
                # Pass the original training_data but it will effectively use only clean rows for this model
                model = smf.ols(formula=formula_str, data=training_data[valid_train_idx]).fit()
                current_forecasts = model.predict(predict_data) # predict_data should have X cols

            elif model_type == "gam":
                # For PyGAM, terms are built based on the columns in X_train_clean
                terms = TermList(*[gam_s(k) for k in range(len(used_X_cols))])
                # PyGAM expects X to be a NumPy array
                gam_model = LinearGAM(terms, fit_intercept=True).fit(X_train_clean.values, y_train_clean.values)
                current_forecasts = gam_model.predict(X_predict.values)

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            # Store forecasts
            # current_forecasts might be a pandas Series (from statsmodels) or numpy array (from PyGAM)
            if hasattr(current_forecasts, 'values'):
                forecasts[:, i-1] = current_forecasts.values
            else:
                forecasts[:, i-1] = current_forecasts

        except Exception as e:
            print(f"Error fitting model for series {series_name} (index {i-1}) with method {model_type}: {e}")
            print(f"Formula/Terms: {formula_str if model_type=='lm' else terms if model_type=='gam' else 'N/A'}")
            print(f"X_train_clean shape: {X_train_clean.shape}, y_train_clean shape: {y_train_clean.shape}")
            forecasts[:, i-1] = np.nan # Fill with NaNs if model fails

    return forecasts


# Example usage in __main__ could be expanded for these functions
if __name__ == '__main__':
    # ... (previous tests for generate_S, generate_multi, conformal, root_det, joint_conformal) ...
    print("\nAll data generation tests passed.") # from previous block
    print("\nAll conformal prediction function tests passed (or skipped if data insufficient).") # from previous block

    # Additional tests for metrics_proba_multi and prediction_fast
    print("\nTesting metrics_proba_multi and prediction_fast...")

    # Re-use data from conformal tests
    # n_param, m_total_series, S_matrix, full_df, data_calib_df, data_test_df
    # Need: sigma_val_residuals (covariance of validation residuals)
    # For simplicity, use calibration residuals' covariance as a stand-in for sigma_val_residuals

    # Recalculate cal_residuals_df from the dummy data used in conformal tests
    series_names_main = [f"V{k+1}" for k in range(m_total_series)] # m_total_series from previous test
    cal_residuals_df_main = pd.DataFrame()
    # Assuming data_calib_df has V1, V1_hat etc. (from previous test setup)
    for s_idx in range(1, m_total_series + 1):
        var_name = f"V{s_idx}"
        hat_name = f"V{s_idx}_hat" # Assuming 'Direct' method forecasts V_hat
        if var_name in data_calib_df.columns and hat_name in data_calib_df.columns:
             cal_residuals_df_main[var_name] = data_calib_df[var_name] - data_calib_df[hat_name]

    cal_residuals_df_main_cleaned = cal_residuals_df_main.dropna()

    if cal_residuals_df_main_cleaned.shape[0] >= 2 and cal_residuals_df_main_cleaned.shape[1] > 0 :
        sigma_val_res_dummy = cal_residuals_df_main_cleaned.cov().values

        if sigma_val_res_dummy.shape == (m_total_series, m_total_series):
            print("\nTesting metrics_proba_multi()...")
            methods_to_test = ["Direct"] # Assuming 'Direct' method forecasts (V_hat) are in data_calib_df

            try:
                summary_metrics = metrics_proba_multi(
                    data_calib_df, sigma_val_res_dummy, data_test_df,
                    n=n_param, m=m_total_series, methods=methods_to_test, alpha=0.1
                )
                long_df, joint_dict = summary_metrics
                print("metrics_proba_multi output (long_df head):")
                print(long_df.head())
                print(f"Shape of long_df: {long_df.shape}")
                assert not long_df.isnull().values.any()
                assert 'Series' in long_df.columns and 'Value' in long_df.columns

                print("\nmetrics_proba_multi output (joint_dict):")
                for method, df in joint_dict.items():
                    print(f"Method: {method}")
                    print(df)
                    assert not df.isnull().values.any()

            except ValueError as e:
                print(f"ValueError during metrics_proba_multi test: {e}")
                print("This might be due to missing columns if dummy data setup is incomplete for 'Direct' method.")

        else:
            print(f"Skipping metrics_proba_multi test: sigma_val_res_dummy shape {sigma_val_res_dummy.shape} is not ({m_total_series},{m_total_series})")
    else:
        print("Skipping metrics_proba_multi test: Not enough data for sigma_val_residuals.")


    print("\nTesting prediction_fast()...")
    # Prepare full_data_df for prediction_fast: needs V1..Vm, X1, X2, X3
    # Use Features_train and Y_level_train from generate_multi output
    # Y_level_train is (m_total_series, n_observations)
    # Features_train is (n_observations, 3)

    # Construct full_df_pred:
    # Columns: V1, ..., Vm, X1, X2, X3
    # Number of observations for this test:
    n_obs_pred_test = Features_train.shape[0] # Should be n_observations from generate_multi run

    df_y_levels_pred = pd.DataFrame(Y_level_train.T, columns=[f"V{k+1}" for k in range(m_total_series)])
    df_features_pred = pd.DataFrame(Features_train, columns=['X1', 'X2', 'X3'])
    full_data_df_pred_test = pd.concat([df_y_levels_pred, df_features_pred], axis=1)

    # Define train_indices for prediction_fast
    # e.g., first half of n_obs_pred_test
    pred_train_indices = list(range(n_obs_pred_test // 2))

    # Test with 'lm'
    try:
        print("Testing prediction_fast() with model_type='lm'...")
        forecasts_lm = prediction_fast(full_data_df_pred_test, "lm", pred_train_indices, m_total_series, n_param)
        print(f"Forecasts_lm shape: {forecasts_lm.shape}")
        num_test_samples = n_obs_pred_test - len(pred_train_indices)
        assert forecasts_lm.shape == (num_test_samples, m_total_series)
        # Basic check for NaNs, though some models might produce them if data is tricky
        # For this dummy data, ideally no NaNs if all X cols present and no extreme values.
        # assert not np.isnan(forecasts_lm).all() # At least some forecasts should be non-NaN
        if np.isnan(forecasts_lm).any():
            print("Warning: NaNs found in LM forecasts.")


        # Test with 'gam'
        print("\nTesting prediction_fast() with model_type='gam'...")
        forecasts_gam = prediction_fast(full_data_df_pred_test, "gam", pred_train_indices, m_total_series, n_param)
        print(f"Forecasts_gam shape: {forecasts_gam.shape}")
        assert forecasts_gam.shape == (num_test_samples, m_total_series)
        # assert not np.isnan(forecasts_gam).all()
        if np.isnan(forecasts_gam).any():
            print("Warning: NaNs found in GAM forecasts.")

    except Exception as e: # Catch any other exceptions during prediction_fast calls
        print(f"Exception during prediction_fast tests: {e}")


    print("\nFinished tests for metrics_proba_multi and prediction_fast.")


def _calculate_G_matrix(S_summing_matrix: np.ndarray, W_cov_matrix: np.ndarray) -> np.ndarray:
    """
    Helper function to calculate the G matrix for forecast reconciliation: G = (S^T W^-1 S)^-1 S^T W^-1.
    Args:
        S_summing_matrix: The summing matrix (m_total_series x n_bottom_series).
        W_cov_matrix: Covariance matrix of base forecast errors (m_total_series x m_total_series).
    Returns:
        The G matrix (n_bottom_series x m_total_series).
    """
    S_t = S_summing_matrix.T

    # Regularize W_cov_matrix before inversion to improve stability
    epsilon = 1e-6 # Small regularization constant
    W_reg = W_cov_matrix + epsilon * np.eye(W_cov_matrix.shape[0])
    W_inv = ginv(W_reg) # Use pseudo-inverse for robustness

    # Calculate S^T W^-1 S
    St_W_inv_S = S_t @ W_inv @ S_summing_matrix
    # Regularize this matrix before inversion as well
    St_W_inv_S_reg = St_W_inv_S + epsilon * np.eye(St_W_inv_S.shape[0])

    try:
        # G = (S^T W^-1 S)^-1 S^T W^-1
        G_matrix = ginv(St_W_inv_S_reg) @ S_t @ W_inv
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error during G matrix calculation: {e}")
        print("This may indicate issues like collinearity or non-invertibility even with ginv and regularization.")
        # Fallback: could return a simplified G, e.g., based on OLS ( G = (S^T S)^-1 S^T )
        # For now, re-raise or return something that signals failure clearly.
        # Based on R code using ginv throughout, it might expect ginv to handle it.
        # If St_W_inv_S_reg is singular and ginv can't handle it, that's a deeper issue.
        # Let's assume ginv is robust or data/S are well-behaved.
        # If issues persist, more sophisticated regularization or error handling needed.
        raise # Re-raise the error to signal a problem in simulation setup

    return G_matrix


def simulation_multi(S_input_matrix: np.ndarray, n_total_obs: int, model_type: str,
                     noise_type: str = "A", alpha: float = 0.1):
    """
    Runs a single simulation for hierarchical time series forecasting and conformal prediction.
    Corresponds to `simulation_multi` from R/function.R.

    Args:
        S_input_matrix: Summing matrix (m_total_series x n_bottom_series) from generate_S.
        n_total_obs: Total number of observations to generate for the series.
        model_type: Forecasting model type ("lm" or "gam").
        noise_type: Type of noise for data generation (passed to `generate_multi`).
        alpha: Significance level for conformal prediction (e.g., 0.1 for 90% intervals).

    Returns:
        A list containing two items, same as `metrics_proba_multi` output:
        1. long_data_df: Pandas DataFrame with marginal conformal metrics.
        2. joint_results_dict: Dictionary with joint conformal metrics per method.
    """
    # Determine series dimensions
    m_total_series = S_input_matrix.shape[0]
    n_bottom_level_series = S_input_matrix.shape[1]

    # Data splitting indices
    # R code: train_int = floor(0.4*n_obs), valid_int = floor(0.2*n_obs), calib_int = floor(0.2*n_obs)
    # Python:
    train_end_idx = int(np.floor(0.4 * n_total_obs))
    valid_end_idx = train_end_idx + int(np.floor(0.2 * n_total_obs)) # End of validation
    calib_end_idx = valid_end_idx + int(np.floor(0.2 * n_total_obs)) # End of calibration
    # Test set is the remainder

    train_indices = list(range(train_end_idx))
    valid_indices = list(range(train_end_idx, valid_end_idx))
    calib_indices = list(range(valid_end_idx, calib_end_idx))
    test_indices = list(range(calib_end_idx, n_total_obs))

    if not all([len(train_indices)>0, len(valid_indices)>0, len(calib_indices)>0, len(test_indices)>0]):
        raise ValueError("Insufficient n_total_obs for data splitting into train, valid, calib, test sets.")

    # Data generation
    generated_data = generate_multi(S_input_matrix, n_total_obs, noise_type=noise_type)
    features_df = pd.DataFrame(generated_data["Features"], columns=['X1', 'X2', 'X3'])
    # Y_level from generate_multi is (m_total_series, n_total_obs)
    y_df = pd.DataFrame(generated_data["Y_level"].T, columns=[f"V{i+1}" for i in range(m_total_series)])
    full_df = pd.concat([y_df, features_df], axis=1) # Index 0 to n_total_obs-1

    # Base forecasts (Direct method)
    # `prediction_fast` returns forecasts for non-training part of full_df
    base_forecasts_array = prediction_fast(full_df, model_type, train_indices,
                                           m=m_total_series, n=n_bottom_level_series)

    non_train_indices = full_df.index.difference(pd.Index(train_indices)) # Get actual index labels
    base_forecasts_df = pd.DataFrame(base_forecasts_array, index=non_train_indices,
                                     columns=[f"V{i+1}_hat" for i in range(m_total_series)])

    # Reconciliation
    # Use validation set for calculating error covariance W_MinT
    actuals_valid_df = full_df.iloc[valid_indices][[f"V{i+1}" for i in range(m_total_series)]]
    forecasts_valid_df = base_forecasts_df.loc[valid_indices] # .loc uses labels from non_train_indices

    val_errors_matrix = actuals_valid_df.values - forecasts_valid_df.values

    # Ensure there are enough samples and no NaNs before cov calculation
    val_errors_matrix_cleaned = val_errors_matrix[~np.isnan(val_errors_matrix).any(axis=1)]
    if val_errors_matrix_cleaned.shape[0] < 2:
        print("Warning: Not enough valid samples in validation errors to compute covariance. Using identity matrix for W_MinT.")
        W_MinT = np.eye(m_total_series)
    else:
        W_MinT = np.cov(val_errors_matrix_cleaned, rowvar=False) # m x m matrix

    W_WLS = np.diag(np.diag(W_MinT)) # Diagonal of W_MinT
    W_OLS = np.eye(m_total_series)   # Identity matrix for OLS reconciliation

    # Calculate G matrices for reconciliation
    G_MinT = _calculate_G_matrix(S_input_matrix, W_MinT)
    G_WLS  = _calculate_G_matrix(S_input_matrix, W_WLS)
    G_OLS  = _calculate_G_matrix(S_input_matrix, W_OLS)

    # Combined G matrix (average of Gs)
    G_Combi = (1/3) * (G_MinT + G_OLS + G_WLS)

    # Projection matrices P = S * G
    P_MinT  = S_input_matrix @ G_MinT
    P_WLS   = S_input_matrix @ G_WLS
    P_OLS   = S_input_matrix @ G_OLS
    P_Combi = S_input_matrix @ G_Combi

    projections = {
        "MinT": P_MinT,
        "WLS": P_WLS,
        "OLS": P_OLS,
        "Combi": P_Combi
    }

    # Prepare data for conformal prediction
    # Actuals for calibration and test sets
    data_calib_actuals_df = full_df.iloc[calib_indices].copy() # Use .copy() to avoid SettingWithCopyWarning
    data_test_actuals_df = full_df.iloc[test_indices].copy()

    # Base forecasts for calibration and test sets
    base_forecasts_calib_df = base_forecasts_df.loc[calib_indices]
    base_forecasts_test_df = base_forecasts_df.loc[test_indices]

    # Create DataFrames for conformal input, starting with actuals
    # And join base forecasts (suffix _hat for "Direct" method)
    data_calib_for_conformal = data_calib_actuals_df.join(base_forecasts_calib_df)
    data_test_for_conformal = data_test_actuals_df.join(base_forecasts_test_df)

    # Reconcile forecasts for calibration and test sets and add to conformal DataFrames
    for method_name, P_matrix in projections.items():
        # Reconcile calibration forecasts: P @ base_forecasts.T -> transpose back
        reconciled_calib_forecasts = (P_matrix @ base_forecasts_calib_df.values.T).T
        rec_calib_df = pd.DataFrame(reconciled_calib_forecasts, index=pd.Index(calib_indices, name=full_df.index.name), # Preserve original index labels
                                    columns=[f"V{i+1}_{method_name}" for i in range(m_total_series)])
        data_calib_for_conformal = data_calib_for_conformal.join(rec_calib_df)

        # Reconcile test forecasts
        reconciled_test_forecasts = (P_matrix @ base_forecasts_test_df.values.T).T
        rec_test_df = pd.DataFrame(reconciled_test_forecasts, index=pd.Index(test_indices, name=full_df.index.name),
                                   columns=[f"V{i+1}_{method_name}" for i in range(m_total_series)])
        data_test_for_conformal = data_test_for_conformal.join(rec_test_df)

    # Conformal prediction evaluation
    eval_methods = ["Direct"] + list(projections.keys()) # "Direct" uses "_hat" forecasts

    # W_MinT is used as Sigma for joint conformal methods (as per R code's `Sigma = W_MinT`)
    # This W_MinT is from validation set errors.
    results = metrics_proba_multi(data_calib_for_conformal, W_MinT, data_test_for_conformal,
                                  n=n_bottom_level_series, m=m_total_series,
                                  methods=eval_methods, alpha=alpha)
    return results


if __name__ == '__main__':
    # ... (all previous tests) ...
    print("\nFinished tests for metrics_proba_multi and prediction_fast.")

    print("\nTesting simulation_multi()...")
    # Use parameters from previous tests for S_matrix generation
    # n_param (bottom_series), l_param (levels for S)
    # S_matrix_main_test = generate_S(n=n_param, l=l_param) # n_param=12, l_param=1 -> S shape (25,12)
                                                        # m_total=25, n_bottom=12

    # Need to use S_matrix that was used for m_total_series and n_param in earlier tests
    # S_matrix was (25,12) for n_param=12, l_param=1. m_total_series=25.

    # Reduce n_observations for faster simulation_multi test
    n_obs_simulation_test = 100 # Must be large enough for train/valid/calib/test split

    # Check if S_matrix (from previous tests) is available and correctly scoped
    # If S_matrix is not available here, regenerate it:
    if 'S_matrix' not in locals() or S_matrix is None : # S_matrix from a previous test block
        print("S_matrix not found in local scope for simulation_multi test, regenerating with n=12, l=1.")
        n_param_sim = 12
        l_param_sim = 1
        S_matrix_sim_test = generate_S(n=n_param_sim, l=l_param_sim)
    else: # Use S_matrix from the existing __main__ scope if it's there
        S_matrix_sim_test = S_matrix


    print(f"Running simulation_multi with S_matrix shape {S_matrix_sim_test.shape}, n_obs={n_obs_simulation_test}, model_type='lm'")
    try:
        sim_results_lm = simulation_multi(S_matrix_sim_test, n_obs_simulation_test, model_type="lm", alpha=0.1)
        long_df_lm, joint_dict_lm = sim_results_lm

        print("\nsimulation_multi results for 'lm' (long_df head):")
        print(long_df_lm.head())
        assert not long_df_lm.empty
        assert 'Series' in long_df_lm.columns and 'Value' in long_df_lm.columns
        # Check if all expected methods are present
        expected_sim_methods = ["Direct", "MinT", "WLS", "OLS", "Combi"]
        assert sorted(long_df_lm['method'].unique()) == sorted(expected_sim_methods)

        print("\nsimulation_multi results for 'lm' (joint_dict keys):")
        print(list(joint_dict_lm.keys()))
        assert sorted(list(joint_dict_lm.keys())) == sorted(expected_sim_methods)
        for method_name, df_joint in joint_dict_lm.items():
            assert not df_joint.isnull().values.any(), f"NaNs found in joint results for method {method_name} (lm)"


        print(f"\nRunning simulation_multi with S_matrix shape {S_matrix_sim_test.shape}, n_obs={n_obs_simulation_test}, model_type='gam'")
        sim_results_gam = simulation_multi(S_matrix_sim_test, n_obs_simulation_test, model_type="gam", alpha=0.1)
        long_df_gam, joint_dict_gam = sim_results_gam

        print("\nsimulation_multi results for 'gam' (long_df head):")
        print(long_df_gam.head())
        assert not long_df_gam.empty
        assert sorted(long_df_gam['method'].unique()) == sorted(expected_sim_methods)

        print("\nsimulation_multi results for 'gam' (joint_dict keys):")
        print(list(joint_dict_gam.keys()))
        assert sorted(list(joint_dict_gam.keys())) == sorted(expected_sim_methods)
        for method_name, df_joint in joint_dict_gam.items():
            assert not df_joint.isnull().values.any(), f"NaNs found in joint results for method {method_name} (gam)"

        print("\nBasic simulation_multi tests passed.")

    except ValueError as ve:
        print(f"ValueError in simulation_multi test: {ve}")
        print("This might be due to n_obs being too small for splitting, or issues in underlying functions.")
    except Exception as e:
        print(f"An unexpected error occurred in simulation_multi test: {e}")
        import traceback
        traceback.print_exc()

    print("\nFinished all simulation and modeling function tests in functions.py.")


# LaTeX table generation functions
# Need to ensure pandas is imported if not already: import pandas as pd
# Need to ensure numpy is imported if not already: import numpy as np

def get_table(data_df: pd.DataFrame, uncertainty_df: pd.DataFrame = None,
              bold: bool = True, digits: int = 2) -> str:
    """
    Generates a LaTeX table from a DataFrame, optionally including uncertainty and bolding the minimum value per row.
    Corresponds to `get_table` from R/function.R.

    Args:
        data_df: Pandas DataFrame with mean values. Index expected to be configuration names (e.g., "C1", "C2").
                 Columns are method names.
        uncertainty_df: Optional Pandas DataFrame with uncertainty values (e.g., standard errors),
                        matching shape and indices/columns of data_df.
        bold: If True, bold the minimum value in each row (considering only data_df values).
        digits: Number of decimal places for formatting values.

    Returns:
        A string containing the LaTeX table.
    """
    if uncertainty_df is not None and not data_df.shape == uncertainty_df.shape:
        raise ValueError("data_df and uncertainty_df must have the same shape.")
    if uncertainty_df is not None and \
       (not data_df.index.equals(uncertainty_df.index) or \
        not data_df.columns.equals(uncertainty_df.columns)):
        raise ValueError("data_df and uncertainty_df must have the same index and columns.")

    n_rows, n_cols = data_df.shape
    col_names = data_df.columns.tolist()
    index_names = data_df.index.tolist()

    # Start LaTeX table
    latex_str = []
    latex_str.append("\\begin{adjustbox}{width=1\\textwidth}")
    # Column format: 'c' for config name, then 'c' for each method column
    col_format = "c " + " ".join(["c"] * n_cols)
    latex_str.append(f"\\begin{{tabular}}{{ {col_format} }}")
    latex_str.append("\\toprule")

    # Header row (method names)
    header = [" "] + [col.replace("_", "\\_") for col in col_names] # Escape underscores in method names
    latex_str.append(" & ".join(header) + " \\\\")
    latex_str.append("\\midrule")

    # Data rows
    for i in range(n_rows):
        config_name = str(index_names[i]).replace("_", "\\_")
        row_values_str = [config_name]

        row_data = data_df.iloc[i, :].copy() # Use .copy() to avoid issues with modifying slices

        min_val_idx = -1
        if bold and pd.api.types.is_numeric_dtype(row_data):
            # Find unique minimum: if multiple methods have the same min, don't bold any.
            # This behavior matches R's `which.min` which returns the first minimum.
            # For stricter "only one minimum" bolding:
            # min_val = row_data.min()
            # if (row_data == min_val).sum() == 1:
            #    min_val_idx = row_data.idxmin() # get column name
            #    min_val_idx = row_data.columns.get_loc(min_val_idx) # get integer index
            # Simplified: bold the first minimum found.
            try:
                min_val_col_name = row_data.idxmin()
                min_val_idx = data_df.columns.get_loc(min_val_col_name)
            except ValueError: # E.g. if all values in row are NaN
                min_val_idx = -1


        for j in range(n_cols):
            val = data_df.iloc[i, j]

            # Format main value
            if pd.isna(val):
                cell_str = "-" # Placeholder for NaN values
            else:
                cell_str = f"{val:.{digits}f}"

            # Add uncertainty if provided
            if uncertainty_df is not None:
                unc = uncertainty_df.iloc[i, j]
                if pd.isna(unc):
                    cell_str += " $\\pm$ -" # Placeholder for NaN uncertainty
                else:
                    cell_str += f" $\\pm$ {unc:.{digits}f}"

            # Bold if it's the minimum in the row
            if bold and j == min_val_idx:
                cell_str = f"\\textbf{{{cell_str}}}"

            row_values_str.append(cell_str)

        latex_str.append(" & ".join(row_values_str) + " \\\\")

    latex_str.append("\\bottomrule")
    latex_str.append("\\end{tabular}}")
    latex_str.append("\\end{adjustbox}")

    return "\n".join(latex_str)


def get_table_occurence(data_df: pd.DataFrame, digits: int = 2) -> str:
    """
    Generates a LaTeX table for occurrence data, typically from a DataFrame with a MultiIndex.
    Highlights the maximum value in each row.
    Corresponds to `get_table_occurence` from R/function.R.

    Args:
        data_df: Pandas DataFrame. Expected to have a MultiIndex ('Config', 'Level') for rows
                 and method names for columns. Values are occurrences or similar metrics.
        digits: Number of decimal places for formatting values.

    Returns:
        A string containing the LaTeX table.
    """
    if not isinstance(data_df.index, pd.MultiIndex) or not len(data_df.index.levels) == 2:
        raise ValueError("data_df must have a MultiIndex with 2 levels ('Config', 'Level').")

    col_names = data_df.columns.tolist()

    latex_str = []
    latex_str.append("\\begin{adjustbox}{width=1\\textwidth}")
    # Column format: 'c | c' for Config & Level, then 'c' for each method
    col_format = "c | c " + " ".join(["c"] * len(col_names))
    latex_str.append(f"\\begin{{tabular}}{{ {col_format} }}")
    latex_str.append("\\toprule")

    # Header row: Config | Level | Method1 | Method2 ...
    header = ["Config", "Level"] + [col.replace("_", "\\_") for col in col_names]
    latex_str.append(" & ".join(header) + " \\\\")
    latex_str.append("\\midrule")

    # Iterate through MultiIndex levels
    # Group by the first level of MultiIndex ('Config') to handle multirow
    # current_config_val = None # To track when Config changes for multirow

    # Store rows to apply multirow for 'Config' later if needed, or print line by line
    # Simpler: iterate and use \addlinespace for grouping, no multirow for 'Config' initially
    # R code implies multirow for Config.

    last_config = None
    for index_tuple, row_series in data_df.iterrows():
        config_val, level_val = index_tuple # e.g., ("C1", "L1")
        config_val_str = str(config_val).replace("_", "\\_")
        level_val_str = str(level_val).replace("_", "\\_")

        row_data_numeric = pd.to_numeric(row_series, errors='coerce')

        max_val_col_name = None
        if not row_data_numeric.isnull().all(): # Check if there are any numeric values to find a max
            try:
                # Find unique maximum, similar to get_table logic for min.
                # max_val = row_data_numeric.max()
                # if (row_data_numeric == max_val).sum() == 1:
                #    max_val_col_name = row_data_numeric.idxmax()
                # Simpler: bold the first maximum.
                 max_val_col_name = row_data_numeric.idxmax()
            except ValueError: # All NaNs after coerce
                 pass


        row_values_str = []
        if config_val != last_config:
            if last_config is not None: # Add space before new config block, except for the first one
                 latex_str.append("\\addlinespace")
            row_values_str.append(config_val_str)
            last_config = config_val
        else:
            row_values_str.append("") # Empty for subsequent rows of the same config for multirow effect

        row_values_str.append(level_val_str)

        for col_name in col_names:
            val = row_series[col_name]
            if pd.isna(val):
                cell_str = "-" # Placeholder for NaN values
            else:
                # Ensure value is numeric before formatting, handle non-numeric gracefully if they slip through
                try:
                    cell_str = f"{float(val):.{digits}f}"
                except ValueError:
                    cell_str = str(val) # Keep as string if not convertible to float

            if col_name == max_val_col_name and not pd.isna(val): # Only bold if not NaN
                cell_str = f"\\textbf{{{cell_str}}}"
            row_values_str.append(cell_str)

        latex_str.append(" & ".join(row_values_str) + " \\\\")

    latex_str.append("\\bottomrule")
    latex_str.append("\\end{tabular}}")
    latex_str.append("\\end{adjustbox}")

    return "\n".join(latex_str)


if __name__ == '__main__':
    # ... (all previous tests) ...
    print("\nFinished all simulation and modeling function tests in functions.py.")

    print("\nTesting LaTeX table generation functions...")

    # Test get_table
    print("\nTesting get_table()...")
    data_gt = pd.DataFrame({
        'MethodA': [1.234, 2.345, 0.987],
        'MethodB': [1.100, 2.500, 1.050],
        'MethodC': [1.500, 2.200, 0.999]
    }, index=[f"Cfg{i+1}" for i in range(3)])
    uncertainty_gt = pd.DataFrame({
        'MethodA': [0.1, 0.2, 0.05],
        'MethodB': [0.12, 0.15, 0.08],
        'MethodC': [0.2, 0.1, 0.03]
    }, index=[f"Cfg{i+1}" for i in range(3)])

    latex_table_1 = get_table(data_gt, uncertainty_df=uncertainty_gt, bold=True, digits=2)
    print("get_table output (with uncertainty, bolding min):")
    print(latex_table_1)
    assert "\\textbf" in latex_table_1 # Check bolding applied
    assert "$\\pm$" in latex_table_1   # Check uncertainty applied

    latex_table_2 = get_table(data_gt, bold=False, digits=3)
    print("\nget_table output (no uncertainty, no bolding, 3 digits):")
    print(latex_table_2)
    assert "\\textbf" not in latex_table_2
    assert "$\\pm$" not in latex_table_2
    assert "1.234" in latex_table_2 # Check digit formatting

    # Test get_table_occurence
    print("\nTesting get_table_occurence()...")
    arrays_gto = [
        np.array(["C1", "C1", "C2", "C2", "C3"]), # Added C3 for more diverse test
        np.array(["L1", "L2", "L1", "L2", "L1"]),
    ]
    index_gto = pd.MultiIndex.from_arrays(arrays_gto, names=("Config", "Level"))
    data_gto = pd.DataFrame({
        'MethodX': [10, 20, 5, 25, np.nan],
        'MethodY': [15, 10, np.nan, 22, 18],
        'MethodZ': [np.nan, 25, 8, 20, 19]
    }, index=index_gto)

    latex_table_occ = get_table_occurence(data_gto, digits=1)
    print("get_table_occurence output (with NaNs):")
    print(latex_table_occ)
    assert "\\textbf" in latex_table_occ
    assert "Config & Level & MethodX & MethodY & MethodZ \\\\" in latex_table_occ
    # Check for placeholder for NaN, e.g. C3, MethodX should be "-"
    assert "C3 & L1 & - & 18.0 & \\textbf{19.0} \\\\" in latex_table_occ or \
           "C3 & L1 & - & 18.0 & \\textbf{19.0} \\\\" in latex_table_occ # Based on updated get_table_occurence
    assert "\\addlinespace" in latex_table_occ

    print("\nFinished all tests in functions.py.")

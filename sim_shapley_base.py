"""
sim_shapley_base.py — Consolidated SIM-Shapley infrastructure for FUSHAP.
"""

import warnings
import numpy as np
from scipy.linalg import cho_factor, cho_solve


# =====================================================================
# 1. Utility Functions (from utils.py)
# =====================================================================

def get_shapley_kernel_weights(feature_num):
    """
    Compute the Shapley kernel distribution p(|z|) over coalition sizes.
    """
    sizes = np.arange(1, feature_num)
    weights = 1.0 / (sizes * (feature_num - sizes))
    weights = weights / np.sum(weights)
    return weights


def sample_coalitions(sample_num, feature_num, weights, seed):
    """
    Sample coalitions z from the Shapley kernel distribution.
    """
    rng = np.random.default_rng(seed=seed)
    num_included = rng.choice(feature_num - 1, size=sample_num, p=weights) + 1
    Z = np.zeros((sample_num, feature_num), dtype=bool)
    for row, num in zip(Z, num_included):
        inds = rng.choice(feature_num, size=num, replace=False)
        row[inds] = True
    return Z


def sample_data_batch(batch_size, X, Y, seed):
    """
    Sample a batch of (X, Y) with replacement.
    """
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(X), size=batch_size, replace=True)
    return X[indices], Y[indices]

def mse_loss(y_true, y_pred, reduction='mean'):
    """
    Mean squared error loss.
    """
    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_true.ndim == 1:
        y_true = y_true[:, np.newaxis]
    per_obs = np.sum((y_pred - y_true) ** 2, axis=1)
    if reduction == 'mean':
        return np.mean(per_obs)
    return per_obs


def cross_entropy_loss(y_true, y_pred, reduction='mean'):
    """
    Cross-entropy loss for classification.
    """
    y_true = np.atleast_1d(y_true)
    y_pred = np.atleast_1d(y_pred)

    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
        y_pred = np.concatenate((1 - y_pred, y_pred), axis=1)

    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)

    if y_pred.shape == y_true.shape:
        # Soft labels
        per_obs = -np.sum(np.log(y_pred) * y_true, axis=1)
    else:
        # Hard labels
        per_obs = -np.log(y_pred[np.arange(len(y_pred)), y_true.astype(int)])

    if reduction == 'mean':
        return np.mean(per_obs)
    return per_obs


# =====================================================================
# 2. MarginalImputer (from imputers.py, unchanged)
# =====================================================================

class MarginalImputer:
    """
    Marginalizes absent features using their empirical distribution.
    """

    def __init__(self, model, background_data, sample_num=10):
        self.model = model
        self.data = background_data
        self.data_repeat = background_data.copy()
        self.sample_num = sample_num
        self.feature_num = background_data.shape[1]

        if len(background_data) > 1024:
            warnings.warn(
                f"Using {len(background_data)} background samples may be slow. "
                "Consider using <= 1024.",
                RuntimeWarning,
            )

    def __call__(self, x, S):
        """
        Compute marginal expectation E[f(x_S, X_{S^c})].
        """
        n_samples = len(x)

        if S.ndim == 1:
            S = np.tile(S.reshape(1, -1), (n_samples, 1))

        assert S.shape == (n_samples, self.feature_num)

        # Repeat each sample for marginal averaging
        x_repeat = np.repeat(x, self.sample_num, axis=0)
        S_repeat = np.repeat(S, self.sample_num, axis=0)
        n = len(x_repeat)

        # Prepare background samples
        if len(self.data_repeat) != n:
            repeat_time = n // len(self.data) + 1
            self.data_repeat = np.tile(self.data, (repeat_time, 1))[:n]

        # Replace masked (absent) features with background values
        x_repeat[~S_repeat] = self.data_repeat[~S_repeat]

        # Model predictions
        pred = self.model(x_repeat)
        pred = pred.reshape(n_samples, self.sample_num, *pred.shape[1:])

        # Average over background samples
        return np.mean(pred, axis=1)


# =====================================================================
# 3. WLS Shapley Solver (extracted from estimator_fast.py)
# =====================================================================

def compute_shapley_kernel_matrix(feature_num):
    """
    Compute the population Shapley kernel second-moment matrix E_z[zz'].
    """
    # Monte Carlo with large sample for accuracy
    n_mc = max(100000, feature_num * 1000)
    weights = get_shapley_kernel_weights(feature_num)
    Z = sample_coalitions(n_mc, feature_num, weights, seed=42)
    Sigma = (Z.astype(float).T @ Z.astype(float)) / n_mc
    return Sigma


def solve_wls_shapley(A, b_bar, c, l2_penalty=0.0):
    """
    Solve the KernelSHAP WLS system via KKT closed form.
    """
    p = A.shape[0]
    vec_1 = np.ones(p)

    A_bar = A + np.eye(p) * l2_penalty
    cho_c, cho_low = cho_factor(A_bar)

    A_bar_inv_b = cho_solve((cho_c, cho_low), b_bar)
    A_bar_inv_1 = cho_solve((cho_c, cho_low), vec_1)

    # Lagrange multiplier from efficiency constraint
    lam = (c - vec_1 @ A_bar_inv_b) / (vec_1 @ A_bar_inv_1)

    phi = A_bar_inv_b + lam * A_bar_inv_1
    return phi


# =====================================================================
# 4. Value Function Evaluator (modified from estimator_fast.py)
# =====================================================================

def evaluate_value_functions(X, Y, Z, imputer, loss_func,
                             importance_weights=None):
    """
    Evaluate value functions V(z) for sampled coalitions, returning both aggregated quantities (A, b_bar, c) and per-observation eta values.
    """
    n, p = X.shape
    m = len(Z)

    if importance_weights is None:
        importance_weights = np.ones(n)

    # Normalize weights to sum to n (so weighted mean = weighted sum / n)
    w = importance_weights / np.mean(importance_weights)

    # --- Compute v(1) and v(0) ---
    pred_all = imputer(X, np.ones(p, dtype=bool))    # predictions with all features
    pred_none = imputer(X, np.zeros(p, dtype=bool))   # predictions with no features

    per_obs_v1 = -loss_func(Y, pred_all, reduction='none')    # (n,)
    per_obs_v0 = -loss_func(Y, pred_none, reduction='none')   # (n,)

    v_1 = np.mean(w * per_obs_v1)
    v_0 = np.mean(w * per_obs_v0)

    # --- Compute eta(z_j, x_i, y_i) for each coalition and observation ---
    eta_matrix = np.zeros((n, m))

    # Batch computation: for each coalition, evaluate on all observations
    for j in range(m):
        z_j = Z[j]  # (p,) boolean mask
        y_hat_j = imputer(X, z_j)  # (n,) or (n, output_dim) predictions

        # Per-observation loss (negated)
        eta_matrix[:, j] = -loss_func(Y, y_hat_j, reduction='none')  # (n,)

    # --- Compute weighted value functions V(z_j) ---
    v_hat = np.zeros(m)
    for j in range(m):
        v_hat[j] = np.mean(w * eta_matrix[:, j])

    # --- Compute A and b_bar ---
    Z_float = Z.astype(float)
    A = (Z_float.T @ Z_float) / m                        
    b_bar = (Z_float.T @ (v_hat - v_0)) / m              
    c = v_1 - v_0                                       

    return {
        'A': A,
        'b_bar': b_bar,
        'c': c,
        'v_hat': v_hat,
        'v_1': v_1,
        'v_0': v_0,
        'eta_matrix': eta_matrix,
        'per_obs_v1': per_obs_v1,
        'per_obs_v0': per_obs_v0,
        'Z': Z,
    }


def evaluate_value_functions_fast(X, Y, Z, imputer, loss_func,
                                   importance_weights=None):
    """
    Fast batch version of evaluate_value_functions.
    """
    n, p = X.shape
    m = len(Z)

    if importance_weights is None:
        importance_weights = np.ones(n)

    w = importance_weights / np.mean(importance_weights)

    # --- v(1) and v(0) ---
    pred_all = imputer(X, np.ones(p, dtype=bool))
    pred_none = imputer(X, np.zeros(p, dtype=bool))

    per_obs_v1 = -loss_func(Y, pred_all, reduction='none')
    per_obs_v0 = -loss_func(Y, pred_none, reduction='none')

    v_1 = np.mean(w * per_obs_v1)
    v_0 = np.mean(w * per_obs_v0)

    # --- Batch eta computation ---
    X_repeat = np.repeat(X, m, axis=0)
    S_repeat = np.tile(Z, (n, 1))  # (n * m, p)

    y_hat_all = imputer(X_repeat, S_repeat)  # (n * m,) or (n * m, d)

    # Reshape to (n, m, ...) and compute per-observation losses
    if y_hat_all.ndim == 1:
        y_hat_all = y_hat_all.reshape(n, m)
    else:
        y_hat_all = y_hat_all.reshape(n, m, -1)

    eta_matrix = np.zeros((n, m))
    for j in range(m):
        if y_hat_all.ndim == 2:
            preds_j = y_hat_all[:, j]
        else:
            preds_j = y_hat_all[:, j, :]
        eta_matrix[:, j] = -loss_func(Y, preds_j, reduction='none')

    # --- Weighted value functions ---
    v_hat = np.array([np.mean(w * eta_matrix[:, j]) for j in range(m)])

    # --- A, b_bar, c ---
    Z_float = Z.astype(float)
    A = (Z_float.T @ Z_float) / m
    b_bar = (Z_float.T @ (v_hat - v_0)) / m
    c = v_1 - v_0

    return {
        'A': A,
        'b_bar': b_bar,
        'c': c,
        'v_hat': v_hat,
        'v_1': v_1,
        'v_0': v_0,
        'eta_matrix': eta_matrix,
        'per_obs_v1': per_obs_v1,
        'per_obs_v0': per_obs_v0,
        'Z': Z,
    }


# =====================================================================
# 5. Convenience: Full LC-only Shapley Estimation
# =====================================================================

def estimate_shapley_lc(X_lc, Y_lc, imputer, loss_func,
                        n_coalitions=1000, l2_penalty=0.1,
                        importance_weights=None,
                        paired_sampling=True,
                        seed=42, fast=True):
    """
    Full LC-only Shapley estimation pipeline (FUSHAP Step 1).
    """
    p = X_lc.shape[1]
    weights = get_shapley_kernel_weights(p)

    # Sample coalitions
    Z = sample_coalitions(n_coalitions, p, weights, seed)

    if paired_sampling:
        Z = np.concatenate([Z, ~Z], axis=0)

    # Evaluate value functions
    eval_func = evaluate_value_functions_fast if fast else evaluate_value_functions
    vf_result = eval_func(
        X_lc, Y_lc, Z, imputer, loss_func,
        importance_weights=importance_weights
    )

    # Solve WLS
    phi = solve_wls_shapley(
        vf_result['A'], vf_result['b_bar'], vf_result['c'],
        l2_penalty=l2_penalty
    )

    return {
        'phi': phi,
        **vf_result,
        'importance_weights': importance_weights if importance_weights is not None else np.ones(len(X_lc)),
    }
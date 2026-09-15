"""
fushap.py — Variance-Reduced Shapley Attribution across Heterogeneous Data Sources.

Implements the FUSHAP pipeline:
  Step 1: LC-only preliminary Shapley estimate (via sim_shapley_base)
  Step 2: Variance reduction using LM sites (influence function → control
          variates → calibrated augmented estimator)
  Step 3: Screening for misaligned sites

Dependencies: sim_shapley_base.py (in same directory)
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import cho_factor, cho_solve

from sim_shapley_base import (
    MarginalImputer,
    get_shapley_kernel_weights,
    sample_coalitions,
    compute_shapley_kernel_matrix,
    solve_wls_shapley,
    estimate_shapley_lc,
)


# =====================================================================
# Step 2a: Influence Function
# =====================================================================

def compute_influence_function(eta_matrix, v_hat, Z, Sigma_inv, c):
    """
    Compute the influence function ψ(x_i, y_i) for each LC observation.

    Returns:
        psi: (n, p) influence function values.
    """
    n, m = eta_matrix.shape
    p = Z.shape[1]
    Z_float = Z.astype(float)
    vec_1 = np.ones(p)

    Sigma_inv_1 = Sigma_inv @ vec_1        
    one_Sigma_inv_one = vec_1 @ Sigma_inv_1  

    surprise = eta_matrix - v_hat[np.newaxis, :]

    g = (surprise @ Z_float) / m

    g_Sigma_inv_1 = g @ Sigma_inv_1 
    lam = -g_Sigma_inv_1 / one_Sigma_inv_one

    g_plus_lam = g + lam[:, np.newaxis] * vec_1[np.newaxis, :]  
    psi = g_plus_lam @ Sigma_inv.T

    return psi


def compute_influence_function_full(eta_matrix, v_hat, Z, Sigma_inv, c,
                                     per_obs_v1, per_obs_v0, importance_weights=None):
    """
    Full version of the influence function that uses per-observation
    eta (1, x_i, y_i) and eta (0, x_i, y_i) for the exact constraint correction.

    Returns:
        psi: (n, p) influence function values.
    """
    n, m = eta_matrix.shape
    p = Z.shape[1]
    Z_float = Z.astype(float)
    vec_1 = np.ones(p)

    Sigma_inv_1 = Sigma_inv @ vec_1
    one_Sigma_inv_one = vec_1 @ Sigma_inv_1

    if importance_weights is not None:
        w = importance_weights / np.mean(importance_weights)
        surprise = w[:, np.newaxis] * eta_matrix - v_hat[np.newaxis, :]
        per_obs_c = w * (per_obs_v1 - per_obs_v0)
    else:
        surprise = eta_matrix - v_hat[np.newaxis, :]
        per_obs_c = per_obs_v1 - per_obs_v0

    g = (surprise @ Z_float) / m

    g_Sigma_inv_1 = g @ Sigma_inv_1
    lam = (per_obs_c - c - g_Sigma_inv_1) / one_Sigma_inv_one

    g_plus_lam = g + lam[:, np.newaxis] * vec_1[np.newaxis, :]
    psi = g_plus_lam @ Sigma_inv.T

    return psi


# =====================================================================
# Step 2b: Control Variates via Ridge Regression
# =====================================================================

def build_control_variate(psi_lc, X_lc, Y_lc, Gamma_r, n_splits=5,
                           alphas=None, random_state=42, poly_degree=2,
                           sample_weights=None):
    """
    Build the control variate for one LM site by ridge regression with cross-fitting.

    Returns:
        result: dict with keys:
            'g_hat_lc': (n, p) cross-fitted predictions on LC.
            'models': list of fitted RidgeCV models.
            'poly': fitted PolynomialFeatures transformer (for reuse).
            'Gamma_r': the feature indices used.
    """
    n, p = psi_lc.shape

    if alphas is None:
        alphas = np.logspace(-3, 3, 20)

    # Build regression inputs with polynomial features
    X_raw = np.column_stack([X_lc[:, Gamma_r], Y_lc])
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_reg = poly.fit_transform(X_raw)

    # Cross-fitted predictions
    g_hat_lc = np.zeros((n, p))
    models = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, test_idx in kf.split(X_reg):
        model = RidgeCV(alphas=alphas)
        # model.fit(X_reg[train_idx], psi_lc[train_idx])

        if sample_weights is not None:
            model.fit(X_reg[train_idx], psi_lc[train_idx],
                      sample_weight=sample_weights[train_idx])
        else:
            model.fit(X_reg[train_idx], psi_lc[train_idx])

        g_hat_lc[test_idx] = model.predict(X_reg[test_idx])
        models.append(model)

    return {
        'g_hat_lc': g_hat_lc,
        'models': models,
        'poly': poly,
        'Gamma_r': Gamma_r,
    }


def evaluate_control_variate(X_lm, Y_lm, Gamma_r, models_from_cv,
                              X_lc_for_refit=None, Y_lc_for_refit=None,
                              psi_lc_for_refit=None, alphas=None):
    """
    Evaluate the control variate ĝ_r on LM_r data.

    For evaluation on LM_r, we refit a single model on all LC data
    (no cross-fitting needed since LM_r observations are out-of-sample
    by definition).

    Returns:
        g_hat_lm: (n_r, p) predictions on LM_r.
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 20)

    # Build regression input for LM_r
    X_reg_lm = np.column_stack([X_lm[:, Gamma_r], Y_lm])

    if X_lc_for_refit is not None:
        # Refit on all LC data
        X_reg_lc = np.column_stack([X_lc_for_refit[:, Gamma_r], Y_lc_for_refit])
        model = RidgeCV(alphas=alphas)
        model.fit(X_reg_lc, psi_lc_for_refit)
        g_hat_lm = model.predict(X_reg_lm)
    else:
        # Average predictions from CV models as fallback
        preds = [m.predict(X_reg_lm) for m in models_from_cv]
        g_hat_lm = np.mean(preds, axis=0)

    return g_hat_lm


# =====================================================================
# Step 2c: Calibration via Quadratic Program
# =====================================================================

def calibrate_per_site(phi_preliminary, psi_lc,
                        g_hat_lc_list, g_hat_lm_list,
                        g_hat_lc_insample_list=None,
                        importance_weights_lc=None,
                        importance_weights_lm_list=None,
                        per_feature=False):
    """
    Calibration: find optimal delta_r for each site by minimizing the empirical variance of the augmented estimator.

    Returns:
        result: dict with keys:
            'delta': (R,) per-site calibration weights.
            'delta_per_feature': (R, p) per-feature weights.
            'corrections': (R, p) correction vectors.
            'phi_augmented': (p,) calibrated augmented estimate.
    """
    R = len(g_hat_lc_list)
    n = len(g_hat_lc_list[0])
    p = g_hat_lc_list[0].shape[1]

    # Use in-sample predictions for covariance if available
    if g_hat_lc_insample_list is None:
        g_hat_lc_insample_list = g_hat_lc_list

    if importance_weights_lc is None:
        importance_weights_lc = np.ones(n)
    w_lc = importance_weights_lc / np.mean(importance_weights_lc)

    # Compute correction vectors using CROSS-FITTED predictions
    corrections = np.zeros((R, p))
    n_r_list = []
    for r in range(R):
        n_r = len(g_hat_lm_list[r])
        n_r_list.append(n_r)

        if importance_weights_lm_list is not None and importance_weights_lm_list[r] is not None:
            w_lm = importance_weights_lm_list[r]
            w_lm = w_lm / np.mean(w_lm)
        else:
            w_lm = np.ones(n_r)

        lm_mean = np.average(g_hat_lm_list[r], weights=w_lm, axis=0)
        lc_mean = np.average(g_hat_lc_list[r], weights=w_lc, axis=0)
        corrections[r] = lm_mean - lc_mean

    # Compute calibration weights using IN-SAMPLE predictions 
    psi_centered = psi_lc - np.mean(psi_lc, axis=0)
    g_centered_insample = [g - np.mean(g, axis=0) for g in g_hat_lc_insample_list]
    g_centered_lm = [g - np.mean(g, axis=0) for g in g_hat_lm_list]

    # Per-feature calibration (joint R-dimensional QP per feature)
    delta_per_feature = np.zeros((R, p))
    A_total = np.zeros((R, R))
    b_total = np.zeros(R)

    reg_strength = 0.1 

    for j in range(p):
        A_j = np.zeros((R, R))
        b_j = np.zeros(R)

        for r in range(R):
            # Covariance between ψ and ĝ_r (in-sample)
            b_j[r] = np.mean(psi_centered[:, j] * g_centered_insample[r][:, j])

            for s in range(R):
                A_j[r, s] = np.mean(g_centered_insample[r][:, j] *
                                     g_centered_insample[s][:, j])

            # LM variance regularization
            var_lm_j = np.mean(g_centered_lm[r][:, j] ** 2)
            A_j[r, r] += (n / n_r_list[r]) * var_lm_j

        avg_diag = np.mean(np.diag(A_j))
        A_j += np.eye(R) * (reg_strength * avg_diag + 1e-10)

        try:
            delta_j = np.linalg.solve(A_j, b_j)
        except np.linalg.LinAlgError:
            delta_j = np.zeros(R)

        delta_per_feature[:, j] = delta_j

        A_total += A_j
        b_total += b_j

    try:
        delta_per_site = np.linalg.solve(A_total, b_total)
    except np.linalg.LinAlgError:
        delta_per_site = np.zeros(R)

    # Compute augmented estimate 
    if per_feature:
        phi_augmented = phi_preliminary.copy()
        for r in range(R):
            phi_augmented += delta_per_feature[r, :] * corrections[r]
    else:
        phi_augmented = phi_preliminary.copy()
        for r in range(R):
            phi_augmented += delta_per_site[r] * corrections[r]

    return {
        'delta': delta_per_site,
        'delta_per_feature': delta_per_feature,
        'corrections': corrections,
        'phi_augmented': phi_augmented,
    }


# =====================================================================
# Step 3: Screening for Misaligned Sites
# =====================================================================

def screen_sites(g_hat_lc_list, g_hat_lm_list,
                 importance_weights_lc=None,
                 importance_weights_lm_list=None,
                 alpha=0.05, n_perm=1000, seed=42):
    """
    Screen LM sites for misalignment using a permutation test.

    Returns:
        result: dict with keys:
            'aligned': list of booleans, True if site passes screening.
            'p_values': (R,) p-value for each site.
            'test_stats': (R,) observed test statistic for each site.
            'aligned_indices': list of indices of aligned sites.
    """
    R = len(g_hat_lc_list)
    n = len(g_hat_lc_list[0])
    rng = np.random.default_rng(seed)

    if importance_weights_lc is None:
        importance_weights_lc = np.ones(n)
    w_lc = importance_weights_lc / np.mean(importance_weights_lc)

    p_values = np.zeros(R)
    test_stats = np.zeros(R)
    aligned = []

    for r in range(R):
        n_r = len(g_hat_lm_list[r])

        if importance_weights_lm_list is not None and importance_weights_lm_list[r] is not None:
            w_lm = importance_weights_lm_list[r]
            w_lm = w_lm / np.mean(w_lm)
        else:
            w_lm = np.ones(n_r)

        # Observed weighted means
        lm_mean = np.average(g_hat_lm_list[r], weights=w_lm, axis=0)
        lc_mean = np.average(g_hat_lc_list[r], weights=w_lc, axis=0)

        # Observed test statistic: sum of squared differences
        obs_stat = np.sum((lm_mean - lc_mean) ** 2)

        # Permutation test: pool LC and LM observations,
        # shuffle labels, recompute test statistic
        tau_all = np.vstack([g_hat_lc_list[r], g_hat_lm_list[r]])
        w_all = np.concatenate([w_lc, w_lm])
        n_total = n + n_r

        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(n_total)
            perm_lc = tau_all[perm[:n]]
            perm_lm = tau_all[perm[n:]]
            w_perm_lc = w_all[perm[:n]]
            w_perm_lm = w_all[perm[n:]]

            perm_lc_mean = np.average(perm_lc, weights=w_perm_lc, axis=0)
            perm_lm_mean = np.average(perm_lm, weights=w_perm_lm, axis=0)
            perm_stat = np.sum((perm_lm_mean - perm_lc_mean) ** 2)

            if perm_stat >= obs_stat:
                count += 1

        p_val = (count + 1) / (n_perm + 1)

        p_values[r] = p_val
        test_stats[r] = obs_stat
        aligned.append(p_val > alpha)

    aligned_indices = [r for r in range(R) if aligned[r]]

    return {
        'aligned': aligned,
        'p_values': p_values,
        'test_stats': test_stats,
        'aligned_indices': aligned_indices,
    }

# =====================================================================
# Full FUSHAP Pipeline
# =====================================================================

def fushap(X_lc, Y_lc, X_uc, lm_sites, model, loss_func,
           n_coalitions=500, l2_penalty=0.1, bg_sample_size=100,
           imputer_sample_num=10, paired_sampling=True,
           importance_weights_lc=None,
           screening_alpha=0.05, do_screening=True,
           n_cv_splits=5, seed=42, verbose=False):
    """
    Full FUSHAP pipeline: Steps 1 → 2 → 3.

    Args:
        X_lc: (n, p) LC feature matrix (all features observed).
        Y_lc: (n,) LC labels.
        X_uc: (N, p) UC feature matrix (all features, no labels).
              Used as background data for the imputer.
        lm_sites: list of dicts, one per LM site. Each dict has:
            'X': (n_r, p) feature matrix (unobserved features can be
                 any value; they are identified by Gamma_r).
            'Y': (n_r,) labels.
            'Gamma_r': list of observed feature indices at this site.
            'importance_weights': (n_r,) optional importance weights.
        model: callable, model(X) -> predictions.
        loss_func: callable with signature loss_func(y_true, y_pred, reduction).
                   Must support reduction='none'.
        n_coalitions: number of coalitions to sample.
        l2_penalty: WLS regularization.
        bg_sample_size: number of UC samples for the background set.
        imputer_sample_num: number of background samples per imputer call.
        paired_sampling: if True, include complement coalitions.
        importance_weights_lc: (n,) density ratios for LC→UC.
        screening_alpha: significance level for screening test.
        do_screening: whether to run the screening step.
        n_cv_splits: number of cross-fitting folds.
        seed: random seed.
        verbose: print progress.

    Returns:
        result: dict with keys:
            'phi_preliminary': (p,) Step 1 estimate.
            'phi_augmented': (p,) Step 2 estimate (after calibration).
            'phi_final': (p,) final estimate (after screening + recalibration).
            'delta': (R,) calibration weights.
            'screening': screening result dict (if do_screening=True).
            'step1': full Step 1 result dict.
            'psi_lc': (n, p) influence function values on LC.
    """
    p = X_lc.shape[1]
    R = len(lm_sites)

    if verbose:
        print(f"FUSHAP: p={p} features, n_LC={len(X_lc)}, "
              f"R={R} LM sites, N_UC={len(X_uc)}")

    rng = np.random.default_rng(seed=seed)
    bg_indices = rng.choice(len(X_uc), size=min(bg_sample_size, len(X_uc)),
                            replace=False)
    background_data = X_uc[bg_indices]

    imputer = MarginalImputer(model, background_data, sample_num=imputer_sample_num)

    # ================================================================
    # Step 1: LC-only preliminary estimate
    # ================================================================
    if verbose:
        print("Step 1: Computing LC-only preliminary estimate...")

    step1 = estimate_shapley_lc(
        X_lc, Y_lc, imputer, loss_func,
        n_coalitions=n_coalitions,
        l2_penalty=l2_penalty,
        importance_weights=importance_weights_lc,
        paired_sampling=paired_sampling,
        seed=seed,
    )
    phi_preliminary = step1['phi']

    if verbose:
        print(f"  φ̃ = {phi_preliminary}")
        print(f"  V(1)={step1['v_1']:.4f}, V(0)={step1['v_0']:.4f}, "
              f"c={step1['c']:.4f}")

    # ================================================================
    # Step 2: Variance reduction using LM sites
    # ================================================================
    if verbose:
        print("Step 2: Computing influence function and control variates...")

    # --- 2a: Influence function ---
    Sigma = compute_shapley_kernel_matrix(p)
    Sigma_inv = np.linalg.inv(Sigma)

    psi_lc = compute_influence_function_full(
        eta_matrix=step1['eta_matrix'],
        v_hat=step1['v_hat'],
        Z=step1['Z'],
        Sigma_inv=Sigma_inv,
        c=step1['c'],
        per_obs_v1=step1['per_obs_v1'],
        per_obs_v0=step1['per_obs_v0'],
        importance_weights=importance_weights_lc,
    )

    if verbose:
        print(f"  ψ̃ mean: {np.mean(psi_lc, axis=0)}")
        print(f"  ψ̃ std:  {np.std(psi_lc, axis=0)}")

    # --- 2b: Control variates for each LM site ---
    g_hat_lc_list = []      
    g_hat_lc_insample = []  
    g_hat_lm_list = []

    for r, site in enumerate(lm_sites):
        Gamma_r = site['Gamma_r']

        if verbose:
            print(f"  Site {r}: Γ_r={Gamma_r}, n_r={len(site['Y'])}")

        w_lc_normalized = None
        if importance_weights_lc is not None:
            w_lc_normalized = importance_weights_lc / np.mean(importance_weights_lc)
 
        cv_result = build_control_variate(
            psi_lc, X_lc, Y_lc, Gamma_r,
            n_splits=n_cv_splits,
            random_state=seed + r,
            sample_weights=w_lc_normalized,
        )
        g_hat_lc_list.append(cv_result['g_hat_lc'])
        poly = cv_result['poly']  # reuse same polynomial transformer

        # In-sample predictions: refit on ALL LC, predict on ALL LC
        X_raw_lc = np.column_stack([X_lc[:, Gamma_r], Y_lc])
        X_reg_lc = poly.transform(X_raw_lc)
        model_full = RidgeCV(alphas=np.logspace(-3, 3, 20))
        if w_lc_normalized is not None:
            model_full.fit(X_reg_lc, psi_lc, sample_weight=w_lc_normalized)
        else:
            model_full.fit(X_reg_lc, psi_lc)
        g_hat_lc_insample.append(model_full.predict(X_reg_lc))

        # Evaluate on LM_r (using same polynomial transformer)
        X_raw_lm = np.column_stack([site['X'][:, Gamma_r], site['Y']])
        X_reg_lm = poly.transform(X_raw_lm)
        g_hat_lm = model_full.predict(X_reg_lm)
        g_hat_lm_list.append(g_hat_lm)

    # 2c: Screening (Step 3)
    screening_result = None
    aligned_indices = list(range(R))  # default: all aligned

    if do_screening and R > 0:
        if verbose:
            print("Step 3: Screening for misaligned sites...")

        iw_lm_list = [site.get('importance_weights', None)
                      for site in lm_sites]

        screening_result = screen_sites(
            g_hat_lc_list, g_hat_lm_list,
            importance_weights_lc=importance_weights_lc,
            importance_weights_lm_list=iw_lm_list,
            alpha=screening_alpha,
        )
        aligned_indices = screening_result['aligned_indices']

        if verbose:
            for r in range(R):
                status = "PASS" if screening_result['aligned'][r] else "FAIL"
                print(f"  Site {r}: {status} "
                      f"(p={screening_result['p_values'][r]:.4f})")

    # 2d: Calibration on aligned sites only
    if len(aligned_indices) > 0:
        aligned_lc = [g_hat_lc_list[r] for r in aligned_indices]
        aligned_lc_insample = [g_hat_lc_insample[r] for r in aligned_indices]
        aligned_lm = [g_hat_lm_list[r] for r in aligned_indices]
        aligned_iw = None
        if importance_weights_lc is not None:
            aligned_iw_lm = [
                lm_sites[r].get('importance_weights', None)
                for r in aligned_indices
            ]
        else:
            aligned_iw_lm = None

        calib_result = calibrate_per_site(
            phi_preliminary, psi_lc, aligned_lc, aligned_lm,
            g_hat_lc_insample_list=aligned_lc_insample,
            importance_weights_lc=importance_weights_lc,
            importance_weights_lm_list=aligned_iw_lm,
        )
        phi_augmented = calib_result['phi_augmented']
        delta = calib_result['delta']
    else:
        phi_augmented = phi_preliminary.copy()
        delta = np.array([])

    if verbose:
        print(f"  δ = {delta}")
        print(f"  φ̂_augmented = {phi_augmented}")

    return {
        'phi_preliminary': phi_preliminary,
        'phi_augmented': phi_augmented,
        'phi_final': phi_augmented,
        'delta': delta,
        'screening': screening_result,
        'aligned_indices': aligned_indices,
        'step1': step1,
        'psi_lc': psi_lc,
        'g_hat_lc_list': g_hat_lc_list,
        'g_hat_lc_insample': g_hat_lc_insample,
        'g_hat_lm_list': g_hat_lm_list,
    }
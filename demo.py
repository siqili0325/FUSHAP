"""
demo.py — Quick demonstration of FUSHAP on simulated data.

Usage:
    python demo.py --n_reps 100 --n_jobs 4
"""

import os
import gc
import numpy as np
import argparse
from joblib import Parallel, delayed
from sim_shapley_base import (
    MarginalImputer,
    estimate_shapley_lc,
    mse_loss,
)
from fushap import fushap


P = 6
BETA = np.array([2.0, 1.5, 1.0, 0.5, 0.3, 0.2])
TRUTH_PATH = 'demo_phi_true.npy'
UC_PATH = 'demo_X_uc.npy'
METHODS = ['(F) Oracle', 'FUSHAP', '(A) LC-only', '(B) LC+IPW', '(C) Impute-mean']


def model(X):
    return X @ BETA


def compute_and_save_ground_truth():
    """Compute high-precision ground truth and save to disk.

    Processes one coalition at a time to avoid memory explosion.
    """
    from scipy.special import comb

    rng = np.random.default_rng(9999)
    n_eval = 50000
    K = 500

    print(f"  Generating {n_eval} evaluation samples...")
    X_eval = rng.normal(0, 1, size=(n_eval, P))
    Y_eval = model(X_eval) + rng.normal(0, 0.5,
                                         size=n_eval)
    bg = rng.normal(0, 1, size=(K, P))
    imputer = MarginalImputer(model, bg, sample_num=K)

    n_coalitions = 2**P
    print(f"  Enumerating {n_coalitions} coalitions "
          f"(one at a time)...")
    V = {}
    for idx in range(n_coalitions):
        z = np.array([(idx >> j) & 1 for j in range(P)],
                      dtype=bool)
        S = tuple(np.where(z)[0])
        preds = imputer(X_eval, z)
        V[S] = -np.mean((Y_eval - preds)**2)
        if (idx + 1) % 128 == 0:
            print(f"    {idx+1}/{n_coalitions} done")

    print("  Computing Shapley values from V(S)...")
    phi_true = np.zeros(P)
    for i in range(P):
        for idx in range(n_coalitions):
            z = np.array([(idx >> j) & 1
                          for j in range(P)], dtype=bool)
            if z[i]:
                continue
            S = tuple(np.where(z)[0])
            S_with_i = tuple(sorted(list(S) + [i]))
            s = len(S)
            weight = 1.0 / (P * comb(P - 1, s, exact=True))
            phi_true[i] += weight * (V[S_with_i] - V[S])

    np.save(TRUTH_PATH, phi_true)
    print(f"  Saved to {TRUTH_PATH}")

    # Free memory
    del X_eval, Y_eval, bg, imputer, V
    gc.collect()


def generate_and_save_uc():
    """Generate UC data and save to disk."""
    rng = np.random.default_rng(0)
    X_uc = rng.normal(0, 1, size=(5000, P))
    np.save(UC_PATH, X_uc)
    print(f"  Saved {X_uc.shape[0]} obs to {UC_PATH}")
    del X_uc
    gc.collect()


def generate_data(seed):
    """Generate LC and LM data for one replication."""
    rng = np.random.default_rng(seed)

    n_lc, n_lm = 300, 2000

    shifts = {
        'LC':  (0.1, 1.05),
        'LM1': (-0.15, 0.9),
        'LM2': (0.2, 1.1),
        'LM3': (-0.1, 0.95),
    }

    def draw(n, key):
        delta, sigma = shifts[key]
        X = rng.normal(delta, sigma, size=(n, P))
        Y = model(X) + rng.normal(0, 0.5, size=n)
        return X, Y

    X_lc, Y_lc = draw(n_lc, 'LC')

    missing_blocks = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    lm_sites = []
    for r in range(3):
        X_lm, Y_lm = draw(n_lm, f'LM{r+1}')
        Gamma_r = [j for j in range(P)
                   if j not in missing_blocks[r]]
        lm_sites.append({
            'X': X_lm, 'Y': Y_lm, 'Gamma_r': Gamma_r,
        })

    return X_lc, Y_lc, lm_sites


def estimate_density_ratio(X_source, X_target,
                            clip_range=(0.1, 10.0)):
    from sklearn.ensemble import GradientBoostingClassifier
    n_s, n_t = len(X_source), len(X_target)
    X_all = np.vstack([X_source, X_target])
    labels = np.concatenate([np.zeros(n_s), np.ones(n_t)])
    clf = GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, random_state=42)
    clf.fit(X_all, labels)
    p_t = clf.predict_proba(X_source)[:, 1]
    p_s = 1 - p_t
    w = (p_t / np.maximum(p_s, 1e-8)) * (n_s / n_t)
    return np.clip(w, clip_range[0], clip_range[1])


def impute_mean(X_lc, X_lm, missing_idx):
    X_imp = X_lm.copy()
    for j in missing_idx:
        X_imp[:, j] = np.mean(X_lc[:, j])
    return X_imp


def run_one_rep(rep):
    """Run one replication. Loads phi_true and X_uc
    from disk inside each worker."""
    phi_true = np.load(TRUTH_PATH)
    X_uc = np.load(UC_PATH)

    seed = 42 + rep * 100
    X_lc, Y_lc, lm_sites = generate_data(seed)

    rng = np.random.default_rng(seed)
    bg_idx = rng.choice(len(X_uc), size=300, replace=False)
    bg = X_uc[bg_idx]
    imputer = MarginalImputer(model, bg, sample_num=30)
    common = dict(n_coalitions=300, l2_penalty=0.01,
                  paired_sampling=True, seed=seed)

    w_lc = estimate_density_ratio(X_lc, X_uc)
    for site in lm_sites:
        Gamma_r = site['Gamma_r']
        site['importance_weights'] = estimate_density_ratio(
            site['X'][:, Gamma_r],
            X_uc[:, Gamma_r])

    def compute_mse(phi):
        return np.sum((phi - phi_true)**2)

    results = {}

    # (A) LC-only
    res = estimate_shapley_lc(
        X_lc, Y_lc, imputer, mse_loss, **common)
    results['(A) LC-only'] = compute_mse(res['phi'])

    # (B) LC + IPW
    w_lc = estimate_density_ratio(X_lc, X_uc)
    res = estimate_shapley_lc(
        X_lc, Y_lc, imputer, mse_loss,
        importance_weights=w_lc, **common)
    results['(B) LC+IPW'] = compute_mse(res['phi'])

    # (C) Impute-mean
    X_pool = [X_lc.copy()]
    Y_pool = [Y_lc.copy()]
    for r in range(3):
        missing_idx = [j for j in range(P)
                       if j not in
                       lm_sites[r]['Gamma_r']]
        X_imp = impute_mean(X_lc, lm_sites[r]['X'],
                             missing_idx)
        X_pool.append(X_imp)
        Y_pool.append(lm_sites[r]['Y'])
    res = estimate_shapley_lc(
        np.vstack(X_pool), np.concatenate(Y_pool),
        imputer, mse_loss, **common)
    results['(C) Impute-mean'] = compute_mse(res['phi'])

    # (F) Oracle
    X_oracle = [X_lc.copy()]
    Y_oracle = [Y_lc.copy()]
    for r in range(3):
        X_oracle.append(lm_sites[r]['X'])
        Y_oracle.append(lm_sites[r]['Y'])
    res = estimate_shapley_lc(
        np.vstack(X_oracle), np.concatenate(Y_oracle),
        imputer, mse_loss, **common)
    results['(F) Oracle'] = compute_mse(res['phi'])

    # FUSHAP
    res = fushap(
        X_lc, Y_lc, X_uc, lm_sites,
        model, mse_loss,
        do_screening=False,
        importance_weights_lc=w_lc,
        **common)
    results['FUSHAP'] = compute_mse(res['phi_augmented'])

    return results


def main():
    parser = argparse.ArgumentParser(
        description='FUSHAP demo on simulated data')
    parser.add_argument('--n_reps', type=int, default=20,
                         help='Number of replications')
    parser.add_argument('--n_jobs', type=int, default=4,
                         help='Number of parallel workers')
    args = parser.parse_args()

    n_reps = args.n_reps
    n_jobs = args.n_jobs

    print("=" * 60)
    print("  FUSHAP Demo: Model I (linear), "
          f"{n_reps} replications")
    print("=" * 60)

    # Step 1: Compute ground truth and save to disk
    print("\nStep 1: Computing ground truth...")
    compute_and_save_ground_truth()
    phi_true = np.load(TRUTH_PATH)
    print(f"  phi_true = "
          f"{np.array2string(phi_true, precision=4)}")

    # Step 2: Generate UC and save to disk
    print("\nStep 2: Generating UC population...")
    generate_and_save_uc()

    # Step 3: Run replications in parallel
    # Each worker loads phi_true and X_uc from disk
    print(f"\nStep 3: Running {n_reps} reps "
          f"({n_jobs} workers)...")

    results_list = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(run_one_rep)(rep)
        for rep in range(n_reps)
    )

    # Aggregate
    all_mse = {m: [] for m in METHODS}
    for res in results_list:
        for m in METHODS:
            all_mse[m].append(res[m])

    # Print summary
    print("\n" + "=" * 60)
    print(f"  Average MSE over {n_reps} replications")
    print("=" * 60)
    print(f"\n  {'Method':<25} {'Mean MSE':<12} "
          f"{'Std':<12}")
    print(f"  {'-'*49}")
    for m in METHODS:
        arr = np.array(all_mse[m])
        print(f"  {m:<25} {arr.mean():<12.4f} "
              f"{arr.std():<12.4f}")

    fushap_mse = np.mean(all_mse['FUSHAP'])
    lc_mse = np.mean(all_mse['(A) LC-only'])
    imp_mse = np.mean(all_mse['(C) Impute-mean'])
    print(f"\n  FUSHAP improvement over LC-only: "
          f"{lc_mse/fushap_mse:.1f}x")
    print(f"  FUSHAP improvement over Impute:  "
          f"{imp_mse/fushap_mse:.1f}x")

    # Clean up temp files
    for f in [TRUTH_PATH, UC_PATH]:
        if os.path.exists(f):
            os.remove(f)

    print("\nDone.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# ================================================================
#  PAL-lookahead experiment (40 cities, E10) with on-disk caching
# ================================================================
from __future__ import annotations
import os
import numpy as np

# ---------------------------- cache ------------------------------
CACHE_DIR  = "E10"
CACHE_FILE = os.path.join(CACHE_DIR, "PAL_lookahead_40.npz")
CACHE_KEY  = "log_likelihood_shared"

os.makedirs(CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------
# 1) Load cached results if available
# -----------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"[cache] Found existing results → {CACHE_FILE}")
    log_likelihood_shared = np.load(CACHE_FILE)[CACHE_KEY]

# -----------------------------------------------------------------
# 2) Otherwise run the original heavy simulation (unmodified)
# -----------------------------------------------------------------
else:
    print("[cache] No cache found – running the full simulation …")

    # ----------------------------------------------------------------
    # >>>>>>>>>>>>>>>>>>>> ORIGINAL CODE — DO NOT EDIT <<<<<<<<<<<<<<<
    # ----------------------------------------------------------------
    import numpy as np
    import time
    import random
    import pandas as pd
    import tensorflow as tf
    import tensorflow_probability as tfp
    import os
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    plt.ioff()

    import sys
    sys.path.append('Scripts/')
    from measles_simulator import *
    from measles_PALSMC import *

    if not os.path.exists("E10"):
        os.makedirs("E10")

    os.environ['PYTHONHASHSEED'] = '42'
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    UKbirths_array = np.load("Data/UKbirths_array.npy")
    UKpop_array = np.load("Data/UKpop_array.npy")
    measles_distance_matrix_array = np.load("Data/measles_distance_matrix_array.npy")
    UKmeasles_array = np.load("Data/UKmeasles_array.npy")

    UKbirths = tf.convert_to_tensor(UKbirths_array, dtype=tf.float32)
    UKpop = tf.convert_to_tensor(UKpop_array, dtype=tf.float32)
    measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype=tf.float32)
    UKmeasles = tf.convert_to_tensor(UKmeasles_array, dtype=tf.float32)

    df = pd.read_csv("Data/M6.csv")
    UKmeasles = tf.convert_to_tensor(df.values, dtype=tf.float32)

    term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype=tf.float32)
    school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=tf.float32)

    n_cities = tf.constant(40, dtype=tf.int64)
    initial_pop = UKpop[:, 0]

    T = 416
    print(T)

    intermediate_steps = 4
    h = tf.constant(14 / tf.cast(intermediate_steps, dtype=tf.float32), dtype=tf.float32)
    is_school_term_array, is_start_school_year_array, *_ = school_term_and_school_year(
        T, intermediate_steps, term, school
    )

    is_school_term_array = tf.convert_to_tensor(is_school_term_array, dtype=tf.float32)
    is_start_school_year_array = tf.convert_to_tensor(is_start_school_year_array, dtype=tf.float32)
    p = tf.constant(0.759, dtype=tf.float32)
    delta_year = tf.convert_to_tensor([[1 / 50]], dtype=tf.float32) * tf.ones((n_cities, 4), dtype=tf.float32)

    n_experiments = 20

    best_parameters = np.load("Data/Parameter/final_parameters_lookahead_A.npy").astype(np.float32)

    pi_0_1, pi_0_2, pi_0_3 = 0.02536, 0.0042, 0.000061
    pi_0 = (
        tf.convert_to_tensor(
            [[pi_0_1, pi_0_2, pi_0_3, 1.0 - pi_0_1 - pi_0_2 - pi_0_3]],
            dtype=tf.float32
        )
        * tf.ones((n_cities, 4), dtype=tf.float32)
    )

    beta_bar = tf.convert_to_tensor(6.30 * tf.ones((n_cities, 1)), dtype=tf.float32)
    rho      = tf.convert_to_tensor([0.142], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)
    gamma    = tf.convert_to_tensor([0.0473], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)

    g = tf.convert_to_tensor([[700]], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)

    a      = tf.constant(0.1476, dtype=tf.float32)
    c      = tf.constant(0.219 , dtype=tf.float32)
    xi_var = tf.convert_to_tensor(0.318 , dtype=tf.float32)
    q_var  = tf.convert_to_tensor(0.305, dtype=tf.float32)

    Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
    Q  = tfp.distributions.TruncatedNormal(loc=0.7, scale=q_var, low=0.0, high=1.0)

    n_particles = 5000
    log_likelihood_shared = np.zeros(n_experiments)

    def logmeanexp(x, se=False, ess=False):
        x = np.asarray(x, dtype=float)
        n = x.shape[0]
        max_x = x.max()
        est = max_x + np.log(np.mean(np.exp(x - max_x)))
        if not se and not ess:
            return est
        results = {"est": est}
        if se:
            jk_vals = np.empty(n)
            for k in range(n):
                x_k = np.delete(x, k)
                max_k = x_k.max()
                jk_vals[k] = max_k + np.log(np.mean(np.exp(x_k - max_k)))
            results["se"] = (n - 1) * jk_vals.std(ddof=1) / np.sqrt(n)
        if ess:
            w = np.exp(x - max_x)
            results["ess"] = (w.sum() ** 2) / (w ** 2).sum()
        return results

    start_time = time.perf_counter()

    for i in range(n_experiments):
        seed_i = 123 + i
        random.seed(seed_i)
        np.random.seed(seed_i)
        tf.random.set_seed(seed_i)

        value = (
            PAL_run_likelihood_lookahead(
                T, intermediate_steps, UKmeasles, UKbirths, UKpop, g,
                measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a,
                is_school_term_array, is_start_school_year_array, h,
                rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year
            )
        )[0].numpy()

        log_likelihood_shared[i] = value
        print(value)

    print(f"Comp.time: {time.perf_counter() - start_time:.2f} s")

    res = logmeanexp(log_likelihood_shared, se=True, ess=True)
    print("Est =", res["est"])
    print("SE  =", res["se"])
    print("ESS =", res["ess"])
    print("Variance:", np.var(log_likelihood_shared, ddof=1))

    np.savetxt(os.path.join("E10", "PAL_lookahead_40.csv"),
               log_likelihood_shared, delimiter=",")

    # ----------------------------------------------------------------
    # >>>>>>>>>>>>>>>>>>>>> ORIGINAL CODE END <<<<<<<<<<<<<<<<<<<<<<<<
    # ----------------------------------------------------------------

    # -------------- cache the results for future runs ----------------
    np.savez(CACHE_FILE, **{CACHE_KEY: log_likelihood_shared})
    print(f"[cache] Results cached → {CACHE_FILE}")

# -------------------------------------------------------------------
# 3) Unified summary – recomputed every run
# -------------------------------------------------------------------
def logmeanexp_and_se(x: np.ndarray) -> tuple[float, float]:
    n = x.size
    max_x = x.max()
    lme   = max_x + np.log(np.mean(np.exp(x - max_x)))
    jk = np.array([
        (np.delete(x, k).max() +
         np.log(np.mean(np.exp(np.delete(x, k) - np.delete(x, k).max()))))
        for k in range(n)
    ])
    se = (n - 1) * jk.std(ddof=1) / np.sqrt(n)
    return lme, se

lme, se = logmeanexp_and_se(log_likelihood_shared)

print("\n[summary]")
print("  log-mean-exp :", lme)
print("  SE           :", se)
print("  mean         :", log_likelihood_shared.mean())
print("  variance     :", log_likelihood_shared.var(ddof=1))

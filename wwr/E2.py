import os
import numpy as np

# ------------- cache settings -----------------------------------
CACHE_DIR  = "wwr/E2"
CACHE_FILE = os.path.join(CACHE_DIR, "PAL_vanilla.npz")
CACHE_KEY  = "log_likelihood_shared"

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 1) Try to load cached results
# ----------------------------------------------------------------
if os.path.exists(CACHE_FILE):
    print(f"[cache] Found existing results → {CACHE_FILE}")
    log_likelihood_shared = np.load(CACHE_FILE)[CACHE_KEY]

# ----------------------------------------------------------------
# 2) No cache → run the original simulation code unmodified
# ----------------------------------------------------------------
else:
    print("[cache] No cache found – running the full simulation …")

    # ----------------------------------------------------------------
    # >>>>>>>>>>>>>>  ORIGINAL CODE — DO NOT EDIT  <<<<<<<<<<<<<<<<<<<
    # ----------------------------------------------------------------
    #  (Everything between the double lines is your untouched script;
    #   only indentation has changed so it nests inside this block.)
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
    from scipy.special import logsumexp
    from sympy.polys.benchmarks.bench_solvers import uk_10

    from Scripts.measles_PALSMC_perstep import PAL_run_likelihood_lookahead

    plt.ioff()

    import sys
    sys.path.append('wwr/Scripts/')
    from measles_simulator import *
    from measles_PALSMC import *

    if not os.path.exists("E2"):
        os.makedirs("E2")

    os.environ['PYTHONHASHSEED'] = '45'
    random.seed(45)
    np.random.seed(45)
    tf.random.set_seed(45)

    UKbirths_array = np.load("Data/UKbirths_array.npy")
    UKpop_array = np.load("Data/UKpop_array.npy")
    measles_distance_matrix_array = np.load("Data/measles_distance_matrix_array.npy")
    UKmeasles_array = np.load("Data/UKmeasles_array.npy")
    modelA_array = np.load("Data/Parameter/final_parameters_lookahead_A.npy")

    UKbirths = tf.convert_to_tensor(UKbirths_array[18:19, :], dtype=tf.float32)
    UKpop = tf.convert_to_tensor(UKpop_array[18:19, :], dtype=tf.float32)
    UKmeasles = tf.convert_to_tensor(UKmeasles_array[18:19, :], dtype=tf.float32)
    measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array[18:19, 18:19],
                                                   dtype=tf.float32)
    df = pd.read_csv("Data/londonsim.csv")

    data_array = df.values
    UKmeasles = tf.convert_to_tensor(data_array, dtype=tf.float32)

    term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype = tf.float32)
    school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = tf.float32)

    n_cities = tf.constant(1, dtype = tf.int64)
    initial_pop = UKpop[:,0]

    T = 416
    print(T)

    intermediate_steps = 4
    h = tf.constant(14/tf.cast(intermediate_steps, dtype = tf.float32), dtype = tf.float32)
    is_school_term_array, is_start_school_year_array, times_total, times_obs = school_term_and_school_year(T, intermediate_steps, term, school)

    is_school_term_array = tf.convert_to_tensor(is_school_term_array, dtype = tf.float32)
    is_start_school_year_array = tf.convert_to_tensor(is_start_school_year_array, dtype = tf.float32)
    p      = tf.constant(0.759, dtype = tf.float32)
    delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

    # increase the n_experiments for proper variance estimates, use n_experiments to just test the log-likelihood
    n_experiments = 20

    # lookahead
    # shared
    best_parameters = np.load("Data/Parameter/final_parameters_lookahead_A.npy")
    best_parameters = np.ndarray.astype(best_parameters, dtype = np.float32)
    q_mean = tf.constant(np.mean(np.load("Data/q_mean.npy")), dtype = tf.float32)

    n_cities = tf.constant(1, dtype = tf.int64)

    # --- parameter block (same format, new values) -----------------------
    pi_0_1 = 0.02536
    pi_0_2 = 0.0042
    pi_0_3 = 0.000061
    pi_0   = (
        tf.convert_to_tensor(
            [[pi_0_1, pi_0_2, pi_0_3, 1.0 - pi_0_1 - pi_0_2 - pi_0_3]],
            dtype=tf.float32
        )
        * tf.ones((n_cities, 4), dtype=tf.float32)
    )

    initial_pop = UKpop[:, 0]

    beta_bar = tf.convert_to_tensor(6.30 * tf.ones((n_cities, 1)), dtype=tf.float32)
    rho      = tf.convert_to_tensor([0.142], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)
    gamma    = tf.convert_to_tensor([0.0473], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)

    g = tf.convert_to_tensor([[0.0]], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)

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
        max_x = np.max(x)
        est = max_x + np.log(np.mean(np.exp(x - max_x)))
        if not se and not ess:
            return est
        results = {"est": est}
        if se:
            jk_vals = np.empty(n)
            for k in range(n):
                x_minus_k = np.delete(x, k)
                max_x_mk = np.max(x_minus_k)
                jk_vals[k] = max_x_mk + np.log(np.mean(np.exp(x_minus_k - max_x_mk)))
            xse = (n - 1) * np.std(jk_vals, ddof=1) / np.sqrt(n)
            results["se"] = xse
        if ess:
            w = np.exp(x - max_x)
            xss = np.sum(w) ** 2 / np.sum(w**2)
            results["ess"] = xss
        return results

    start_time = time.perf_counter()

    for i in range(n_experiments):
        seed_i = 123 + i
        random.seed(seed_i)
        np.random.seed(seed_i)
        tf.random.set_seed(seed_i)

        value = (
            PAL_run_likelihood_res(
                T, intermediate_steps, UKmeasles, UKbirths, UKpop, g,
                measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a,
                is_school_term_array, is_start_school_year_array, h,
                rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year
            )
        )[0].numpy()

        log_likelihood_shared[i] = value
        print(value)

    elapsed_time = time.perf_counter() - start_time
    print(f"Comp.time: {elapsed_time:.4f} seconds")

    res = logmeanexp(log_likelihood_shared, se=True, ess=True)
    print("Est =", res["est"])
    print("SE  =", res["se"])
    print("ESS =", res["ess"])

    variance_log = np.var(log_likelihood_shared, ddof=1)
    mean_log = np.mean(log_likelihood_shared)

    print("Variance of log likelihoods:", variance_log)
    print("mean of log likelihoods:", mean_log)

    out_file_path = os.path.join("E2", "PAL_vanilla.csv")
    np.savetxt(out_file_path, log_likelihood_shared, delimiter=",")

    # ----------------------------------------------------------------
    # >>>>>>>>>>>>>>>>>>>>  ORIGINAL CODE END  <<<<<<<<<<<<<<<<<<<<<<<
    # ----------------------------------------------------------------

    # ------ save array to cache so future runs can skip heavy work ---
    np.savez(CACHE_FILE, **{CACHE_KEY: log_likelihood_shared})
    print(f"[cache] Results cached → {CACHE_FILE}")

# --------------------------------------------------------------------
# 3) Quick summary (identical whether loaded or freshly computed)
# --------------------------------------------------------------------
def logmeanexp_and_se(x: np.ndarray) -> tuple[float, float]:
    """
    Return (log-mean-exp, jackknife SE) for a 1-D array of log-likelihoods.
    """
    n = x.size
    max_x = x.max()
    lme   = max_x + np.log(np.mean(np.exp(x - max_x)))

    # jackknife
    jk_vals = np.empty(n)
    for k in range(n):
        x_k = np.delete(x, k)
        max_k = x_k.max()
        jk_vals[k] = max_k + np.log(np.mean(np.exp(x_k - max_k)))

    se = (n - 1) * jk_vals.std(ddof=1) / np.sqrt(n)
    return lme, se

lme, se = logmeanexp_and_se(log_likelihood_shared)

print("\n[summary]")
print("  log-mean-exp :", lme)
print("  SE           :", se)
print("  mean         :", log_likelihood_shared.mean())
print("  variance     :", log_likelihood_shared.var(ddof=1))

E2_est = float(lme)    
E2_se  = float(se)      
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

if not os.path.exists("E9"):
    os.makedirs("E9")


os.environ['PYTHONHASHSEED'] = '42'


random.seed(42)

np.random.seed(42)

tf.random.set_seed(42)

UKbirths_array = np.load("Data/UKbirths_array.npy")
UKpop_array = np.load("Data/UKpop_array.npy")
measles_distance_matrix_array = np.load("Data/measles_distance_matrix_array.npy")
UKmeasles_array = np.load("Data/UKmeasles_array.npy")

UKbirths = tf.convert_to_tensor(UKbirths_array, dtype = tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array, dtype = tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype = tf.float32)
UKmeasles = tf.convert_to_tensor(UKmeasles_array, dtype = tf.float32)

df = pd.read_csv("Data/M6.csv")

data_array = df.values

UKmeasles = tf.convert_to_tensor(data_array, dtype=tf.float32)



term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype = tf.float32)
school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = tf.float32)

n_cities = tf.constant(40, dtype = tf.int64)

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

n_cities = tf.constant(40, dtype = tf.int64)

# π₀ —— initial state fractions (restored from current x vector)
pi_0_1 = 0.0304
pi_0_2 = 0.0056
pi_0_3 = 0.000034
pi_0   = (
    tf.convert_to_tensor(
        [[pi_0_1, pi_0_2, pi_0_3, 1.0 - pi_0_1 - pi_0_2 - pi_0_3]],
        dtype=tf.float32
    )
    * tf.ones((n_cities, 4), dtype=tf.float32)
)

initial_pop = UKpop[:, 0]

# β̄ , ρ , γ  (restored from log-space)
beta_bar = tf.convert_to_tensor(6.35 * tf.ones((n_cities, 1)), dtype=tf.float32)   # e^(1.840549)
rho      = tf.convert_to_tensor([0.159], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)  # e^(-1.949856)
gamma    = tf.convert_to_tensor([0.045], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)  # e^(-3.051457)

g = tf.convert_to_tensor([[595]], dtype=tf.float32) * tf.ones((n_cities, 1), dtype=tf.float32)

# a , c , σξ , qvar
a      = tf.constant(0.052, dtype=tf.float32)                    # e^(-1.918923)
c      = tf.constant(0.077 , dtype=tf.float32)                    # e^(-1.517331)
xi_var = tf.convert_to_tensor(0.219 , dtype=tf.float32)            # 10·e^(-3.446486)
q_var  = tf.convert_to_tensor(0.185, dtype=tf.float32)            # e^(-1.186399)

# prior distributions
Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
Q  = tfp.distributions.TruncatedNormal(loc=0.7, scale=q_var, low=0.0, high=1.0)


n_particles = 5000
log_likelihood_shared = np.zeros(n_experiments)

def logmeanexp(x, se=False, ess=False):
    """
    Python implementation of the 'logmeanexp' function, modeling the R version:

    logmeanexp(x, se=FALSE, ess=FALSE)

    Parameters
    ----------
    x : array-like
        Input data for which the log-mean-exp will be calculated.
    se : bool, default=False
        If True, compute the standard error based on a jackknife approach.
    ess : bool, default=False
        If True, compute the effective sample size (ESS).

    Returns
    -------
    float or dict
        If se=False and ess=False, returns a single float:
            log( mean( exp(x) ) )

        If se=True or ess=True, returns a dictionary that always contains
        'est' (the estimate) and may include 'se' (standard error) and/or
        'ess' (effective sample size) depending on the parameters.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]

    # ---- 1. Calculate a stable log-mean-exp estimate ----
    # For numerical stability, subtract max_x before exponentiation
    max_x = np.max(x)
    est = max_x + np.log(np.mean(np.exp(x - max_x)))

    # If neither se nor ess is requested, return just the estimate
    if not se and not ess:
        return est

    # ---- 2. Use jackknife for SE or calculate ESS as needed ----
    results = {"est": est}

    # (Optional) Compute the standard error using jackknife
    if se:
        jk_vals = np.empty(n)
        for k in range(n):
            # Remove x[k], compute logmeanexp on the remaining n-1 values
            x_minus_k = np.delete(x, k)
            max_x_mk = np.max(x_minus_k)
            jk_vals[k] = max_x_mk + np.log(np.mean(np.exp(x_minus_k - max_x_mk)))

        # Jackknife formula: SE = (n-1) * std(jk_vals) / sqrt(n)
        # ddof=1 makes it sample standard deviation
        xse = (n - 1) * np.std(jk_vals, ddof=1) / np.sqrt(n)
        results["se"] = xse

    # (Optional) Compute effective sample size
    if ess:
        # w_i = exp(x_i - max_x)
        # ESS = (sum(w_i))^2 / sum(w_i^2)
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

    value = (PAL_run_likelihood_res(T, intermediate_steps, UKmeasles , UKbirths, UKpop, g, measles_distance_matrix, initial_pop, pi_0, beta_bar, p, a, is_school_term_array, is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, n_particles, delta_year))[0].numpy()

    log_likelihood_shared[i] = value

    print(value)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Comp.time: {elapsed_time:.4f} seconds")

res = logmeanexp(log_likelihood_shared, se=True, ess=True)
print("Est =", res["est"])
print("SE  =", res["se"])
print("ESS =", res["ess"])

variance_log = np.var(log_likelihood_shared, ddof=1)
print("Variance of log likelihoods:", variance_log)

out_file_path = os.path.join("E9", "PAL_res_40.csv")

np.savetxt(out_file_path, log_likelihood_shared, delimiter=",")



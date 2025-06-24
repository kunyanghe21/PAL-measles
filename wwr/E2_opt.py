import os
import sys
import time
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.optimize import minimize

sys.path.append("wwr/Scripts/")
from measles_simulator import *
from measles_PALSMC import *


os.environ["PYTHONHASHSEED"] = "45"
random.seed(45)
np.random.seed(45)
tf.random.set_seed(45)

CACHE_DIR = "wwr/E2"
CACHE_FILE = os.path.join(CACHE_DIR, "E2_param_exp.npz")
CACHE_KEY = "E2_param_exp"

os.makedirs(CACHE_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"[cache] Found existing results → {CACHE_FILE}")
    E2_param_exp = np.load(CACHE_FILE)[CACHE_KEY]
else:
    print("[cache] No cache found – running optimization …")

    UKbirths_array = np.load("wwr/Data/UKbirths_array.npy")
    UKpop_array = np.load("wwr/Data/UKpop_array.npy")
    measles_distance_matrix_array = np.load("wwr/Data/measles_distance_matrix_array.npy")
    UKmeasles_array = np.load("wwr/Data/UKmeasles_array.npy")

    UKbirths = tf.convert_to_tensor(UKbirths_array[18:19, :], dtype=tf.float32)
    UKpop = tf.convert_to_tensor(UKpop_array[18:19, :], dtype=tf.float32)
    UKmeasles = tf.convert_to_tensor(UKmeasles_array[18:19, :], dtype=tf.float32)
    measles_distance_matrix = tf.convert_to_tensor(
        measles_distance_matrix_array[18:19, 18:19], dtype=tf.float32
    )

    df = pd.read_csv("wwr/Data/londonsim.csv")

    data_array = df.values
    UKmeasles = tf.convert_to_tensor(data_array, dtype=tf.float32)

    n_cities = tf.constant(1, dtype=tf.int64)
    initial_pop = UKpop[:, 0]

    p = tf.constant(0.759, dtype=tf.float32)
    p_bound = -np.log(2) - np.log(float(p.numpy()))

    delta_year = tf.convert_to_tensor([[1 / 50]], dtype=tf.float32) * tf.ones((n_cities, 4), dtype=tf.float32)

    T = UKmeasles.shape[1]
    intermediate_steps = 4
    h = tf.constant(14 / tf.cast(intermediate_steps, dtype=tf.float32), dtype=tf.float32)

    is_school_term_array = tf.zeros((T, intermediate_steps), dtype=tf.float32)
    is_start_school_year_array = tf.zeros((T, intermediate_steps), dtype=tf.float32)

    x_0 = np.array(
        [
 -3.292756,   # log(0.0371514202)
 -5.053540,   # log(0.00637310651)
 -9.321924,   # log(8.93450088e-05)
  1.988469,   # log(7.30434201)
 -1.806411,   # log(0.164242541)
 -2.909371,   # log(0.0545073885)
 -1.910585,   # log(0.147916508)
 -2.604172,   # log(0.0740463307)
 -3.710095,   # log(0.0244794055)
 -1.980585    # log(0.138058450)
]
    )

    g_fixed = 0.0
    print("We are fixing g to:", g_fixed)

    n_particles = 15000

    def optimization_func(x_optim):
        x = np.exp(x_optim)
        if (
            x_optim[0] > 0
            or x_optim[1] > 0
            or x_optim[2] > 0
            or x_optim[6] > p_bound
            or x_optim[7] > 0
            or x_optim[9] > 1
        ):
            return 2e5
        if 1 - x[0] - x[1] - x[2] < 0:
            return 2e5
        pi_0_init = np.array([x[0], x[1], x[2], 1 - x[0] - x[1] - x[2]], dtype=np.float32)
        beta_bar_init = tf.convert_to_tensor([x[3]], dtype=tf.float32)
        rho_init = tf.convert_to_tensor([x[4]], dtype=tf.float32)
        gamma_init = tf.convert_to_tensor([x[5]], dtype=tf.float32)
        g_init = tf.constant([g_fixed], dtype=tf.float32)
        a_init = tf.convert_to_tensor(x[6], dtype=tf.float32)
        c_init = tf.convert_to_tensor(x[7], dtype=tf.float32)
        xi_var = 10.0 * tf.convert_to_tensor(x[8], dtype=tf.float32)
        q_var = tf.convert_to_tensor(x[9], dtype=tf.float32)
        q_mean_ = tf.convert_to_tensor(0.7, dtype=tf.float32)
        Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
        Q = tfp.distributions.TruncatedNormal(loc=q_mean_, scale=q_var, low=0.0, high=1.0)
        value = -PAL_run_likelihood_res(
            T,
            intermediate_steps,
            UKmeasles,
            UKbirths,
            UKpop,
            g_init * tf.ones((n_cities, 1), dtype=tf.float32),
            measles_distance_matrix,
            initial_pop,
            pi_0_init * tf.ones((n_cities, 4), dtype=tf.float32),
            beta_bar_init * tf.ones((n_cities, 1), dtype=tf.float32),
            p,
            a_init,
            is_school_term_array,
            is_start_school_year_array,
            h,
            rho_init * tf.ones((n_cities, 1), dtype=tf.float32),
            gamma_init * tf.ones((n_cities, 1), dtype=tf.float32),
            Xi,
            Q,
            c_init,
            n_cities,
            n_particles,
            delta_year,
        )[0].numpy()
        print("[DEBUG] x_optim =", x_optim, " -> value =", value)
        return value

    bounds = (
        (-20, -0.5),
        (-20, -0.5),
        (-20, -0.5),
        (0, 4),
        (-4, 0),
        (-4, 0),
        (-20, p_bound),
        (-20, -0.01),
        (-10, 3),
        (-10, 1)
    )

    initial_loss = optimization_func(x_0)
    print("initial_loss =", initial_loss)

    res = minimize(
        optimization_func,
        x_0,
        bounds=bounds,
        method="SLSQP",
        options={"eps": 0.3, "maxiter": 30},
    )

    E2_param = res.x
    print("Optimization done! final x =", E2_param)

    E2_param_exp = np.exp(E2_param)
    np.savez(CACHE_FILE, **{CACHE_KEY: E2_param_exp})
    print(f"[cache] Results cached → {CACHE_FILE}")

print(E2_param_exp)



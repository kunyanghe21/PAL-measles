import os, sys, random, time
import numpy as np, pandas as pd, tensorflow as tf, tensorflow_probability as tfp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

os.environ["PYTHONHASHSEED"] = "41"
random.seed(41)
np.random.seed(41)
tf.random.set_seed(41)

CACHE_DIR = "wwr/E10"
CACHE_FILE = os.path.join(CACHE_DIR, "E10_param_exp.npz")
CACHE_KEY = "E10_param_exp"
os.makedirs(CACHE_DIR, exist_ok=True)

if os.path.exists(CACHE_FILE):
    print(f"[cache] Found existing results → {CACHE_FILE}")
    E10_param_exp = np.load(CACHE_FILE)[CACHE_KEY]
    print("Optimization done! final x =", E10_param_exp)
else:
    plt.ioff()
    sys.path.append("wwr/Scripts/")
    from measles_simulator import *
    from measles_PALSMC import *

    UKbirths_array = np.load("wwr/Data/UKbirths_array.npy")
    UKpop_array = np.load("wwr/Data/UKpop_array.npy")
    measles_distance_matrix_array = np.load("wwr/Data/measles_distance_matrix_array.npy")
    UKmeasles_array = np.load("wwr/Data/UKmeasles_array.npy")

    UKbirths = tf.convert_to_tensor(UKbirths_array, dtype=tf.float32)
    UKpop = tf.convert_to_tensor(UKpop_array, dtype=tf.float32)
    measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype=tf.float32)
    UKmeasles = tf.convert_to_tensor(UKmeasles_array, dtype=tf.float32)

    df = pd.read_csv("wwr/Data/M40.csv")
    data_array = df.values
    UKmeasles = tf.convert_to_tensor(data_array, dtype=tf.float32)

    n_cities = tf.constant(40, dtype=tf.int64)
    initial_pop = UKpop[:, 0]
    p = tf.constant(0.759, dtype=tf.float32)
    q_mean = tf.constant(np.mean(np.load("wwr/Data/q_mean.npy")), dtype=tf.float32)
    delta_year = tf.convert_to_tensor([[1 / 50]], dtype=tf.float32) * tf.ones((n_cities, 4), dtype=tf.float32)

    T = UKmeasles.shape[1]
    intermediate_steps = 4
    h = tf.constant(14 / tf.cast(intermediate_steps, dtype=tf.float32), dtype=tf.float32)
    is_school_term_array = tf.zeros((T, intermediate_steps), dtype=tf.float32)
    is_start_school_year_array = tf.zeros((T, intermediate_steps), dtype=tf.float32)

    x_0 = np.array([
 -3.674239,  # x[0]
    -5.472271,  # x[1]
    -9.705021,  # x[2]
    1.840549,  # x[3]
    -1.949856,  # x[4]
    -3.051457,  # x[5]
    1.7917,  # x[6]
    -1.918923,  # x[7]
    -1.517331,  # x[8]
    -3.446486,  # x[9]
    -1.186399  # x[10]
]          
    )

    n_particles = 5000

    def optimization_func(x_optim):
        if (
            x_optim[0] > 0
            or x_optim[1] > 0
            or x_optim[2] > 0
            or x_optim[7] > (-np.log(2) - np.log(p.numpy()))
            or x_optim[8] > 0
            or x_optim[10] > 1
        ):
            return 200000
        x = np.exp(x_optim)
        if 1 - x[0] - x[1] - x[2] < 0:
            return 200000
        pi_0_init = np.array([x[0], x[1], x[2], 1 - x[0] - x[1] - x[2]], dtype=np.float32)
        beta_bar_init = tf.convert_to_tensor([x[3:4]], dtype=tf.float32)
        rho_init = tf.convert_to_tensor([x[4:5]], dtype=tf.float32)
        gamma_init = tf.convert_to_tensor([x[5:6]], dtype=tf.float32)
        g_init = 100 * tf.convert_to_tensor([x[6:7]], dtype=tf.float32)
        a_init = tf.convert_to_tensor(x[7], dtype=tf.float32)
        c_init = tf.convert_to_tensor(x[8], dtype=tf.float32)
        xi_var = 10 * tf.convert_to_tensor(x[9], dtype=tf.float32)
        q_var = tf.convert_to_tensor(x[10], dtype=tf.float32)
        pi_0_init_transform = pi_0_init * tf.ones((n_cities, 4), dtype=tf.float32)
        beta_bar_init_transform = beta_bar_init * tf.ones((n_cities, 1), dtype=tf.float32)
        rho_init_transform = rho_init * tf.ones((n_cities, 1), dtype=tf.float32)
        gamma_init_transform = gamma_init * tf.ones((n_cities, 1), dtype=tf.float32)
        g_init_transform = g_init * tf.ones((n_cities, 1), dtype=tf.float32)
        Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
        Q = tfp.distributions.TruncatedNormal(0.7, q_var, 0, 1)
        value = -(
            PAL_run_likelihood_lookahead(
                T,
                intermediate_steps,
                UKmeasles,
                UKbirths,
                UKpop,
                g_init_transform,
                measles_distance_matrix,
                initial_pop,
                pi_0_init_transform,
                beta_bar_init_transform,
                p,
                a_init,
                is_school_term_array,
                is_start_school_year_array,
                h,
                rho_init_transform,
                gamma_init_transform,
                Xi,
                Q,
                c_init,
                n_cities,
                n_particles,
                delta_year,
            )
        )[0].numpy()
        print("[DEBUG] x_optim =", x_optim, " -> value =", value)
        return value

    bnds = (
        (-20, -0.5),
        (-20, -0.5),
        (-20, -0.5),
        (0, 2),
        (-4, 0),
        (-4, 0),
        (-1, 5),
        (-20, -np.log(2) - np.log(p)),
        (-20, -0.01),
        (-10, 3),
        (-10, -0.1),
    )

    res = minimize(
        optimization_func,
        x_0,
        bounds=bnds,
        method="SLSQP",
        options={"eps": 0.5, "maxiter": 50},
    )

    E10_param = res.x
    print("Optimization done! final x =", E10_param)

    E10_param_exp = np.exp(E10_param)
    np.savez(CACHE_FILE, **{CACHE_KEY: E10_param_exp})
    print(f"[cache] Results cached → {CACHE_FILE}")

print(E10_param_exp)
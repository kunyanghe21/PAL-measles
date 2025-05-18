import numpy as np
import time
import random
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys

from Scripts.measles_PALSMC_perstep import PAL_run_likelihood_res

sys.path.append('Scripts/')
from measles_simulator import *
from measles_PALSMC import *

plt.ioff()



UKbirths_array = np.load("Data/UKbirths_array.npy")
UKpop_array = np.load("Data/UKpop_array.npy")
measles_distance_matrix_array = np.load("Data/measles_distance_matrix_array.npy")
UKmeasles_array = np.load("Data/UKmeasles_array.npy")


UKbirths = tf.convert_to_tensor(UKbirths_array[18:19, :], dtype=tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array[18:19, :], dtype=tf.float32)
UKmeasles = tf.convert_to_tensor(UKmeasles_array[18:19, :], dtype=tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array[18:19, 18:19],
                                               dtype=tf.float32)

df = pd.read_csv("Data/londonsim.csv")


data_array = df.values

UKmeasles = tf.convert_to_tensor(data_array, dtype=tf.float32)

n_cities = tf.constant(1, dtype=tf.int64)

initial_pop = UKpop[:, 0]
p = tf.constant(0.759, dtype=tf.float32)
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
    -1.918923,  # x[7]
    -1.517331,  # x[8]
    -3.446486,  # x[9]
    -1.186399  # x[10]
])

q_mean = tf.constant(0.7, dtype=tf.float32)
n_particles = 5000

# ------------------------- #
# 关键改动2：定义一个固定的 g 值
# 这里假设我们固定 g = exp(2.30383363)*100
# 你也可以直接固定成一个具体数，例如 g_fixed=1000.0
# ------------------------- #
g_fixed = float(0)
print("We are fixing g to:", g_fixed)


def optimization_func(x_optim):
    x = np.exp(x_optim)
    if (
            x_optim[0] > 0 or
            x_optim[1] > 0 or
            x_optim[2] > 0 or
            x_optim[6] > (-np.log(2) - np.log(p.numpy())) or
            x_optim[7] > 0 or
            x_optim[9] > 1
    ):
        return 2e5

    if (1 - x[0] - x[1] - x[2]) < 0:
        return 2e5

    pi_0_init = np.array([x[0], x[1], x[2], 1 - x[0] - x[1] - x[2]], dtype=np.float32)

    # 提取 beta_bar, rho, gamma
    beta_bar_init = tf.convert_to_tensor([x[3:4]], dtype=tf.float32)  # x[3]
    rho_init = tf.convert_to_tensor([x[4:5]], dtype=tf.float32)  # x[4]
    gamma_init = tf.convert_to_tensor([x[5:6]], dtype=tf.float32)  # x[5]

    g_init = tf.constant([g_fixed], dtype=tf.float32)  # shape=[1]

    a_init = tf.convert_to_tensor(x[6], dtype=tf.float32)
    c_init = tf.convert_to_tensor(x[7], dtype=tf.float32)
    xi_var = 10 * tf.convert_to_tensor(x[8], dtype=tf.float32)
    q_var = tf.convert_to_tensor(x[9], dtype=tf.float32)

    pi_0_init_transform = pi_0_init * tf.ones((n_cities, 4), dtype=tf.float32)
    beta_bar_init_transform = beta_bar_init * tf.ones((n_cities, 1), dtype=tf.float32)
    rho_init_transform = rho_init * tf.ones((n_cities, 1), dtype=tf.float32)
    gamma_init_transform = gamma_init * tf.ones((n_cities, 1), dtype=tf.float32)
    g_init_transform = g_init * tf.ones((n_cities, 1), dtype=tf.float32)

    Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
    Q = tfp.distributions.TruncatedNormal(q_mean, q_var, 0, 1)

    value = -(
        PAL_run_likelihood_res(
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
            delta_year
        )[0].numpy()
    )

    print("[DEBUG] x_optim =", x_optim, " -> value =", value)
    return value

bnds = (
    (-20, -0.5),  # x[0]
    (-20, -0.5),  # x[1]
    (-20, -0.5),  # x[2]
    (0, 2),  # x[3]
    (-4, 0),  # x[4]
    (-4, 0),  # x[5]
    (-20, -np.log(2) - np.log(p)),  # x[7] => 现在对应 x_optim[6]
    (-20, -0.01),  # x[8] => 现在对应 x_optim[7]
    (-10, 3),  # x[9] => 现在对应 x_optim[8]
    (-10, -0.1)  # x[10]=> 现在对应 x_optim[9]
)

initial_loss = optimization_func(x_0)
print("initial_loss =", initial_loss)

res = minimize(
    optimization_func,
    x_0,
    bounds=bnds,
    method='SLSQP',
    options={"eps": 0.5, "maxiter": 100}
)

final_parameters_lookahead_A = res.x
print("Optimization done! final x =", final_parameters_lookahead_A)
print("Fixed g value was:", g_fixed)

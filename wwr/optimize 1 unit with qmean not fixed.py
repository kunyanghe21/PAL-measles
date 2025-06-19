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
sys.path.append('wwr/Scripts/')
from measles_simulator import *
from measles_PALSMC import *

plt.ioff()

UKbirths_array = np.load("wwr/Data/UKbirths_array.npy")
UKpop_array = np.load("wwr/Data/UKpop_array.npy")
measles_distance_matrix_array = np.load("wwr/Data/measles_distance_matrix_array.npy")
UKmeasles_array = np.load("wwr/Data/UKmeasles_array.npy")

UKbirths = tf.convert_to_tensor(UKbirths_array[18:19, :], dtype=tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array[18:19, :], dtype=tf.float32)
UKmeasles = tf.convert_to_tensor(UKmeasles_array[18:19, :], dtype=tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array[18:19, 18:19],
                                               dtype=tf.float32)

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
 -3.827630,   # x[0] = log(S_0)             ,  S_0  = 0.021761124566
 -5.844465,   # x[1] = log(E_0)             ,  E_0  = 0.002895884298
 -9.720592,   # x[2] = log(I_0)             ,  I_0  = 6.003442028e-05
  1.201501,   # x[3] = log(betabar)         ,  betabar = 3.325105802
 -3.138990,   # x[4] = log(rho)             ,  rho = 0.0433265237
 -2.806183,   # x[5] = log(gamma)           ,  gamma = 0.0604352097
 -0.493285,   # x[6] = log(a)               ,  a = 0.6106172198
 -0.114045,   # x[7] = log(c)               ,  c = 0.8922177383
 -2.943745,   # x[8] = log(sigma_xi / 10)   ,  sigma_xi = 0.526681063
 -2.994803    # x[9] = log(q_var = psi)     ,  q_var = 0.112499902
 -0.69
])


g_fixed = float(0)
print("We are fixing g to:", g_fixed)

n_particles = 5000

def optimization_func(x_optim):


    x = np.exp(x_optim)
    if (
        x_optim[0] > 0 or
        x_optim[1] > 0 or
        x_optim[2] > 0 or
        x_optim[6] > (-np.log(2) - np.log(p.numpy())) or
        x_optim[7] > 0 or
        x_optim[9] > 1 or
        x_optim[10] >= 0
    ):
        return 2e5

    # pi_0 的 4 个分量 (S, E, I, R)
    if (1 - x[0] - x[1] - x[2]) < 0:
        return 2e5

    pi_0_init = np.array(
        [x[0], x[1], x[2], 1 - x[0] - x[1] - x[2]],
        dtype=np.float32
    )

    beta_bar_init = tf.convert_to_tensor([x[3]], dtype=tf.float32)  # shape=[1]
    rho_init      = tf.convert_to_tensor([x[4]], dtype=tf.float32)  # shape=[1]
    gamma_init    = tf.convert_to_tensor([x[5]], dtype=tf.float32)  # shape=[1]
    g_init        = tf.constant([g_fixed], dtype=tf.float32)        # shape=[1]

    a_init  = tf.convert_to_tensor(x[6], dtype=tf.float32)
    c_init  = tf.convert_to_tensor(x[7], dtype=tf.float32)
    xi_var  = 10.0 * tf.convert_to_tensor(x[8], dtype=tf.float32)  # Xi 的形参、率参 = xi_var
    q_var   = tf.convert_to_tensor(x[9],  dtype=tf.float32)
    q_mean_ = tf.convert_to_tensor(x[10], dtype=tf.float32)  # 新增: q_mean

    # 扩展到 n_cities 维
    pi_0_init_transform   = pi_0_init[None, :]      # shape=[1,4]
    beta_bar_init_trans   = beta_bar_init[None, :]  # shape=[1,1]
    rho_init_trans        = rho_init[None, :]       # shape=[1,1]
    gamma_init_trans      = gamma_init[None, :]     # shape=[1,1]
    g_init_trans          = g_init[None, :]         # shape=[1,1]

    # Xi, Q 是随机分布
    Xi = tfp.distributions.Gamma(
        concentration=xi_var,
        rate=xi_var
    )
    Q  = tfp.distributions.TruncatedNormal(
        loc=q_mean_,
        scale=q_var,
        low=0.0,
        high=1.0
    )


    value = -(
        PAL_run_likelihood_lookahead(
            T,
            intermediate_steps,
            UKmeasles,
            UKbirths,
            UKpop,
            g_init_trans,
            measles_distance_matrix,
            initial_pop,
            pi_0_init_transform,
            beta_bar_init_trans,
            p,
            a_init,
            is_school_term_array,
            is_start_school_year_array,
            h,
            rho_init_trans,
            gamma_init_trans,
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
    (-20, -0.5),                 # x[0]
    (-20, -0.5),                 # x[1]
    (-20, -0.5),                 # x[2]
    (0, 4),                      # x[3]
    (-4, 0),                     # x[4]
    (-4, 0),                     # x[5]
    (-20, -np.log(2) - np.log(p)),  # x[6]: a
    (-20, -0.01),               # x[7]: c
    (-10, 3),                   # x[8]: xi_var factor
    (-10, 1),                   # x[9]: q_var
    (-10, 0)                    # x[10]: q_mean (log space)
)

initial_loss = optimization_func(x_0)
print("initial_loss =", initial_loss)


res = minimize(
    optimization_func,
    x_0,
    bounds=bnds,
    method='SLSQP',
    options={"eps": 0.1, "maxiter": 100}
)

final_parameters_lookahead_A = res.x
print("Optimization done! final x =", final_parameters_lookahead_A)
print("Fixed g value was:", g_fixed)


param_exp = np.exp(final_parameters_lookahead_A)
pi_0_S = param_exp[0]
pi_0_E = param_exp[1]
pi_0_I = param_exp[2]
pi_0_R = 1 - pi_0_S - pi_0_E - pi_0_I
beta_bar = param_exp[3]
rho      = param_exp[4]
gamma    = param_exp[5]
a_       = param_exp[6]
c_       = param_exp[7]
xi_var_  = 10.0 * param_exp[8]
q_var_   = param_exp[9]
q_mean_  = param_exp[10]

print("\n=== Transformed back to real scale ===")
print("pi_0 =", [pi_0_S, pi_0_E, pi_0_I, pi_0_R])
print("beta_bar =", beta_bar)
print("rho =", rho)
print("gamma =", gamma)
print("a =", a_)
print("c =", c_)
print("xi_var =", xi_var_)
print("q_var =", q_var_)
print("q_mean =", q_mean_, "(should be <1 if final x_optim[10]<0)")

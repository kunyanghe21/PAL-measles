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

# -------------------------
# 初始参数: 扩展到 11 个维度
# 最后一维 (x_optim[10]) 用来表示 q_mean 的对数
# 这里给出一个与之前注释更匹配的初始化
# 其中 exp(-0.356675) ≈ 0.7
# -------------------------
x_0 = np.array([
    -3.674239,  # x[0]: pi_0 fraction S
    -5.472271,  # x[1]: pi_0 fraction E
    -9.705021,  # x[2]: pi_0 fraction I
     1.840549,  # x[3]: beta_bar
    -1.949856,  # x[4]: rho
    -3.051457,  # x[5]: gamma
    -1.918923,  # x[6]: a
    -1.517331,  # x[7]: c
    -3.446486,  # x[8]: xi_var (要再乘以 10)
    -1.186399,  # x[9]: q_var
    -0.356675   # x[10]: q_mean 的对数，使 exp(-0.356675)=0.7
])

# -------------------------
# 关键改动：固定 g 不参与优化
# -------------------------
g_fixed = float(0)
print("We are fixing g to:", g_fixed)

n_particles = 5000

def optimization_func(x_optim):
    """
    x_optim: 参数在 log 空间 (或类似约束后的空间)，长度为 11.
    """
    # 先将前 11 个都用 np.exp() 方式映射到正数域
    x = np.exp(x_optim)

    # 做一些边界检查，若不满足就返回一个很大的值（相当于拒绝）
    # ---------------------------------------------------------
    # 1) 要求 x_optim[0..2]<0, 这样 x[0..2]<1, 用于 pi_0 的三部分
    # 2) 要求 x_optim[6] < -np.log(2) - np.log(p) (来自 a 的某个物理含义？)
    # 3) 要求 x_optim[7]<0 (比如 c 也要小于1？你原先的注释如此)
    # 4) 要求 x_optim[9] <= 1 (控制 q_var 不要太大)
    # 5) 新增：要求 x_optim[10]<0, 这样确保 q_mean < 1
    # ---------------------------------------------------------
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

    # 调用 PAL_run_likelihood_lookahead(...) 计算似然
    # 注意把新的 q_mean, q_var 传入
    value = -(
        PAL_run_likelihood_res(
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

# 定义 11 维的 bounds, 对应 x_optim[0..10]
#   - 其中最后一项 (-10, 0) 用来保证 q_mean = exp(x_optim[10]) 在 (0, 1)
bnds = (
    (-20, -0.5),                 # x[0]
    (-20, -0.5),                 # x[1]
    (-20, -0.5),                 # x[2]
    (0, 2),                      # x[3]
    (-4, 0),                     # x[4]
    (-4, 0),                     # x[5]
    (-20, -np.log(2) - np.log(p)),  # x[6]: a
    (-20, -0.01),               # x[7]: c
    (-10, 3),                   # x[8]: xi_var factor
    (-10, 1),                   # x[9]: q_var
    (-10, 0)                    # x[10]: q_mean (log space)
)

# 查看初始 loss
initial_loss = optimization_func(x_0)
print("initial_loss =", initial_loss)

# 执行优化
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

# 如果需要查看最后一次迭代得到的真实物理参数，可做类似：
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

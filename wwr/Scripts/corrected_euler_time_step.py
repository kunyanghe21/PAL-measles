import numpy as np
import time
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def is_school_term(t_days, term, school):
    t_days_norm = t_days % 365
    time = int(np.sum(term < t_days_norm))
    return school[time]


def is_start_school_year(t_days):
    t_days_norm = t_days % 365
    return t_days_norm == 248.5


def school_term_and_school_year(T, intermediate_steps, term, school):
    is_school_term_array = np.zeros((T, intermediate_steps))
    is_start_school_year_array = np.zeros((T, intermediate_steps))
    times_total = np.zeros((T * intermediate_steps))
    times_obs = np.zeros((T))
    index = 0

    for t_obs in range(0, T):
        for t_intermediate in range(1, intermediate_steps + 1):
            t_days = 14 * t_obs + t_intermediate * (14 / intermediate_steps) + tf.math.floor(t_obs / 26)
            times_total[index] = t_days
            index += 1

            is_school_term_array[t_obs, t_intermediate - 1] = (is_school_term(t_days, term, school))
            is_start_school_year_array[t_obs, t_intermediate - 1] = (is_start_school_year(t_days))

        times_obs[t_obs] = t_days

    return is_school_term_array, is_start_school_year_array, times_total, times_obs


def log_factorial(y_t):
    """此处如原代码，只是对前40个城市做循环。若n_cities>40，会有越界风险。"""
    def cond(city_index, log_sum):
        return city_index < 40  # 保持原逻辑

    def body(city_index, log_sum):
        return city_index + 1, log_sum + tf.reduce_sum(tf.math.log(
            tf.linspace(
                tf.constant(1, dtype=tf.float32),
                y_t[city_index, 0, 0],
                tf.cast(y_t[city_index, 0, 0], dtype=tf.int64)
            )
        ))

    output = tf.while_loop(cond, body, loop_vars=[0, tf.constant(0, dtype=tf.float32)])
    return output[1]


def log_correction(T, UKmeasles):
    def body(input, t_obs):
        UKmeasles_t = UKmeasles[:, t_obs + 1 : t_obs + 2]
        return log_factorial(tf.expand_dims(UKmeasles_t, axis=-1))

    return tf.scan(body, tf.range(0, T, dtype=tf.int64), initializer=(tf.constant(0, dtype=tf.float32)))


def PAL_assemble_K(h, infection_rate, rho, gamma):
    prob_inf = tf.expand_dims(1 - tf.exp(-h * infection_rate), axis=2)
    K_r1 = tf.concat((1 - prob_inf, prob_inf, tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf))), axis=-1)

    prob_latent = tf.ones((tf.shape(prob_inf))) * tf.expand_dims(tf.expand_dims(1 - tf.exp(-h * rho), axis=2), axis=0)
    K_r2 = tf.concat((tf.zeros(tf.shape(prob_inf)), 1 - prob_latent, prob_latent, tf.zeros(tf.shape(prob_inf))),
                     axis=-1)

    prob_recover = tf.ones((tf.shape(prob_inf))) * tf.expand_dims(tf.expand_dims(1 - tf.exp(-h * gamma), axis=2),
                                                                  axis=0)
    K_r3 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), 1 - prob_recover, prob_recover,),
                     axis=-1)

    K_r4 = tf.concat((tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)), tf.zeros(tf.shape(prob_inf)),
                      tf.ones(tf.shape(prob_inf))), axis=-1)

    K_t = tf.concat((K_r1, K_r2, K_r3, K_r4), axis=2)
    return K_t

@tf.function
def PAL_scan_intermediate(
    bar_lambda_tprev,
    is_start_school_year_array_t_obs,
    intermediate_steps,
    UKbirths_t,
    c,
    n_cities,
    n_particles,
    delta_year,
    is_school_term_array_t,
    v,
    pop_t,
    xi_t,  # 保留原形参，但不再使用
    h,
    rho,
    gamma,
    beta_bar,
    p,
    a,
    Xi      # 新增: Gamma/xi 分布，用于在inner step里采样
):
    def body(input_, t_intermediate):
        bar_lambda_tm1, Lambda_tm1 = input_

        infected_prop_t = tf.einsum("pcm,c->pcm", bar_lambda_tm1[..., 2:3], 1 / pop_t)

        beta_t = (1 + 2 * (1 - p) * a) * beta_bar * is_school_term_array_t[t_intermediate] \
                 + (1 - 2 * p * a) * beta_bar * (1 - is_school_term_array_t[t_intermediate])

        spatial_infection = infected_prop_t + tf.reduce_sum(
            (v / pop_t) * (tf.transpose(infected_prop_t, perm=[0, 2, 1]) - infected_prop_t),
            axis=2, keepdims=True
        )

        # ★★ 在每个子步 Euler step 中采样 xi_sub
        xi_sub = Xi.sample((n_particles, n_cities, 1))

        infection_rate = beta_t * xi_sub * spatial_infection

        K_tprev = PAL_assemble_K(h, infection_rate, rho, gamma)

        alpha_t = (
            c * UKbirths_t * is_start_school_year_array_t_obs[t_intermediate]
            + ((1 - c) / (26 * intermediate_steps - 1)) * UKbirths_t
              * (1 - is_start_school_year_array_t_obs[t_intermediate])
        )
        alpha_t = tf.expand_dims(alpha_t, axis=0)
        alpha_t = tf.concat(
            (alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))),
            axis=-1
        )

        surv_prob = 1 - tf.expand_dims(delta_year / (26 * intermediate_steps), axis=0)
        Lambda_t = tf.einsum("pnm,pnmk->pnmk", bar_lambda_tm1 * surv_prob, K_tprev)
        bar_lambda_t = tf.reduce_sum(Lambda_t, axis=2) + alpha_t

        return bar_lambda_t, Lambda_t

    Lambda_tprev = tf.zeros((n_particles, n_cities, 4, 4))
    lambda_, Lambda_ = tf.scan(
        body,
        tf.range(0, intermediate_steps, dtype=tf.int64),
        initializer=(bar_lambda_tprev, Lambda_tprev)
    )

    return lambda_, Lambda_


###############################################################################
# PAL_body_run_res_low: 保留xi_t形参,但在调用 scan_intermediate 时再传 Xi
###############################################################################
@tf.function
def PAL_body_run_res_low(
    bar_lambda_tm1,
    intermediate_steps,
    UKmeasles_t,
    UKbirths_t,
    pop_t,
    beta_bar,
    p,
    a,
    is_school_term_array_t,
    is_start_school_year_array_t_obs,
    h,
    rho,
    gamma,
    xi_t,  # 保留，但不在inner step中使用
    Q,
    c,
    n_cities,
    n_particles,
    delta_year,
    v,
    Xi      # 新增
):
    lambda_, Lambda_ = PAL_scan_intermediate(
        bar_lambda_tm1,
        is_start_school_year_array_t_obs,
        intermediate_steps,
        UKbirths_t,
        c,
        n_cities,
        n_particles,
        delta_year,
        is_school_term_array_t,
        v,
        pop_t,
        xi_t,   # 仍保留，但无实际用
        h,
        rho,
        gamma,
        beta_bar,
        p,
        a,
        Xi      # 在子步里真正用它
    )

    f_xi = tf.reduce_sum(Lambda_, axis=0)[:, :, 2, 3]

    b = (-Q.parameters["scale"]**2 * f_xi + Q.parameters["loc"])
    mu_r = (b + tf.math.sqrt(b*b + 4*Q.parameters["scale"]**2*tf.transpose(UKmeasles_t))) / 2
    mu_r_norm = mu_r + tf.cast((mu_r == 0), dtype=tf.float32)
    sigma_r = tf.math.sqrt(
        1. / (
            (tf.transpose(UKmeasles_t)/(mu_r_norm*mu_r_norm))
            + 1./(Q.parameters["scale"]**2)
        )
    )

    q_t = tfp.distributions.TruncatedNormal(mu_r, sigma_r, 0., 1.).sample()
    q_t = tf.expand_dims(tf.expand_dims(q_t, axis=-1), axis=-1)

    Q_t_r3 = tf.concat(
        (tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), tf.zeros(tf.shape(q_t)), q_t),
        axis=-1
    )
    Q_t = tf.concat(
        (tf.zeros(tf.shape(Q_t_r3)), tf.zeros(tf.shape(Q_t_r3)), Q_t_r3, tf.zeros(tf.shape(Q_t_r3))),
        axis=-2
    )

    y_t = tf.expand_dims(UKmeasles_t, axis=-1)
    Y_t_r3 = tf.concat(
        (tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), tf.zeros(tf.shape(y_t)), y_t),
        axis=-1
    )
    Y_t = tf.concat(
        (tf.zeros(tf.shape(Y_t_r3)), tf.zeros(tf.shape(Y_t_r3)), Y_t_r3, tf.zeros(tf.shape(Y_t_r3))),
        axis=-2
    )

    M = tf.reduce_sum(tf.expand_dims(Q_t, axis=0)*Lambda_, axis=0)

    bar_Lambda_t = (
        (1 - Q_t)*Lambda_[-1,...]
        + tf.where(
            (Y_t*Lambda_[-1,...]*Q_t) == 0,
            Y_t*Lambda_[-1,...]*Q_t,
            Y_t*Lambda_[-1,...]*Q_t/M
        )
    )

    likelihood_t_tm1 = (
        tfp.distributions.Poisson(M[...,2,3]).log_prob(tf.transpose(UKmeasles_t))
        + Q.log_prob(q_t[...,0,0])
        - tfp.distributions.TruncatedNormal(mu_r, sigma_r, 0., 1.).log_prob(q_t[...,0,0])
    )

    return likelihood_t_tm1, bar_Lambda_t


###############################################################################
# PAL_run_likelihood_lookahead: 额外多传 Xi 参数进 body_run_res_low
###############################################################################
@tf.function
def PAL_run_likelihood_lookahead(
    T,
    intermediate_steps,
    UKmeasles,
    UKbirths,
    UKpop,
    g,
    measles_distance_matrix,
    initial_pop,
    pi_0,
    beta_bar,
    p,
    a,
    is_school_term_array,
    is_start_school_year_array,
    h,
    rho,
    gamma,
    Xi,  # 新增
    Q,
    c,
    n_cities,
    n_particles,
    delta_year
):
    v = (g * measles_distance_matrix)

    def cond(t_obs, input_):
        return t_obs < (T - 1)

    def body(t_obs, input_):
        bar_lambda_tm1, log_weights_tm1, log_alpha_tm1, loglikelihood = input_

        # Resampling with correction
        log_alpha_tm1_corrected = tf.where(
            tf.math.is_nan(log_alpha_tm1),
            -500*tf.ones(tf.shape(log_alpha_tm1)),
            log_alpha_tm1
        )
        alpha_tm1_unorm = tf.exp(
            log_alpha_tm1_corrected - tf.reduce_max(log_alpha_tm1_corrected, axis=0, keepdims=True)
        )
        alpha_tm1 = alpha_tm1_unorm / tf.reduce_sum(alpha_tm1_unorm, axis=0)

        indeces = tfp.distributions.Categorical(probs=tf.transpose(alpha_tm1)).sample(n_particles)
        res_bar_lambda_tm1 = tf.transpose(
            tf.gather(
                tf.transpose(bar_lambda_tm1, [1, 0, 2]),
                tf.transpose(indeces),
                axis=1,
                batch_dims=1
            ),
            [1, 0, 2]
        )
        res_log_weights_tm1 = tf.transpose(
            tf.gather(
                tf.transpose(log_weights_tm1, [1, 0]),
                tf.transpose(indeces),
                axis=1,
                batch_dims=1
            ),
            [1, 0]
        )
        res_log_alpha_tm1 = tf.transpose(
            tf.gather(
                tf.transpose(log_alpha_tm1, [1, 0]),
                tf.transpose(indeces),
                axis=1,
                batch_dims=1
            ),
            [1, 0]
        )
        res_log_weights_tm1 = res_log_weights_tm1 - res_log_alpha_tm1

        # t
        pop_index = tf.cast(t_obs / 26, dtype=tf.int64)
        pop_t = UKpop[:, pop_index]

        birth_index = tf.cast(t_obs / 26, dtype=tf.int64)
        UKbirths_t = UKbirths[:, birth_index : (birth_index + 1)]

        # 依旧外面采 xi_t，但实际 Euler 步中用 xi_sub
        xi_t = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs, :]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:, t_obs : (t_obs + 1)]

        log_likelihood_t_tm1, bar_Lambda_t = PAL_body_run_res_low(
            res_bar_lambda_tm1,
            intermediate_steps,
            UKmeasles_t,
            UKbirths_t,
            pop_t,
            beta_bar,
            p,
            a,
            is_school_term_array_t,
            is_start_school_year_array_t_obs,
            h,
            rho,
            gamma,
            xi_t,  # 形参保留
            Q,
            c,
            n_cities,
            n_particles,
            delta_year,
            v,
            Xi     # 在inner step中真正用
        )

        alpha_t = (
            c*UKbirths_t*is_start_school_year_array_t_obs[-1]
            + ((1 - c)/(26*intermediate_steps))*UKbirths_t*(1 - is_start_school_year_array_t_obs[-1])
        )
        alpha_t = tf.expand_dims(alpha_t, axis=0)
        alpha_t = tf.concat(
            (alpha_t, tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t)), tf.zeros(tf.shape(alpha_t))),
            axis=-1
        )
        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis=2) + alpha_t

        # t+1
        t_obs = t_obs + 1
        pop_index = tf.cast(t_obs / 26, dtype=tf.int64)
        pop_t = UKpop[:, pop_index]
        birth_index = tf.cast(t_obs / 26, dtype=tf.int64)
        UKbirths_t = UKbirths[:, birth_index : (birth_index + 1)]
        xi_tp1 = Xi.sample((n_particles, n_cities, 1))

        is_school_term_array_t = is_school_term_array[t_obs, :]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]
        UKmeasles_t = UKmeasles[:, t_obs : (t_obs + 1)]

        log_alpha_t, _ = PAL_body_run_res_low(
            bar_lambda_t,
            intermediate_steps,
            UKmeasles_t,
            UKbirths_t,
            pop_t,
            beta_bar,
            p,
            a,
            is_school_term_array_t,
            is_start_school_year_array_t_obs,
            h,
            rho,
            gamma,
            xi_tp1,
            Q,
            c,
            n_cities,
            n_particles,
            delta_year,
            v,
            Xi
        )

        weights_flow = tf.math.exp(
            res_log_weights_tm1 - tf.reduce_max(res_log_weights_tm1, axis=0, keepdims=True)
        )
        log_weights_tm1 = tf.math.log(weights_flow / tf.reduce_sum(weights_flow, axis=0, keepdims=True))

        log_weights_t = log_weights_tm1 + log_likelihood_t_tm1
        log_alpha_t = log_alpha_t + log_weights_t

        likelihood_t_tm1_norm = tf.exp(
            (log_weights_t) - tf.reduce_max((log_weights_t), axis=0, keepdims=True)
        )
        log_increment = tf.reduce_sum(
            tf.math.log(tf.reduce_sum(likelihood_t_tm1_norm, axis=0))
            + tf.reduce_max((log_weights_t), axis=0)
        )

        return t_obs, (bar_lambda_t, log_weights_t, log_alpha_t, loglikelihood + log_increment)

    # 初始化
    bar_lambda_0 = tf.expand_dims(tf.expand_dims(initial_pop, axis=1)*pi_0, axis=0) * tf.ones(
        (n_particles, n_cities, 4)
    )
    log_weights_0 = tf.zeros((n_particles, n_cities), dtype=tf.float32)
    log_alpha_0 = log_weights_0
    loglikelihood_0 = tf.zeros((1,), dtype=tf.float32)

    time, output = tf.while_loop(
        cond,
        body,
        loop_vars=[0, (bar_lambda_0, log_weights_0, log_alpha_0, loglikelihood_0)]
    )

    bar_lambda_tm1, log_weights_tm1, log_alpha_tm1, loglikelihood = output
    t_obs = time

    # 收尾
    log_alpha_tm1_corrected = tf.where(
        tf.math.is_nan(log_alpha_tm1),
        -500. * tf.ones(tf.shape(log_alpha_tm1)),
        log_alpha_tm1
    )
    alpha_tm1_unorm = tf.exp(
        log_alpha_tm1_corrected - tf.reduce_max(log_alpha_tm1_corrected, axis=0, keepdims=True)
    )
    alpha_tm1 = alpha_tm1_unorm / tf.reduce_sum(alpha_tm1_unorm, axis=0)

    indeces = tfp.distributions.Categorical(probs=tf.transpose(alpha_tm1)).sample(n_particles)
    res_bar_lambda_tm1 = tf.transpose(
        tf.gather(
            tf.transpose(bar_lambda_tm1, [1, 0, 2]),
            tf.transpose(indeces),
            axis=1,
            batch_dims=1
        ),
        [1, 0, 2]
    )
    res_log_weights_tm1 = tf.transpose(
        tf.gather(
            tf.transpose(log_weights_tm1, [1, 0]),
            tf.transpose(indeces),
            axis=1,
            batch_dims=1
        ),
        [1, 0]
    )
    res_log_alpha_tm1 = tf.transpose(
        tf.gather(
            tf.transpose(log_alpha_tm1, [1, 0]),
            tf.transpose(indeces),
            axis=1,
            batch_dims=1
        ),
        [1, 0]
    )
    res_log_weights_tm1 = res_log_weights_tm1 - res_log_alpha_tm1

    pop_index = tf.cast(t_obs / 26, dtype=tf.int64)
    pop_t = UKpop[:, pop_index]
    birth_index = tf.cast(t_obs / 26, dtype=tf.int64)
    UKbirths_t = UKbirths[:, birth_index:(birth_index + 1)]
    xi_t = Xi.sample((n_particles, n_cities, 1))

    is_school_term_array_t = is_school_term_array[t_obs, :]
    is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]
    UKmeasles_t = UKmeasles[:, t_obs:(t_obs + 1)]

    log_likelihood_t_tm1, _ = PAL_body_run_res_low(
        res_bar_lambda_tm1,
        intermediate_steps,
        UKmeasles_t,
        UKbirths_t,
        pop_t,
        beta_bar,
        p,
        a,
        is_school_term_array_t,
        is_start_school_year_array_t_obs,
        h,
        rho,
        gamma,
        xi_t,
        Q,
        c,
        n_cities,
        n_particles,
        delta_year,
        v,
        Xi
    )

    weights_flow = tf.math.exp(
        res_log_weights_tm1 - tf.reduce_max(res_log_weights_tm1, axis=0, keepdims=True)
    )
    log_weights_tm1 = tf.math.log(weights_flow / tf.reduce_sum(weights_flow, axis=0, keepdims=True))

    log_weights_t = log_weights_tm1 + log_likelihood_t_tm1
    likelihood_t_tm1_norm = tf.exp(
        (log_weights_t) - tf.reduce_max((log_weights_t), axis=0, keepdims=True)
    )
    log_increment = tf.reduce_sum(
        tf.math.log(tf.reduce_sum(likelihood_t_tm1_norm, axis=0))
        + tf.reduce_max((log_weights_t), axis=0)
    )

    return loglikelihood + log_increment

def PAL_body_run_res(
    bar_lambda_tm1,
    intermediate_steps,
    UKmeasles_t,
    UKbirths_t,
    pop_t,
    beta_bar,
    p,
    a,
    # -- here is where the main fix is:
    # we want 'is_school_term_array_t' to be shape [intermediate_steps],
    # not just a scalar from [t_obs,0]
    is_school_term_array_t,
    is_start_school_year_array_t_obs,
    h,
    rho,
    gamma,
    xi_t,  # not used in substeps but we keep for compatibility
    Q,
    c,
    n_cities,
    n_particles,
    delta_year,
    v,
    Xi
):
    """
    Single iteration from time t_{obs} to t_{obs}+1,
    with 'intermediate_steps' Euler sub‐steps inside.
    """
    # run the sub‐step scan
    lambda_seq, Lambda_seq = PAL_scan_intermediate(
        bar_lambda_tm1,
        is_start_school_year_array_t_obs,
        intermediate_steps,
        UKbirths_t,
        c,
        n_cities,
        n_particles,
        delta_year,
        is_school_term_array_t,
        v,
        pop_t,
        xi_t,
        h,
        rho,
        gamma,
        beta_bar,
        p,
        a,
        Xi
    )

    # final Lambda from the last sub‐step
    Lambda_final = Lambda_seq[-1, ...]  # shape: [n_particles, n_cities, 4, 4]

    # sum of infection transitions over substeps, for Q / data likelihood
    f_xi = tf.reduce_sum(Lambda_seq, axis=0)[:, :, 2, 3]  # shape: [n_particles, n_cities]

    # define parameter b
    b = -Q.parameters["scale"]**2 * f_xi + Q.parameters["loc"]

    # mu_r
    meas_t = tf.transpose(UKmeasles_t)  # shape => [1, n_cities]; will broadcast
    mu_r = (b + tf.sqrt(b**2 + 4*Q.parameters["scale"]**2 * meas_t)) / 2

    # guard against zero
    mu_r_norm = mu_r + tf.cast(mu_r == 0, tf.float32)

    # sigma_r
    sigma_r = tf.sqrt(
        1.0 / (
            (meas_t / (mu_r_norm * mu_r_norm))
            + 1.0 / (Q.parameters["scale"]**2)
        )
    )

    # sample q_t from TruncatedNormal( mu_r, sigma_r, 0, 1 )
    q_dist = tfp.distributions.TruncatedNormal(loc=mu_r, scale=sigma_r, low=0., high=1.)
    q_t = q_dist.sample()  # shape => [n_particles, n_cities]

    # expand dims for combining
    q_t_expanded = q_t[..., None, None]  # shape => [n_particles, n_cities, 1, 1]

    # build Q_t
    Q_t_r3 = tf.concat(
        [
            tf.zeros_like(q_t_expanded),
            tf.zeros_like(q_t_expanded),
            tf.zeros_like(q_t_expanded),
            q_t_expanded,
        ],
        axis=-1,
    )
    Q_t = tf.concat(
        [
            tf.zeros_like(Q_t_r3),
            tf.zeros_like(Q_t_r3),
            Q_t_r3,
            tf.zeros_like(Q_t_r3),
        ],
        axis=-2,
    )

    # build Y_t
    y_t = UKmeasles_t[..., None]  # shape => [n_cities, 1]
    y_t = tf.expand_dims(y_t, axis=0)  # shape => [1, n_cities, 1]
    Y_t_r3 = tf.concat(
        [tf.zeros_like(y_t), tf.zeros_like(y_t), tf.zeros_like(y_t), y_t],
        axis=-1
    )
    Y_t = tf.concat(
        [
            tf.zeros_like(Y_t_r3),
            tf.zeros_like(Y_t_r3),
            Y_t_r3,
            tf.zeros_like(Y_t_r3),
        ],
        axis=-2,
    )
    # broadcast Y_t to [n_particles, n_cities, 4, 4]
    Y_t = tf.tile(Y_t, [n_particles, 1, 1, 1])

    # M = sum of Q_t * Lambda over substeps
    M = tf.reduce_sum(Q_t[None, ...] * Lambda_seq, axis=0)

    # now form bar_Lambda_t
    # (1 - Q_t) * lastLambda +  Y_t * lastLambda * Q_t / M
    # but if Y_t*... == 0, just 0, else the ratio
    # shape => [n_particles, n_cities, 4, 4]
    numerator = Y_t * Lambda_final * Q_t
    bar_Lambda_t = (
        (1.0 - Q_t) * Lambda_final
        + tf.where(
            numerator == 0,
            numerator,
            numerator / M
        )
    )

    # log p(data), for partial: Poisson part, + Q prior part, – truncatedNormal
    log_pois = tfp.distributions.Poisson(M[..., 2, 3]).log_prob(meas_t)
    log_q_prior = Q.log_prob(q_t)  # shape => [n_particles, n_cities]
    log_q_trunc = q_dist.log_prob(q_t)  # shape => [n_particles, n_cities]

    likelihood_t_tm1 = log_pois + log_q_prior - log_q_trunc

    # return:
    #   new bar_lambda (summed over compartments),
    #   the log-likelihood increments,
    #   the 4D bar_Lambda_t, M, q_t
    return (
        tf.reduce_sum(bar_Lambda_t, axis=2),  # shape => [n_particles, n_cities, 4]
        likelihood_t_tm1,                    # shape => [n_particles, n_cities]
        bar_Lambda_t,                        # shape => [n_particles, n_cities, 4, 4]
        M,                                   # shape => [n_particles, n_cities, 4, 4]
        q_t
    )

###############################################################################
# The (corrected) PAL_run_likelihood_res function
###############################################################################
def PAL_run_likelihood_res(
    T,
    intermediate_steps,
    UKmeasles,
    UKbirths,
    UKpop,
    g,
    measles_distance_matrix,
    initial_pop,
    pi_0,
    beta_bar,
    p,
    a,
    is_school_term_array,
    is_start_school_year_array,
    h,
    rho,
    gamma,
    Xi,  # gamma dist for xi
    Q,
    c,
    n_cities,
    n_particles,
    delta_year
):
    """
    Outer loop over T observations, with resampling each time,
    using PAL_body_run_res with Euler sub-steps in between.
    """
    v = g * measles_distance_matrix

    def cond(t_obs, *_):
        return t_obs < T

    def body(t_obs, state):
        bar_lambda_tm1, _, log_likelihood, _ = state

        # pick correct indices
        pop_index = tf.cast(t_obs // 26, tf.int64)
        pop_t = UKpop[:, pop_index]  # shape => [n_cities]

        birth_index = tf.cast(t_obs // 26, tf.int64)
        UKbirths_t = UKbirths[:, birth_index : birth_index + 1]  # shape => [n_cities, 1]

        # sample xi_t (not used inside sub‐steps, but needed for signature)
        xi_t = Xi.sample((n_particles, n_cities, 1))

        # Here is the critical fix: pass the entire row for t_obs
        # is_school_term_array[t_obs, :] => shape = (intermediate_steps,)
        # is_start_school_year_array[t_obs, :] => shape = (intermediate_steps,)
        is_school_term_array_t = is_school_term_array[t_obs, :]
        is_start_school_year_array_t_obs = is_start_school_year_array[t_obs, :]

        UKmeasles_t = UKmeasles[:, t_obs : t_obs + 1]  # [n_cities, 1]

        (
            bar_lambda_t,
            loglikelihood_t_tm1,
            bar_Lambda_t,
            M,
            _
        ) = PAL_body_run_res(
            bar_lambda_tm1,
            intermediate_steps,
            UKmeasles_t,
            UKbirths_t,
            pop_t,
            beta_bar,
            p,
            a,
            is_school_term_array_t,
            is_start_school_year_array_t_obs,
            h,
            rho,
            gamma,
            xi_t,
            Q,
            c,
            n_cities,
            n_particles,
            delta_year,
            v,
            Xi
        )

        # births for next iteration's initial
        alpha_t = (
            c * UKbirths_t * is_start_school_year_array_t_obs[-1]
            + ((1 - c) / (26 * intermediate_steps)) * UKbirths_t * (1 - is_start_school_year_array_t_obs[-1])
        )
        alpha_t = tf.concat(
            [alpha_t, tf.zeros_like(alpha_t), tf.zeros_like(alpha_t), tf.zeros_like(alpha_t)],
            axis=-1
        )  # shape => [n_cities, 4]
        alpha_t = alpha_t[None, ...]  # => [1, n_cities, 4], then broadcast

        # new bar_lambda with births added
        bar_lambda_t = tf.reduce_sum(bar_Lambda_t, axis=2) + alpha_t

        # do 'weights' for resampling
        # shape => [n_particles, n_cities]
        loglikelihood_t_tm1 = tf.where(
            tf.math.is_nan(loglikelihood_t_tm1),
            -500 * tf.ones_like(loglikelihood_t_tm1),
            loglikelihood_t_tm1
        )

        # normalized weights across n_particles for each city?
        # Typically we either combine across cities or not.  The code below
        # does a per-city sampling.  If that is your design, it is OK.
        # shape => [n_particles, n_cities]
        ll_centered = loglikelihood_t_tm1 - tf.reduce_max(loglikelihood_t_tm1, axis=0, keepdims=True)
        likelihood_t_tm1_flow = tf.exp(ll_centered)
        norm_weights = likelihood_t_tm1_flow / tf.reduce_sum(likelihood_t_tm1_flow, axis=0, keepdims=True)

        # resampling index
        indices = tfp.distributions.Categorical(probs=tf.transpose(norm_weights)).sample(n_particles)
        # gather
        res_bar_lambda_t = tf.transpose(
            tf.gather(
                tf.transpose(bar_lambda_t, [1, 0, 2]),
                tf.transpose(indices),
                axis=1,
                batch_dims=1
            ),
            [1, 0, 2]
        )
        res_bar_Lambda_t = tf.transpose(
            tf.gather(
                tf.transpose(bar_Lambda_t, [1, 0, 2, 3]),
                tf.transpose(indices),
                axis=1,
                batch_dims=1
            ),
            [1, 0, 2, 3]
        )

        # for the increment in the total log-likelihood:
        # we do effectively log(average(likelihood_t_tm1_flow)) + max
        # shape => [n_cities]
        max_per_city = tf.reduce_max(loglikelihood_t_tm1, axis=0)
        mean_flow_per_city = tf.reduce_mean(tf.exp(loglikelihood_t_tm1 - max_per_city), axis=0)
        log_increment_per_city = tf.math.log(mean_flow_per_city) + max_per_city
        log_increment = tf.reduce_sum(log_increment_per_city)

        return (t_obs + 1, (res_bar_lambda_t, res_bar_Lambda_t, log_likelihood + log_increment, M))

    # initialization
    bar_lambda_0 = (
        tf.expand_dims(tf.expand_dims(initial_pop, axis=1) * pi_0, axis=0)
        * tf.ones([n_particles, n_cities, 4], dtype=tf.float32)
    )
    bar_Lambda_0 = tf.zeros([n_particles, n_cities, 4, 4], dtype=tf.float32)
    likelihood_0 = tf.zeros([1], dtype=tf.float32)
    M_0 = tf.zeros([n_particles, n_cities, 4, 4], dtype=tf.float32)

    # run the while_loop
    time, output = tf.while_loop(
        cond,
        body,
        loop_vars=[0, (bar_lambda_0, bar_Lambda_0, likelihood_0, M_0)]
    )

    # final (log‐)likelihood is output[2]
    return output[2]
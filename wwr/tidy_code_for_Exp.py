# ======================================================================
#  PAL‑measles experiment 
#  author : hky   2025‑07‑03
# ======================================================================
from __future__ import annotations
from pathlib import Path                   # ← New
import os, sys, time, random, numpy as np, pandas as pd, tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, Any    
# ……………………………………………………………………………………………………………………………

ROOT = Path("wwr").resolve()      
BASE = str(ROOT)
DATA = os.path.join(BASE, "Data")
PARAM = os.path.join(DATA, "Parameter")

# Shared data (to avoid repeated disk IO)
UKbirths_arr        = np.load(os.path.join(DATA, "UKbirths_array.npy"))
UKpop_arr           = np.load(os.path.join(DATA, "UKpop_array.npy"))
distance_arr        = np.load(os.path.join(DATA, "measles_distance_matrix_array.npy"))

sys.path.append(os.path.join(BASE, "Scripts"))
from measles_simulator import *
from measles_PALSMC import *

# ────────────────────────── Utility functions ──────────────────────────────────
def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def logmeanexp(x: np.ndarray, jackknife: bool = True) -> tuple[float, float]:
    n, x_max = x.size, x.max()
    lme = x_max + np.log(np.mean(np.exp(x - x_max)))
    if not jackknife:
        return lme, np.nan
    jk = np.array([
        np.delete(x, k).max() +
        np.log(np.mean(np.exp(np.delete(x, k) - np.delete(x, k).max())))
        for k in range(n)
    ])
    se = (n - 1) * jk.std(ddof=1) / np.sqrt(n)
    return lme, se

# ────────────────────────── Core runner ────────────────────────────────
def run_experiment(cfg: Dict) -> None:
    """Single experiment: cache → data preparation → Monte‑Carlo → summary print"""
    cache_path = os.path.join(cfg["cache_dir"], cfg["cache_file"])
    os.makedirs(cfg["cache_dir"], exist_ok=True)

    # ---------- 1. Load / compute ----------
    if os.path.exists(cache_path):
        log_like = np.load(cache_path)[cfg["cache_key"]]
    else:
        # ---------- 2. Data loading ----------
        if cfg["n_cities"] == 1:     # Single city (London)
            births  = tf.convert_to_tensor(UKbirths_arr[18:19], tf.float32)
            pop     = tf.convert_to_tensor(UKpop_arr[18:19],  tf.float32)
            dist    = tf.convert_to_tensor(distance_arr[18:19, 18:19], tf.float32)

            measles = tf.convert_to_tensor(
                pd.read_csv(os.path.join(DATA, "londonsim.csv")).values,
                tf.float32
            )
        else:                        # 40 cities
            births = tf.convert_to_tensor(UKbirths_arr, tf.float32)
            pop    = tf.convert_to_tensor(UKpop_arr,  tf.float32)
            dist   = tf.convert_to_tensor(distance_arr, tf.float32)

            # Compatible with both CSV and NPY formats
            dfile = os.path.join(DATA, cfg["data_file"])
            measles = (
                tf.convert_to_tensor(pd.read_csv(dfile).values, tf.float32)
                if dfile.endswith(".csv") else
                tf.convert_to_tensor(np.load(dfile), tf.float32)
            )

        # Academic year calendar
        term   = tf.constant([6, 99, 115, 198, 252, 299, 308, 355, 366], tf.float32)
        school = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0], tf.float32)
        T_int  = int(measles.shape[1])
        inter_steps = 4
        h = tf.constant(14 / inter_steps, tf.float32)
        is_term, is_year, *_ = school_term_and_school_year(T_int, inter_steps, term, school)
        is_term = tf.convert_to_tensor(is_term, tf.float32)
        is_year = tf.convert_to_tensor(is_year, tf.float32)

        # ---------- 3. Load best parameters (.npz or .npy) ----------
        pfile = os.path.join(cfg["param_dir"], cfg["param_file"])
        raw   = np.load(pfile, allow_pickle=False)
        best_par = (
            raw[cfg["param_key"]].astype(np.float32) if cfg["param_key"] else
            raw.astype(np.float32)
        )

        # ---------- 4. Assemble fixed tensors ----------
        n_cities = cfg["n_cities"]
        pi_0 = tf.constant(
            [[best_par[0], best_par[1], best_par[2],
              1.0 - best_par[0] - best_par[1] - best_par[2]]],
            tf.float32
        ) * tf.ones((n_cities, 4), tf.float32)

        beta_bar = tf.ones((n_cities, 1), tf.float32) * best_par[3]
        rho      = tf.ones((n_cities, 1), tf.float32) * best_par[4]
        gamma    = tf.ones((n_cities, 1), tf.float32) * best_par[5]
        g_scalar = cfg["g_factory"](best_par)
        g        = tf.ones((n_cities, 1), tf.float32) * g_scalar

        a       = tf.constant(best_par[cfg["idx_a"]], tf.float32)
        c       = tf.constant(best_par[cfg["idx_c"]], tf.float32)
        xi_var  = 10.0 * tf.constant(best_par[cfg["idx_xi"]], tf.float32)
        q_var   = tf.constant(best_par[cfg["idx_q"]], tf.float32)

        Xi = tfp.distributions.Gamma(concentration=xi_var, rate=xi_var)
        Q  = tfp.distributions.TruncatedNormal(
                 loc=cfg["q_loc_factory"](best_par),
                 scale=q_var,
                 low=0.0,
                 high=1.0,
             )

        p_const       = tf.constant(0.759, tf.float32)
        delta_year    = tf.constant([[1 / 50]], tf.float32) * tf.ones((n_cities, 4), tf.float32)

        # ---------- 5. PAL_vanilla / PAL_lookahead ----------
        log_like = np.empty(cfg["n_experiments"], dtype=np.float32)
        fn_call  = PAL_run_likelihood_res if cfg["mode"] == "res" else PAL_run_likelihood_lookahead

        for i in range(cfg["n_experiments"]):
            set_seeds(cfg["seed_offset"] + i)
            log_like[i] = fn_call(
                T_int, inter_steps, measles, births, pop, g, dist, pop[:, 0],
                pi_0, beta_bar, p_const, a, is_term, is_year, h,
                rho, gamma, Xi, Q, c, n_cities, cfg["n_particles"], delta_year
            )[0].numpy()

        # Cache
        np.savez(cache_path, **{cfg["cache_key"]: log_like})

    # ---------- 6. Summary ----------
    lme, se = logmeanexp(log_like)
    RESULTS[cfg["label"]] = (float(lme), float(se))
    print(f"[{cfg['label']}]  LME={lme:.4f}  SE={se:.4f}")
    return float(lme), float(se)

q_mean_path = os.path.join(DATA, "q_mean.npy")      
Q_MEAN: float = float(np.mean(np.load(q_mean_path)))  

# ────────────────────────── Experiment configurations ────────────────────────────────
EXPERIMENTS = [
    # label  n_cities  mode   n_particles  seed_offset
    # g_factory                    q_loc_factory
    # cache dir / file / key
    # param dir / file / key
    # data_file
    # a‑idx c‑idx xi‑idx q‑idx
    dict(
        label="E2",  n_cities=1,  mode="res",       n_particles=5_000,   seed_offset=113,
        g_factory=lambda bp: 0.0,
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E2"),
        cache_file="PAL_vanilla_new.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E2"),
        param_file="E2_param_exp.npz",
        param_key="E2_param_exp",
        data_file="londonsim.csv",
        n_experiments=20,
        idx_a=6, idx_c=7, idx_xi=8, idx_q=9,
    ),
    dict(
        label="E3",  n_cities=1,  mode="res",       n_particles=100_000, seed_offset=123,
        g_factory=lambda bp: 0.0,
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E3"),
        cache_file="PAL_vanilla.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E2"),
        param_file="E2_param_exp.npz",
        param_key="E2_param_exp",
        data_file="londonsim.csv",
        n_experiments=20,
        idx_a=6, idx_c=7, idx_xi=8, idx_q=9,
    ),
    dict(
        label="E4",  n_cities=1,  mode="lookahead", n_particles=5_000,   seed_offset=123,
        g_factory=lambda bp: 0.0,
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E4"),
        cache_file="PAL_lookahead.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E4"),
        param_file="E4_param_exp.npz",
        param_key="E4_param_exp",
        data_file="londonsim.csv",
        n_experiments=20,
        idx_a=6, idx_c=7, idx_xi=8, idx_q=9,
    ),
    dict(
        label="E5",  n_cities=1,  mode="lookahead", n_particles=100_000, seed_offset=123,
        g_factory=lambda bp: 0.0,
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E5"),
        cache_file="PAL_lookahead.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E4"),
        param_file="E4_param_exp.npz",
        param_key="E4_param_exp",
        data_file="londonsim.csv",
        n_experiments=20,
        idx_a=6, idx_c=7, idx_xi=8, idx_q=9,
    ),
    dict(
        label="E6",  n_cities=1,  mode="lookahead", n_particles=5_000,   seed_offset=123,
        g_factory=lambda bp: 0.0,
        q_loc_factory=lambda bp: float(bp[10]),
        cache_dir=os.path.join(BASE, "E6"),
        cache_file="PAL_lookahead.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E6"),
        param_file="E6_param_exp.npz",
        param_key="E6_param_exp",
        data_file="UKmeasles_array.npy",
        n_experiments=20,
        idx_a=6, idx_c=7, idx_xi=8, idx_q=9,
    ),
    # ─────────────── 40‑city series ───────────────────────────────────
    dict(
        label="E9",  n_cities=40, mode="res",       n_particles=5_000,   seed_offset=123,
        g_factory=lambda bp: 100.0 * bp[6],         
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E9"),
        cache_file="PAL_res_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E9"),
        param_file="E9_param_exp.npz",
        param_key="E9_param_exp",
        data_file="M40.csv",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,   # ≈E9 shifted by 1
    ),
    dict(
        label="E10", n_cities=40, mode="lookahead", n_particles=5_000,   seed_offset=100,
        g_factory=lambda bp: 100.0 * bp[6],
        q_loc_factory=lambda bp: 0.7,
        cache_dir=os.path.join(BASE, "E10"),
        cache_file="PAL_lookahead_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E10"),
        param_file="E10_param_exp.npz",
        param_key="E10_param_exp",
        data_file="M40.csv",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,
    ),
    dict(
        label="E12", n_cities=40, mode="res",       n_particles=5_000,   seed_offset=123,
        g_factory=lambda bp: 100.0 * bp[6],
        q_loc_factory=lambda bp: Q_MEAN,
        cache_dir=os.path.join(BASE, "E12"),
        cache_file="PAL_res_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E12"),
        param_file="E12_param_exp.npz",
        param_key="E12_param_exp",
        data_file="UKmeasles_array.npy",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,
    ),
    dict(
        label="E13", n_cities=40, mode="res",       n_particles=100_000, seed_offset=123,
        g_factory=lambda bp: 100.0 * bp[6],
        q_loc_factory=lambda bp: Q_MEAN,
        cache_dir=os.path.join(BASE, "E13"),
        cache_file="PAL_res_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=os.path.join(BASE, "E12"),     # Same parameters as E12
        param_file="E12_param_exp.npz",
        param_key="E12_param_exp",
        data_file="UKmeasles_array.npy",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,
    ),
    dict(
        label="E14", n_cities=40, mode="lookahead", n_particles=5_000,   seed_offset=123,
        g_factory=lambda bp: 100.0 * bp[6],
        q_loc_factory=lambda bp: Q_MEAN,
        cache_dir=os.path.join(BASE, "E14"),
        cache_file="PAL_lookahead_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=PARAM,                          # final_parameters_lookahead_A.npy
        param_file="final_parameters_lookahead_A.npy",
        param_key=None,
        data_file="UKmeasles_array.npy",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,
    ),
    dict(
        label="E15", n_cities=40, mode="lookahead", n_particles=100_000, seed_offset=123,
        g_factory=lambda bp: 100.0 * bp[6],
        q_loc_factory=lambda bp: Q_MEAN,
        cache_dir=os.path.join(BASE, "E15"),
        cache_file="PAL_lookahead_40.npz",
        cache_key="log_likelihood_shared",
        param_dir=PARAM,
        param_file="final_parameters_lookahead_A.npy",
        param_key=None,
        data_file="UKmeasles_array.npy",
        n_experiments=20,
        idx_a=7, idx_c=8, idx_xi=9, idx_q=10,
    ),
]

RESULTS: dict[str, tuple[float, float]] = {}

if __name__ == "__main__":
    for cfg in EXPERIMENTS:
        run_experiment(cfg)          

    print("\n────────── All experiments ──────────")
    print(f"{'Experiment':<10} {'LME_est':>12} {'SE':>10}")
    for lbl in sorted(RESULTS, key=lambda x: int(x[1:])):    
        est, se = RESULTS[lbl]
        print(f"{lbl:<10} {est:>12.4f} {se:>10.4f}")

    globals().update({f"{lbl}_est": RESULTS[lbl][0] for lbl in RESULTS})
    globals().update({f"{lbl}_se":  RESULTS[lbl][1] for lbl in RESULTS})
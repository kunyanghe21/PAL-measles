import os
os.environ["PYTHONHASHSEED"] = "12345"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import sys
sys.path.append('Scripts/')
from measles_simulator_KH import *

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

UKbirths_array = np.load("Data/UKbirths_array.npy")
UKpop_array = np.load("Data/UKpop_array.npy")
measles_distance_matrix_array = np.load("Data/measles_distance_matrix_array.npy")
UKmeasles_array = np.load("Data/UKmeasles_array.npy")
modelA_array = np.load("Data/Parameter/final_parameters_lookahead_A.npy")

UKbirths = tf.convert_to_tensor(UKbirths_array, dtype = tf.float32)
UKpop = tf.convert_to_tensor(UKpop_array, dtype = tf.float32)
measles_distance_matrix = tf.convert_to_tensor(measles_distance_matrix_array, dtype = tf.float32)
UKmeasles = tf.convert_to_tensor(UKmeasles_array, dtype = tf.float32)

df = pd.read_csv("Data/londonbirth.csv")
data_array = df.values
UKbirths = tf.convert_to_tensor(data_array, dtype=tf.float32)

df1 = pd.read_csv("Data/londonpop.csv")
data_array1 = df1.values
UKpop = tf.convert_to_tensor(data_array1, dtype=tf.float32)

term   = tf.convert_to_tensor([6, 99, 115, 198, 252, 299, 308, 355, 366], dtype = tf.float32)
school = tf.convert_to_tensor([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype = tf.float32)

n_cities = tf.constant(40, dtype = tf.int64)

initial_pop = UKpop[:,0]

T = UKmeasles.shape[1]
intermediate_steps = 4
h = tf.constant(14/tf.cast(intermediate_steps, dtype = tf.float32), dtype = tf.float32)
is_school_term_array, is_start_school_year_array, times_total, times_obs = school_term_and_school_year(T, intermediate_steps, term, school)

is_school_term_array = tf.convert_to_tensor(is_school_term_array, dtype = tf.float32)
is_start_school_year_array = tf.convert_to_tensor(is_start_school_year_array, dtype = tf.float32)

pi_0_1 = 0.015
pi_0_2 = 0.0002
pi_0_3 = 0.0002
pi_0 = tf.convert_to_tensor([[pi_0_1, pi_0_2, pi_0_3, 1 - pi_0_1 - pi_0_2 - pi_0_3]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

initial_pop = UKpop[:,0]

beta_bar  = tf.convert_to_tensor( [[10]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)
rho   = tf.convert_to_tensor([[0.1]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)
gamma = tf.convert_to_tensor([[0.1]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)

g = tf.convert_to_tensor([[0]], dtype = tf.float32)*tf.ones((n_cities, 1), dtype = tf.float32)
p = tf.constant(0.759, dtype = tf.float32)
a = tf.constant(0.3,   dtype = tf.float32)
c = tf.constant(0.1,   dtype = tf.float32)

Xi = tfp.distributions.Gamma(concentration = 2, rate = 2)
Q  = tfp.distributions.TruncatedNormal( 0.7, 0.2, 0, 1)

delta_year = tf.convert_to_tensor([[1/50]], dtype = tf.float32)*tf.ones((n_cities, 4), dtype = tf.float32)

T_small = tf.constant(415, dtype = tf.float32)

# Initialize result lists
means = np.zeros((40, 25))
variances = np.zeros((40, 25))
medians = np.zeros((40, 25))

# Perform 100 simulations
for i in range(25):
    X_t, Y_t, Xi_t, Q_t = run(T_small, intermediate_steps, UKbirths, UKpop, g, measles_distance_matrix,
                              initial_pop, pi_0, beta_bar, p, a, is_school_term_array,
                              is_start_school_year_array, h, rho, gamma, Xi, Q, c, n_cities, delta_year)

    max_time = 415

    # Calculate the log values for each city Y_t_log
    for city in range(40):
        Y_t_log = Y_t[1:(max_time + 1), city, 0]
        # Calculate mean, variance, median
        means[city, i] = np.mean(Y_t_log)
        variances[city, i] = np.var(Y_t_log)
        medians[city, i] = np.median(Y_t_log)

# Initialize an empty list to store the results for each city
all_results = []

for city_index in range(40):
    # Create a DataFrame for the current city
    results_df_city = pd.DataFrame({
        'Simulation': np.arange(25),
        'Mean': means[city_index, :],
        'Variance': variances[city_index, :],
        'Median': medians[city_index, :]
    })

    # Add city column
    results_df_city['City'] = city_index

    # Add the current city’s results to the total list
    all_results.append(results_df_city)

# Combine results from all cities
combined_results_df = pd.concat(all_results, ignore_index=True)

# Save to a new CSV file
combined_results_df.to_csv("/Users/mac/Desktop/PAL_measles/combined_simulation_results_for_1000_times.csv", index=False)
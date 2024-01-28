"""Testing MCMC performance on simulated data from a partially observed source-measurement model"""
import numpy as np
from scripts.thesis.sample_udgs import gen_udgs_mcmc
from scripts.thesis.sample_sm_model import sample_fm_dag
from scripts.thesis.mcmc_helpers import dag_from_udg
from scripts.thesis.mcmc_algorithm import InputData
from scripts.thesis.mcmc_performance_plot_results import display_results

########################################
# Source-measurement model on 10 nodes #
########################################
n_models = 100
n_samples = 10**3
d = 10
max_moves = 10**4
burn_in = None
conv_crit = 500
significance_level = 0.05
lmbda = 1 # 1 yields BIC
seed = 1312

# Recommendation is to run this statement separately to prepare the set of UDGs (long runtime). 
# Once prepared, the statement gen_udgs_mcmc(...) will load the contents in ~/storage rather than do a new sampling
# See ~/experiments/sampling_udgs/prep_udgs.py for an example preparatory script.
buckets = gen_udgs_mcmc(d=d, init="udg", desired_ed=0.5, n_models=n_models, explore=True, rng=np.random.default_rng(0))
prng = list(buckets.keys())

stored_results = {p:{} for p in prng}

for p in prng:
    models = buckets[p]
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        udg = models[i]
        X = sample_fm_dag(udg, N=n_samples, rng=rng)
        idt = InputData(samples=X, rng=np.random.default_rng(0))
        # IT-informed initialization
        init = dag_from_udg(idt.it_udg(significance_level=significance_level))
        idt.mcmc(init=init,
                 max_moves=max_moves, 
                 burn_in=burn_in, 
                 conv_crit=conv_crit,
                 lmbda=lmbda)

        stored_results[p][i] = (idt, udg)

display_results(stored_results=stored_results,
                significance_level=significance_level,
                directory="results_d10") # Ensure that this subdirectory exists within current directory
""" Prepping the set of UDGs before running the benchmark 
    (if the UDG set is already prepared in '~/storage', the benchmark script will load it rather than re-generate it)
"""
import numpy as np
from scripts.thesis.sample_udgs import gen_udgs_mcmc

# Settings corresponding to our experiment in mcmc_performance/partially_observed_source_measurement_model
n_models = 100
d = 10
# This stores the UDGs in ~/storage
buckets = gen_udgs_mcmc(d=d, 
                        init="udg", 
                        desired_ed=0.5, 
                        n_models=n_models, 
                        explore=True, 
                        rng=np.random.default_rng(0),
                        plot=True)

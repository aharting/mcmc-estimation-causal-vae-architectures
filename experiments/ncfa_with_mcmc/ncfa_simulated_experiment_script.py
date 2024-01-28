"""NCFA on simulated randomized control trial"""

import numpy as np
from scripts.thesis.sample_udgs import gen_udgs_mcmc
from scripts.thesis.ncfa_simulated_experiment import ncfa_simulated_experiment


d = 10
p = 0.5
# First choose a UDG of non-empty UEC
# To make use of the existing UDG sample in /storage from pervious experiments, 
# choose the first UDG in the n=100-sample stored in /storage
buckets = gen_udgs_mcmc(d=d, 
                        init="udg", 
                        desired_ed=0.5, 
                        n_models=100, 
                        explore=True, 
                        rng=np.random.default_rng(0),
                        plot=False)
models = buckets[p]
udg = models[0]
n_latents_treated = 2 # Checked that the mmcm model has at least these many latents
n_samples = 10**4

#################
# ATE = 1*sigma #
#################

ate = 1 # Before changing, ensure that there is a subdirectory in current directory named "ate_<ate>sigma"
ncfa_simulated_experiment(ate=ate, udg=udg, n_latents_treated=n_latents_treated, n_samples=n_samples)

#################
# ATE = 3*sigma #
#################

ate = 3 # Before changing, ensure that there is a subdirectory in current directory named "ate_<ate>sigma"
ncfa_simulated_experiment(ate=ate, udg=udg, n_latents_treated=n_latents_treated, n_samples=n_samples)

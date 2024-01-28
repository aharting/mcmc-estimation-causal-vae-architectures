"""Testing MCMC performance on simulated data from a partially observed source-measurement model while varying hyperparameters."""
import numpy as np
import itertools
from scripts.thesis.tuning_hyperparameters_fcns import (tuning_hyperparameters,
display_results_tuning_hyperparameters)


########################
# IT-informed initialization #
########################
d = 10
prng = [p / 10 for p in range(1,10)]
initialization = "it_informed" # 'empty', 'complete' or 'it_informed'
directory = f"{initialization}_initialization" # Ensure that this directory exists

n_models = 100
max_moves = 10**3
n_samples = 10**3
burn_in = None
conv_crit = 500
significance_level = 0.05

alpha_base = 10
alpha_exps = [-np.inf] + list(np.linspace(start=-2, stop=2, num=5))
alpha_range = [alpha_base**x for x in alpha_exps]

lmbda_base = 3
lmbda_exps = list(np.linspace(start=-1, stop=3, num=5))
lmbda_range = [lmbda_base**x for x in lmbda_exps]
# Generate all combinations of parameters
param_grid = {"lmbda":lmbda_range,
              "alpha_stay":alpha_range,
              "alpha_exit":[1]}
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
# Add alpha_exit=0
param_grid = {"lmbda":lmbda_range,
              "alpha_stay":[1],
              "alpha_exit":[0]}
all_params.extend(
    [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())])
alpha_exps.append(np.inf)

# Run tuning
for p in prng:
    tuning_hyperparameters(d=d,
                           p=p,
                           directory=directory,
                           n_models=n_models,
                           max_moves=max_moves,
                           n_samples=n_samples,
                           burn_in=burn_in,
                           conv_crit=conv_crit,
                           significance_level=significance_level,
                           initialization=initialization,
                           all_params=all_params,
                           model_type="partially_observed_sm_model")
# Plot results
display_results_tuning_hyperparameters(prng=prng,
                                       directory=directory,
                                       alpha_base=alpha_base,
                                       alpha_exps=alpha_exps,
                                       lmbda_base=lmbda_base,
                                       lmbda_exps=lmbda_exps)

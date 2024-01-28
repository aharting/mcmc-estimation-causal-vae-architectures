"""Testing MCMC performance on simulated data from a fully observed DAG model while varying hyperparameter configuration.
Experimental set-up is similar to GrUES setting 2 (Markham, A., Liu, M., Aragam, B., & Solus, L., 2023).
"""
import numpy as np
import itertools
from scripts.thesis.tuning_hyperparameters_fcns import (tuning_hyperparameters,
display_results_tuning_hyperparameters, generalized_performance,
display_results_generalized_performance)


########################
# Empty initialization #
########################
d = 5
prng = [p / 10 for p in range(1,10)]
initialization = "empty" # 'empty', 'complete' or 'it_informed'
directory = f"{initialization}_initialization" # Ensure that this directory exists

n_models = 100
max_moves = 10**3
n_samples = 10**3
burn_in = None
conv_crit = 500
significance_level = 0.05

alpha_base = 10
alpha_exps = [-np.inf] + list(np.linspace(start=-2, stop=3, num=6))
alpha_range = [alpha_base**x for x in alpha_exps]

lmbda_base = 3
lmbda_exps = list(np.linspace(start=-2, stop=2, num=5))
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
                           model_type="observable_dag")
# Plot results
display_results_tuning_hyperparameters(prng=prng,
                                       directory=directory,
                                       alpha_base=alpha_base,
                                       alpha_exps=alpha_exps,
                                       lmbda_base=lmbda_base,
                                       lmbda_exps=lmbda_exps)

##############################
# IT-informed initialization #
##############################
n_models = 100
prng = [p / 10 for p in range(1,10)]
initialization = "it_informed" # 'empty', 'complete' or 'it_informed'
directory = f"{initialization}_initialization" # Ensure that this directory exists

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
                           model_type="observable_dag")
# Plot results
display_results_tuning_hyperparameters(prng=prng,
                                   directory=directory,
                                   alpha_base=alpha_base,
                                   alpha_exps=alpha_exps,
                                   lmbda_base=lmbda_base,
                                   lmbda_exps=lmbda_exps)

####################################################
# Performance improvement using optimal parameters #
# for five-node models on ten-node models #
####################################################

# Optimize proportion correct MAP estimates
d = 10
initialization = "it_informed"
directory = "generalizeability_optimal_parameters/optimize_bic"
directory_params = "it_informed_initialization"
optimizing_metric = "incorr_bic"

# Compare performance using retrieved params and default params
for p in prng:
    generalized_performance(d=d,
                            p=p,
                            directory=directory,
                            n_models=n_models,
                            max_moves=max_moves,
                            n_samples=n_samples,
                            burn_in=burn_in,
                            conv_crit=conv_crit,
                            significance_level=significance_level,
                            initialization=initialization,
                            directory_params=directory_params,
                            optimizing_metric=optimizing_metric)

display_results_generalized_performance(prng=prng,
                                        directory=directory,
                                        optimizing_metric=optimizing_metric)

# Optimize proportion correct MAP estimates
directory = "generalizeability_optimal_parameters/optimize_map"
optimizing_metric = "incorr_map"

# Compare performance using retrieved params and default params
for p in prng:
    generalized_performance(d=d,
                            p=p,
                            directory=directory,
                            n_models=n_models,
                            max_moves=max_moves,
                            n_samples=n_samples,
                            burn_in=burn_in,
                            conv_crit=conv_crit,
                            significance_level=significance_level,
                            initialization=initialization,
                            directory_params=directory_params,
                            optimizing_metric=optimizing_metric)

display_results_generalized_performance(prng=prng,
                                        directory=directory,
                                        optimizing_metric=optimizing_metric)

"""Testing MCMC performance on simulated data from a fully observed DAG model while varying DAG initialization mode.
Experimental set-up is similar to GrUES setting 2 (Markham, A., Liu, M., Aragam, B., & Solus, L., 2023)."""

from scripts.thesis.tuning_initialization_fcns import tuning_initialization, display_results_tuning_initialization


##########################################
# DAG model on 5 nodes (GrUES benchmark) #
##########################################

prng = [p / 10 for p in range(1,10)]
d = 5
directory = "results_d5" # Ensure that this directory exists

n_models = 100
max_moves = 10**3
n_samples = 10**3
burn_in = None
conv_crit = 500
significance_level = 0.05

# Run tuning
for p in prng:
    tuning_initialization(d=d,
                          p=p,
                          directory=directory,
                          n_models=n_models,
                          max_moves=max_moves,
                          n_samples=n_samples,
                          burn_in=burn_in,
                          conv_crit=conv_crit,
                          significance_level=significance_level,
                          model_type="observable_dag")
# Plot results
display_results_tuning_initialization(prng=prng, directory=directory)

#########################
# DAG model on 10 nodes #
#########################

prng = [p / 10 for p in range(1,10)]
d = 10
directory = "results_d10" # Ensure that this directory exists

n_models = 100
max_moves = 10**3
n_samples = 10**3
burn_in = None
conv_crit = 500
significance_level = 0.05

# Run tuning
for p in prng:
    tuning_initialization(d=d,
                          p=p,
                          directory=directory,
                          n_models=n_models,
                          max_moves=max_moves,
                          n_samples=n_samples,
                          burn_in=burn_in,
                          conv_crit=conv_crit,
                          significance_level=significance_level,
                          model_type="observable_dag")
# Plot results
display_results_tuning_initialization(prng=prng, directory=directory)

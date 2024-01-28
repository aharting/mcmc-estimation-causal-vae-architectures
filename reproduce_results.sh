#!/bin/bash

# mcmc performance benchmark on simulated data
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/observable_dag_model/benchmark/
python3 observable_dag_model_script.py
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/partially_observed_source_measurement_model/benchmark/
python3 partially_observed_source_measurement_model_script.py

# mcmc tuning on simulated data
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/observable_dag_model/tuning/initialization/
python3 tuning_initialization_script.py
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/observable_dag_model/tuning/hyperparameters/
python3 tuning_hyperparameters_script.py
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/partially_observed_source_measurement_model/tuning/initialization/
python3 tuning_initialization_script.py
cd ~/mcmc-estimation-causal-vae-architectures/experiments/mcmc_performance/partially_observed_source_measurement_model/tuning/hyperparameters/
python3 tuning_hyperparameters_script.py

# ncfa on simulated data
cd ~/mcmc-estimation-causal-vae-architectures/experiments/ncfa_with_mcmc/
python3 ncfa_simulated_experiment_script.py


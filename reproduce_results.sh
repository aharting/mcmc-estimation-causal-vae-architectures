#!/bin/bash

# Get the directory of the script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH=$SCRIPT_DIR:$PYTHONPATH

# mcmc performance benchmark on simulated data
cd "$SCRIPT_DIR/experiments/mcmc_performance/observable_dag_model/benchmark/" || exit
python3 observable_dag_model_script.py || exit

cd "$SCRIPT_DIR/experiments/mcmc_performance/partially_observed_source_measurement_model/benchmark/" || exit
python3 partially_observed_source_measurement_model_script.py || exit

# mcmc tuning on simulated data
cd "$SCRIPT_DIR/experiments/mcmc_performance/observable_dag_model/tuning/initialization/" || exit
python3 tuning_initialization_script.py || exit

cd "$SCRIPT_DIR/experiments/mcmc_performance/observable_dag_model/tuning/hyperparameters/" || exit
python3 tuning_hyperparameters_script.py || exit

cd "$SCRIPT_DIR/experiments/mcmc_performance/partially_observed_source_measurement_model/tuning/initialization/" || exit
python3 tuning_initialization_script.py || exit

cd "$SCRIPT_DIR/experiments/mcmc_performance/partially_observed_source_measurement_model/tuning/hyperparameters/" || exit
python3 tuning_hyperparameters_script.py || exit

# ncfa on simulated data
cd "$SCRIPT_DIR/experiments/ncfa_with_mcmc/" || exit
python3 ncfa_simulated_experiment_script.py || exit

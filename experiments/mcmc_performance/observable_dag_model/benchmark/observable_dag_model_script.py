"""
Testing MCMC performance on simulated data from a fully observed DAG model. 
Experimental set-up is similar to GrUES setting 2 (Markham, A., Liu, M., Aragam, B., & Solus, L., 2023).
"""
import numpy as np
from scripts.grues.sample import sample_causal_dag
from scripts.thesis.mcmc_helpers import dag_from_udg
from scripts.thesis.mcmc_algorithm import InputData
from scripts.thesis.mcmc_performance_plot_results import display_results

if False:
    ##########################################
    # DAG model on 5 nodes (GrUES benchmark) #
    ##########################################
    n_models = 100
    n_samples = 10**3
    d = 5
    prng = [p / 10 for p in range(1,10)]
    max_moves = 10**4
    burn_in = int(max_moves / 2)
    conv_crit = None
    significance_level = 0.05
    lmbda = 1 # 1 yields BIC
    seed = 1312

    stored_results = {p:{} for p in prng}

    for p in prng:
        rng = np.random.default_rng(seed)
        for i in range(n_models):
            udg, dag, X = sample_causal_dag(num_nodes=d,
                                            edge_prob=p,
                                            weight_interv=(0, 1),
                                            samp_size=n_samples,
                                            return_uec=True,
                                            rng=rng,)
            np.fill_diagonal(udg, True)
            dag = dag.astype(int)

            idt = InputData(X, rng=np.random.default_rng(0))
            init = dag_from_udg(idt.it_udg(significance_level=significance_level))
            idt.mcmc(init=init,
                     max_moves=max_moves, 
                     burn_in=burn_in, 
                     conv_crit=conv_crit,
                     lmbda=lmbda)

            stored_results[p][i] = (idt, udg)

    display_results(stored_results=stored_results,
                    significance_level=significance_level,
                    directory="results_d5") # Ensure that this subdirectory exists within current directory

    #########################
    # DAG model on 10 nodes #
    #########################
    n_models = 100
    n_samples = 10**3
    d = 10
    prng = [p / 10 for p in range(1,10)]
    max_moves = 10**4
    burn_in = None
    conv_crit = 500
    significance_level = 0.05
    lmbda = 1 # 1 yields BIC
    seed = 1312

    stored_results = {p:{} for p in prng}

    for p in prng:
        rng = np.random.default_rng(seed)
        for i in range(n_models):
            udg, dag, X = sample_causal_dag(num_nodes=d,
                                            edge_prob=p,
                                            weight_interv=(0, 1),
                                            samp_size=n_samples,
                                            return_uec=True,
                                            rng=rng,)
            np.fill_diagonal(udg, True)
            dag = dag.astype(int)

            idt = InputData(X, rng=np.random.default_rng(0))
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

#########################
# DAG model on 15 nodes #
#########################
n_models = 100
n_samples = 10**3
d = 15
prng = [p / 10 for p in range(1,10)]
max_moves = 10**4
burn_in = None
conv_crit = 500
significance_level = 0.05
lmbda = 1 # 1 yields BIC
seed = 1312

stored_results = {p:{} for p in prng}

for p in prng:
    rng = np.random.default_rng(seed)
    for i in range(n_models):
        udg, dag, X = sample_causal_dag(num_nodes=d,
                                        edge_prob=p,
                                        weight_interv=(0, 1),
                                        samp_size=n_samples,
                                        return_uec=True,
                                        rng=rng,)
        np.fill_diagonal(udg, True)
        dag = dag.astype(int)

        idt = InputData(X, rng=np.random.default_rng(0))
        init = dag_from_udg(idt.it_udg(significance_level=significance_level))
        idt.mcmc(init=init,
                 max_moves=max_moves,
                 burn_in=burn_in, 
                 conv_crit=conv_crit,
                 lmbda=lmbda)

        stored_results[p][i] = (idt, udg)

display_results(stored_results=stored_results,
                significance_level=significance_level,
                directory="results_d15") # Ensure that this subdirectory exists within current directory

"""Functions for tuning of the initialization DAG in simulated randomized control trials"""
from scripts.thesis.mcmc_helpers import sfd, dag_from_udg
from tqdm import tqdm
import numpy as np
import pandas as pd
from scripts.thesis.mcmc_algorithm import InputData
from scripts.grues.sample import sample_causal_dag
from scripts.medil.ecc_algorithms import find_clique_min_cover
from scripts.thesis.sample_udgs import gen_udgs_mcmc
from scripts.thesis.sample_sm_model import sample_fm_dag


def tuning_initialization(d=5,
                          p=0.5,
                          directory="",
                          n_models=100,
                          max_moves=10**3,
                          n_samples=10**3,
                          burn_in=None,
                          conv_crit=500,
                          significance_level=0.05,
                          model_type="observable_dag"):
    """Records performance over different choices of initialization
    Args
    ----
    model_type: str, one of ["observable_dag", "partially_observed_sm_model"]
        Sampling model
    """
    total_num_udgs = 2**(np.triu(np.ones(shape=(d,d)), k=1)).sum()
    seed = 1312
    rng = np.random.default_rng(seed)

    init_range = ["empty", "complete", "it_informed"]

    df = pd.DataFrame(columns=["init",
                               "incorr_bic",
                               "incorr_map",
                               "sfd_bic",
                               "sfd_map"])
    if model_type == "partially_observed_sm_model":
        buckets = gen_udgs_mcmc(d=d, init="udg", desired_ed=0.5, n_models=n_models, explore=True, rng=np.random.default_rng(0))
        models = buckets[p]
    
    for i in tqdm(range(n_models)):
        if model_type == "observable_dag":
            true_udg, dag, X = sample_causal_dag(num_nodes=d,
                                                 edge_prob=p,
                                                 weight_interv=(0, 1),
                                                 samp_size=n_samples,
                                                 return_uec=True,
                                                 rng=rng,)
            np.fill_diagonal(true_udg, True)
        elif model_type == "partially_observed_sm_model":
            true_udg = models[i]
            X = sample_fm_dag(true_udg, N=n_samples, rng=rng)
        else:
            raise Exception("Invalid model_type")            
        true_mmcm = find_clique_min_cover(true_udg)

        for method in init_range:
            idt = InputData(samples=X, rng=np.random.default_rng(0))
            if method == "it_informed":
                init = dag_from_udg(idt.it_udg(significance_level=significance_level))
            else:
                init = method
            idt.mcmc(init=init, 
                     max_moves=max_moves,
                     burn_in=burn_in,
                     conv_crit=conv_crit)

            est_udg = {"bic": idt.optimal_udg,
                       "map": idt.map_udg()}

            est_mmcm = {key: find_clique_min_cover(est_udg[key]) for key in ["bic", "map"]}

            record = [method]
            record += [1-(est_udg[key] == true_udg).all() for key in ["bic", "map"]]
            record += [sfd(est_mmcm[key], true_mmcm) for key in ["bic", "map"]]

            df.loc[len(df)] = record

    df_scores = df.groupby(["init"]).agg({"incorr_bic": np.mean,
                                          "incorr_map": np.mean,
                                          "sfd_bic": np.median,
                                          "sfd_map": np.median}).reset_index()
    optimal_scores = df_scores.agg(np.min).drop(index=["init"])
    optimal_params = {metric: dict(df_scores.iloc[np.argmin(df_scores[metric])][["init"]])
                      for metric in ["incorr_bic", "incorr_map", "sfd_bic", "sfd_map"]}
    pd.DataFrame(optimal_params).to_csv(f"{directory}/optimal_params_{p}.csv")
    pd.DataFrame(optimal_scores).to_csv(f"{directory}/optimal_scores_{p}.csv")
    df.to_csv(f"{directory}/all_scores_{p}.csv")
    df_scores.to_csv(f"{directory}/all_scores_aggregated_{p}.csv")
    return


def display_results_tuning_initialization(prng, directory=""):
    """Plot results from tuning_initialization()
    """
    scores = pd.DataFrame()
    for p in prng:
        df = pd.read_csv(f"{directory}/all_scores_aggregated_{p}.csv")
        df[["prop_corr_bic", "prop_corr_map"]] = 1 - df[["incorr_bic", "incorr_map"]]
        df = df.drop(columns=["incorr_bic", "incorr_map", "Unnamed: 0"])
        df["p"] = p
        scores = scores.append(df, ignore_index=True)

    for metric in ["prop_corr_bic", "prop_corr_map", "sfd_bic", "sfd_map"]:
        df_plot = scores.pivot(index="p", columns="init", values=metric)
        plot = df_plot.plot.bar()
        plot.set(title="Performance by initialization",
                 ylabel=(f"{'Proportion correct' if 'prop_corr' in metric else 'SFD'} {metric[-3:].upper()} estimate"),
                 xlabel="Edge density")
        fig = plot.get_figure()
        fig.tight_layout()
        fig.savefig(f"{directory}/optimal_init_{metric}.png", bbox_inches='tight')
    return

"""Functions for hyperparameter tuning in simulated experiments"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from collections import Counter
from scripts.thesis.mcmc_helpers import sfd, dag_from_udg
from scripts.thesis.mcmc_algorithm import InputData
from scripts.grues.sample import sample_causal_dag
from scripts.medil.ecc_algorithms import find_clique_min_cover
from scripts.thesis.sample_udgs import gen_udgs_mcmc
from scripts.thesis.sample_sm_model import sample_fm_dag


def tuning_hyperparameters(d=5,
                           p=0.5,
                           directory="",
                           n_models=100,
                           max_moves=10**3,
                           n_samples=10**3,
                           burn_in=None,
                           conv_crit=500,
                           significance_level=0.05,
                           initialization="it_informed",
                           all_params=[{"lmbda": 1, "alpha_stay": 1, "alpha_exit": 1}],
                           model_type="observable_dag"):
    """Records performance over different choices of hyperparameters
    Args
    ----
    model_type: str, one of ["observable_dag", "partially_observed_sm_model"]
        Sampling model
    """
    total_num_udgs = 2**(np.triu(np.ones(shape=(d,d)), k=1)).sum()

    seed = 1312
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(columns=["lmbda",
                               "alpha_stay",
                               "alpha_exit",
                               "incorr_bic",
                               "incorr_map",
                               "sfd_bic", 
                               "sfd_map", 
                               "num_udgs_visited",
                               "num_udg_changes"])
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

        for params in all_params:
            idt = InputData(samples=X, rng=np.random.default_rng(0))
            if initialization == "it_informed":
                init = dag_from_udg(idt.it_udg(significance_level=significance_level))
            else:
                init = initialization
            idt.mcmc(init=init, 
                     max_moves=max_moves,
                     burn_in=burn_in,
                     conv_crit=conv_crit,
                     **params)

            est_udg = {"bic": idt.optimal_udg,
                       "map": idt.map_udg()}

            est_mmcm = {key: find_clique_min_cover(est_udg[key]) for key in ["bic", "map"]}

            num_udgs_visited = len(Counter(hash(udg.tobytes()) for udg in idt.markov_chain_udg).most_common())
            num_udgs_visited /= total_num_udgs

            chain = [hash(udg.tobytes()) for udg in idt.markov_chain_udg]
            num_udg_changes = sum([chain[i] != chain[i+1] for i in range(len(chain)-1)])
            num_udg_changes /= idt.moves

            record = [value for key, value in params.items()]
            record += [1-(est_udg[key] == true_udg).all() for key in ["bic", "map"]]
            record += [sfd(est_mmcm[key], true_mmcm) for key in ["bic", "map"]]
            record += [num_udgs_visited, num_udg_changes]

            df.loc[len(df)] = record

    df_scores = (df.groupby(["lmbda", "alpha_stay", "alpha_exit"])
                   .agg({"incorr_bic": np.mean,
                         "incorr_map": np.mean,
                         "sfd_bic": np.median,
                         "sfd_map": np.median,
                         "num_udgs_visited": np.mean,
                         "num_udg_changes": np.mean})
                   .reset_index())

    optimal_params = {metric: dict(df_scores.iloc[np.argmin(df_scores[metric])][["lmbda", "alpha_stay", "alpha_exit"]])
                      for metric in ["incorr_bic",
                                     "incorr_map",
                                     "sfd_bic",
                                     "sfd_map",
                                     "num_udgs_visited",
                                     "num_udg_changes"]}
    optimal_scores = df_scores.agg(np.min).drop(index=["lmbda", "alpha_stay", "alpha_exit"])
    pd.DataFrame(optimal_params).to_csv(f"{directory}/optimal_params_{p}.csv")
    pd.DataFrame(optimal_scores).to_csv(f"{directory}/optimal_scores_{p}.csv")
    df.to_csv(f"{directory}/all_scores_{p}.csv")
    df_scores.to_csv(f"{directory}/all_scores_aggregated_{p}.csv")
    return


def display_results_tuning_hyperparameters(prng, 
                                           directory="", 
                                           alpha_base=10, 
                                           alpha_exps=[-np.inf, 0, np.inf],
                                           lmbda_base=3, 
                                           lmbda_exps=[0]):
    """Plot results from tuning_hyperparameters()
    """
    optimal_scores = pd.DataFrame()
    for p in prng:
        tmp = pd.read_csv(f"{directory}/optimal_scores_{p}.csv").set_index("Unnamed: 0")
        optimal_scores = pd.concat([optimal_scores, tmp.rename(columns={"0":p})], axis=1)

    default_scores = pd.DataFrame()
    for p in prng:
        all_scores = pd.read_csv(f"{directory}/all_scores_aggregated_{p}.csv").set_index("Unnamed: 0")
        default_lmbda = default_alpha_stay = default_alpha_exit = 1
        tmp = (all_scores.loc[(all_scores["lmbda"]==default_lmbda) & 
                              (all_scores["alpha_stay"]==default_alpha_stay) & 
                              (all_scores["alpha_exit"]==default_alpha_exit)]
                         .drop(columns=["lmbda", "alpha_stay", "alpha_exit"]))
        tmp = tmp.reset_index(drop=True).transpose()
        default_scores = pd.concat([default_scores, tmp.rename(columns={0:p})], axis=1)

    for optimizing_metric in default_scores.index:
        df_plot = pd.concat([default_scores.loc[default_scores.index==optimizing_metric].transpose().rename(columns={optimizing_metric:"default"}),
                             optimal_scores.loc[optimal_scores.index==optimizing_metric].transpose().rename(columns={optimizing_metric:"tuned"})], axis=1)

        if "incorr" in optimizing_metric:
            df_plot = 1 - df_plot
            title = optimizing_metric.replace("incorr", "prop_corr")
        else:
            title = optimizing_metric
        plot = df_plot.plot.bar()
        plot.set(title=f"Tuning impact", 
                ylabel=f"{'Proportion correct' if 'corr' in optimizing_metric else 'SFD'} {optimizing_metric[-3:].upper()} estimate",
                xlabel="Edge density")
        fig = plot.get_figure()
        fig.tight_layout()
        fig.savefig(f"{directory}/tuning_impact_{title}.png", bbox_inches='tight')

    lmbdas = [lmbda_base**x for x in lmbda_exps]
    clrs = ["pink", "blue", "orange", "green", "purple"]
    colors = {lmbdas[i]:clrs[i%5] for i in range(len(lmbdas))} # to ensure colors are the same
    alpha_exps_cleaned = [e for e in alpha_exps if abs(e)<np.inf]
    for metric in ["prop_corr_bic", "prop_corr_map", "sfd_bic", "sfd_map"]:
        fig, axs = plt.subplots((len(prng)+2)//3, min(len(prng),3))
        for i in range(len(prng)):
            record = pd.read_csv(f"{directory}/all_scores_aggregated_{prng[i]}.csv").set_index("Unnamed: 0")
            record["stay_vs_exit"] = record[["alpha_stay", "alpha_exit"]].apply(lambda row: row["alpha_stay"] / row["alpha_exit"], axis=1)
            record["stay_vs_exit"] = record["stay_vs_exit"].replace({np.inf:alpha_base**(max(alpha_exps_cleaned)+1), 
                                                                     0:alpha_base**(min(alpha_exps_cleaned)-1)})
            record[["prop_corr_bic", "prop_corr_map"]] = 1 - record[["incorr_bic", "incorr_map"]]
            if len(prng)>1:
                if len(prng)//3 > 1:
                    ax =  axs[i//3, i%3]
                else:
                    ax =  axs[i%3]
            else:
                ax = axs
            ax.set(xscale="log")
            min_xtick = min(alpha_exps_cleaned) - 1 if -np.inf in alpha_exps else min(alpha_exps_cleaned)            
            max_xtick = max(alpha_exps_cleaned) + 1 if np.inf in alpha_exps else max(alpha_exps_cleaned)
            ax.set_xticks(np.logspace(min_xtick, max_xtick, num=len(alpha_exps), base=alpha_base), 
                          [0 for i in range(1) if -np.inf in alpha_exps] + [fr'${int(alpha_base)}^{{{int(i)}}}$' for i in alpha_exps_cleaned] + [r'$\infty$' for i in range(1) if np.inf in alpha_exps], rotation=45)
            for lmbda in record.lmbda.unique():
                df_plot = record[record["lmbda"]==lmbda]
                ax.scatter(x=df_plot["stay_vs_exit"], y=df_plot[metric], color=colors[lmbda])
        for ax in fig.get_axes():
            ax.label_outer()

        fig.text(0.5, 0, r'$\alpha_{stay}$ / $\alpha_{exit}$', ha='center')
        fig.text(0, 0.5, f"{'Proportion correct' if 'prop_corr' in metric else 'SFD'} {metric[-3:].upper()} estimate", va='center', rotation='vertical')
        fig.suptitle(r'Performance by $\lambda$, $\alpha_{stay}$ / $\alpha_{exit}$')

        hndles = [mpatches.Patch(color=clrs[i%5], label=fr'${lmbda_base}^{{{int(lmbda_exps[i])}}}$') for i in range(len(lmbda_exps))]
        fig.legend(handles=hndles, loc='upper left', title=r'$\lambda$', bbox_to_anchor=(1, 0.5))

        rnge = [prng[i] for i in list(range(0, len(prng), 3)) + list(range(1, len(prng), 3)) + list(range(2, len(prng), 3))]
        hndles = [mlines.Line2D([], [], color='white', marker='s', linestyle='solid', markeredgecolor="black", markersize=10, label=p) for p in rnge]
        fig.legend(handles=hndles, loc='lower left', title=r'Edge density', bbox_to_anchor=(1, 0.5), ncol=3)

        fig.tight_layout()
        fig.savefig(f"{directory}/tuning_scores_grid_{metric}.png", bbox_inches='tight')

    for p in prng:
        df_scores = pd.read_csv(f"{directory}/all_scores_aggregated_{p}.csv").set_index("Unnamed: 0")
        df_plot = df_scores[(df_scores["alpha_stay"]==1)&(df_scores["alpha_exit"]==1)]
        plot = df_plot.plot.bar("lmbda", "num_udgs_visited")
        plot.set(title=f"Number of UDGs visited (p={p})",
                 ylabel="% of all UDGs visited",
                 xlabel="lmbda")
        fig = plot.get_figure()
        plot.set_xticklabels([str(round(float(item.get_text()), 3)) for item in plot.get_xticklabels()])
        fig.tight_layout()
        fig.savefig(f"{directory}/num_udgs_visited_lmbda_p{p}.png", bbox_inches='tight')

        plot = df_plot.plot.bar("lmbda", "num_udg_changes")
        plot.set(title=f"Number of UDG changes (p={p})",
                 ylabel="% of moves that was a UDG change",
                 xlabel="lmbda")
        fig = plot.get_figure()
        plot.set_xticklabels([str(round(float(item.get_text()), 3)) for item in plot.get_xticklabels()])
        fig.tight_layout()
        fig.savefig(f"{directory}/num_udg_changes_lmbda_p{p}.png", bbox_inches='tight')

        df_plot = df_scores[(df_scores["lmbda"]==1)&(df_scores["alpha_exit"]==1)]
        plot = df_plot.plot.bar("alpha_stay", "num_udgs_visited")
        plot.set(title=f"Number of UDGs visited (p={p})", 
                 ylabel="% of all UDGs visited",
                 xlabel=r'$\alpha_{stay}$ / $\alpha_{exit}$')
        fig = plot.get_figure()
        plot.set_xticklabels([str(round(float(item.get_text()), 3)) for item in plot.get_xticklabels()])
        fig.tight_layout()
        fig.savefig(f"{directory}/num_udgs_visited_alpha_stay_p{p}.png", bbox_inches='tight')

        plot = df_plot.plot.bar("alpha_stay", "num_udg_changes")
        plot.set(title=f"Number of UDG changes (p={p})",
                 ylabel="% of moves that was a UDG change",
                 xlabel=r'$\alpha_{stay}$ / $\alpha_{exit}$')
        fig = plot.get_figure()
        plot.set_xticklabels([str(round(float(item.get_text()), 3)) for item in plot.get_xticklabels()])
        fig.tight_layout()
        fig.savefig(f"{directory}/num_udg_changes_alpha_stay_p{p}.png", bbox_inches='tight')
    return


def generalized_performance(d=10,
                            p=0.5,
                            directory="",
                            n_models=100,
                            max_moves=10**3,
                            n_samples=10**3,
                            burn_in=None,
                            conv_crit=500,
                            significance_level=0.05,
                            initialization="it_informed",
                            directory_params="it_informed_initialization",
                            optimizing_metric="incorr_bic"):
    """Records performance over different choices of parameters (default vs configuration loaded from defined directory)
    """
    seed = 1312
    rng = np.random.default_rng(seed)

    record = pd.read_csv(f"{directory_params}/optimal_params_{p}.csv").set_index("Unnamed: 0")
    optimal_params = dict(record[optimizing_metric])

    param_options = ["default", "tuned"]

    df = pd.DataFrame(columns=["params",
                               "incorr_bic",
                               "incorr_map",
                               "sfd_bic",
                               "sfd_map"])

    for i in tqdm(range(n_models)):
        true_udg, dag, X = sample_causal_dag(num_nodes=d,
                                             edge_prob=p,
                                             weight_interv=(0, 1),
                                             samp_size=n_samples,
                                             return_uec=True,
                                             rng=rng,)
        np.fill_diagonal(true_udg, True)
        dag = dag.astype(int)
        true_mmcm = find_clique_min_cover(true_udg)

        for method in param_options:
            idt = InputData(samples=X, rng=np.random.default_rng(0))
            if initialization == "it_informed":
                init = dag_from_udg(idt.it_udg(significance_level=significance_level))
            else:
                init = initialization

            if method == "default":
                idt.mcmc(init=init, 
                         max_moves=max_moves,
                         burn_in=burn_in,
                         conv_crit=conv_crit)
            else:
                idt.mcmc(init=init, 
                         max_moves=max_moves,
                         burn_in=burn_in,
                         conv_crit=conv_crit,
                         **optimal_params)

            est_udg = {"bic": idt.optimal_udg,
                       "map": idt.map_udg()}

            est_mmcm = {key: find_clique_min_cover(est_udg[key]) for key in ["bic", "map"]}

            record = [method]
            record += [1-(est_udg[key] == true_udg).all() for key in ["bic", "map"]]
            record += [sfd(est_mmcm[key], true_mmcm) for key in ["bic", "map"]]

            df.loc[len(df)] = record

    df_scores = df.groupby(["params"]).agg({"incorr_bic": np.mean,
                                          "incorr_map": np.mean,
                                          "sfd_bic": np.median,
                                          "sfd_map": np.median}).reset_index()

    optimal_scores = df_scores.agg(np.min).drop(index=["params"])
    optimal_params = {metric:dict(df_scores.iloc[np.argmin(df_scores[metric])][["params"]]) 
                      for metric in ["incorr_bic","incorr_map","sfd_bic","sfd_map"]}

    pd.DataFrame(optimal_params).to_csv(f"{directory}/optimal_params_{p}.csv")
    pd.DataFrame(optimal_scores).to_csv(f"{directory}/optimal_scores_{p}.csv")
    df.to_csv(f"{directory}/all_scores_{p}.csv")
    df_scores.to_csv(f"{directory}/all_scores_aggregated_{p}.csv")
    return


def display_results_generalized_performance(prng,
                                            directory="",
                                            optimizing_metric="incorr_bic"):
    """Plot results from generalized_performance()
    """
    scores = pd.DataFrame()
    for p in prng:
        df = pd.read_csv(f"{directory}/all_scores_aggregated_{p}.csv")
        df[["prop_corr_bic", "prop_corr_map"]] = 1 - df[["incorr_bic", "incorr_map"]]
        df = df.drop(columns=["incorr_bic", "incorr_map", "Unnamed: 0"])
        df["p"] = p
        scores = scores.append(df, ignore_index=True)
    
    optimizing_metric = optimizing_metric.replace("incorr", "prop_corr")
    df_plot = scores.pivot(index="p", columns="params", values=optimizing_metric)
    plot = df_plot.plot.bar()
    plot.set(title="Performance by initialization",
             ylabel=(f"{'Proportion correct' if 'prop_corr' in optimizing_metric else 'SFD'} {optimizing_metric[-3:].upper()} estimate"),
             xlabel="Edge density")
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"{directory}/tuning_impact_{optimizing_metric}.png", bbox_inches='tight')
    return

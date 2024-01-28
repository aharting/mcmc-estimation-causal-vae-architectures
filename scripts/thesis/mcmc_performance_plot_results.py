"""Function to display results from experiments"""
from scripts.medil.ecc_algorithms import find_clique_min_cover
from scripts.thesis.mcmc_helpers import precision_recall_f1, sfd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def display_results(stored_results,
                    significance_level=0.05,
                    directory=""):
    """
    Unifying helper function to summarize and plot results from experiments
    in experiments/mcmc_performance 

    Args
    ----------
    stored_results : dict
        Dictionary of {p:{i:(idt, udg) for i in n_models} for p in prng}
        where idt is the InputData object, udg is the true UDG,
        prng is the edge density, and n_models is the number of models
    significance_level: float
        Significance level of independence testing for benchmarking
    directory: str
        Directory relative to current (i.e. where script is triggered)
        where plots should be stored
    """
    df_corr = pd.DataFrame()
    df_scores = pd.DataFrame()
    df_sfd = pd.DataFrame()
    df_incorr_present = pd.DataFrame()
    df_hpd_size = pd.DataFrame()
    iw_range = np.arange(start=0.1,stop=1,step=0.1)
    for p in stored_results.keys():
        for i in stored_results[p].keys():

            idt, udg = stored_results[p][i]

            est_udg = {"bic": idt.optimal_udg,
                       "map": idt.map_udg(),
                       "it": idt.it_udg(significance_level=significance_level)}

            est_mmcm = {key: find_clique_min_cover(est) for key, est in est_udg.items()}
            true_mmcm = find_clique_min_cover(udg)

            # Proportion correct
            corr_model = {key: (est == udg).all() for key, est in est_udg.items()}
            corr_model_hpd = {f"hpd_{iw*100}": idt.isin_hpd(true_udg=udg, interval_width=iw) for iw in [0.1, 0.2]}
            corr_model.update(corr_model_hpd)
            corr_model["p"] = p
            df_corr = df_corr.append(corr_model, ignore_index=True)

            # Scores
            scores_model = {key: precision_recall_f1(est=est, true=udg) for key, est in est_udg.items()} 
            scores_model["p"] = p
            scores_model = pd.DataFrame(scores_model)
            scores_model.index.name="score"
            scores_model = scores_model.reset_index()
            df_scores = df_scores.append(scores_model, ignore_index=True)

            # SFD
            sfd_model = {key: sfd(est, true_mmcm) for key, est in est_mmcm.items()}
            sfd_model["p"] = p
            df_sfd = df_sfd.append(sfd_model, ignore_index=True)

            # True UDG present in chain yielding incorrect estimate
            incorr_present_model = {key: idt.isin_mchn(true=udg, graph="udg", burn_in=(0 if key == "bic" else None))
                                    for key, corr in corr_model.items() if key != "it" and corr==False}

            incorr_present_model["p"] = p
            df_incorr_present = df_incorr_present.append(incorr_present_model, ignore_index=True)

            # HPD size
            hpd_size_model = {iw: len(idt.hpd(interval_width=iw)) for iw in iw_range}
            hpd_size_model["p"] = p
            df_hpd_size = df_hpd_size.append(hpd_size_model, ignore_index=True)

    # Proportion correct
    prop_corr = df_corr.groupby("p").mean()
    plot = prop_corr.plot.bar()
    plot.set(title="Proportion correctly estimated UDG",
             ylabel="Proportion",
             xlabel="Edge density")
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"{directory}/prop_corr.png", bbox_inches='tight')

    # Scores
    grpd = df_scores.groupby(["p", "score"]).mean()
    for key in grpd.columns:
        df_plot = grpd[key].reset_index().pivot(index="p", columns="score", values=key)
        plot = df_plot.plot.bar()
        plot.set(title=f"{key.upper()} optimal estimate",
                 ylabel="f1, precision, recall",
                 xlabel="Edge density")
        fig = plot.get_figure()
        fig.tight_layout()
        fig.savefig(f"{directory}/mean_scores_{key}.png", bbox_inches='tight')

    # SFD
    for key in df_sfd.columns:
        if key == "p":
            continue
        fig, ax = plt.subplots()
        df_sfd.boxplot(column=key, by="p", ax=ax)
        ax.set_xlabel("Edge density")
        ax.set_ylabel("SFD")
        fig.suptitle(f"SFD between {key.upper()} estimate and true UDG")
        ax.set_title("")
        fig.savefig(f"{directory}/sfd_{key}.png", bbox_inches='tight')

    # HPD size
    hpd_size_table = df_hpd_size.groupby("p").mean()
    hpd_size_table.to_csv(f"{directory}/avg_hpd_size.csv")

    mean_scores = df_incorr_present.groupby("p").mean().fillna(0)

    # Presence in chain when incorrectly estimates
    plot = mean_scores.plot.bar()
    plot.set(title=f"Incorrect UDG estimates where true UDG present in the chain",
             ylabel="Propotion",
             xlabel="Edge density")
    fig = plot.get_figure()
    fig.tight_layout()
    fig.savefig(f"{directory}/mean_scores_inchn_udg.png", bbox_inches='tight')

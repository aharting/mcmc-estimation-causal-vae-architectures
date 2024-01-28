"""NCFA application on simulated randomized control trial"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors
import matplotlib.patches as mpatches
import glob
import os
import re
import pickle
from sklearn.cluster import KMeans

from scripts.thesis.ncfa_pipeline import pipeline
from scripts.thesis.ncfa_simulated_experiments_helpers import sample_fm_dag_iv, prep_dataset, get_latent_repr
from scripts.thesis.mcmc_helpers import udg_from_dag
from scripts.medil.ecc_algorithms import find_clique_min_cover


def ncfa_simulated_experiment(ate=1, udg=np.ones(shape=(3,3)), n_latents_treated=1, n_samples=10**4):
    """Performs NCFA on a simulated experiment and observes shifts between treatment and control group in the latent space
    Args
    ----
    ate: int
        Average treatment effect in terms of # of standard deviations 
    
    n_latents_treated: int
        Number of latents exposed to treatment
    n_samples: int
        Number of samples of data
    """
    parent_directory = f"ate_{ate}sigma"

    #############################################
    # Synthetic data generation (simulated RCT) #
    #############################################
    data_directory = f"{parent_directory}/illustration_simulation_data"
    
    rng = np.random.default_rng(0)
    X, treated_idxs, biadj_matrix = sample_fm_dag_iv(udg, N=n_samples, rng=rng, n_latents_treated=n_latents_treated, ate=ate)

    # Illustrate the minimum MCM
    fig, ax = plt.subplots()
    cmap = colors.ListedColormap(['orange', 'b'])
    grid_plot = biadj_matrix.astype(int)
    ax.imshow(grid_plot, cmap=cmap)
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, grid_plot.shape[1]-0.5, 1));
    ax.set_yticks(np.arange(-0.5, grid_plot.shape[0]-0.5, 1));
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_yticklabels(range(grid_plot.shape[0]))
    ax.invert_yaxis()
    ax.set_xlabel("Measurement variable")
    ax.set_ylabel("Latent variable")
    ax.set_title("Matrix representation of minimum MCM")

    blue_patch = mpatches.Patch(color='blue', label='Adjacent')
    orange_patch = mpatches.Patch(color='orange', label='Non-adjacent')

    ax.legend(handles=[blue_patch, orange_patch])
    fig.tight_layout()
    fig.savefig(f"{data_directory}/mmcm_bic.png", bbox_inches='tight')

    # Prepare treatment and control group dataframes
    ctrl_idxs = list(set(range(n_samples))-set(treated_idxs))
    df_reduced = pd.DataFrame(X[ctrl_idxs])
    df_reduced_iv = pd.DataFrame(X[treated_idxs])

    # Illustrate simulated outcomes of measurement variables
    fig, ax = plt.subplots()
    ax.scatter(df_reduced_iv[0].to_numpy(), df_reduced_iv[6].to_numpy(), label="Treatment", color="b")
    ax.scatter(df_reduced[0].to_numpy(), df_reduced[6].to_numpy(), label="Control", color="orange")
    fig.suptitle("Synthetic data")
    ax.set_xlabel("Measured variable, unaffected")
    ax.set_ylabel("Measured variable, affected")
    ax.legend()
    fig.savefig(f"{data_directory}/synthetic_data_illustration.png")

    ##############################
    # NCFA on control group data #
    ##############################
    ctrl_directory = f"{parent_directory}/ncfa_control_group"

    # graph estimation params
    heuristic = False # Do not use heuristic for 1-pure child UDGs!
    method = "dcov_fast"
    alpha = 0.05
    dof = biadj_matrix.shape[0] # No excessive degrees of freedom
    dof_method = "uniform"
    seed = np.random.seed(0)
    scaler = "NA" # "sc" # "log" # "NA"

    biadj_mat_recons = {"true":biadj_matrix}
    dataset, num_features = prep_dataset(df_reduced, scaler=scaler)
    pipeline(dataset, heuristic, method, alpha, dof, dof_method, ctrl_directory, seed, biadj_mat_recons=biadj_mat_recons)

    for e in ["loss", "error"]:
        for i in range(2): # i = 0 for training, i = 1 for validation
            filenames = glob.glob(os.path.join(ctrl_directory, f"{e}_*.pkl"))
            fig, ax = plt.subplots()
            fig.suptitle(f"VAE {'Validation' if i == 1 else 'Training'} {e.capitalize()}")
            for f in filenames:
                file = open(f, 'rb')
                data = pickle.load(file)
                file.close()
                method = re.match(fr"{ctrl_directory}/{e}_(\w+).pkl", f).group(1)
                if method == "vanilla":
                    lbl = method
                else:
                    lbl = method.split("recon_")[1].upper()
                ax.plot(data[i], label=lbl)
                print(method, np.round(np.mean(data[i][-5:]),2))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(e.capitalize())
            ax.legend()
            fig.savefig(f"{ctrl_directory}/vae_{'valid' if i == 1 else 'train'}_{e}.png")
            fig.show()


    ################################
    # NCFA on treatment group data #
    ################################
    iv_directory = f"{parent_directory}/ncfa_treatment_group"

    dataset, num_features = prep_dataset(df_reduced_iv, scaler=scaler)
    method = "dcov_fast"

    pipeline(dataset, heuristic, method, alpha, dof, dof_method, iv_directory, seed, biadj_mat_recons=biadj_mat_recons)

    for e in ["loss", "error"]:
        for i in range(2): # i = 0 for training, i = 1 for validation
            filenames = glob.glob(os.path.join(iv_directory, f"{e}_*.pkl"))
            fig, ax = plt.subplots()
            fig.suptitle(f"VAE {'Validation' if i == 1 else 'Training'} {e.capitalize()}")
            for f in filenames:
                file = open(f, 'rb')
                data = pickle.load(file)
                file.close()
                method = re.match(fr"{iv_directory}/{e}_(\w+).pkl", f).group(1)
                if method == "vanilla":
                    lbl = method
                else:
                    lbl = method.split("recon_")[1].upper()
                ax.plot(data[i], label=lbl)
                print(method, np.round(np.mean(data[i][-5:]),2))
            ax.set_xlabel("Epoch")
            ax.set_ylabel(e.capitalize())
            ax.legend()
            fig.savefig(f"{iv_directory}/vae_{'valid' if i == 1 else 'train'}_{e}.png")
            fig.show()

    ###################
    # Compare results #
    ###################
    output_directory = f"{parent_directory}/results_treatment_vs_control"

    data = {}
    data_recon = {}
    data_actual = {}
    model = "recon_true"
    data["iv"], data_recon["iv"] = get_latent_repr(df_reduced_iv, scaler=scaler, model=model, directory=iv_directory)
    data["ctrl"], data_recon["ctrl"] = get_latent_repr(df_reduced, scaler=scaler, model=model, directory=ctrl_directory)

    dataset, _ = prep_dataset(df_reduced_iv, scaler=scaler)
    data_actual["iv"], _ = dataset
    dataset, _ = prep_dataset(df_reduced, scaler=scaler)
    data_actual["ctrl"], _ = dataset

    # Compute ATE among measurement variables
    ate = pd.DataFrame()
    ate["recon"] = (np.mean(data_recon["iv"].detach().numpy(), axis=0) / np.mean(data_recon["ctrl"].detach().numpy(), axis=0)) - 1
    ate["actual"] = (np.mean(data_actual["iv"], axis=0) / np.mean(data_actual["ctrl"], axis=0)) - 1
    ate.to_csv(f"{output_directory}/ate.csv")

    # Treatment / control latent space representation
    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle('Treatment / control latent space representation')
    ax.set_xlabel('Value')
    ax.set_ylabel('Latent variable')

    clrs = {"iv":"blue", "ctrl":"orange"}
    x_clrs = {"iv":"red", "ctrl":"green"}
    labels = {"iv":"treatment", "ctrl":"control"}

    # Compute centroids
    for latent_num in range(data["iv"].shape[1]):
        cs = []
        for key, d in data.items():
            data_use = pd.DataFrame(pd.DataFrame(data[key]))
            kmeans = KMeans(n_clusters=1)
            kmeans.fit(data_use)
            centroids = kmeans.cluster_centers_
            cs.append(centroids)
            x = data_use.loc[:, latent_num]
            y = np.ones(len(x))*latent_num
            ax.scatter(x, y, c=clrs[key], label=labels[key], alpha=0.5, s=200)  # plot different colors per cluster
            # ax.legend()
        for i in range(len(cs)):
            centroids = cs[i]
            cx = centroids[:, latent_num]
            cy = np.ones(len(cx))*latent_num
            ax.scatter(cx, cy,
                       marker='X', s=200, linewidths=1.5,
                       color=list(x_clrs.items())[i][1], edgecolors="black", lw=1.5)

    blue_patch = mpatches.Patch(color='blue', label='Treatment')
    red_x = mlines.Line2D([], [], color='red', marker='X', linestyle='None',
                              markersize=10, label='Treatment centroid')
    orange_patch = mpatches.Patch(color='orange', label='Control')
    green_x = mlines.Line2D([], [], color='green', marker='X', linestyle='None',
                              markersize=10, label='Control centroid')
    ax.legend(handles=[blue_patch, red_x, orange_patch, green_x], loc="upper right")

    ax.set_yticks(np.arange(0, data["iv"].shape[1], 1));
    ax.set_yticklabels([int(s) for s in range(data["iv"].shape[1])])        

    plt.show()

    # Illustrate shift for two selected latent variables
    fig.savefig(f"{output_directory}/latent_space_shift_all_latents.png")

    fig = plt.figure()
    ax = fig.add_subplot()
    fig.suptitle('Treatment / control latent space representation')
    ax.set_xlabel('Latent variable 0')
    ax.set_ylabel('Latent variable 1')

    clrs = {"iv":"blue", "ctrl":"orange"}
    x_clrs = {"iv":"red", "ctrl":"green"}
    labels = {"iv":"treatment", "ctrl":"control"}

    cs = []
    # Compute centroids
    for key, d in data.items():
        data_use = pd.DataFrame(pd.DataFrame(data[key]))
        x = data_use.loc[:, 0]
        y = data_use.loc[:, 1]
        ax.scatter(x, y, c=clrs[key], label=labels[key], alpha=0.5, s=200)  # plot different colors per cluster
    for key, d in data.items():
        data_use = pd.DataFrame(pd.DataFrame(data[key]))
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(data_use)
        centroids = kmeans.cluster_centers_
        cs.append(centroids)
        cx = centroids[:, 0]
        cy = centroids[:, 1]
        ax.scatter(cx, cy,
                   marker='X', s=200, linewidths=1.5,
                   color=x_clrs[key], edgecolors="black", lw=1.5)
            # ax.legend()

    ax.legend(handles=[blue_patch, red_x, orange_patch, green_x], loc="upper right")
    plt.show()
    fig.savefig(f"{output_directory}/latent_space_shift.png")

    dist = np.linalg.norm(cs[0] - cs[1])
    print("Centroids dist", dist)
    return

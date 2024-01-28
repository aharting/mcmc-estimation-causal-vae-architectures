"""Pipeline for NCFA model fitting, similar to pipeline specified below
SOURCE: Markham, A., Liu, M., Aragam, B., & Solus, L. (2023), https://gitlab.com/alex-markham/medil/-/blob/ncfa/exp/pipeline.py"""
from scripts.medil.data_loader import load_dataset_real
from scripts.medil.estimation import estimation
from scripts.medil.functional_MCM import assign_DoF
from scripts.medil.train import train_vae
from datetime import datetime
import numpy as np
import pickle
import os
import torch
import glob
import re

def run_vae_suite(train_loader, valid_loader, cov_train, cov_valid, path, seed):
    """ Run training loop for exact VAE
    Parameters
    ----------
    biadj_mat_recon: adjacency matrix for heuristic graph
    train_loader: loader for training data
    valid_loader: loader for validation data
    cov_train: covariance matrix for training data
    cov_valid: covariance matrix for validation data
    path: path to save the experiments
    seed: random seed for the experiments
    """

    filenames = glob.glob(os.path.join(path, f"biadj_mat_redundant*.npy"))
    # train & validate heuristic MeDIL VAE
    for mat_path in filenames:
        biadj_mat_recon = np.load(mat_path)
        mh, nh = biadj_mat_recon.shape
        model_recon, loss_recon, error_recon = train_vae(
            mh, nh, biadj_mat_recon, train_loader, valid_loader, cov_train, cov_valid, seed
        )
        method = re.match(fr"{path}/biadj_mat_redundant(\w+).npy", mat_path).group(1)
        torch.save(model_recon, os.path.join(path, f"model_recon{method}.pt"))
        with open(os.path.join(path, f"loss_recon{method}.pkl"), "wb") as handle:
            pickle.dump(loss_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(path, f"error_recon{method}.pkl"), "wb") as handle:
            pickle.dump(error_recon, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # train & validate vanilla VAE
    biadj_mat_vanilla = np.ones((mh, nh))
    model_vanilla, loss_vanilla, error_vanilla = train_vae(
        mh, nh, biadj_mat_vanilla, train_loader, valid_loader, cov_train, cov_valid, seed
    )
    torch.save(model_vanilla, os.path.join(path, "model_vanilla.pt"))
    with open(os.path.join(path, "loss_vanilla.pkl"), "wb") as handle:
        pickle.dump(loss_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path, "error_vanilla.pkl"), "wb") as handle:
        pickle.dump(error_vanilla, handle, protocol=pickle.HIGHEST_PROTOCOL)


def pipeline(dataset, heuristic, method, alpha, dof, dof_method, path, seed, biadj_mat_recons={}):
    """ Pipeline function for estimating the shd and number of reconstructed latent
    Parameters
    ----------
    dataset: dataset for real experiments
    heuristic: whether to use heuristic or not
    method: method for udg estimation
    alpha: significance level
    dof: desired size of latent space of VAE
    dof_method: how to distribute excess degrees of freedom to latent causal factors
    path: path for saving the files
    seed: random seed
    """

    # load parameters
    np.random.seed(seed)
    batch_size = 128
    samples, valid_samples = dataset

    # learn MeDIL model using provided biadj_mat and save graph
    for key, biadj_mat_recon in biadj_mat_recons.items():
        biadj_mat_redundant = assign_DoF(biadj_mat_recon, deg_of_freedom=dof, method=dof_method)
        np.save(os.path.join(path, f"biadj_mat_recon_{key}.npy"), biadj_mat_recon)
        np.save(os.path.join(path, f"biadj_mat_redundant_{key}.npy"), biadj_mat_redundant)

    # learn MeDIL model using IT testing and save graph
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Learning the MeDIL model")
    biadj_mat_recon = estimation(samples, heuristic=heuristic, method=method, alpha=alpha)
    biadj_mat_redundant = assign_DoF(biadj_mat_recon, deg_of_freedom=dof, method=dof_method)
    np.save(os.path.join(path, "biadj_mat_recon_IT.npy"), biadj_mat_recon)
    np.save(os.path.join(path, "biadj_mat_redundant_IT.npy"), biadj_mat_redundant)

    info = {"heuristic": heuristic, "method": method, "alpha": alpha, "dof": dof, "dof_method": dof_method}
    with open(os.path.join(path, "info.pkl"), "wb") as f:
        pickle.dump(info, f)

    # define VAE training and validation sample
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preparing training and validation data for VAE")
    train_loader = load_dataset_real(samples, batch_size)
    valid_loader = load_dataset_real(valid_samples, batch_size)

    # perform vae training
    # biadj_mat_recon = np.load(os.path.join(path, "biadj_mat_redundant.npy"))
    cov_train, cov_valid = np.eye(samples.shape[1]), np.eye(samples.shape[1])
    run_vae_suite(train_loader, valid_loader, cov_train, cov_valid, path, seed)


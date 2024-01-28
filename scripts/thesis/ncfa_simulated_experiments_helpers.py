"""Helper functions to NCFA application on simulated randomized control trial"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import copy
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scripts.medil.ecc_algorithms import find_clique_min_cover
from scripts.thesis.mcmc_helpers import udg_from_dag

def sample_fm_dag_iv(U, 
                     N=10**3, 
                     n_latents_treated=0, 
                     ate=1, 
                     sigma=1, 
                     low=0.5, 
                     high=2, 
                     sanity_check=False, 
                     deterministic=False, 
                     weighted=False, 
                     rng=np.random.default_rng(0)):
    """Sample from a UDG using a factor model DAG
    """
    biadj_matrix = find_clique_min_cover(U)
    k, m = biadj_matrix.shape
    A = np.concatenate([np.concatenate([np.zeros((k,k)), biadj_matrix.astype(int)], axis=1),
                        np.zeros((m, k+m))]
                       , axis=0)
    assert (udg_from_dag(A)[k:, k:] == U).all()
    X, treated_idxs = sample_dag_iv(A, 
                                    n_latents_treated=n_latents_treated, 
                                    ate=ate, 
                                    N=N, 
                                    sigma=sigma, 
                                    low=low, 
                                    high=high, 
                                    sanity_check=sanity_check, 
                                    deterministic=deterministic, 
                                    weighted=weighted, 
                                    rng=rng)
    return X[:, k:], treated_idxs, biadj_matrix

def sample_dag_iv(A, 
                  n_latents_treated=0, 
                  ate=1, 
                  N=10**3, 
                  sigma=1, 
                  low=0.5, 
                  high=2, 
                  sanity_check=False, 
                  deterministic=False, 
                  weighted=False, 
                  rng=np.random.default_rng(0)):
    d = A.shape[0]
    if weighted:
        D = A
    else:
        # Randomly generate edge weights
        W = np.multiply(
          rng.uniform(low=low, high=high, size=(d,d)),
          rng.choice([-1,1], size=(d,d))
        )
        # Obtain DAG with edge weights
        D = np.multiply(A, W)

    if deterministic:
        # Generate samples
        X = np.zeros((N,d))
        for j in range(N):
            smpl = np.zeros(d)
            # Sample according to topological order
            for i in range(d):
                if A[:,i].sum() == 0:
                    smpl[i] = rng.normal()
                else:
                    smpl[i] = np.dot(smpl, D[:,i])
            X[j,:] = smpl
    else:
        # Generate samples
        draws = rng.normal(size=(N,d))*sigma
        # The below gives equivalent results to      
        # X = (np.linalg.inv(np.eye(d) - D.T) @ draws.T).T
        X = np.zeros((N,d))
        treated_idxs = np.sort(rng.choice(range(N), size=int(N/2), replace=False))
        for j in range(N):
            smpl = np.zeros(d)
            # Sample according to topological order
            for i in range(d):
                smpl[i] = np.dot(smpl, D[:,i]) + draws[j,i]
                if i < n_latents_treated and j in treated_idxs: # treatment acts on first n_latents_treated latents
                    smpl[i] += ate*sigma
            X[j,:] = smpl
    return X, treated_idxs

def log_transform(df):
    df = copy.deepcopy(df)
    for col in df.columns:
        q10 = df[col].quantile(.1)
        if q10 == 0:
            print(col)
            continue
        df[col] = df[col].replace(0, q10)
        df[col] = np.log(df[col])
    return df

def prep_dataset(df, scaler="sc"):
    """
    Preserves column order or original dataframe
    """
    df_transformed = pd.get_dummies(df, prefix_sep=':', drop_first=False)
    # Transform
    scale_cols = [col for col in df.columns if df_transformed[col].max()>1]
    if scale_cols:
        if scaler == "sc":
            sc = StandardScaler() # consider log transforms instead, with log percentile
            df_transformed[scale_cols] = sc.fit_transform(df_transformed[scale_cols])
        elif scaler == "log":
            df_transformed[scale_cols] = log_transform(df_transformed[scale_cols])
    # Train test split
    dataset_train, dataset_valid = train_test_split(df_transformed, test_size=0.25)
    dataset_train = dataset_train.values
    dataset_valid = dataset_valid.values
    dataset = [dataset_train.astype(float), dataset_valid.astype(float)]
    num_features = dataset_train.shape[1]
    return dataset, num_features


def get_latent_repr(df, directory, scaler="NA", model="vanilla"):
    dataset, num_features = prep_dataset(df, scaler=scaler)
    samples, valid_samples = dataset
    batch_size = len(samples)

    samples_x = samples.astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for x_batch in data_loader:
        break
    x_batch = x_batch[0]
    model = torch.load(os.path.join(directory, f"model_{model}.pt"))
    model.eval()
    x_recon, mu_batch, logvar_batch = model(x_batch)
    data = mu_batch.detach().numpy() # np.concatenate([mu_batch.detach().numpy(), logvar_batch.detach().numpy()], axis=1)
    return data, x_recon

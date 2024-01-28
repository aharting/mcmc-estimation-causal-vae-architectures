"""Function to sample from a source-measurement (factor) model"""
import numpy as np
from tqdm import tqdm
from scripts.thesis.mcmc_helpers import udg_from_dag
from scripts.medil.ecc_algorithms import find_clique_min_cover
from scripts.thesis.mcmc_algorithm import InputData

def sample_dag(A, 
               N=10**3, 
               sigma=1, 
               low=0.5, 
               high=2, 
               sanity_check=False, 
               deterministic=False, 
               weighted=False, 
               rng=np.random.default_rng(0)):
    """Generates a sample from an adjacency matrix.
    Parameters
    ----------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.
        The matrix is sorted in topological order
    
    N: int
       The number of samples
    
    sigma: float
           Standard deviation in local models
    
    low: float
             Minimum abs edge weight

    high: float
             Maximum abs edge weight
                 
    Returns
    -------
    X : np.array
        A sample of size N from the DAG with d vertices in the shape of a (N,d)-array
    """
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
        for j in range(N):
            smpl = np.zeros(d)
            # Sample according to topological order
            for i in range(d):
                smpl[i] = np.dot(smpl, D[:,i]) + draws[j,i]
            X[j,:] = smpl
    
    # Sanity check
    if sanity_check:
        udg = udg_from_dag(A)
        cov = np.zeros(A.shape)
        cov[np.where(abs(np.cov(X, rowvar=False)) > 0.07)] = 1
        cov = cov.astype(bool)
        assert (cov == udg).all(), "Cause for concern"
    return X


def sample_fm_dag(U, N=10**3, sigma=1, low=0.5, high=2, sanity_check=False, deterministic=False, weighted=False, rng=np.random.default_rng(0)):
    """Sample from a UDG using a factor model DAG
    """
    biadj_matrix = find_clique_min_cover(U)
    k, m = biadj_matrix.shape
    A = np.concatenate([np.concatenate([np.zeros((k,k)), biadj_matrix.astype(int)], axis=1),
                        np.zeros((m, k+m))]
                       , axis=0)
    assert (udg_from_dag(A)[k:, k:] == U).all()
    X = sample_dag(A, N=N, sigma=sigma, low=low, high=high, sanity_check=sanity_check, deterministic=deterministic, weighted=weighted, rng=rng)
    return X[:, k:]



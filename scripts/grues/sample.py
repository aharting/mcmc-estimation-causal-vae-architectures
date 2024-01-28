"""Function to sample DAG models and simulate data
SOURCE: Deligeorgaki, D., Markham, A., Misra, P., & Solus, L. (2022), https://codeberg.org/alex-markham/GUES/src/branch/main/gues/sample.py
"""
import numpy as np
from numpy.random import default_rng


def sample_causal_dag(
    num_nodes,
    edge_prob,
    weight_interv=(0.25, 1),
    samp_size=1000,
    return_uec=False,
    rng=default_rng(0),
):
    """Draw sample from a causal model described by a DAG
    Parameters
    ----------
    num_nodes: number of nodes in the DAG
    edge_prob: edge probability
    weight_interv: given the pair (a, b), weights are drawn from the interval union [-b, -a)  union (a, b]
    samp_size: number of samples
    rng: random generator

    Returns
    -------
    sample: sample
    dag: adjacency matrix of the DAG, with entry (i,j) indicating an edge from node i to j
    """
    dag_adj = np.zeros((num_nodes, num_nodes), bool)

    max_edges = (num_nodes * (num_nodes - 1)) // 2
    num_edges = np.round(edge_prob * max_edges).astype(int)
    edges = np.ones(max_edges)
    edges[num_edges:] = 0
    dag_adj[np.triu_indices(num_nodes, k=1)] = rng.permutation(edges)

    if return_uec:
        trans_closure = np.linalg.matrix_power(
            (np.eye(num_nodes) + dag_adj).astype(int), num_nodes - 1
        )
        uec = (trans_closure.T @ trans_closure).astype(bool)
        np.fill_diagonal(uec, 0)

    a, b = weight_interv
    weights = rng.uniform(-b, -a, num_edges)
    weights[rng.choice((True, False), num_edges)] *= -1

    precision = np.eye(num_nodes, dtype=float)
    precision[dag_adj] = weights
    precision = precision.dot(precision.T)

    cov = np.linalg.inv(precision)

    sample = rng.multivariate_normal(np.zeros(num_nodes), cov, samp_size)

    causal_order = rng.permutation(num_nodes)

    dag_adj = dag_adj[:, causal_order][causal_order, :]
    sample = sample[:, causal_order]

    if return_uec:
        uec = uec[:, causal_order][causal_order, :]
        return uec, dag_adj, sample
    else:
        return dag_adj, sample

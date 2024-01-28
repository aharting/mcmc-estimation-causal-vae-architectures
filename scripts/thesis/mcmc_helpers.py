"""Various helper functions"""
import numpy as np


def precision_recall_f1(est, true):
    """Computes the precision, recall amd f1 scores for a udg
    """
    true = {tuple(i) for i in np.argwhere(true)}
    pos = {tuple(i) for i in np.argwhere(est)}
    tp = true.intersection(pos)
    p = len(tp) / len(pos)
    r = len(tp) / len(true)
    f1 = 2 * (p * r) / (p + r)
    return {"p": p, "r": r, "f1": f1}


def sfd(est, true):
    """Computes the structural Frobenius distance
    """
    b1 = est.astype(int)
    b2 = true.astype(int)
    diff = b1.T @ b1 - b2.T @ b2
    return np.linalg.norm(diff, ord="fro")


def check_dag(A, an_graph=None):
    """Checks whether the adjacency matrix indeed encodes a DAG.
    Args
    -------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.
    an_graph: np.array
              The ancestor graph (\\sum_{p=0}^{d-1}A^p)
    Raises
    -------
    Assertion error if A encodes cycles 
    
    Returns
    -------
    """
    num_nodes = A.shape[0]
    if an_graph is None:
        graph = np.copy(A)
        for p in range(2, num_nodes):
            graph += np.linalg.matrix_power(A, p)
    else:
        assert A.shape == an_graph.shape
        graph = np.copy(an_graph) - np.eye(A.shape[0])
    assert np.diag(graph).sum() == 0, "Encoded graph contains cycles"


def a(M):
    """Helper function to udg_from_dag()
    Args
    -------
    M : 2d numpy array of float
        Symmetric
    """
    aM = np.copy(M)
    aM[np.where(aM != 0)] = 1
    np.fill_diagonal(aM, 0)
    return aM


def udg_from_dag(A, an_graph=None):
    """Retrieves the UDG from a DAG.
    Args
    -------
    A : np.array
        The adjacency matrix of a DAG, where A[i,j] != 0 => i -> j.
    
    Returns
    -------
    udg : 2d numpy array of bool
          A :math:`M \times M` matrix with :math:`M` random variables.
    """
    num_nodes = A.shape[0]
    if an_graph is None:
        graph = np.eye(A.shape[0])
        for p in range(1, num_nodes):
            graph += np.linalg.matrix_power(A, p)
    else:
        assert A.shape == an_graph.shape
        graph = np.copy(an_graph)
    udg = a(graph.T @ graph).astype(bool)
    np.fill_diagonal(udg, True)
    return udg


def get_max_cpdag(udg):
    r"""Return maximal CPDAG in the UEC.
    SOURCE: Deligeorgaki, D., Markham, A., Misra, P., & Solus, L. (2022),
    https://codeberg.org/alex-markham/GUES/src/branch/main/gues/grues.py#L181
    """
    U = np.copy(udg)
    compliment_U = ~U
    np.fill_diagonal(compliment_U, False)

    # V_ij == 1 if and only if there's a k adjacent to j but not i
    V = compliment_U @ U

    # W_ij == 1 if and only if there's k such that i--j--k is an induced path
    W = np.logical_and(V, U).T

    # This orients all v-structures and removes edges violating CI relations
    U[W] = False
    
    return U


def dag_from_cpdag(cpdag):
    """Given a CPDAG, return a DAG in the corresponding MEC
    """
    bidirected = (np.multiply(cpdag , cpdag.T)).astype(int)
    unidirected = cpdag - bidirected
    dag = unidirected + np.triu(bidirected, k=1)
    return dag


def dag_from_udg(udg):
    """Given a UDG, return a DAG in the corresponding UEC (assuming it is non-empty)
    """
    cpdag = get_max_cpdag(udg).astype(int)
    return dag_from_cpdag(cpdag)

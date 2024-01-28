"""Data loader functions
SOURCE: Markham, A., Liu, M., Aragam, B., & Solus, L. (2023), https://gitlab.com/alex-markham/medil/-/blob/ncfa/learning/data_loader.py"""
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def load_dataset(samples, num_latent, batch_size):
    """ Generate dataset given the adjacency matrix, number of samples and batch size
    Parameters
    ----------
    samples: samples from the MCM
    num_latent: number of latent variables
    batch_size: batch size

    Returns
    -------
    data_loader: data loader
    """

    samples_x = samples[:, num_latent:].astype(np.float32)
    samples_z = samples[:, :num_latent].astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x), torch.tensor(samples_z))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader


def load_dataset_real(samples, batch_size):
    """ Generate dataset given the adjacency matrix, number of samples and batch size
    Parameters
    ----------
    samples: samples from the MCM
    batch_size: batch size

    Returns
    -------
    data_loader: data loader
    """

    samples_x = samples.astype(np.float32)
    samples_z = np.empty(shape=(samples_x.shape[0], 0)).astype(np.float32)
    dataset = TensorDataset(torch.tensor(samples_x), torch.tensor(samples_z))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

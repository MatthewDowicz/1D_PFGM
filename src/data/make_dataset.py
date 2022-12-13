import jax
import jax.numpy as jnp
from jax import random

## PyTorch for Dataloaders
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np

from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union

import pathlib
from pathlib import Path
import os

def rho(rng,
        batchsize: int,
        mu1: float = -0.5,
        sig1: float = 0.2,
        mu2: float = 0.5,
        sig2: float = 0.2):

        """
        Function to create the initial charge (data) distribution needed for
        training. Function is to replicate the same initial distribution as 
        presented in this implementation of the PFGM: 
        https://observablehq.com/@dkirkby/poisson-flow

        Args:
        -----
        rng: KeyLike
            PRNG state.
        batchsize: int
            Number of samples from the distribution.
        mu1: Float
            Center of the first gaussian.
        sig1: Float
            Width of the first gaussian.
        mu2: Float
            Center of second gaussian.
        sig2: Float
            Width of second gaussian.

        Returns:
        --------
            x: jnp.DeviceArray
                DeviceArray containing 'batchsize' samples of the training distribution.
        """

        rng, subkey1, subkey2 = jax.random.split(rng, num=3)
        sub_batchsize = batchsize // 2
        x1 = mu1 + sig1 * random.normal(subkey1, shape=(sub_batchsize, 1))
        x2 = mu2 + sig2 * random.normal(subkey2, shape=(sub_batchsize, 1))
        x = jnp.concatenate([x1, x2])

        return x

def perturb(batchsize: int,
            rng: Any,
            sigma: float = 0.01, 
            tau: float = 0.03, 
            M: int = 291,
            restrict_M: bool = True):
    """
    Perturbing the augmented training data. See algorithm 2 in the PFGM paper
    (https://arxiv.org/pdf/2209.11178.pdf.). Found under models/utils_poisson.py
    on Github.
    
    Args:
    -----
        batchsize: int
            A batch of un-augmented training data.
        rng: jax.random.PRNGKey.
            PRNGKey needed for functional style required by JAX.
        sigma: float
            Noise parameter. Specifically, it's the standard deviation of the 
            gaussian distribution that is sampled from to get the noise in x/y
            (eps_x/eps_y).
        tau: float
            Hyperparameter. Not sure what it really is.. :)
        M: float
            Measure how far out you go from the distribution. 
            Used to sample m, which is the exponent of (1 + \tau).
        restrict_M: bool
            Flag to allow confing the norms of the data to be....

    Returns:
    --------
        y: jnp.array
            The perturbed samples.
    """
    # Splitting of keys to allow for PRNG.
    rng, subkey1, subkey2, subkey3, subkey4, subkey5, subkey6 = jax.random.split(rng, num=7)

    # Sample the unperturbed samples from the "charge" distribution.
    x = rho(subkey1, batchsize)
    # Sample the exponents of (1+tau) from m ~ U[0,M].
    m = random.uniform(subkey2, shape=(batchsize,1), minval=0, maxval=M)
    # Sample the noise parameter for the perturbed augmented data, z.
    eps_z = 0 + sigma * random.normal(subkey3, shape=(batchsize,1))
    eps_z = jnp.abs(eps_z)
    # Confine the norms of perturbed data.
    # See Appendix B.1.1 of the paper
    if restrict_M:
        idx = jnp.squeeze(eps_z < 0.005)
        num = int(jnp.sum(idx))
        restrict_m = int(M * 0.7)
        m = m.at[idx, 0].set(random.uniform(subkey4, shape=(num,), minval=0, maxval=restrict_m))

    data_dim = 1
    factor = (1 + tau) ** m

    # Create the noise parameter for the perturbed data, x.
    eps_x = 0 + sigma * random.normal(subkey5, shape=(batchsize, data_dim))
    # Note for our 1D case this just returns eps_x. If our data was higher
    # dimension, we'd actually have ||eps_x||.
    norm_eps_x = jnp.linalg.norm(eps_x, ord=2, axis=1) * factor[:, 0]

    # Perturb z
    perturbed_z = jnp.squeeze(eps_z) * factor[:,0]
    perturbed_z = jnp.reshape(perturbed_z, newshape=(len(x), data_dim))
    # Sample uniform angle over unit sphere (this is for calculating u in Eqn. 5).
    gaussian = random.normal(subkey6, shape=(len(x),data_dim))
    unit_gaussian = gaussian / jnp.linalg.norm(gaussian, ord=2, axis=1, keepdims=True)
    # Construct the perturbation for x.

    perturbation_x = unit_gaussian[:, 0] * norm_eps_x
    perturbation_x = jnp.reshape(perturbation_x, newshape=(-1, data_dim))
    # Perturb x.
    perturbed_x = x + perturbation_x
    # Augment the data with the extra dimension, z.
    perturbed_samples_vec = jnp.concatenate((perturbed_x, perturbed_z), axis=1)
    return x, perturbed_samples_vec


def empirical_field(x,y):
    """
    Function to calculate the empirical (seen) Poisson Field.
    This function does the brute force calculation of what the field
    looks like and is the "labels" for our supervised learning problem.
    This is the answer that we want the NN to learn to emulate.
    Code from losses.py from:
         https://github.com/Newbeeer/Poisson_flow/blob/main/losses.py
    
    Args:
    -----
        x: jax.DeviceArray
            The unperturbed data. These are the samples that approximate
            the charge distribution. Shape = (batchsize, 2). x = (x,0)
        y: jax.DeviceArray
            The perturbed data that have been lifted off the data hyperplane
            in z=0. These are the training samples for the NN. 
            Shape = (batchsize, 2). y = (x_perturbed, z)
            
    Returns:
    --------
        target: jax.DeviceArray
            The value of the empirical field. These are the "labels" that
            the NN will try to learn for each training sample in the N+1
            hemisphere. Shape = (batchsize, 2). target[0] = [Ex, Ez]
    """
    # Get distance between vector on hyperplane (z=0) and their perturbed versions
    # Expand dims here, so that the 2nd dim of the array isn't collapsed
    # ie. making sure that gt_distance.shape = (batchsize, 2)
    x = jnp.concatenate((x, jnp.zeros((len(x),1))), axis=1)
    gt_distance = jnp.sqrt(jnp.sum((jnp.expand_dims(x, axis=1) - y) ** 2, axis=-1, keepdims=False))
    
    # For numerical stability, we multiply each row by its minimum value
    # keepdims=True, so that we don't lose a dimension
    distance = jnp.min(gt_distance, keepdims=True)[0] / (gt_distance + 1e-7)
    distance = distance ** 2 # 2 == data_dim
    distance = distance[:, :, None] # add an extra dim for the sum below

    # Normalize the coefficients (effectively multiply by c(x_tilde))
    # Expand dims again to avoid losing dimension because we sum on line 40    
    coeff = distance / (jnp.sum(distance, axis=1, keepdims=True) + 1e-7)
    diff = - ((jnp.expand_dims(y, axis=1) - x))
    
    # Calculate empirical Poisson Field (N+1 dimension in the augmented space)
    gt_direction = jnp.sum(coeff * diff, axis=1, keepdims=False)
    gt_norm = jnp.linalg.norm(gt_direction, axis=1)
    # Normalizing the N+1-dimensional Poisson Field as in Sect. 3.2
    gt_direction /= jnp.reshape(gt_norm, (-1,1))
    gt_direction *= jnp.sqrt(2) # 2 == data_dim

    target = gt_direction
    return target

def save_gen_data(filename: str,
                  data: jnp.array):
    """
    Save the created dataset to the 'saved_data' directory.

    Args:
    -----
        filename: str
            Name of the file of the saved data.
        data: jnp.array
            The tuple of (perturbed data, labels)

    Returns:
    --------
        Saved jnp.array
    """
    
    dir_path = Path(os.path.abspath(os.path.join('..')))
    data_dir = dir_path / 'saved_data'
    file_path = data_dir / str(filename)
    
    return jnp.save(file_path, data)

def load_gen_data(filename: str):
    """
    Loads the saved data for use in training/evaluating a NN.

    Args:
    -----
        filename: str
            Name of the file to load

    Returns:
    --------
        The specified jnp.array
    """
    dir_path = Path(os.path.abspath(os.path.join('..')))
    data_dir = dir_path / 'saved_data'
    file_path = data_dir / f'{filename}.npy'

    return jnp.load(file_path)

def get_NN_data(rng, samplesize, save=False, filename=''):
    """
    Function that outputs the necessary data to train the NN.
    
    Args:
    -----
        samplesize: int
            Number of samples you want to create.
        rng: PRNGkey
            Key to allow PRNG for the creation of the data.
        save: bool
            Save the generated data.
        filename: str
            Filename of the data that will be saved
            
    Returns:
    --------
        dataloader: torch.DataLoader
            Returns an iterable dataloader for training/evaluation containing
            the tuple (y,E) where:
                y: jax.DeviceArray
                    The perturbed training samples.
                E: jax.DeviceArray
                    The "labels" for training. These are the empirical
                    values of the E/poisson field.
    """
    x, y = perturb(batchsize=samplesize,
                        rng = rng,
                        sigma = 0.01, 
                        tau = 0.03, 
                        M = 291,
                        restrict_M = True)

    E = empirical_field(x,y)

    if save == True:
        save_gen_data(filename, (y, E))
    else:
        pass 

    return (y, E)  


class JaxDataset(Dataset):
    def __init__(self, dataset):
        """
        Custom Pytorch Dataset for Jax.DeviceArrays.
        """
        self.data_set = dataset
        
    def __len__(self):
        return len(self.data_set[0])
    
    def __getitem__(self, idx):
        data = self.data_set
        image = np.array(data[0][idx])
        labels = np.array(data[1][idx])
        
        return image, labels
    
def numpy_collate(batch):
    """
    Function to allow jnp.arrays to be used in PyTorch Dataloaders.
    """
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(DataLoader):
    """Custom pyTorch DataLoader for numpy/jax arrays"""
    def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)


def get_dataloader(file_name: str, batchsize: int, shuffle: bool = True):
    """
    Returns a dataloader of the specified filename ie. returns a
    Training/Validation/Testing Dataloader.

    Args:
    -----
        filename: str
            Name of the data file
        batchsize: int
            Number of samples per batch.
        shuffle: bool
            Flag to tell the dataloader to shuffle the data or not.
            Default is True.
    """
    data = load_gen_data(str(file_name))
    dataset = JaxDataset(data)
    loader = NumpyLoader(dataset, batchsize, shuffle=shuffle)
    return loader

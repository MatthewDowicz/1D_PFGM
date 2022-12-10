import jax
import jax.numpy as jnp
from jax import random
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union



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
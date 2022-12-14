# JAX
import jax
import jax.numpy as jnp
from jax import random
# Numpy
import numpy as np
# For ODESolver
from scipy import integrate
# Basic imports
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
# 
from .train_model import MLPRegressTrainer
# from .module import TrainerModule



def prior_sampling(num_samps: int,
                   zmax: int = 30):
    """
    Sampling from the radially projected uniform distribution on the 
    hemisphere with the radius of the hemisphere being r=z_max. I.e.
    we are sampling from the hyperplane at z=z_max.
    See sect 3.3 in PFGM paper.
    Found in methods.py on Github.

    Args:
    -----
        num_samps: int
            Number of samples to sample from the prior distribution.
        zmax: int
            The height of the hyperplane that the samples of the radially
            projected uniform distribution have.

    Returns:
    --------
        init_samples: jnp.array
            The samples of the prior distribution e.g. (x_prior, z_max)
    """
    dshape = (num_samps,)
    rng = jax.random.PRNGKey(32)
    rng, subkey1, subkey2 = jax.random.split(key=rng, num=3)
    # Sample the radius from p_radius (details in Appendix A.4 in PFGM paper)
    max_z = zmax
    N = 2 # 1 dimensional data + 1 additional dimension (z)
    # Sampling from inverse-beta distribution
    samples_norm = random.beta(key=subkey1, a=N / 2. - 0.5, b=0.5, shape=(dshape[0], 1)) #R1 
    inverse_beta = samples_norm / (1 - samples_norm) # R2
    # Sampling from p_radius(R) by change-of-variable
    samples_norm = jnp.sqrt(max_z ** 2 * inverse_beta) # R3
    # clip the sample norm (radius)
    samples_norm = jnp.clip(samples_norm, 1, 3000) # 3000 in place of sampling.upper_norm

    # Uniformly sample the angle direction
    gaussian = random.normal(key=subkey2, shape=(dshape[0], 1))
    unit_gaussian = gaussian / jnp.linalg.norm(gaussian, axis=1, keepdims=True)

    # Radius times the angle direction
    init_samples = unit_gaussian * samples_norm
    zmax_coord = jnp.ones((len(init_samples), 1)) * max_z
    init_samples = jnp.concatenate((init_samples, zmax_coord), axis=1)

    return init_samples


def ode_sampler(num_samps: int,
                trainer: MLPRegressTrainer,
                zmax: int = 30,
                eps: float = 1e-3, 
                verbose: bool = False, 
                method: str = 'RK45'):
    """
    RK45 ODE sampler for basic PFGM.

    Args:
    -----
        num_samps: int
            Number of samples.
        trainer: train_mode.MLPRegressTrainer
            NN model for approximating the Poisson field
        zmax: int
            The height of the projected prior distribution.
            Defaults to 30.
        eps: float
           The reverse ODE will be integrated to 'eps' for numerical
           instability.
           Defaults to 1e-3.
        verbose: bool
            Flag for toggling between verbose output and non-verbose output
                verbose = True:
                    (data, zs), and number of function calls (nfe)
                verbose = False:
                    data, nfe
            Defaults to False
        method: str
            Type of ODESolver to use.
            Defaults to RK45

    Returns:
    --------
        xs: jnp.array
            The transformed prior datas x-values at each time step.
            The last xs entry should resemble the distribution of interest.
        zs: jnp.array
            The transformed prior datas z-values at each time step.
        nfe: int 
            Number of function calls ie. how many times the ODESolver had to run
            to reach the target distribution.
    """



    samples = prior_sampling(num_samps=num_samps, zmax=zmax)
    samples = samples.ravel()
    
    def ode_func(t, x):
        # Convert the flattened array into the shape the NN expects
        x = x.reshape(((-1, 2)))
        # Change-of-variable z=exp(t) for faster sampling
        z = np.exp(t)
        # Update the z-dimension before passing to NN
        if type(x) == np.ndarray:
            x[:, 1] = z
            input_vec = x
        else:
            input_vec = x.at[:, 1].set(z)
        # Get trained model
        model_bd = trainer.bind_model()
        # Predicted normalized Poisson field
        v = model_bd(input_vec)
        # Get dx/dz as shown in Sect. 3.3
        dt_dz = 1 / (v[:, -1] + 1e-5)
        dx_dt = v[:, 0]     
        dx_dz = dx_dt * dt_dz
        driftx = v.at[:, 0].set(dx_dz * z)
        drift = driftx.at[:, 1].set(z)
        del v
        del driftx
        return drift.reshape((-1,))

    solution = integrate.solve_ivp(ode_func,
                                   t_span=(np.log(zmax), np.log(eps)),
                                   y0=samples.ravel(),
                                   method=str(method),
                                   rtol=1e-4,
                                   atol=1e-4)
    
    nfe = solution.nfev
    data = solution.y

    xs = data[::2]
    zs = data[1::2]

    if verbose:
        return (xs, zs), nfe

    else:
        return xs[:, -1], nfe
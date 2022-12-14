# Basic Imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Sequence, Optional

# Changing fonts to be latex typesetting
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'dejavuserif'
rcParams['font.family'] = 'serif'

# Import make_dataset.py for visualizations.
import sys
sys.path.append("../..") # ..usually means going up one directory
from data import make_dataset as mkds

from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union


def perturbed_dist_vis(batchsize, rng, range_vals=(-1.5, 1.5), bin_num=30, sigma=0.01, tau=0.03, M=291, restrict_M = True):
    """
    Function to visualize the perturbed samples in the N+1 dimension.

    Args:
    -----
        batchsize: int
            Number of samples to visualize.
        zoom: bool
            Flag to focus on the unperturbed data range.
    """
    x, train_samps = mkds.perturb(batchsize=batchsize,
                                    rng = rng,
                                    sigma = sigma, 
                                    tau = tau, 
                                    M = M,
                                    restrict_M = restrict_M)
                    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].hist(x[:,0], alpha=0.5, range=range_vals, bins=bin_num, label='original charges (x)')
    ax[0].hist(train_samps[:,0], alpha=0.5, range=range_vals, bins=bin_num, label='perturbed charges (y)');
    ax[0].set(xlabel='Charge Values', ylabel='Frequency', title='Distributions of charges')
    ax[0].legend()
    ax[1].hist(train_samps[:,1], alpha=0.5, bins=bin_num, label='perturbed charges z-coord')
    ax[1].set(xlabel='Distance from dist. plane', ylabel='Frequency', title='Distribution of heights in N+1 (z)')

    # bins = np.histogram(np.hstack((x[:,0], train_samps[:,0])), bins=bin_num)[1]

    # if zoom:
    #     ax[0].hist(x[:,0], alpha=0.5, bins=bins, label='original charges (x)');
    #     ax[0].hist(train_samps[:,0], alpha=0.5, bins=bins, label='perturbed charges (y)');
    #     ax[0].set(xlabel='Charge Values', ylabel='Frequency', title='Distributions of charges')
    #     ax[0].set_xlim([-1.5, 1.5])
    #     ax[0].legend()
    #     ax[1].hist(train_samps[:,1], alpha=0.5, bins=bins, label='perturbed charges z-coord')
    #     ax[1].set(xlabel='Distance from dist. plane', ylabel='Frequency', title='Distribution of heights in N+1 (z)')

    # else:
    #     ax[0].hist(x[:,0], alpha=0.5, bins=bins, label='original charges (x)');
    #     ax[0].hist(train_samps[:,0], alpha=0.5, bins=bins, label='perturbed charges (y)');
    #     ax[0].set(xlabel='Charge Values', ylabel='Frequency', title='Distributions of charges')
    #     ax[0].legend()
    #     ax[1].hist(train_samps[:,1], alpha=0.5, bins=bins, label='perturbed charges z-coord')
    #     ax[1].set(xlabel='Distance from dist. plane', ylabel='Frequency', title='Distribution of heights in N+1 (z)')

    plt.show()

def perturb_vis(batchsize, rng):
    """
    Visualization of "lifting" the training data into the N+1 dimension.
    
    Args:
    -----
    data: jnp.array
        The non-perturbed training data.
    turb_data: jnp.array
        The perturbed training data ie. the data in the N+1 hemisphere.
    """

    x, train_samps = mkds.perturb(batchsize=batchsize,
                                  rng=rng)
    
    fig, ax = plt.subplots(1,2, figsize=(10,6))
    ax[0].scatter(train_samps[:,0], train_samps[:,1], label='perturbed samples')
    ax[0].scatter(x[:,0], x[:,0] * 0, label='original samples')
    ax[0].set(xlabel='X-values (values on hyperplane z=0)',
              ylabel='Z-values (values lifted in N+1 hemisphere)',
              title='Visualization of how training data is lifted into N+1')
    ax[0].legend()
    ax[1].scatter(train_samps[:,0], train_samps[:,1], label='perturbed samples')
    ax[1].scatter(x[:,0], x[:,0] * 0, label='original samples')
    ax[1].set_xlim([-1.5, 1.5])
    ax[1].set(xlabel='X-values (values on hyperplane z=0)',
              ylabel='Z-values (values lifted in N+1 hemisphere)',
              title='Zoom in on limits of the original samples')
    ax[1].legend()

    plt.show()


def quiver_plot(dataloader, xlim: Tuple[float, float] = (-10,10), ylim: Tuple[float, float] = (-1,10)):
    """
    Visualizing a batch of data via a quiver plot. The arrow direction is 
    provided by the E-field and the coordinate for the arrow is provided
    by the perturbed data vector.
    """
    batch = next(iter(dataloader))
    imgs, labels = batch
    plt.quiver(imgs[:,0], imgs[:,1], labels[:,0], labels[:,1], label='True Poisson Field');
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.show()
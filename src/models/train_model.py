# Standard Libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state, checkpoints

from .module import TrainerModule
class MLPRegressor(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    
    @nn.compact
    def __call__(self, x, **kwargs):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.silu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class MLPRegressTrainer(TrainerModule):
    
    def __init__(self,
                 hidden_dims : Sequence[int],
                 output_dim : int,
                 trial: Any = None,
                 **kwargs):
        super().__init__(model_class=MLPRegressor,
                         model_hparams={
                             'hidden_dims': hidden_dims,
                             'output_dim': output_dim
                         },
                         **kwargs)
    
    def create_functions(self):
        def mse_loss(params, batch):
            x, y = batch
            pred = self.model.apply({'params': params}, x)
            loss = ((pred - y) ** 2).mean()
            return loss
        
        def train_step(state, batch):
            loss_fn = lambda params: mse_loss(params, batch)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            metrics = {'loss': loss}
            return state, metrics
        
        def eval_step(state, batch):
            loss = mse_loss(state.params, batch)
            return {'loss': loss}
        
        return train_step, eval_step
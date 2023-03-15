"""
Modified from original IQL implementation to store window size of k previous states
"""

import functools
from typing import Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.dlpack
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
from common import MLP, Params, PRNGKey, default_init
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Input
import tensorflow as tf
from jax import random

from flax.linen import initializers


tfd = tfp.distributions
tfb = tfp.bijectors

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class NonMarkovPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        # print(f'observations: {observations.shape}')
        batch_size = self.hidden_dims[0]
        dropout_rate = 0.2
        # (256, k, 29)

        #(1, 6, 29) -> (1, 6, 8)
        # model = LSTM(256, input_shape=(observations.shape[1], observations.shape[2]), return_sequences=True, dropout=dropout_rate, activation='tanh')
        
        dim1, dim2, dim3 = observations.shape

        
        hidden_dim = 256
        hiddens = []
        carry = nn.LSTMCell.initialize_carry(random.PRNGKey(0), (dim1,), 256) # size 1, 256 
        # https://flax.readthedocs.io/en/latest/_modules/flax/linen/recurrent.html

        for i in range(6):
          x = observations[:, i, :]
          # print(x.shape)
          # print(carry[0].shape, carry[1].shape)
          new_carry, y = nn.LSTMCell()(carry, x) #(new_c, new_h), new_h
          carry = new_carry
          hiddens.append(y)

        # print(f'hiddens[0]: {hiddens[0].shape}')

        outputs = jnp.array(hiddens).reshape(dim1, dim2, 256)

        # print(f'outputs: {outputs.shape}')

        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim,)) # don't change dimension

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                                          temperature)
        # print(f'after {base_dist} {type(base_dist)}')

        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist

        return base_dist


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        
        # (1, 6, 29)
        print(f'observations: {observations.shape}')
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)
        print(f'outputs: {outputs.shape} {outputs.dtype} {type(outputs)}')
        means = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.log_std_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        # Add flag that says if non-Markov -> LSTM architecture using k
        k = 5 # Just for testing - set this in config / as a hyperparameter later

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)

        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


@functools.partial(jax.jit, static_argnames=('actor_def', 'distribution'))
def _sample_actions(rng: PRNGKey,
                    actor_def: nn.Module,
                    actor_params: Params,
                    observations: np.ndarray,
                    temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, temperature)
    rng, key = jax.random.split(rng)
    toreturn = dist.sample(seed=key)
    print(f'toreturn: {toreturn.shape}')
    return rng, toreturn


# Sample k times
def sample_actions(rng: PRNGKey,
                   actor_def: nn.Module,
                   actor_params: Params,
                   observations: np.ndarray,
                   temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations,
                           temperature)

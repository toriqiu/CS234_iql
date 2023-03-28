import functools
import logging
import time
from typing import NamedTuple, Optional, Any, Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

###############   MODEL ##########################


class SelfAttention(hk.MultiHeadAttention):
    """Self attention with a causal mask applied."""

    def __call__(
            self,
            query: jnp.ndarray,
            key: Optional[jnp.ndarray] = None,
            value: Optional[jnp.ndarray] = None,
            mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        key = key if key is not None else query
        value = value if value is not None else query
        seq_len = query.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))
        mask = mask * causal_mask if mask is not None else causal_mask
        # print('self attention', query.shape, key.shape, mask.shape)
        return super().__call__(query, key, value, mask)


class DenseBlock(hk.Module):
    """A 2-layer MLP"""

    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(hiddens, w_init=initializer)(x)


class Transformer(hk.Module):
    """A transformer stack."""

    def __init__(self,
                 num_heads: int,
                 num_layers: int,
                 dropout_rate: float,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._dropout_rate = dropout_rate

    def __call__(self,
                 h: jnp.ndarray,
                 mask: Optional[jnp.ndarray],
                 is_training: bool) -> jnp.ndarray:
        """Connects the transformer.
        Args:
          h: Inputs, [B
          , T, H].
          mask: Padding mask, [B, T].
          is_training: Whether we're training or not.
        Returns:
          Array of shape [B, T, H].
        """
        mask = np.zeros((h.shape[0],5))
        init_scale = 2. / self._num_layers
        dropout_rate = self._dropout_rate if is_training else 0.
        if mask is not None:
            mask = mask[:, None, None, :]

        for i in range(self._num_layers):
            h_norm = layer_norm(h, name=f'h{i}_ln_1')
            # print(h_norm.shape, mask)
            h_attn = SelfAttention(
                num_heads=self._num_heads,
                key_size=29,
                w_init_scale=init_scale,
                name=f'h{i}_attn')(h_norm, mask=mask)
            print('transformer')
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            print('h shapes',h_attn.shape, h.shape)
            h = h + h_attn
            h_norm = layer_norm(h, name=f'h{i}_ln_2')
            h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense
        h = layer_norm(h, name='ln_f')

        return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
    """Apply a unique LayerNorm to x with default settings."""
    return hk.LayerNorm(axis=-1,
                        create_scale=True,
                        create_offset=True,
                        name=name)(x)


######################################### TRAIN #########################################


batch_size = 16  # Train batch size per core
sequence_length = 128  # Sequence length to learn on

d_model = 256  # model width
num_heads = 4  # Number of attention heads
num_layers = 6  # Number of transformer layers
dropout_rate = 0.1  # Dropout rate

learning_rate = 2e-4  # Max learning-rate
grad_clip_value = 0.25  # Gradient norm clip value

checkpoint_dir = '/jax-transformer'  # Directory to store checkpoints
LOG_EVERY = 50
MAX_STEPS = 10 ** 6


def embeddings(data: Mapping[str, jnp.ndarray], vocab_size: int) :
    tokens = data['obs']
    input_mask = jnp.greater(tokens, 0)
    seq_length = tokens.shape[1]

    # Embed the input tokens and positions.
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    token_embedding_map = hk.Embed(vocab_size, d_model, w_init=embed_init)
    token_embs = token_embedding_map(tokens)
    positional_embeddings = hk.get_parameter(
        'pos_embs', [seq_length, d_model], init=embed_init)
    input_embeddings = token_embs + positional_embeddings
    return input_embeddings, input_mask


def build_forward_fn(vocab_size: int, d_model: int, num_heads: int,
                     num_layers: int, dropout_rate: float):
    """Create the model's forward pass."""

    def forward_fn(data: Mapping[str, jnp.ndarray],
                   is_training: bool = True) -> jnp.ndarray:
        """Forward pass."""
        # input_embeddings, input_mask = embeddings(data, vocab_size)
        input_mask = None

        # Run the transformer over the inputs.
        transformer = Transformer(
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate)
        output_embeddings = transformer(data, input_mask, is_training)

        return output_embeddings

    return forward_fn


def lm_loss_fn(forward_fn,
               vocab_size: int,
               params,
               rng,
               data: Mapping[str, jnp.ndarray],
               is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data, is_training)
    targets = jax.nn.one_hot(data['target'], vocab_size)
    assert logits.shape == targets.shape

    mask = jnp.greater(data['obs'], 0)
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    loss = jnp.sum(loss * mask) / jnp.sum(mask)

    return loss


class GradientUpdater:
    """A stateless abstraction around an init_fn/update_fn pair.
    This extracts some common boilerplate from the training loop.
    """

    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    @functools.partial(jax.jit, static_argnums=0)
    def init(self, master_rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(master_rng)
        params = self._net_init(init_rng, data)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    @functools.partial(jax.jit, static_argnums=0)
    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)

        updates, opt_state = self._opt.update(g, state['opt_state'])
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


# def main():
#     # Create the dataset.
#     train_dataset, vocab_size = load(batch_size,
#                                      sequence_length)
#     # Set up the model, loss, and updater.
#     forward_fn = build_forward_fn(vocab_size, d_model, num_heads,
#                                   num_layers, dropout_rate)
#     forward_fn = hk.transform(forward_fn)
#     loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, vocab_size)

#     optimizer = optax.chain(
#         optax.clip_by_global_norm(grad_clip_value),
#         optax.adam(learning_rate, b1=0.9, b2=0.99))

#     updater = GradientUpdater(forward_fn.init, loss_fn, optimizer)

#     # Initialize parameters.
#     logging.info('Initializing parameters...')
#     rng = jax.random.PRNGKey(428)
#     data = next(train_dataset)
#     state = updater.init(rng, data)

#     logging.info('Starting train loop...')
#     prev_time = time.time()
#     for step in range(MAX_STEPS):
#         data = next(train_dataset)
#         state, metrics = updater.update(state, data)
#         # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
#         # Using values from state/metrics too often will block the runahead and can
#         # cause these overheads to become more prominent.
#         # if step % LOG_EVERY == 0:
#         #     steps_per_sec = LOG_EVERY / (time.time() - prev_time)
#         #     prev_time = time.time()
#         #     metrics.update({'steps_per_sec': steps_per_sec})
#         #     logging.info({k: float(v) for k, v in metrics.items()})


# main()
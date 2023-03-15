from typing import Tuple

import jax
import jax.numpy as jnp

from common import Batch, InfoDict, Model, Params, PRNGKey

def update(key: PRNGKey, actor: Model, critic: Model, value: Model,
           batch: Batch, temperature: float) -> Tuple[Model, InfoDict]:
    v = value(batch.observations)
    # Update policy distribution based on critic
    q1, q2 = critic(batch.observations, batch.actions)
    q = jnp.minimum(q1, q2)
    exp_a = jnp.exp((q - v) * temperature) # exp ^ [Q(theta hat) - V(phi)]
    exp_a = jnp.minimum(exp_a, 100.0) # Why are we using 100 here?

    # Equation 7 from paper - modified from original IQL
    # Sample k timesteps from dist.log_prob
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params},
                           batch.observations,
                           training=True,
                           rngs={'dropout': key})
        # pi (a|s) -> pi (a|s_t, a_{t-1}, ...., s_{t-k}, a_{t-k-1}
        log_probs = dist.log_prob(batch.actions) 
        actor_loss = -(exp_a * log_probs).mean()

        return actor_loss, {'actor_loss': actor_loss, 'adv': q - v}

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

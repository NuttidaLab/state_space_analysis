import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import optax
import tensorflow_probability.substrates.jax as tfp
from typing import Tuple

tfd = tfp.distributions
tfb = tfp.bijectors

class BlockOneRE:
    def __init__(self):
        # total input dim = 8 (rt) + 7 (ra) + 8 (re) = 23
        self.input_dim = 8
        self.params = None

    def initialize(self, key: jr.PRNGKey, method="prior", w_scale: float = 1e-2) -> Tuple[dict, dict]:
        """
        Initialize exactly the same set of parameters as BlockHMMEmissions for num_states=1.
        """
        ks = jr.split(key, 9)
        params = {
            # Error GLM
            "weights_re":       jr.normal(ks[7], (8,)) * w_scale,
            "alpha_re":         tfb.Softplus()(jr.normal(ks[8], ()) * 0.1 + 0.5) + 1.0,
        }
        # no properties needed here (empty dict matches your existing signature)
        return params, {}

    def distribution(self, params: dict, inputs: jnp.ndarray):
        """
        """
        if inputs.ndim == 2:
            inputs = inputs[jnp.newaxis, ...]
        # split
        x_re = inputs

        # 3) Error ~ Gamma(GLM log-link)
        lp_re = jnp.einsum('...i,i->...', x_re, params["weights_re"])
        mu_re = jnp.exp(lp_re)
        return tfd.Independent(
            tfd.Gamma(
                concentration=params["alpha_re"],
                rate=params["alpha_re"] / mu_re
            ),
            reinterpreted_batch_ndims=0
        )

    def neg_log_likelihood(self, params, inputs, emissions):
        dist = self.distribution(params, inputs)
        ll   = dist.log_prob(emissions[..., 0])
        return -jnp.sum(ll)

    def fit_em(self, params, props, emissions, inputs, num_iters=100, lr=1e-3, verbose=True):
        """
        A simple Adam‐based MLE fit (no EM here, since single-state)
        """
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        @jax.jit
        def step(p, o):
            loss, grads = jax.value_and_grad(self.neg_log_likelihood)(p, inputs, emissions)
            updates, o2 = optimizer.update(grads, o)
            return optax.apply_updates(p, updates), o2, loss

        losses = []
        p, o = params, opt_state
        for _ in range(num_iters):
            p, o, loss = step(p, o)
            losses.append(loss)
        self.params = p
        return p, losses

    # def marginal_log_prob(self, params, emissions, inputs):
    #     single = False
    #     if emissions.ndim == 2:
    #         emissions = emissions[jnp.newaxis, ...]
    #         inputs    = inputs[jnp.newaxis, ...]
    #         single = True
    #     dist = self.distribution(params, inputs)
    #     rt_obs, ra_obs, re_obs = emissions[...,0], emissions[...,1], emissions[...,2]
    #     ll = dist.log_prob([rt_obs, ra_obs, re_obs])
    #     return jnp.sum(ll, axis=-1)

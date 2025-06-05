import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

class BlockOneRegressor:
    def __init__(self,
                 # feature dims reflect final GLM formulas:
                 # RT: 1 + Error* (Attention+Coh+Exp) → 8 features
                 # RA: 1 + stim + prev_stim + prev_resp + Attention + Coh + Exp → 7 features
                 # RE: 1 + rt* (Attention+Coh+Exp) → 8 features
                 input_dim_rt: int = 8,
                 input_dim_ra: int = 7,
                 input_dim_re: int = 8):
        self.input_dim_rt = input_dim_rt
        self.input_dim_ra = input_dim_ra
        self.input_dim_re = input_dim_re
        self.input_dim    = input_dim_rt + input_dim_ra + input_dim_re
        self.params = None

    def initialize(self, key, method="prior"):
        """Randomly initialize weights & positive parameters."""
        keys = jax.random.split(key, 6)
        # linear weights
        w_scale = 1e-2
        params = {
            "weights_rt": jax.random.normal(keys[0], (self.input_dim_rt,)) * w_scale,
            "weights_ra": jax.random.normal(keys[1], (self.input_dim_ra,)) * w_scale,
            "weights_re": jax.random.normal(keys[2], (self.input_dim_re,)) * w_scale,
        }
        # positive dispersion parameters
        for name, k in [("alpha_rt", keys[3]),
                        ("kappa_ra", keys[4]),
                        ("phi_re",   keys[5])]:
            raw = jax.random.normal(k, ()) * 0.1 + 0.5
            params[name] = tfb.Softplus().forward(raw)
        self.params = params
        return params, {}

    def distribution(self, params, inputs):
        """
        inputs: shape (..., input_dim)
        returns a JointDistributionSequential [dist_rt, dist_ra, dist_re]
        """
        # ensure batch dim exists
        if inputs.ndim == 2:
            inputs = inputs[jnp.newaxis, ...]
        # split features
        rt_in = inputs[..., :self.input_dim_rt]
        ra_in = inputs[..., self.input_dim_rt:self.input_dim_rt+self.input_dim_ra]
        re_in = inputs[..., -self.input_dim_re:]
        # 1) RT ~ Gamma(log-link)
        lp_rt = jnp.einsum('...i,i->...', rt_in, params['weights_rt'])
        mu_rt = jnp.exp(lp_rt)
        dist_rt = tfd.Independent(
            tfd.Gamma(
                concentration=params['alpha_rt'],
                rate=params['alpha_rt'] / mu_rt
            ),
            reinterpreted_batch_ndims=0
        )
        # 2) RA ~ Von Mises on mean direction
        lp_ra = jnp.einsum('...i,i->...', ra_in, params['weights_ra'])
        dist_ra = tfd.Independent(
            tfd.VonMises(
                loc=lp_ra,
                concentration=params['kappa_ra']
            ),
            reinterpreted_batch_ndims=0
        )
        # 3) RE ~ Beta(logit link)
        lp_re = jnp.einsum('...i,i->...', re_in, params['weights_re'])
        mu_re = jax.nn.sigmoid(lp_re)
        alpha = mu_re * params['phi_re']
        beta  = (1 - mu_re) * params['phi_re']
        dist_re = tfd.Independent(
            tfd.Beta(
                concentration1=alpha,
                concentration0=beta
            ),
            reinterpreted_batch_ndims=0
        )
        return tfd.JointDistributionSequential([dist_rt, dist_ra, dist_re])

    def neg_log_likelihood(self, params, inputs, emissions):
        rt_obs = emissions[..., 0]
        ra_obs = emissions[..., 1]
        re_obs = jnp.clip(emissions[..., 2], 1e-6, 1 - 1e-6)
        dist = self.distribution(params, inputs)
        ll = dist.log_prob([rt_obs, ra_obs, re_obs])
        return -jnp.sum(ll)

    def fit_em(self,
               init_params,
               props,
               emissions,
               inputs,
               num_iters=100,
               learning_rate=1e-3,
               verbose=True):
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(init_params)
        params    = init_params
        lls       = []
        @jax.jit
        def step(params, opt_state, inputs, emissions):
            loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, inputs, emissions)
            updates, opt_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), opt_state, loss
        for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting EM"):
            params, opt_state, loss = step(params, opt_state, inputs, emissions)
            lls.append(-loss)
        self.params = params
        return params, lls

    def marginal_log_prob(self, params, emissions, inputs):
        # batchify
        single = False
        if emissions.ndim == 2:
            emissions = emissions[jnp.newaxis, ...]
            inputs    = inputs[jnp.newaxis, ...]
            single = True
        rt_obs, ra_obs, re_obs = emissions[...,0], emissions[...,1], emissions[...,2]
        dist = self.distribution(params, inputs)
        ll   = dist.log_prob([rt_obs, ra_obs, re_obs])
        mll  = jnp.sum(ll, axis=-1)
        return mll[0] if single else mll

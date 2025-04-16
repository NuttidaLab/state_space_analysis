import jax
import jax.numpy as jnp
from jax import vmap
import optax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions

class GaussianModel:
	
	def __init__(self, output_dim=2):
		self.output_dim = output_dim
		self.params = None
		
	def initialize(self, key, method="prior"):
		key, subkey = jax.random.split(key)
		means = jax.random.normal(subkey, shape=(self.output_dim,))
		key, subkey = jax.random.split(key)
		covs = jax.random.uniform(subkey, shape=(self.output_dim,), minval=0.1, maxval=0.9)
		params = {"means": means, "covs": covs}
		props = {}  # No extra properties for now.
		self.params = params
		return params, props
	
	def distribution(self, params):
		jitter = 1e-5 * jnp.eye(self.output_dim)
		cov = params["covs"] + jitter
		return tfd.MultivariateNormalFullCovariance(
			params["means"], cov)
		
	def neg_log_likelihood(self, params, emissions):
		dist = self.distribution(params)
		return -jnp.sum(dist.log_prob(emissions))

	def fit_em(self, init_params, props, emissions, num_iters, learning_rate=0.01, verbose=True):

		optimizer = optax.adam(learning_rate)
		opt_state = optimizer.init(init_params)
		lls = []

		@jax.jit
		def step(params, opt_state, emissions):
			loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, emissions)
			updates, opt_state = optimizer.update(grads, opt_state, params)
			new_params = optax.apply_updates(params, updates)
			return new_params, opt_state, loss

		params = init_params
		for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting"):
			params, opt_state, loss = step(params, opt_state, emissions)
			lls.append(-loss)  # Record log likelihood
		return params, lls
	
	def marginal_log_prob(self, params, emissions):
		if emissions.ndim == 2:
			emissions = jnp.expand_dims(emissions, axis=0)
			single_trial = True
		else:
			single_trial = False

		dist = self.distribution(params)
		log_probs = dist.log_prob(emissions)
		mll = jnp.sum(log_probs, axis=1)
		return mll[0] if single_trial else mll
import jax
import jax.numpy as jnp
from jax import vmap
import optax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions
tfb = tfp.bijectors

class GaussianModel:

	def __init__(self, output_dim=2, jitter=1e-5):
		self.D = output_dim
		self.jitter = jitter

		# bijector to take a flat vector → a valid lower-triangular matrix with strictly positive diag
		self._fill_tril = tfb.FillScaleTriL(
			diag_shift=self.jitter
		)

	def initialize(self, key, method="prior"):
		k1, k2 = jax.random.split(key)
		means = jax.random.normal(k1, (self.D,))
		# how many entries in a D×D lower-triangular? D*(D+1)//2
		n_tril = (self.D * (self.D + 1)) // 2
		raw_tril = jax.random.normal(k2, (n_tril,))
		params = {"means": means, "raw_tril": raw_tril}
		return params, {}

	def distribution(self, params):
		means   = params["means"]
		raw_tril = params["raw_tril"]
		# Turn flat vector → lower‑triangular L
		L = self._fill_tril.forward(raw_tril)            # shape (D, D), L lower-tri
		Sigma = L @ L.T                                  # full PD covariance
		return tfd.MultivariateNormalFullCovariance(
			loc=means,
			covariance_matrix=Sigma
		)

	def neg_log_likelihood(self, params, emissions):
		dist = self.distribution(params)
		return -jnp.sum(dist.log_prob(emissions))

	def fit_em(self, init_params, props, emissions, num_iters=100, lr=1e-2, verbose=True):
		optimizer = optax.adam(lr)
		opt_state = optimizer.init(init_params)
		lls = []

		@jax.jit
		def step(params, opt_state, batch):
			loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, batch)
			updates, opt_state = optimizer.update(grads, opt_state, params)
			new_params = optax.apply_updates(params, updates)
			return new_params, opt_state, loss

		params = init_params
		for _ in range(num_iters):
			params, opt_state, loss = step(params, opt_state, emissions)
			lls.append(-loss)
		return params, lls

	# def marginal_log_prob(self, params, emissions):
	#     return self.distribution(params).log_prob(emissions)
	
	def marginal_log_prob(self, params, emissions):
		if emissions.ndim == 2:
			emissions = jnp.expand_dims(emissions, axis=0)
			single = True
		else:
			single = False

		dist = self.distribution(params) # tfd.MultivariateNormalFullCovariance
		# for (n_trials, T, D), log_prob → (n_trials, T)
		log_ps = dist.log_prob(emissions)
		# sum over time → (n_trials,)
		mll = jnp.sum(log_ps, axis=1)
		return mll[0] if single else mll



# class GaussianModel:
	
# 	def __init__(self, output_dim=2):
# 		self.output_dim = output_dim
# 		self.params = None
		
# 	def initialize(self, key, method="prior"):
# 		key, subkey = jax.random.split(key)
# 		means = jax.random.normal(subkey, shape=(self.output_dim,))
# 		key, subkey = jax.random.split(key)
# 		covs = jax.random.uniform(subkey, shape=(self.output_dim,), minval=0.1, maxval=0.9)
# 		params = {"means": means, "covs": covs}
# 		props = {}  # No extra properties for now.
# 		self.params = params
# 		return params, props
	
# 	def distribution(self, params):
# 		jitter = 1e-5 * jnp.eye(self.output_dim)
# 		cov = params["covs"] + jitter
# 		return tfd.MultivariateNormalFullCovariance(
# 			params["means"], cov)
		
# 	def neg_log_likelihood(self, params, emissions):
# 		dist = self.distribution(params)
# 		return -jnp.sum(dist.log_prob(emissions))

# 	def fit_em(self, init_params, props, emissions, num_iters, learning_rate=0.01, verbose=True):

# 		optimizer = optax.adam(learning_rate)
# 		opt_state = optimizer.init(init_params)
# 		lls = []

# 		@jax.jit
# 		def step(params, opt_state, emissions):
# 			loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, emissions)
# 			updates, opt_state = optimizer.update(grads, opt_state, params)
# 			new_params = optax.apply_updates(params, updates)
# 			return new_params, opt_state, loss

# 		params = init_params
# 		for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting"):
# 			params, opt_state, loss = step(params, opt_state, emissions)
# 			lls.append(-loss)  # Record log likelihood
# 		return params, lls
	
# 	def marginal_log_prob(self, params, emissions):
# 		if emissions.ndim == 2:
# 			emissions = jnp.expand_dims(emissions, axis=0)
# 			single_trial = True
# 		else:
# 			single_trial = False

# 		dist = self.distribution(params)
# 		log_probs = dist.log_prob(emissions)
# 		mll = jnp.sum(log_probs, axis=1)
# 		return mll[0] if single_trial else mll
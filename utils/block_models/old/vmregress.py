import jax
import jax.numpy as jnp
from jax import vmap
import optax
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd = tfp.distributions

# def distribution(self, params, inputs):
#     prediction = params.weights @ inputs
#     prediction +=  params.biases
#     base_dist = tfd.VonMises(prediction, 1 / params.covs)
#     return tfd.Independent(base_dist, reinterpreted_batch_ndims=2)

class VonMisesRegressor:
    def __init__(self, input_dim=3, output_dim=1):
        """
        A JAX-based Von Mises regressor.
          - input_dim: Dimensionality of inputs.
          - output_dim: Dimensionality of outputs.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        # We do not initialize parameters here; use initialize() instead.
        self.params = None

    def initialize(self, key, method="prior"):
        """
        Initialize the model parameters.
        For now, only "prior" initialization is supported.
        Returns:
          - params: Dictionary with "weights", "biases", "covs"
          - props: Additional properties (empty dict for now)
        """
        key, subkey = jax.random.split(key)
        weights = jax.random.normal(subkey, shape=(self.input_dim, self.output_dim))
        key, subkey = jax.random.split(key)
        biases = jax.random.normal(subkey, shape=(self.output_dim,))
        key, subkey = jax.random.split(key)
        covs = jax.random.uniform(subkey, shape=(self.output_dim,), minval=0.1, maxval=1.0)
        params = {"weights": weights, "biases": biases, "covs": covs}
        props = {}  # No extra properties for now.
        self.params = params
        return params, props

    def distribution(self, params, inputs):
        """
        Compute the predicted Von Mises distribution.
        Args:
          - params: Dictionary containing "weights", "biases", and "covs"
          - inputs: jnp.array of shape (n, input_dim)
        Returns:
          - A tfd.VonMises distribution with location given by the linear predictor.
        """
        # Linear prediction: shape (n, output_dim)
        prediction = jnp.dot(inputs, params["weights"]) + params["biases"]
        # Compute concentration as 1/covs (ensuring positivity)
        concentration = 1.0 / jnp.maximum(params["covs"], 1e-6)
        concentration = jnp.broadcast_to(concentration, prediction.shape)
        return tfd.VonMises(loc=prediction, concentration=concentration)

    def neg_log_likelihood(self, params, inputs, emissions):
        """
        Compute the negative log-likelihood of the observed emissions given inputs.
        Args:
          - params: Dictionary of model parameters.
          - inputs: jnp.array of shape (n, input_dim)
          - emissions: jnp.array of shape (n, output_dim)
        Returns:
          - Scalar negative log-likelihood.
        """
        dist = self.distribution(params, inputs)
        return -jnp.sum(dist.log_prob(emissions))

    def fit_em(self, init_params, props, emissions, num_iters, inputs, learning_rate=0.01, verbose=True):
        """
        Fit the model parameters by minimizing the negative log-likelihood.
        Args:
          - init_params: Initial parameters (dictionary).
          - props: Additional properties (unused here).
          - emissions: jnp.array of shape (n, output_dim)
          - inputs: jnp.array of shape (n, input_dim)
          - num_iters: Number of optimization iterations.
          - learning_rate: Learning rate for the optimizer.
          - verbose: If True, show progress.
        Returns:
          - fit_params: Fitted parameters (dictionary).
          - lls: List of log-likelihood values (one per iteration).
        """
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(init_params)
        lls = []

        @jax.jit
        def step(params, opt_state, inputs, emissions):
            loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, inputs, emissions)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss

        params = init_params
        for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting"):
            params, opt_state, loss = step(params, opt_state, inputs, emissions)
            lls.append(-loss)  # Record log likelihood
        return params, lls

    def marginal_log_prob(self, fit_params, emissions, inputs):
        """
        Compute the (marginal) log probability of emissions given inputs using fitted parameters.
        In this model (with no latent variables), this is equivalent to the sum of log probabilities.
        Args:
          - fit_params: Dictionary of fitted parameters.
          - emissions: jnp.array of shape (n, output_dim)
          - inputs: jnp.array of shape (n, input_dim)
        Returns:
          - Scalar log probability.
        """
        dist = self.distribution(fit_params, inputs)
        return jnp.sum(dist.log_prob(emissions))


class VonMisesTimeRegressor:
	def __init__(self, input_dim=3, output_dim=1):
		"""
		A JAX-based Von Mises regressor.
			- input_dim: Dimensionality of inputs.
			- output_dim: Dimensionality of outputs.
		"""
		self.input_dim = input_dim
		self.output_dim = output_dim
		# We do not initialize parameters here; use initialize() instead.
		self.params = None

	def initialize(self, key, method="prior"):
		"""
		Initialize the model parameters.
		For now, only "prior" initialization is supported.
		Returns:
			- params: Dictionary with "weights", "biases", "covs"
			- props: Additional properties (empty dict for now)
		"""
		key, subkey = jax.random.split(key)
		weights = jax.random.normal(subkey, shape=(self.input_dim, self.output_dim))
		key, subkey = jax.random.split(key)
		biases = jax.random.normal(subkey, shape=(self.output_dim,))
		key, subkey = jax.random.split(key)
		covs = jax.random.uniform(subkey, shape=(self.output_dim,), minval=0.1, maxval=1.0)
		params = {"weights": weights, "biases": biases, "covs": covs}
		props = {}  # No extra properties for now.
		self.params = params
		return params, props

	def distribution(self, params, inputs):
		"""
		Compute the predicted Von Mises distribution.
		Args:
			- params: Dictionary containing "weights", "biases", and "covs"
			- inputs: jnp.array of shape (n_trials, T, input_dim)
		Returns:
			- Returns a VonMises distribution where the location is computed for each time step.
		"""
		# Use einsum to compute a linear prediction for each trial and each timestep.
		# prediction shape: (n_trials, T, output_dim)
		prediction = jnp.einsum('ntd,dk->ntk', inputs, params["weights"]) + params["biases"]
		# Compute concentration for each output dimension, then broadcast to (n_trials, T, output_dim)
		concentration = 1.0 / jnp.maximum(params["covs"], 1e-6)
		concentration = jnp.broadcast_to(concentration, prediction.shape)
		return tfd.VonMises(loc=prediction, concentration=concentration)

	def neg_log_likelihood(self, params, inputs, emissions):
		"""
		Compute the negative log-likelihood of the observed emissions given inputs.
		Args:
			- params: Dictionary of model parameters.
			- inputs: jnp.array of shape (n, input_dim)
			- emissions: jnp.array of shape (n, output_dim)
		Returns:
			- Scalar negative log-likelihood.
		"""
		dist = self.distribution(params, inputs)
		return -jnp.sum(dist.log_prob(emissions))

	def fit_em(self, init_params, props, emissions, num_iters, inputs, learning_rate=0.01, verbose=True):
		"""
		Fit the model parameters by minimizing the negative log-likelihood.
		Args:
			- init_params: Initial parameters (dictionary).
			- props: Additional properties (unused here).
			- emissions: jnp.array of shape (n, output_dim)
			- inputs: jnp.array of shape (n, input_dim)
			- num_iters: Number of optimization iterations.
			- learning_rate: Learning rate for the optimizer.
			- verbose: If True, show progress.
		Returns:
			- fit_params: Fitted parameters (dictionary).
			- lls: List of log-likelihood values (one per iteration).
		"""
		optimizer = optax.adam(learning_rate)
		opt_state = optimizer.init(init_params)
		lls = []

		@jax.jit
		def step(params, opt_state, inputs, emissions):
				loss, grads = jax.value_and_grad(self.neg_log_likelihood)(params, inputs, emissions)
				updates, opt_state = optimizer.update(grads, opt_state, params)
				new_params = optax.apply_updates(params, updates)
				return new_params, opt_state, loss

		params = init_params
		for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting"):
				params, opt_state, loss = step(params, opt_state, inputs, emissions)
				lls.append(-loss)  # Record log likelihood
		return params, lls

	def marginal_log_prob(self, params, emissions, inputs):
		"""
		Compute the marginal log probability of emissions given inputs.
		
		This function handles both batched inputs (shape: (n_trials, T, ...))
		and single-trial inputs (shape: (T, ...)).
		
		Args:
		- params: Dictionary of fitted parameters.
		- emissions: jnp.array, shape (T, output_dim) or (n_trials, T, output_dim)
		- inputs: jnp.array, shape (T, input_dim) or (n_trials, T, input_dim)
		
		Returns:
		- If batched: jnp.array of shape (n_trials,) containing the per-trial marginal log likelihood.
		- If single-trial: a scalar marginal log likelihood.
		"""
		# Check if inputs are single-trial (i.e. 2D instead of 3D)
		if inputs.ndim == 2:
			inputs = jnp.expand_dims(inputs, axis=0)
			emissions = jnp.expand_dims(emissions, axis=0)
			single_trial = True
		else:
			single_trial = False

		dist = self.distribution(params, inputs)
		log_probs = dist.log_prob(emissions)
		mll = jnp.sum(log_probs, axis=1)
		return mll[0] if single_trial else mll

from jax import numpy as jnp

def calc_aic_bic_ghmm(n_state, log_likelihood, emission_dim, T):

  m = n_state
  k = emission_dim + (emission_dim*(emission_dim+1))/2

  p = m**2 + k*m - 1

  aic = -2*log_likelihood + 2*p
  bic = -2*log_likelihood + p*jnp.log(T)

  return aic, bic

def calc_aic_bic_vhmm(n_state, log_likelihood, input_dim, output_dim, T):

  if n_state == 1:
    return calc_aic_bic_regressor(log_likelihood, input_dim, output_dim, T)

  m = n_state
  # Transition matrix: m*(m-1) free parameters
  # Initial state: m - 1 free parameters
  transition_initial = m**2 - 1
  # Emission parameters per state:
  k_prime = input_dim * output_dim + 2 * output_dim
  p = transition_initial + m * k_prime
  aic = -2 * log_likelihood + 2 * p
  bic = -2 * log_likelihood + p * jnp.log(T)
  return aic, bic

def calc_aic_bic_regressor(log_likelihood, input_dim, output_dim, T):
  p = input_dim * output_dim + 2 * output_dim  # only emission parameters
  aic = -2 * log_likelihood + 2 * p
  bic = -2 * log_likelihood + p * jnp.log(T)
  return aic, bic
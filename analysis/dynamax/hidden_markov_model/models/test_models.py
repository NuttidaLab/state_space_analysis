import pytest
from datetime import datetime
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import dynamax.hidden_markov_model as models
from dynamax.utils.utils import monotonically_increasing


NUM_TIMESTEPS = 50

CONFIGS = [
    (models.BernoulliHMM, dict(num_states=4, emission_dim=3), None),
    (models.CategoricalHMM, dict(num_states=4, emission_dim=3, num_classes=5), None),
    (models.CategoricalRegressionHMM, dict(num_states=4, num_classes=3, input_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.GammaHMM, dict(num_states=4), None),
    (models.GaussianHMM, dict(num_states=4, emission_dim=3, emission_prior_concentration=1.0, emission_prior_scale=1.0), None),
    (models.DiagonalGaussianHMM, dict(num_states=4, emission_dim=3), None),
    (models.SphericalGaussianHMM, dict(num_states=4, emission_dim=3), None),
    (models.SharedCovarianceGaussianHMM, dict(num_states=4, emission_dim=3), None),
    (models.LowRankGaussianHMM, dict(num_states=4, emission_dim=3, emission_rank=1), None),
    (models.GaussianMixtureHMM, dict(num_states=4, num_components=2, emission_dim=3, emission_prior_mean_concentration=1.0), None),
    (models.DiagonalGaussianMixtureHMM, dict(num_states=4, num_components=2, emission_dim=3, emission_prior_mean_concentration=1.0), None),
    (models.LinearRegressionHMM, dict(num_states=4, emission_dim=3, input_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.CircularRegressionHMM, dict(num_states=4, emission_dim=3, input_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.OCCircularRegressionHMM, dict(num_states=4, emission_dim=3, input_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.LogisticRegressionHMM, dict(num_states=4, input_dim=5), jnp.ones((NUM_TIMESTEPS, 5))),
    (models.MultinomialHMM, dict(num_states=4, emission_dim=3, num_classes=5, num_trials=10), None),
    (models.PoissonHMM, dict(num_states=4, emission_dim=3), None),
]


@pytest.mark.parametrize(["cls", "kwargs", "inputs"], CONFIGS)
def test_sample_and_fit(cls, kwargs, inputs):
    hmm = cls(**kwargs)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(42))
    params, param_props = hmm.initialize(key1)
    states, emissions = hmm.sample(params, key2, num_timesteps=NUM_TIMESTEPS, inputs=inputs)
    fitted_params, lps = hmm.fit_em(params, param_props, emissions, inputs=inputs, num_iters=10)
    assert monotonically_increasing(lps, atol=1e-2, rtol=1e-2)
    fitted_params, lps = hmm.fit_sgd(params, param_props, emissions, inputs=inputs, num_epochs=10)


## A few model-specific tests
def test_categorical_hmm_viterbi():
    # From http://en.wikipedia.org/wiki/Viterbi_algorithm:
    hmm = models.CategoricalHMM(num_states=2, emission_dim=1, num_classes=3)
    params, props = hmm.initialize(
        jr.PRNGKey(0),
        initial_probs=jnp.array([0.6, 0.4]),
        transition_matrix=jnp.array([[0.7, 0.3], [0.4, 0.6]]),
        emission_probs=jnp.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]).reshape(2, 1, 3))

    emissions = jnp.arange(3).reshape(3, 1)
    state_sequence = hmm.most_likely_states(params, emissions)
    assert jnp.allclose(jnp.squeeze(state_sequence), jnp.array([1, 0, 0]))


def test_gmm_hmm_vs_gmm_diag_hmm(key=jr.PRNGKey(0), num_states=4, num_components=3, emission_dim=2):
    key1, key2, key3 = jr.split(key, 3)
    diag_hmm = models.DiagonalGaussianMixtureHMM(num_states, num_components, emission_dim)
    diag_params, _ = diag_hmm.initialize(key1)

    full_hmm = models.GaussianMixtureHMM(num_states, num_components, emission_dim)
    emission_covariances = vmap(lambda ss: vmap(lambda s: jnp.diag(s**2))(ss))(diag_params.emissions.scale_diags)
    full_params, _ = full_hmm.initialize(key2,
        initial_probs=diag_params.initial.probs,
        transition_matrix=diag_params.transitions.transition_matrix,
        emission_weights=diag_params.emissions.weights,
        emission_means=diag_params.emissions.means,
        emission_covariances=emission_covariances)

    states_diag, emissions_diag = diag_hmm.sample(diag_params, key3, NUM_TIMESTEPS)
    states_full, emissions_full = full_hmm.sample(full_params, key3, NUM_TIMESTEPS)
    assert jnp.allclose(emissions_full, emissions_diag)
    assert jnp.allclose(states_full, states_diag)

    posterior_diag = diag_hmm.smoother(diag_params, emissions_diag)
    posterior_full = full_hmm.smoother(full_params, emissions_full)

    assert jnp.allclose(posterior_diag.marginal_loglik, posterior_full.marginal_loglik)
    assert jnp.allclose(posterior_diag.filtered_probs, posterior_full.filtered_probs)
    assert jnp.allclose(posterior_diag.predicted_probs, posterior_full.predicted_probs)
    assert jnp.allclose(posterior_diag.smoothed_probs, posterior_full.smoothed_probs)
    assert jnp.allclose(posterior_diag.initial_probs, posterior_full.initial_probs)

    states_diag = diag_hmm.most_likely_states(diag_params, emissions_diag)
    states_full = full_hmm.most_likely_states(full_params, emissions_full)
    assert jnp.allclose(states_full, states_diag)


def test_sample_and_fit_arhmm():
    arhmm = models.LinearAutoregressiveHMM(num_states=4, emission_dim=2, num_lags=1)
    #key1, key2 = jr.split(jr.PRNGKey(int(datetime.now().timestamp())))
    key1, key2 = jr.split(jr.PRNGKey(42))
    params, param_props = arhmm.initialize(key1)
    states, emissions = arhmm.sample(params, key2, num_timesteps=NUM_TIMESTEPS)
    inputs = arhmm.compute_inputs(emissions)
    fitted_params, lps = arhmm.fit_em(params, param_props, emissions, inputs=inputs, num_iters=10)
    assert monotonically_increasing(lps, atol=1e-2, rtol=1e-2)
    fitted_params, lps = arhmm.fit_sgd(params, param_props, emissions, inputs=inputs, num_epochs=10)
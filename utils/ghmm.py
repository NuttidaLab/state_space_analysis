"""Gaussian Hidden Markov Models with various constraints on the covariance matrix."""
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
from jax import vmap
import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jaxtyping import Float, Array
import optax

from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.inference import HMMPosterior
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import IntScalar, Scalar
from dynamax.utils.distributions import InverseWishart
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart
from dynamax.utils.distributions import nig_posterior_update
from dynamax.utils.distributions import niw_posterior_update
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import pytree_sum


class ParamsGaussianHMMEmissions(NamedTuple):
    """Parameters for Gaussian emissions in an HMM."""
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class GaussianHMMEmissions(HMMEmissions):
    """Gaussian emissions for an HMM.

    Args:
        num_states: number of discrete states
        emission_dim: dimension of the emission vector
        emission_prior_mean: prior mean for emissions
        emission_prior_concentration: concentration parameter for the normal inverse Wishart prior
        emission_prior_scale: scale matrix for the normal inverse Wishart  prior
        emission_prior_extra_df: extra degrees of freedom for the normal inverse Wishart prior
    """
    def __init__(self,
                 num_states: int,
                 emission_dim: int,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]] = 0.0,
                 emission_prior_concentration: Scalar = 1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]] = 1e-4,
                 emission_prior_extra_df: Scalar = 0.1):
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
                else emission_prior_scale * jnp.eye(emission_dim)
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self) -> Tuple[int]:
        """Shape of the emission vector."""
        return (self.emission_dim,)

    def distribution(
            self, 
            params: ParamsGaussianHMMEmissions,
            state: IntScalar,
            inputs: Optional[Array] = None
            ) -> tfd.Distribution:
        """Return the emission distribution for a given state."""
        # Add jitter to the covariance for numerical stability:
        jitter = 1e-6 * jnp.eye(self.emission_dim)
        cov = params.covs[state] + jitter
        return tfd.MultivariateNormalFullCovariance(
            params.means[state], cov)

    def log_prior(self, params: ParamsGaussianHMMEmissions) -> Float[Array, ""]:
        """Return the log prior probability of the emission parameters."""
        return NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params.covs, params.means)).sum()

    def initialize(self, 
                   key: Array = jr.PRNGKey(0),
                   method: str = "prior",
                   emission_means: Optional[Float[Array, "num_states emission_dim"]] = None,
                   emission_covariances: Optional[Float[Array, "num_states emission_dim emission_dim"]] = None,
                   emissions: Optional[Float[Array, "num_timesteps emission_dim"]] = None
        ) -> Tuple[ParamsGaussianHMMEmissions, ParamsGaussianHMMEmissions]:
        """Initialize the model parameters and their corresponding properties.

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            emission_means: manually specified emission means.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Tuple of (params, props) where params are the initialized parameters and props are their properties.
        """
        if method.lower() == "kmeans":
            assert emissions is not None, "Need emissions to initialize the model with K-Means!"
            from sklearn.cluster import KMeans
            key, subkey = jr.split(key)  # Create a random seed for SKLearn.
            sklearn_key = jr.randint(subkey, shape=(), minval=0, maxval=2147483647)  # Max int32 value.
            km = KMeans(self.num_states, random_state=int(sklearn_key)).fit(emissions.reshape(-1, self.emission_dim))

            _emission_means = jnp.array(km.cluster_centers_)
            _emission_covs = jnp.tile(jnp.eye(self.emission_dim)[None, :, :], (self.num_states, 1, 1))

        elif method.lower() == "prior":
            this_key, key = jr.split(key)
            prior = NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                         self.emission_prior_df, self.emission_prior_scale)
            (_emission_covs, _emission_means) = prior.sample(seed=this_key, sample_shape=(self.num_states,))

        else:
            raise Exception("Invalid initialization method: {}".format(method))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsGaussianHMMEmissions(
            means=default(emission_means, _emission_means),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsGaussianHMMEmissions(
            means=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def collect_suff_stats(
            self, 
            params: ParamsGaussianHMMEmissions, 
            posterior: HMMPosterior,
            emissions: Float[Array, "num_timesteps emission_dim"], 
            inputs: Optional[Array] = None
        ) -> Dict[str, Float[Array, "..."]]:
        """Collect sufficient statistics for the M-step of the EM algorithm."""
        expected_states = posterior.smoothed_probs
        return dict(
            sum_w=jnp.einsum("tk->k", expected_states),
            sum_x=jnp.einsum("tk,ti->ki", expected_states, emissions),
            sum_xxT=jnp.einsum("tk,ti,tj->kij", expected_states, emissions, emissions)
        )

    def initialize_m_step_state(self, params: ParamsGaussianHMMEmissions, 
                                props: ParamsGaussianHMMEmissions) -> None:
        """Initialize the M-step state."""
        return None

    def m_step(
            self, 
            params: ParamsGaussianHMMEmissions, 
            props: ParamsGaussianHMMEmissions, 
            batch_stats: Dict[str, Float[Array, "..."]], 
            m_step_state: Any
        ) -> Tuple[ParamsGaussianHMMEmissions, Any]:
        """Perform the M-step of the EM algorithm."""
        if props.covs.trainable and props.means.trainable:
            niw_prior = NormalInverseWishart(loc=self.emission_prior_mean,
                                            mean_concentration=self.emission_prior_conc,
                                            df=self.emission_prior_df,
                                            scale=self.emission_prior_scale)

            # Find the posterior parameters of the NIW distribution
            def _single_m_step(stats):
                """Perform the M-step for a single state."""
                niw_posterior = niw_posterior_update(niw_prior, (stats['sum_x'], stats['sum_xxT'], stats['sum_w']))
                return niw_posterior.mode()

            emission_stats = pytree_sum(batch_stats, axis=0)
            covs, means = vmap(_single_m_step)(emission_stats)
            params = params._replace(means=means, covs=covs)

        elif props.covs.trainable and not props.means.trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed means and trainable covariance")

        elif not props.covs.trainable and props.means.trainable:
            raise NotImplementedError("GaussianHMM.fit_em() does not yet support fixed covariance and trainable means")

        return params, m_step_state
    

### Now for the models ###
class ParamsGaussianHMM(NamedTuple):
    """Parameters for a Gaussian HMM."""
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsGaussianHMMEmissions


class GaussianHMM(HMM):
    r"""An HMM with multivariate normal (i.e. Gaussian) emissions.

    Let $y_t \in \mathbb{R}^N$ denote a vector-valued emissions at time $t$. In this model,
    the emission distribution is,

    $$p(y_t \mid z_t, \theta) = \mathcal{N}(y_{t} \mid \mu_{z_t}, \Sigma_{z_t})$$

    with $\theta = \{\mu_k, \Sigma_k\}_{k=1}^K$ denoting the *emission means* and *emission covariances*.

    The model has a conjugate normal-inverse-Wishart_ prior,

    $$p(\theta) = \prod_{k=1}^K \mathcal{N}(\mu_k \mid \mu_0, \kappa_0^{-1} \Sigma_k) \mathrm{IW}(\Sigma_{k} \mid \nu_0, \Psi_0)$$

    .. _normal-inverse-Wishart: https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution

    :param num_states: number of discrete states $K$
    :param emission_dim: number of conditionally independent emissions $N$
    :param initial_probs_concentration: $\alpha$
    :param transition_matrix_concentration: $\beta$
    :param transition_matrix_stickiness: optional hyperparameter to boost the concentration on the diagonal of the transition matrix.
    :param emission_prior_mean: $\mu_0$
    :param emission_prior_concentration: $\kappa_0$
    :param emission_prior_extra_df: $\nu_0 - N > 0$, the "extra" degrees of freedom, above and beyond the minimum of $\\nu_0 = N$.
    :param emission_prior_scale: $\Psi_0$

    """
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, " num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, " emission_dim"]]=0.0,
                 emission_prior_concentration: Scalar=1e-4,
                 emission_prior_scale: Union[Scalar, Float[Array, "emission_dim emission_dim"]]=1e-4,
                 emission_prior_extra_df: Scalar=0.1):
        self.emission_dim = emission_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = GaussianHMMEmissions(num_states, emission_dim,
                                                  emission_prior_mean=emission_prior_mean,
                                                  emission_prior_concentration=emission_prior_concentration,
                                                  emission_prior_scale=emission_prior_scale,
                                                  emission_prior_extra_df=emission_prior_extra_df)

        super().__init__(num_states, initial_component, transition_component, emission_component)

    def initialize(self,
                   key: Array=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, " num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        """Initialize the model parameters and their corresponding properties.

        You can either specify parameters manually via the keyword arguments, or you can have
        them set automatically. If any parameters are not specified, you must supply a PRNGKey.
        Parameters will then be sampled from the prior (if `method==prior`).

        Args:
            key: random number generator for unspecified parameters. Must not be None if there are any unspecified parameters.
            method: method for initializing unspecified parameters. Both "prior" and "kmeans" are supported.
            initial_probs: manually specified initial state probabilities.
            transition_matrix: manually specified transition matrix.
            emission_means: manually specified emission means.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(
            key1, method=method, initial_probs=initial_probs
        )
        params["transitions"], props["transitions"] = self.transition_component.initialize(
            key2, method=method, transition_matrix=transition_matrix
        )
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            key3, method=method, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions
        )
        return ParamsGaussianHMM(**params), ParamsGaussianHMM(**props)
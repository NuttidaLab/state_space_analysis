import jax.numpy as jnp
import jax.random as jr
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import vmap
from jaxtyping import Float, Array
import optax
from dynamax.parameters import ParameterProperties
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.types import Scalar
from dynamax.utils.distributions import InverseWishart
from dynamax.utils.distributions import NormalInverseGamma
from dynamax.utils.distributions import NormalInverseWishart
from dynamax.utils.distributions import nig_posterior_update
from dynamax.utils.distributions import niw_posterior_update
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.utils.utils import pytree_sum
from typing import NamedTuple, Optional, Tuple, Union


class ParamsGaussianHMMEmissions(NamedTuple):
    means: Union[Float[Array, "state_dim emission_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class GaussianHMMEmissions(HMMEmissions):

    def __init__(self,
                 num_states,
                 emission_dim,
                 emission_prior_mean=0.0,
                 emission_prior_concentration=1e-4,
                 emission_prior_scale=1e-4,
                 emission_prior_extra_df=0.1,
                 m_step_optimizer=optax.adam(1e-3),
                 m_step_num_iters=50
                 ):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.emission_dim = emission_dim
        self.emission_prior_mean = emission_prior_mean * jnp.ones(emission_dim)
        self.emission_prior_conc = emission_prior_concentration
        self.emission_prior_scale = emission_prior_scale if jnp.ndim(emission_prior_scale) == 2 \
                else emission_prior_scale * jnp.eye(emission_dim)
        self.emission_prior_df = emission_dim + emission_prior_extra_df

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, inputs=None):
        return tfd.MultivariateNormalFullCovariance(
            params.means[state], params.covs[state])

    def log_prior(self, params):
        return NormalInverseWishart(self.emission_prior_mean, self.emission_prior_conc,
                                   self.emission_prior_df, self.emission_prior_scale).log_prob(
            (params.covs, params.means)).sum()

    def initialize(self, key=jr.PRNGKey(0),
                   method="prior",
                   emission_means=None,
                   emission_covariances=None,
                   emissions=None):
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

### Now for the models ###
class ParamsGaussianHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsGaussianHMMEmissions

class GaussianHMM(HMM):
    def __init__(self, num_states: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0,
                 emission_prior_mean: Union[Scalar, Float[Array, "emission_dim"]]=0.0,
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
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_means: Optional[Float[Array, "num_states emission_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:

        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emission_means=emission_means, emission_covariances=emission_covariances, emissions=emissions)
        return ParamsGaussianHMM(**params), ParamsGaussianHMM(**props)

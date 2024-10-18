import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Float, Array
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet, HMMTransitions
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Optional, Tuple, Union

tfd = tfp.distributions
tfb = tfp.bijectors

class ParamsET_Emissions(NamedTuple):
    weights: Union[Float[Array, "state_dim emission_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]


class ET_Emissions(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim,
                 emission_dim):
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim

    def initialize(self,
                   key=jr.PRNGKey(0),
                   emission_weights=None,
                   emission_covariances=None):

        key1, key2, key = jr.split(key, 3)
        _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.input_dim))
        # _emission_covs = 0.01 * jr.normal(key2, (self.num_states))
        _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsET_Emissions(
            weights=default(emission_weights, _emission_weights),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsET_Emissions(
            weights=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def distribution(self, params, state, inputs):
        return tfd.VonMises(params.weights[state] @ inputs, params.covs[state])

    def log_prior(self, params):
        return 0.0


class ParamsET_Transitions(NamedTuple):
    transition_matrix: Union[Float[Array, "state_dim state_dim"], ParameterProperties]
    transition_weights: Union[Float[Array, "state_dim input_dim"], ParameterProperties]

class ET_Transitions(HMMTransitions):
    def __init__(self, num_states, input_dim, concentration=1.1, stickiness=0.0):
        self.num_states = num_states
        self.input_dim = input_dim
        self.concentration = \
            concentration * jnp.ones((num_states, num_states)) + \
                stickiness * jnp.eye(num_states)

    def initialize(self, key=None, transition_matrix=None, transition_weights=None):

        if transition_matrix is None:
            this_key, key = jr.split(key)
            transition_matrix = tfd.Dirichlet(self.concentration).sample(seed=this_key)

        if transition_weights is None:
            this_key, key = jr.split(key)
            transition_weights = 0.01 * jr.normal(this_key, (self.num_states, self.input_dim))

        # Package the results into dictionaries
        params = ParamsET_Transitions(transition_matrix=transition_matrix, transition_weights=transition_weights)
        props = ParamsET_Transitions(transition_matrix=ParameterProperties(constrainer=tfb.SoftmaxCentered()), transition_weights=ParameterProperties())
        return params, props

    def distribution(self, params, state, inputs=None):
        logits = params.transition_matrix[state] + params.transition_weights[state] @ inputs
        return tfd.Categorical(logits = logits)

    def log_prior(self, params):
        return tfd.Dirichlet(self.concentration).log_prob(params.transition_matrix).sum()

    # def _compute_transition_matrices(self, params, inputs=None):
    #     return params.transition_matrix

    def collect_suff_stats(self, params, posterior, inputs=None):
        return posterior.trans_probs


class ParamsET_HMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsET_Transitions
    emissions: ParamsET_Emissions

class ET_HMM(HMM):
    def __init__(self,
                 num_states: int,
                 input_dim: int,
                 emission_dim: int,
                 initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]]=1.1,
                 transition_matrix_stickiness: Scalar=0.0):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial_component = StandardHMMInitialState(num_states, initial_probs_concentration=initial_probs_concentration)
        transition_component = ET_Transitions(num_states, input_dim, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = ET_Emissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   transition_weights: Optional[Float[Array, "num_states input_dim"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
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
            emission_weights: manually specified emission weights.
            emission_covariances: manually specified emission covariances.
            emissions: emissions for initializing the parameters with kmeans.

        Returns:
            Model parameters and their properties.

        """
        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, transition_matrix=transition_matrix, transition_weights=transition_weights)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, emission_weights=emission_weights, emission_covariances=emission_covariances)
        return ParamsET_HMM(**params), ParamsET_HMM(**props)

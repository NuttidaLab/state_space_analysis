import jax.numpy as jnp
import jax.random as jr
from jax import vmap
from jaxtyping import Float, Array
from dynamax.hidden_markov_model.models.abstractions import HMM, HMMEmissions, HMMParameterSet, HMMPropertySet
from dynamax.hidden_markov_model.models.initial import StandardHMMInitialState, ParamsStandardHMMInitialState
from dynamax.hidden_markov_model.models.transitions import StandardHMMTransitions, ParamsStandardHMMTransitions
from dynamax.parameters import ParameterProperties
from dynamax.types import Scalar
from dynamax.utils.utils import pytree_sum
from dynamax.utils.bijectors import RealToPSDBijector
from tensorflow_probability.substrates import jax as tfp
from typing import NamedTuple, Optional, Tuple, Union

tfd = tfp.distributions
tfb = tfp.bijectors

class ParamsE_HMMEmissions(NamedTuple):
    weights: Union[Float[Array, "state_dim emission_dim input_dim"], ParameterProperties]
    covs: Union[Float[Array, "state_dim emission_dim emission_dim"], ParameterProperties]

class E_HMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim,
                 emission_dim):
        
        self.num_states = num_states
        self.input_dim = input_dim
        self.emission_dim = emission_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   emission_weights=None,
                   emission_covariances=None):

        key1, key2, key = jr.split(key, 3)
        _emission_weights = 0.01 * jr.normal(key1, (self.num_states, self.emission_dim, self.input_dim))
        _emission_covs = jnp.tile(jnp.eye(self.emission_dim), (self.num_states, 1, 1))

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0
        params = ParamsE_HMMEmissions(
            weights=default(emission_weights, _emission_weights),
            covs=default(emission_covariances, _emission_covs))
        props = ParamsE_HMMEmissions(
            weights=ParameterProperties(),
            covs=ParameterProperties(constrainer=RealToPSDBijector()))
        return params, props

    def distribution(self, params, state, inputs = None):
        # print("inputs: ", inputs)
        # print("params.weights[state]: ", params.weights[state])
        # print("params.weights[state] @ inputs: ", params.weights[state] @ inputs)
        # print("params.covs[state]: ", params.covs[state])
        return tfd.VonMises(params.weights[state] @ inputs, params.covs[state])

    def log_prior(self, params):
        return 0.0


class ParamsE_HMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsE_HMMEmissions


class E_HMM(HMM):
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
        transition_component = StandardHMMTransitions(num_states, concentration=transition_matrix_concentration, stickiness=transition_matrix_stickiness)
        emission_component = E_HMMEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   emission_weights: Optional[Float[Array, "num_states emission_dim input_dim"]]=None,
                   emission_covariances:  Optional[Float[Array, "num_states emission_dim emission_dim"]]=None,
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:

        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, emission_weights=emission_weights, emission_covariances=emission_covariances)
        return ParamsE_HMM(**params), ParamsE_HMM(**props)

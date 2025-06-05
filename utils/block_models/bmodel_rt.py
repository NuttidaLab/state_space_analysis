
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn
from jax import vmap
import optax
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

# New emission parameterization reflecting final GLM/VM formulas
class ParamsBlockHMMrtEmissions(NamedTuple):
    # RT GLM: 1 + Error * (Attention + Coh + Exp) => intercept + 1 Error + 3 flags + 3 interactions = 8
    weights_rt:    Union[Float[Array, "num_states 8"], ParameterProperties]
    alpha_rt:      Union[Float[Array, "num_states 1"],   ParameterProperties]

class ParamsBlockHMMrt(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsBlockHMMrtEmissions

class BlockHMMrtEmissions(HMMEmissions):
    def __init__(self,
                 num_states: int,
                 input_dim: int = 8,       # 8(rt) + 7(ra) + 8(error)
                 emission_dim: int = 3,
                 m_step_optimizer=optax.adam(1e-3),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states   = num_states
        self.input_dim    = input_dim
        self.emission_dim = emission_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize(self,
                   key=jr.PRNGKey(0),
                   method="prior",
                   weights_rt=None, alpha_rt=None,
                   emissions=None):

        if method == "prior":
            # RT
            weights_rt = jnp.zeros((self.num_states, 8))
            alpha_rt   = jnp.ones((self.num_states, 1))
        
        params = ParamsBlockHMMrtEmissions(
            weights_rt, alpha_rt,
        )
        props = ParamsBlockHMMrtEmissions(
            ParameterProperties(),                           # weights_rt
            ParameterProperties(constrainer=tfb.Softplus()), # alpha_rt > 0
        )
        return params, props

    def distribution(self, params, state, inputs):
        # if inputs.ndim == 2:
        #     inputs = inputs[jnp.newaxis, ...]
        x_rt = inputs

        # 1) Gamma GLM for RT
        lp_rt = params.weights_rt[state] @ x_rt
        mu_rt = jnp.exp(lp_rt)
        base = tfd.Gamma(
                concentration=params.alpha_rt[state],
                rate=params.alpha_rt[state] / mu_rt
        )
        
        dist_rt = tfd.Independent(
            base,
            reinterpreted_batch_ndims=1
        )
        return dist_rt
    
    def log_prior(self, params):
        return 0.0

class BlockHMMrt(HMM):
    def __init__(
        self,
        num_states: int,
        input_dim: int = 8,  # 8(rt) + 7(ra) + 8(error)
        emission_dim: int = 3,
        initial_probs_concentration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_matrix_concentration: Union[Scalar, Float[Array, "num_states"]] = 1.1,
        transition_matrix_stickiness: Scalar = 0.0
    ):
        self.emission_dim = emission_dim
        self.input_dim = input_dim
        initial     = StandardHMMInitialState(
            num_states,
            initial_probs_concentration=initial_probs_concentration
        )
        transitions = StandardHMMTransitions(
            num_states,
            concentration=transition_matrix_concentration,
            stickiness=transition_matrix_stickiness
        )
        emissions   = BlockHMMrtEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial, transitions, emissions)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(
        self,
        key: jr.PRNGKey = jr.PRNGKey(0),
        method: str = "prior",
        initial_probs: Optional[Float[Array, "num_states"]] = None,
        transition_matrix: Optional[Float[Array, "num_states num_states"]] = None,
        # Emission init args:
        weights_rt: Optional[Float[Array, "num_states 8"]] = None,
        alpha_rt:   Optional[Float[Array, "num_states 1"]]   = None,
        
        emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
    ) -> Tuple[HMMParameterSet, HMMPropertySet]:
        # Split RNG
        # Initialize each component
        k1, k2, k3 = jr.split(key, 3)
        params, props = dict(), dict()
        
        params["initial"], props["initial"] = self.initial_component.initialize(k1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(k2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(
            k3, method=method,
            weights_rt=weights_rt, alpha_rt=alpha_rt,
            emissions=emissions
        )
        return ParamsBlockHMMrt(**params), ParamsBlockHMMrt(**props)

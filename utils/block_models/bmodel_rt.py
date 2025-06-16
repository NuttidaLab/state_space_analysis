
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
    w1:    Union[Float[Array, "num_states 8"], ParameterProperties] # weights
    w2:      Union[Float[Array, "num_states 1"],   ParameterProperties] # alpha

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
                   w1=None, w2=None,
                   emissions=None):

        if method == "prior":
            # RT
            w1 = jnp.zeros((self.num_states, 8))
            w2   = jnp.ones((self.num_states, 1))
        
        params = ParamsBlockHMMrtEmissions(
            w1, w2,
        )
        props = ParamsBlockHMMrtEmissions(
            ParameterProperties(),                           # w1
            ParameterProperties(constrainer=tfb.Softplus()), # w2 > 0
        )
        return params, props

    def distribution(self, params, state, inputs):
        # if inputs.ndim == 2:
        #     inputs = inputs[jnp.newaxis, ...]
        x_rt = inputs

        # 1) Gamma GLM for RT
        lp_rt = params.w1[state] @ x_rt
                
        # ─── clamp to a “safe” window [−20, +20] ─────────────────────────────────
        lp_rt_clamped = jnp.clip(lp_rt, a_min=-20.0, a_max=20.0)
        mu_rt         = jnp.exp(lp_rt_clamped)
        # ─── ensure the Gamma rate never becomes exactly 0 ─────────────────────────
        
        base = tfd.Gamma(
                concentration=params.w2[state],
                rate=params.w2[state] / mu_rt
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
        w1: Optional[Float[Array, "num_states 8"]] = None,
        w2:   Optional[Float[Array, "num_states 1"]]   = None,
        
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
            w1=w1, w2=w2,
            emissions=emissions
        )
        return ParamsBlockHMMrt(**params), ParamsBlockHMMrt(**props)

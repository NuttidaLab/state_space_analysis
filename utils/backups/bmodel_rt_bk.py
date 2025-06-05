
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
    alpha_rt:      Union[Float[Array, "num_states"],   ParameterProperties]
    # # RA mixture: 1 + stim + prev_stim + prev_resp + Attention + Coh + Exp = 7 features each
    # weights_ra1:   Union[Float[Array, "num_states 7"], ParameterProperties]
    # weights_ra2:   Union[Float[Array, "num_states 7"], ParameterProperties]
    # mixture_logits_ra: Union[Float[Array, "num_states 2"], ParameterProperties]
    # kappa1_ra:     Union[Float[Array, "num_states"],   ParameterProperties]
    # kappa2_ra:     Union[Float[Array, "num_states"],   ParameterProperties]
    # # Error GLM: 1 + rt * (Attention + Coh + Exp) => intercept + 1 rt + 3 flags + 3 interactions = 8
    # weights_re:    Union[Float[Array, "num_states 8"], ParameterProperties]
    # alpha_re:        Union[Float[Array, "num_states"],   ParameterProperties]

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
                #    weights_ra1=None, weights_ra2=None, mixture_logits_ra=None,
                #    kappa1_ra=None,  kappa2_ra=None,
                #    weights_re=None, alpha_re=None,
                   emissions=None):

        if method == "prior":
            # RT
            weights_rt = jnp.zeros((self.num_states, 8))
            alpha_rt   = jnp.ones((self.num_states,))
            # # RA mixture
            # weights_ra1       = jnp.zeros((self.num_states, 7))
            # weights_ra2       = jnp.zeros((self.num_states, 7))
            # mixture_logits_ra = jnp.zeros((self.num_states, 2))
            # kappa1_ra         = jnp.ones((self.num_states,))
            # kappa2_ra         = jnp.ones((self.num_states,))
            # # Error
            # weights_re = jnp.zeros((self.num_states, 8))
            # alpha_re     = jnp.ones((self.num_states,))
        
        params = ParamsBlockHMMrtEmissions(
            weights_rt, alpha_rt,
            # weights_ra1, weights_ra2, mixture_logits_ra, kappa1_ra, kappa2_ra,
            # weights_re, alpha_re
        )
        props = ParamsBlockHMMrtEmissions(
            ParameterProperties(),                           # weights_rt
            ParameterProperties(constrainer=tfb.Softplus()), # alpha_rt > 0
            # ParameterProperties(),                           # weights_ra1
            # ParameterProperties(),                           # weights_ra2
            # ParameterProperties(),                           # mixture_logits_ra
            # ParameterProperties(constrainer=tfb.Softplus()), # kappa1_ra > 0
            # ParameterProperties(constrainer=tfb.Softplus()), # kappa2_ra > 0
            # ParameterProperties(),                           # weights_re
            # ParameterProperties(constrainer=tfb.Softplus())  # alpha_re > 0
        )
        return params, props

    def distribution(self, params, state, inputs):
        if inputs.ndim == 2:
            inputs = inputs[jnp.newaxis, ...]
        # Inputs ordering (length 23):
        # [0:8]   --> rt features: [1, Error, Attention, Coh, Exp, Error:Att, Error:Coh, Error:Exp]
        # [8:15]  --> ra features: [1, stim, prev_stim, prev_resp, Attention, Coh, Exp]
        # [15:23] --> error features: [1, rt, Attention, Coh, Exp, rt:Att, rt:Coh, rt:Exp]
        x_rt = inputs
        # x_ra = inputs[8:15]
        # x_re = inputs[15:23]

        # 1) Gamma GLM for RT
        lp_rt = params.weights_rt[state] @ x_rt
        mu_rt = jnp.exp(lp_rt)
        base = tfd.Gamma(
                concentration=params.alpha_rt[state],
                rate=params.alpha_rt[state] / mu_rt
        )
        print("Gamma.batch_shape:", base.batch_shape)
        
        dist_rt = tfd.Independent(
            base,
            reinterpreted_batch_ndims=0
        )

        # # 2) 2-component Von Mises mixture for RA
        # mu1 = params.weights_ra1[state] @ x_ra
        # mu2 = params.weights_ra2[state] @ x_ra
        # w_raw = params.mixture_logits_ra[state]
        # w     = jnn.softmax(w_raw)
        # dist_ra = tfd.Independent(
        #     tfd.MixtureSameFamily(
        #         mixture_distribution=tfd.Categorical(probs=w),
        #         components_distribution=tfd.VonMises(
        #             loc=jnp.stack([mu1, mu2], axis=-1),
        #             concentration=jnp.stack([params.kappa1_ra[state],
        #                                     params.kappa2_ra[state]], axis=-1)
        #         )
        #     ),
        #     reinterpreted_batch_ndims=0
        # )

        # # 3) Gamma GLM for Error
        # lp_re = params.weights_re[state] @ x_re
        # mu_re = jnp.exp(lp_re)
        # dist_re = tfd.Independent(
        #     tfd.Gamma(
        #         concentration=params.alpha_re[state],
        #         rate=params.alpha_re[state] / mu_re
        #     ),
        #     reinterpreted_batch_ndims=0
        # )

        # return tfd.JointDistributionSequential([dist_rt, dist_ra, dist_re])
        # return tfd.JointDistributionSequential([dist_rt])
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
        alpha_rt:   Optional[Float[Array, "num_states"]]   = None,
        # weights_ra1: Optional[Float[Array, "num_states 7"]] = None,
        # weights_ra2: Optional[Float[Array, "num_states 7"]] = None,
        # mixture_logits_ra: Optional[Float[Array, "num_states 2"]] = None,
        # kappa1_ra: Optional[Float[Array, "num_states"]]   = None,
        # kappa2_ra: Optional[Float[Array, "num_states"]]   = None,
        # weights_re: Optional[Float[Array, "num_states 8"]] = None,
        # alpha_re:    Optional[Float[Array, "num_states"]]   = None,
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
            # weights_ra1=weights_ra1, weights_ra2=weights_ra2, mixture_logits_ra=mixture_logits_ra,
            # kappa1_ra=kappa1_ra, kappa2_ra=kappa2_ra,
            # weights_re=weights_re, alpha_re=alpha_re,
            emissions=emissions
        )
        return ParamsBlockHMMrt(**params), ParamsBlockHMMrt(**props)

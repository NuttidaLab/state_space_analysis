# Block model implementation

import jax.numpy as jnp
import jax.random as jr
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

class ParamsBlockHMMEmissions(NamedTuple):    
    weights_rt: Union[Float[Array, "num_states 6"], ParameterProperties]
    alpha_rt: Union[Float[Array, "num_states 1"], ParameterProperties]
    weights_ra: Union[Float[Array, "num_states 7"], ParameterProperties]
    kappa_ra: Union[Float[Array, "num_states 1"], ParameterProperties]
    weights_re: Union[Float[Array, "num_states 4"], ParameterProperties]
    phi_re: Union[Float[Array, "num_states 1"], ParameterProperties]

class ParamsBlockHMM(NamedTuple):
    initial: ParamsStandardHMMInitialState
    transitions: ParamsStandardHMMTransitions
    emissions: ParamsBlockHMMEmissions
    
class BlockHMMEmissions(HMMEmissions):
    def __init__(self,
                 num_states,
                 input_dim, # 2+3 from rt, 3 + 3 from ra, 3 from re 
                 emission_dim, # There are 3 of them -> rt, ra, re
                 m_step_optimizer=optax.adam(1e-3),
                 m_step_num_iters=50):
        super().__init__(m_step_optimizer=m_step_optimizer, m_step_num_iters=m_step_num_iters)
        self.num_states = num_states
        self.input_dim = input_dim # input shapes # 2 + 3 + 1 from rt, 3 + 3 + 1 from ra, 3 + 1 from re
        self.emission_dim = emission_dim

    @property
    def emission_shape(self):
        return (self.emission_dim,)

    def initialize( self,
                    key=jr.PRNGKey(0),
                    method="prior",
                    weights_rt = None,
                    alpha_rt = None,
                    weights_ra = None,
                    kappa_ra = None,
                    weights_re = None,
                    phi_re = None,
                    emissions=None):

        if method == "prior":
            
            # Reaction time component 
            weights_rt = jnp.zeros((self.num_states, 6)) # error, surprise | attention, coherence, expectiation | bias
            # learn a positive shape α[state]
            alpha_rt = jnp.ones((self.num_states, ))
            
            # Response angle component 
            weights_ra = jnp.zeros((self.num_states, 7)) # stim, prev. stim, pre. resp | attention, coherence, expectation | bias
            # learn a positive concentration \kappa[state]
            kappa_ra = jnp.ones((self.num_states, ))
            
            # Response error component
            weights_re = jnp.zeros((self.num_states, 4)) # | attention, coherence, expectation | bias
            # learn a positive “precision” φ per state  
            phi_re = jnp.ones((self.num_states, ))
            
        params = ParamsBlockHMMEmissions(
            
            weights_rt=weights_rt,
            alpha_rt=alpha_rt,
            
            weights_ra=weights_ra,
            kappa_ra=kappa_ra,
            
            weights_re=weights_re,
            phi_re=phi_re,
        )
        
        props = ParamsBlockHMMEmissions(
            
            weights_rt=ParameterProperties(),
            alpha_rt=ParameterProperties(constrainer=tfb.Softplus()), #Positive reals
            
            weights_ra=ParameterProperties(),
            kappa_ra=ParameterProperties(constrainer=tfb.Softplus()), #Positive reals
            
            weights_re=ParameterProperties(),
            phi_re=ParameterProperties(constrainer=tfb.Softplus()), #Positive reals
        )
        
        return params, props
    
    def distribution(self, params, state, inputs):
        
        # Reaction time component
        # ~ Gamma(concentration, rate) where concentration = α[state], rate = α[state] / μ
        # μ = w^T * x, where w = weights[state]
        lp_rt = params.weights_rt[state] @ inputs[:6]
        mu_rt = jnp.exp(lp_rt)
        dist_rt = tfd.Independent(
            tfd.Gamma(
                concentration=params.alpha_rt[state],
                rate=params.alpha_rt[state] / mu_rt
            ),
            reinterpreted_batch_ndims=0
        )
        
        # Response angle component 
        # ~ VonMises(loc, kappa) where loc = w^T * x, concentration = kappa
        lp_ra = params.weights_ra[state] @ inputs[6:13]
        dist_ra = tfd.Independent(
            tfd.VonMises(
                loc=lp_ra,
                concentration=params.kappa_ra[state]
            ),
            reinterpreted_batch_ndims=0
        )
        
        # Response error component
        # ~ Beta(α, β) where α = μ * φ, β = (1 - μ) * φ
        # μ = sigmoid(w^T * x) maps real → (0,1)
        lp_re = params.weights_re[state] @ inputs[13:17]
        mu_re = 1.0 / (1.0 + jnp.exp(-lp_re)) # maps real → (0,1)
        phi_re = params.phi_re[state]
        alpha = mu_re * phi_re
        beta  = (1. - mu_re) * phi_re
        dist_re = tfd.Independent(
            tfd.Beta(concentration1=alpha,
                    concentration0=beta),
            reinterpreted_batch_ndims=0
        )

        # Joint them so `log_prob` = sum of three components
        return tfd.JointDistributionSequential([
            dist_rt,
            dist_ra,
            dist_re,
        ])

    def log_prior(self, params):
        return 0.0

class BlockHMM(HMM):
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
        emission_component = BlockHMMEmissions(num_states, input_dim, emission_dim)
        super().__init__(num_states, initial_component, transition_component, emission_component)

    @property
    def inputs_shape(self):
        return (self.input_dim,)

    def initialize(self,
                   key: jr.PRNGKey=jr.PRNGKey(0),
                   method: str="prior",
                   initial_probs: Optional[Float[Array, "num_states"]]=None,
                   transition_matrix: Optional[Float[Array, "num_states num_states"]]=None,
                   weights_rt: Optional[Float[Array, "num_states 6"]]=None,
                   alpha_rt: Optional[Float[Array, "num_states 1"]]=None,
                   weights_ra: Optional[Float[Array, "num_states 7"]]=None,
                   kappa_ra: Optional[Float[Array, "num_states 1"]]=None,
                   weights_re: Optional[Float[Array, "num_states 4"]]=None,
                   phi_re: Optional[Float[Array, "num_states 1"]]=None,
                   emissions:  Optional[Float[Array, "num_timesteps emission_dim"]]=None
        ) -> Tuple[HMMParameterSet, HMMPropertySet]:

        key1, key2, key3 = jr.split(key , 3)
        params, props = dict(), dict()
        params["initial"], props["initial"] = self.initial_component.initialize(key1, method=method, initial_probs=initial_probs)
        params["transitions"], props["transitions"] = self.transition_component.initialize(key2, method=method, transition_matrix=transition_matrix)
        params["emissions"], props["emissions"] = self.emission_component.initialize(key3, method=method, emissions=emissions, 
            weights_rt=weights_rt, alpha_rt=alpha_rt, weights_ra=weights_ra, kappa_ra=kappa_ra, weights_re=weights_re, phi_re=phi_re)
        return ParamsBlockHMM(**params), ParamsBlockHMM(**props)

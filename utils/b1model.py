import jax
import jax.numpy as jnp
import optax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm

tfd   = tfp.distributions
tfb   = tfp.bijectors

class BlockOneRegressor:
    def __init__(self,
                 input_dim_rt=6,   # #features feeding RT
                 input_dim_ra=7,   # #features feeding RA
                 input_dim_re=4):  # #features feeding RE
        self.input_dim_rt = input_dim_rt
        self.input_dim_ra = input_dim_ra
        self.input_dim_re = input_dim_re
        self.input_dim    = input_dim_rt + input_dim_ra + input_dim_re

        # placeholders for fitted parameters
        self.params = None

    def initialize(self, key, method="prior"):
        """Randomly initialize weights & positive parameters."""
        keys = jax.random.split(key, 6)

        # 1) Linear weights (no explicit bias: assume last input dim is 1 if you want an intercept)
        w_scale = 1e-2
        params = {
            "weights_rt": jax.random.normal(keys[0], (self.input_dim_rt,)) * w_scale,
            "weights_ra": jax.random.normal(keys[2], (self.input_dim_ra,)) * w_scale,
            "weights_re": jax.random.normal(keys[4], (self.input_dim_re,)) * w_scale,
        }

        # 2) Positive “shape” for Gamma, VonMises concentration, and Beta precision
        #    we draw a small raw value around 0.5, then Softplus → >0
        for name, k in [("alpha_rt", keys[1]),
                        ("kappa_ra", keys[3]),
                        ("phi_re",   keys[5])]:
            raw = jax.random.normal(k, (1,)) * 0.1 + 0.5
            params[name] = tfb.Softplus().forward(raw)

        self.params = params
        return params, {}

    def distribution(self, params, inputs):
        """
        inputs: jnp.array of shape (n, input_dim)
        returns: a JointDistributionSequential([dist_rt,dist_ra,dist_re])
        """
        # if single input, add batch dimension
        if len(inputs.shape) == 2:
            inputs = jnp.expand_dims(inputs, axis=0)
        # split the inputs
        rt_in = inputs[:, :, :6]
        ra_in = inputs[:, :, 6:13]
        re_in = inputs[:, :, 13:17]

        # --- 1) Reaction Time (Gamma with log link) ---
        lp_rt = rt_in @ params["weights_rt"]     
        mu_rt = jnp.exp(lp_rt)
        dist_rt = tfd.Independent(
            tfd.Gamma(
                concentration=params["alpha_rt"],
                rate         = params["alpha_rt"] / mu_rt
            ),
            reinterpreted_batch_ndims=0
        )

        # --- 2) Response Angle (Von Mises) ---
        lp_ra = ra_in @ params["weights_ra"]     
        dist_ra = tfd.Independent(
            tfd.VonMises(
                loc          = lp_ra,
                concentration= params["kappa_ra"]
            ),
            reinterpreted_batch_ndims=0
        )

        # --- 3) Response Error (Beta over (0,1)) ---
        lp_re = re_in @ params["weights_re"]     
        mu_re = jax.nn.sigmoid(lp_re)                           # in (0,1)
        alpha = mu_re * params["phi_re"]
        beta  = (1 - mu_re) * params["phi_re"]
        dist_re = tfd.Independent(
            tfd.Beta(
                concentration1=alpha,
                concentration0=beta
            ),
            reinterpreted_batch_ndims=0
        )

        return tfd.JointDistributionSequential([dist_rt, dist_ra, dist_re])

    def neg_log_likelihood(self, params, inputs, emissions):
        """
        emissions: jnp.array of shape (n, 3) in the order [rt, ra, re]
        """
        # clip to safe range
        rt_obs = emissions[..., 0]               # shape (288,120)
        ra_obs = emissions[..., 1]               # shape (288,120)
        re_obs = jnp.clip(emissions[..., 2],      # shape (288,120)
                          1e-6, 1-1e-6)

        dist = self.distribution(params, inputs)
        # need to hand in a list/tuple in the same order we declared them
        ll = dist.log_prob([rt_obs, ra_obs, re_obs])
        return -jnp.sum(ll)  # negative log‑likelihood

    def fit_em(self,
               init_params,
               props,
               emissions,
               inputs,
               num_iters=100,
               learning_rate=1e-3,
               verbose=True):
        """Gradient‐based fit of neg-log-likelihood."""
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(init_params)
        params    = init_params
        lls       = []

        @jax.jit
        def step(params, opt_state, inputs, emissions):
            loss, grads = jax.value_and_grad(self.neg_log_likelihood)(
                params, inputs, emissions
            )
            updates, opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, opt_state, loss

        for _ in tqdm(range(num_iters), disable=not verbose, desc="Fitting"):
            params, opt_state, loss = step(params, opt_state, inputs, emissions)
            lls.append(-loss)  # record log‑likelihood

        self.params = params
        return params, lls

    # def marginal_log_prob(self, params, emissions, inputs):
    #     """
    #     Returns a length‑n array of per‐trial log‑likelihoods.
    #     """
    #     # prepare and clip
    #     rt_obs = emissions[..., 0]
    #     ra_obs = emissions[..., 1]
    #     re_obs = jnp.clip(emissions[..., 2],
    #                       1e-6, 1-1e-6)
    #     dist = self.distribution(params, inputs)
    #     ll   = dist.log_prob([rt_obs, ra_obs, re_obs])
    #     return ll 

    def marginal_log_prob(self, params, emissions, inputs):
        """
        Compute per‐trial marginal log‐likelihoods, handling both:
        - emissions: (T, 3), inputs: (T, D)  → returns scalar
        - emissions: (n, T, 3), inputs: (n, T, D) → returns (n,)
        """
        # 1) batch‐ify single‐trial inputs
        if emissions.ndim == 2:
            # (T,3) → (1,T,3)
            emissions = jnp.expand_dims(emissions, axis=0)
            # (T,D) → (1,T,D)
            inputs    = jnp.expand_dims(inputs,    axis=0)
            single    = True
        else:
            single = False

        # 2) unpack & clip
        rt_obs = emissions[..., 0]
        ra_obs = emissions[..., 1]
        re_obs = emissions[..., 2]

        # 3) compute per‐step log probs → shape (n_trials, T)
        dist = self.distribution(params, inputs)
        ll   = dist.log_prob([rt_obs, ra_obs, re_obs])

        # 4) sum over time → (n_trials,)
        mll = jnp.sum(ll, axis=1)

        # 5) unwrap if single trial
        return mll[0] if single else mll
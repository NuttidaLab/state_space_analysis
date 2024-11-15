data {
  int<lower=1> K;  // Number of latent states
  int<lower=0> T;  // Number of time points

  vector[2] wmu;
  vector[2] tau;

  array[T] real<lower=-pi(), upper=pi()> y;  // Observed circular emissions
  array[T] int<lower=1, upper=K> z;  // Observed latent states

  vector<lower=0>[K] pi_init;  // Initial distribution over states

  array[T] vector[3] x_tr;  // External inputs for transitions
  array[T] real x_em;  // External inputs for emission
}
parameters {
  // Transition coefficients
  array[K] real a;
  array[K] vector[3] b;  // Transition weights
}

model {
  // Priors for transition weights and emissions
  for (k in 1:K) {
    a[k] ~ normal(0, 1);
    b[k] ~ normal(0, 1);
  }

  // Initial state
  z[1] ~ categorical(pi_init);

  // Observed data model
  for (t in 1:T) {

    // Compute mean direction (mu)
    real mu = wmu[z[t]] * x_em[t];
    y[t] ~ von_mises(mu, tau[z[t]]);

    if (t < T) {
      // Compute eta for transitions from state z[t]
      vector[K] eta;
      for (k in 1:K) {
        eta[k] = dot_product(b[k], x_tr[t]) + a[k];
      }
      // Transition to next state
      z[t + 1] ~ categorical_logit(eta);
    }
  }
}
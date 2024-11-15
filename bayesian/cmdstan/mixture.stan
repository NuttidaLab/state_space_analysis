data { // This is the data block
  int<lower=0> T;  // Number of time points

  vector[2] wmu;
  vector[2] intercepts;
  vector[2] kappa;

  array[T] real<lower=-pi(), upper=pi()> y;  // Observed circular emissions

  vector<lower=0>[2] pi_init;  // Initial distribution over states

  array[T] vector[3] x_tr;  // External inputs for transitions
  array[T] real x_em;  // External inputs for emission
}

transformed data {}

parameters {
  real a;
  vector[3] b;
}

transformed parameters {}

model {

    array[T] int z;

    // Priors on the weights
    a ~ normal(0, 1);
    b ~ normal(0, 1);

    z[1] ~ categorical(pi_init);

    for (t in 1:T) {
        // Compute mean direction (mu)
        real mu = wmu[z[t]] * x_em[t] + intercepts[z[t]];
        y[t] ~ von_mises(mu, kappa[z[t]]);

        if (t < T) {
            // Compute eta for transitions from state z[t]
            z[t + 1] ~ bernoulli_logit(dot_product(b, x_tr[t]) + a);
        }
    }
}

generated quantities {}
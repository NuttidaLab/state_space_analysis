data {
  int<lower=1> N;       // number of trials in this subset
  vector[N] y;          // observed responses for one condition/time subset
}

parameters {
  real mu_prior;                       // mean of the prior distribution
  real<lower=0> sigma_expectation;     // std dev of the prior distribution
  real<lower=0> sigma_attention;      // std dev of the likelihood distribution

  vector[N] s;                         // latent states
}

model {
  // Priors for parameters (weakly informative)
  mu_prior ~ normal(0, 5);
  sigma_expectation ~ normal(0, 1);
  sigma_attention ~ normal(0, 1);

  // Latent state priors
  s ~ normal(mu_prior, sigma_expectation);

  // Likelihood
  y ~ normal(s, sigma_attention);
}

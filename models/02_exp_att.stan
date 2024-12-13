data {
  int<lower=1> N;            // Number of trials
  vector[N] y;               // Observed responses
  vector[N] x;               // Presented target angles
}

parameters {
  real b_exp;                // Expectation bias parameter
  real<lower=0> sigma_exp;   // Expectation uncertainty (std dev)
  real<lower=0> sigma_att;   // Attention-modulated uncertainty (std dev)
  vector[N] s;               // Latent states for each trial
}

model {
  // Priors (adjust as needed)
  b_exp ~ normal(0, 10);
  sigma_exp ~ normal(0, 5);
  sigma_att ~ normal(0, 5);

  // Likelihood
  for (i in 1:N) {
    s[i] ~ normal(x[i] + b_exp, sigma_exp);
    y[i] ~ normal(s[i], sigma_att);
  }
}
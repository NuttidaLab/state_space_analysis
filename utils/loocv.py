import jax.numpy as jnp
from jax import vmap
import numpy as np

def cross_validate_regressor(model, emissions, inputs, key, num_iters=100, init = "default", num_folds="auto"):
    # Initialize the parameters using K-Means on the full training set.
    if init == "default": params, props = model.initialize(key=key, method="prior")
    else: params, props = init

    n = len(emissions)  # number of trials
    if num_folds == "auto":
        nfolds = np.array(list(range(len(emissions))))
    else:
        nfolds = np.random.choice(np.arange(0, emissions.shape[0]), num_folds, replace=False)

    # Create leave-one-out training folds for emissions and inputs.
    emissions_train_folds = jnp.stack([
        jnp.concatenate([emissions[:i], emissions[i+1:]])
        for i in nfolds
    ])
    inputs_train_folds = jnp.stack([
        jnp.concatenate([inputs[:i], inputs[i+1:]])
        for i in nfolds
    ])

    # Validation data for each fold (the left-out trial).
    emissions_val = jnp.array([ emissions[x] for x in nfolds])  # Shape: (nfolds, 120, 1)
    inputs_val = jnp.array([ inputs[x] for x in nfolds])        # Shape: (nfolds, 120, 3)

    def _fit_fold(y_train, y_val, x_train, x_val):
        # Fit the model on the training fold using the training inputs.
        fit_params, _ = model.fit_em(params, props, y_train, num_iters=num_iters, inputs=x_train, verbose=False)
        return model.marginal_log_prob(fit_params, y_val, inputs=x_val)

    # Vectorize the fold-fitting over all folds.
    val_lls = vmap(_fit_fold)(emissions_train_folds, emissions_val, inputs_train_folds, inputs_val)
    return val_lls.mean(), val_lls

def cross_validate_dist(model, emissions, key, num_iters=100, init = "default", num_folds="auto"):
    # Initialize the parameters using K-Means on the full training set
    if init == "default": params, props = model.initialize(key=key, method="prior")
    else: params, props = init
    
    if num_folds == "auto":
        nfolds = list(range(len(emissions)))
    else:
        nfolds = np.random.choice(np.arange(0, emissions.shape[0]), num_folds, replace=False)

    # Split the training data into folds.
    # Note: this is memory inefficient but it highlights the use of vmap.
    folds = jnp.stack([
        jnp.concatenate([emissions[:i], emissions[i+1:]])
        for i in nfolds
    ])
    
    def _fit_fold(y_train, y_val):
        fit_params, train_lps = model.fit_em(params, props, y_train, num_iters=num_iters, verbose=False)
        return model.marginal_log_prob(fit_params, y_val)
    
    val_lls = vmap(_fit_fold)(folds, emissions[nfolds])
    return val_lls.mean(), val_lls

def cross_validate_dist_unvec(model, emissions, key, num_iters=100, init="default", num_folds="auto"):
    # Initialize parameters
    if init == "default":
        params, props = model.initialize(key=key, method="prior")
    else:
        params, props = init

    if num_folds == "auto":
        folds = list(range(emissions.shape[0]))
    else:
        folds = np.random.choice(np.arange(0, emissions.shape[0]), num_folds, replace=False)
    val_lls = []
    
    # Loop over each index without building the full folds array
    for i in folds:
        # Use the i-th emission as validation, and the rest as training
        y_val = emissions[i]
        y_train = jnp.concatenate([emissions[:i], emissions[i+1:]])
        
        # Fit the model on the training data
        fit_params, _ = model.fit_em(params, props, y_train, num_iters=num_iters, verbose=False)
        # Compute the marginal log-probability on the validation data
        val_ll = model.marginal_log_prob(fit_params, y_val)
        val_lls.append(val_ll)
    
    val_lls = jnp.array(val_lls)
    return np.nanmean(val_lls), val_lls
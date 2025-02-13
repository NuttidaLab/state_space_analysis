import numpy as np
from joblib import Parallel, delayed

def fnc_time_bootstrap_optimized(time_vec1, time_vec2, nboot, CI_int, n_jobs=-1, random_seed=None):
    """
    Optimized bootstrapping function to compute confidence intervals and p-values across time points.

    Parameters:
    time_vec1 : numpy array (n_neurons1, n_timepoints)
        Data for condition 1
    time_vec2 : numpy array (n_neurons2, n_timepoints)
        Data for condition 2
    nboot : int
        Number of bootstrap iterations
    CI_int : tuple of two floats
        Confidence interval bounds (e.g., (2.5, 97.5) for 95% CI)

    Returns:
    t1_CI : numpy array (n_timepoints, 2)
        Confidence intervals for time_vec1
    t2_CI : numpy array (n_timepoints, 2)
        Confidence intervals for time_vec2
    diff_CI : numpy array (n_timepoints, 2)
        Confidence intervals for the difference between time_vec1 and time_vec2
    p_diff : numpy array (n_timepoints,)
        p-values for the difference between time_vec1 and time_vec2
    """
    if random_seed is not None:
        np.random.seed(random_seed)  # Fix the random seed for reproducibility
    
    nt = time_vec1.shape[1]  # number of time points
    n_num = min(time_vec1.shape[0], time_vec2.shape[0])  # number of neurons (minimum of both conditions)

    # Precompute random indices for bootstrapping outside the loop
    bootstrap_indices = np.random.randint(0, n_num, size=(nboot, n_num))

    def compute_bootstrap_for_timepoint(ti):
        # Get data for current time point
        t1_temp = time_vec1[:, ti]
        t2_temp = time_vec2[:, ti]

        # Resample data with replacement using precomputed indices
        t1_resampled = t1_temp[bootstrap_indices]
        t2_resampled = t2_temp[bootstrap_indices]

        # Compute mean for each bootstrap sample
        t1_avg = np.nanmean(t1_resampled, axis=1)
        t2_avg = np.nanmean(t2_resampled, axis=1)

        # Compute difference between t1 and t2
        diff_avg = t1_avg - t2_avg

        # Calculate confidence intervals
        t1_ci = np.percentile(t1_avg, CI_int)
        t2_ci = np.percentile(t2_avg, CI_int)
        diff_ci = np.percentile(diff_avg, CI_int)

        # Calculate p-value for the difference
        pos_diff = np.nansum(diff_avg > 0) / nboot
        neg_diff = np.nansum(diff_avg < 0) / nboot
        p_diff_value = min(2 * min(pos_diff, neg_diff), 1 - pos_diff, 1 - neg_diff)

        return t1_ci, t2_ci, diff_ci, p_diff_value

    # Run computations in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_for_timepoint)(ti) for ti in range(nt))

    # Unpack results
    t1_CI, t2_CI, diff_CI, p_diff = zip(*results)

    # Convert to numpy arrays
    t1_CI = np.array(t1_CI)
    t2_CI = np.array(t2_CI)
    diff_CI = np.array(diff_CI)
    p_diff = np.array(p_diff)

    return t1_CI, t2_CI, diff_CI, p_diff

def fnc_time_bootstrap_optimized_retX(time_vec1, time_vec2, nboot, CI_int, n_jobs=-1, random_seed=None):
    """
    Optimized bootstrapping function to compute confidence intervals and p-values across time points.

    Parameters:
    time_vec1 : numpy array (n_neurons1, n_timepoints)
        Data for condition 1
    time_vec2 : numpy array (n_neurons2, n_timepoints)
        Data for condition 2
    nboot : int
        Number of bootstrap iterations
    CI_int : tuple of two floats
        Confidence interval bounds (e.g., (2.5, 97.5) for 95% CI)

    Returns:
    t1_CI : numpy array (n_timepoints, 2)
        Confidence intervals for time_vec1
    t2_CI : numpy array (n_timepoints, 2)
        Confidence intervals for time_vec2
    diff_CI : numpy array (n_timepoints, 2)
        Confidence intervals for the difference between time_vec1 and time_vec2
    p_diff : numpy array (n_timepoints,)
        p-values for the difference between time_vec1 and time_vec2
    """
    if random_seed is not None:
        np.random.seed(random_seed)  # Fix the random seed for reproducibility
    
    nt = time_vec1.shape[1]  # number of time points
    n_num = min(time_vec1.shape[0], time_vec2.shape[0])  # number of neurons (minimum of both conditions)

    # Precompute random indices for bootstrapping outside the loop
    bootstrap_indices = np.random.randint(0, n_num, size=(nboot, n_num))

    def compute_bootstrap_for_timepoint(ti):
        # Get data for current time point
        t1_temp = time_vec1[:, ti]
        t2_temp = time_vec2[:, ti]

        # Resample data with replacement using precomputed indices
        t1_resampled = t1_temp[bootstrap_indices]
        t2_resampled = t2_temp[bootstrap_indices]

        # Compute mean for each bootstrap sample
        t1_avg = np.nanmean(t1_resampled, axis=1)
        t2_avg = np.nanmean(t2_resampled, axis=1)

        # Compute difference between t1 and t2
        diff_avg = t1_avg - t2_avg

        # Calculate confidence intervals
        t1_ci = np.percentile(t1_avg, CI_int)
        t2_ci = np.percentile(t2_avg, CI_int)
        diff_ci = np.percentile(diff_avg, CI_int)

        # Calculate p-value for the difference
        pos_diff = np.nansum(diff_avg > 0) / nboot
        neg_diff = np.nansum(diff_avg < 0) / nboot
        p_diff_value = min(2 * min(pos_diff, neg_diff), 1 - pos_diff, 1 - neg_diff)

        return t1_avg, t1_ci, t2_avg, t2_ci, diff_avg, diff_ci, p_diff_value

    # Run computations in parallel using joblib
    results = Parallel(n_jobs=n_jobs)(delayed(compute_bootstrap_for_timepoint)(ti) for ti in range(nt))

    # Unpack results
    t1_avg, t1_CI, t2_avg, t2_CI, diff_avg, diff_CI, p_diff = zip(*results)

    # Convert to numpy arrays
    t1_avg = np.array(t1_avg)
    t1_CI = np.array(t1_CI)
    t2_avg = np.array(t2_avg)
    t2_CI = np.array(t2_CI)
    diff_avg = np.array(diff_avg)
    diff_CI = np.array(diff_CI)
    p_diff = np.array(p_diff)

    return t1_avg, t1_CI, t2_avg, t2_CI, diff_avg.mean(axis=1), diff_CI, p_diff

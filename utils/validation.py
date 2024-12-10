''' Utility functions for analysis '''
import numpy as np

# preallocate empty array and assign slice by chrisaycock
def shift(arr, num, fill_value=np.nan):
    ''' Shifts the values of arr by num indices'''
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def validate(exp_ts, jx, jy, tgonset, noise_thresh, noise_gap, do_noise_thresh = True):
    ''' Removes 'too early' trials and returns the response angle for each trial 
    
    Logic for Getting Valid Trials:

        1. Get all the trials
        2. Align all target onset at 250 ts
        3. Get distance from center and angle from center
        4. If the distance moves <0.4 au n frames before target onset. Then that trial is "too early" and all response set nan.
        5. If ts has distance > 1 au then set its response angle to the last valid angle (if first, the nan) and set that distance to 1.
        6. get angle first distance = 1 a.u. instance after target onset. or if it never reaches 1, 
            then angle at max distance after target onset. as the response angle.

    '''

    n_subjects, n_sessions, n_runs, n_trials, n_ts = jx.shape

    # shif by tgonset
    shifted_jx = np.ndarray(shape=(n_subjects, n_sessions, n_runs, n_trials, exp_ts))
    shifted_jy = np.ndarray(shape=(n_subjects, n_sessions, n_runs, n_trials, exp_ts))

    shifted_jx[:] = np.nan
    shifted_jy[:] = np.nan

    for sub in range(n_subjects):
        for sess in range(n_sessions):
            for run in range(n_runs):
                for trial in range(n_trials):
                    
                    # Align to tgonset on bigger array
                    onset = tgonset[sub, sess, run, trial]
                    shifted_jx[sub, sess, run, trial, :n_ts] = jx[sub, sess, run, trial, :]
                    shifted_jy[sub, sess, run, trial, :n_ts] = jy[sub, sess, run, trial, :]
                    shifted_jx[sub, sess, run, trial, :] = shift(shifted_jx[sub, sess, run, trial, :], 250 - onset, fill_value=np.nan)
                    shifted_jy[sub, sess, run, trial, :] = shift(shifted_jy[sub, sess, run, trial, :], 250 - onset, fill_value=np.nan)

    # Calculate distance from center
    dist_from_cent = np.sqrt(shifted_jx ** 2 + shifted_jy ** 2)
    dist_from_cent[np.isnan(dist_from_cent)] = 0

    # Calculate response angle
    resp_angle = np.arctan2( shifted_jy.flatten(), shifted_jx.flatten() )
    resp_angle = np.rad2deg(resp_angle)
    resp_angle = (resp_angle + 360) % 360;
    resp_angle = resp_angle.reshape(n_subjects, n_sessions, n_runs, n_trials, exp_ts)


    if do_noise_thresh:
        # Make responses nan if distance from center is > noise_thresh
        
        for sub in range(n_subjects):
            for sess in range(n_sessions):
                for run in range(n_runs):
                    for trial in range(n_trials):
                        # if the distance from center is >0.4 duing first targetonset - 30 frames, disregard respose angle
                        if np.any(dist_from_cent[sub, sess, run, trial, :250-noise_gap] > noise_thresh):
                            resp_angle[sub, sess, run, trial, :] = np.nan

    # Maintain last valid response angle if distance from center is > 1

    for sub in range(n_subjects):
        for sess in range(n_sessions):
            for run in range(n_runs):
                for trial in range(n_trials):
                    last_valid_angle = np.nan
                    for ts in range(exp_ts):
                        
                        # if distance is > 1, set that response angle to last valid response angle
                        # else if distance is < 1, set last valid response angle to current response angle

                        if dist_from_cent[sub, sess, run, trial, ts] > 1:
                            dist_from_cent[sub, sess, run, trial, ts] = 1
                            resp_angle[sub, sess, run, trial, ts] = last_valid_angle
                        else:
                            last_valid_angle = resp_angle[sub, sess, run, trial, ts]


    # Response of the trial

    final_resp_angles = np.ndarray(shape=(n_subjects, n_sessions, n_runs, n_trials))
    final_resp_angles[:] = np.nan

    final_resp_idx = np.ndarray(shape=(n_subjects, n_sessions, n_runs, n_trials))
    final_resp_idx[:] = np.nan

    for sub in range(n_subjects):
        for sess in range(n_sessions):
            for run in range(n_runs):
                for trial in range(n_trials):

                    # start from onset (250)
                    # get the argmax of the distance from center.
                    #   Either this argmax will be 1
                    #   or it will be the largest spike after onset
                    
                    resp_idx = np.argmax(dist_from_cent[sub, sess, run, trial, 250:])
                    final_resp_idx[sub, sess, run, trial] = 250 + resp_idx
                    final_resp_angles[sub, sess, run, trial] = resp_angle[sub, sess, run, trial, 250 + resp_idx]

    return shifted_jx, shifted_jy, dist_from_cent, resp_angle, final_resp_angles, final_resp_idx.astype(int)
from scipy.io import loadmat
from scipy.stats import circmean, circstd
import numpy as np
from tqdm import tqdm
import pandas as pd
import itertools

from .validation import validate
from .circular_math import circdist, circmedian

data_path = 'axej_eeg/'

exp_ts = 1000

# noise frames = 0 to 220 (250 - 30)
noise_thresh = 0.5
noise_gap = 30

experiment_orientations = [159, 123, 87, 51, 15]
subject_names = ["01", "02", "03", "04", "05", "06", "07", "08" ,"09", "10", "11", "12", "14"]

def get_calib(subj, sess, run=None):
    mat_contents = loadmat(data_path + f'AxeJEEG_Subj{subject_names[subj]}_S{sess+1}_Cali1.mat.mat', struct_as_record=False, squeeze_me=True)
    return mat_contents["p"].__dict__

def get_run(subj, sess, run):
    mat_contents = loadmat(data_path + f'AxeJEEG_Subj{subject_names[subj]}_S{sess+1}_Run{run+1}.mat.mat', struct_as_record=False, squeeze_me=True)
    return mat_contents["p"].__dict__

def load_and_validate_data(which="run", do_noise_thresh=True):
    n_subjects = 13
    n_sessions = 4
    # n_runs = 1
    n_trials = 120
    n_ts = 500

    # Attention (attCue): tr_foc = 1, tr_div = 2
    # Coherence (tgCoh): tr_lo = 1, tr_hi = 2

    if which == "run":
        getter = get_run
        n_runs = 6

    elif which == "calib":
        getter = get_calib
        n_runs = 1

    jx, jy, stimdir, tgonset, att, coh = [], [], [], [], [], []

    for subj,sess,run in tqdm(itertools.product(range(n_subjects), range(n_sessions), range(n_runs)), total=n_subjects*n_sessions*n_runs, desc="Loading data"):
        data = getter(subj, sess, run)
        jx.append(data["joyx"])
        jy.append(data["joyy"])
        stimdir.append(data["stimDirREAL"])

        f_tgonset = data["f_precuedur"] + data["f_cuedur"]
        tgonset.append(f_tgonset)

        att.append(data["attCue"])
        coh.append(data["tgCoh"])

    # Shape the data
    jx = np.array(jx, dtype=np.float64).reshape(n_subjects, n_sessions, n_runs, n_trials, n_ts)
    jy = np.array(jy, dtype=np.float64).reshape(n_subjects, n_sessions, n_runs, n_trials, n_ts)
    stimdir = np.array(stimdir).reshape(n_subjects, n_sessions, n_runs, n_trials).astype(np.float64)
    tgonset = np.array(tgonset).reshape(n_subjects, n_sessions, n_runs, n_trials).astype(int)
    att = np.array(att).reshape(n_subjects, n_sessions, n_runs, n_trials).astype(int)
    coh = np.array(coh).reshape(n_subjects, n_sessions, n_runs, n_trials).astype(int)

    # replace 1 with -1 and 2 with 1
    attention = np.where(att == 2, -1, 1)
    coherence = np.where(coh == 1, -1, 1)

    shifted_jx, shifted_jy, dist_from_cent, resp_angle, final_resp_angles, final_resp_idx = validate(
        exp_ts, 
        jx, 
        jy, 
        tgonset, 
        noise_thresh, 
        noise_gap,
        do_noise_thresh = do_noise_thresh,
    )

    return shifted_jx, shifted_jy, dist_from_cent, resp_angle,\
        final_resp_angles, final_resp_idx, attention, coherence, stimdir, tgonset

def package_calib_data(do_clip = True, do_noise_thresh = True):
    shifted_jx, shifted_jy, dist_from_cent, resp_angle, final_resp_angles, final_resp_idx, att, coh, stimdir, tgonset = load_and_validate_data(which="calib", do_noise_thresh=do_noise_thresh)

    n_subjects, n_sessions, n_runs, n_trials = shifted_jx.shape[:4]

    # Do clipping
    if do_clip:
        resp_angle = np.clip(resp_angle, 0, 180)
        final_resp_angles = np.clip(final_resp_angles, 0, 180)

    # Make calib_df 
    calib_df = pd.DataFrame({
        "subject": np.repeat(np.arange(n_subjects), n_sessions*n_runs*n_trials),
        "session": np.repeat(np.tile(np.arange(n_sessions), n_runs*n_trials), n_subjects),
        "run": np.repeat(np.tile(np.arange(n_runs), n_trials), n_subjects*n_sessions),
        "trial": np.tile(np.arange(n_trials), n_subjects*n_sessions*n_runs),
        "final_resp_angle": final_resp_angles.flatten(),
        "final_resp_idx": final_resp_idx.flatten(),
        "att": att.flatten(),
        "coh": coh.flatten(),
        "stimdir": stimdir.flatten(),
        "tgonset": tgonset.flatten(),
    })

    # Group according to different stimdirs
    # 5 angles in total and each subject has 4 * 120/ 5 = 96 trials per angle
    stim_resp = np.zeros((n_subjects, 5, 96))

    for sub in range(n_subjects):
        # for each unique stimdir
        for i, unique_stim in enumerate(np.unique(stimdir)):
            # get the response angle for that stimdir by masking the stimdir with the unique stimdir
            stim_resp[sub, i] = final_resp_angles[sub][stimdir[sub]==unique_stim]

    # Make the median table
    # Make the std table
    circ_median = np.zeros((n_subjects, 5))
    circ_std = np.zeros((n_subjects, 5))
    for sub in range(n_subjects):
        for i, unique_stim in enumerate(np.unique(stimdir)):
            circ_median[sub, i] = circmedian(stim_resp[sub, i])
            circ_std[sub, i] = np.rad2deg(circstd(np.deg2rad(stim_resp[sub, i]), nan_policy='omit'))

    return calib_df, circ_median, circ_std, (shifted_jx, shifted_jy, dist_from_cent, resp_angle)

def calc_acc(calib_median, calib_std, final_resp_angles, stimdir):

    n_subjects, n_sessions, n_runs, n_trials = final_resp_angles.shape[:4]

    median_key = {15:0, 51:1, 87:2, 123:3, 159:4}
    std_key = {15:0, 51:1, 87:2, 123:3, 159:4}

    # One and two sigma distance from the calibration median

    resp_correct_sigma = np.zeros_like(final_resp_angles)
    resp_correct_sigma[:] = np.nan

    resp_correct_two_sigma = np.zeros_like(final_resp_angles)
    resp_correct_two_sigma[:] = np.nan

    targ_calib = np.zeros_like(final_resp_angles)
    targ_calib[:] = np.nan

    nn_neighbor = np.zeros_like(final_resp_angles)
    nn_neighbor[:] = np.nan

    nn_correct = np.zeros_like(final_resp_angles)
    nn_correct[:] = np.nan

    exp_angle = np.zeros_like(final_resp_angles)
    exp_angle[:] = np.nan

    expected = np.zeros_like(final_resp_angles)
    expected[:] = np.nan

    resp_err = np.zeros_like(final_resp_angles)
    resp_err[:] = np.nan

    v_err = np.zeros_like(final_resp_angles)
    v_err[:] = np.nan

    for subj in range(n_subjects):
        for sess in range(n_sessions):
            for run in range(n_runs):
                if run != 5:
                    vals, counts = np.unique(stimdir[subj,sess,run], return_counts=True)
                    block_exp_angle = vals[np.argmax(counts)]

                for trial in range(n_trials):
                    actual_stimdir = stimdir[subj,sess,run,trial]

                    # Get the median and std for the actual stimdir
                    calib_resp_median = calib_median[subj,median_key[actual_stimdir]]
                    calib_resp_sigma = calib_std[subj,std_key[actual_stimdir]]
                    targ_calib[subj,sess,run,trial] = calib_resp_median

                    # Get the response angle
                    trial_response = final_resp_angles[subj,sess,run,trial]

                    # Skip if the trial is 'too early'
                    if np.isnan(trial_response): continue

                    if run != 5:
                        calib_block_exp_median = calib_median[subj,median_key[block_exp_angle]]
                        exp_angle[subj,sess,run,trial] = calib_block_exp_median
                        expected[subj,sess,run,trial] = int(calib_block_exp_median == calib_resp_median)

                    neighbor = min(calib_median[subj], key=lambda x:abs(x-trial_response))
                    nn_neighbor[subj,sess,run,trial] = neighbor
                    nn_correct[subj,sess,run,trial] = int(neighbor == calib_resp_median)

                    resp_err[subj,sess,run,trial] = circdist(calib_resp_median, trial_response)
                    v_err[subj,sess,run,trial] = trial_response - calib_resp_median

                    # Check if the response is within 1 sigma or 2 sigma
                    resp_correct_sigma[subj,sess,run,trial] = circdist(calib_resp_median, trial_response) <= calib_resp_sigma
                    resp_correct_two_sigma[subj,sess,run,trial] = circdist(calib_resp_median, trial_response) <= calib_resp_sigma*2
    
    return resp_correct_sigma, resp_correct_two_sigma, targ_calib,\
        nn_neighbor, nn_correct, exp_angle, expected, resp_err, v_err

def package_run_data(median, std, do_clip = True, do_noise_thresh = True):
    
    shifted_jx, shifted_jy, dist_from_cent, resp_angle, final_resp_angles, final_resp_idx, attention, coherence, stimdir, tgonset = load_and_validate_data(which="run", do_noise_thresh=do_noise_thresh)

    n_subjects, n_sessions, n_runs, n_trials = shifted_jx.shape[:4]

    # Do clipping
    if do_clip:
        resp_angle = np.clip(resp_angle, 0, 180)
        final_resp_angles = np.clip(final_resp_angles, 0, 180)

    
    resp_correct_sigma, resp_correct_two_sigma, targ_calib, nn_neighbor, nn_correct, exp_angle, expected, resp_err, v_err = calc_acc(median, std, final_resp_angles, stimdir)

    resp_df = pd.DataFrame({
        "subj": np.repeat(np.arange(n_subjects), n_sessions*n_runs*n_trials),
        "sess": np.tile(np.repeat(np.arange(n_sessions), n_runs*n_trials), n_subjects),
        "run": np.tile(np.repeat(np.arange(n_runs), n_trials), n_subjects*n_sessions),
        "trial": np.tile(np.arange(n_trials), n_subjects*n_sessions*n_runs),
        "target": stimdir.flatten(),
        "t_calib": targ_calib.flatten(),
        "resp": final_resp_angles.flatten(),
        "resp_err": resp_err.flatten(),
        "v_err": v_err.flatten(),
        "exp": exp_angle.flatten(),
        "nn": nn_neighbor.flatten(),
        "1sig": resp_correct_sigma.flatten(),
        "2sig": resp_correct_two_sigma.flatten(),
        "nn_c": nn_correct.flatten(),
        "att": attention.flatten(),
        "coh": coherence.flatten(),
        "exp_al": expected.flatten(),
        "f_onset": tgonset.flatten(),
        "f_resp": final_resp_idx.flatten(),
    })

    # if resp > than t_calib then "higher" else "lower"
    resp_df["dir"] = np.where(resp_df["resp"] > resp_df["t_calib"], "higher", "lower")

    return resp_df, (shifted_jx, shifted_jy, dist_from_cent, resp_angle)


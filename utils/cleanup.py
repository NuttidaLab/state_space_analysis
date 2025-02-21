import numpy as np
from scipy.ndimage import gaussian_filter1d

def discretize_nearly_static_segments(data, movement_threshold=0.1, min_length=5, floor_threshold=0.4):
    """
    For each time series (row) in `data`, this function checks for segments
    where the change from one frame to the next is less than `movement_threshold`.
    If a segment spans more than `min_length` frames, then the entire segment is 
    replaced ("floored") to 0 if its value is less than `floor_threshold`, or to 1 otherwise.

    Parameters:
      data: 2D numpy array of shape (n_series, n_frames) with values between 0 and 1.
      movement_threshold: Maximum allowed difference between consecutive frames 
                          to consider the segment nearly static.
      min_length: Minimum number of frames in a segment to trigger the flooring.
                  (Only segments longer than this will be modified.)
      floor_threshold: If the nearly static segment’s value is below this threshold,
                       the segment is set to 0; otherwise, it is set to 1.
    
    Returns:
      A new numpy array with the modified time series.
    """
    modified = data.copy()
    mask = np.zeros_like(modified, dtype=bool)
    
    n_series, n_frames = modified.shape

    for i in range(n_series):
        ts = modified[i]
        j = 0
        while j < n_frames:
            start = j
            # Extend the segment while consecutive differences are below the threshold.
            while j < n_frames - 1 and np.abs(ts[j + 1] - ts[j]) < movement_threshold:
                j += 1
            # Now indices start to j (inclusive) form a nearly static segment.
            if (j - start + 1) > min_length:
                seg_val = ts[start]  # representative value (since segment is nearly constant)
                if seg_val < floor_threshold:
                    ts[start:j + 1] = 0
                    # Mark these indices in the mask
                    mask[i, start:j + 1] = True
                else:
                    # ts[start:j + 1] = 1
                    pass
            j += 1  # move to the next frame
    
    modified = gaussian_filter1d(modified, sigma=2, axis=1)
    return modified, mask
# -*- coding: utf-8 -*-
"""
helpers   Private helper functions for neural_analysis code


Created on Fri Apr  9 14:08:15 2021

@author: sbrincat
"""
import numpy as np

from scipy.stats import norm,mode


def _check_window_lengths(windows,tol=1):
    """ 
    Ensures a set of windows are the same length. If not equal, but within given tolerance,
    windows are trimmed or expanded to the modal window length.
    
    ARGS
    windows (n_wins,2) array-like. Set of windows to test, given as series of [start,end].
    
    tol     Scalar. Max tolerance of difference of each window length from the modal value.
    
    RETURNS
    windows (n_wins,2) ndarray. Same windows, possibly slightly trimmed/expanded to uniform length
    """    
    windows = np.asarray(windows)
    
    window_lengths  = np.diff(windows,axis=1).squeeze()
    window_range    = np.ptp(window_lengths)
    
    # If all window lengths are the same, windows are OK and we are done here
    if np.allclose(window_lengths, window_lengths[0]): return windows
    
    # Compute mode of windows lengths and max difference from it
    modal_length    = mode(window_lengths)[0][0]    
    max_diff        = np.max(np.abs(window_lengths - modal_length))
    
    # If range is beyond our allowed tolerance, throw an error
    assert max_diff <= tol, \
        ValueError("Variable-length windows unsupported (range=%.1f). All windows must have same length" \
                    % window_range)
        
    # If range is between 0 and tolerance, we trim/expand windows to the modal length
    windows[:,1]    = windows[:,1] + (modal_length - window_lengths)
    return windows


def _enclose_in_object_array(data):
    """ Enclose array within an object array """
    out = np.empty((1,),dtype=object)
    out[0] = data
    return out

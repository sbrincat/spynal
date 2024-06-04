
# -*- coding: utf-8 -*-
""" Preprocessing functions for LFP/EEG/continuous data and spectral analysis """
import numpy as np

from sklearn.linear_model import LinearRegression

from spynal.helpers import _check_window_lengths
from spynal.utils import index_axis, standardize_array, undo_standardize_array


def cut_trials(data, trial_lims, smp_rate, axis=0):
    """
    Cut continuous (eg LFP) data into trials

    Parameters
    ----------
    data : ndarray, shape=(...,n_timepts,...)
        Continuous data unsegmented into trials.
        Arbitrary dimensionality, could include multiple channels, etc.

    trial_lims : array-like, shape=(n_trials,2)
        List of [start,end] of each trial (in s) to use to cut data.

    smp_rate : float
        Sampling rate of data (Hz).

    axis : int, default: 0 (1st axis)
        Axis of data array corresponding to time samples

    Returns
    -------
    cut_data : ndarray, shape=(...,n_trial_timepts,...,n_trials)
        Continuous data segmented into trials.
        Trial axis is appended to end of all axes in input data.
    """
    trial_lims = np.asarray(trial_lims)
    assert (trial_lims.ndim == 2) and (trial_lims.shape[1] == 2), \
        "trial_lims argument should be a (n_trials,2) array of trial [start,end] times"
    n_trials = trial_lims.shape[0]

    # Convert trial_lims in s -> indices into continuous data samples
    trial_idxs = np.round(smp_rate*trial_lims).astype(int)
    assert trial_idxs.min() >= 0, \
        ValueError("trial_lims are attempting to index before start of data")
    assert trial_idxs.max() < data.shape[axis], \
        ValueError("trial_lims are attempting to index beyond end of data")
    # Ensure all windows have same length
    trial_idxs = _check_window_lengths(trial_idxs,tol=1)

    # Samples per trial = end - start + 1
    n_smp_per_trial = trial_idxs[0,1] - trial_idxs[0,0] + 1

    # Create array to hold trial-cut data. Same shape as data, with time sample axis
    # reduced to n_samples_per_trial and trial axis appended.
    cut_shape = [*data.shape,n_trials]
    cut_shape[axis] = n_smp_per_trial
    cut_data = np.empty(tuple(cut_shape),dtype=data.dtype)

    # Extract segment of continuous data for each trial
    for trial,lim in enumerate(trial_idxs):
        cut_data[...,trial] = index_axis(data, axis, slice(lim[0],lim[1]+1))

    return cut_data


def realign_data(data, align_times, time_range, timepts, time_axis=0, trial_axis=-1):
    """
    Realigns trial-cut continuous (eg LFP) data to new set of within-trial times
    (eg new trial event) so that t=0 on each trial at given event.
    For example, data aligned to a start-of-trial event might
    need to be relaligned to the behavioral response.

    Parameters
    ----------
    data : ndarray, shape=(...,n_timepts,...,n_trials,...)
        Continuous data segmented into trials.
        Arbitrary dimensionality, could include multiple channels, etc.

    align_times : array-like, shape=(n_trials,)
        New set of times (in old reference frame) to realign data to (in s)

    time_range : array-like, shape=(2,)
        Time range to extract from each trial around new align time
        ([start,end] in s relative to align_times).
        eg, time_range=(-1,1) -> extract 1 s on either side of align event.

    timepts : array-like, shape=(n_timepts)
        Time sampling vector for data (in s)

    time_axis : int, default: 0 (1st axis of array)
        Axis of data corresponding to time samples

    trial_axis : int, default: -1 (last axis of array)
        Axis of data corresponding to distinct trials

    Returns
    -------
    realigned : ndarray, shape=(...,n_timepts_out,...,n_trials,...)
        Data realigned to given within-trial times.
        Time axis is reduced to length implied by `time_range`, but otherwise
        array has same shape as input data.
    """
    assert time_range is not None, \
        "Desired time range to extract from each trial must be given in  `time_range`"
    assert timepts is not None, "Data time sampling vector must be given in `timepts`"

    timepts     = np.asarray(timepts)
    align_times = np.asarray(align_times)
    time_range  = np.asarray(time_range)

    if time_axis < 0:   time_axis = data.ndim + time_axis
    if trial_axis < 0:  trial_axis = data.ndim + trial_axis

    # Move array axes so time axis is 1st and trials last (n_timepts,...,n_trials)
    if (time_axis == data.ndim-1) and (trial_axis == 0):
        data = np.swapaxes(data,time_axis,trial_axis)
    else:
        if time_axis != 0:              data = np.moveaxis(data,time_axis,0)
        if trial_axis != data.ndim-1:   data = np.moveaxis(data,trial_axis,-1)

    # Convert align times and time epochs to nearest integer sample indexes
    dt = np.mean(np.diff(timepts))
    align_smps = np.round((align_times - timepts[0])/dt).astype(int)
    range_smps = np.round(time_range/dt).astype(int)
    # Compute [start,end] sample indexes for each trial epoch = align time +/- time range
    trial_range_smps = align_smps[:,np.newaxis] + range_smps[np.newaxis,:]

    assert (trial_range_smps[:,0] >= 0).all(), \
        "Some requested time epochs extend before start of data"
    assert (trial_range_smps[:,1] < len(timepts)).all(), \
        "Some requested time epochs extend beyond end of data"

    n_timepts_out   = range_smps[1] - range_smps[0] + 1
    return_shape    = (n_timepts_out, *(data.shape[1:]))
    realigned       = np.empty(return_shape)

    # Extract timepoints corresponding to realigned time epoch from each trial in data
    for trial,t in enumerate(trial_range_smps):
        # Note: '+1' below makes the selection inclusive of the right endpoint in each trial
        realigned[...,trial] = data[t[0]:t[1]+1,...,trial]

    # Move array axes back to original locations
    if (time_axis == data.ndim-1) and (trial_axis == 0):
        realigned = np.swapaxes(realigned,trial_axis,time_axis)
    else:
        if time_axis != 0:              realigned = np.moveaxis(realigned,0,time_axis)
        if trial_axis != data.ndim-1:   realigned = np.moveaxis(realigned,-1,trial_axis)

    return realigned


def realign_data_on_event(data, event_data, event, timepts, align_times, time_range,
                          time_axis=0, trial_axis=-1):
    """
    Convenience wrapper around `realign_data` for relaligning to a given
    named event within a per-trial dataframe or dict variable.

    Only parameters differing from :func:`realign_data` are described here.

    Parameters
    ----------
    event_data : dict, {str : ndarray, shape=(n_trials,)} or DataFrame, shape=(n_trials,n_events)
        Per-trial event timing data to use to realign spike timestamps.

    event : str
        Dict key or DataFrame column name whose associated values are to be used to realign data
    """
    # Extract vector of times to realign on
    align_times = event_data[event]

    # Compute the realignment and return
    return realign_data(data, timepts, align_times, time_range,
                        time_axis=time_axis, trial_axis=trial_axis)


def remove_dc(data, axis=None):
    """
    Remove constant DC component of signals, estimated as across-time mean
    for each time series (ie trial,channel,etc.)

    Parameters
    ----------
    data : ndarray, shape=(...,n_timepoints,...)
        Raw data to remove DC component of.
        Can be any arbitary shape, with time sampling along `axis`

    axis : int, Default: None (remove DC component computed across *full* data array)
        Data axis corresponding to time

    Returns
    -------
    data : ndarray, shape=(...,n_timepoints,...)
        Data with DC component removed (same shape as input)
    """
    return data - data.mean(axis=axis, keepdims=True)


def remove_evoked(data, axis=0, method='mean', design=None, return_evoked=False):
    """
    Remove estimate of evoked potentials phase-locked to trial events,
    returning data with (in theory) only non-phase-locked induced components

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Raw data to remove evoked components from.
        Can be any arbitary shape, with observations (trials) along `axis`.

    axis : int, default: 0 (1st axis)
        Data axis corresponding to distinct observations/trials

    method : {'mean','groupmean','regress'}, default: 'mean'
        Method to use for estimating evoked potentials:

        - 'mean'      : Grand mean signal across all observations (trials)
        - 'groupmean' : Mean signal across observations with each group in `design`
        - 'regress'   : OLS regresion fit of design matrix `design` to data

    design : array-like, shape=(n_obs,...), optional
        Design matrix to fit to data (`method` == 'regress')
        or group/condition labels for each observation (`method` == 'groupmean').
        Not used for `method` == 'mean'.

    return_evoked : bool, default: False
        If True, also returns second output = estimated evoked potential on each trial.

    Returns
    -------
    data_induced : ndarray, shape=(...,n_obs,...)
        Data with estimated evoked component removed, leaving only non-phase-locked
        "induced" component. Same shape as input `data`.

    evoked : ndarray, shape=(...,n_obs,...), optional
        Evoked potential for each trial, estimated with given method. Same shape as `data`.
        Only returned if `return_evoked` is True.
    """
    assert axis is not None, "<axis> should correspond to data trials/observations dimension"
    
    method = method.lower()
    design = np.asarray(design)

    data = data.copy()  # Copy input data to avoid overwriting it in caller
    
    # Subtract off grand mean potential across all trials
    if method in ['mean','grandmean']:
        evoked_grandmean = np.mean(data, axis=axis, keepdims=True)
        data -= evoked_grandmean
        
        # Expand grand-mean evoked potential to n_trials if returning evoked
        if return_evoked:
            reps = np.ones((data.ndim,),dtype=int)
            reps[axis] = data.shape[axis]
            evoked = np.tile(evoked_grandmean, reps)

    # Subtract off mean potential across all trials within each group/condition
    # todo  can we do this with an xarray or pandas groupby() instead??
    elif method == 'groupmean':
        assert (design.ndim == 1) or ((design.ndim == 2) and (design.shape[1] == 1)), \
            "Design matrix <design> must be vector-like (1d or 2d w/ shape[1]=1)"

        groups = np.unique(design)

        data, data_shape = standardize_array(data, axis=axis, target_axis=0)
        if return_evoked: evoked = np.empty_like(data)
        
        for group in groups:
            idxs = design == group
            evoked_group = np.mean(data[idxs,...], axis=0, keepdims=True)
            data[idxs,...] -= evoked_group
            if return_evoked: evoked[idxs,...] = evoked_group

        data = undo_standardize_array(data, data_shape, axis=axis, target_axis=0)
        if return_evoked:
            evoked = undo_standardize_array(evoked, data_shape, axis=axis, target_axis=0)

    # Regress data on given design matrix and return residuals
    elif method == 'regress':
        assert design.ndim in [1,2], \
            "Design matrix <design> must be matrix-like (2d) or vector-like (1d)"

        # Only fit explicit intercept if design doesn't already have a constant (intercept) column in it
        has_constant_col = np.any(np.all(design==1, axis=0))
        model = LinearRegression(fit_intercept=True if not has_constant_col else False)

        data, data_shape = standardize_array(data, axis=axis, target_axis=0)

        evoked = model.fit(design,data).predict(design)
        data -= evoked

        data = undo_standardize_array(data, data_shape, axis=axis, target_axis=0)
        if return_evoked:
            evoked = undo_standardize_array(evoked, data_shape, axis=axis, target_axis=0)

    if return_evoked:   return data, evoked
    else:               return data


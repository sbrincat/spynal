#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of oscillatory neural synchrony

Function list
-------------
Field-field synchrony
^^^^^^^^^^^^^^^^^^^^^
- synchrony :               General synchrony between pair of analog channels using given method
- coherence :               Time-frequency coherence between pair of analog channels
- ztransform_coherence  :   Z-transform coherence so ~ normally distributed
- plv :                     Phase locking value (PLV) between pair of analog channels
- ppc :                     Pairwise phase consistency (PPC) btwn pair of analog channels

Spike-field synchrony
^^^^^^^^^^^^^^^^^^^^^
- spike_field_coupling :    General spike-field coupling/synchrony btwn spike/LFP pair
- spike_field_coherence :   Spike-field coherence between a spike/LFP pair
- spike_field_plv :         Spike-field PLV between a spike/LFP pair
- spike_field_ppc :         Spike-field PPC between a spike/LFP pair

Data simulation
^^^^^^^^^^^^^^^
- simulate_multichannel_oscillation : Generates simulated oscillatory paired data

Function reference
------------------
"""
# Created on Thu Oct  4 15:28:15 2018
#
# @author: sbrincat
# TODO  Reformat jackknife functions to match randstats functions and move there?

import numpy as np

from spynal.utils import set_random_seed, index_axis, axis_index_slices
from spynal.spectra import spectrogram, simulate_oscillation
from spynal.randstats import jackknifes
from spynal.sync.coherence import coherence, spike_field_coherence
from spynal.sync.phasesync import plv, ppc, spike_field_plv, spike_field_ppc


# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def synchrony(data1, data2, axis=0, method='PPC', return_phase=False, single_trial=None,
              data_type=None, spec_method='wavelet', smp_rate=None, time_axis=None,
              taper_axis=None, keepdims=True, **kwargs):
    """
    Compute measure of synchrony between pair of channels of continuous (eg LFP)
    raw or spectral (time-frequency) data, using given estimation method

    Parameters
    ----------
    data1,data2 : ndarray, shape=(...,n_obs,...), dtype=float or complex
        Single-channel continuous (eg LFP) data for 2 distinct channels.
        Can be given as raw LFPs or complex-valued time-frequency transform.

        Trial/observation axis is assumed to be axis 0 unless given in `axis`.
        For raw data, axis corresponding to time must be given in `time_axis`.

        Other than those constraints, data can have arbitrary shape, with
        analysis performed in mass-bivariate fashion independently
        along each dimension other than observation `axis` (eg different
        frequencies, timepoints, conditions, etc.)

    axis : scalar, default: 0
        Axis corresponding to distinct observations/trials

    method : {'PPC','PLV','coherence'}, default: 'PPC'
        Synchrony estimation method:

        - 'PPC' : Pairwise Phase Consistency, measure of phase synchrony (see :func:`ppc`)
        - 'PLV' : Phase Locking Value, measure of phase synchrony (see :func:`plv`)
        - 'coherence' : coherence, measure of linear oscillatory coupling (see :func:`coherence`)

    return_phase : bool, default: False
        If False, only return measure of synchrony magnitude/strength between data1 and data2.
        If True, also returns mean phase difference (or coherence phase) between
        data1 and data2 (in radians) in additional output.

    single_trial : str or None, default: None
        What type of coherence estimator to compute:

        - None :       standard across-trial estimator
        - 'pseudo' :   single-trial estimates using jackknife pseudovalues
        - 'richter' :  single-trial estimates using actual jackknife estimates cf. Richter 2015

    data_type : {'raw','spectral'}, default: assume 'raw' if data is real; 'spectral' if complex
        What kind of data is input: 'raw' or 'spectral'?

    spec_method : {'wavelet','multitaper','bandfilter'}, default: 'wavelet'
        Method to use for spectral analysis (only used for *raw* data)

    smp_rate : scalar
        Sampling rate of data (only needed for *raw* data)

    time_axis : int
        Axis of data corresponding to time (only needed for *raw* data)

    taper_axis : int
        Axis of spectral data corresponding to tapers (only needed for *multitaper spectral* data)

    keepdims : bool, default: True
        If True, retains reduced trial and/or taper axes as length-one axes in output.
        If False, removes reduced trial,taper axes from outputs.

    **kwargs :
        All other kwargs passed as-is to specific synchrony estimation function.

    Returns
    -------
    sync : ndarray, shape=
        Synchrony magnitude/strength (PPC/PLV/coherence) between data1 and data2.
        If data is spectral, this has same shape as data, but with `axis` reduced
        to a singleton or removed, depending on value of `keepdims`.
        If data is raw, this has same shape as data with `axis` reduced or removed
        and a new frequency axis inserted immediately before `time_axis`.

    freqs : ndarray, shape=(n_freqs,)
        List of frequencies in `sync`. Only returned for raw data, [] otherwise.

    timepts : ndarray, shape=(n_timepts_out,)
        List of timepoints in `sync` (in s, referenced to start of data).
        Only returned for raw data, [] otherwise.

    dphi : ndarray, optional
        Mean phase difference (or coherence phase) between data1 and data2 in radians.
        Same shape as `sync`.
        Positive values correspond to data1 leading data2.
        Negative values correspond to data1 lagging behind data2.
        Optional: Only returned if return_phase is True.

    Examples
    --------
    sync, freqs, timepts = synchrony(data1, data2, axis=0, method='PPC', return_phase=False)

    sync, freqs, timepts, dphi = synchrony(data1, data2, axis=0, method='PPC', return_phase=True)

    References
    ----------
    - Single-trial method:    Womelsdorf et al (2006) https://doi.org/10.1038/nature04258
    - Single-trial method:    Richter et al (2015) https://doi.org/10.1016/j.neuroimage.2015.04.040
    """
    method = method.lower()
    if method in ['ppc','pairwise_phase_consistency']:  sync_func = ppc
    elif method in ['plv','phase_locking_value']:       sync_func = plv
    elif method in ['coh','coherence']:                 sync_func = coherence
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    return sync_func(data1, data2, axis=axis, return_phase=return_phase, single_trial=single_trial,
                     data_type=data_type, spec_method=spec_method, smp_rate=smp_rate,
                     time_axis=time_axis, taper_axis=taper_axis, keepdims=keepdims, **kwargs)


# =============================================================================
# Spike-Field Synchrony functions
# =============================================================================
def spike_field_coupling(spkdata, lfpdata, axis=0, method='PPC', return_phase=False,
                         time_axis=None, taper_axis=None, timepts=None,
                         data_type=None, spec_method='multitaper', smp_rate=None,
                         keepdims=True, **kwargs):
    """
    Computs measure of pairwise coupling between a pair of spike and continuous (eg LFP)
    raw or spectral (time-frequency) data, using given estimation method

    Parameters
    ----------
    spkdata : ndarray, shape=(...,n_obs,...), dtype=bool
        Single-channel binary spike trains (with 1's labelling spike times, 0's otherwise).

        For coherence: Can be given either as raw binary spike trains or as their spectral
        transform, but must have same data type (raw or spectral) and shape as `lfpdata`.

        For PLV/PPC: Shape is arbitrary, but MUST have same shape as `lfpdata`
        for raw `lfpdata`, and same dimensionality as `lfpdata` for spectral `lfpdata`.
        Thus, if `lfpdata` is spectral, must pre-pend singleton (length-1) axis to
        spkdata to match (eg using np.newaxis).

    lfpdata : ndarray, shape=(...,n_obs,...)
        Single-channel continuous (eg LFP) data.
        Can be given as raw LFPs or complex-valued time-frequency transform.

        Trial/observation axis is assumed to be axis 0 unless given in `axis`.
        For raw data, axis corresponding to time must be given in `time_axis`.

        Other than those constraints, data can have arbitrary shape, with analysis performed
        in mass-bivariate fashion independently along each axis other than observation `axis`
        (eg different frequencies, timepoints, conditions, etc.)

    axis : int, default: 0
        Axis corresponding to distinct observations/trials

    method : {'PPC','PLV','coherence'}, default: 'PPC'
        Spike-field coupling estimation method

        - 'PPC' : Pairwise Phase Consistency (see :func:`spike_field_ppc`)
        - 'PLV' : Phase Locking Value (see :func:`spike_field_plv`)
        - 'coherence' : coherence (see :func:`spike_field_coherence`)

    return_phase : bool, default: False
        If False, only return measure of synchrony magnitude/strength between data1 and data2.
        If True, also returns mean LFP phase of spikes (or coherence phase) in additional output.

    time_axis : int
        Axis of data corresponding to time. Required input for phase-based methods (PLV/PPC).
        For coherence, only needed for `data_type` = 'raw' AND `spec_method` = 'multitaper'.

    taper_axis : int
        Axis of spectral data corresponding to tapers. Only needed for multitaper spectral data.

    timepts : array-like, shape=(n_timepts,), Default: (0 - n_timepts-1)/smp_rate
        Time sampling vector for data. Default value starts at 0, with spacing = 1/smp_rate.
        For phase methods, should be in same time units as `width`/`spacing`/`lims` or `timewins`.

    data_type : {'raw','spectral'}, default: assume 'raw' if data is real; 'spectral' if complex
        What kind of data is input: 'raw' or 'spectral'?

    spec_method : {'wavelet','multitaper','bandfilter'}, default: 'wavelet'
        Method to use for (or already used for) spectral analysis.
        NOTE: Must be input for multitaper spectral data, so taper axis is handled appropriately.
        Otherwise, only used for *raw* data.

    smp_rate : scalar
        Sampling rate of data (only needed for *raw* data)

    keepdims : bool, default: True
        If True, retains reduced trial and/or taper axes as length-one axes in output.
        If False, removes reduced trial,taper axes from outputs.

    **kwargs :
        All other kwargs passed as-is to specific synchrony estimation function.

    Returns
    -------
    sync : ndarray
        Magnitude/strength of spike-field coupling (coherence or PLV/PPC magnitude).

        For phase-based methods, time windows without any spikes are set = np.nan.

        If data is spectral, this has same shape as data, but with `axis` removed.
        If data is raw, this has same shape with `axis` removed and a new
        frequency axis inserted immediately before `time_axis`.

        TODO
        If lfpdata is spectral, this has same shape, but with `axis` removed
        (and taper_axis as well for multitaper), and time axis reduced to n_timewins.
        If lfpdata is raw, this has same shape with `axis` removed, `time_axis`
        reduced to n_timewins, and a new frequency axis inserted immediately
        before `time_axis`.

    freqs : ndarray, shape=(n_freqs,)
        List of frequencies in `sync`. Only returned for raw data, [] otherwise.

    timepts : ndarray, shape=(n_timepts,)
        List of timepoints in `sync` (in s, referenced to start of data).
        Only returned for raw data, [] otherwise.

    n : ndarray, shape=(n_timepts,), dtype=int
        Number of spikes contributing to synchrony computations.
        Value only returned for phase-based measures (PLV/PPC), for coherence, returns None.

    phi : ndarray, optional
        Mean phase of LFPs at spike times (or phase of complex coherency) in radians.
        Optional: Only returned if return_phase is True.

    Examples
    --------
    sync,freqs,timepts,n = spike_field_coupling(spkdata,lfpdata,return_phase=False)

    sync,freqs,timepts,n,phi = spike_field_coupling(spkdata,lfpdata,return_phase=True)
    """
    method = method.lower()
    if method in ['ppc','pairwise_phase_consistency']:  sfc_func = spike_field_ppc
    elif method in ['plv','phase_locking_value']:       sfc_func = spike_field_plv
    elif method in ['coh','coherence']:                 sfc_func = spike_field_coherence
    else:
        raise ValueError("Unsupported value '%s' given for <method>. \
                         Should be 'PPC'|'PLV'|'coherence'" % method)

    return sfc_func(spkdata, lfpdata, axis=axis, time_axis=time_axis, taper_axis=taper_axis,
                    timepts=timepts, data_type=data_type, spec_method=spec_method,
                    smp_rate=smp_rate, return_phase=return_phase, keepdims=keepdims, **kwargs)


# =============================================================================
# Utility functions for generating single-trial jackknife pseudovalues
# =============================================================================
def two_sample_jackknife(func, data1, data2, *args, axis=0, shape=None, dtype=None, **kwargs):
    """
    Jackknife resampling of arbitrary statistic computed by `func` on `data1` and `data2`

    Parameters
    ----------
    func : callable
        Takes data1,data2 (and any other args) as input, returns statistic to be jackknifed

    data1,data2 : ndarray, shape:Any
        Data to compute jackknife resamples of. Shape is arbitrary but must be same.

    axis : int, default: 0
        Observation (trial) axis of `data1` and `data2`

    shape : tuple, shape=(ndims,), dtype=int, default: data1.shape
        Shape for output array `stat`. Default to same as input.

    dtype : str, default: data1.dtype
        dtype for output array `stat`. Default to same as input.

    *args, **kwargs :
        Any additional arguments passed directly to `func`

    Returns
    -------
    stat : ndarray, shape=Any
        Jackknife resamples of statistic. Same size as data1,data2.
    """
    if shape is None: shape = data1.shape
    if dtype is None: dtype = data1.dtype

    n = data1.shape[axis]
    ndim = data1.ndim

    # Create generator with n length-n vectors, each of which excludes 1 trial
    resamples = jackknifes(n)

    stat = np.empty(shape, dtype=dtype)

    # Do jackknife resampling -- estimate statistic w/ each observation left out
    for trial,sel in enumerate(resamples):
        # Index into <axis> of data and stat, with ':' for all other axes
        slices_in   = axis_index_slices(axis, sel, ndim)
        slices_out  = axis_index_slices(axis, [trial], ndim)
        stat[slices_out] = func(data1[slices_in], data2[slices_in], *args, **kwargs)

    return stat


def jackknife_to_pseudoval(x, xjack, n):
    """
    Calculate single-trial pseudovalues from leave-one-out jackknife estimates

    Parameters
    ----------
    x : ndarray, shape=Any
        Statistic computed on full observed data. Any arbitrary shape.

    xjack : ndarray, shape=Any
        Statistic computed on jackknife resampled data

    n : int
        Number of observations (trials) used to compute x

    Returns
    -------
    pseudo : ndarray, shape=Any
        Single-trial jackknifed pseudovalues. Same shape as xjack.
    """
    return n*x - (n-1)*xjack


# =============================================================================
# Data simulation and testing functions
# =============================================================================
def simulate_multichannel_oscillation(n_chnls, *args, **kwargs):
    """
    Generate synthetic multichannel data with oscillations at given parameters.

    For each channel, generate multiple trials with constant oscillatory signal +
    random additive Gaussian noise. Parameters can be shared across channels or set
    independently for each channel.

    Parameters
    ----------
    n_chnls : int
        Number of channels to simulate

    *args :
    **kwargs :
        Rest of place and keyword arguments are passed to :func:`simulate_oscillation`
        for each channel. Each argument can be given in one of two forms:

        (1) Scalar. A single value equivalent to the same argument to simulate_oscillation().
            That same value will be copied for all `n_chnls` channels.
        (2) Array-like, shape=(n_chnls,). List with different values for each channel,
            which will be iterated through.

        See :func:`simulate_oscillation` for details on arguments.

        Exceptions: If a value is set for `seed`, it will be used to set the
        random number generator only ONCE at the start of this function, so that
        the generation of all channel signals follow a reproducible random sequence.

        Simulated data must have same shape for all channels, so all channels must
        have the same value set for `time_range`, `smp_rate`, and `n_trials`.

    Returns
    -------
    data : ndarray, shape=(n_timepts,n_trials,n_chnls)
        Simulated multichannel data
    """
    # Set a single random seed, so all multichannel data follows reproducible sequence
    seed = kwargs.pop('seed',None)
    if seed is not None:
        if not np.isscalar(seed) and (len(seed) > 1):
            seed = seed[0]
            print("Multiple values not permitted for <seed>. Using first one.")
        set_random_seed(seed)

    # Ensure all channels have same values for these parameters that determine data size
    for param in ['time_range','smp_rate','n_trials']:
        assert (param not in kwargs) or np.isscalar(kwargs[param]) or \
                np.allclose(np.diff(kwargs[param]), 0), \
            ValueError("All simulated channels must have same value for '%s'" % param)

    # Replicate any single-valued arguments (args and kwargs) into (n_chnls,) lists
    args = list(args)
    for i,value in enumerate(args):
        if np.isscalar(value) or (len(value) != n_chnls): args[i] = n_chnls * [value]

    for key,value in kwargs.items():
        if np.isscalar(value) or (len(value) != n_chnls): kwargs[key] = n_chnls * [value]

    # Simulate oscillatory data for each channel
    for chnl in range(n_chnls):
        # Extract the i-th value for each arg and kwarg
        chnl_args = [value[chnl] for value in args]
        chnl_kwargs = {key:value[chnl] for key,value in kwargs.items()}

        chnl_data = simulate_oscillation(*chnl_args, **chnl_kwargs)

        # HACK Create the array to hold all channel data after we know the shape
        if chnl == 0:
            n_timepts,n_trials = chnl_data.shape
            data = np.empty((n_timepts,n_trials,n_chnls))

        data[:,:,chnl] = chnl_data

    return data


# =============================================================================
# Other helper functions
# =============================================================================
def _infer_data_type(data):
    """ Infer type of data signal given -- 'raw' (real) | 'spectral' (complex) """
    if np.isrealobj(data):  return 'raw'
    else:                   return 'spectral'


def _sync_raw_to_spectral(data1, data2, smp_rate, axis, time_axis, taper_axis,
                          spec_method, data_type, **kwargs):
    """
    Check data input to lfp-lfp synchrony methods.
    Determine if input data is raw or spectral, compute spectral transform if raw.
    """
    if data_type is None: data_type = _infer_data_type(data1)

    assert _infer_data_type(data2) == data_type, \
        ValueError("data1 and data2 must have same data_type (raw or spectral)")

    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, \
            "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, \
            "For raw/time-series data, need to input value for <time_axis>"
        if spec_method == 'multitaper': kwargs.update(keep_tapers=True)
        data1,freqs,timepts = spectrogram(data1, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        # Account for new frequency (and/or taper) axis prepended before time_axis
        n_new_axes = 2 if spec_method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if spec_method == 'multitaper': taper_axis = time_axis-1

    else:
        freqs = []
        timepts = []
        if spec_method == 'multitaper':
            assert taper_axis is not None, \
                ValueError("Must set value for taper_axis for multitaper spectral inputs")

    return data1, data2, freqs, timepts, axis, time_axis, taper_axis


def _sfc_raw_to_spectral(spkdata, lfpdata, smp_rate, axis, time_axis, taper_axis, timepts,
                         method, spec_method, data_type, **kwargs):
    """
    Check data input to spike-field coupling methods.
    Determine if input data is raw or spectral, compute spectral transform if raw.
    """
    if data_type is None:
        spk_data_type = _infer_data_type(spkdata)
        lfp_data_type = _infer_data_type(lfpdata)
        # Spike and field data required to have same type (both raw or spectral) for coherence
        if method == 'coherence':
            assert spk_data_type == lfp_data_type, \
                ValueError("Spiking (%s) and LFP (%s) data must have same data type" % \
                            (spk_data_type,lfp_data_type))
        # Spike data must be raw (not spectral) for phase-based methods
        else:
            assert _infer_data_type(spkdata) == 'raw', \
                ValueError("Spiking data must be given as raw, not spectral, data")

        data_type = lfp_data_type

    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert (timepts is not None) or (smp_rate is not None), \
            ValueError("If no value is input for <timepts>, must input value for <smp_rate>")

        # Ensure spkdata is boolean array with 1's for spiking times (ie not timestamps)
        assert spkdata.dtype != object, \
            TypeError("Spiking data must be converted from timestamps to boolean format")
        spkdata = spkdata.astype(bool)

        # Default timepts to range from 0 - n_timepts/smp_rate
        if timepts is None:     timepts = np.arange(lfpdata.shape[time_axis]) / smp_rate
        elif smp_rate is None:  smp_rate = 1 / np.diff(timepts).mean()

        # For multitaper, keep tapers, to be averaged across like trials below
        if spec_method == 'multitaper': kwargs.update(keep_tapers=True)
        # For multitaper phase sync, spectrogram window spacing must = sampling interval (eg 1 ms)
        if (spec_method == 'multitaper') and (method != 'coherence'):
            kwargs.update(spacing=1/smp_rate)

        # All spike-field coupling methods require spectral data for LFPs
        lfpdata,freqs,times = spectrogram(lfpdata, smp_rate, axis=time_axis, method=spec_method,
                                          data_type='lfp', **kwargs)

        # Coherence requires spectral data for both spikes and LFPs
        if method == 'coherence':
            spkdata,_,_ = spectrogram(spkdata, smp_rate, axis=time_axis, method=spec_method,
                                      data_type='spike', **kwargs)

        timepts_raw = timepts
        timepts = times + timepts[0]

        # Multitaper spectrogram loses window width/2 timepoints at either end of data
        # due to windowing. For phase-based SFC methods, must remove these timepoints
        # from spkdata to match. (Note: not issue for coherence, which operates in spectral domain)
        if (spec_method == 'multitaper') and (method != 'coherence'):
            retained_times = (timepts_raw >= timepts[0]) & (timepts_raw <= timepts[-1])
            spkdata = index_axis(spkdata, time_axis, retained_times)

        # Frequency axis always inserted just before time axis, so if
        # observation/trial axis is later, must increment it
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if spec_method == 'multitaper' else 1

        if method != 'coherence':
            # Set up indexing to preserve axes before/after time axis,
            # but insert n_new_axis just before it
            slicer = [slice(None)]*time_axis + \
                    [np.newaxis]*n_new_axes + \
                    [slice(None)]*(spkdata.ndim-time_axis)
            # Insert singleton dimension(s) into spkdata to match freq/taper dim(s) in lfpdata
            spkdata = spkdata[tuple(slicer)]

        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
        if spec_method == 'multitaper': taper_axis = time_axis - 1

    else:
        freqs = []
        # timepts = []

    return spkdata, lfpdata, freqs, timepts, smp_rate, axis, time_axis, taper_axis

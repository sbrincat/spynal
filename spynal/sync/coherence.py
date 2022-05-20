#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis of oscillatory neural coherence
"""
import numpy as np

from spynal.utils import axis_index_slices, setup_sliding_windows
from spynal.randstats.sampling import jackknifes
from spynal.randstats.utils import jackknife_to_pseudoval
from spynal.sync.helpers import _sync_raw_to_spectral, _sfc_raw_to_spectral


# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def coherence(data1, data2, axis=0, return_phase=False, transform=None, single_trial=None,
              spec_method='wavelet', data_type=None, smp_rate=None, time_axis=None,
              taper_axis=None, keepdims=True, **kwargs):
    """
    Compute coherence between pair of channels of raw or spectral (time-frequency) data
    (LFP or spikes)

    Coherence is a spectral analog of linear correlation that takes both phase and amplitude
    into account.

    Only parameters differing from :func:`synchrony` are described here.

    Parameters
    ----------
    transform : 'Z' or None, default: None
        Transform to apply to all computed coherence values.
        Set=None to return raw, untransformed coherence.
        Set='Z' to Z-transform coherence using Jarvis & Mitra (2001) method.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function.
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")

    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    # Check if raw data input. If so, compute spectral transform first
    data1, data2, freqs, timepts, axis, time_axis, taper_axis = \
        _sync_raw_to_spectral(data1, data2, smp_rate, axis, time_axis, taper_axis,
                              spec_method, data_type, **kwargs)

    # For multitaper, compute means across trials, tapers; degrees of freedom = 2*n_trials*n_tapers
    if spec_method == 'multitaper':
        reduce_axes = (axis,taper_axis)
        df = 2*np.prod([data1.shape[ax] for ax in reduce_axes])
    # Otherwise just do means across trials; df = 2*n_trials
    else:
        reduce_axes = axis
        df = 2*data1.shape[axis]

    # Setup actual function call for any transform to compute on raw coherence values
    if (transform is None) or callable(transform):
        transform_ = transform
    elif transform.lower() in ['z','ztransform']:
        transform_ = lambda coh: ztransform_coherence(coh, df)
    else:
        raise ValueError("Unsupported value '%s' set for <transform>" % transform)

    # Compute cross-spectrum and auto-spectrum of each channel
    auto_spec1 = data1*data1.conj()
    auto_spec2 = data2*data2.conj()
    cross_spec = data1*data2.conj()

    def _cross_auto_to_coh(cross_spec, auto_spec1, auto_spec2, axis,
                           return_phase, transform, keepdims):
        """ Compute coherence from cross-spectrum and pair of auto-spectra """
        # Average spectra across observatations (trials and/or tapers)
        # Note: .real deals with floating point error, converts complex dtypes to float
        auto_spec1 = np.mean(auto_spec1, axis=axis, keepdims=keepdims).real
        auto_spec2 = np.mean(auto_spec2, axis=axis, keepdims=keepdims).real
        cross_spec = np.mean(cross_spec, axis=axis, keepdims=keepdims)

        # Calculate complex coherency as cross-spectrum / product of square root of spectra
        coherency = cross_spec / np.sqrt(auto_spec1*auto_spec2)

        # Convert complex coherency -> coherence (magnitude of coherency)
        coherence = np.abs(coherency).real

        # Perform any requested tranform on coherence (eg z-scoring)
        if transform is not None: coherence = transform(coherence)

        # Optionally also extract coherence phase angle
        if return_phase:    return coherence, np.angle(coherency)
        else:               return coherence, None


    # Standard across-trial coherence estimator
    if single_trial is None:
        coh, dphi = _cross_auto_to_coh(cross_spec, auto_spec1, auto_spec2, reduce_axes,
                                        return_phase, transform_, keepdims)

    # Single-trial coherence estimator using jackknife resampling method
    else:
        # Jackknife resampling of coherence statistic (this is the 'richter' estimator)
        # Note: Allow for reduction along taper axis within resampled stat function, but only
        #       resample across trial axis--want sync estimates for single-trials, not tapers
        jackfunc = lambda s12,s1,s2: _cross_auto_to_coh(s12, s1, s2, reduce_axes,
                                                        False, transform_, True)[0]
        coh_shape = list(cross_spec.shape)
        if spec_method == 'multitaper': coh_shape[taper_axis] = 1
        n_jack = coh_shape[axis]
        ndim = len(coh_shape)

        # Create generator with n length-n vectors, each of which excludes 1 trial
        resamples = jackknifes(n_jack)

        # Do jackknife resampling -- estimate statistic w/ each observation left out
        coh = np.empty(coh_shape, dtype=float)
        for trial,sel in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            slices_in   = axis_index_slices(axis, sel, ndim)
            slices_out  = axis_index_slices(axis, [trial], ndim)
            coh[slices_out] = jackfunc(cross_spec[slices_in],
                                       auto_spec1[slices_in], auto_spec2[slices_in])

        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        # Note: n here is the number of jackknifes computed = n_trials
        if single_trial == 'pseudo':
            coh_full = jackfunc(cross_spec, auto_spec1, auto_spec2)
            coh = jackknife_to_pseudoval(coh_full, coh, data1.shape[axis])

        if not keepdims and (spec_method == 'multitaper'): coh = coh.squeeze(axis=taper_axis)

    if return_phase:    return coh, freqs, timepts, dphi
    else:               return coh, freqs, timepts


# =============================================================================
# Spike-Field Synchrony functions
# =============================================================================
def spike_field_coherence(spkdata, lfpdata, axis=0, time_axis=None, taper_axis=None,
                          timepts=None, transform=None,
                          data_type=None, spec_method='multitaper', smp_rate=None,
                          return_phase=False, keepdims=True, **kwargs):
    """
    Compute pairwise coherence between single-channel spiking data and LFP data

    Only parameters differing from :func:`spike_field_coupling` are described here.

    Parameters
    ----------
    transform : 'Z' or None, default: None
        Transform to apply to all computed coherence values.
        Set=None to return raw, untransformed coherence.
        Set='Z' to Z-transform coherence using Jarvis & Mitra (2001) method.

    **kwargs :
        Any other keyword args passed as-is to spectrogram() function
    """
    if axis < 0: axis = lfpdata.ndim + axis
    if time_axis < 0: time_axis = lfpdata.ndim + time_axis

    # Check if raw data input. If so, compute spectral transform first
    spkdata, lfpdata, freqs, timepts, smp_rate, axis, time_axis, taper_axis = \
        _sfc_raw_to_spectral(spkdata, lfpdata, smp_rate, axis, time_axis, taper_axis, timepts,
                             'coherence', spec_method, data_type, **kwargs)

    extra_args = dict(axis=axis, data_type='spectral', spec_method=spec_method,
                      transform=transform, return_phase=return_phase)
    if spec_method == 'multitaper': extra_args.update(taper_axis=taper_axis)

    if return_phase:
        coh,_,_,phi = coherence(spkdata, lfpdata, keepdims=keepdims, **extra_args)
        return coh,freqs,timepts,None,phi
    else:
        coh,_,_ = coherence(spkdata, lfpdata, keepdims=keepdims, **extra_args)
        return coh,freqs,timepts,None


# =============================================================================
# Helper functions
# =============================================================================
def ztransform_coherence(coh, df, beta=23/20):
    """
    z-transform coherence values to render them approximately normally distributed

    Parameters
    ----------
    coh : ndarray, shape=Any
        Raw coherence values

    df : int
        Degrees of freedom of coherence estimates.
        For multitaper spectral estimates, this is usually df = 2*n_trials*n_tapers.
        For other estimates, this is usually df = 2*n_trials.

    beta : scalar, default: 23/20
        Mysterious number from Jarvis & Mitra to make output z-scores approximately normal

    Returns
    -------
    z : ndarray, shape=Any
        z-transformed coherence. Same shape as `coh`.

    References
    ----------
    - Jarvis & Mitra (2001) Neural Computation https://doi.org/10.1162/089976601300014312
    - Hipp, Engel, Siegel (2011) Neuron https://doi.org/10.1016/j.neuron.2010.12.027
    """
    return beta*(np.sqrt(-(df-2)*np.log(1-coh**2)) - beta)

# -*- coding: utf-8 -*-
"""
Spectral analysis, signal processing, and continuous (LFP/EEG) data preprocessing

This module is the base module for the `spectra` module. It contains all the
high-level API functions. Lower-level functions can be found in the other spectra modules.

Overview
--------
Functionality for computing frequency spectra as well as time-frequency (spectrogram) transforms.

Options to compute spectral analysis using multitaper, wavelet, band-pass filtering, or spectral
burst analysis methods.

Options to return full complex spectral data, spectral power, phase, real or imaginary part, etc.

Also includes functions for preprocessing, postprocessing, plotting of continuous/spectral data.

Most functions perform operations in a mass-univariate manner. This means that
rather than embedding function calls in for loops over channels, trials, etc., like this::

    for channel in channels:
        for trial in trials:
            results[trial,channel] = compute_something(data[trial,channel])

You can instead execute a single call on ALL the data, labeling the relevant axis
for the computation (usually time here), and it will run in parallel (vectorized)
across all channels, trials, etc. in the data, like this:

``results = compute_something(data, axis)``


Function list
-------------
General spectral analysis
^^^^^^^^^^^^^^^^^^^^^^^^^
- spectrum :          Frequency spectrum of data
- spectrogram :       Time-frequency spectrogram of data
- power_spectrum :    Power spectrum of data
- power_spectrogram : Power of time-frequency transform
- phase_spectrogram : Phase of time-frequency transform

Wavelet spectral analysis (spectra.wavelet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- wavelet_spectrum :    Wavelet-based frequency spectrum
- wavelet_spectrogram : Time-frequency continuous wavelet transform
- compute_wavelets :    Compute wavelets for use in wavelet spectral analysis
- wavelet_bandwidth :   Compute time,frequency bandwidths for set of wavelets
- wavelet_edge_extent : Compute extent of edge effects for set of wavelets

Multitaper spectral analysis (spectra.multitaper)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- multitaper_spectrum :     Multitaper (DPSS) frequency spectrum
- multitaper_spectrogram :  Multitaper (DPSS) time-frequency spectrogram
- compute_tapers :          Compute DPSS tapers for use in multitaper spectral analysis

Bandpass-filtering spectral analysis (spectra.bandfilter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- bandfilter_spectrum :     Band-filtered frequency spectrum
- bandfilter_spectrogram :  Band-filtered, Hilbert-transformed time-frequency of data
- set_filter_params :       Set filter coefficients for use in band-filtered analysis

Other spectral analyses
^^^^^^^^^^^^^^^^^^^^^^^
- itpc :                Intertrial phase clustering (analysis of phase locking to trial events)
- burst_analysis :      Compute oscillatory burst analysis of Lundqvist et al 2016

Preprocessing (spectra.preprocess)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- cut_trials :          Cut LFPs/continuous data into trial segments
- realign_data :        Realign LFPs/continuous data to new within-trial event
- remove_dc :           Remove constant DC component of signals
- remove_evoked :       Remove phase-locked evoked potentials from signals

Postprocesssing (spectra.postprocess)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- pool_freq_bands :         Average spectral data within set of frequency bands
- pool_time_epochs :        Average spectral data within set of time epochs
- one_over_f_norm :         Normalize to correct for 1/f distribution of spectral power

Utilities (spectra.utils)
^^^^^^^^^^^^^^^^^^^^^^^^^
- get_freq_sampling :       Frequency sampling vector for a given FFT-based computation
- complex_to_spec_type :    Convert complex Fourier transform output to power/phase/real/imag/etc.
- one_sided_to_two_sided :  Convert 1-sided Fourier transform output to 2-sided equivalent
- simulate_oscillation :    Generates simulated oscillation-in-noise data

Plotting
^^^^^^^^
- plot_spectrum :       Plot frequency spectrum as a line plot, handling freq axis properly
- plot_spectrogram :    Plot time-frequency spectrogram as a heatmap plot


Dependencies
------------
- pyfftw :              Python wrapper around FFTW, the speedy FFT library

Function reference
------------------
"""
# Created on Thu Oct  4 15:28:15 2018
#
# @author: sbrincat

from math import floor, ceil, log2, sqrt
import numpy as np
import matplotlib.pyplot as plt

from spynal.utils import set_random_seed, iarange
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.plots import plot_line_with_error_fill, plot_heatmap
from spynal.spectra.wavelet import wavelet_spectrum, wavelet_spectrogram, compute_wavelets, \
                                   wavelet_bandwidth, wavelet_edge_extent
from spynal.spectra.multitaper import multitaper_spectrum, multitaper_spectrogram
from spynal.spectra.bandfilter import bandfilter_spectrum, bandfilter_spectrogram
from spynal.spectra.postprocess import one_over_f_norm, pool_freq_bands 
from spynal.spectra.helpers import _infer_freq_scale, _frequency_plot_settings

try:
    import torch
    torch_avail = True
except:
    torch_avail = False

# =============================================================================
# General spectral analysis functions
# =============================================================================
def spectrum(data, smp_rate, axis=0, method='multitaper', data_type='lfp', spec_type='complex',
             removeDC=True,torch_avail=torch_avail, max_bin_size=1e9, **kwargs):
    """
    Compute frequency spectrum of data using given method

    Parameters
    ----------
    data : ndarray,shape=(...,n_samples,...)
        Data to compute spectral analysis of.
        Arbitrary shape; spectral analysis is computed along `axis`.

    smp_rate : scalar
        Data sampling rate (Hz)

    axis : int, default: 0 (1st axis)
        Axis of `data` to do spectral analysis on (usually time dimension).

    method : {'multitaper','wavelet','bandfilter'}, default: 'multitaper'
        Specific spectral analysis method to use:

        - 'multitaper' : Multitaper spectral analysis in :func:`.multitaper_spectrum`
        - 'wavelet' : Wavelet analysis in :func:`.wavelet_spectrum`
        - 'bandfilter' : Bandpass filtering in :func:`.bandfilter_spectrum`

    data_type : {'lfp','spike'}, default: 'lfp'
        Type of signal in data

    spec_type : {'complex','power','phase','real','imag'}, default: 'complex'
        Type of spectral signal to return. See :func:`.complex_to_spec_type` for details.

    removeDC : bool, default: True
        If True, subtracts off mean DC component across `axis`, making signals zero-mean
        before spectral analysis.

    **kwargs :
        All other kwargs passed directly to method-specific spectrum function

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs,...), dtype=complex or float.
        Frequency spectrum of given type computed with given method.
        Frequency axis is always inserted in place of `axis`.
        Note: 'multitaper' method will return with additional taper
        axis inserted after just after `axis` if `keep_tapers` is True.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqs,) or (n_freqbands,2)
        For `method` == 'bandfilter': List of (low,high) cut frequencies (Hz) used to
        generate `spec`, shape=(n_freqbands,2)
        For other methods: List of frequencies in `spec` (Hz), shape=(n_freqs,)
    """
    
    method = method.lower()
    assert data_type in ['lfp','spike'], \
        ValueError("<data_type> must be 'lfp' or 'spike' ('%s' given)" % data_type)

    if method == 'multitaper':      spec_fun = multitaper_spectrum
    elif method == 'wavelet':       spec_fun = wavelet_spectrum
    elif method == 'bandfilter':    spec_fun = bandfilter_spectrum
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    spec,freqs = spec_fun(data,smp_rate,axis=axis,data_type=data_type,spec_type=spec_type,
                          removeDC=removeDC,torch_avail=torch_avail,max_bin_size=max_bin_size,**kwargs)

    return spec, freqs


def spectrogram(data, smp_rate, axis=0, method='wavelet', data_type='lfp', spec_type='complex',
                removeDC=True, torch_avail=torch_avail, max_bin_size=1e9, **kwargs):
    """
    Compute time-frequency transform of data using given method

    Parameters
    ----------
    data : ndarray, shape=(...,n_samples,...)
        Data to compute spectral analysis of.
        Arbitrary shape; spectral analysis is computed along `axis`.

    smp_rate : scalar
        Data sampling rate (Hz)

    axis : int, default: 0 (1st axis)
        Axis of `data` to do spectral analysis on (usually time dimension).

    method : {'multitaper','wavelet','bandfilter','burst'}, default: 'wavelet'
        Specific spectral analysis method to use:

        - 'multitaper' : Multitaper spectral analysis in :func:`.multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`.wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`.bandfilter_spectrogram`
        - 'burst' : Oscillatory burst analysis in :func:`.burst_analysis`

    data_type : {'lfp','spike'}, default: 'lfp'
        Type of signal in data

    spec_type : {'complex','power','phase','real','imag'}, default: 'complex'
        Type of spectral signal to return. See :func:`.complex_to_spec_type` for details.

    removeDC : bool, default: True
        If True, subtracts off mean DC component across `axis`, making signals zero-mean
        before spectral analysis.

    **kwargs :
        All other kwargs passed directly to method-specific spectrogram function

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs,n_timepts,...), dtype=complex or float.
        Time-frequency spectrogram of given type computed with given method.
        Frequency axis is always inserted just before time axis.
        Note: 'multitaper' method will return with additional taper
        axis inserted between freq and time axes if `keep_tapers` is True.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqs,) or (n_freqbands,2)
        For `method` == 'bandfilter': List of (low,high) cut frequencies (Hz) used to
        generate `spec`, shape=(n_freqbands,2)
        For other methods: List of frequencies in `spec` (Hz), shape=(n_freqs,)

    timepts : ndarray, shape=(n_timepts,)
        List of time points / time window centers in `spec` (in s, referenced to start of data)
    """
    method = method.lower()

    # Special case: Lundqvist oscillatory burst analysis
    if (spec_type == 'burst') or (method == 'burst'):
        assert data_type == 'lfp', ValueError("<data_type> must be 'lfp' for burst analysis")

        spec,freqs,timepts = burst_analysis(data, smp_rate, axis=axis, removeDC=removeDC, **kwargs)

    else:
        assert data_type in ['lfp','spike'], \
            ValueError("<data_type> must be 'lfp' or 'spike' ('%s' given)" % data_type)

        if method == 'wavelet':         spec_fun = wavelet_spectrogram
        elif method == 'multitaper':    spec_fun = multitaper_spectrogram
        elif method == 'bandfilter':    spec_fun = bandfilter_spectrogram
        else:
            raise ValueError("Unsupported value set for <method>: '%s'" % method)

        spec,freqs,timepts = spec_fun(data, smp_rate, axis=axis, data_type=data_type,
                                      spec_type=spec_type, removeDC=removeDC, torch_avail=torch_avail,max_bin_size=max_bin_size,**kwargs)

    return spec, freqs, timepts


def power_spectrum(data, smp_rate, axis=0, method='multitaper',torch_avail=torch_avail,max_bin_size=1e9, **kwargs):
    """
    Convenience wrapper around spectrum() to compute **power** spectrum of data with given method

    See :func:`spectrum` for details
    """
    return spectrum(data, smp_rate, axis=axis, method=method, torch_avail=torch_avail,max_bin_size=max_bin_size,spec_type='power', **kwargs)


def power_spectrogram(data, smp_rate, axis=0, method='wavelet',torch_avail=torch_avail, max_bin_size=1e9,**kwargs):
    """
    Convenience wrapper around spectrogram() to compute time-frequency **power** with given method

    See :func:`spectrogram` for details
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, torch_avail=torch_avail,max_bin_size=max_bin_size,spec_type='power', **kwargs)


def phase_spectrogram(data, smp_rate, axis=0, method='wavelet',torch_avail=torch_avail, max_bin_size=1e9,**kwargs):
    """
    Convenience wrapper around spectrogram() to compute **phase** of time-frequency transform

    See :func:`spectrogram` for details
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, torch_avail=torch_avail,max_bin_size=max_bin_size,spec_type='phase', **kwargs)


# =============================================================================
# Other spectral analysis functions
# =============================================================================
def itpc(data, smp_rate, axis=0, method='wavelet', itpc_method='PLV', trial_axis=None, **kwargs):
    """
    Intertrial phase clustering (ITPC) measures frequency-specific phase locking of continuous
    neural activity (eg LFPs) to trial events. A spectral analog (roughly) of evoked potentials.

    Complex time-frequency representation is first computed (using some method), then the
    complex vector mean is computed across trials, and it's magnitude is returned as ITPC.

    aka "intertrial coherence", "intertrial phase-locking value/factor"

    Parameters
    ----------
    data : ndarray, shape=(...,n_timepts,...,n_trials,...)
        Data to compute ITPC of, aligned to some within-trial event.
        Arbitrary shape; spectral analysis is computed along `axis` (usually time),
        and ITPC is computed along `trial_axis`.

    smp_rate : scalar
        Data sampling rate (Hz)

    axis : int, default: 0 (1st axis)
        Axis of `data` to do spectral analysis on (usually time dimension).

    method : {'multitaper','wavelet','bandfilter'}, default: 'wavelet'
        Specific underlying spectral analysis method to use:

        - 'multitaper' : Multitaper spectral analysis in :func:`.multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`.wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`.bandfilter_spectrogram`

    itpc_method : {'PLV','Z','PPC'}, default: 'PLV'
        Method to use for computing intertrial phase clustering:

        - 'PLV' : Phase locking value (length of cross-trial complex vector mean)
            Standard/traditional measure of ITPC, but is positively biased
            for small n, and is biased > 0 even for no phase clustering.
        - 'Z' :   Rayleigh's Z normalization of PLV (Z = n*PLV**2). Reduces small-n bias.
        - 'PPC' : Pairwise Phase Consistency normalization (PPC = (n*PLV**2 - 1)/(n - 1))
            of PLV. Debiased and has expected value 0 for no clustering.

    **kwargs :
        All other kwargs passed directly to method-specific spectrogram function

    Returns
    -------
    ITPC : ndarray, shape=(...,n_freqs,n_timepts,...)
        Time-frequency spectrogram representation of intertrial phase clustering.
        Frequency axis is always inserted just before time axis, and trial axis is
        removed, but otherwise shape is same as input `data`.

    freqs : ndarray, shape=(n_freqs,) or (n_freqbands,2)
        For `method` == 'bandfilter': List of (low,high) cut frequencies (Hz) used to
        generate `ITPC`, shape=(n_freqbands,2)
        For other methods: List of frequencies in `spec` (Hz), shape=(n_freqs,)

    timepts : ndarray, shape=(n_timepts,)
        List of time points / time window centers in `ITPC` (in s, referenced to start of data)

    References
    ----------
    Cohen "Analyzing Neural Time Series Data" http://dx.doi.org/10.7551/mitpress/9609.001.0001 Ch. 19
    """
    method = method.lower()
    itpc_method = itpc_method.lower()
    if axis < 0:        axis = data.ndim + axis
    if trial_axis < 0:  trial_axis = data.ndim + trial_axis

    n = data.shape[trial_axis]

    if method == 'wavelet':         spec_fun = wavelet_spectrogram
    elif method == 'multitaper':    spec_fun = multitaper_spectrogram
    elif method == 'bandfilter':    spec_fun = bandfilter_spectrogram
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    if method == 'multitaper': kwargs.update(keep_tapers=True)

    # Compute spectrogram using given method
    spec,freqs,timepts = spec_fun(data, smp_rate, axis=axis, spec_type='complex', **kwargs)
    # Account for new frequency (and/or taper) axis prepended before time axis
    n_new_axes = 2 if method == 'multitaper' else 1
    if trial_axis >= axis: trial_axis += n_new_axes
    axis += n_new_axes
    if method == 'multitaper': taper_axis = axis - 1

    spec = spec / np.abs(spec)     # Normalize spectrum to unit length

    # Compute mean resultant length (aka PLV) = length of complex vector mean across trials
    # For multitaper spectral analysis, take mean across both trials and tapers
    reduce_axes = (trial_axis,taper_axis) if method == 'multitaper' else trial_axis
    ITPC = np.abs(spec.mean(axis=reduce_axes))

    # PLV -- we are done, return ITPC as-is
    if itpc_method in ['plv','plf']:         pass

    # PPC -- debiasing normalization
    elif itpc_method in ['ppc']:             ITPC = (n*ITPC**2 - 1) / (n - 1)

    # Rayleigh's Z -- debiasing normalization
    elif itpc_method in ['z','rayleighz']:   ITPC = n*ITPC**2

    else:
        raise ValueError("%s in an unsupported ITPC method" % itpc_method)

    return ITPC, freqs, timepts

intertrial_phase_clustering = itpc
""" Alias of :func:`itpc`. See there for details. """


def burst_analysis(data, smp_rate, axis=0, trial_axis=-1, threshold=2, min_cycles=3,
                   method='wavelet', spec_type='power', freq_exp=None,
                   bands=((20,35),(40,65),(55,90),(70,100)),
                   window=None, timepts=None,max_bin_size=1e9,torch_avail=torch_avail, **kwargs):
    """
    Oscillatory burst analysis of Lundqvist et al 2016.

    Computes oscillatory power, z-scores within each frequency band, thresholds at given
    z `threshold`, labels as burst "ON" times timepoints > `threshold` for at least
    `min_cycles` duration.

    To compute burst rate, simply take mean of computed `bursts` across trial axis.

    Default argument values approximate analysis as implemented in Lundqvist 2016.

    Parameters
    ----------
    data : ndarray, shape=(...,n_samples,...)
        Data to compute spectral analysis of.
        Arbitrary shape; spectral analysis is computed along `axis`.

    smp_rate : scalar
        Data sampling rate (Hz)

    smp_rate : scalar
        Data sampling rate (Hz)

    axis : int, default: 0 (1st axis)
        Axis of `data` to do spectral analysis on (usually time dimension).

    trial_axis : int, default: -1 (last axis of `data`)
        Axis of `data` corresponding to trials/observations.

    threshold : scalar, default: 2 (2 SDs above mean)
        Threshold power level for detecting bursts, given in SDs above the mean

    min_cycles : scalar, default: 3
        Minimal length of contiguous above-threshold period to be counted as a burst,
        given in number of oscillatory cycles at each frequency (or band center).

    method : {'multitaper','wavelet','bandfilter'}, default: 'wavelet'
        Specific spectral analysis method to use:

        - 'multitaper' : Multitaper spectral analysis in :func:`.multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`.wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`.bandfilter_spectrogram`

        Note: In the original paper, multitaper was used, but all three
        methods were claimed to produced similar results.

    spec_type : {'power','magnitude'}, default: 'power'
        Type of spectral signal to compute:

        - 'power' : Spectral power, ie square of signal envelope
        - 'magnitude' : Square root of power, ie signal envelope

    freq_exp : float, default: None (do no frequency normalization)
        This can be used to normalize out 1/f^a effects in power before
        band-pooling and burst detection). This gives the exponent on the frequency
        ('a' in 1/f^a).  Set = 1 to norm by 1/f.  Set = None for no normalization.

    bands : array-like, shape=(n_freqbands,2), default: ((20,35),(40,65),(55,90),(70,100))
        List of (low,high) cut frequencies for each band to compute bursts within.
        Set low cut = 0 for low-pass, set high cut >= smp_rate/2 for high-pass,
        otherwise assumes band-pass. Default samples ~ beta, low/med/high gamma.
        Set = None to compute bursts at each frequency in spectral transform.

    window : array-like, shape=(2,), default: None (compute over entire data time range)
        Optional (start,end) of time window to compute mean,SD for burst amplitude threshold
        within (in same units as `timepts`).

    timepts : array-like, shape=(n_timepts,), default: None
        Time sampling vector for data (usually in s).
        Necessary if `window` is set, but unused otherwise.

    **kwargs :
        Any other kwargs passed directly to :func:`spectrogram` function

    Returns
    -------
    bursts : ndarray, shape=(...,n_freq[band]s,n_timepts_out,...), dtype=bool
        Binary array labelling timepoints within detected bursts in each trial and frequency band.
        Same shape as `data`, but with frequency axis prepended immediately before time `axis`.

    freqs : ndarray, shape=(n_freq[band],)
        List of center frequencies of bands in `bursts`

    timepts : ndarray=(n_timepts_out,)
        List of time points / time window centers in `bursts` (in s, referenced to start of data).

    References
    ----------
    Lundqvist et al (2016) Neuron https://doi.org/10.1016/j.neuron.2016.02.028
    Lundqvist et al (2018) Nature Comms https://doi.org/10.1038/s41467-017-02791-8
    """
    # todo  Gaussian fits for definining burst f,t extent?
    # todo  Option input of spectral data?
    # todo  Add optional sliding trial window for mean,SD
    method = method.lower()
    spec_type = spec_type.lower()
    if bands is not None: bands = np.asarray(bands)
    if bands.ndim == 1: bands = bands[np.newaxis,:] # Ensure bands is (n_bands,2) even if n_bands=1
    if axis < 0: axis = data.ndim + axis
    if (trial_axis is not None) and (trial_axis < 0):  trial_axis = data.ndim + trial_axis

    assert axis != trial_axis, \
        ValueError("Time and trial axes can't be same. Set trial_axis=None if no trials in data.")
    if window is not None:
        assert len(window) == 2, \
            ValueError("Window for computing mean,SD should be given as (start,end) (len=2)")
        assert timepts is not None, \
            ValueError("Need to input <timepts> to set a window for computing mean,SD")
    assert spec_type in ['power','magnitude'], \
        ValueError("spec_type must be 'power'|'magnitude' for burst analysis (%s given)"
                   % spec_type)

    # Move array axes so time axis is 1st and trials last (n_timepts,...,n_trials)
    if (axis == data.ndim-1) and (trial_axis == 0):
        data = np.swapaxes(data, axis, trial_axis)
    else:
        if axis != 0:
            data = np.moveaxis(data,axis,0)
        # If data has no trial axis, temporarily append singleton to end to simplify code
        if trial_axis is None:
            data = data[...,np.newaxis]
        elif trial_axis != data.ndim-1:
            data = np.moveaxis(data, trial_axis, -1)
    data_shape = data.shape
    data_ndim = data.ndim
    # Standardize data array to shape (n_timepts,n_data_series,n_trials)
    if data_ndim > 3:       data = data.reshape((data.shape[0],-1,data.shape[-1]))
    elif data_ndim == 2:    data = data[:,np.newaxis,:]
    n_timepts,n_series,n_trials = data.shape

    # Set default values to appoximate Lundqvist 2016 analysis, unless overridden by inputs
    if method == 'wavelet':
        # Sample frequency at 1 Hz intervals from min to max frequency in requested bands
        if ('freqs' not in kwargs) and (bands is not None):
            kwargs['freqs'] = np.arange(bands.min(),bands.max()+1,1)

    # For bandfilter method, if frequency bands not set explicitly, set it with value for <bands>
    elif method == 'bandfilter':
        if ('freqs' not in kwargs) and (bands is not None): kwargs['freqs'] = bands

    # Compute time-frequency power from raw data -> (n_freqs,n_timepts,n_data_series,n_trials)
    data,freqs,times = spectrogram(data, smp_rate, axis=0, method=method, spec_type=spec_type,torch_avail=torch_avail,max_bin_size=max_bin_size,
                                   **kwargs)
    timepts = times + timepts[0] if timepts is not None else times
    n_timepts = len(times)
    dt = np.mean(np.diff(times))

    # Normalize computed power by 1/f**exp to normalize out 1/f distribution of power
    if freq_exp is not None:
        data = one_over_f_norm(data, axis=0, freqs=freqs, exponent=freq_exp)

    # If requested, pool data within given frequency bands
    # (skip for bandfilter spectral analysis, which already returns frequency bands)
    if (method != 'bandfilter') and (bands is not None):
        data = pool_freq_bands(data, bands, axis=0, freqs=freqs, func='mean')
        freqs = bands
    n_freqs = data.shape[0]

    # Compute mean,SD of each frequency band and data series (eg channel)
    # across all trials (axis -1) and timepoints (axis 1)
    if window is None:
        mean = data.mean(axis=(1,-1), keepdims=True)
        sd   = data.std(axis=(1,-1), ddof=0, keepdims=True)

    # Compute mean,SD of each freq band/channel across all trials and timepoints w/in time window
    else:
        tbool = (timepts >= window[0]) & (timepts <= window[1])
        mean = data.compress(tbool,axis=1).mean(axis=(1,-1), keepdims=True)
        sd   = data.compress(tbool,axis=1).std(axis=(1,-1), ddof=0, keepdims=True)

    # Compute z-score of data and threshold -> boolean array of candidate burst times
    z = (data - mean) / sd
    bursts = z > threshold

    tsmps = np.arange(n_timepts)


    def _screen_bursts(burst_bool, min_samples, start):
        """ Subfunction to evaluate/detect bursts in boolean time series of candidate bursts """
        # Find first candidate burst in trial (timepoints of all candidate burst times)
        if start is None:   on_times = np.nonzero(burst_bool)[0]
        # Find next candidate burst in trial
        else:               on_times = np.nonzero(burst_bool & (tsmps > start))[0]

        # If no (more) bursts in time series, return data as is, we are done
        if len(on_times) == 0:  return burst_bool
        # Otherwise, get onset index of first/next candidate burst
        else:                   onset = on_times[0]

        # Find non-burst timepoints in remainder of time series
        off_times = np.nonzero(~burst_bool & (tsmps > onset))[0]

        # If no offset found, burst must extend to end of data
        if len(off_times) == 0: offset = len(burst_bool)
        # Otherwise, get index of offset of current burst = next off time - 1
        else:                   offset = off_times[0] - 1

        # Determine if length of current candidate burst meets minimum duration
        # If not, delete it from data (set all timepoints w/in it to False)
        burst_len = offset - onset + 1
        if burst_len < min_samples:  burst_bool[onset:(offset+1)] = False

        # todo trim bursts to half-max point? (using Gaussian fits or raw power?)

        # If offset is less than minimum burst length from end of data, we are done.
        # Ensure no further timepoints are labelled "burst on" and return data
        if (len(burst_bool) - offset) < min_samples:
            burst_bool[(offset+1):] = False
            return burst_bool
        # Otherwise, call function recursively, now starting search just after current burst offset
        else:
            return _screen_bursts(burst_bool,min_samples,offset+1)


    # Screen all candidate bursts across freqs/trials/chnls to ensure they meet minimum duration
    for i_freq,freq in enumerate(freqs):
        # Compute center frequency of frequency band
        if not np.isscalar(freq): freq = np.mean(freq)
        # Convert minimum length in oscillatory cycles -> samples
        min_samples = ceil(min_cycles * (1/freq) / dt)

        for i_trial in range(n_trials):
            for i_series in range(n_series):
                # Extract burst time series for current (frequency, data series, trial)
                series = bursts[i_freq,:,i_series,i_trial]
                if not series.any(): continue

                bursts[i_freq,:,i_series,i_trial] = _screen_bursts(series,min_samples,None)

    # Reshape data array to ~ original dimensionality -> (n_freqs,n_timepts,...,n_trials)
    if data_ndim > 3:       bursts = bursts.reshape((n_freqs,n_timepts,data_shape[1:-1],n_trials))
    elif data_ndim == 2:    bursts = bursts.squeeze(axis=2)

    # Move array axes back to original locations
    if (axis == data_ndim-1) and (trial_axis == 0):
        bursts = np.moveaxis(bursts,-1,0)   # Move trial axis back to 0
        bursts = np.moveaxis(bursts,1,-1)   # Move freq axis to end
        bursts = np.moveaxis(bursts,1,-1)   # Move time axis to end (after freq)
    else:
        if axis != 0:                   bursts = np.moveaxis(bursts,(0,1),(axis,axis+1))
        if trial_axis is None:          bursts = np.squeeze(bursts,-1)
        elif trial_axis != data_ndim-1:
            if trial_axis > axis:       bursts = np.moveaxis(bursts,-1,trial_axis+1)
            else:                       bursts = np.moveaxis(bursts,-1,trial_axis)

    return bursts, freqs, timepts


# =============================================================================
# Plotting functions
# =============================================================================
def plot_spectrum(freqs, data, ax=None, ylim=None, color=None, **kwargs):
    """
    Plot frequency spectrum as a line plot.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqs,)
        Frequency sampling (x-axis) vector for data (Hz).
        May be linearly or logarithmically sampled; we handle appropriately.

    data : ndarray, shape=(n_freqs,)
        Frequency spectrum data to plot (y-axis)

    ax : Pyplot Axis object, default: plt.gca()
        Axis to plot into

    ylim : array-like, shape=(2,), Default: (data.min(),data.max()) +/- 5%
        Plot y-axis limits: (min,max)

    color : Color spec, default: <Matplotlib default plot color>
        Color to plot in

    **kwargs :
        Any additional keyword args are interpreted as parameters of plt.axes()
        (settable Axes object attributes) or plt.plot() (Line2D object attributes),
        and passsed to the proper function.

    Returns
    -------
    lines : List of Line2D objects
        Output of plt.plot()

    ax : Axis object
        Axis plotted into
    """
    freqs,fticks,fticklabels = _frequency_plot_settings(freqs)

    lines, _, ax = plot_line_with_error_fill(freqs, data, ax=ax, ylim=ylim, color=color, **kwargs)

    plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
    ax.set_xticks(fticks)
    ax.set_xticklabels(fticklabels)
    # No need to return list of lists if only plotting one data series
    if (data.ndim == 1) or (data.shape[0] == 1): lines = lines[0]

    return lines, ax


def plot_spectrogram(timepts, freqs, data, ax=None, clim=None, cmap='viridis', **kwargs):
    """
    Plot time-frequency spectrogram as a heatmap plot.

    Parameters
    ----------
    timepts : array-like, shape=(n_timepts,)
        Time sampling (x-axis) vector for data

    freqs : array-like, shape=(n_freqs,)
        Frequency sampling (y-axis) vector for data (Hz).
        May be linearly or logarithmically sampled; we handle appropriately.

    data : ndarray, shape=(n_freqs,n_timepts)
        Time-frequency (spectrogam) data to plot on color axis

    ax : Pyplot Axis object, default: plt.gca()
        Axis to plot into

    clim : array-like, shape=(2,), default: (data.min(),data.max())
        Color axis limits: (min,max)

    cmap  : str | Colormap object. default: 'viridis' (linear dark-blue to yellow colormap)
        Colormap to plot heatmap in, given either as name of matplotlib colormap or custom
        matplotlib.colors.Colormap object instance.

    **kwargs :
        Any additional keyword args are interpreted as parameters of :func:`plt.axes`
        (settable Axes object attributes) or :func:`plt.imshow` (AxesImage object attributes).

    Returns
    -------
    img : AxesImage object
        Output of ax.imshow(). Allows access to image properties.

    ax : Axis object
        Axis plotted into.
    """
    freqs,fticks,fticklabels = _frequency_plot_settings(freqs)

    img, ax = plot_heatmap(timepts, freqs, data, ax=ax, clim=clim, cmap=cmap,
                           origin='lower', **kwargs)

    plt.grid(axis='y',color=[0.75,0.75,0.75],linestyle=':')
    ax.set_yticks(fticks)
    ax.set_yticklabels(fticklabels)

    return img, ax

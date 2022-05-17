# -*- coding: utf-8 -*-
"""
Spectral analysis, signal processing, and continuous (LFP/EEG) data preprocessing

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

Multitaper spectral analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- multitaper_spectrum :     Multitaper (DPSS) frequency spectrum
- multitaper_spectrogram :  Multitaper (DPSS) time-frequency spectrogram
- compute_tapers :          Compute DPSS tapers for use in multitaper spectral analysis

Wavelet spectral analysis
^^^^^^^^^^^^^^^^^^^^^^^^^
- wavelet_spectrum :    Wavelet-based frequency spectrum
- wavelet_spectrogram : Time-frequency continuous wavelet transform
- compute_wavelets :    Compute wavelets for use in wavelet spectral analysis
- wavelet_bandwidth :   Compute time,frequency bandwidths for set of wavelets
- wavelet_edge_extent : Compute extent of edge effects for set of wavelets

Bandpass-filtering spectral analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- bandfilter_spectrum :     Band-filtered frequency spectrum
- bandfilter_spectrogram :  Band-filtered, Hilbert-transformed time-frequency of data
- set_filter_params :       Set filter coefficients for use in band-filtered analysis

Other spectral analyses
^^^^^^^^^^^^^^^^^^^^^^^
- itpc :                Intertrial phase clustering (analysis of phase locking to trial events)
- burst_analysis :      Compute oscillatory burst analysis of Lundqvist et al 2016

Preprocessing
^^^^^^^^^^^^^
- cut_trials :          Cut LFPs/continuous data into trial segments
- realign_data :        Realign LFPs/continuous data to new within-trial event
- get_freq_sampling :   Frequency sampling vector for a given FFT-based computation
- remove_dc :           Remove constant DC component of signals
- remove_evoked :       Remove phase-locked evoked potentials from signals

Postprocesssing
^^^^^^^^^^^^^^^
- pool_freq_bands :         Average spectral data within set of frequency bands
- pool_time_epochs :        Average spectral data within set of time epochs
- one_over_f_norm :         Normalize to correct for 1/f distribution of spectral power
- complex_to_spec_type :    Convert complex Fourier transform output to power/phase/real/imag/etc.
- one_sided_to_two_sided :  Convert 1-sided Fourier transform output to 2-sided equivalent

Plotting
^^^^^^^^
- plot_spectrum :       Plot frequency spectrum as a line plot, handling freq axis properly
- plot_spectrogram :    Plot time-frequency spectrogram as a heatmap plot

Data simulation
^^^^^^^^^^^^^^^
- simulate_oscillation :    Generates simulated oscillation-in-noise data


Dependencies
------------
- pyfftw :              Python wrapper around FFTW, the speedy FFT library

Function reference
------------------
"""
# Created on Thu Oct  4 15:28:15 2018
#
# @author: sbrincat

from warnings import warn
from math import floor,ceil,log2,pi,sqrt
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from pyfftw.interfaces.scipy_fftpack import fft,ifft # ~ 46/16 s on benchmark

# from numpy.fft import fft,ifft        # ~ 15 s on benchmark
# from scipy.fftpack import fft,ifft    # ~ 11 s on benchmark
# from mkl_fft import fft,ifft    # ~ 15.2 s on benchmark
# from pyfftw import empty_aligned, byte_align
# from pyfftw.interfaces.cache import enable as enable_pyfftw_cache
# import pyfft
# enable_pyfftw_cache()

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spynal.utils import set_random_seed, iarange, index_axis, axis_index_slices, \
                         standardize_array, undo_standardize_array, interp1
from spynal.helpers import _check_window_lengths
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.plots import plot_line_with_error_fill, plot_heatmap
from spynal.spectra.wavelet import wavelet_spectrum, wavelet_spectrogram, compute_wavelets, \
                                   wavelet_bandwidth, wavelet_edge_extent
from spynal.spectra.multitaper import multitaper_spectrum, multitaper_spectrogram
from spynal.spectra.bandfilter import bandfilter_spectrum, bandfilter_spectrogram

# Set default arguments for pyfftw functions: Fast planning, use all available threads
_FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE',
                        'threads': cpu_count()}


# =============================================================================
# General spectral analysis functions
# =============================================================================
def spectrum(data, smp_rate, axis=0, method='multitaper', data_type='lfp', spec_type='complex',
             removeDC=True, **kwargs):
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

        - 'multitaper' : Multitaper spectral analysis in :func:`multitaper_spectrum`
        - 'wavelet' : Wavelet analysis in :func:`wavelet_spectrum`
        - 'bandfilter' : Bandpass filtering in :func:`bandfilter_spectrum`

    data_type : {'lfp','spike'}, default: 'lfp'
        Type of signal in data

    spec_type : {'complex','power','phase','real','imag'}, default: 'complex'
        Type of spectral signal to return. See :func:`complex_to_spec_type` for details.

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
                          removeDC=removeDC, **kwargs)

    return spec, freqs


def spectrogram(data, smp_rate, axis=0, method='wavelet', data_type='lfp', spec_type='complex',
                removeDC=True, **kwargs):
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

        - 'multitaper' : Multitaper spectral analysis in :func:`multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`bandfilter_spectrogram`
        - 'burst' : Oscillatory burst analysis in :func:`burst_analysis`

    data_type : {'lfp','spike'}, default: 'lfp'
        Type of signal in data

    spec_type : {'complex','power','phase','real','imag'}, default: 'complex'
        Type of spectral signal to return. See :func:`complex_to_spec_type` for details.

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
                                      spec_type=spec_type, removeDC=removeDC, **kwargs)

    return spec, freqs, timepts


def power_spectrum(data, smp_rate, axis=0, method='multitaper', **kwargs):
    """
    Convenience wrapper around spectrum() to compute **power** spectrum of data with given method

    See :func:`spectrum` for details
    """
    return spectrum(data, smp_rate, axis=axis, method=method, spec_type='power', **kwargs)


def power_spectrogram(data, smp_rate, axis=0, method='wavelet', **kwargs):
    """
    Convenience wrapper around spectrogram() to compute time-frequency **power** with given method

    See :func:`spectrogram` for details
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, spec_type='power', **kwargs)


def phase_spectrogram(data, smp_rate, axis=0, method='wavelet', **kwargs):
    """
    Convenience wrapper around spectrogram() to compute **phase** of time-frequency transform

    See :func:`spectrogram` for details
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, spec_type='phase', **kwargs)


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

        - 'multitaper' : Multitaper spectral analysis in :func:`multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`bandfilter_spectrogram`

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
                   window=None, timepts=None, **kwargs):
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

        - 'multitaper' : Multitaper spectral analysis in :func:`multitaper_spectrogram`
        - 'wavelet' : Wavelet analysis in :func:`wavelet_spectrogram`
        - 'bandfilter' : Bandpass filtering in :func:`bandfilter_spectrogram`

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
    data,freqs,times = spectrogram(data, smp_rate, axis=0, method=method, spec_type=spec_type,
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


def _frequency_plot_settings(freqs):
    """ Return settings for plotting a frequency axis: plot freqs, ticks, tick labels """
    freqs = np.asarray(freqs).squeeze()
    # For freqs given as (low,high) bands, convert to band means
    if (freqs.ndim == 2) and (freqs.shape[1] == 2): freqs = freqs.mean(axis=1)

    freq_scale = _infer_freq_scale(freqs)

    # For log-sampled freqs, plot in log2(freq) but label with actual freqs
    if freq_scale == 'log':
        freqs           = np.log2(freqs)            # Log2-transform plotting freqs
        fmin            = ceil(freqs[0])
        fmax            = floor(freqs[-1])
        freq_ticks      = np.arange(fmin,fmax+1)    # Plot ticks every octave: [2,4,8,16,...]
        freq_tick_labels= 2**np.arange(fmin,fmax+1)

    # For linear-sampled freqs, just plot in actual freqs
    elif freq_scale == 'linear':
        fmin            = ceil(freqs[0]/10.0)*10.0  # Plot ticks every 10 Hz
        fmax            = floor(freqs[-1]/10.0)*10.0
        freq_ticks      = np.arange(fmin,fmax+1,10).astype(int)
        freq_tick_labels= freq_ticks

    # For arbitrary unevenly-sampled freqs (eg bandfilter or burst analyis),
    # plot freqs categorically as range 0 - n_freqs-1, but label with actual freqs
    else:
        freq_tick_labels= freqs
        freq_ticks      = np.arange(len(freqs))
        freqs           = np.arange(len(freqs))

    return freqs, freq_ticks, freq_tick_labels


# =============================================================================
# Preprocessing functions
# =============================================================================
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

    smp_rate : scalar
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


def get_freq_sampling(smp_rate,n_fft,freq_range=None,two_sided=False):
    """
    Return frequency sampling vector (axis) for a given FFT-based computation

    Parameters
    ----------
    smp_rate : scalar
        Data sampling rate (Hz)

    n_fft : scalar
        Number of samples (timepoints) in FFT output

    freq_range : array-like, shape=(2,) or scalar, default: all frequencies from FFT
        Range of frequencies to retain in output, either given as an explicit [low,high]
        range or just a scalar giving the highest frequency to return.


    two_sided : bool, default: False
        If True, return freqs for two-sided spectrum, including both positive and
        negative frequencies (which have same amplitude for all real signals).
        If False, only return positive frequencies, in range (0,smp_rate/2).

    Returns
    -------
    freqs : ndarray, shape=(n_freqs,)
        Frequency sampling vector (in Hz)

    freq_bool : ndarray, shape=(n_fft,), dtype=bool
        Boolean vector flagging frequencies in full FFT output to retain, given desired freq_range
    """
    freqs   = np.fft.fftfreq(n_fft,d=1/smp_rate) # All possible frequencies

    # If no range requested, keep all frequencies
    if freq_range is None:
        # Include both positive and negative frequencies
        if two_sided:
            freq_bool = np.ones((n_fft,),dtype=bool)
        # Limit to positive frequencies
        else:
            if n_fft%2 == 0: n = (n_fft/2 + 1, n_fft/2 - 1)
            else:           n = ((n_fft-1)/2, (n_fft-1)/2 + 1)
            freq_bool = np.concatenate((np.ones((int(n[0]),),dtype=bool),
                                        np.zeros((int(n[1]),),dtype=bool)))

    # Limit frequencies to requested range
    else:
        # Only keep frequencies < max freq, or w/in given range
        if len(freq_range) == 1:
            freq_bool = np.abs(freqs) <= freq_range
        elif len(freq_range) == 2:
            freq_bool = (np.abs(freqs) >= freq_range[0]) & \
                        (np.abs(freqs) <= freq_range[1])
        else:
            raise ValueError("freq_range must be given as 2-length vector = [min,max]" \
                             "or scalar max frequency")

        # Limit to positive frequencies. Special case to also get f = (-)smp_rate/2
        if not two_sided:
            freq_bool = freq_bool & ((freqs >= 0) | np.isclose(freqs,-smp_rate/2))

    # Extract only desired freqs from sampling vector
    freqs = freqs[freq_bool]

    # Again, special case to deal with (-)smp_rate/2
    if not two_sided: freqs = np.abs(freqs)

    return freqs,freq_bool


def remove_dc(data, axis=None):
    """
    Remove constant DC component of signals, estimated as across-time mean
    for each time series (ie trial,channel,etc.)

    Parameters
    ----------
    data : ndarray, shape=(...,n_samples,...)
        Raw data to remove DC component of.
        Can be any arbitary shape, with time sampling along `axis`

    axis : int, Default: None (remove DC component computed across *full* data array)
        Data axis corresponding to time

    Returns
    -------
    data : ndarray, shape=(...,n_samples,...)
        Data with DC component removed (same shape as input)
    """
    return data - data.mean(axis=axis,keepdims=True)


def remove_evoked(data, axis=0, method='mean', design=None):
    """
    Remove estimate of evoked potentials phase-locked to trial events,
    returning data with (in theory) only non-phase-locked induced components

    Parameters
    ----------
    data : ndarray, shape=(...,n_samples,...)
        Raw data to remove evoked components from.
        Can be any arbitary shape, with observations (trials) along `axis`.

    axis : int, default: 0 (1st axis)
        Data axis corresponding to distinct observations/trials

    method : {'mean','groupMean','regress'}, default: 'mean'
        Method to use for estimating evoked potentials:

        - 'mean'      : Grand mean signal across all observations (trials)
        - 'groupMean' : Mean signal across observations with each group in `design`
        - 'regress'   : OLS regresion fit of design matrix `design` to data

    design : array-like, shape=(n_samples,...), optional
        Design matrix to fit to data (`method` == 'regress')
        or group/condition labels for each observation (`method` == 'groupMean').
        Not used for `method` == 'mean'.

    Returns
    -------
    data : ndarray, shape=(...,n_samples,...)
        Data with estimated evoked component removed. Same shape as input `data`.
    """
    design = np.asarray(design)

    # Subtract off grand mean potential across all trials
    if method.lower() == 'mean':
        return data - np.mean(data,axis=axis,keepdims=True)

    # Subtract off mean potential across all trials within each group/condition
    # todo  can we do this with an xarray or pandas groupby() instead??
    elif method.lower() == 'groupmean':
        assert (design.ndim == 1) or ((design.ndim == 2) and (design.shape[1] == 1)), \
            "Design matrix <design> must be vector-like (1d or 2d w/ shape[1]=1)"

        data, data_shape = standardize_array(data, axis=axis, target_axis=0)

        groups = np.unique(design)
        for group in groups:
            idxs = design == group
            data[idxs,...] -= np.mean(data[idxs,...],axis=0,keepdims=True)

        data = undo_standardize_array(data, data_shape, axis=axis, target_axis=0)

    # Regress data on given design matrix and return residuals
    elif method.lower() == 'regress':
        assert design.ndim in [1,2], \
            "Design matrix <design> must be matrix-like (2d) or vector-like (1d)"

        data, data_shape = standardize_array(data, axis=axis, target_axis=0)

        model = LinearRegression()
        data -= model.fit(design,data).predict(design)

        data = undo_standardize_array(data, data_shape, axis=axis, target_axis=0)

    return data


# =============================================================================
# Post-processing helper functions
# =============================================================================
def complex_to_spec_type(data, spec_type):
    """
    Converts complex spectral data to given spectral signal type

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=complex
        Complex spectral (or time-frequency) data. Arbitrary shape.

    spec_type : {'power','phase','magnitude','real','imag'}
        Type of spectral signal to return:

        - 'power'     Spectral power of data
        - 'phase'     Phase of complex spectral data (in radians)
        - 'magnitude' Magnitude (square root of power) of complex data = signal envelope
        - 'real'      Real part of complex data
        - 'imag'      Imaginary part of complex data

    Returns
    -------
    data : ndarray, shape=Any, dtype=complex
        Computed spectral signal. Same shape as input.
    """
    if spec_type == 'complex':      return data
    elif spec_type == 'power':      return power(data)
    elif spec_type == 'phase':      return phase(data)
    elif spec_type == 'magnitude':  return magnitude(data)
    elif spec_type == 'real':       return data.real
    elif spec_type == 'imag':       return np.imag(data)
    else:
        raise ValueError("%s is an unsupported option for spec_type" % spec_type)


def power(data):
    """ Compute power from complex spectral data  """
    return (data*data.conj()).real  # Note: .real fixes small float errors

def magnitude(data):
    """ Compute magnitude (square root of power) from complex spectral data  """
    return np.abs(data)

def phase(data):
    """ Compute phase of complex spectral data  """
    return np.angle(data)

def real(data):
    """ Return real part of complex spectral data  """
    return data.real

def imag(data):
    """ Return imaginary part of complex spectral data  """
    return np.imag(data)


def pool_freq_bands(data, bands, axis=None, freqs=None, func='mean'):
    """
    Pool (eg average) spectral data within each of a given set of frequency bands

    Parameters
    ----------
    data : ndarray or xarray DataArray, shape=(...,n_freqs,...)
        Raw data to pool within frequency bands. Any arbitary shape.

    bands : array-like, shape=(n_bands,2) or dict {str : array-like, shape=(2,)}
        Frequency bands to pool data within. Input either as a list of [low-cut, high-cut]
        values or as a dict, with keys being frequency band names and their associated
        values being the corresponding [low-cut, high-cut] pair.
        Band edges are inclusive.

    axis : int
        Data axis corresponding to frequency.
        Only needed if `data` is not an xarray DataArray with dimension named 'freq'/'frequency'.

    freqs : array-like, shape=(n_freqs,)
        Frequency sampling in `data`. Only needed if `data` is not an xarray DataArray.

    func : str or callable, default: 'mean' (mean across all frequencies in band)
        Function to use to pool values within each frequency band, given either as a
        string specifier (options: 'mean' or 'sum') or a custom function that takes as input
        an ndarray and returns an ndarray with its first axis reduced to length 1.

    Returns
    -------
    data : ndarray or xarray DataArray, shape=(...,n_freqbands,...)
        Data with values pooled within each of given frequency bands
    """
    # Convert list of frequency band ranges to {'name':freq_range} dict
    if not isinstance(bands,dict):
        bands = {'band_'+str(i_band):frange for i_band,frange in enumerate(bands)}

    # Convert frequency bands into 1-d list of bin edges
    bins = []
    for value in bands.values(): bins.extend(value)

    func_ = _str_to_pool_func(func)

    # Figure out data dimensionality and standardize so frequency axis = 0 (1st axis)
    if HAS_XARRAY and isinstance(data,xr.DataArray):
        dims = np.asarray(data.dims)
        # Find frequency dimension if not given explicitly
        if axis is None:  axis = ((dims == 'freq') | (dims == 'frequency')).nonzero()[0][0]
        freq_dim = dims[axis]   # Name of frequency dim

        if freqs is None: freqs = data.coords[freq_dim].values

        # Permute array dims so freq is 1st dim
        if axis != 0:
            temp_dims = np.concatenate(([dims[axis]], dims[dims != freq_dim]))
            data = data.transpose(*temp_dims)
        else:
            temp_dims = dims

        # Initialize new DataArray with freq dim = freq bands, indexed by band names
        coords = {dim : data.coords[dim].values for dim in data.coords}
        coords[freq_dim] = list(bands.keys())
        data_shape = (len(bands), *data.shape[1:])
        band_data = xr.DataArray(np.zeros(data_shape,dtype=data.dtype),
                                 dims=temp_dims, coords=coords)

    else:
        assert axis is not None, \
        ValueError("For ndarray data, must give frequency axis in <axis>")
        assert freqs is not None, \
        ValueError("For ndarray data, must give frequency sampling vector in <freqs>")

        if axis != 0: data = data.swapaxes(0,axis)

        data_shape= (len(bands), *data.shape[1:])
        band_data = np.zeros(data_shape,dtype=data.dtype)

    # Pool data over each frequency band
    for i_band,(_,frange) in enumerate(bands.items()):
        fbool = (freqs >= frange[0]) & (freqs <= frange[1])
        band_data[i_band,...] = func_(data[fbool,...])

    # Permute back to original data dimension order
    if axis != 0:
        if HAS_XARRAY and isinstance(data,xr.DataArray):
            band_data = band_data.transpose(*dims)
        else:
            band_data = band_data.swapaxes(axis,0)

    return band_data


def pool_time_epochs(data, epochs, axis=None, timepts=None, func='mean'):
    """
    Pool (eg average) spectral data within each of a given set of time epochs

    Parameters
    ----------
    data : ndarray or xarray DataArray, shape=(...,n_timepts,...)
        Raw data to pool within time epochs. Any arbitary shape.

    epochs : array-like, shape=(n_epochs,2) or dict {str : array-like, shape=(2,)}
        Time epochs to pool data within. Input either as a list of [start,end] times
        or as a dict, with keys being time epoch names and their associated
        values being the corresponding [start,end] pair.
        Epoch edges are inclusive.

    axis : int
        Data axis corresponding to time.
        Only needed if `data` is not an xarray DataArray with dimension named 'time'.

    timepts : array-like, shape(n_timepts,)
        Time sampling in `data`.
        Only needed if `data` is not an xarray DataArray with dimension named 'time'.

    func : str or callable, default: 'mean' (mean across all frequencies in band)
        Function to use to pool values within each time epoch, given either as a
        string specifier (options: 'mean' or 'sum') or a custom function that takes as input
        an ndarray and returns an ndarray with its first axis reduced to length 1.

    Returns
    -------
    data : ndarray or xarray DataArray, shape=(...,n_time_epochs,...)
        Data with values pooled within each of given time epochs
    """
    # Convert list of time epoch ranges to {'name':time_range} dict
    if not isinstance(epochs,dict):
        epochs = {'epochs_'+str(i_epoch):trange for i_epoch,trange in enumerate(epochs)}

    func_ = _str_to_pool_func(func)

    # Figure out data dimensionality and standardize so time axis = 0 (1st axis)
    if HAS_XARRAY and isinstance(data,xr.DataArray):
        dims = np.asarray(data.dims)
        if timepts is None: timepts = data.coords['time'].values
        # Find 'time' dimension if not given explicitly
        if axis is None:  axis = (dims == 'time').nonzero()[0][0]
        # Permute array dims so time is 1st dim
        if axis != 0:
            temp_dims = np.concatenate(([dims[axis]], dims[dims != 'time']))
            data = data.transpose(*temp_dims)
        else:
            temp_dims = dims

        # Initialize new DataArray with time dim = time epochs, indexed by epoch names
        coords = {dim : data.coords[dim].values for dim in data.coords}
        coords['time'] = list(epochs.keys())
        data_shape= (len(epochs), *data.shape[1:])
        epoch_data = xr.DataArray(np.zeros(data_shape,dtype=data.dtype),
                                  dims=temp_dims, coords=coords)

    else:
        assert axis is not None, \
        ValueError("For ndarray data, must give time axis in <axis>")
        assert timepts is not None, \
        ValueError("For ndarray data, must give time sampling vector in <timepts>")

        if axis != 0: data = data.swapaxes(0,axis)

        data_shape= (len(epochs), *data.shape[1:])
        epoch_data = np.zeros(data_shape,dtype=data.dtype)

    # Pool data over each time epoch
    for i_epoch,(_,trange) in enumerate(epochs.items()):
        tbool = (timepts >= trange[0]) & (timepts <= trange[1])
        epoch_data[i_epoch,...] = func_(data[tbool,...])

    # Permute back to original data dimension order
    if axis != 0:
        if HAS_XARRAY and isinstance(data,xr.DataArray):
            epoch_data = epoch_data.transpose(*dims)
        else:
            epoch_data = epoch_data.swapaxes(axis,0)

    return epoch_data


def one_over_f_norm(data, axis=None, freqs=None, exponent=1.0):
    """
    Normalize to correct for ~ 1/frequency**alpha baseline distribution of power
    by multiplying by frequency, raised to a given exponent

    Parameters
    ----------
    data : ndarray or xarray DataArray, shape=(...,n_freqs,...)
        Raw data to pool within frequency bands. Any arbitary shape.

    axis : int
        Data axis corresponding to frequency.
        Only needed if `data` is not an xarray DataArray with dimension named 'freq'/'frequency'.

    freqs : array-like, shape=(n_freqs,)
        Frequency sampling in `data`. Only needed if `data` is not an xarray DataArray.

    exponent : float, default: 1 (correct for 1/f, w/o exponent)
        Exponent ('alpha') to raise freqs to for normalization.

    Returns
    -------
    data : ndarray or xarray DataArray, shape=(...,n_freqs,...)
        1/f normalized data. Same shape as input.
    """
    if HAS_XARRAY and isinstance(data,xr.DataArray):
        dims = np.asarray(data.dims)
        # Find frequency dimension if not given explicitly
        if axis is None:  axis = ((dims == 'freq') | (dims == 'frequency')).nonzero()[0][0]
        freq_dim = dims[axis]   # Name of frequency dim
        if freqs is None: freqs = data.coords[freq_dim].values

    assert axis is not None, \
        ValueError("Frequency axis must be given in <axis> (or input xarray data)")
    assert freqs is not None, \
        ValueError("Frequency sampling vector must be given in <freqs> (or input xarray data)")

    # Ensure that freqs will broadcast against data
    freqs = np.asarray(freqs)
    if data.ndim != freqs.ndim:
        slicer          = [np.newaxis]*data.ndim    # Create (data.ndim,) list of np.newaxis
        slicer[axis]    = slice(None)               # Set <axis> element to slice as if set=':'
        freqs           = freqs[tuple(slicer)]      # Expand freqs to dimensionality of data

    return data * freqs**exponent


def one_sided_to_two_sided(data, freqs, smp_rate, axis=0):
    """
    Convert a one-sided Fourier/wavelet transform output to the two-sided equivalent.

    Assumes conjugate symmetry across positive and negative frequencies (as is the case
    only when the original raw signals were real).

    Also extrapolates values for f=0, as is necessary for wavelet transforms.

    Parameters
    ----------
    data : ndarray, shape=(...,n_freqs,...), dtype=complex
        Complex (1-sided) frequency-transformed data. Any arbitary shape.

    freqs : array-like, shape=(n_freqs,)
        Frequency sampling in `data`

    smp_rate : scalar
        Data sampling rate (Hz)

    axis : int, default: 0 (1st axis)
        Data axis corresponding to frequency

    Returns
    -------
    data : ndarray, shape=(...,2*n_freqs+1,...), dtype=complex
        2-sided equivalent of input `data`

    freqs : ndarray, shape=(2*n_freqs+1,)
        List of (positive and negative) freqs in 2-sided output `data`
    """
    assert np.isclose(freqs[-1],smp_rate/2), \
        "Need to have sampling up to 1/2 sampling rate (Nyquist freq=%d Hz)" % (smp_rate/2)

    # If f=0 is not in data, numerically extrapolate values for it
    if not np.isclose(freqs,0).any():
        f0 = interp1(freqs,data,0,axis=axis,kind='cubic',fill_value='extrapolate')
        f0 = np.expand_dims(f0,axis)
        data = np.concatenate((f0,data),axis=axis)
        freqs = np.concatenate(([0],freqs))

    # Convert values at Nyquist freq to complex conjugate at negative frequency
    slices = axis_index_slices(axis,-1,data.ndim)
    data[slices] = data[slices].conj()
    freqs[-1] *= -1

    # Replicate values for all freqs (s.t. 0 < f < nyquist)
    # as complex conjugates at negative frequencies
    idxs    = slice(-2,1,-1)
    slices  = axis_index_slices(axis,idxs,data.ndim)
    data    = np.concatenate((data, data[slices].conj()), axis=axis)
    freqs   = np.concatenate((freqs, -freqs[idxs]))

    return data, freqs


# =============================================================================
# Data simulation and testing functions
# =============================================================================
def simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0, n_trials=1000,
                         freq_sd=0, amp_sd=0, phase_sd=0,
                         smp_rate=1000, time_range=1.0, burst_rate=0, burst_width=4, seed=None):
    """
    Generate synthetic data with oscillation at given parameters.

    Generate multiple trials with constant oscillatory signal + random additive Gaussian noise.

    Parameters
    ----------
    frequency : scalar
        Frequency to simulate oscillation at (Hz)

    amplitude : scalar, default: 5.0
        Amplitude of simulated oscillation (a.u.)

    phase : scalar, default: 0
        Phase of oscillation (rad)

    noise : scalar, default: 1.0
        Amplitude of additive Gaussian noise (a.u)

    n_trials : int, default: 1000
        Number of trials/observations to simulate

    freq_sd,amp_sd,phase_sd : scalar, Default: 0 (no inter-trial variation)
        Inter-trial variation in frequency/amplitude/phase, given as Gaussian SD
        (same units as base parameters, which are used as Gaussian mean)

    smp_rate : int, default: 1000
        Sampling rate for simulated data (Hz)

    time_range : scalar, default: 1 s
        Full time range to simulate oscillation over (s)

    burst_rate : scalar, default: 0 (not bursty)
        Oscillatory burst rate (bursts/trial). Set=0 to simulate constant, non-bursty oscillation.

    burst_width : scalar, default: 4
        Half-width of oscillatory bursts (Gaussian SD, in cycles)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Returns
    -------
    data : ndarray, shape=(n_timepts,n_trials)
        Simulated oscillation-in-noise data
    """
    if seed is not None: set_random_seed(seed)

    def _randn(*args):
        """
        Generate unit normal random variables in a way that reproducibly matches output
        of Matlab with same seed != 0. (np.random.randn() does not work here for unknown reasons)
        stackoverflow.com/questions/3722138/is-it-possible-to-reproduce-randn-of-matlab-with-numpy?noredirect=1&lq=1
        """
        return norm.ppf(np.random.rand(*args))

    # Set per-trial frequency, amplitude, phase from base parameter + any spread
    freq    = frequency if freq_sd == 0 else frequency + freq_sd*_randn(1,n_trials)
    amp     = amplitude if amp_sd == 0 else amplitude + amp_sd*_randn(1,n_trials)
    phi     = phase if phase_sd == 0 else phase + phase_sd*_randn(1,n_trials)

    # Simulate oscillatory bursts if burst_rate is set != 0
    bursty = burst_rate > 0
    # Convert burst width from cycles to s
    burst_sd = burst_width/freq

    # Set time sampling vector (in s)
    n_timepts = round(time_range * smp_rate)
    t = np.arange(n_timepts) / smp_rate

    # Generate oscillatory signal = sinusoid wave at given amplitude(s),frequency(s),phase(s)
    if np.isscalar(amp) and np.isscalar(freq) and np.isscalar(phi):
        data = np.tile((amp * np.cos(2*pi*freq*t + phi))[:,np.newaxis], (1,n_trials))
    else:
        data = amp * np.cos(2*pi*freq*t[:,np.newaxis] + phi)

    # Make oscillations bursty, if requested
    if bursty:
        # Function to generate unit-height Gaussian function with given mean,SD
        def _gaussian(mu,sd,t):
            z = (t - mu)/sd
            return np.exp(-0.5*(z**2))

        # Use burst rate to determine which trials will have bursts ~ Bernoulli(p=rate)
        burst_trials = np.random.binomial(1,burst_rate, size=(n_trials,)).astype(bool)

        # Generate random burst times within full time range of data
        burst_times = np.empty((n_trials,))
        burst_times[burst_trials] = t[0] + (t[-1]-t[0]) * np.random.rand(burst_trials.sum())

        # Weight current trial by random Gaussian envelope if it has a burst
        # otherwise, remove signal from non-burst trials
        for trial in range(n_trials):
            if burst_trials[trial]:
                data[:,trial] *= _gaussian(burst_times[trial],burst_sd,t)
            else:
                data[:,trial] *= 0

    # Generate additive Gaussian noise of given amplitude
    if noise != 0:  data += noise * _randn(n_timepts,n_trials)

    return data


# =============================================================================
# Helper functions
# =============================================================================
def _undo_standardize_array_newaxis(data,data_shape,axis=0):
    """
    Reshape data array from unwrapped form back to ~ original
    multi-dimensional form in special case where a new frequency axis was
    inserted before time axis (<axis>)

    Parameters
    ----------
    data : ndarray, shape=(axis_len,m)
        Data array w/ all axes > 0 unwrapped into single dimension, where m = prod(shape[1:])

    data_shape : tuple, len=Any
        Original shape of data array

    axis : int, default: 0
        Axis of original data corresponding to distinct observations,
        which has become axis 1, but will be permuted back to original axis.

    Returns
    -------
    data: ndarray, shape=(...,n_freqs,n_timepts,...)
        Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)
    if axis < 0: axis = len(data_shape) + axis

    data_ndim   = len(data_shape) # Number of dimensions in original data
    n_freqs      = data.shape[0]
    n_timepts    = data.shape[1]

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    if len(data_shape) > 2:
        shape   = (n_freqs, n_timepts, *data_shape[np.arange(data_ndim) != axis])
        data    = np.reshape(data,shape,order='F')

    # Squeeze (n,1) array back down to 1d (n,) vector,
    #  and extract value from scalar array -> float
    elif data_ndim == 1:
        data = data.squeeze(axis=-1)
        if data.size == 1: data = data.item()

    # If <axis> wasn't 0, move axis back to original position
    if (axis != 0) and isinstance(data,np.ndarray):
        data = np.moveaxis(data,(0,1),(axis,axis+1))

    return data


def _next_power_of_2(n):
    """ Find next power of 2 (smallest power of 2 greater than n) """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


def _infer_freq_scale(freqs):
    """ Determine if frequency sampling vector is linear, logarithmic, or uneven """
    # Determine if frequency scale is linear (all equally spaced)
    if np.allclose(np.diff(np.diff(freqs)),0):
        return 'linear'

    # Determine if frequency scale is logarithmic (all equally spaced in log domain)
    elif np.allclose(np.diff(np.diff(np.log2(freqs))),0):
        return 'log'

    # Otherwise assume arbitrary unevenly-sampled frequency axis (as in bandfilter/burst analysis)
    else:
        warn("Unable to determine scale of frequency sampling vector. Assuming it's arbitrary")
        return 'uneven'


def _extract_triggered_data(data, smp_rate, event_times, window):
    """
    Extracts windowed chunks of data around given set of event times

    Parameters
    ----------
    data : ndarray, shape=(n_samples,...)
        Data to cut triggered snippets out of
        NOTE: Not coded up for arbitrary-shaped data

    smp_rate : int, default: 1000
        Sampling rate for `data` (Hz)

    event_times : array-like, shape=(n_events,)
        List of times (s) of event triggers to extract data around.
        Times are referenced to 1st data sample (t=0).

    window : array-like, shape=(2,)
        [start,end] of window (in s) to extract around each trigger

    Returns
    -------
    data : ndarray, shape=(n_samples_per_window,n_events,...)
        Data cut at event triggers
    """
    # Convert event_times, window from s -> samples
    event_times = np.floor(np.asarray(event_times)*smp_rate).astype(int)
    window      = np.round(np.asarray(window)*smp_rate).astype(int)

    n_per_event = window[1] - window[0]
    n_events    = len(event_times)
    data_shape  = data.shape
    data_out    = np.zeros((n_per_event,n_events,*data_shape[1:]))

    for i_event,event in enumerate(event_times):
        idxs    = np.arange(event-window[0],event+window[1])
        data_out[:,i_event,...] = data[idxs,...]

    return data_out


def _str_to_pool_func(func):
    """ Convert string specifier to callable pooling function """
    # If it's already a callable, return as-is
    if callable(func):      return func
    else:
        assert isinstance(func,str), "'func' must be a string or callable function"

        if func == 'mean':  return lambda x: np.mean(x, axis=0)
        elif func == 'sum': return lambda x: np.sum(x, axis=0)
        else:
            raise ValueError("Unsupported value '%s' for func. Set='mean'|'sum'" % func)

# -*- coding: utf-8 -*-
"""
spectra     A module for spectral analysis of neural oscillations

Functionality for computing frequency spectra as well as time-frequency (spectrogram) transforms.

Options to compute spectral analysis using multitaper, wavelet, band-pass filtering or spectral
burst analysis methods.

Options to return full complex spectral data, spectral power, or spectral phase.

Also includes functions for preprocessing, postprocessing, and plotting of analog/spectral data.

Most functions perform operations in a mass-univariate manner. This means that
rather than embedding function calls in for loops over channels, trials, etc., like this:

for channel in channels:
    for trial in trials:
        results[trial,channel] = compute_something(data[trial,channel])

You can instead execute a single call on ALL the data, labeling the relevant axis
for the computation (usually time here), and it will run in parallel (vectorized)
across all channels, trials, etc. in the data, like this:

results = compute_something(data, axis)


FUNCTIONS
### General spectral analysis ###
spectrum            Frequency spectrum of data
spectrogram         Time-frequency spectrogram of data
power_spectrum      Power spectrum of data
power_spectrogram   Power of time-frequency transform
phase_spectrogram   Phase of time-frequency transform

### Multitaper spectral analysis ###
multitaper_spectrum Multitaper (DPSS) frequency spectrum
multitaper_spectrogram  Multitaper (DPSS) time-frequency spectrogram
compute_tapers      Computes DPSS tapers for use in multitaper spectral analysis

### Wavelet spectral analysis ###
wavelet_spectrum    Wavelet-based frequency spectrum
wavelet_spectrogram Time-frequency continuous wavelet transform
compute_wavelets    Computes wavelets for use in wavelet spectral analysis
wavelet_bandwidth   Computes time,frequency bandwidths for set of wavelets
wavelet_edge_extent Computes extent of edge effects for set of wavelets

### Bandpass-filtering spectral analysis ###
bandfilter_spectrum Band-filtered frequency spectrum
bandfilter_spectrogram Band-filtered, Hilbert-transformed time-frequency of data
set_filter_params   Sets filter coefficients for use in band-filtered analysis

### Other spectral analyses ###
itpc                Intertrial phase clustering (analysis of phase locking to trial events)
burst_analysis      Computes oscillatory burst analysis of Lundqvist et al 2016

### Preprocessing ###
cut_trials          Cut LFPs/continuous data into trial segments
realign_data        Realigns LFPs/continuous data to new within-trial event

get_freq_sampling   Frequency sampling vector for a given FFT-based computation

remove_dc           Removes constant DC component of signals
remove_evoked       Removes phase-locked evoked potentials from signals

### Postprocesssing ###
pool_freq_bands     Averages spectral data within set of frequency bands
pool_time_epochs    Averages spectral data within set of time epochs
one_over_f_norm     Normalizes to correct for 1/f distribution of spectral power
complex_to_spec_type    Converts complex Fourier transform output to power/phase/real/imag/etc.
one_sided_to_two_sided  Converts 1-sided Fourier transform output to 2-sided equivalent

### Plotting ###
plot_spectrum       Plots frequency spectrum as a line plot, handling freq axis properly
plot_spectrogram    Plots time-frequency spectrogram as a heatmap plot

### Data simulation ###
simulate_oscillation Generates simulated oscillation-in-noise data


DEPENDENCIES
pyfftw              Python wrapper around FFTW, the speedy FFT library


Created on Thu Oct  4 15:28:15 2018

@author: sbrincat
"""

from warnings import warn
from math import floor,ceil,log2,pi,sqrt
from collections import OrderedDict
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal.windows import dpss
from scipy.signal import filtfilt,hilbert,zpk2tf,butter,ellip,cheby1,cheby2
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

from neural_analysis.utils import set_random_seed, iarange, index_axis, axis_index_slices, \
                                  standardize_array, undo_standardize_array, interp1
from neural_analysis.helpers import _check_window_lengths
from neural_analysis.spikes import _spike_data_type, times_to_bool
from neural_analysis.plots import plot_line_with_error_fill, plot_heatmap


# Set default arguments for pyfftw functions: Fast planning, use all available threads
_FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE',
                        'threads': cpu_count()}


# =============================================================================
# General spectral analysis functions
# =============================================================================
def spectrum(data, smp_rate, axis=0, method='multitaper', data_type='lfp', spec_type='complex',
             removeDC=True, **kwargs):
    """
    Computes frequency spectrum of data using given method

    spec,freqs = spectrum(data,smp_rate,axis=0,method='multitaper',
                          data_type='lfp',spec_type='complex',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'multitaper' [default] (only value currently supported)

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    **kwargs    All other kwargs passed directly to method-specific
                spectrum function. See there for details.

    RETURNS
    spec        (...,n_freqs,...) ndarray of complex | float.
                Frequency spectrum of given type computed with given method.
                Frequency axis is always inserted in place of time axis
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after time axis if keep_tapers=True
                dtype is complex for spec_type='complex', 'float' otherwise

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <spec>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <spec> (Hz).
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
    Computes time-frequency transform of data using given method

    spec,freqs,timepts = spectrogram(data,smp_rate,axis=0,method='wavelet',
                                     data_type='lfp', spec_type='complex',removeDC=True,**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform
                'burst' :       Oscillatory burst analysis

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    **kwargs    All other kwargs passed directly to method-specific
                spectrogram function. See there for details.

    RETURNS
    spec        (...,n_freqs,n_timepts,...) ndarray of complex | float.
                Time-frequency spectrogram of given type computed with given method.
                Frequency axis is always inserted just before time axis.
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after freq axis if keep_tapers=True.
                dtype is complex for spec_type='complex', 'float' otherwise.

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <spec>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <spec> (Hz).

    timepts     (n_timepts,) ndarray. List of time points / time window centers in <spec>
                (in s, referenced to start of data).
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
    Convenience wrapper around spectrum to compute power spectrum of data with given method

    spec,freqs = power_spectrum(data,smp_rate,axis=0,method='multitaper',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'multitaper' [default] (only value currently supported)

    **kwargs    All other kwargs passed directly to method-specific
                spectrum function. See there for details.

    RETURNS
    spec        (...,n_freqs,...) ndarray of complex floats.
                Compplex frequency spectrum computed with given method.
                Frequency axis is always inserted in place of time axis
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after time axis.

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <spec>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <spec> (Hz).
    """
    return spectrum(data, smp_rate, axis=axis, method=method, spec_type='power', **kwargs)


def power_spectrogram(data, smp_rate, axis=0, method='wavelet', **kwargs):
    """
    Convenience wrapper around spectrogram() to compute time-frequency power with given method

    spec,freqs,timepts = power_spectrogram(data,smp_rate,axis=0,method='wavelet',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform

    **kwargs    All other kwargs passed directly to method-specific
                spectrogram function. See there for details.

    RETURNS
    spec        (...,n_freqs,n_timepts,...) ndarray of floats.
                Time-frequency power spectrogram computed with given method.
                Frequency axis is always inserted just before time axis.
                For 'multitaper' method, power is averaged across tapers.

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <spec>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <spec> (Hz).

    timepts     (n_timepts,) ndarray. List of time points / time window centers in <spec>
                (in s, referenced to start of data).
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, spec_type='power', **kwargs)


def phase_spectrogram(data, smp_rate, axis=0, method='wavelet', **kwargs):
    """
    Convenience wrapper around spectrogram() to compute phase of time-frequency transform
    of data with given method

    spec,freqs,timepts = phase_spectrogram(data,smp_rate,axis=0,method='wavelet',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform

    **kwargs    All other kwargs passed directly to method-specific
                spectrogram function. See there for details.

    RETURNS
    spec        (...,n_freqs,n_timepts,...) ndarray of floats.
                Time-frequency power spectrogram computed with given method.
                Frequency axis is always inserted just before time axis.
                For 'multitaper' method, phase is averaged (circular mean)
                across tapers.

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <spec>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <spec> (Hz).

    timepts     (n_timepts,) ndarray. List of time points / time window centers in <spec>
                (in s, referenced to start of data).
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, spec_type='phase', **kwargs)


# =============================================================================
# Multitaper spectral analysis functions
# =============================================================================
def multitaper_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freq_range=None, removeDC=True, freq_width=4, n_tapers=None,
                        keep_tapers=False, tapers=None, pad=True, **kwargs):
    """
    Multitaper Fourier spectrum computation for continuous (eg LFP) or point process (spike) data
    
    Multitaper methods project the data onto orthogonal Slepian (DPSS) "taper" functions, which
    increases the data's effective signal-to-noise. It allows a principled tradeoff btwn time
    resolution (data.shape[axis]), frequency resolution (freq_width), and the number of taper
    functions (n_tapers), which determines the signal-to-noise increase (see ARGS for details).
    
    Note: By default, data is zero-padded to the next power of 2 greater than its input length.
    This will change the frequency sampling (number of freqs and exact freqs sampled) from what
    would be obtained from the original raw data, but can be skipped by inputtng pad=False.

    spec,freqs = multitaper_spectrum(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',
                                     freq_range=None,removeDC=True,freq_width=4,n_tapers=None,
                                     keep_tapers=False,tapers=None,pad=True,**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of data to perform spectral analysis on (usually time dim)
                Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freq_range  (2,) array-like | Scalar. Range of frequencies to keep in output,
                either given as an explicit [low,high] range or just a scalar
                giving the highest frequency to return.
                Default: all frequencies from FFT, ranging from
                0 Hz - Nyquist frequency (smp_rate/2)

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    freq_width  Scalar. Frequency bandwidth 'W' (Hz). Default: 4 Hz
                Note: Time bandwidth 'T' is set to full length of data.

    n_tapers    Scalar. Number of tapers to compute. Must be <= 2TW-1, as this is
                the max number of spectrally delimited tapers. Default: 2TW-1

    tapers      (n_samples,n_tapers). Precomputed tapers (as computed by compute_tapers()).
                Input either (freq_width, n_tapers) -or- tapers.
                Default: computed from (time range, freq_width, n_tapers)

    keep_tapers Bool. If True, retains all tapered versions of spectral data in output.
                If False [default], returns the mean across tapers.

    pad         Bool. If True [default], zero-pads data to next power of 2 length

    RETURNS
    spec        (...,n_freqs,[n_tapers,]...) ndarray of complex | float.
                Multitaper spectrum of given type of data. Sampling (time) axis is
                replaced by frequency and taper axes (if keep_tapers=True), but
                shape is otherwise preserved.
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    REFERENCE   Mitra & Pesaran (1999) "Analysis of dynamic brain imaging data"
                Jarvis & Mitra (2001) Neural Computation

    SOURCE      Adapted from Chronux functions mtfftc.m, mtfftpb.m
    """
    if axis < 0: axis = data.ndim + axis

    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        lims    = kwargs.pop('lims',None)
        bins    = kwargs.pop('bins',None)
        data,_  = times_to_bool(data, width=1/smp_rate, lims=lims, bins=bins)
        axis    = data.ndim
    assert len(kwargs) == 0, \
        TypeError("Incorrect or misspelled variable(s) in keyword args: "+', '.join(kwargs.keys()))

    # If observation axis != 0, permute axis to make it so
    if axis != 0: data = np.moveaxis(data,axis,0)

    n_timepts = data.shape[0]
    # Set FFT length = data length if no padding; else pad to next power of two
    if not pad: n_fft = n_timepts
    else:       n_fft = _next_power_of_2(n_timepts)
    # Set frequency sampling vector
    freqs,fbool = get_freq_sampling(smp_rate,n_fft,freq_range=freq_range)

    # Compute DPSS taper functions (if not precomputed)
    if tapers is None:
        tapers = compute_tapers(smp_rate,time_width=n_timepts/smp_rate,
                                freq_width=freq_width,n_tapers=n_tapers)

    # Reshape tapers to (n_timepts,n_tapers) (if not already)
    if (tapers.ndim == 2) and (tapers.shape[1] == n_timepts): tapers = tapers.T
    assert tapers.shape[0] == n_timepts, \
        ValueError("tapers must have same length (%d) as number of timepoints in data (%d)"
                   % (tapers.shape[0],n_timepts))

    # Reshape tapers array to pad end of it w/ singleton dims
    taper_shape  = (*tapers.shape,*np.ones((data.ndim-1,),dtype=int))

    # DELETE Results are identical with just subtracting of DC from data before fft
    # # Compute values needed for normalizing point process (spiking) signals
    # if data_type == 'spike' and removeDC:
    #     # Compute Fourier transform of tapers
    #     taper_fft= fft(tapers,n=n_fft,axis=0)
    #     if data.ndim > 1:
    #         taper_fft_shape = list(taper_shape)
    #         taper_fft_shape[0] = n_fft
    #         taper_fft = np.reshape(taper_fft,taper_fft_shape)
    #     # Compute mean spike rate across all timepoints in each data series
    #     mean_rate = np.sum(data,axis=0,keepdims=True)/n_timepts

    # Reshape tapers and data to have appropriate shapes to broadcast together
    if data.ndim > 1:  tapers = np.reshape(tapers,taper_shape)

    if removeDC: data = remove_dc(data,axis=0)

    # Insert dimension for tapers in data axis 1 -> (n_timepts,1,...)
    data    = data[:,np.newaxis,...]

    # Project data onto set of taper functions
    data    = data * tapers

    # Compute Fourier transform of projected data, normalizing appropriately
    spec    = fft(data,n=n_fft,axis=0)
    if data_type != 'spike': spec = spec/smp_rate

    # DELETE Results are identical with just subtracting of DC from data before fft
    # Subtract off the DC component (average spike rate) for point process signals
    # if data_type == 'spike' and removeDC: spec -= taper_fft*mean_rate

    # Extract desired set of frequencies
    spec    = spec[fbool,...]

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)

    # Compute mean across tapers if requested
    if not keep_tapers:
        if spec_type == 'phase':    spec = phase(np.exp(1j*spec).mean(axis=1))
        else:                       spec = spec.mean(axis=1)

    # If observation axis wasn't 0, permute (freq,tapers) back to original position
    if axis != 0:
        if keep_tapers: spec = np.moveaxis(spec,[0,1],[axis,axis+1])
        else:           spec = np.moveaxis(spec,0,axis)

    return spec, freqs


def multitaper_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                           freq_range=None, removeDC=True, time_width=0.5, freq_width=4,
                           n_tapers=None, spacing=None, tapers=None, keep_tapers=False,
                           pad=True, **kwargs):
    """
    Computes multitaper time-frequency spectrogram for continuous (eg LFP)
    or point process (eg spike) data

    Multitaper methods project the data onto orthogonal Slepian (DPSS) "taper" functions, which
    increases the data's effective signal-to-noise. It allows a principled tradeoff btwn time
    resolution (data.shape[axis]), frequency resolution (freq_width), and the number of taper
    functions (n_tapers), which determines the signal-to-noise increase (see ARGS for details).
    
    Note: By default, data is zero-padded to the next power of 2 greater than its input length.
    This will change the frequency sampling (number of freqs and exact freqs sampled) from what
    would be obtained from the original raw data, but can be skipped by inputtng pad=False.

    spec,freqs,timepts = multitaper_spectrogram(data,smp_rate,axis=0,data_type='lfp',
                                                spec_type='complex',freq_range=None,removeDC=True,
                                                time_width=0.5,freq_width=4,n_tapers=None,
                                                spacing=None,tapers=None,pad=True,**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of data to perform spectral analysis on (usually time dim)
                Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freq_range  (2,) array-like | Scalar. Range of frequencies to keep in output,
                either given as an explicit [low,high] range or just a scalar
                giving the highest frequency to return.
                Default: all frequencies from FFT

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    time_width  Scalar. Time bandwidth 'T' (s). Width of sliding time window is
                set equal to this. Default: 0.5 (500 ms)

    freq_width  Scalar. Frequency bandwidth 'W' (Hz). Default: 4 Hz

    n_tapers    Scalar. Number of tapers to compute. Must be <= 2TW-1, as this is
                the max number of spectrally delimited tapers. Default: 2TW-1

    spacing     Scalar. Spacing between successive sliding time windows (s).
                Default: Set = window width (so each window exactly non-overlapping)

    tapers      (n_win_samples,n_tapers). Precomputed tapers (as computed by compute_tapers()).
                Input either (time_width, freq_width, n_tapers) -or- tapers.
                Default: computed from (time_width, freq_width, n_tapers)

    pad         Bool. If True [default], zero-pads data to next power of 2 length

    RETURNS
    spec        (...,n_freqs[,n_tapers],n_timewins,...) ndarray of complex | float.
                Multitaper time-frequency spectrogram of data.
                Sampling (time) axis is replaced by frequency, taper (if keep_tapers=True),
                and time window axes but shape is otherwise preserved.
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timewins,...) ndarray. List of timepoints in <spec> (in s, referenced
                to start of data). Timepoints here are centers of each time window.

    REFERENCE   Mitra & Pesaran (1999) "Analysis of dynamic brain imaging data"
                Jarvis & Mitra (2001) Neural Computation

    SOURCE      Adapted from Chronux function mtfftc.m
    """
    if axis < 0: axis = data.ndim + axis

    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        lims    = kwargs.pop('lims',None)
        bins    = kwargs.pop('bins',None)
        data,_  = times_to_bool(data, width=1/smp_rate, lims=lims, bins=bins)
        axis    = data.ndim

    # If observation axis != 0, permute axis to make it so
    if axis != 0: data = np.moveaxis(data,axis,0)
    n_timepts = data.shape[0]

    window = time_width
    if spacing is None: spacing = window
    # Compute DPSS taper functions (if not precomputed)
    if tapers is None:
        tapers = compute_tapers(smp_rate,time_width=time_width,freq_width=freq_width,
                                n_tapers=n_tapers)

    # Set up parameters for data time windows
    # Set window starts to range from time 0 to time n - window width
    win_starts  = iarange(0,n_timepts/smp_rate - window,spacing)
    # Set sampled timepoint vector = center of each window
    timepts     = win_starts + window/2.0

    # Extract time-windowed version of data -> (n_timepts_per_win,n_wins,n_dataseries)
    data = _extract_triggered_data(data,smp_rate,win_starts,[0,window])

    if removeDC: data = remove_dc(data,axis=0)

    # Do multitaper analysis on windowed data
    # Note: Set axis=0 and removeDC=False bc already dealt with above
    spec, freqs = multitaper_spectrum(data,smp_rate,axis=0,data_type=data_type,spec_type=spec_type,
                                      freq_range=freq_range,tapers=tapers,pad=pad,
                                      removeDC=False,keep_tapers=keep_tapers,**kwargs)

    # If time axis wasn't 0, permute (freq,tapers,timewin) axes back to original position
    if axis != 0:
        if keep_tapers: spec = np.moveaxis(spec,[0,1,2],[axis,axis+1,axis+2])
        else:           spec = np.moveaxis(spec,[0,1],[axis,axis+1])

    return spec, freqs, timepts


def compute_tapers(smp_rate, time_width=0.5, freq_width=4, n_tapers=None):
    """
    Computes Discrete Prolate Spheroidal Sequence (DPSS) tapers for use in
    multitaper spectral analysis.

    Uses scipy.signal.windows.dpss, but arguments are different here

    tapers = compute_tapers(smp_rate,time_width=0.5,freq_width=4,n_tapers=None)

    ARGS
    smp_rate    Scalar. Data sampling rate (Hz)

    time_width  Scalar. Time bandwidth 'T' (s). Should match data window length.
                Default: 0.5 (500 ms)

    freq_width  Scalar. Frequency bandwidth 'W' (Hz). Default: 4 Hz

    n_tapers    Scalar. Number of tapers to compute. Must be <= 2TW-1, as this is
                the max number of spectrally delimited tapers. Default: 2TW-1

    RETURNS
    tapers      (n_samples,n_tapers) ndarray. Computed dpss taper functions
                (n_samples = T*smp_rate)

    SOURCE  Adapted from Cronux function dpsschk.m
    """
    # Time-frequency bandwidth product 'TW' (s*Hz)
    time_freq_prod  = time_width*freq_width

    # Up to 2TW-1 tapers are bounded; this is both the default and max value for n_tapers
    n_tapers_max = floor(2*time_freq_prod - 1)
    if n_tapers is None: n_tapers = n_tapers_max

    assert n_tapers <= n_tapers_max, \
        ValueError("For time-freq product = %.1f, %d tapers are tightly bounded in" \
                    "frequency (n_tapers set = %d)" \
                    % (time_freq_prod,n_tapers_max,n_tapers))

    # Convert time bandwidth from s to window length in number of samples
    n_samples = int(round(time_width*smp_rate))

    # Compute the tapers for given window length and time-freq product
    # Note: dpss() normalizes by sum of squares; x sqrt(smp_rate)
    #       converts this to integral of squares (see Chronux function dpsschk())
    # Note: You might imagine you'd want sym=False, but sym=True gives same values
    #       as Chronux dpsschk() function...
    return dpss(n_samples, time_freq_prod, Kmax=n_tapers, sym=True, norm=2).T * sqrt(smp_rate)


# =============================================================================
# Wavelet analysis functions
# =============================================================================
def wavelet_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                     freqs=2**np.arange(1,7.5,0.25), removeDC=True,
                     wavelet='morlet', wavenumber=6, pad=False, buffer=0, **kwargs):
    """
    Computes continuous wavelet transform of data, then averages across timepoints to
    reduce it down to a frequency spectrum.

    Not really the best way to compute 1D frequency spectra, but included for completeness

    spec,freqs,timepts = wavelet_spectrum(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',
                                          freqs=2**np.arange(1,7.5,0.25),removeDC=True,
                                          wavelet='morlet',wavenumber=6,
                                          pad=True,buffer=0, **kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies
                Default: 2**np.irange(1,7.5,0.25) (log sampled in 1/4 octaves from 2-128)

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    wavelet     String. Name of wavelet type. Default: 'morlet'

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
                Default: 6

    pad         Bool. If True, zero-pads data to next power of 2 length. Default: False

    buffer      Float. Time (s) to trim off each end of time dimension of data.
                Removes symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    RETURNS
    spec        (...,n_freqs,...) ndarray of complex | float.
                Wavelet-derived spectrum of data.
                Same shape as data, with frequency axis replacing time axis
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)
    """
    if axis < 0: axis = data.ndim + axis

    spec, freqs, _ = wavelet_spectrogram(data, smp_rate, axis=axis, data_type=data_type,
                                         spec_type=spec_type, freqs=freqs, removeDC=removeDC,
                                         wavelet=wavelet, wavenumber=wavenumber, pad=pad,
                                         buffer=buffer, **kwargs)

    # Take mean across time axis (which is now shifted +1 b/c of frequency axis)
    return spec.mean(axis=axis+1), freqs


def wavelet_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freqs=2**np.arange(1,7.5,0.25), removeDC=True,
                        wavelet='morlet', wavenumber=6, pad=False, buffer=0, downsmp=1, **kwargs):
    """
    Computes continuous time-frequency wavelet transform of data at given frequencies.

    spec,freqs,timepts = wavelet_spectrogram(data,smp_rate,axis=0,data_type='lfp',
                                             spec_type='complex',freqs=2**np.arange(1,7.5,0.25),
                                             removeDC=True,wavelet='morlet',wavenumber=6,
                                             pad=True,buffer=0,downsmp=1, **kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies
                Default: 2**np.irange(1,7.5,0.25) (log sampled in 1/4 octaves from 2-128)

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    wavelet     String. Name of wavelet type. Default: 'morlet'

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
                Default: 6

    pad         Bool. If True, zero-pads data to next power of 2 length. Default: False

    buffer      Float. Time (s) to trim off each end of time dimension of data.
                Removes symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    downsmp     Int. Factor to downsample time sampling by (after spectral analysis).
                eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)
                Default: 1 (no downsampling)

    RETURNS
    spec        (...,n_freqs,n_timepts_out,...) ndarray of complex | float.
                Wavelet time-frequency spectrogram of data, transformed to requested spectral type.
                Same shape as data, with frequency axis prepended before time, and time axis
                possibly reduces via downsampling.
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timepts_out,) ndarray. List of timepoints in <spec> (in s, referenced
                to start of data).

    REFERENCE   Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"
    """
    if axis < 0: axis = data.ndim + axis

    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        lims    = kwargs.pop('lims',None)
        bins    = kwargs.pop('bins',None)
        data,_  = times_to_bool(data, width=1/smp_rate, lims=lims, bins=bins)
        axis    = data.ndim
    assert len(kwargs) == 0, \
        TypeError("Incorrect or misspelled variable(s) in keyword args: "+', '.join(kwargs.keys()))

    # Convert buffer from s -> samples
    if buffer != 0:  buffer  = int(ceil(buffer*smp_rate))

    # Reshape data array -> (n_timepts_in,n_dataseries) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_timepts_in = data.shape[0]

    # Time indexes to extract from spectrogram for output (accounting for buffer, downsampling)
    time_idxs_out = np.arange(buffer,n_timepts_in-buffer,downsmp)

    # Set FFT length = data length if no padding; else pad to next power of two
    if not pad: n_fft = n_timepts_in
    else:       n_fft = _next_power_of_2(n_timepts_in)

    # Compute set of Fourier-transformed wavelet functions (if not already given)
    if isinstance(wavelet,str):
        wavelets_fft = compute_wavelets(n_fft,smp_rate,freqs=freqs,
                                        wavelet=wavelet,wavenumber=wavenumber,
                                        do_fft=True)
    else:
        wavelets_fft = wavelet

    if removeDC: data = remove_dc(data,axis=0)

    # Compute FFT of data
    data = fft(data, n=n_fft,axis=0, **_FFTW_KWARGS_DEFAULT)

    # Reshape data -> (1,n_timepts,n_series) (insert axis 0 for wavelet scales/frequencies)
    # Reshape wavelets -> (n_freqs,n_timepts,1) to broadcast
    #  (except for special case of 1D data with only a single time series)
    data = data[np.newaxis,...]
    if data.ndim == 3: wavelets_fft = wavelets_fft[:,:,np.newaxis]

    # Convolve data with wavelets (multiply in Fourier domain)
    # -> inverse FFT to get wavelet transform
    spec = ifft(data*wavelets_fft, n=n_fft,axis=1, **_FFTW_KWARGS_DEFAULT)[:,time_idxs_out,...]

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)

    spec    = _undo_standardize_array_newaxis(spec,data_shape,axis=axis)

    timepts = time_idxs_out.astype(float)/smp_rate  # Convert time sampling from samples -> s

    return spec, freqs, timepts


def compute_wavelets(n, smp_rate, freqs=2**np.arange(1,7.5,0.25),
                     wavelet='morlet', wavenumber=6, do_fft=False):
    """
    Computes set of (Fourier transformed) wavelets for use in wavelet spectral analysis

    wavelets = compute_wavelets(n,smp_rate,freqs=2**np.arange(1,7.5,0.25),
                                 wavelet='morlet',wavenumber=6,do_fft=False)

    ARGS
    n           Int. Total number of samples (time points) in analysis,
                including any padding.

    smp_rate     Scalar. Data sampling rate (Hz)

    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies
                Default: 2**np.irange(1,7.5,0.25) (log sampled in 1/4 octaves from 2-128)

    wavelet     String. Name of wavelet type. Default: 'morlet'
                Currently 'morlet' is only supported value.

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. For Morlet, must be >= 6 to meet "admissibility constraint".
                Default: 6

    do_fft      Bool. If True, returns Fourier transform of wavelet functions;
                If False [default], returns original time-domain functions

    RETURNS
    wavelets    (n_freqs,n_timepts). Computed set of wavelet functions at multiple
                frequencies/scales. (either the time domain wavelets or their
                Fourier transform, depending on <do_fft> argument)

    REFERENCE   Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"

    SOURCE      Adapted from Torrence & Compo's Matlab wavelet toolbox
    """
    freqs   = np.asarray(freqs)
    wavelet = wavelet.lower()
    dt      = 1/smp_rate         # Convert sampling rate -> sampling interval

    if wavelet == 'morlet':
        # Conversion factor from scale to Fourier period for Morlet wavelets [T&C Table 1]
        # period = 1/frequency = scale * scale_to_period
        scale_to_period = (4.0*pi)/(wavenumber + sqrt(2.0 + wavenumber**2))

        # Convert desired frequencies -> scales for full set of wavelets
        scales          = (1.0/freqs) / scale_to_period
        scales          = scales[:,np.newaxis] # -> (n_freqs,1)

        # Construct wavenumber array used in transform [T&C Eqn(5)]
        k   = np.arange(1, int(np.fix(n/2.0)+1))
        k   = k*((2.0*pi)/(n*dt))
        k   = np.hstack((np.zeros((1,)), k, -k[int(np.fix((n-1)/2)-1) : : -1]))
        k   = k[np.newaxis,:]   # -> (1,n_timepts_in)
        k0  = wavenumber

        # Total energy=N   [T&C Eqn(7)]
        normalization   = np.sqrt(scales*k[:,1])*(pi**(-0.25))*sqrt(n)
        # Wavelet exponent
        exponent        = -0.5*(scales*k - k0)**2 * (k > 0)

        # Fourier transform of Wavelet function
        if do_fft:
            wavelets = normalization*np.exp(exponent) * (k > 0)
        else:
            raise NotImplementedError("non-FFT wavelet output not coded up yet (TODO)")

    else:
        raise ValueError("Unsupported value '%s' given for <wavelet>." \
                         "Currently only 'Morlet' suppported")

    return wavelets


def wavelet_bandwidth(freqs, wavelet='morlet', wavenumber=6, full=True):
    """
    Returns frequency and time bandwidths for set of wavelets at given frequencies

    freq_widths,time_widths = wavelet_bandwidth(freqs,wavelet='morlet',wavenumber=6,full=True)

    ARGS
    freqs       (n_freqs,) array-like. Set of center wavelet frequencies.

    wavelet     String. Name of wavelet type. Default: 'morlet'
                Currently 'morlet' is only supported value.

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                Default: 6

    full        Bool. If True [default], returns full-bandwidths.
                If False, returns full bandwidths.

    RETURNS
    freq_widths (n_freqs,) ndarray. Frequency bandwidth (Hz) for each given frequency

    time_widths (n_freqs,) ndarray. Time bandwidth (s) for each given frequency
    """
    wavelet = wavelet.lower()
    freqs = np.asarray(freqs)

    if wavelet == 'morlet':
        freq_widths = freqs / wavenumber
        time_widths = 1 / (2*pi*freq_widths)

    else:
        raise ValueError("Unsupported value '%s' given for <wavelet>." \
                         "Currently only 'Morlet' suppported")

    # Convert half-bandwidths to full-bandwidths
    if full:
        freq_widths = 2 * freq_widths
        time_widths = 2 * time_widths

    return freq_widths, time_widths


def wavelet_edge_extent(freqs, wavelet='morlet', wavenumber=6):
    """
    Returns temporal extent of edge effects for set of wavelets at given frequencies

    Computes time period over which edge effects might effect output of wavelet transform,
    and over which the effects of a single spike-like artifact in data will extend.

    Computed as time for wavelet power to drop by a factor of exp(−2), ensuring that
    edge effects are "negligible" beyond this point.

    edge_extent = wavelet_edge_extent(freqs,wavelet='morlet',wavenumber=6)

    ARGS
    freqs       (n_freqs,) array-like. Set of center wavelet frequencies.

    wavelet     String. Name of wavelet type. Default: 'morlet'
                Currently 'morlet' is only supported value.

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                Default: 6

    RETURNS
    edge_extent (n_freqs,) ndarray. Time period (s) over which edge effects
                extend for each given frequency

    REFERENCE   Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis" Sxn.3g

    SOURCE      Adapted from Torrence & Compo's Matlab wavelet toolbox
    """
    wavelet = wavelet.lower()
    freqs = np.asarray(freqs)

    if wavelet == 'morlet':
        # "Fourier factor" (conversion factor from scale to Fourier period [T&C Table 1])
        scale_to_period = (4.0*pi)/(wavenumber + sqrt(2.0 + wavenumber**2))
        # Convert given frequencies -> scales for set of wavelets
        scales          = (1.0/freqs) / scale_to_period
        # Cone-of-influence for Morlet = sqrt(2)*scale [T&C Table 1]
        edge_extent     = sqrt(2.0) * scales

    else:
        raise ValueError("Unsupported value '%s' given for <wavelet>." \
                         "Currently only 'Morlet' suppported")

    return edge_extent


# =============================================================================
# Band-pass filtering functions
# =============================================================================
def bandfilter_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freqs=((2,8),(10,32),(40,100)), removeDC=True,
                        filt='butter', params=None, buffer=0, **kwargs):
    """
    Computes band-filtered and Hilbert-transformed signal from data
    for given frequency band(s), then reduces it to 1D frequency spectra by averaging across time.

    Not really the best way to compute 1D frequency spectra, but included for completeness

    spec,freqs,timepts = bandfilter_spectrum(data,smp_rate,axis=0,data_type='lfp',
                                             spec_type='complex',freqs=((2,8),(10,32),(40,100)),
                                             filt='butter',params=None,buffer=0,**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freqs       (n_freqbands,) array-like of (2,) sequences | (n_freqbands,2) ndarray.
                List of (low,high) cut frequencies for each band to use.
                Set 1st value = 0 for low-pass; set 2nd value >= smp_rate/2 for
                high-pass; otherwise assumes band-pass.
                Default: ((2,8),(10,32),(40,100)) (~theta, alpha/beta, gamma)
                ** Only used if filter <params> not explicitly given **

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    filt        String. Name of filter to use. Default: 'butter' (Butterworth)
                ** Only used if filter <params> not explitly given **

    params      Dict. Contains parameters that define filter for each freq band.
                Can precompute params with set_filter_params() and input explicitly
                OR input values for <freqs> and params will be computed here.
                One of two forms: 'ba' or 'zpk', with key/values as follows:

        b,a     (n_freqbands,) lists of vectors. Numerator <b> and
                denominator <a> polynomials of the filter for each band.

        z,p,k   Zeros, poles, and system gain of the IIR filter transfer function

    buffer      Float. Time (s) to trim off each end of time dimension of data.
                Removes symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    **kwargs    Any other kwargs passed directly to set_filter_params()

    RETURNS
    spec        (...,n_freqbands,...) ndarray of complex floats.
                Band-filtered, Hilbert-transformed data, transformed to requested spectral
                type and averaged across the time axis to 1D frequency spectra.
                Same shape as input data, but with frequency axis replacing time axis.
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqbands,2) ndarray. List of (low,high) cut frequencies (Hz)
                for each band used.
    """
    if axis < 0: axis = data.ndim + axis

    spec, freqs, _ = bandfilter_spectrogram(data, smp_rate, axis=axis, data_type=data_type,
                                            spec_type=spec_type, freqs=freqs, removeDC=removeDC,
                                            filt=filt, params=params, buffer=buffer, **kwargs)

    # Take mean across time axis (which is now shifted +1 b/c of frequency axis)
    return spec.mean(axis=axis+1), freqs


def bandfilter_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                           freqs=((2,8),(10,32),(40,100)), removeDC=True,
                           filt='butter', order=4, params=None, buffer=0, downsmp=1, **kwargs):
    """
    Computes zero-phase band-filtered and Hilbert-transformed signal from data
    for given frequency band(s).

    Function aliased as bandfilter_spectrogram() or bandfilter().

    spec,freqs,timepts = bandfilter_spectrogram(data,smp_rate,axis=0,data_type='lfp',
                                                spec_type='complex',freqs=((2,8),(10,32),(40,100)),
                                                filt='butter', order=4, params=None,
                                                buffer=0 ,downsmp=1, **kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    data_type   String. Type of signal in data: 'lfp' [default] or 'spike'

    spec_type   String. Type of spectral signal to return: 'complex' [default] |
                'power' | 'phase' | 'real' | 'imag'. See complex_to_spec_type for details.

    freqs       (n_freqbands,) array-like of (2,) sequences | (n_freqbands,2) ndarray.
                List of (low,high) cut frequencies for each band to use.
                Set 1st value = 0 for low-pass; set 2nd value >= smp_rate/2 for
                high-pass; otherwise assumes band-pass.
                Default: ((2,8),(10,32),(40,100)) (~theta, alpha/beta, gamma)
                ** Only used if filter <params> not explicitly given **

    removeDC    Bool. If True, removes mean DC component across time axis,
                making signals zero-mean for spectral analysis. Default: True

    filt        String. Name of filter to use. Default: 'butter' (Butterworth)
                Ignored if filter <params> explitly given.

    order       Int. Filter order. Default: 4
                Ignored if filter <params> explitly given.

    NOTE: Can specify filter implictly using <filt,order> OR explicitly using <params>.
          If <params> is input, filt and order are ignored.

    params      Dict. Contains parameters that define filter for each freq band.
                Can precompute params with set_filter_params() and input explicitly
                OR input values for <freqs> and params will be computed here.
                One of two forms: 'ba' or 'zpk', with key/values as follows:

        b,a     (n_freqbands,) lists of vectors. Numerator <b> and
                denominator <a> polynomials of the filter for each band.

        z,p,k   Zeros, poles, and system gain of the IIR filter transfer function

    buffer      Float. Time (s) to trim off each end of time dimension of data.
                Removes symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    downsmp     Int. Factor to downsample time sampling by (after spectral analysis).
                eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)
                Default: 1 (no downsampling)

    **kwargs    Any other kwargs passed directly to set_filter_params()

    RETURNS
    spec        (...,n_freqbands,n_timepts_out,...) ndarray of complex | float.
                Band-filtered, Hilbert-transformed "spectrogram" of data, transformed to
                requested spectral type.
                Same shape as input data, but with frequency axis prepended immediately
                before time <axis>.
                dtype is complex if spec_type='complex', float otherwise.

    freqs       (n_freqbands,2) ndarray. List of (low,high) cut frequencies (Hz)
                for each band used.

    timepts     (n_timepts_out,) ndarray. List of timepoints in <spec> (in s, referenced
                to start of data).
    """
    if axis < 0: axis = data.ndim + axis

    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        lims    = kwargs.pop('lims',None)
        bins    = kwargs.pop('bins',None)
        data,_  = times_to_bool(data, width=1/smp_rate, lims=lims, bins=bins)
        axis    = data.ndim

    # Convert buffer from s -> samples
    if buffer != 0:  buffer  = int(ceil(buffer*smp_rate))

    # Set filter parameters from frequency bands if <params> not explicitly passed in
    if params is None:
        assert freqs is not None, \
            ValueError("Must input a value for either filter <params> or band <freqs>")

        freqs   = np.asarray(freqs)  # Convert freqs to (n_freqbands,2)
        n_freqs = freqs.shape[0]
        params  = set_filter_params(freqs, smp_rate, filt=filt, order=order,
                                    form='ba', return_dict=True, **kwargs)

    # Determine form of filter parameters given: b,a or z,p,k
    else:
        assert len(kwargs) == 0, \
            TypeError("Incorrect or misspelled variable(s) in keyword args: " +
                      ', '.join(kwargs.keys()))

        if np.all([(param in params) for param in ['b','a']]):       form = 'ba'
        elif np.all([(param in params) for param in ['z','p','k']]): form = 'zpk'
        else:
            raise ValueError("<params> must be a dict with keys 'a','b' or 'z','p','k'")

        # Convert zpk form to ba
        if form == 'zpk':
            n_freqs = len(params['z'])
            params['b'] = [None] * n_freqs
            params['a'] = [None] * n_freqs
            for i_freq in range(n_freqs):
                b,a = zpk2tf(params['z'][i_freq],params['p'][i_freq],params['k'][i_freq])
                params['b'][i_freq] = b
                params['a'][i_freq] = a
        else:
            n_freqs = len(params['b'])

    # Set any freqs > Nyquist equal to Nyquist
    if freqs is not None: freqs[freqs > smp_rate/2] = smp_rate/2

    # Reshape data array -> (n_timepts_in,n_dataseries) matrix
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    # Temporarily append singleton axis to vector-valued data to simplify code
    vector_data = data.ndim == 1
    if vector_data: data = data[:,np.newaxis]

    n_timepts_in,n_series = data.shape

    # Time indexes to extract from spectrogram for output (accounting for buffer, downsampling)
    time_idxs_out   = np.arange(buffer,n_timepts_in-buffer,downsmp)
    n_timepts_out   = len(time_idxs_out)

    if removeDC: data = remove_dc(data,axis=0)

    dtype = float if spec_type == 'real' else complex
    spec = np.empty((n_freqs,n_timepts_out,n_series),dtype=dtype)

    # For each frequency band, band-filter raw signal and
    # compute complex analytic signal using Hilbert transform
    for i_freq,(b,a) in enumerate(zip(params['b'],params['a'])):
        bandfilt = filtfilt(b, a, data, axis=0, method='gust')
        # Note: skip Hilbert transform for real output
        spec[i_freq,:,:] = bandfilt[time_idxs_out,:] if spec_type == 'real' else \
                           hilbert(bandfilt[time_idxs_out,:],axis=0)

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)

    if vector_data: spec = spec.squeeze(axis=-1)
    spec = _undo_standardize_array_newaxis(spec,data_shape,axis=axis)

    timepts = time_idxs_out.astype(float)/smp_rate  # Convert time sampling from samples -> s

    return spec, freqs, timepts

bandfilter = bandfilter_spectrogram  # Alias function as bandfilter()


def set_filter_params(bands, smp_rate, filt='butter', order=4, form='ba',
                      return_dict=False, **kwargs):
    """
    Sets coefficients for desired filter(s) using scipy.signal
    "Matlab-style IIR filter design" functions

    params = set_filter_params(bands,smp_rate,filt='butter',order=4,form='ba',
                               return_dict=True,**kwargs)
    b,a = set_filter_params(bands,smp_rate,filt='butter',order=4,form='ba',
                            return_dict=False,**kwargs)
    z,p,k = set_filter_params(bands,smp_rate,filt='butter',order=4,form='zpk',
                              return_dict=False,**kwargs)

    ARGS
    bands       (n_freqbands,) array-like of (2,) sequences | (n_freqbands,2) ndarray.
                [low,high] cut frequencies for each band to use.
                Set 1st value = 0 for low-pass; set 2nd value >= smp_rate/2 for
                high-pass; otherwise assumes band-pass.

    smp_rate    Scalar. Data sampling rate (Hz)

    filt        String. Name of filter to use: 'butter'|'ellip'|'cheby1'|'cheby2'
                Default: 'butter'

    order       Int. Filter order. Default: 4

    form        String. Type of parameters output. 'ba': numerator/denominator
                b,a or 'zpk': pole-zero z, p, k. Default: ‘ba’.

    return_dict Bool. If True, params returned in a dict; else as a tuple.

    Any additional kwargs passed directly to filter function

    RETURNS
    If return_dict is False, outputs are returned as a tuple, as described below;
    else, outputs are packaged in a single dict, with param names as keys.

    b,a         (n_freqbands,) list of vectors. Numerator <b> and
                denominator <a> polynomials of the filter for each band.
                Returned if form='ba'.

    z,p,k       Zeros, poles, and system gain of the IIR filter transfer
                function. Returned if form='zpk'.
    """
    # Convert bands to (n_freqbands,2)
    bands       = np.asarray(bands)
    if bands.ndim == 1: bands = np.reshape(bands,(1,len(bands)))
    n_bands     = bands.shape[0]
    nyquist     = smp_rate/2.0   # Nyquist freq at given sampling freq

    # Setup filter-generating function for requested filter type
    # Butterworth filter
    if filt.lower() in ['butter','butterworth']:
        gen_filt = lambda band,btype: butter(order,band,btype=btype,output=form)
    # Elliptic filter
    elif filt.lower() in ['ellip','butterworth']:
        rp = kwargs.pop('rp',5)
        rs = kwargs.pop('rs',40)
        gen_filt = lambda band,btype: ellip(order,rp,rs,band,btype=btype,output=form)
    # Chebyshev Type 1 filter
    elif filt.lower() in ['cheby1','cheby','chebyshev1','chebyshev']:
        rp = kwargs.pop('rp',5)
        gen_filt = lambda band,btype: cheby1(order,rp,band,btype=btype,output=form)
    # Chebyshev Type 2 filter
    elif filt.lower() in ['cheby2','chebyshev2']:
        rs = kwargs.pop('rs',40)
        gen_filt = lambda band,btype: cheby2(order,rs,band,btype=btype,output=form)
    else:
        raise ValueError("Filter type '%s' is not supported (yet)" % filt)
    assert len(kwargs) == 0, \
        TypeError("Incorrect or misspelled variable(s) in keyword args: "+', '.join(kwargs.keys()))

    # Setup empty lists to hold filter parameters
    if form == 'ba':    params = OrderedDict({'b':[None]*n_bands, 'a':[None]*n_bands})
    elif form == 'zpk': params = OrderedDict({'z':[None]*n_bands, 'p':[None]*n_bands,
                                              'k':[None]*n_bands})
    else:
        raise ValueError("Output form '%s' is not supported. Should be 'ba' or 'zpk'" % form)

    for i_band,band in enumerate(bands):
        band_norm = band/nyquist  # Convert band to normalized frequency

        # If low-cut freq = 0, assume low-pass filter
        if band_norm[0] == 0:   btype = 'lowpass';  band_norm = band_norm[1]
        # If high-cut freq >= Nyquist freq, assume high-pass filter
        elif band_norm[1] >= 1: btype = 'highpass'; band_norm = band_norm[0]
        # Otherwise, assume band-pass filter
        else:                   btype = 'bandpass'

        if form == 'ba':
            params['b'][i_band],params['a'][i_band] = gen_filt(band_norm,btype)
        else:
            params['z'][i_band],params['p'][i_band],params['k'][i_band] = gen_filt(band_norm,btype)

    if return_dict: return params
    else:           return params.values()


# =============================================================================
# Other spectral analysis functions
# =============================================================================
def itpc(data, smp_rate, axis=0, method='wavelet', itpc_method='PLV', trial_axis=None, **kwargs):
    """
    Intertrial phase clustering (ITPC) measures frequency-specific phase locking of continuous
    neural activity (eg LFPs) to trial events. A spectral analog (roughly) of evoked potentials.

    aka "intertrial coherence", "intertrial phase-locking value/factor"

    ARGS
    data        (...,n_timepts,...,n_trials,...) ndarray. Data to compute ITPC of,
                aligned to some within-trial event.
                Arbitrary shape; spectral analysis is computed along <axis>,
                and ITPC is computed along <trial_axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform
                'burst' :       Oscillatory burst analysis

    itpc_method String. Method to use for computing intertrial phase clustering:
                'PLV' : Phase locking value (length of cross-trial complex vector mean)
                        Standard/traditional measure of ITPC, but is positively biased
                        for small n, and is biased > 0 even for no phase clustering.
                'Z' :   Rayleigh's Z normalization of PLV (Z = n*PLV**2). Reduces small-n bias.
                'PPC' : Pairwise Phase Consistency normalization (PPC = (n*PLV**2 - 1)/(n - 1))
                        of PLV. Debiased and has expected value 0 for no clustering.
                Default: 'PLV'

    **kwargs    All other kwargs passed directly to method-specific
                spectrogram function. See there for details.

    RETURNS
    ITPC        (...,n_freqs,n_timepts,...) ndarray.
                Time-frequency spectrogram representation of intertrial phase clustering.
                Frequency axis is always inserted just before time axis, and trial axis is
                removed, but otherwise shape is same as input data

    freqs       For method='bandfilter': (n_freqbands,2) ndarray. List of (low,high) cut
                frequencies (Hz) used to generate <ITPC>.
                For other methods: (n_freqs,) ndarray. List of frequencies in <ITPC> (Hz).

    timepts     (n_timepts,) ndarray. List of time points / time window centers in <ITPC>
                (in s, referenced to start of data).

    REFERENCE
    Cohen _Analyzing Neural Time Series Data_ Ch. 19
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

# Also alias itpc as intertrial_phase_clustering
intertrial_phase_clustering = itpc


def burst_analysis(data, smp_rate, axis=0, trial_axis=-1, threshold=2, min_cycles=3,
                   method='wavelet', spec_type='power', freq_exp=None,
                   bands=((20,35),(40,65),(55,90),(70,100)),
                   window=None, timepts=None, **kwargs):
    """
    Computes oscillatory burst analysis of Lundqvist et al 2016.
    Default argument values approximate analysis as implemented in there.

    Computes oscillatory power
    To compute burst rate, simply take mean across trial axis.


    bursts,freqs,timepts = burst_analysis(data, smp_rate, axis=0, trial_axis=-1,
                                          threshold=2, min_cycles=3,
                                          method='wavelet', spec_type='power', freq_exp=None,
                                          bands=((20,35),(40,65),(55,90),(70,100)),
                                          window=None, timepts=None,
                                          **kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    trial_axis  Int. Axis of <data> corresponding to trials/observations.
                Default: -1 (last axis of data)

    threshold   Scalar. Threshold power level for detecting bursts, given in SDs above the mean.
                Default: 2 SDs

    min_cycles  Scalar. Minimal length of contiguous above-threshold period to be counted as a
                burst, given in number of oscillatory cycles at each frequency (or band center).
                Default: 3

    method      String. Underlying time-frequency spectral analysis method to use,
                which burst analysis is computed on.
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform
                Note: In the original paper, multitaper was used, but all three
                methods were claimed to produced similar results.

    spec_type   String. Type of spectral signal to compute. Default: 'power'
                'power' : Spectral power, ie square of signal envelope
                'magnitude' : Square root of power, ie signal envelope

    freq_exp    Float. This can be used to normalize out 1/f^a effects in power before
                band-pooling and burst detection). This gives the exponent on the frequency
                ('a' in 1/f^a).  Set = 1 to norm by 1/f.  Set = None for no normalization.
                Default: None (do no frequency normalization)

    bands       (n_freqbands,) array-like of (2,) sequences | (n_freqbands,2) ndarray.
                List of (low,high) cut frequencies for each band to compute bursts within.
                Set 1st value = 0 for low-pass; set 2nd value >= smp_rate/2 for
                high-pass; otherwise assumes band-pass.
                Set = None to compute bursts at each frequency in spectral transform.
                Default: ((20,35),(40,65),(55,90),(70,100)) (beta, low/med/high gamma)

    window      (2,) array-like. (start,end) of time window to compute mean,SD for
                burst amplitude threshold within (in same units as <timepts>).
                Default: None (compute over entire data time range)

    timepts     (n_timepts,) array-like. Time sampling vector for data (usually in s).
                Necessary if <window> is set, but unused otherwise.  Default: None

    **kwargs    Any other kwargs passed directly to power_spectrogram() function

    RETURNS
    bursts      (...,n_freq[band]s,n_timepts_out,...) ndarray of bool. Binary array labelling
                timepoints within detected bursts in each trial and frequency band.
                Same shape as input data, but with frequency axis prepended immediately
                before time <axis>.

    freqs       (n_freq[band]s,) ndarray. List of center frequencies of bands in <bursts>

    timepts     (n_timepts_out,) ndarray. List of time points / time window centers in <spec>
                (in s, referenced to start of data).

    REFERENCE
    Lundqvist, ..., & Miller (2016) Neuron "Gamma and Beta Bursts Underlie Working Memory"
    Lundqvist, ..., & Miller (2018) Nature Comm "Gamma and beta bursts during working memory
                                                readout suggest roles in its volitional control"
    """
    # todo  Gaussian fits for definining burst f,t extent?
    # todo  Option input of spectral data?
    # todo  Add optional sliding trial window for mean,SD
    method = method.lower()
    spec_type = spec_type.lower()
    if bands is not None: bands = np.asarray(bands)
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
    Plots frequency spectrum as a line plot.

    lines,ax = plot_spectrum(freqs, data, ylim=None, color=None, **kwargs)

    ARGS
    freqs   (n_freqs,) array-like. Frequency sampling vector for data (Hz).
            May be linearly or logarithmically sampled; we deal appropriately.

    data    (n_freqs,) ndarray. Frequency spectrum data.

    ax      Pyplot Axis object. Axis to plot into. Default: plt.gca()

    ylim    (2,) array-like. y-axis limits: (min,max)
            Default: min/max over all data +/- 5%  (data.min(),data.max())

    color   (3,) array-like | Color spec. Color to plot in.
            Default: <default plot color>

    **kwargs All other keyword args passed directly to plt.plot()

    ACTION
    Plots spectrum data into given axes using plt.plot()

    RETURNS
    lines   List of Line2D objects. Output of plt.plot()
    ax      Pyplot Axis object. Axis for plot
    """
    freqs,fticks,fticklabels = frequency_plot_settings(freqs)
    
    lines, _, ax = plot_line_with_error_fill(freqs, data, ax=ax, ylim=ylim, color=color, **kwargs)
    
    plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
    ax.set_xticks(fticks)
    ax.set_xticklabels(fticklabels)
    # No need to return list of lists if only plotting one data series
    if (data.ndim == 1) or (data.shape[0] == 1): lines = lines[0]    
    
    return lines, ax


def plot_spectrogram(timepts, freqs, data, ax=None, clim=None, cmap='viridis', **kwargs):
    """
    Plots time-frequency spectrogram as a heatmap plot.

    img,ax = plot_spectrogram(timepts, freqs, data, clim=None, cmap='viridis', **kwargs)

    ARGS
    timepts (n_timepts,) array-like. Time sampling vector for data

    freqs   (n_freqs,) array-like. Frequency sampling vector for data (Hz).
            May be linearly or logarithmically sampled; we deal appropriately.

    data    (n_freqs,n_timepts) ndarray. Time-frequency (spectrogam) data

    ax      Pyplot Axis object. Axis to plot into. Default: plt.gca()

    clim    (2,) array-like. Color axis limits: (min,max)
            Default: min/max over all data (data.min(),data.max())

    cmap    String. Colormap to plot in. Default: 'viridis'

    **kwargs All other keyword args passed directly to plt.imshow()

    ACTION
    Plots spectrogram data into given axes using plt.imshow()

    RETURNS
    img     AxesImage object. Output of plt.imshow()
    ax      Pyplot Axis object. Axis for plot
    """
    freqs,fticks,fticklabels = frequency_plot_settings(freqs)
    
    img, ax = plot_heatmap(timepts, freqs, data, ax=ax, clim=clim, cmap=cmap,
                           origin='lower', **kwargs)

    plt.grid(axis='y',color=[0.75,0.75,0.75],linestyle=':')
    ax.set_yticks(fticks)
    ax.set_yticklabels(fticklabels)

    return img, ax


def frequency_plot_settings(freqs):
    """ Returns settings for plotting a frequency axis: plot freqs, ticks, tick labels """
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
    Cuts continuous (eg LFP) data into trials

    cut_data = cut_trials(data, trial_lims, smp_rate, axis=0)

    ARGS
    data        (...,n_timepts,...) array. Continuous data unsegmented into trials.
                Arbitrary dimensionality, could include multiple channels, etc.

    trial_lims  (n_trials,2) array-like. List of [start,end] of each trial (in s)
                to use to cut data.

    smp_rate    Scalar. Sampling rate of data (Hz).

    axis        Int. Axis of data array corresponding to time samples. Default: 0

    RETURNS
    cut_data    (...,n_trial_timepts,...,n_trials) array.
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


def realign_data(data, align_times, time_range=None, timepts=None, time_axis=0, trial_axis=-1):
    """
    Realigns trial-cut continuous (eg LFP) data to new set of within-trial times
    (eg new trial event) so that t=0 on each trial at given event.
    For example, data aligned to a start-of-trial event might
    need to be relaligned to the behavioral response.

    realigned = realign_data(data,align_times,time_range,timepts,time_axis=0,trial_axis=-1)

    ARGS
    data        ndarray. Continuous data segmented into trials.
                Arbitrary dimensionality, could include multiple channels, etc.

    align_times (n_trials,) array-like. New set of times (in old
                reference frame) to realign data to (in s)

    time_range  (2,) array-like. Time range to extract from each trial around
                new align time ([start,end] in s relative to align_times).
                eg, time_range=(-1,1) -> extract 1 s on either side of align event.
                Must set value for this.

    timepts     (n_timepts) array-like. Time sampling vector for data (in s).
                Must set value for this.

    time_axis   Int. Axis of data corresponding to time samples.
                Default: 0 (1st axis of array)

    trial_axis  Int. Axis of data corresponding to distinct trials.
                Default: -1 (last axis of array)

    RETURNS
    realigned   Data realigned to given within-trial times.
                Time axis is reduced to length implied by time_range, but otherwise
                array has same shape as input data.
    """
    assert time_range is not None, \
        "Desired time range to extract from each trial must be given in  <time_range>"
    assert timepts is not None, "Data time sampling vector must be given in <timepts>"

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
    Realigns trial-cut continuous (eg LFP) data to new within-trial event
    so that t=0 on each trial at given event.

    Convenience wrapper around realign_data() for relaligning to a given
    named event within a per-trial dataframe or dict variable.

    realigned = realign_data_on_event(data,event_data,event,timepts,align_times,time_range,
                                      time_axis=0,trial_axis=-1)

    ARGS
    data        ndarray. Continuous data segmented into trials.
                Arbitrary dimensionality, could include multiple channels, etc.

    event_data  {string:(n_trials,) array} dict | (n_trials,n_events) DataFrame.
                Per-trial event timing data to use to realign spike timestamps.

    event       String. Dict key or DataFrame column name whose associated values
                are to be used to realign spike timestamps

    See realign_data() for details on rest of arguments

    RETURNS
    realigned   Data realigned to given trial event.
                Time axis is reduced to length implied by time_range, but otherwise
                array has same shape as input data.
    """
    # Extract vector of times to realign on
    align_times = event_data[event]

    # Compute the realignment and return
    return realign_data(data, timepts, align_times, time_range,
                        time_axis=time_axis, trial_axis=trial_axis)


def get_freq_sampling(smp_rate,n_fft,freq_range=None,two_sided=False):
    """
    Returns frequency sampling vector (axis) for a given FFT-based computation

    ARGS
    smp_rate     Scalar. Data sampling rate (Hz)

    n_fft        Scalar. Number of samples (timepoints) in FFT output

    freq_range   (2,) array-like | Scalar. Range of frequencies to keep in output,
                either given as an explicit [low,high] range or just a scalar
                giving the highest frequency to return.
                Default: all frequencies from FFT

    two_sided    Bool. If True, returns freqs for two-sided spectrum, including
                both positive and negative frequencies (which have same amplitude
                for all real signals).  If False [default], only returns positive
                frequencies, in range (0,smp_rate/2).

    RETURNS
    freqs       (n_freqs,) ndarray. Frequency sampling vector (in Hz)

    freq_bool   (n_fft,) ndarray of bools. Boolean vector flagging frequencies
                in FFT output to retain, given desired freq_range
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
    Removes constant DC component of signals, estimated as across-time mean
    for each time series (ie trial,channel,etc.)

    data = remove_dc(data,axis=None)

    ARGS
    data    (...,n_obs,...) ndarray. Raw data to remove DC component of.
            Can be any arbitary shape, with time sampling along axis <axis>

    axis    Int. Data axis corresponding to time.
            Default: remove DC component computed across *full* data array
            (mirrors behavior of np.mean)

    RETURNS
    data    (...,n_obs,...) ndarray.  Data with DC component removed
    """
    return data - data.mean(axis=axis,keepdims=True)


def remove_evoked(data, axis=0, method='mean', design=None):
    """
    Removes estimate of evoked potentials phase-locked to trial events,
    returning data with (in theory) only non-phase-locked induced components

    data = remove_evoked(data,axis=0,method='mean',design=None)

    ARGS
    data    (...,n_obs,...) ndarray. Raw data to remove evoked components from.
            Can be any arbitary shape, with observations (trials) along axis <axis>

    axis    Int. Data axis corresponding to distinct observations/trials. Default: 0

    method  String. Method to use for estimating evoked potentials (default: 'mean'):
            'mean'      : Grand mean signal across all observations (trials)
            'groupMean' : Mean signal across observations with each group in <design>
            'regress'   : OLS regresion fit of design matrix <design> to data

    design  (n_obs,...) array-like. Design matrix to fit to data (method=='regress')
            or group/condition labels for each observation (method=='groupMean')

    RETURNS
    data    (...,n_obs,...) ndarray.  Data with estimated evoked component removed
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

    ARGS
    data        ndarray of complex. Complex spectral (or time-frequency) data. Arbitrary shape.
    spec_type   String. Type of spectral signal to return:
                'power'     Spectral power of data
                'phase'     Phase of complex spectral data (in radians)
                'magnitude' Magnitude (square root of power) of complex data = signal envelope
                'real'      Real part of complex data
                'imag'      Imaginary part of complex data

    RETURNS
    data        ndarray of float. Computed spectral signal. Same shape as input.
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
    """ Computes power from complex spectral data  """
    return (data*data.conj()).real  # Note: .real fixes small float errors

def magnitude(data):
    """ Computes magnitude (square root of power) from complex spectral data  """
    return np.abs(data)

def phase(data):
    """ Computes phase of complex spectral data  """
    return np.angle(data)

def real(data):
    """ Returns real part of complex spectral data  """
    return data.real

def imag(data):
    """ Returns imaginary part of complex spectral data  """
    return np.imag(data)


def pool_freq_bands(data, bands, axis=None, freqs=None, func='mean'):
    """
    Pools (averages) spectral data within each of a given set of frequency bands

    data = pool_freq_bands(data,axis=0,method='mean',design=None)

    ARGS
    data    (...,n_freqs,...) ndarray | xarray DataArray.
            Raw data to pool within frequency bands. Any arbitary shape.

    bands   {'name':[low,high]} dict | [[low_1,high_1],...,[low_n,high_n].
            Frequency bands to pool data within, each given as pairs of
            [low-cut, high-cut] values in a list or dict keyed by band names.
            Band edges are inclusive.

    axis    Int. Data axis corresponding to frequency.
            Only needed if <data> is not an xarray DataArray
            with dimension named 'freq' or 'frequency'

    freqs   (n_freqs,) array-like. Frequency sampling in <data>.
            Only needed if <data> is not an xarray DataArray

    func    String | callable. Function to use to pool values within each frequency band.
            Default: 'mean' (mean across all frequencies in band)

    RETURNS
    data    (...,n_freqbands,...) ndarray | xarray DataArray.
            Data with values averaged within each of given frequency bands
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
    Pools (averages) spectral data within each of a given set of time epochs

    data = pool_time_epochs(data,epochs,axis=None,timepts=None,func='mean')

    ARGS
    data    (...,n_timepts,...) ndarray | xarray DataArray.
            Raw data to pool within time epochs. Any arbitary shape.

    epochs  {'name':[start,end]} dict | [[start_1,end_1],...,[start_n,end_n].
            Time epochs to pool data within, each given as a window of
            [start time, end time] values in a list or dict keyed by epoch names.
            Epoch edges are inclusive.

    axis    Int. Data axis corresponding to time.
            Only needed if <data> is not an xarray DataArray

    timepts (n_timepts,) array-like. Time sampling in <data>.
            Only needed if <data> is not an xarray DataArray

    func    String | callable. Function to use to pool values within each time epoch.
            Default: 'mean' (mean across all timepoints in band)

    RETURNS
    data    (...,nTimeEpochs,...) ndarray | xarray DataArray.
            Data with values averaged within each of given time epochs
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
    Normalizes for ~ 1/frequency**alpha baseline distribution of power
    by multiplying by frequency, raised to a given exponent

    data = one_over_f_norm(data, axis=None, freqs=None, exponent=1)

    ARGS
    data    (...,n_freqs,...) ndarray | xarray DataArray.
            Raw power data to normalize. Any arbitary shape.

    axis    Int. Data axis corresponding to frequency.
            Only needed if <data> is not an xarray DataArray
            with dimension named 'freq' or 'frequency'

    freqs   (n_freqs,) array-like. Frequency sampling in <data>.
            Only needed if <data> is not an xarray DataArray

    exponent Float. Exponent ('alpha') to raise freqs to for normalization.
            Default: 1

    RETURNS
    data    1/f normalized data. Same shape as input.
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


def one_sided_to_two_sided(data,freqs,smp_rate,freq_axis=0):
    """
    Converts a one-sided Fourier or wavelet transform output to the equivalent
    two-sided output, assuming conjugate symmetry across positive and negative
    frequencies (as is the case when the original signals were real).  Also
    extrapolates values for f=0, as is necessary for wavelet transforms.
    """
    assert np.isclose(freqs[-1],smp_rate/2), \
        "Need to have sampling up to 1/2 sampling rate (Nyquist freq=%d Hz)" % (smp_rate/2)

    # If f=0 is not in data, numerically extrapolate values for it
    if not np.isclose(freqs,0).any():
        f0 = interp1(freqs,data,0,axis=freq_axis,kind='cubic',fill_value='extrapolate')
        f0 = np.expand_dims(f0,freq_axis)
        data = np.concatenate((f0,data),axis=freq_axis)
        freqs = np.concatenate(([0],freqs))

    # Convert values at Nyquist freq to complex conjugate at negative frequency
    slices = axis_index_slices(freq_axis,-1,data.ndim)
    data[slices] = data[slices].conj()
    freqs[-1] *= -1

    # Replicate values for all freqs (s.t. 0 < f < nyquist)
    # as complex conjugates at negative frequencies
    idxs    = slice(-2,1,-1)
    slices  = axis_index_slices(freq_axis,idxs,data.ndim)
    data    = np.concatenate((data, data[slices].conj()), axis=freq_axis)
    freqs   = np.concatenate((freqs, -freqs[idxs]))

    return data,freqs


# =============================================================================
# Data simulation and testing functions
# =============================================================================
def simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=1.0, n_trials=1000,
                         freq_sd=0, amp_sd=0, phase_sd=0,
                         smp_rate=1000, time_range=1.0, burst_rate=0, burst_width=4, seed=None):
    """
    Generates synthetic data with oscillation at given parameters.

    Generates multiple trials with constant oscillatory signal +
    random additive Gaussian noise.

    data = simulate_oscillation(frequency,amplitude=5.0,phase=0,noise=1.0,n_trials=1000,
                                smp_rate=1000,time_range=1.0,burst_rate=0,burst_width=4,seed=None)

    ARGS
    frequency   Scalar. Frequency to simulation oscillation at (Hz)

    amplitude   Scalar. Amplitude of oscillation (a.u.). Default: 5.0

    phase       Scalar. Phase of oscillation (rad). Default: 0

    noise       Scalar. Amplitude of additive Gaussian noise (a.u). Default: 1.0

    n_trials    Int. Number of trials/observations to simulate. Default: 1000

    freq/amp/phase_sd Scalar. Inter-trial variation in frequency/amplitude/phase,
                given as Gaussian SD (same units as base parameters, which are used
                as Gaussian mean). Default: 0 (no inter-trial variation)

    smp_rate    Int. Sampling rate for simulate data (Hz). Default: 1000

    time_range  Scalar. Full time range to simulate oscillation over (s). Default: 1 s

    burst_rate  Scalar. Oscillatory burst rate (bursts/trial). Set=0 to simulate
                constant, non-bursty oscillation. Default: 0 (not bursty)

    burst_width Scalar. Half-width of oscillatory bursts (Gaussian SD, in cycles). Default: 4

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for fully random numbers.

    RETURNS
    data        (n_timepts,n_trials) ndarray. Simulated oscillation-in-noise data.
    """
    if seed is not None: set_random_seed(seed)

    def _randn(*args):
        """
        Generates unit normal random variables in a way that reproducibly matches output
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
    Reshapes data array from unwrapped form back to ~ original
    multi-dimensional form in special case where a new frequency axis was
    inserted before time axis (<axis>)

    data = _undo_standardize_array_newaxis(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where m = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    axis    Int. Axis of original data corresponding to distinct observations,
            which has become axis 1, but will be permuted back to original axis.
            Default: 0

    RETURNS
    data       (...,n_freqs,n_timepts,...) ndarray. Data array reshaped back to original shape
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
    """
    Rounds x up to the next power of 2 (smallest power of 2 greater than n)
    """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


def _infer_freq_scale(freqs):
    """ Determines if frequency sampling vector is linear, logarithmic, or uneven """
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

    ARGS
    event_times (n_events,) array-like. List of times (s) of event triggers to
                extract data around. Times are referenced to 1st data sample (t=0)

    window      (2,) array-like. [start,end] of window (in s) to extract
                around each trigger.

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
    """ Converts string specifier to callable pooling function """
    # If it's already a callable, return as-is
    if callable(func):      return func
    else:
        assert isinstance(func,str), "'func' must be a string or callable function"

        if func == 'mean':  return lambda x: np.mean(x, axis=0)
        elif func == 'sum': return lambda x: np.sum(x, axis=0)
        else:
            raise ValueError("Unsupported value '%s' for func. Set='mean'|'sum'" % func)

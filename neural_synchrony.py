# -*- coding: utf-8 -*-
"""
A module for analysis of neural oscillations and synchrony

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

burst_analysis      Computes oscillatory burst analysis of Lundqvist et al 2016

### Field-field synchrony ###
synchrony           Synchrony between pair of channels using given method

coherence           Time-frequency coherence between pair of channels
ztransform_coherence Z-transform coherence so ~ normally distributed

phase_locking_value Phase locking value (PLV) between pair of channels
pairwise_phase_consistency Pairwise phase consistency (PPC) btwn pair of channels

### Preprocessing ###
cut_trials          Cut LFPs/continuous data into trial segments
realign_data        Realigns LFPs/continuous data to new within-trial event

get_freq_sampling   Frequency sampling vector for a given FFT-based computation
setup_sliding_windows Generates set of sliding windows from given parameters

remove_dc           Removes constant DC component of signals
remove_evoked       Removes phase-locked evoked potentials from signals

### Postprocesssing ###
pool_freq_bands     Averages spectral data within set of frequency bands
pool_time_epochs    Averages spectral data within set of time epochs
one_sided_to_two_sided Converts 1-sided Fourier transform output to 2-sided equivalent
    
### Data simulation ###
simulate_oscillation Generates simulated oscillation-in-noise data

simulate_mvar        Simulates network activity with given connectivity
network_simulation   Canned network simulations


DEPENDENCIES
pyfftw              Python wrapper around FFTW, the speedy FFT library
spike_analysis      A module for basic analyses of neural spiking activity


Created on Thu Oct  4 15:28:15 2018

@author: sbrincat
"""

import os
import time
from math import floor,ceil,log2,pi,sqrt
from collections import OrderedDict
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal.windows import dpss
from scipy.signal import filtfilt,hilbert,zpk2tf,butter,ellip,cheby1,cheby2
from scipy.stats import norm,mode
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

try:
    from .spike_analysis import _spike_data_type, times_to_bool
# TEMP    
except ImportError:
    from spike_analysis import _spike_data_type, times_to_bool
    

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

    timepts     (n_timepts,...) ndarray. List of time points / time window centers
                for each time index in <spec>
    """
    method = method.lower()
    assert data_type in ['lfp','spike'], \
        ValueError("<data_type> must be 'lfp' or 'spike' ('%s' given)" % data_type)

    if method == 'wavelet':         spec_fun = wavelet_spectrogram
    elif method == 'multitaper':    spec_fun = multitaper_spectrogram
    elif method == 'bandfilter':    spec_fun = bandfilter_spectrogram
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    spec,freqs,timepts = spec_fun(data,smp_rate,axis=axis,data_type=data_type,spec_type=spec_type,
                                  removeDC=removeDC, **kwargs)

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

    timepts     (n_timepts,...) ndarray. List of time points / time window
                centers for each time index in <spec>
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

    timepts     (n_timepts,...) ndarray. List of time points / time window centers
                for each time index in <spec>
    """
    return spectrogram(data, smp_rate, axis=axis, method=method, spec_type='phase', **kwargs)


# =============================================================================
# Multitaper spectral analysis functions
# =============================================================================
def multitaper_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex', freq_range=None, 
                        removeDC=True, freq_width=4, n_tapers=None, keep_tapers=False,
                        tapers=None, pad=True, **kwargs):
    """
    Multitaper Fourier spectrum computation for continuous (eg LFP)
    or point process (eg spike) data

    spec,freqs = multitaper_spectrum(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',freq_range=None,
                                     removeDC=True,freq_width=4,n_tapers=None,keep_tapers=False,
                                     tapers=None,pad=True,**kwargs)

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

    SOURCE  Adapted from Chronux functions mtfftc.m, mtfftpb.m
    """
    if axis < 0: axis = data.ndim + axis
    
    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data, width=1/smp_rate, **kwargs)
        axis = data.ndim
    
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


def multitaper_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex', freq_range=None,
                           removeDC=True, time_width=0.5, freq_width=4, n_tapers=None, spacing=None,
                           tapers=None, keep_tapers=False, pad=True, **kwargs):
    """
    Computes multitaper time-frequency spectrogram for continuous (eg LFP)
    or point process (eg spike) data

    spec,freqs,timepts = multitaper_spectrogram(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',
                                                freq_range=None,removeDC=True,
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

    timepts     (n_timewins,...) ndarray. List of timepoints (center of each
                time window) in <spec>.

    REFERENCE   Mitra & Pesaran (1999) "Analysis of dynamic brain imaging data"
                Jarvis & Mitra (2001) Neural Computation

    SOURCE      Adapted from Chronux function mtfftc.m
    """
    if axis < 0: axis = data.ndim + axis
    
    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data,**kwargs)
        axis = data.ndim

    # If observation axis != 0, permute axis to make it so
    if axis != 0: data = np.moveaxis(data,axis,0)
    n_timepts = data.shape[0]

    window = time_width
    if spacing is None: spacing = window
    # Compute DPSS taper functions (if not precomputed)
    if tapers is None:
        tapers = compute_tapers(smp_rate,time_width=time_width,freq_width=freq_width,n_tapers=n_tapers)

    # Set up parameters for data time windows
    # Set window starts to range from time 0 to time n - window width (1e-12 for fp err)
    win_starts  = np.arange(0,n_timepts/smp_rate-window+1e-12,spacing)
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
    tapers (n_samples,n_tapers) ndarray. Computed dpss taper functions (n_samples = T*smp_rate)

    SOURCE  Adapted from Cronux function dpsschk.m
    """
    # Time-frequency bandwidth product 'TW' (s*Hz)
    TW  = time_width*freq_width
    
    # Up to 2TW-1 tapers are bounded; this is both the default and max value for n_tapers    
    n_tapers_max = floor(2*TW - 1)
    if n_tapers is None: n_tapers = n_tapers_max
    
    assert n_tapers <= n_tapers_max, \
        ValueError("For TW = %.1f, %d tapers are tightly bounded in" \
                    "frequency (n_tapers set = %d)" \
                    % (TW,n_tapers_max,n_tapers))

    # Convert time bandwidth from s to window length in number of samples
    n_samples = int(round(time_width*smp_rate))

    # Compute the tapers for given window length and time-freq product
    # Note: dpss() normalizes by sum of squares; x sqrt(smp_rate)
    #       converts this to integral of squares (see Chronux function dpsschk())
    # Note: You might imagine you'd want sym=False, but sym=True gives same values
    #       as Chronux dpsschk() function...
    return dpss(n_samples, TW, Kmax=n_tapers, sym=True, norm=2).T * sqrt(smp_rate)


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
    
    spec, freqs, _ = wavelet_spectrogram(data, smp_rate, axis=axis, data_type=data_type, spec_type=spec_type,
                                         freqs=freqs, removeDC=removeDC, wavelet=wavelet,
                                         wavenumber=wavenumber, pad=pad, buffer=buffer, **kwargs)
    
    # Take mean across time axis (which is now shifted +1 b/c of frequency axis)
    return spec.mean(axis=axis+1), freqs

    
def wavelet_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freqs=2**np.arange(1,7.5,0.25), removeDC=True,
                        wavelet='morlet', wavenumber=6, pad=False, buffer=0, downsmp=1, **kwargs):
    """
    Computes continuous time-frequency wavelet transform of data at given frequencies.


    spec,freqs,timepts = wavelet_spectrogram(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',
                                            freqs=2**np.arange(1,7.5,0.25),removeDC=True,
                                            wavelet='morlet',wavenumber=6,
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

    timepts     (n_timepts_out,...) ndarray. List of timepoints (indexes into original
                data time series) in <spec>.

    REFERENCE   Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"
    """
    if axis < 0: axis = data.ndim + axis
    
    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data, width=1/smp_rate, **kwargs)
        axis = data.ndim
            
    # Convert buffer from s -> samples
    if buffer != 0:  buffer  = int(ceil(buffer*smp_rate))

    # Reshape data array -> (n_timepts_in,n_dataseries) matrix
    data, data_shape = _reshape_data(data,axis)
    n_timepts_in = data.shape[0]

    timepts_out = np.arange(buffer,n_timepts_in-buffer,downsmp)

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

    # Convolve data with wavelets (multiply in Fourier domain) -> inverse FFT to get wavelet transform
    spec = ifft(data*wavelets_fft, n=n_fft,axis=1, **_FFTW_KWARGS_DEFAULT)[:,timepts_out,...]

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)
    
    spec = _unreshape_data_newaxis(spec,data_shape,axis=axis)

    return spec, freqs, timepts_out


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
        norm    = np.sqrt(scales*k[:,1])*(pi**(-0.25))*sqrt(n)
        # Wavelet exponent
        exponent= -0.5*(scales*k - k0)**2 * (k > 0)

        # Fourier transform of Wavelet function
        if do_fft:
            wavelets = norm*np.exp(exponent) * (k > 0)
        else:
            raise "non-FFT wavelet output not coded up yet (TODO)"

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
# Band-pass filtering analysis functions
# =============================================================================
def bandfilter_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freqs=((2,8),(10,32),(40,100)), removeDC=True,
                        filt='butter', params=None, buffer=0, **kwargs):
    """
    Computes band-filtered and Hilbert-transformed signal from data
    for given frequency band(s), then reduces it to 1D frequency spectra by averaging across time.

    Not really the best way to compute 1D frequency spectra, but included for completeness

    spec,freqs,timepts = bandfilter_spectrum(data,smp_rate,axis=0,data_type='lfp',spec_type='complex',
                                             freqs=((2,8),(10,32),(40,100)),filt='butter',
                                              params=None,buffer=0,**kwargs):

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

    spec,freqs,timepts = bandfilter_spectrogram(data,smp_rate,axis=0,data_type='lfp', spec_type='complex',
                                               freqs=((2,8),(10,32),(40,100)),
                                               filt='butter', order=4, params=None,
                                               buffer=0 ,downsmp=1, **kwargs):

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

    timepts     (n_timepts_out,...) ndarray. List of timepoints (indexes into original
                data time series) in <spec>.
    """
    if axis < 0: axis = data.ndim + axis
    
    # Convert spike timestamp data to boolean spike train format
    if (data_type == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data, width=1/smp_rate, **kwargs)
        axis = data.ndim
            
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
    data, data_shape = _reshape_data(data,axis)
    # Temporarily append singleton axis to vector-valued data to simplify code
    vector_data = data.ndim == 1
    if vector_data: data = data[:,np.newaxis]

    n_timepts_in,n_series = data.shape
    
    timepts_out     = np.arange(buffer,n_timepts_in-buffer,downsmp)
    n_timepts_out   = len(timepts_out)

    if removeDC: data = remove_dc(data,axis=0)
    
    dtype = float if spec_type == 'real' else complex
    spec = np.empty((n_freqs,n_timepts_out,n_series),dtype=dtype)

    # For each frequency band, band-filter raw signal and
    # compute complex analytic signal using Hilbert transform
    for i_freq,(b,a) in enumerate(zip(params['b'],params['a'])):
        bandfilt = filtfilt(b, a, data, axis=0, method='gust')
        # Note: skip Hilbert transform for real output
        spec[i_freq,:,:] = bandfilt[timepts_out,:] if spec_type == 'real' else \
                           hilbert(bandfilt[timepts_out,:],axis=0) 

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)
    
    if vector_data: spec = spec.squeeze(axis=-1)    
    spec = _unreshape_data_newaxis(spec,data_shape,axis=axis)

    return spec, freqs, timepts_out

bandfilter = bandfilter_spectrogram  # Alias function to bandfilter()


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

    # Setup empty lists to hold filter parameters
    if form == 'ba':    params = OrderedDict({'b':[None]*n_bands, 'a':[None]*n_bands})
    elif form == 'zpk': params = OrderedDict({'z':[None]*n_bands, 'p':[None]*n_bands, 'k':[None]*n_bands})
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


def burst_analysis(data, smp_rate, axis=0, trial_axis=-1, method='wavelet', 
                   freq_exp=None, bands=((20,35),(40,65),(55,90),(70,100)), 
                   window=None, timepts=None, threshold=2, min_cycles=3, **kwargs):
    """
    Computes oscillatory burst analysis of Lundqvist et al 2016.
        
    To compute burst rate, simply take mean across trial axis.
    
    Default argument values approximate analysis as implemented in Lundqvist 2016.    

    bursts,freqs,timepts = burst_analysis(data,smp_rate,axis=0,method='bandfilter',
                                          freq_bands=None,**kwargs):

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    trial_axis  Int. Axis of <data> corresponding to trials/observations. 
                Default: -1 (last axis of data)
                
    method      String. Underlying time-frequency spectral analysis method to use,
                which burst analysis is computed on.
                'wavelet' :     Continuous Morlet wavelet analysis [default]
                'multitaper' :  Multitaper spectral analysis
                'bandfilter' :  Band-pass filtering and Hilbert transform
                Note: In the original paper, multitaper was used, but all three 
                methods were claimed to produced similar results.
                
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
    
    threshold   Scalar. Threshold power level for detecting bursts, given in SDs above the mean.
                Default: 2 SDs
                
    min_cycles  Scalar. Minimal length of contiguous above-threshold period to be counted as a
                burst, given in number of oscillatory cycles at each frequency (or band center freq).
                Default: 3               

    **kwargs    Any other kwargs passed directly to power_spectrogram() function

    RETURNS
    bursts      (...,n_freqbands,n_timepts_out,...) ndarray of bool. Binary array labelling
                timepoints within detected bursts in each trial and frequency band.
                Same shape as input data, but with frequency axis prepended immediately
                before time <axis>.
                
    freqs       (n_freqbands,) ndarray. List of center frequencies of bands in <bursts>

    timepts     (n_timepts_out,...) ndarray. List of timepoints (indexes into original
                data time series) in <spec>.
                
    REFERENCE
    Lundqvist, ..., & Miller (2016) Neuron "Gamma and Beta Bursts Underlie Working Memory"
    Lundqvist, ..., & Miller (2018) Nature Comm "Gamma and beta bursts during working memory
                                                readout suggest roles in its volitional control"
    """
    # TODO  Add optional sliding trial window for mean,SD; Gaussian fits for definining burst f,t extent?
    # TODO  Option input of spectral data?
    method = method.lower()
    bands = np.asarray(bands)
    if axis < 0:        axis = data.ndim + axis
    if trial_axis < 0:  trial_axis = data.ndim + trial_axis
    
    if window is not None:
        assert len(window) == 2, \
            ValueError("Window for computing mean,SD should be given as (start,end) (len=2)")
        assert timepts is not None, \
            ValueError("To set a window for computing mean,SD need to input time sampling vector <timepts>")
    
    # Set default values to appoximate Lundqvist 2016 analysis, unless overridden by inputs
    if method == 'wavelet':
        # Sample frequency at 1 Hz intervals from min to max frequency in requested bands
        if 'freqs' not in kwargs: kwargs['freqs'] = np.arange(bands.min(),bands.max()+1,1)                
        
    # For bandfilter method, if frequency bands not set explicitly, set it with value for <bands>
    elif method == 'bandfilter':
        if 'freqs' not in kwargs: kwargs['freqs'] = bands
        
    # Compute time-frequency power from raw data
    data,freqs,tidxs = power_spectrogram(data, smp_rate, axis=axis, method=method, **kwargs)
    timepts = timepts[tidxs] if timepts is not None else tidxs 
    dt = np.mean(np.diff(tidxs)) * (1/smp_rate)
    
    # Update axes to reflect new frequency dim inserted before them
    if trial_axis > axis: trial_axis += 1
    axis += 1
    freq_axis = axis - 1
    
    # Normalize computed power by 1/f**exp to normalize out 1/f distribution of power
    if freq_exp is not None:
        data = data*freqs[:,np.newaxis,np.newaxis]**freq_exp
        
    # If requested, pool data within given frequency bands 
    # (skip for bandfilter spectral analysis, which already returns frequency bands)
    if (method != 'bandfilter') and (bands is not None):
        data = pool_freq_bands(data, bands, axis=freq_axis, freqs=freqs, func='mean')
        # Set sampled frequency vector = center frequency of each band
        freqs = bands.mean(axis=1)
            
    # Compute mean,SD of each frequency band across all trials and timepoints
    if window is None:
        mean = data.mean(axis=(axis,trial_axis), keepdims=True)
        sd   = data.std(axis=(axis,trial_axis), ddof=0, keepdims=True)
        
    # Compute mean,SD of each frequency band across all trials and timepoints within given time window
    else:        
        tbool = (timepts >= window[0]) & (timepts <= window[1])        
        mean = data.compress(tbool,axis=axis).mean(axis=(axis,trial_axis), keepdims=True)
        sd   = data.compress(tbool,axis=axis).std(axis=(axis,trial_axis), ddof=0, keepdims=True)
                               
    # Compute z-score of data and threshold -> boolean array of candidate burst times
    bursts = ((data - mean) / sd) > threshold
    
    n_trials = data.shape[trial_axis]
    tidxs = range(bursts.shape[axis])
    
    # TODO  Generalize to arbitrary dimensionality? -- transpose here?  OR just demand standard input???    
    for i_freq,freq in enumerate(freqs):
        # Convert minimum length in oscillatory cycles -> samples
        min_samples = ceil(min_cycles * (1/freq) / dt)
        
        for i_trial in range(n_trials):        
            # Extract time series for current freq,trial
            series = bursts[i_freq,:,i_trial]            
            if not series.any(): continue
            
            bursts[i_freq,:,i_trial] = _screen_bursts(series,tidxs,min_samples,start=None)
       
    return bursts, freqs, timepts


def _screen_bursts(data, t, min_samples, start=None):
    """ Subfunction to evaluate/detect bursts in boolean time series of candidate-burst times """
    # Find first candidate burst in trial (timepoints of all candidate burst times)
    if start is None:   on_times = np.nonzero(data)[0]
    # Find next candidate burst in trial
    else:               on_times = np.nonzero(data & (t > start))[0]
    
    # If no (more) bursts in time series, return data as is, we are done
    if len(on_times) == 0:  return data
    # Onset index of first/next candidate burst (if there is one)
    else:                   onset = on_times[0]
    
    # Find non-burst timepoints in remainder of time series
    off_times = np.nonzero(~data & (t > onset))[0]
    
    # Offset index of current burst = next off time - 1
    if len(off_times) != 0: offset = off_times[0] - 1
    # If no offset found, burst must extend to end of data
    else:                   offset = len(data)
    
    # Determine if length of current candidate burst meets minimum duration
    # If not, delete it from data (set all timepoints w/in it to False)
    burst_len = offset - onset + 1
    if burst_len < min_samples:  data[onset:(offset+1)] = False
    
    # TODO  trimming bursts to half-max point? (using Gaussian fits or raw power?)

    # If offset is less than minimum burst length from end of data, we are done, return data
    if (len(data) - offset) < min_samples:  return data
    # Otherwise, call function recursively, now starting search just after current burst offset
    else:                                   return _screen_bursts(data,t,min_samples,start=offset+1)
    
    
    
# =============================================================================
# Field-Field Synchrony functions
# =============================================================================
def synchrony(data1, data2, axis=0, method='PPC', spec_method='wavelet', **kwargs):
    """
    Computes pairwise synchrony between pair of channels of continuous raw or
    spectral (time-frequency) data, using given estimation method

    sync,freqs,timepts= synchrony(data1,data2,axis=0,method='PPC',
                                  spec_method='wavelet',**kwargs)
                                  
    Convenience wrapper function around specific synchrony estimation functions
    coherence, phase_locking_value, pairwise_phase_consistency

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have arbitrary shape, with
            analysis performed in mass-bivariate fashion independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    method  String. Synchrony estimation method. Options: 'PPC' [default] | 'PLV' | 'coherence'
            
    **kwargs    All other kwargs passed as-is to synchrony estimation function;=.
            See there for details.

    RETURNS
    sync    ndarray. Synchrony between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in coh (only for raw data)
    timepts (n_timepts,). List of timepoints in coh (only for raw data)

    """
    if method.lower() in ['ppc','pairwise_phase_consistency']:  sync_fun = pairwise_phase_consistency
    elif method.lower() in ['plv','phase_locking_value']:       sync_fun = phase_locking_value
    elif method.lower() in ['coh','coherence']:                 sync_fun = coherence
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)
    
    return sync_fun(data1, data2, axis=axis, method=spec_method, **kwargs)
    
        
def coherence(data1, data2, axis=0, return_phase=False, single_trial=None, ztransform=False,
              method='wavelet', data_type=None, smp_rate=None, time_axis=None,
              **kwargs):
    """
    Computes pairwise coherence between pair of channels of raw or
    spectral (time-frequency) data (LFP or spikes)

    coh,freqs,timepts[,dphi] = coherence(data1,data2,axis=0,return_phase=False,
                                         single_trial=None,ztransform=False,
                                         method='wavelet',data_type=None,smp_rate=None,
                                         time_axis=None,**kwargs)

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have arbitrary shape, with
            analysis performed in mass-bivariate fashion independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of coherence estimator to compute:
            None        standard across-trial estimator [default]
            'pseudo'    single-trial estimates using jackknife pseudovalues
            'richter'   single-trial estimates using actual jackknife estimates
                        as in Richter & Fries 2015

    ztransform Bool. If True, returns z-transformed coherence using Jarvis &
            Mitra (2001) method. If false [default], returns raw coherence.

    data_type Str. What kind of data are we given in data1,data2:
            'raw' or 'spectral'
            Default: assume 'raw' if data is real; 'spectral' if complex

    Following args are only used for spectral analysis for data_type == 'raw'

    method  String. Spectral method. 'wavelet' [default] | 'multitaper'

    smp_rate Scalar. Sampling rate of data (only needed for raw data)

    time_axis Scalar. Axis of data corresponding to time (only needed for raw data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    coh     ndarray. Magnitude of coherence between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in coh (only for raw data)
    timepts (n_timepts,). List of timepoints in coh (only for raw data)

    dphi   ndarray. Mean phase difference between data1 and data2 in radians.
           Positive values correspond to data1 leading data2.
           Negative values correspond to data1 lagging behind data2.
           Optional: Only returned if return_phase is True.

    REFERENCE
    Single-trial method:    Womelsdorf, Fries, Mitra, Desimone (2006) Science
    Single-trial method:    Richter, ..., Fries (2015) NeuroImage
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")
            
    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    if data_type is None: data_type = _infer_data_type(data1)
    
    # If raw data is input, compute spectral transform first
    # print(axis, time_axis, data1.shape, data1.mean(), data2.mean())
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        data1,freqs,timepts = spectrogram(data1, smp_rate, axis=time_axis, method=method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2, smp_rate, axis=time_axis, method=method,
                                          data_type='lfp', spec_type='complex', **kwargs)
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
    else:
        freqs = []
        timepts = []

    # For multitaper, compute means across trials, tapers; df = 2*n_trials*n_tapers
    if method == 'multitaper':
        reduce_axes = (axis,time_axis-1)
        df = 2*data1.shape[axis]*data1.shape[time_axis-1]
    # Otherwise, just compute means across trials; df = 2*n_trials (TODO is this true?)
    else:
        reduce_axes = axis
        df = 2*data1.shape[axis]

    # Standard across-trial coherence estimator
    if single_trial is None:
        if return_phase:
            coh,dphi = _spec_to_coh_with_phase(data1, data2, axis=reduce_axes)
        else:
            coh = _spec_to_coh(data1, data2, axis=reduce_axes)
        
        if ztransform: coh = ztransform_coherence(coh,df)

    # Single-trial coherence estimator using jackknife resampling method
    else:
        # If observation axis != 0, permute axis to make it so
        if axis != 0:
            data1 = np.moveaxis(data1,axis,0)
            data2 = np.moveaxis(data2,axis,0)
        n = data1.shape[0]

        # Compute cross spectrum and auto-spectra for each observation/trial
        S12,S1,S2 = _spec_to_csd(data1,data2)

        coh = np.zeros_like(data1,dtype=float)

        # Do jackknife resampling -- estimate statistic w/ each observation left out
        # (this is the 'richter' estimator)
        trials = np.arange(n)
        for trial in trials:
            idxs = trials != trial
            coh[trial,...] = _csd_to_coh(S12[idxs,...],S1[idxs,...],S2[idxs,...])

        if ztransform: coh = ztransform_coherence(coh,df/n)

        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            coh_full = _csd_to_coh(S12,S1,S2)
            if ztransform: coh_full = ztransform_coherence(coh_full,df)
            coh = jackknife_to_pseudoval(coh_full[np.newaxis,...],coh,n)

        # If observation axis wasn't 0, permute axis back to original position
        if axis != 0: coh = np.moveaxis(coh,0,axis)

    if return_phase:    return coh, freqs, timepts, dphi
    else:               return coh, freqs, timepts


def ztransform_coherence(coh, df, beta=23/20):
    """
    z-transforms coherence values to render them approximately normally distributed

    z = ztransform_coherence(coh,df,beta=23/20)

    ARGS
    coh     ndarray. Raw coherence values.

    df      Int. Degrees of freedom of coherence estimates. For multitaper
            estimates, this is usually df = 2*n_trials*n_tapers; for other
            estimates, this is usually df = 2*n_trials

    beta    Scalar. Mysterious number from Jarvis & Mitra to make output z-scores
            approximately normal. Default: 23/20 (see J & M)

    RETURNS
    z       ndarray of same shape as <coh>. z-transformed coherence

    REFERENCE
    Jarvis & Mitra (2001) Neural Computation
    Hipp, Engel, Siegel (2011) Neuron
    """
    return beta*(np.sqrt(-(df-2)*np.log(1-coh**2)) - beta)


def _spec_to_coh(data1, data2, axis=0):
    """ Compute coherence from a pair of spectra/spectrograms """
    # Compute cross spectrum and auto-spectra, average across trials/tapers
    # Note: .real deals with floating point error, converts complex dtypes to float        
    S12 = np.abs(np.mean(data1*data2.conj(), axis=axis)).real
    S1  = np.mean(data1*data1.conj(), axis=axis).real
    S2  = np.mean(data2*data2.conj(), axis=axis).real

    # Calculate coherence as cross-spectrum / product of spectra
    # Note: absolute value of cross-spectrum computed above
    return S12 / np.sqrt(S1*S2)
    
        
def _spec_to_coh_with_phase(data1, data2, axis=0):
    """ Compute coherence, relative phase angle from a pair of spectra/spectrograms """
    # Compute cross spectrum and auto-spectra, average across trials/tapers
    S12 = np.mean(data1*data2.conj(), axis=axis)
    S1  = np.mean(data1*data1.conj(), axis=axis).real
    S2  = np.mean(data2*data2.conj(), axis=axis).real

    # Calculate complex coherency as cross-spectrum / product of spectra
    coherency = S12 / np.sqrt(S1*S2)

    # Absolute value converts complex coherency -> coherence
    # Angle extracts mean coherence phase angle
    # Note: .real deals with floating point error, converts complex dtypes to float
    return np.abs(coherency).real, np.angle(coherency)        


def _spec_to_csd(data1, data2):
    """ Compute cross spectrum, auto-spectra from a pair of spectra/spectrograms """
    S12 = data1*data2.conj()
    S1  = data1*data1.conj()
    S2  = data2*data2.conj()
    
    return S12, S1, S2
      
        
def _csd_to_coh(S12, S1, S2, axis=0):
    """ Compute coherence from cross spectrum, auto-spectra """
    # Average cross and individual spectra across observations/trials
    S12 = np.abs(np.mean(S12, axis=axis)).real
    S1  = np.mean(S1, axis=axis).real
    S2  = np.mean(S2, axis=axis).real

    # Calculate coherence as cross-spectrum / product of spectra
    return S12 / np.sqrt(S1*S2)


def phase_locking_value(data1, data2, axis=0, return_phase=False,
                        single_trial=None, method='wavelet', data_type=None,
                        smp_rate=None, time_axis=None, **kwargs):
    """
    Computes phase locking value (PLV) between raw or spectral (time-frequency) LFP data

    PLV is the mean resultant length (magnitude of the vector mean) of phase
    differences dphi btwn phases of data1 and data2:
        dphi = phase(data1) - phase(data2)
        plv  = abs( trialMean(exp(i*dphi)) )

    plv,freqs,timepts[,dphi] = phase_locking_value(data1,data2,axis=0,return_phase=False,
                                                 single_trial=None,
                                                 method='wavelet',data_type=None,
                                                 smp_rate=None,time_axis=None,**kwargs)

    ARGS
    data1,2 (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have
            Can have arbitrary shape, with analysis performed independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of coherence estimator to compute:
            None        standard across-trial estimator [default]
            'pseudo'    single-trial estimates using jackknife pseudovalues
            'richter'   single-trial estimates using actual jackknife estimates
                        as in Richter & Fries 2015

    Following args are only used for spectral analysis for data_type == 'raw'

    method  String. Spectral method. 'wavelet' [default] | 'multitaper'

    data_type Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
            Default: assume 'raw' if data is real; 'spectral' if complex

    smp_rate Scalar. Sampling rate of data (only needed for raw data)

    time_axis Scalar. Axis of data corresponding to time (only needed for raw data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    plv     ndarray. Phase locking value between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in plv (only for raw data)
    timepts (n_timepts,). List of timepoints in plv (only for raw data)

    dphi   ndarray. Mean phase difference between data1 and data2 in radians.
           Positive values correspond to data1 leading data2.
           Negative values correspond to data1 lagging behind data2.
           Optional: Only returned if return_phase is True.

    REFERENCES
    Lachaux et al. (1999) Human Brain Mapping
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp', spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp', spec_type='complex', **kwargs)
        
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
    else:
        freqs = []
        timepts = []

    # For multitaper, compute means across trials, tapers
    if method == 'multitaper':  reduce_axes = (axis,time_axis-1)
    # Otherwise, just compute means across trials
    else:                       reduce_axes = axis
        
    # Standard across-trial PLV estimator
    if single_trial is None:
        if return_phase:
            plv,dphi = _spec_to_plv_with_phase(data1,data2,axis=reduce_axes)
            return  plv, freqs, timepts, dphi

        else:
            plv = _spec_to_plv(data1,data2,axis=reduce_axes)
            return  plv, freqs, timepts

    # Single-trial PLV estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_plv(data1,data2,axis=0)
        # Jackknife resampling of PLV statistic (this is the 'richter' estimator)
        plv = two_sample_jackknife(jackfunc,data1,data2,axis=reduce_axes)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            plv_full = _spec_to_plv(data1,data2,axis=reduce_axes,keepdims=True)
            plv = jackknife_to_pseudoval(plv_full,plv,n_obs)

        return  plv, freqs, timepts


def _spec_to_plv(data1, data2, axis=0, keepdims=False):
    """ Compute PLV from a pair of spectra/spectrograms """
    # Cross-spectrum-based method adapted from FieldTrip ft_conectivity_ppc()
    csd = data1*data2.conj()    # Compute cross-spectrum
    csd = csd / np.abs(csd)     # Normalize cross-spectrum
    # Compute vector mean across trial/observations -> absolute value
    return np.abs(np.mean(csd,axis=axis,keepdims=keepdims))

    ## Alt version using circular mean -- ~ 3x slower in timetest
    # # Compute phase difference btwn data1 and data2
    # dphi    = circular_subtract_complex(data1,data2)
    # # Compute absolute value of circular mean of phase diffs across trials
    # return np.abs( np.mean( np.exp(1j*dphi), axis=axis,keepdims=keepdims) )


def _spec_to_plv_with_phase(data1, data2, axis=0, keepdims=False):
    """ Compute PLV, relative phase from a pair of spectra/spectrograms """
    # Cross-spectrum-based method adapted from FieldTrip ft_conectivity_ppc()
    csd = data1*data2.conj()    # Compute cross-spectrum
    csd = csd / np.abs(csd)     # Normalize cross-spectrum
    # Compute vector mean across trial/observations
    vector_mean =  np.mean(csd,axis=axis,keepdims=keepdims)
    # Compute mean across trial/observations -> absolute value
    return np.abs(vector_mean), np.angle(vector_mean)


def pairwise_phase_consistency(data1, data2, axis=0, return_phase=False,
                               single_trial=None, method='wavelet',
                               data_type=None, smp_rate=None, time_axis=None,
                               **kwargs):
    """
    Computes pairwise phase consistency (PPC) between raw or spectral
    (time-frequency) LFP data, which is unbiased by n (unlike PLV and coherence)

    PPC is an unbiased estimator of PLV^2, and can be expressed (and computed
    efficiently) in terms of PLV and n:
        PPC = (n*PLV^2 - 1) / (n-1)

    ppc,freqs,timepts[,dphi] = pairwise_phase_consistency(data1,data2,axis=0,
                                                        return_phase=False,single_trial=None,
                                                        method='wavelet',data_type=None,
                                                        smp_rate=None,time_axis=None,**kwargs)

    ARGS
    data1,data2   (...,n_obs,...) ndarrays. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have
            Can have arbitrary shape, with analysis performed independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns additional output with mean phase difference

    single_trial String or None. What type of estimator to compute:
            None        standard across-trial estimator [default]
            'pseudo'    single-trial estimates using jackknife pseudovalues
            'richter'   single-trial estimates using actual jackknife estimates
                        as in Richter & Fries 2015

    Following args are only used for spectral analysis for data_type == 'raw'

    method  String. Spectral method. 'wavelet' [default] | 'multitaper'

    data_type Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
            Default: assume 'raw' if data is real; 'spectral' if complex

    smp_rate Scalar. Sampling rate of data (only needed for raw data)

    time_axis Scalar. Axis of data corresponding to time (only needed for raw data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    ppc     Pairwise phase consistency between data1 and data2.
            If data is spectral, this has shape as data, but with <axis> removed.
            If data is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in ppc (only for raw data)
    timepts (n_timepts,). List of timepoints in ppc (only for raw data)

    dphi   ndarray. Mean phase difference between data1 and data2 in radians.
           Positive values correspond to data1 leading data2.
           Negative values correspond to data1 lagging behind data2.
           Optional: Only returned if return_phase is True.

    REFERENCES
    Original concept:   Vinck et al. (2010) NeuroImage
    Relation to PLV:    Kornblith, Buschman, Miller (2015) Cerebral Cortex
    """
    assert not((single_trial is not None) and return_phase), \
        ValueError("Cannot do both single_trial AND return_phase together")
    assert (single_trial is None) or (single_trial in ['pseudo','richter']), \
        ValueError("Parameter <single_trial> must be = None, 'pseudo', or 'richter'")

    if axis < 0: axis = data1.ndim + axis
    if (time_axis is not None) and (time_axis < 0): time_axis = data1.ndim + time_axis

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        if method == 'multitaper': kwargs.update(keep_tapers=True)
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp',spec_type='complex', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,data_type='lfp',spec_type='complex', **kwargs)
        
        # Account for new frequency (and/or taper) axis
        n_new_axes = 2 if method == 'multitaper' else 1
        if axis >= time_axis: axis += n_new_axes
        time_axis += n_new_axes
    else:
        freqs = []
        timepts = []

    # For multitaper, compute means across trials, tapers
    if method == 'multitaper':  reduce_axes = (axis,time_axis-1)
    # Otherwise, just compute means across trials
    else:                       reduce_axes = axis
    
    # Standard across-trial PPC estimator
    if single_trial is None:
        if return_phase:
            ppc,dphi = _spec_to_ppc_with_phase(data1,data2,axis=reduce_axes)
            return ppc, freqs, timepts, dphi

        else:
            ppc = _spec_to_ppc(data1,data2,axis=reduce_axes)
            return ppc, freqs, timepts

    # Single-trial PPC estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_ppc(data1,data2,axis=0)
        # Jackknife resampling of PPC statistic (this is the 'richter' estimator)
        ppc = two_sample_jackknife(jackfunc,data1,data2,axis=reduce_axes)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            ppc_full = _spec_to_ppc(data1,data2,axis=reduce_axes,keepdims=True)
            ppc = jackknife_to_pseudoval(ppc_full,ppc,n_obs)

        return ppc, freqs, timepts


def _spec_to_ppc(data1, data2, axis=0, keepdims=False):
    """ Compute PPC from a pair of spectra/spectrograms """
    if np.isscalar(axis):   n = data1.shape[axis]
    else:                   n = np.prod([data1.shape[ax] for ax in axis])
    return plv_to_ppc(_spec_to_plv(data1,data2,axis=axis,keepdims=keepdims), n)


def _spec_to_ppc_with_phase(data1, data2, axis=0, keepdims=False):
    """ Compute PPC and mean relative phase from a pair of spectra/spectrograms """
    if np.isscalar(axis):   n = data1.shape[axis]
    else:                   n = np.prod([data1.shape[ax] for ax in axis])    
    plv,dphi = _spec_to_plv_with_phase(data1,data2,axis=axis,keepdims=keepdims)
    return plv_to_ppc(plv,n), dphi


def plv_to_ppc(plv, n):
    """ Converts PLV to PPC as PPC = (n*PLV^2 - 1)/(n-1) """
    return (n*plv**2 - 1) / (n - 1)


# =============================================================================
# Plotting functions
# =============================================================================
def plot_spectrum(freqs, data, ylim=None, color=None, **kwargs):
    """
    Plots frequency spectrum as a line plot.

    plot_spectrum(freqs, data, ylim=None, color=None, **kwargs)

    ARGS    
    freqs   (n_freqs,) array-like. Frequency sampling vector for data (Hz).
            May be linearly or logarithmically sampled; we deal appropriately.
                        
    data    (n_freqs,) ndarray. Frequency spectrum data.

    ylim    (2,) array-like. y-axis limits: (min,max)
            Default: min/max over all data +/- 5%  (data.min(),data.max())

    color   (3,) array-like | Color spec. Color to plot in. 
            Default: <default plot color>

    **kwargs All other keyword args passed directly to plt.plot()
    
    ACTION 
    Plots spectrum data into current axes using plt.plot()
    
    RETURNS
    lines   List of Line2D objects. Output of plt.plot()
    """    
    freqs   = np.asarray(freqs)
    if ylim is None: 
        ylim = (data.min(), data.max())
        ylim = (ylim[0]-0.05*np.diff(ylim), ylim[1]+0.05*np.diff(ylim))
    
    freqs,fticks,fticklabels = frequency_plot_settings(freqs)

    df      = np.diff(freqs).mean()
    flim    = [freqs[0]-df/2, freqs[-1]+df/2]
    
    if color is None: color = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]

    lines = plt.plot(freqs, data, '-', color=color, **kwargs)

    plt.xlim(flim)
    plt.ylim(ylim)
    plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
    plt.xticks(fticks,fticklabels)
    
    return lines


def plot_spectrogram(timepts, freqs, data, clim=None, cmap='viridis', **kwargs):
    """
    Plots time-frequency spectrogram as a pseudocolor plot.

    plot_spectrogram(timepts, freqs, data, clim=None, cmap='viridis', **kwargs)

    ARGS
    timepts (n_timepts,) array-like. Time sampling vector for data
    
    freqs   (n_freqs,) array-like. Frequency sampling vector for data (Hz).
            May be linearly or logarithmically sampled; we deal appropriately.
                        
    data    (n_freqs,n_timepts) ndarray. Time-frequency (spectrogam) data

    clim    (2,) array-like. Color axis limits: (min,max)
            Default: min/max over all data (data.min(),data.max())

    cmap    String. Colormap to plot in. Default: 'viridis'

    **kwargs All other keyword args passed directly to plt.imshow()
    
    ACTION 
    Plots spectrogram data into current axes using plt.imshow()
    
    RETURNS
    img    AxesImage object. Output of plt.imshow()
    """    
    timepts = np.asarray(timepts)
    freqs   = np.asarray(freqs)
    if clim is None: clim = (data.min(), data.max())
    
    freqs,fticks,fticklabels = frequency_plot_settings(freqs)

    df      = np.diff(freqs).mean()
    flim    = [freqs[0]-df/2, freqs[-1]+df/2]    

    dt      = np.diff(timepts).mean()
    tlim    = [timepts[0]-dt/2, timepts[-1]+dt/2]

    img = plt.imshow(data, extent=[*tlim,*flim], vmin=clim[0], vmax=clim[1],
                     aspect='auto', origin='lower', cmap=cmap, **kwargs)

    plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
    plt.yticks(fticks,fticklabels)
    
    return img
    
    
def frequency_plot_settings(freqs):
    """ Returns settings for plotting a frequency axis: plot freqs, ticks, tick labels """
    freq_scale = _infer_freq_scale(freqs)
    
    # For log-sampled freqs, plot in log(freq) but label with actual freqs
    if freq_scale == 'log':
        freqs           = np.log2(freqs)
        fmin            = ceil(freqs[0])
        fmax            = floor(freqs[-1])
        freq_ticks      = np.arange(fmin,fmax+1)
        freq_tick_labels= 2**np.arange(fmin,fmax+1)
        
    # For linear-sampled freqs, just plot in actual freqs    
    else:
        fmin            = ceil(freqs[0]/10.0)*10.0
        fmax            = floor(freqs[-1]/10.0)*10.0                
        freq_ticks      = np.arange(fmin,fmax+1,10).astype(int)
        freq_tick_labels= freq_ticks           

    return freqs,freq_ticks,freq_tick_labels


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
    assert trial_idxs.min() >= 0, ValueError("trial_lims are attempting to index before start of data")
    assert trial_idxs.max() < data.shape[axis], ValueError("trial_lims are attempting to index beyond end of data")
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
        cut_data[...,trial] = _index_axis(data, axis, slice(lim[0],lim[1]+1))
        
        # Note: The following is several orders of magnitude slower (mainly bc need to form explicit index):
        # cut_data[...,trial] = data.take(np.arange(lim[0],lim[1]+1), axis=axis)
                
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
    if time_axis != 0:              data = np.moveaxis(data,0,time_axis)
    if trial_axis != data.ndim-1:   data = np.moveaxis(data,-1,trial_axis)
    
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
    for iTrial,t in enumerate(trial_range_smps):
        # Note: '+1' below makes the selection inclusive of the right endpoint in each trial
        realigned[...,iTrial] = data[t[0]:t[1]+1,...,iTrial]    
    
    # Move array axes back to original locations
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


def setup_sliding_windows(width, lims, step=None, reference=None,
                          force_int=False, exclude_end=None):
    """
    Generates set of sliding windows using given parameters

    windows = setup_sliding_windows(width,lims,step=None,
                                  reference=None,force_int=False,exclude_end=None)

    ARGS
    width       Scalar. Full width of each window. Required arg.

    lims        (2,) array-like. [start end] of full range of domain you want
                windows to sample. Required.

    step        Scalar. Spacing between start of adjacent windows
                Default: step = width (ie, perfectly non-overlapping windows)

    reference   Bool. Optionally sets a reference value at which one window
                starts and the rest of windows will be determined from there.
                eg, set = 0 to have a window start at x=0, or
                    set = -width/2 to have a window centered at x=0
                Default: None = just start at lim[0]

    force_int   Bool. If True, rounds window starts,ends to integer values.
                Default: False (don't round)

    exclude_end Bool. If True, excludes the endpoint of each (integer-valued)
                sliding win from the definition of that win, to prevent double-sampling
                (eg, the range for a 100 ms window is [1 99], not [1 100])
                Default: True if force_int==True, otherwise default=False

    OUTPUT
    windows     (n_wins,2) ndarray. Sequence of sliding window [start end]
    """
    # Default: step is same as window width (ie windows perfectly disjoint)
    if step is None: step = width
    # Default: Excluding win endpoint is default for integer-valued win's,
    #  but not for continuous wins
    if exclude_end is None:  exclude_end = True if force_int else False

    # Standard sliding window generation
    if reference is None:
        if exclude_end: win_starts = _iarange(lims[0], lims[-1]-width+1, step)
        else:           win_starts = _iarange(lims[0], lims[-1]-width, step)

    # Origin-anchored sliding window generation
    #  One window set to start at given 'reference', position of rest of windows
    #  is set around that window
    else:
        if exclude_end:
            # Series of windows going backwards from ref point (flipped to proper order),
            # followed by Series of windows going forwards from ref point
            win_starts = np.concatenate(np.flip(_iarange(reference, lims[0], -step)),
                                        _iarange(reference+step, lims[-1]-width+1, step))

        else:
            win_starts = np.concatenate(np.flip(_iarange(reference, lims[0], -step)),
                                        _iarange(reference+step, lims[-1]-width, step))

    # Set end of each window
    if exclude_end: win_ends = win_starts + width - 1
    else:           win_ends = win_starts + width

    # Round window starts,ends to nearest integer
    if force_int:
        win_starts = np.round(win_starts)
        win_ends   = np.round(win_ends)

    return np.stack((win_starts,win_ends),axis=1)


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

        data,data_shape = _reshape_data(data,axis=axis)

        groups = np.unique(design)
        for group in groups:
            idxs = design == group
            data[idxs,...] -= np.mean(data[idxs,...],axis=0,keepdims=True)

        data = _unreshape_data(data,data_shape,axis=axis)

    # Regress data on given design matrix and return residuals
    elif method.lower() == 'regress':
        assert design.ndim in [1,2], \
            "Design matrix <design> must be matrix-like (2d) or vector-like (1d)"

        data,data_shape = _reshape_data(data,axis=axis)

        model = LinearRegression()
        data -= model.fit(design,data).predict(design)

        data = _unreshape_data(data,data_shape,axis=axis)

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
                'magnitude' Magnitude (square root of power) of complex data
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

    freqs   (n_freqs,) array-like. Frequency sampling in <data>.
            Only needed if <data> is not an xarray DataArray

    func    String | calable. Function to apply to each freqBand.
            Default: mean

    RETURNS
    data    (...,n_freqbands,...) ndarray | xarray DataArray.
            Data with values averaged within each of given frequency bands
    """
    # TODO  Deal with more complicated band edge situations (currently assumed non-overlapping)

    # Convert list of frequency band ranges to {'name':freq_range} dict
    if not isinstance(bands,dict):
        bands = {'band_'+str(i_band):frange for i_band,frange in enumerate(bands)}

    # Convert frequency bands into 1-d list of bin edges
    bins = []
    for value in bands.values(): bins.extend(value)

    # xarray: Pool values in bands using DataArray groupby_bins() method
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

        for i_band,(_,frange) in enumerate(bands.items()):
            fbool = (freqs >= frange[0]) & (freqs <= frange[1])
            band_data[i_band,...] = data[fbool,...].mean(axis=0)

        # Permute back to original data dimension order
        if axis != 0: band_data = band_data.transpose(*dims)

    # ndarray: loop thru freq bands, pooling values in each
    else:
        assert axis is not None, \
        ValueError("For ndarray data, must give frequency axis in <axis>")
        assert freqs is not None, \
        ValueError("For ndarray data, must give frequency sampling vector in <freqs>")

        if axis != 0: data = data.swapaxes(0,axis)

        data_shape= (len(bands), *data.shape[1:])
        band_data = np.zeros(data_shape,dtype=data.dtype)

        for i_band,(_,frange) in enumerate(bands.items()):
            fbool = (freqs >= frange[0]) & (freqs <= frange[1])
            if func == 'mean':
                band_data[i_band,...] = data[fbool,...].mean(axis=0)
            else:
                band_data[i_band,...] = func(data[fbool,...])

        if axis != 0: band_data = band_data.swapaxes(axis,0)

    return band_data


def pool_time_epochs(data, epochs, axis=None, timepts=None):
    """
    Pools (averages) spectral data within each of a given set of time epochs

    data = pool_time_epochs(data,epochs,axis=None,timepts=None)

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

    RETURNS
    data    (...,nTimeEpochs,...) ndarray | xarray DataArray.
            Data with values averaged within each of given time epochs
    """
    # todo  Deal with more complicated band edge situations (currently assumed non-overlapping)

    # Convert list of time epoch ranges to {'name':time_range} dict
    if not isinstance(epochs,dict):
        epochs = {'epochs_'+str(i_epoch):trange for i_epoch,trange in enumerate(epochs)}

    # xarray: Pool values in epochs using DataArray groupby_bins() method
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

        for i_epoch,(_,trange) in enumerate(epochs.items()):
            tbool = (timepts >= trange[0]) & (timepts <= trange[1])
            epoch_data[i_epoch,...] = data[tbool,...].mean(axis=0)

        # Permute back to original data dimension order
        if axis != 0: epoch_data = epoch_data.transpose(*dims)

    # ndarray: loop thru freq bands, mean-pooling values in each
    else:
        assert axis is not None, \
        ValueError("For ndarray data, must give time axis in <axis>")
        assert timepts is not None, \
        ValueError("For ndarray data, must give time sampling vector in <timepts>")

        if axis != 0: data = data.swapaxes(0,axis)

        data_shape= (len(epochs), *data.shape[1:])
        epoch_data = np.zeros(data_shape,dtype=data.dtype)

        for i_epoch,(_,trange) in enumerate(epochs.items()):
            tbool = (timepts >= trange[0]) & (timepts <= trange[1])
            epoch_data[i_epoch,...] = data[tbool,...].mean(axis=0)

        if axis != 0: epoch_data = epoch_data.swapaxes(axis,0)

    return epoch_data


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
        f0 = _interp1(freqs,data,0,axis=freq_axis,kind='cubic',fill_value='extrapolate')
        f0 = np.expand_dims(f0,freq_axis)
        data = np.concatenate((f0,data),axis=freq_axis)
        freqs = np.concatenate(([0],freqs))

    # Convert values at Nyquist freq to complex conjugate at negative frequency
    slices = _axis_slices(freq_axis,-1,data.ndim)
    data[slices] = data[slices].conj()
    freqs[-1] *= -1

    # Replicate values for all freqs (s.t. 0 < f < nyquist)
    # as complex conjugates at negative frequencies
    idxs    = slice(-2,1,-1)
    slices  = _axis_slices(freq_axis,idxs,data.ndim)
    data    = np.concatenate((data, data[slices].conj()), axis=freq_axis)
    freqs   = np.concatenate((freqs, -freqs[idxs]))

    return data,freqs


# =============================================================================
# Utility functions for circular and complex data (phase) analsysis
# =============================================================================
def amp_phase_to_complex(amp,theta):
    """ Converts amplitude and phase angle to complex variable """
    return amp * np.exp(1j*theta)


def complex_to_amp_phase(c):
    """ Converts complex variable to amplitude (magnitude) and phase angle """
    return magnitude(c), phase(c)
    

# =============================================================================
# Utility functions for generating single-trial jackknife pseudovalues
# =============================================================================
def two_sample_jackknife(func, data1, data2, *args, axis=0, **kwargs):
    """
    Jackknife resampling of arbitrary statistic
    computed by <func> on <data1> and <data2>

    ARGS
    func    Callable. Takes data1,data2 (and optionally any *args and **kwargs) as input,
            returns statistic to jackknife

    data1,2 ndarrays of same arbitrary shape. Data to compute jackknife resamples of

    axis    Int. Observation (trial) axis of data1 and data2. Default: 0

    *args,**kwargs  Any additional arguments passed directly to <func>

    RETURNS
    stat    ndarray of same size as data1,data2. Jackknife resamples of statistic.
    """

    # If observation axis != 0, permute axis to make it so
    if axis != 0:
        data1 = np.moveaxis(data1,axis,0)
        data2 = np.moveaxis(data2,axis,0)

    n = data1.shape[0]

    stat = np.zeros_like(data1)

    # Do jackknife resampling -- estimate statistic w/ each observation left out
    trials = np.arange(n)
    for trial in trials:
        idxs = trials != trial
        stat[trial,...] = func(data1[idxs,...],data2[idxs,...],*args,**kwargs)

    # If observation axis wasn't 0, permute axis back to original position
    if axis != 0: stat = np.moveaxis(stat,0,axis)

    return stat


def jackknife_to_pseudoval(x, xjack, n):
    """
    Calculate single-trial pseudovalues from leave-one-out jackknife estimates

    pseudo = jackknife_to_pseudoval(x,xjack,axis=0)

    ARGS
    x       ndarray of arbitrary shape (observations/trials on <axis>).
            Statistic (eg coherence) computed on full observed data

    xjack   ndarray of arbitrary shape (must be same shape as x).
            Statistic (eg coherence) computed on jackknife resampled data

    n       Int. Number of observations (trials) used to compute x

    RETURNS
    pseudo  ndarray of same shape as x. Single-trial jackknifed pseudovalues.
    """
    return n*x - (n-1)*xjack


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
    if seed is not None: np.random.seed(seed)
    
    def _randn(*args):
        """ 
        Generates unit normal random variables in a way that reproducibly matches output of Matlab (with same seed != 0)
        (np.random.randn() does not work here for unknown reasons)
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


def simulate_multichannel_oscillation(n_chnls, *args, **kwargs):
    """
    Generates synthetic multichannel data with oscillations at given parameters.
    
    For each channel, generates multiple trials with constant oscillatory signal +  
    random additive Gaussian noise. Parameters can be shared across channels or set
    independently for each channel
    
    data = simulate_multichannel_oscillation(n_chnls, *args, **kwargs)
    
    ARGS
    n_chnls     Int. Number of channels to simulate
    
    *args       Rest of place and keyword arguments are passed to simulate_oscillation()
    **kwargs    for each channel. Each argument can be given in one of two forms:
                (1) A single value equivalent to the same argument to simulate_oscillation().
                    That same value will be used for all n_chnls channels.  
                (2) (n_chnls,) list can be given with different values for each channel,
                    which will be iterated through.
                See simulate_oscillation() for details on arguments.
                
                Exceptions: If a value is set for <seed>, it will be used to set the 
                random number generator only ONCE at the start of this function, so that
                the generation of all channel signals follow a reproducible random sequence.
                
                Simulated data must have same shape for all channels, so all channels must 
                have the same value set for time_range, smp_rate, and n_trials.                
    
    RETURNS
    data        (n_timepts,n_trials,n_chnls) ndarray. Simulated multichannel data.
    """
    # Set a single random seed, so simulation of all multichannel data follows a reproducible sequence
    seed = kwargs.pop('seed',None)    
    if seed is not None:
        if not np.isscalar(seed) and (len(seed) > 1):
            seed = seed[0]
            print("Using only first value given for <seed> (%d). Multiple values not permitted" % seed)
        np.random.seed(seed)
    
    # Ensure all channels have same values for these parameters that determine data size
    for param in ['time_range','smp_rate','n_trials']:
        assert (param not in kwargs) or np.isscalar(kwargs[param]) or np.allclose(np.diff(kwargs[param]), 0), \
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

                
def simulate_mvar(coeffs, cov=None, n_timepts=100, n_trials=1, burnin=100, seed=None):
    """
    Simulates activity in a network with given connectivity coefficients and
    noise covariance, over the given number of time points, channels, and trials.
    Useful for testing code for measuring synchrony or causality.

    data = simulate_mvar(coeffs,cov=None,n_timepts=100,n_trials=1,burnin=100,seed=None)

    ARGS
    coeffs      ([n_timepts,]n_lags,n_channels,n_channels) ndarray.
                Network connectivity coefficients between all channels/nodes
                at one or more lags. Optionally can have time-varying
                connectivity, with time given in axis 0.

    cov         (n_channels,n_channels) ndarray. Between-channel noise covariance matrix.
                Default: np.eye(n_channels) (no covariance)

    n_timepts   Scalar. Number of timepoints to simulate.
    n_trials    Scalar. Number of trials to simulate.
    burnin      Scalar. Number of additional timepoints to include at start,
                but remove from output, to account for burn-in of network dynamics.

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for fully random numbers.
                
    RETURNS
    data        (n_timepts,n_trials,n_channels) ndarray. Simulated data

    ACKNOWLEDGMENTS Adapted slightly from spectral_connectivity:simulate_MVAR()
    """
    if seed is not None: np.random.seed(seed)

    # Does network have stationary or time-varying connectivity matrices?
    is_stationary = coeffs.ndim < 4
    if is_stationary:   n_lags,n_channels,_ = coeffs.shape
    else:               n_coeff_times,n_lags,n_channels,_ = coeffs.shape

    if cov is None:  cov = np.eye(n_channels)

    assert is_stationary or (n_coeff_times == n_timepts), \
    "coeffs matrices must be 3-dimensional (stationary) or have size(4) = n_timepts (time-varying)"

    # For time-varying connectivity, replicate starting connectivity for full burn-in period
    if not is_stationary and (burnin != 0):
        coeffs = np.concatenate((coeffs[0,:,:,:][np.newaxis,:,:,:]*np.ones((burnin,1,1,1)),
                                 coeffs),axis=0)

    # Initialize starting state + noise at each time step
    #  ~ multivariate normal(0,cov)
    data = np.random.multivariate_normal(np.zeros((n_channels,)),
                                         cov,
                                         size=(burnin+n_timepts,n_trials))

    # Step thru each timepoint in simulation
    for time in np.arange(n_lags,burnin+n_timepts):
        # Step thru each lag of delayed network influence
        for lag in np.arange(n_lags):
            # Propagate network activity with given starting state and coeffs
            if is_stationary:
                data[time,...] += np.matmul(coeffs[np.newaxis,np.newaxis,lag,...],
                                            data[time-(lag+1),...,np.newaxis]).squeeze()
            else:
                data[time,...] += np.matmul(coeffs[np.newaxis,time,lag,...],
                                            data[time-(lag+1),...,np.newaxis]).squeeze()

    # Remove timepoints at start to account for network burn-in
    return data[burnin:, ...]


def network_simulation(simulation='DhamalaFig3'):
    """
    Generates one of several canned simulations of network connectivity

    data,smp_rate = network_simulation(simulation='DhamalaFig3')

    ARGS
    simulation  String. Name of canned simulation to generate. Default: 'DhamalaFig3'
                'DhamalaFig3' : Sim from Dhamala...Ding 2008. Two channels with
                time-dependent cauasality -- x1->x2 switches to x2->x1

    RETURNS
    data        (n_timepts,n_trials,n_channels) ndarray. Simulated data
    
    smp_rate    Int. Sampling rate of simulated data (Hz)

    REFERENCES
    Dhamala, M., Rangarajan, G., and Ding, M. (2008). Analyzing information flow in
    brain networks with nonparametric Granger causality. NeuroImage 41, 354-362.

    ACKNOWLEDGMENTS
    Adapted slightly from scripts provided with Python spectral_connectivity module
    """
    n_trials = 1000
    burnin  = 1000

    # Simulation from Dhamala...Ding NeuroImage 2008, Fig. 3
    # Switches from strong influence in 1 direction to opposite direction
    if simulation == 'DhamalaFig3':
        smp_rate = 200
        n_timepts,n_lags,n_channels = 900,2,2

        cov = np.eye(n_channels) * [0.25,0.25]

        # Stationary within-node connectivity (same for all timepoints)
        coeffs = np.zeros((n_timepts,n_lags,n_channels,n_channels))
        coeffs[:,0,0,0] = 0.53
        coeffs[:,1,0,0] = -0.80
        coeffs[:,0,1,1] = 0.53
        coeffs[:,1,1,1] = -0.80

        # Time-varying between-node connectivity
        t = np.arange(n_timepts)
        offset = (n_timepts-1)/2
        # Causal influence from x1 -> x2 : Goes smoothly from 0.5 to 0
        coeffs[:,0,0,1] = 0.25*np.tanh(-0.05*(t-offset)) + 0.25
        # Causal influence from x1 -> x2 : Goes smoothly from 0 to 0.5
        coeffs[:,0,1,0] = 0.25*np.tanh(0.05*(t-offset)) + 0.25

        data  = simulate_mvar(coeffs,cov=cov,
                              n_timepts=n_timepts,n_trials=n_trials,burnin=burnin)

    else:
        raise ValueError("%s simulation is not (yet) supported" % simulation)
    
    return data,smp_rate

            
# =============================================================================
# Data reshaping helper functions
# =============================================================================
def _index_axis(data, axis, idxs):
    """ 
    Utility to dynamically index into a arbitrary axis of an ndarray 
    
    data = _index_axis(data, axis, idxs)
    
    ARGS
    data    ndarray. Array of arbitrary shape, to index into given axis of.
    
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    RETURNS
    data    ndarray. Input array with indexed values selected from given axis.
    """
    # Generate list of slices, with ':' for all axes except <idxs> for <axis>
    slices = _axis_slices(axis, idxs, data.ndim)

    # Use slices to index into data, and return sliced data
    return data[slices]


def _axis_slices(axis, idxs, ndim):
    """
    Generate list of slices, with ':' for all axes except <idxs> for <axis>,
    to use for dynamic indexing into an arbitary axis of an ndarray
    
    slices = _axis_slices(axis, idxs, ndim)
    
    ARGS
    axis    Int. Axis of ndarray to index into.
    
    idxs    (n_selected,) array-like of int | (axis_len,) array-like of bool | slice object
            Indexing into given axis of array, given either as list of
            integer indexes or as boolean vector.
    
    ndim    Int. Number of dimensions in ndarray to index into
    
    RETURNS
    slices  Tuple of slices. Index tuple to use to index into given 
            axis of ndarray as: selected_values = array[slices]  
    """
    # Initialize list of null slices, equivalent to [:,:,:,...]
    slices = [slice(None)] * ndim

    # Set slice for <axis> to desired indexing
    slices[axis] = idxs

    # Convert to tuple bc indexing arrays w/ a list is deprecated
    return tuple(slices)


def _reshape_data(data, axis=0):
    """
    Reshapes multi-dimensional data array to 2D (matrix) form for analysis

    data, data_shape = _reshape_data(data,axis=0)

    ARGS
    data    (...,n,...) ndarray. Data array of arbitrary shape.

    axis    Int. Axis of data to move to axis 0 for subsequent analysis. Default: 0

    RETURNS
    data    (n,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all axes > 0 unwrapped into single dimension, where 
            m = prod(shape[1:])

    data_shape (data.ndim,) tuple. Original shape of data array
    """
    if axis < 0: axis = data.ndim + axis    
    
    # Save original shape/dimensionality of <data>
    data_ndim  = data.ndim
    data_shape = data.shape

    if ~data.flags.c_contiguous:
        # If observation axis != 0, permute axis to make it so
        if axis != 0:       data = np.moveaxis(data,axis,0)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(data_shape[axis],-1),order='F')

    # Faster method for c-contiguous arrays
    else:
        # If observation axis != last dim, permute axis to make it so
        if axis != data_ndim - 1: data = np.moveaxis(data,axis,-1)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix, then transpose
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T
        else:               data = data.T

    return data, data_shape


def _unreshape_data(data, data_shape, axis=0):
    """
    Reshapes data array from unwrapped 2D (matrix) form back to ~ original
    multi-dimensional form

    data = _unreshape_data(data,data_shape,axis=0)

    ARGS
    data    (axis_len,m) ndarray. Data array w/ <axis> moved to axis=0, 
            and all axes > 0 unwrapped into single dimension, where 
            m = prod(shape[1:])

    data_shape (data.ndim,) tuple. Original shape of data array

    axis    Int. Axis of original data moved to axis 0, which will be shifted 
            back to original axis.. Default: 0

    RETURNS
    data    (...,axis_len,...) ndarray. Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)
    if axis < 0: axis = data.ndim + axis    

    data_ndim   = len(data_shape) # Number of dimensions in original data
    axis_len    = data.shape[0]   # Length of dim 0 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axis_length>)
    if data_ndim > 2:
        # Reshape data -> (axis_len,<original shape w/o <axis>>)
        shape = (axis_len,*data_shape[np.arange(data_ndim) != axis])
        # Note: I think you want the order to be 'F' regardless of memory layout
        # TODO test this!!!
        data  = np.reshape(data,shape,order='F')

    # If <axis> wasn't 0, move axis back to original position
    if axis != 0: data = np.moveaxis(data,0,axis)

    return data


def _unreshape_data_newaxis(data,data_shape,axis=0):
    """
    Reshapes data array from unwrapped form back to ~ original
    multi-dimensional form in special case where a new frequency axis was
    inserted before time axis (<axis>)

    data = _unreshape_data_newaxis(data,data_shape,axis=0)

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

    # If <axis> wasn't 0, move axis back to original position
    if axis != 0: data = np.moveaxis(data,(0,1),(axis,axis+1))
    
    return data


def _remove_buffer(data, buffer, axis=1):
    """
    Removes a temporal buffer (eg of zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    data = _remove_buffer(data,buffer,axis=1)

    ARGS
    data    Data array where a buffer has been appended on both ends of time dimension.
            Can be any arbitrary size, typically (n_freqs,n_timepts+2*buffer,...).
    buffer  Int. Length (number of samples) of buffer appended to each end.
    axis    Int. Array axis to remove buffer from (ie time dim). Default: 1

    RETURNS
    data    Data array with buffer removed, reducing time axis to n_timepts
            (typically shape (n_freqs,n_timepts,...))
    """
    if axis == 1:
        return data[:,buffer:-buffer,...] # special case for default
    elif axis == 0:
        return data[buffer:-buffer,...]
    else:
        return (data.swapaxes(0,axis)[buffer:-buffer,...]
                    .swapaxes(axis,0))


# =============================================================================
# Other helper functions
# =============================================================================
def _infer_data_type(data):
    """ Infers type of data signal given -- 'raw' (real) | 'spectral' (complex) """
    if np.isrealobj(data):  return 'raw'
    else:                   return 'spectral'


def _iarange(start=0, stop=0, step=1):
    """
    Implements Numpy arange() with an inclusive endpoint. Same inputs as arange(), same
    output, except ends at stop, not stop - 1 (or more generally stop - step)

    r = _iarange(start=0,stop=0,step=1)
    
    Note: Must input all 3 arguments or use keywords (unlike flexible arg's in arange)    
    """
    if isinstance(step,int):    return np.arange(start,stop+1,step)
    else:                       return np.arange(start,stop+1e-12,step)


def _next_power_of_2(n):
    """
    Rounds x up to the next power of 2 (smallest power of 2 greater than n)
    """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


def _infer_freq_scale(freqs):
    """ Determines if frequency sampling vector is linear or logarithmic """
    # Determine if frequency scale is linear (all equally spaced)
    if np.allclose(np.diff(np.diff(freqs)),0):
        return 'linear'
    
    # Determine if frequency scale is logarithmic (all equally spaced in log domain)
    elif np.allclose(np.diff(np.diff(np.log2(freqs))),0):
        return 'log'
    
    else:
        raise "Unable to determine scale of frequency sampling vector"


def _freq_to_scale(freqs, wavelet='morlet', wavenumber=6):
    """
    Converts wavelet center frequency(s) to wavelet scales. Typically used to
    convert set of frequencies you want to set of scales, which pycwt understands.
    Currently only supports Morlet wavelets.

    scales = _freq_to_scale(freqs,mother,wavenumber)

    ARGS
    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies

    wavelet     String. Type of wavelet. Default: 'morlet'

    wavenumber  Scalar. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
                Default: 6

    RETURNS
    scales      (n_freqs,) ndarray. Set of equivlent wavelet scales
    """
    freqs = np.asarray(freqs)
    if wavelet.lower() == 'morlet':
        return (wavenumber + sqrt(2 + wavenumber**2)) / (4*pi*freqs)
    else:
        raise ValueError("Unsupported value set for <wavelet>: '%s'" % wavelet)


def _interp1(x, y, xinterp, **kwargs):
    """
    Interpolates 1d data vector <y> sampled at index values <x> to
    new sampling vector <xinterp>
    Convenience wrapper around scipy.interpolate.interp1d w/o weird call structure
    """
    return interp1d(x,y,**kwargs).__call__(xinterp)


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
    
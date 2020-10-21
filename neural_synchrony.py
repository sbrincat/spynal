# -*- coding: utf-8 -*-
"""
A module for analysis of neural oscillations and synchrony

FUNCTIONS
### General spectral analysis ###
spectrum            Complex spectrum (Fourier coefficients) of data
spectrogram         Complex time-frequency transform (Fourier coeffs)
power_spectrum      Power spectrum of data
power_spectrogram   Power of time-frequency transform
phase_spectrum      TODO?
phase_spectrogram   Phase of time-frequency transform

### Multitaper spectral analysis ###
multitaper_spectrum Multitaper (DPSS) Fourier spectrum
multitaper_spectrogram  Multitaper (DPSS) time-frequency spectrogram
compute_tapers      Computes DPSS tapers for use in multitaper spectral analysis

### Wavelet spectral analysis ###
wavelet_spectrum    TODO?
wavelet_spectrogram Complex time-frequency continuous wavelet transform
compute_wavelets    Computes wavelets for use in wavelet spectral analysis

### Bandpass-filtering spectral analysis ###
bandfilter_spectrum TODO?
bandfilter_spectrogram Complex band-filtered, Hilbert-transformed time-freq(band)
set_filter_params   Sets filter coefficients for use in band-filtered analysis

### Field-field synchrony ###
coherence           Time-frequency coherence between pair of channels
ztransform_coherence Z-transform coherence so ~ normally distributed

phase_locking_value Phase locking value (PLV) between pair of channels
pairwise_phase_consistency Pairwise phase consistency (PPC) btwn pair of channels

### Spike-field synchrony ###
spike_field_coupling  General spike-field coupling/synchrony btwn spike/LFP pair
spike_field_coherence Spike-field coherence between a spike/LFP pair
spike_field_phase_locking_value Spike-field PLV between a spike/LFP pair
spike_field_pairwise_phase_consistency Spike-field PPC between a spike/LFP pair

### Preprocessing ###
remove_dc           Removes constant DC component of signals
remove_evoked       Removes phase-locked evoked potentials from signals

get_freq_sampling   Frequency sampling vector for a given FFT-based computation
setup_sliding_windows Generates set of sliding windows from given parameters

### Postprocesssing ###
pool_freq_bands     Averages spectral data within set of frequency bands
pool_time_epochs    Averages spectral data within set of time epochs

### Simulation/testing ###
simulate_mvar        Simulates network activity with given connectivity
network_simulation   Canned network simulations


DEPENDENCIES
pyfftw              Python wrapper around FFTW, the speedy FFT library
spike_analysis      A module for basic analyses of neural spiking activity


Created on Thu Oct  4 15:28:15 2018

@author: sbrincat
"""
# TODO  Figure out how to switch temporal output (spectrum vs spectrogram,
#       cont vs window vs epoch?) for each analysis. Code up missing functions:
#       wavelet_spectrum, waveletSpikeSpectrogram, multitaperSpikeSpectrogram
# TODO  Also option fixed-width vs variable width tapers (cf Fieldtrip)
# TODO  Also option output signal type to ***Spectrum/ogram func's (complex/power/phase)?
#       Same for synchrony funcs or stick with return_phase option?
# TODO  Add Hanning tapers as another spectral analysis option
# TODO  Build in ability to compute more than just pairwise sync measures?

from math import floor,ceil,log2,pi,sqrt
from collections import OrderedDict
from multiprocessing import cpu_count
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal.windows import dpss
from scipy.signal import filtfilt,hilbert,zpk2tf,butter,ellip,cheby1,cheby2
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

from spike_analysis import _spike_data_type, times_to_bool


# Set default arguments for pyfftw functions: Fast planning, use all available threads
_FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE',
                        'threads': cpu_count()}


# =============================================================================
# General spectral analysis functions
# =============================================================================
def spectrum(data, smp_rate, axis=0, method='multitaper', signal='lfp',
             **kwargs):
    """
    Computes complex frequency spectrum of data using given method

    spec,freqs = spectrum(data,smp_rate,axis=0,method='multitaper',
                          signal='lfp',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'multitaper' [default] (only value currently supported)

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    **kwargs    All other kwargs passed directly to method-specific
                spectrum function. See there for details.

    RETURNS
    spec        (...,n_freqs,...) ndarray of complex floats.
                Compplex frequency spectrum computed with given method.
                Frequency axis is always inserted in place of time axis
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after time axis.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)
    """
    assert signal in ['lfp','spike'], \
        ValueError("<signal> must be 'lfp' or 'spike' ('%s' given)" % signal)

    if method.lower() == 'multitaper':  spec_fun = multitaper_spectrum
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    spec,freqs = spec_fun(data,smp_rate,axis=axis,signal=signal, **kwargs)

    return spec, freqs


def spectrogram(data, smp_rate, axis=0, method='wavelet', signal='lfp',
                **kwargs):
    """
    Computes complex time-frequency transform of data using given method

    spec,freqs,timepts = spectrogram(data,smp_rate,axis=0,method='wavelet',
                                     signal='lfp',**kwargs)

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

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    **kwargs    All other kwargs passed directly to method-specific
                spectrogram function. See there for details.

    RETURNS
    spec        (...,n_freqs,n_timepts,...) ndarray of complex floats.
                Complex time-frequency spectrogram computed with given method.
                Frequency axis is always inserted just before time axis.
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after time axis.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timepts,...) ndarray. List of time points / time window centers
                for each time index in <spec>
    """
    assert signal in ['lfp','spike'], \
        ValueError("<signal> must be 'lfp' or 'spike' ('%s' given)" % signal)

    if method.lower() == 'wavelet':         spec_fun = wavelet_spectrogram
    elif method.lower() == 'multitaper':    spec_fun = multitaper_spectrogram
    elif method.lower() == 'bandfilter':    spec_fun = bandfilter_spectrogram
    else:
        raise ValueError("Unsupported value set for <method>: '%s'" % method)

    spec,freqs,timepts = spec_fun(data,smp_rate,axis=axis,signal=signal, **kwargs)

    return spec, freqs, timepts


def power_spectrum(data, smp_rate, axis=0, method='multitaper', **kwargs):
    """
    Computes power spectrum of data using given method

    spec,freqs = power_spectrum(data,smp_rate,axis=0,method='multitaper',**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    method      String. Specific spectral analysis method to use:
                'multitaper' [default] (only value currently supported)

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    **kwargs    All other kwargs passed directly to method-specific
                spectrum function. See there for details.

    RETURNS
    spec        (...,n_freqs,...) ndarray of complex floats.
                Compplex frequency spectrum computed with given method.
                Frequency axis is always inserted in place of time axis
                Note: 'multitaper' method will return with additional taper
                axis inserted after just after time axis.

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)
    """
    # Compute full complex spectrum
    spec,freqs = spectrum(data,smp_rate,axis=axis,method=method, **kwargs)

    # Compute power from complex spectral data (.real fixes small float errors)
    spec = (spec*spec.conj()).real

    # Compute mean across tapers for multitaper method (taper axis = <axis>+1)
    if method == 'multitaper': spec = spec.mean(axis=axis+1)

    return spec, freqs


def power_spectrogram(data, smp_rate, axis=0, method='wavelet', **kwargs):
    """
    Computes power of time-frequency transform of data using given method

    spec,freqs,timepts = power_spectrogram(data,smp_rate,axis=0,method='wavelet',
                                          **kwargs)

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

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timepts,...) ndarray. List of time points / time window
                centers for each time index in <spec>
    """
    # Compute full complex spectrogram
    spec,freqs,timepts = spectrogram(data,smp_rate,axis=axis,method=method,
                                     **kwargs)

    # Compute power from complex spectral data (.real fixe small float errors)
    spec = (spec*spec.conj()).real

    # Compute mean across tapers for multitaper method (taper axis = <axis>+1)
    if method == 'multitaper': spec = spec.mean(axis=axis+1)

    return spec, freqs, timepts


def phase_spectrogram(data,smp_rate,axis=0,method='wavelet',**kwargs):
    """
    Computes phase of time-frequency transform of data using given method

    spec,freqs,timepts = phase_spectrogram(data,smp_rate,axis=0,method='wavelet',
                                          **kwargs)

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

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timepts,...) ndarray. List of time points / time window centers
                for each time index in <spec>
    """
    spec,freqs,timepts = spectrogram(data,smp_rate,axis=axis,method=method, **kwargs)

    # Compute phase angle of spectrogram
    spec = np.angle(spec)

    # Compute circular mean across tapers for multitaper method (taper axis = <axis>+1)
    if method == 'multitaper': spec = np.exp(1j*spec).mean(axis=axis+1)

    return spec.real, freqs, timepts


# =============================================================================
# Multitaper spectral analysis functions
# =============================================================================
def multitaper_spectrum(data, smp_rate, axis=0, signal='lfp', freq_range=None,
                        tapers=None, pad=True, **kwargs):
    """
    Multitaper Fourier spectrum computation for continuous (eg LFP)
    or point process (eg spike) data

    spec,freqs = multitaper_spectrum(data,smp_rate,axis=0,signal='lfp',
                                    freq_range=None,tapers=None,pad=True,
                                    **kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of data to perform spectral analysis on (usually time dim)
                Default: 0

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    freq_range  (2,) array-like | Scalar. Range of frequencies to keep in output,
                either given as an explicit [low,high] range or just a scalar
                giving the highest frequency to return.
                Default: all frequencies from FFT

    tapers      (n_samples,n_tapers). Computed tapers (as computed by
                compute_tapers())

    pad         Bool. If True [default], zero-pads data to next power of 2 length

    RETURNS
    spec        (...,n_freqs,n_tapers,...) ndarray of complex floats.
                Complex multitaper spectrum of data. Sampling (time) axis is
                replaced by frequency, taper axes, but shape is otherwise preserved

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    REFERENCE   Mitra & Pesaran (1999) "Analysis of dynamic brain imaging data"
                Jarvis & Mitra (2001) Neural Computation

    SOURCE  Adapted from Chronux functions mtfftc.m, mtfftpb.m
    """
    # TODO  Do we need to remove pad after spectral analysis?
    # TODO  Think abt arg's to spikeTimes2bool -- do we need to hard-code some?

    # Convert spike timestamp data to boolean spike train format
    if (signal == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data,**kwargs)
        axis = data.ndim

    # If observation axis != 0, permute axis to make it so
    if axis != 0: data = np.moveaxis(data,axis,0)

    n_timepts = data.shape[0]
    # Set FFT length = data length if no padding; else pad to next power of two
    if not pad: n_fft = n_timepts
    else:       n_fft = _next_power_of_2(n_timepts)
    # Set frequency sampling vector
    freqs,fbool = get_freq_sampling(smp_rate,n_fft,freq_range=freq_range)

    # Reshape tapers to (n_timepts,n_tapers) (if not already)
    assert tapers is not None, \
        "Must input taper functions in <tapers>. Use compute_tapers()"

    if (tapers.ndim == 2) and (tapers.shape[1] == n_timepts): tapers = tapers.T
    assert tapers.shape[0] == n_timepts, \
        ValueError("tapers must have same length (%d) as number of timepoints in data (%d)"
                   % (tapers.shape[0],n_timepts))

    # Reshape tapers array to pad end of it w/ singleton dims
    taper_shape  = (*tapers.shape,*np.ones((data.ndim-1,),dtype=int))

    # Compute values needed for normalizing point process (spiking) signals
    if signal == 'spike':
        # Compute Fourier transform of tapers
        taper_fft= fft(tapers,n=n_fft,axis=0)
        if data.ndim > 1:  taper_fft = np.reshape(taper_fft,taper_shape)
        # Compute mean spike rate across all timepoints in each data series
        n_spikes = np.sum(data,axis=0,keepdims=True)/n_timepts

    # Reshape tapers and data to have appropriate shapes to broadcast together
    if data.ndim > 1:  tapers = np.reshape(tapers,taper_shape)

    # Insert dimension for tapers in data axis 1 -> (n_timepts,1,...)
    data    = data[:,np.newaxis,...]

    # Project data onto set of taper functions
    data    = data * tapers

    # Compute Fourier transform of projected data, normalizing appropriately
    spec    = fft(data,n=n_fft,axis=0)/smp_rate

    # Subtract off the DC component (average spike rate) for point process signals
    if signal == 'spike':  spec = data - taper_fft*n_spikes

    # Extract desired set of frequencies
    spec    = spec[fbool,...]

    # If observation axis wasn't 0, permute (freq,tapers) back to original position
    if axis != 0: data = np.moveaxis(data,[0,1],[axis,axis+1])

    return spec, freqs


def multitaper_spectrogram(data, smp_rate, axis=0, signal='lfp',
                           freq_range=None, window=0.5, spacing=None,
                           tapers=None, pad=True, **kwargs):
    """
    Multitaper time-frequency spectrogram computation for continuous (eg LFP)
    or point process (eg spike) data

    spec,freqs,timepts = multitaper_spectrogram(data,smp_rate,axis=0,
                                               signal='lfp',freq_range=None,
                                               window=0.5,spacing=None,
                                               tapers=None,pad=True,**kwargs)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate    Scalar. Data sampling rate (Hz)

    axis        Int. Axis of data to perform spectral analysis on (usually time dim)
                Default: 0

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    freq_range  (2,) array-like | Scalar. Range of frequencies to keep in output,
                either given as an explicit [low,high] range or just a scalar
                giving the highest frequency to return.
                Default: all frequencies from FFT

    window      Scalar. Width of sliding time window (s) [default: 0.5 s].
                Should have same size as tapers (ie smp_rate*window = tapers.shape[0])

    spacing     Scalar. Spacing between successive sliding time windows (s).
                Default: Set = window width (so each window exactly non-overlapping)

    tapers      (nPerWin,n_tapers). Computed tapers (as computed by compute_tapers()).

    pad         Bool. If True [default], zero-pads data to next power of 2 length

    RETURNS
    spec        (...,n_freqs,n_timewins,n_tapers,...) ndarray of complex floats.
                Complex multitaper time-frequency spectrogram of data.
                Sampling (time) axis is replaced by frequency, taper, time window
                axes but shape is otherwise preserved

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timewins,...) ndarray. List of timepoints (center of each
                time window) in <spec>.

    REFERENCE   Mitra & Pesaran (1999) "Analysis of dynamic brain imaging data"
                Jarvis & Mitra (2001) Neural Computation

    SOURCE      Adapted from Chronux function mtfftc.m
    """
    # Convert spike timestamp data to boolean spike train format
    if (signal == 'spike') and (_spike_data_type(data) == 'timestamp'):
        data,_ = times_to_bool(data,**kwargs)
        axis = data.ndim

    # If observation axis != 0, permute axis to make it so
    if axis != 0: data = np.moveaxis(data,axis,0)
    n_timepts = data.shape[0]

    if spacing is None: spacing = window
    if tapers is None:  tapers = compute_tapers(smp_rate,T=window,W=4,n_tapers=3)

    # Set up parameters for data time windows
    # Set window starts to range from time 0 to time n - window width (1e-12 for fp err)
    win_starts  = np.arange(0,n_timepts/smp_rate-window+1e-12,spacing)
    # Set sampled timepoint vector = center of each window
    timepts     = win_starts + window/2.0

    # Extract time-windowed version of data -> (nWinTimePts,n_wins,nDataSeries)
    data = _extract_triggered_data(data,smp_rate,win_starts,[0,window])

    # Do multitaper analysis on windowed data
    spec, freqs = multitaper_spectrum(data,smp_rate,axis=0,signal=signal,
                                      freq_range=freq_range,tapers=tapers,pad=pad,
                                      **kwargs)

    # If time axis wasn't 0, permute (freq,tapers,timewin) axes back to original position
    if axis != 0: data = np.moveaxis(data,[0,1,2],[axis,axis+1,axis+2])

    return spec, freqs, timepts


def compute_tapers(smp_rate,T=None,W=None,TW=None,n_tapers=None):
    """
    Computes Discrete Prolate Spheroidal Sequence (DPSS) tapers for use in
    multitaper spectral analysis.

    Uses scipy.signal.windows.dpss, but arguments are different here

    tapers = compute_tapers(smp_rate,T=None,W=None,TW=None,n_tapers=None)
    ARGS
    smp_rate Scalar. Data sampling rate (Hz)

    Note: Must input at least 2 of 3 args from set T,W,TW. The 3rd is computed
          using TW = T*W. If values for all 3 arg's given, must have TW = T*W.

    T       Scalar. Time bandwidth (s). Should match data window length (in s).
    W       Scalar. Frequency bandwidth (Hz)
    TW      Scalar. Time-frequency bandwidth product.

    n_tapers Scalar. Number of tapers to compute. Must be <= 2TW-1, as this is
            the max number of spectrally delimited tapers. Default: 2TW-1

    RETURNS
    tapers (n_samples,n_tapers). Computed tapers (n_samples = T*smp_rate)

    SOURCE  Adapted from Cronux function dpsschk.m
    """
    n_args = (T is not None) + (W is not None) + (TW is not None)
    assert n_args >= 2, ValueError("At least 2/3 arguments from set T,W,TW must be given")

    # If all 3 arg's are set, make sure they are consistent
    if n_args == 3:
        assert TW == T*W, \
            ValueError("If all arg's are given, T (%1f) * W (%.1f) must = TW (%.1f)"
                       % (T,W,TW))
    # If 2 arg's are set, set the 3rd one appopriately
    elif TW is None:    TW  = T*W
    elif T is None:     T   = TW/W
    elif W is None:     W   = TW/T

    # Up to 2TW-1 tapers are bounded; this is both the default and max value for n_tapers
    n_tapers_max = floor(2*TW - 1)
    if n_tapers is None:     n_tapers = n_tapers_max
    else:
        assert n_tapers <= n_tapers_max, \
            ValueError("For TW = %.1f, %d tapers are tightly bounded in" \
                       "frequency (n_tapers set = %d)" \
                       % (TW,n_tapers_max,n_tapers))

    # Convert time bandwidth from s to window length in number of samples
    M = int(round(T*smp_rate))

    # Compute the tapers for given window length and time-freq product
    # Note: dpss() normalizes by sum of squares; x sqrt(smp_rate)
    #       converts this to integral of squares (see Chronux function dpsschk())
    # Note: You might imagine you'd want sym=False, but sym=True gives same values
    #       as Chronux dpsschk() function...
    return dpss(M, TW, Kmax=n_tapers, sym=True, norm=2).T * sqrt(smp_rate)


# =============================================================================
# Wavelet analysis functions
# =============================================================================
def wavelet_spectrogram(data, smp_rate, axis=0, signal='lfp',
                        freqs=2**np.arange(1,7.5,0.25), wavelet='morlet',
                        wavenumber=6, pad=False, buffer=0, downsmp=1):
    """
    Computes continuous wavelet transform of given data signal at given frequencies.


    spec,freqs,timepts = wavelet_spectrogram(data,smp_rate,axis=0,signal='lfp',
                                            freqs=2**np.arange(1,7.5,0.25),
                                            wavelet='morlet',wavenumber=6,
                                            pad=True,buffer=0,downsmp=1)

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    freqs       (n_freqs,) array-like. Set of desired wavelet frequencies
                Default: 2**np.irange(1,7.5,0.25) (log sampled in 1/4 octaves from 2-128)

    wavelet     String. Name of wavelet type. Default: 'morlet'

    wavenumber  Int. Wavelet wave number parameter ~ number of oscillations
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
                Default: 6

    pad         Bool. If True, zero-pads data to next power of 2 length. Default: False

    buffer      Int. Length (number of samples in original sampling rate) of
                buffer to trim off each end of time dimension of data. Removes
                symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    downsmp     Int. Factor to downsample time sampling by (after spectral analysis).
                eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)
                Default: 1 (no downsampling)

    RETURNS
    spec        (n_freqs,n_timepts_out,...) ndarray of complex.
                Complex wavelet spectrogram of data

    freqs       (n_freqs,) ndarray. List of frequencies in <spec> (in Hz)

    timepts     (n_timepts_out,...) ndarray. List of timepoints (indexes into original
                data time series) in <spec>.

    REFERENCE   Torrence & Compo (1998) "A Practical Guide to Wavelet Analysis"
    """
    # Convert buffer from s -> samples
    if buffer != 0:  buffer  = int(ceil(buffer*smp_rate))

    # Reshape data array -> (n_timepts_in,nDataSeries) matrix
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

    # Compute FFT of data
    data = fft(data, n=n_fft,axis=0, **_FFTW_KWARGS_DEFAULT)
    
    # Reshape data -> (1,n_timepts,n_series) (insert axis 0 for wavelet scales/frequencies)
    # Reshape wavelets -> (n_freqs,n_timepts,1) to broadcast 
    #  (except for special case of 1D data with only a single time series)
    data = data[np.newaxis,...]
    if data.ndim == 3: wavelets_fft = wavelets_fft[:,:,np.newaxis]

    spec = ifft(data*wavelets_fft, n=n_fft,axis=1, **_FFTW_KWARGS_DEFAULT)[:,timepts_out,...]

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
                in each wavelet. Must be >= 6 to meet "admissibility constraint".
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
    dt      = 1/smp_rate         # Convert sampling rate -> sampling interval
    freqs   = np.asarray(freqs)

    if wavelet.lower() == 'morlet':
        # Conversion factor from scale to Fourier period for Morlet wavelets
        # (~ 1.033... for wavenumber = 6)
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

        # DEL wavelets = empty_aligned((len(freqs),len(k)))
        # Fourier transform of Wavelet function
        if do_fft:
            wavelets = norm*np.exp(exponent) * (k > 0)
        else:
            raise "non-FFT wavelet output not coded up yet (TODO)"

    else:
        raise ValueError("Unsupported value '%s' given for <wavelet>." \
                         "Currently only 'Morlet' suppported")

    return wavelets


# =============================================================================
# Band-pass filtering analysis functions
# =============================================================================
def bandfilter_spectrogram(data, smp_rate, axis=0, signal='lfp',
                           freqs=((2,8),(10,32),(40,100)), filt='butter',
                           params=None, buffer=0, downsmp=1, **kwargs):
    """
    Computes zero-phase band-filtered and Hilbert-transformed signal from data
    for given frequency band(s).

    Function aliased as bandfilter_spectrogram() or bandfilter().

    spec,freqs,timepts = bandfilter_spectrogram(data,smp_rate,axis=0,signal='lfp',
                                               freqs=((2,8),(10,32),(40,100)),
                                               filt='butter',
                                               params=None,buffer=0,downsmp=1,
                                               **kwargs):

    ARGS
    data        (...,n_samples,...) ndarray. Data to compute spectral analysis of.
                Arbitrary shape; spectral analysis is computed along axis <axis>.

    smp_rate     Scalar. Data sampling rate (Hz)

    axis        Int. Axis of <data> to do spectral analysis on
                (usually time dimension). Default: 0

    signal      String. Type of signal in data: 'lfp' [default] or 'spike'

    freqs       (nFreqBands,) array-like of (2,) sequences | (nFreqBands,2) ndarray.
                [low,high] cut frequencies for each band to use.
                Set 1st value = 0 for low-pass; set 2nd value >= smp_rate/2 for
                high-pass; otherwise assumes band-pass.
                Default: ((2,8),(10,32),(40,100)) (~theta, alpha/beta, gamma)
                ** Only used if filter <params> not explitly given **

    filt        String. Name of filter to use. Default: 'butter' (Butterworth)
                ** Only used if filter <params> not explitly given **

    params      Dict. Contains parameters that define filter for each freq band.
                Can precompute params with set_filter_params() and input explicitly
                OR input values for <freqs> and params will be computed here.
                One of two forms: 'ba' or 'zpk', with key/values as follows:

        b,a     (nFreqBands,) lists of vectors. Numerator <b> and
                denominator <a> polynomials of the filter for each band.

        z,p,k   Zeros, poles, and system gain of the IIR filter transfer function

    buffer      Int. Length (number of samples in original sampling rate) of
                buffer to trim off each end of time dimension of data. Removes
                symmetric buffer previously added (outside of here) to prevent
                edge effects. Default: 0 (no buffer)

    downsmp     Int. Factor to downsample time sampling by (after spectral analysis).
                eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)
                Default: 1 (no downsampling)

    Any other kwargs passed directly to set_filter_params()

    RETURNS
    spec        (nFreqBands,n_timepts_out,...) ndarray of complex floats.
                Complex band-filtered, Hilbert-transformed "spectrogram" of data

    freqs       (nFreqBands,) ndarray. List of center frequencies of bands in <spec>

    timepts     (n_timepts_out,...) ndarray. List of timepoints (indexes into original
                data time series) in <spec>.
    """
    if signal == 'spike':
        raise ValueError("bandfilter for spike signals not supported (yet)") # TODO

    # Set filter parameters from frequency bands if <params> not passed in
    if params is None:
        assert freqs is not None, \
            ValueError("Must input a value for either filter <params> or band <freqs>")

        freqs   = np.asarray(freqs)  # Convert freqs to (nFreqBands,2)
        n_freqs = freqs.shape[0]
        params  = set_filter_params(freqs,smp_rate,filt,return_dict=True,**kwargs)

    # Determine form of filter parameters given: b,a or z,p,k
    else:
        if np.all([(param in params) for param in ['b','a']]):       form = 'ba'
        elif np.all([(param in params) for param in ['z','p','k']]): form = 'zpk'
        else:
            raise ValueError("<params> must be a dict with keys 'a','b' or 'z','p','k'")

        # Convert zpk form to ba
        if form == 'zpk':
            n_freqs = len(params['z'])
            params['b'] = [None for j in range(n_freqs)]
            params['a'] = [None for j in range(n_freqs)]
            for i_freq in range(n_freqs):
                b,a = zpk2tf(params['z'][i_freq],params['p'][i_freq],params['k'][i_freq])
                params['b'][i_freq] = b
                params['a'][i_freq] = a
        else:
            n_freqs = len(params['b'])


    # Compute center freq of each band, to have uniform output w/ other func's
    if freqs is not None:
        freqs[freqs > smp_rate/2] = smp_rate/2
        freqs = freqs.mean(axis=1)

    # Reshape data array -> (n_timepts_in,nDataSeries) matrix
    data, data_shape = _reshape_data(data,axis)
    if data.ndim == 1:  n_timepts_in,n_series = data.shape[0],1
    else:               n_timepts_in,n_series = data.shape
    n_freqs = len(freqs)

    timepts_out     = np.arange(buffer,n_timepts_in-buffer,downsmp)
    n_timepts_out   = len(timepts_out)

    spec = np.zeros((n_freqs,n_timepts_out,n_series),dtype=complex)

    # For each frequency band, band-filter raw signal and
    # compute complex analytic signal using Hilbert transform
    for i_freq in range(n_freqs):
        b, a = params['b'][i_freq], params['a'][i_freq]
        bandfilt = filtfilt(b,a,data,axis=0)
        spec[i_freq,:,:] = hilbert(bandfilt[timepts_out,:],axis=0)

    spec = _unreshape_data_newaxis(spec,data_shape,axis=axis)

    return spec, freqs, timepts_out

bandfilter = bandfilter_spectrogram  # Alias function to bandfilter()


def set_filter_params(bands, smp_rate, filt, order=4, form='ba',
                      return_dict=False, **kwargs):
    """
    Sets coefficients for desired filter(s) using scipy.signal
    "Matlab-style IIR filter design" functions

    params = set_filter_params(bands,smp_rate,filt,order=4,form='ba',return_dict=False,**kwargs)
    b,a = set_filter_params(bands,smp_rate,filt,order=4,form='ba',return_dict=False,**kwargs)
    z,p,k = set_filter_params(bands,smp_rate,filt,order=4,form='ba',return_dict=False,**kwargs)

    ARGS
    bands       (nFreqBands,) array-like of (2,) sequences | (nFreqBands,2) ndarray.
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

    b,a         (nFreqBands,) list of vectors. Numerator <b> and
                denominator <a> polynomials of the filter for each band.
                Returned if form='ba'.

    z,p,k       Zeros, poles, and system gain of the IIR filter transfer
                function. Returned if form='zpk'.
    """
    # Convert bands to (nFreqBands,2)
    bands       = np.asarray(bands)
    if bands.ndim == 1: bands = np.reshape(bands,(1,len(bands)))
    n_bands     = bands.shape[0]
    nyquist     = smp_rate/2.0   # Nyquist freq at given sampling freq

    # Setup filer-generating function for requested filter type
    # Butterworth filter
    if filt.lower() in ['butter','butterworth']:
        filt = lambda band,btype: butter(order,band,btype=btype,output=form)
    # Elliptic filter
    elif filt.lower() in ['ellip','butterworth']:
        rp = kwargs['rp'] if 'rp' in kwargs else 5
        rs = kwargs['rs'] if 'rs' in kwargs else 40
        filt = lambda band,btype: ellip(order,rp,rs,band,btype=btype,output=form)
    # Chebyshev Type 1 filter
    elif filt.lower() in ['cheby1','cheby','chebyshev1','chebyshev']:
        rp = kwargs['rp'] if 'rp' in kwargs else 5
        filt = lambda band,btype: cheby1(order,rp,band,btype=btype,output=form)
    # Chebyshev Type 2 filter
    elif filt.lower() in ['cheby2','chebyshev2']:
        rs = kwargs['rs'] if 'rs' in kwargs else 40
        filt = lambda band,btype: cheby2(order,rs,band,btype=btype,output=form)
    else:
        raise ValueError("Filter type '%s' is not supported (yet)" % filt)

    # Setup empty lists to hold filter parameters
    plist = [None for j in range(n_bands)]
    if form == 'ba':    params = OrderedDict({'b':plist, 'a':plist})
    elif form == 'zpk': params = OrderedDict({'z':plist, 'p':plist, 'k':plist})
    else:
        raise ValueError("Output form '%s' is not supported. Should be 'ba' or 'zpk'" % form)

    for i_band in range(n_bands):
        band = bands[i_band,:]/nyquist  # Convert band to normalized frequency

        # If low-cut freq = 0, assume low-pass filter
        if band[0] == 0:    btype = 'lowpass';  band = band[1]
        # If high-cut freq >= Nyquist freq, assume high-pass filter
        elif band[1] >= 1:  btype = 'highpass'; band = band[0]
        # Otherwise, assume band-pass filter
        else:               btype = 'bandpass'

        if form == 'ba':
            params['b'][i_band],params['a'][i_band] = filt(band,btype)
        else:
            params['z'][i_band],params['p'][i_band],params['k'][i_band] = filt(band,btype)

    if return_dict: return params
    else:           return params.values


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

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,signal='lfp',**kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,signal='lfp',**kwargs)
        if axis >= time_axis: axis += 1   # Account for new frequency axis
    else:
        freqs = []
        timepts = []

    # For multitaper, compute means across trials, tapers; df = 2*n_trials*n_tapers
    if method == 'multitaper':
        reduce_axes = (axis,time_axis+1)
        df = 2*data1.shape[axis]*data1.shape[time_axis+1]
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

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,signal='lfp', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,signal='lfp', **kwargs)
        if axis >= time_axis: axis += 1   # Account for new frequency axis
    else:
        freqs = []
        timepts = []

    # Standard across-trial PLV estimator
    if single_trial is None:
        if return_phase:
            plv,dphi = _spec_to_plv_with_phase(data1,data2,axis=axis)
            return  plv, freqs, timepts, dphi

        else:
            plv = _spec_to_plv(data1,data2,axis=axis)
            return  plv, freqs, timepts

    # Single-trial PLV estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_plv(data1,data2,axis=0)
        # Jackknife resampling of PLV statistic (this is the 'richter' estimator)
        plv = two_sample_jackknife(jackfunc,data1,data2,axis=axis)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            plv_full = _spec_to_plv(data1,data2,axis=axis,keepdims=True)
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

    n_obs    = data1.shape[axis]

    if data_type is None: data_type = _infer_data_type(data1)
    # If raw data is input, compute spectral transform first
    if data_type == 'raw':
        assert smp_rate is not None, "For raw/time-series data, need to input value for <smp_rate>"
        assert time_axis is not None, "For raw/time-series data, need to input value for <time_axis>"
        
        data1,freqs,timepts = spectrogram(data1,smp_rate,axis=time_axis,
                                          method=method,signal='lfp', **kwargs)
        data2,freqs,timepts = spectrogram(data2,smp_rate,axis=time_axis,
                                          method=method,signal='lfp', **kwargs)
        if axis >= time_axis: axis += 1   # Account for new frequency axis
    else:
        freqs = []
        timepts = []

    # Standard across-trial PPC estimator
    if single_trial is None:
        if return_phase:
            ppc,dphi = _spec_to_ppc_with_phase(data1,data2,axis=axis)
            return ppc, freqs, timepts, dphi

        else:
            ppc = _spec_to_ppc(data1,data2,axis=axis)
            return ppc, freqs, timepts

    # Single-trial PPC estimator using jackknife resampling method
    else:
        # Note: two_sample_jackknife() (temporarily) shifts trial axis to 0, so axis=0 here
        jackfunc = lambda data1,data2: _spec_to_ppc(data1,data2,axis=0)
        # Jackknife resampling of PPC statistic (this is the 'richter' estimator)
        ppc = two_sample_jackknife(jackfunc,data1,data2,axis=axis)
        # Convert to jackknife pseudovalues = n*stat_full - (n-1)*stat_jackknife
        if single_trial == 'pseudo':
            ppc_full = _spec_to_ppc(data1,data2,axis=axis,keepdims=True)
            ppc = jackknife_to_pseudoval(ppc_full,ppc,n_obs)

        return ppc, freqs, timepts


def _spec_to_ppc(data1, data2, axis=0, keepdims=False):
    """ Compute PPC from a pair of spectra/spectrograms """
    n = data1.shape[axis]
    return plv_to_ppc(_spec_to_plv(data1,data2,axis=axis,keepdims=keepdims), n)


def _spec_to_ppc_with_phase(data1, data2, axis=0, keepdims=False):
    """ Compute PPC and mean relative phase from a pair of spectra/spectrograms """
    n = data1.shape[axis]
    plv,dphi = _spec_to_plv_with_phase(data1,data2,axis=axis,keepdims=keepdims)
    return plv_to_ppc(plv,n), dphi


def plv_to_ppc(plv, n):
    """ Converts PLV to PPC as PPC = (n*PLV^2 - 1)/(n-1) """
    return (n*plv**2 - 1) / (n - 1)


# =============================================================================
# Spike-Field Synchrony functions
# =============================================================================
def spike_field_coupling(spkdata, lfpdata, method='PPC', **kwargs):
    """
    Wrapper around all spike-field coupling functions

    ARGS
    method  String. Spike-field coupling method to use.
            Options: 'PPC' [default] | 'PLV' | 'coherence'

    All other arguments passed directly to specific spike-field coupling function.
    See those functions for rest of arguments
    """
    method = method.lower()

    if method.lower() == 'ppc':
        sfc_func = spike_field_pairwise_phase_consistency
    elif method.lower() == 'plv':
        sfc_func = spike_field_phase_locking_value
    elif method.lower() in ['coherence','coh']:
        sfc_func = spike_field_coherence
    else:
        raise ValueError("Unsuuported value '%s' given for <method>. \
                         Should be 'PPC'|'PLV'|'coherence'" % method)

    return sfc_func(spkdata,lfpdata, **kwargs)


def spike_field_coherence(spkdata, lfpdata, axis=0, data_type=None,
                          spec_method='multitaper',
                          smp_rate=None,time_axis=None,**kwargs):
    """
    Computes pairwise coherence between single-channel spiking data and LFP data

    coh,freqs,timepts = spike_field_coherence(spkdata,lfpdata,axis=0,data_type=None,
                                            spec_method='multitaper',smp_rate=None,
                                            time_axis=None,**kwargs)

    ARGS
    todo Fix documentation
    data1,data2   (...,n_obs,...) ndarray. Single-channel LFP data for 2 distinct channels.
            Can be given as raw LFPs or complex-valued time-frequency transform.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have
            Can have arbitrary shape, with analysis performed independently
            along each dimension other than observation <axis> (eg different
            frequencies, timepoints, conditions, etc.)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    single_trial String or None. What type of coherence estimator to compute:
            None        standard across-trial estimator [default]
            'pseudo'    single-trial estimates using jackknife pseudovalues
            'richter'   single-trial estimates using actual jackknife estimates
                        as in Richter & Fries 2015

    ztransform Bool. If True, returns z-transformed coherence using Jarvis &
            Mitra (2001) method. If false [default], returns raw coherence.

    data_type Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
            Default: assume 'raw' if data is real; 'spectral' if complex

    Following args are only used for spectral analysis for data_type == 'raw'

    method  String. Spectral method. 'wavelet' | 'multitaper' [default]

    smp_rate Scalar. Sampling rate of data (only needed for raw data)

    time_axis Scalar. Axis of data corresponding to time (only needed for raw data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    coh     ndarray. Magnitude of coherence between spikeData and lfpdata.
            If lfpdata is spectral, this has shape as lfpdata, but with <axis> removed.
            If lfpdata is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in coh (only for raw data)
    timepts (n_timepts,). List of timepoints in coh (only for raw data)
    """
    # If raw data is input, compute spectral transform first
    if (data_type == 'raw') or (_infer_data_type(spkdata) == 'raw'):
        # Ensure spkdata is boolean array flagging spiking times
        assert spkdata.dtype != object, \
            "Spiking data must be converted from timestamps to boolean format for this function"
        spkdata = spkdata.astype(bool)

        spkdata,freqs,timepts = spectrogram(spkdata,smp_rate,axis=time_axis,
                                            method=spec_method,signal='spike',
                                            **kwargs)

    if (data_type == 'raw') or (_infer_data_type(lfpdata) == 'raw'):
        lfpdata,freqs,timepts = spectrogram(lfpdata,smp_rate,axis=time_axis,
                                            method=spec_method,signal='lfp',
                                            **kwargs)
        # Frequency axis always inserted just before time axis, so if
        # observation/trial axis is later, must increment it
        if axis >= time_axis: axis += 1
    else:
        freqs = []
        timepts = []

    coh,_,_ = coherence(spkdata,lfpdata,axis=axis,data_type='spectral')

    return coh,freqs,timepts


def spike_field_phase_locking_value(spkdata, lfpdata, axis=0, return_phase=False,
                                    timepts=None, timewins=None,
                                    spec_method='wavelet', data_type=None, smp_rate=None,
                                    time_axis=None, **kwargs):
    """
    Computes phase locking value (PLV) of spike-triggered LFP phase

    PLV is the mean resultant length (magnitude of the vector mean) of the
    spike-triggered LFP phase phi:
        plv  = abs( trialMean(exp(i*phi)) )

    plv,freqs,timepts = spike_field_phase_locking_value(spkdata,lfpdata,axis=0,return_phase=False,
                                          spec_method='wavelet',data_type=None,
                                          smp_rate=None,time_axis=None,
                                          **kwargs)

    ARGS
    TODO Redo docs
    spkdata (...,n_obs,...) ndarray of bool. Binary spike trains (with 1's labelling)
            spike times. Shape is arbitrary, but MUST have same shape as lfpdata.
            Thus, if lfpdata is spectral, must pre-pend singleton dimension to
            spkdata to match (eg using np.newaxis).

    lfpdata (...,n_obs,...) ndarray of float or complex. LFP data, given either
            as (real) raw data, or as complex spectral data.

            For raw data, axis corresponding to time must be given in <time_axis>.
            Trial/observation axis is assumed to be axis 0 unless given in <axis>.

            Other than those constraints, data can have arbitrary shape, with
            analysis performed independently in mass-bivariate fashion along
            each dimension other than observation <axis> (eg different conditions)

    axis    Scalar. Axis corresponding to distinct observations/trials. Default: 0

    return_phase Bool. If True, returns add'l output with mean spike-triggered phase

    timewins (n_wins,2) ndarray. Time windows to compute PLV within, given as
            series of window [start end]'s

            Can instead give window parameters as kwargs:
            window  Scalar. Window width (s). Default: 0.5
            spacing Scalar. Window spacing (s)
                    Default: <window> (giving exactly non-overlapping windows)
            lims    (2,) array-like. [Start,end] limits for windows
                    Default: (timepts[0],timepts[-1]) (full sampled time of LFPs)

    Following args are only used for spectral analysis for data_type == 'raw'

    method  String. Spectral method. 'wavelet' [default] | 'multitaper'

    data_type Str. What kind of data are we given in data1,data2: 'raw' or 'spectral'
            Default: assume 'raw' if data is real; 'spectral' if complex

    smp_rate Scalar. Sampling rate of data (only needed for raw data)
    time_axis Scalar. Axis of data corresponding to time (only needed for raw data)

    Any other kwargs passed as-is to spectrogram() function.

    RETURNS
    plv     ndarray. Phase locking value between spike and LFP data.
            If lfpdata is spectral, this has same shape, but with <axis> removed.
            If lfpdata is raw, this has same shape with <axis> removed and a new
            frequency axis inserted immediately before <time_axis>.

    freqs   (n_freqs,). List of frequencies in plv (only for raw data)
    timepts (n_timepts,). List of timepoints in plv (only for raw data)

    phi     ndarray. If return_phase is True, mean spike-triggered LFP phase
            (in radians) is also returned here

    REFERENCES
    Lachaux et al. (1999) Human Brain Mapping
    """
    # FIXME I think we need to require timepts argument, no?
    # Ensure spkdata boolean array (not timestamps), is same shape as lfpdata
    assert spkdata.dtype != object, \
        "Spiking data must be converted from timestamps to boolean format for this function"
    assert (spkdata.ndim == lfpdata.ndim) and (spkdata.shape[1:] == lfpdata.shape[1:]), \
        "Spiking data must have same size/shape as LFP data (minus any frequency axis)"

    spkdata = spkdata.astype(bool)  # Ensure spkdata is boolean array

    # If raw data is input, compute spectral transform first
    if (data_type == 'raw') or (_infer_data_type(lfpdata) == 'raw'):
        lfpdata,freqs,time_idxs = spectrogram(lfpdata,smp_rate,axis=time_axis,
                                              method=spec_method,signal='lfp',
                                              **kwargs)
        timepts = timepts[time_idxs]

        # Frequency axis always inserted just before time axis, so if
        # observation/trial axis is later, must increment it
        if axis >= time_axis: axis += 1

        # Insert singleton dimension into spkdata to match freq dim in lfpdata
        spkdata = spkdata[np.newaxis,:,:]

    else:
        freqs = []

    # Set timewins based on given parameters if not set in args
    if timewins is None:
        window  = kwargs.pop('window',0.5)
        spacing = kwargs.pop('spacing',window)
        lims    = kwargs.pop('lims',(timepts[0],timepts[-1]))
        timewins = setup_sliding_windows(window,lims,spacing)
    else:
        timewins = np.asarray(timewins)

    # Move trials/observations axis to end of data arrays
    if axis not in [-1,lfpdata.ndim]:
        lfpdata = np.moveaxis(lfpdata,axis,-1) # -> (n_freqs,n_timepts,n_trials)
        spkdata = np.moveaxis(spkdata,axis,-1) # -> (1,n_timepts,n_trials)

    # Remove singleton freq dim from spkdata (yes, kinda pointless to have it in
    # the first place, but makes axis bookeeping tractable)
    spkdata = spkdata.squeeze(axis=0)

    n_freqs = lfpdata.shape[0]
    n_timewins = timewins.shape[0]

    # Normalize LFP spectrum/spectrogram so data is all unit-length complex vectors
    lfpdata = lfpdata / np.abs(lfpdata)

    # Convert time sampling vector and time windows to int-valued ms,
    #  to avoid floating-point issues in indexing below
    timepts_ms  = np.round(timepts*1000)
    timewins_ms = np.round(timewins*1000)

    vector_mean = np.full((n_freqs,n_timewins),np.nan,dtype=complex)
    n = np.zeros((n_timewins,))

    for i_timewin,timewin in enumerate(timewins_ms):
        # Boolean vector flagging all time points within given time window
        tbool = (timepts_ms >= timewin[0]) & (timepts_ms <= timewin[1])

        # Logical AND btwn window and spike train booleans to get spikes in window
        win_spikes = spkdata & tbool[:,np.newaxis] # Expand win bool to trial dim

        # Count of all spikes within time window across all trials/observations
        n[i_timewin] = win_spikes.sum()

        # If no spikes in window, can't compute PLV. Skip and leave = nan.
        if n[i_timewin] == 0: continue

        # Compute complex vector mean of all spike-triggered LFP phase angles
        for i_freq in range(n_freqs):
            # Extract LFP data for current frequency
            lfpdata_f = lfpdata[i_freq,:,:]
            # Complex vector mean across all trials and all timepts in window
            vector_mean[i_freq,i_timewin] = lfpdata_f[win_spikes].mean()

    # Compute absolute value of complex vector mean = mean resultant = PLV
    # and optionally the mean phase angle as well. Also return spike counts.
    if return_phase:
        return np.abs(vector_mean), freqs, timepts, n, np.angle(vector_mean)
    else:
        return np.abs(vector_mean), freqs, timepts, n


def spike_field_pairwise_phase_consistency(spkdata, lfpdata, axis=0,
                                           return_phase=False, **kwargs):
    """
    Computes pairwise phase consistency (PPC) of spike-triggered LFP phase,
    which is unbiased by n (unlike PLV and coherence)

    PPC is an unbiased estimator of PLV^2, and can be expressed (and computed
    efficiently) in terms of PLV and n:
        PPC = (n*PLV^2 - 1) / (n-1)

    TODO Rest of docs
    """
    if return_phase:
        plv,freqs,timepts,n,mean_phase = \
        spike_field_phase_locking_value(spkdata,lfpdata,axis=axis,
                                        return_phase=True, **kwargs)
        return plv_to_ppc(plv,n), freqs, timepts, n, mean_phase

    else:
        plv,freqs,timepts,n = \
        spike_field_phase_locking_value(spkdata,lfpdata,axis=axis,
                                        return_phase=False, **kwargs)
        return plv_to_ppc(plv,n), freqs, timepts, n



# =============================================================================
# Preprocessing functions
# =============================================================================
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


def _iarange(start=0, stop=0, step=1):
    """
    Implements Numpy arange() with an inclusive endpoint. Same inputs as arange(), same
    output, except ends at stop, not stop - 1 (or more generally stop - step)

    r = _iarange(start=0,stop=0,step=1)
    """
    # TODO  Generalize to deal w/ different input configs (eg iarange(stop))
    if isinstance(step,int):    return np.arange(start,stop+1,step)
    else:                       return np.arange(start,stop+1e-12,step)


def remove_dc(data, axis=None):
    """
    Removes constant DC component of signals, estimated as across-time mean
    for each time series (ie trial,channel,etc.)

    data = remove_dc(data,axis=None)

    ARGS
    data    (...,n_obs,...) ndarray. Raw data to remove DC component of.
            Can be any arbitary shape, with time sampling along axis <axis>

    axis    Int. Data axis corresponding to time.


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

    axis    Int. Data axis corresponding to distinct observations. Default: 0

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
    data    (...,nFreqBands,...) ndarray | xarray DataArray.
            Data with values averaged within each of given frequency bands
    """
    # todo  Deal with more complicated band edge situations (currently assumed non-overlapping)

    # Convert list of frequency band ranges to {'name':freq_range} dict
    if not isinstance(bands,dict):
        bands = {'band_'+str(i_band):frange for i_band,frange in enumerate(bands)}

    # Convert frequency bands into 1-d list of bin edges
    bins = []
    for value in bands.values(): bins.extend(value)

    # xarray: Pool values in bands using DataArray groupby_bins() method
    if HAS_XARRAY and isinstance(data,xr.DataArray):
        dims = np.asarray(data.dims)
        # TODO Generalize to allow frequency dim to be named 'frequency'
        if freqs is None: freqs = data.coords['freq'].values
        # Find 'freq' dimension if not given explicitly
        if axis is None:  axis = (dims == 'freq').nonzero()[0][0]
        # Permute array dims so freq is 1st dim
        if axis != 0:
            temp_dims = np.concatenate(([dims[axis]], dims[dims != 'freq']))
            data = data.transpose(*temp_dims)
        else:
            temp_dims = dims

        # Initialize new DataArray with freq dim = freq bands, indexed by band names
        coords = {dim : data.coords[dim].values for dim in data.coords}
        coords['freq'] = list(bands.keys())
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

        ## DELETE Is cleverer, but only works in restricted cases
        ##Pool values in bands using DataArray groupby_bins() method
        ## Convert frequency bands into 1-d list of bin edges
        #bins = []
        #[bins.extend(value) for value in epochs.values()];
        #
        ## HACK Remove bins "in between" desired freq bands using isel(::2) and
        ##       rename 'freq_bins"'dim back to original 'freq'
        #epoch_data = (data.groupby_bins('time',bins,include_lowest=True)
        #                 .mean(dim='time')
        #                 .isel(time_bins=slice(None,None,2))
        #                 .rename({'time_bins':'time'}))

        ## HACK Remove bins "in between" desired freq bands using isel(::2) and
        ##       rename 'freq_bins"'dim back to original 'freq'
        #if func == 'mean':
        #    band_data = (data.groupby_bins('freq',bins,include_lowest=True)
        #                    .mean(dim='freq')
        #                    .isel(freq_bins=slice(None,None,2))
        #                    .rename({'freq_bins':'freq'}))
        #else:
        #    band_data = (data.groupby_bins('freq',bins,include_lowest=True)
        #                    .apply(func)
        #                    .isel(freq_bins=slice(None,None,2))
        #                    .rename({'freq_bins':'freq'}))
        #
        ## Use frequency band name strings as coordinates for new freq dim
        #band_data.coords['freq'] = list(bands.keys())



# =============================================================================
# Helper functions for circular data (phase) analsysis
# =============================================================================
def circular_subtract_complex(x1, x2):
    """ Circular subtraction of complex numbers, returned in radians """
    return np.angle(x1/x2)

def circular_subtract(theta1, theta2):
    """ Circular subtraction of angles (in radians), returned in radians """
    return np.angle(np.exp(1j*theta1) / np.exp(1j*theta2))


# =============================================================================
# Helper functions for generating single-trial jackknife pseudovalues
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
# Data simulation functions
# =============================================================================
def simulate_mvar(coeffs, cov=None, n_timepts=100, n_trials=1, burnin=100):
    """
    Simulates activity in a network with given connectivity coefficients and
    noise covariance, over the given number of time points, channels, and trials.
    Useful for testing code for measuring synchrony or causality.

    data = simulate_mvar(coeffs,cov=None,n_timepts=100,n_trials=1,burnin=100)

    INPUTS
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

    RETURNS
    data        (n_timepts,n_trials,n_channels) ndarray. Simulated data

    ACKNOWLEDGMENTS Adapted slightly from spectral_connectivity:simulate_MVAR()
    """
    # TODO Add way to easily set random seed for repeatability
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

    INPUTS
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


def test_power(method, plot=False, simulation='DhamalaFig3', n_trials=1000, **kwargs):
    """
    Basic testing for functions estimating time-frequency spectral power 
    
    Generates synthetic LFP data using given network simulation,
    estimates spectrogram using given function, and compares estimated to expected.
    
    means,sems = test_power(method,plot=False,simulation='DhamalaFig3',n_trials=1000, **kwargs)
                              
    INPUTS
    method  String. Name of time-frequency spectral estimation function to test
            
    plot    Bool. Set=True to plot test results. Default: False
    
    simulation  String. Name of canned simulation to generate. Default: 'DhamalaFig3'
            See network_simulation() for details
            
    n_trials Int. Number of trials to include in simulated data. Default: 1000

    **kwargs All other keyword args passed to spectral estimation function given by <method>.
    
    RETURNS
    means   (n_freqs,n_timepts,n_channels) ndarray. Estimated mean spectrogram
    sems    (n_freqs,n_timepts,n_channels) ndarray. SEM of mean spectrogram
    
    ACTION
    Throws an error if any estimated power value is too far from expected value
    If <plot> is True, also generates a plot summarizing expected vs estimated power
    """
    # Set expected peak frequency for given simulation
    if simulation == 'DhamalaFig3':     peak_freq = 40
    
    # Generate simulated LFP data -> (n_timepts,n_trials,n_channels)
    data,smp_rate = network_simulation(simulation=simulation)    
    
    # Compute time-frequency/spectrogram representation of data 
    # -> (n_freqs,n_timepts,n_trials,n_channels)
    spec,freqs,timepts = power_spectrogram(data,smp_rate,axis=0,method=method,**kwargs)
    
    # Compute across-trial mean and SEM of time-frequency data
    # -> (n_freqs,n_timepts,n_channels)
    means = spec.mean(axis=2)
    sems  = spec.std(axis=2,ddof=0) / sqrt(n_trials)

    # Compute mean across all timepoints and channels -> (n_freqs,) frequency marginal
    marginal_means = means.mean(axis=(2,1))
    marginal_sems = sems.mean(axis=(2,1))    
    
    # For wavelets, evaluate and plot frequency on log scale
    if method == 'wavelet':
        freqs_eval      = np.log2(freqs)
        peak_freq_eval  = np.log2(peak_freq)
        fMin            = ceil(log2(freqs[0]))
        fMax            = floor(log2(freqs[-1]))    
        freqTicks       = np.arange(fMin,fMax+1)
        freqTickLabels  = 2**np.arange(fMin,fMax+1)
        
    # For other spectral analysis, evaluate and plot frequency on linear scale        
    else:
        freqs_eval      = freqs
        peak_freq_eval  = peak_freq
        fMin            = ceil(freqs[0]/10.0)*10.0
        fMax            = floor(freqs[-1]/10.0)*10.0                
        freqTicks       = np.arange(fMin,fMax+1,10).astype(int)
        freqTickLabels  = freqTicks        
            
    if plot:
        plt.figure()
        plt.subplot(2,1,1)    
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        dt = np.diff(timepts).mean()
        df = np.diff(freqs_eval).mean()
        xlim = [timepts[0]-dt/2, timepts[-1]+dt/2]
        ylim = [freqs_eval[0]-df/2, freqs_eval[-1]+df/2]
        plt.plot(xlim, [peak_freq_eval,peak_freq_eval], '-', color='r')
        plt.imshow(means[:,:,0], aspect='auto', origin='lower',extent=[*xlim,*ylim])
        plt.yticks(freqTicks,freqTickLabels)
        
        plt.subplot(2,1,2)    
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.errorbar(freqs_eval, marginal_means, 3*marginal_sems, marker='o')                
        plt.plot([peak_freq_eval,peak_freq_eval], plt.ylim(), '-', color='r')        
        plt.xticks(freqTicks,freqTickLabels)
        
    ## Determine if test actually produced the expected values
    # Absolute difference of each sampled frequeny from expected peak frequency
    dist_from_peak = np.abs(freqs_eval - peak_freq_eval)    
    # Indexes needed to sort spectrum by distance from expected peak
    sortIdxs = np.argsort(dist_from_peak)
    sorted_means = marginal_means[sortIdxs]
    pct_max = 100.0 * sorted_means[0:-1]/sorted_means.max()
    
    # Does power decrease monotonically from expected peak frequency (for values > 5% of max)
    assert not ((np.diff(sorted_means) > 0) & (pct_max >= 5.0)).any(), \
        AssertionError("Power does not decrease monotonically from expected peak frequency (%d)" % peak_freq)
                
    return means,sems
    
    
def test_synchrony(method, spec_method='wavelet', plot=False, simulation='DhamalaFig3', n_trials=1000, **kwargs):
    """
    Basic testing for functions estimating bivariate time-frequency synchrony/coherence 
    
    Generates synthetic LFP data using given network simulation,
    estimates t-f synchrony using given function, and compares estimated to expected.
    
    means,sems = test_synchrony(method,spec_method='wavelet',plot=False,
                                simulation='DhamalaFig3',n_trials=1000, **kwargs)
                              
    INPUTS
    method  String. Name of synchrony estimation function to test:
            'PPC' | 'PLV' | 'coherence'

    spec_method  String. Name of spectral estimation function to use to 
            generate time-frequency representation to input into synchrony function
            
    plot    Bool. Set=True to plot test results. Default: False
    
    simulation  String. Name of canned simulation to generate. Default: 'DhamalaFig3'
            See network_simulation() for details
            
    n_trials Int. Number of trials to include in simulated data. Default: 1000

    **kwargs All other keyword args passed to synchrony estimation function given by <method>.
            Can also include args to time-frequency spectral estimation function given by <spec_method>.
    
    RETURNS
    sync    (n_freqs,n_timepts) ndarray. Estimated synchrony
    
    ACTION
    Throws an error if any estimated synchrony value is too far from expected value
    If <plot> is True, also generates a plot summarizing expected vs estimated synchrony
    """    
    # Set expected peak frequency for given simulation
    if simulation == 'DhamalaFig3':     peak_freq = 40
    
    # Generate simulated LFP data -> (n_timepts,n_trials,n_channels)
    data,smp_rate = network_simulation(simulation=simulation)    
    
    # Compute time-frequency/spectrogram representation of data and
    # bivariate measure of synchrony -> (n_freqs,n_timepts)
    sync,freqs,timepts = synchrony(data[:,:,0], data[:,:,1], axis=1, method=method,
                                   spec_method=spec_method, smp_rate=smp_rate,
                                   time_axis=0, **kwargs)
    
    # Compute mean across all timepoints -> (n_freqs,) frequency marginal
    marginal_sync = sync.mean(axis=1)
    
    # For wavelets, evaluate and plot frequency on log scale
    if spec_method == 'wavelet':
        freqs_eval      = np.log2(freqs)
        peak_freq_eval  = np.log2(peak_freq)
        fMin            = ceil(log2(freqs[0]))
        fMax            = floor(log2(freqs[-1]))    
        freqTicks       = np.arange(fMin,fMax+1)
        freqTickLabels  = 2**np.arange(fMin,fMax+1)
        
    # For other spectral analysis, evaluate and plot frequency on linear scale        
    else:
        freqs_eval      = freqs
        peak_freq_eval  = peak_freq
        fMin            = ceil(freqs[0]/10.0)*10.0
        fMax            = floor(freqs[-1]/10.0)*10.0                
        freqTicks       = np.arange(fMin,fMax+1,10).astype(int)
        freqTickLabels  = freqTicks        
            
    if plot:
        plt.figure()
        plt.subplot(2,1,1)    
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        dt = np.diff(timepts).mean()
        df = np.diff(freqs_eval).mean()
        xlim = [timepts[0]-dt/2, timepts[-1]+dt/2]
        ylim = [freqs_eval[0]-df/2, freqs_eval[-1]+df/2]
        plt.plot(xlim, [peak_freq_eval,peak_freq_eval], '-', color='r')
        plt.imshow(sync, aspect='auto', origin='lower',extent=[*xlim,*ylim])
        plt.yticks(freqTicks,freqTickLabels)
        
        plt.subplot(2,1,2)    
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.plot(freqs_eval, marginal_sync)                
        plt.plot([peak_freq_eval,peak_freq_eval], plt.ylim(), '-', color='r')        
        plt.xticks(freqTicks,freqTickLabels)
        
    ## Determine if test actually produced the expected values
    # Absolute difference of each sampled frequeny from expected peak frequency
    dist_from_peak = np.abs(freqs_eval - peak_freq_eval)
    # Does synchrony peak at expected value?
    assert marginal_sync.argmax() == dist_from_peak.argmin(), \
        AssertionError("Synchrony does not peak at frequency (%d)" % peak_freq)        
    
    # # Indexes needed to sort spectrum by distance from expected peak
    # sortIdxs = np.argsort(dist_from_peak)
    # sorted_sync = marginal_sync[sortIdxs]
    # pct_max = 100.0 * sorted_sync[0:-1]/sorted_sync.max()
        
    # Does synchrony decrease monotonically from expected peak frequency (for values > 5% of max)
    # assert not ((np.diff(sorted_sync) > 0) & (pct_max >= 5.0)).any(), \
    #     AssertionError("Synchrony does not decrease monotonically from expected peak frequency (%d)" % peak_freq)
                
    return sync   
    
        
# =============================================================================
# Data reshaping helper functions
# =============================================================================
def _index_axis(data, axis, idxs):
    """ Utility to dynamically index into a arbitrary axis of an ndarray """
    # Generate list of slices, with ':' for all axes except <idxs> for <axis>
    slices = _axis_slices(axis,idxs,data.ndim)

    # Use slices to index into data, and return sliced data
    return data[slices]


def _axis_slices(axis, idxs, ndim):
    """
    Generate list of slices, with ':' for all axes except <idxs> for <axis>,
    to use for dynamic indexing into arbitary axis of array
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
    data    (n_obs,...) ndarray. Data array where zxis 0 is observations (trials),
            and rest of axis(s) are any independent data series.

    axis    Int. Axis of data to move to axis 0 for subsequent analysis. Default: 0

    RETURNS
    data    (n_obs,n_series) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where n_series = prod(shape[1:])

    data_shape Tuple. Original shape of data array
    """
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
        last_dim = data_ndim - 1
        if axis != last_dim: data = np.moveaxis(data,axis,last_dim)

        # If data array data has > 2 dims, keep axis 0 and unwrap other dims into a matrix
        if data_ndim > 2:   data = np.reshape(data,(-1,data_shape[axis]),order='C').T

    return data, data_shape


def _unreshape_data(data, data_shape, axis=0):
    """
    Reshapes data array from unwrapped 2D (matrix) form back to ~ original
    multi-dimensional form

    data = _unreshape_data(data,data_shape,axis=0)

    ARGS
    data    (axis_len,n_series) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where n_series = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    axis    Int. Axis of original data corresponding to distinct observations,
            which has become axis 0, but will be permuted back to original axis.
            Default: 0

    RETURNS
    data       (axis_len,...) ndarray. Data array reshaped back to original shape
    """
    data_shape  = np.asarray(data_shape)

    data_ndim   = len(data_shape) # Number of dimensions in original data
    axis_len     = data.shape[0]   # Length of dim 0 (will become dim <axis> again)

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    # (but with length of dimension <axis> = <axis_length>)
    if data_ndim > 2:
        # Reshape data -> (axis_len,<original shape w/o <axis>>)
        shape = (axis_len,*data_shape[np.arange(data_ndim) != axis])
        data = np.reshape(data,shape,order='F')

    # If observation axis wasn't 0, permute axis back to original position
    if axis != 0: data = np.moveaxis(data,0,axis)

    return data


def _unreshape_data_newaxis(data,data_shape,axis=0):
    """
    Reshapes data array from unwrapped form back to ~ original
    multi-dimensional form in special case where a new frequency axis was
    inserted before time axis (<axis>)

    data = _unreshape_data_newaxis(data,data_shape,axis=0)

    ARGS
    data    (axis_len,n_series) ndarray. Data array w/ all axes > 0 unwrapped into
            single dimension, where n_series = prod(shape[1:])

    data_shape Tuple. Original shape of data array

    axis    Int. Axis of original data corresponding to distinct observations,
            which has become axis 1, but will be permuted back to original axis.
            Default: 0

    RETURNS
    data       (n_freqs,n_timepts,...) ndarray. Data array reshaped back to original shape
    """
    # todo  would be great if this could be generalized...
    data_shape  = np.asarray(data_shape)

    data_ndim   = len(data_shape) # Number of dimensions in original data
    n_freqs      = data.shape[0]
    n_timepts    = data.shape[1]

    # If data array data had > 2 dims, reshape matrix back into ~ original shape
    if len(data_shape) > 2:
        shape   = (n_freqs, n_timepts, *data_shape[np.arange(data_ndim) != axis])
        data    = np.reshape(data,shape,order='F')

    # If observation axis wasn't 0, permute axis back to original position
    if axis != 0:
        axes    = (*data_shape[:axis], 0, 1, *data_shape[(axis+1):])
        data    = np.transpose(data,axes=axes)

    return data


def _remove_buffer(data, buffer, axis=1):
    """
    Removes a temporal buffer (eg of zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    data = _remove_buffer(data,buffer,axis=1)

    ARGS
    data    Data array where a buffer has been appended on both ends of time dimension.
            Can be any arbitrary size, typically (n_freqs,n_timepts+2*buffer,...).
    buffer  Scalar. Length (number of samples) of buffer appended to each end.
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
# Other helper functions
# =============================================================================
def _infer_data_type(data):
    """ Infers type of data signal given -- 'raw' (real) | 'spectral' (complex) """
    if np.isrealobj(data):  return 'raw'
    else:                   return 'spectral'


def _next_power_of_2(n):
    """
    Rounds x up to the next power of 2 (smallest power of 2 greater than n)
    """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


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

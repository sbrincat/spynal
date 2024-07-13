# -*- coding: utf-8 -*-
""" Utility functions for LFP/EEG/continuous data and spectral analysis """
from math import pi, ceil, log2
import numpy as np
import scipy as sp

from multiprocessing import cpu_count
from scipy.stats import norm

from spynal.utils import axis_index_slices, set_random_seed, interp1

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pyfftw.interfaces.scipy_fftpack as fftw
    HAS_FFTW = True
    # Set default arguments for pyfftw functions: Fast planning, use all available threads
    FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE', 'threads': cpu_count()}
except ImportError:
    HAS_FFTW = False


# =============================================================================
# Preprocessing/helper functions
# =============================================================================
def next_power_of_2(n):
    """ Find next power of 2 (smallest power of 2 greater than n) """
    # todo  Think about switching this to use scipy.fftpack.next_fast_len
    return 1 if n == 0 else 2**ceil(log2(n))


def get_freq_sampling(smp_rate, n_fft, freq_range=None, pad=False, two_sided=False):
    """
    Return frequency sampling vector (axis) for a given FFT-based computation

    Parameters
    ----------
    smp_rate : scalar
        Data sampling rate (Hz)

    n_fft : scalar
        Number of samples (timepoints) in FFT output (excluding any padding due to `pad`)

    freq_range : array-like, shape=(2,) or scalar, default: all frequencies from FFT
        Range of frequencies to retain in output, either given as an explicit [low,high]
        range or just a scalar giving the highest frequency to return.

    pad : bool, default: False
        If True, `n_fft` is increased to length = the next power of 2

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
    # If requested, pad data to next power of 2 > input n
    if pad: n_fft = next_power_of_2(n_fft)

    freqs   = np.fft.fftfreq(n_fft, d=1/smp_rate) # All possible frequencies

    # If no range requested, keep all frequencies
    if freq_range is None:
        # Include both positive and negative frequencies
        if two_sided:
            freq_bool = np.ones((n_fft,),dtype=bool)
        # Limit to positive frequencies
        else:
            if n_fft % 2 == 0:  n = (n_fft/2 + 1, n_fft/2 - 1)
            else:               n = ((n_fft-1)/2, (n_fft-1)/2 + 1)
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
            raise ValueError("freq_range must be given as 2-length vector = [min,max]"
                             "or scalar max frequency")

        # Limit to positive frequencies. Special case to also get f = (-)smp_rate/2
        if not two_sided:
            freq_bool = freq_bool & ((freqs >= 0) | np.isclose(freqs,-smp_rate/2))

    # Extract only desired freqs from sampling vector
    freqs = freqs[freq_bool]

    # Again, special case to deal with (-)smp_rate/2
    if not two_sided: freqs = np.abs(freqs)

    return freqs, freq_bool


def get_freq_length(smp_rate, n_fft, freq_range=None, pad=False, two_sided=False):
    """
    Return length of frequency sampling vector (axis) for a given FFT-based computation.

    Wrapper around :func:`get_freq_sampling` that simply returns the length of the list
    of sampled freqs. Input arguments are identical; see there for details.

    Returns
    -------
    n_freqs : int
        Length of frequency sampling vector for given FFT, which could be used to set axis
        length for array to hold FFT results.
    """
    freqs,_ = get_freq_sampling(smp_rate, n_fft, freq_range=freq_range,
                                pad=pad, two_sided=two_sided)

    return len(freqs)


# =============================================================================
# Core FFT/IFFT functions
# =============================================================================
def fft(data, n_fft=None, axis=0, fft_method=None):
    """
    Compute 1d discrete Fast Fourier transform along given array axis, using given method.

    Thin wrapper around multiple low-level FFT implementations, which may be chosen based on
    performance and availability (in our hands, torch is fastest for virtually all tests).

    Parameters
    ----------
    data : ndarray, shape=(...,n_samples,...)
        Data to compute 1d FFT on. Shape is arbitrary; FFT is performed along `axis`
        (usually corresponding to time) independently across any/all other array axes
        (which might be data channels, trials, etc.).

    n_fft : int, default: data.shape[axis]
        Length of the transformed axis of the output. If smaller than the length of the input,
        the input is cropped. If larger, the input is zero-padded with zeros. If not given,
        the length of the input along the axis specified by `axis` is used.

    axis : int, default: 0 (1st array axis)
        Axis over which to compute the FFT. If not given, the first axis is used.

    fft_method : str, default: 'torch' (if available)
        Which underlying low-level FFT implementation to use. Options:

        - 'torch' : Torch's FFT on the GPU, using :func:torch.fft.fft.
            Across a range of tests, this is by far the fastest method.
        - 'fftw' : FFTW library's optimized FFT, using :func:pyfftw.interfaces.scipy_fftpack.fft
        - 'numpy' : Numpy's FFT implementation, using :func:np.fft.fft

        Depending on what is installed, default order is: torch -> fftw -> numpy

    Returns
    -------
    spec : ndarray, shape=(...,n_fft,...)
        1d Fast-Fourier Transformed data, along `axis`. Same shape as `data`, except `axis` now
        has length `n_fft`.
    """
    if n_fft is None: n_fft = data.shape[axis]

    if fft_method is None:
        if HAS_TORCH:   fft_method = 'torch'
        elif HAS_FFTW:  fft_method = 'fftw'
        else:           fft_method = 'numpy'

    if fft_method == 'torch':
        return torch.fft.fft(torch.from_numpy(data), n=n_fft, dim=axis).numpy()

    elif fft_method == 'fftw':
        return fftw.fft(data, n=n_fft, axis=axis, **FFTW_KWARGS_DEFAULT)

    elif fft_method == 'numpy':
        return np.fft.fft(data, n=n_fft, axis=axis)

    elif fft_method == 'scipy':
        return sp.fft.fft(data, n=n_fft, axis=axis)

    else:
        raise ValueError("Unsupported value '%s' set for `fft_method`" % fft_method)


def ifft(spec, n_fft, axis=0, fft_method=None):
    """
    Compute inverse 1d discrete Fast Fourier transform along given array axis, using given method

    Thin wrapper around multiple low-level iFFT implementations, which may be chosen based on
    performance and availability (in our hands, torch is fastest for virtually all tests).

    Parameters
    ----------
    spec : ndarray, shape=(...,n_freqs,...)
        Spectral data  to compute inverse 1d FFT on. Shape is arbitrary; iFFT is performed along
        `axis` (corresponding to frequency), independently across any/all other array axes
        (which might be data channels, trials, etc.).

    n_fft : int, default: spec.shape[axis]
        Length of the transformed axis of the output. If smaller than the length of the input,
        the input is cropped. If larger, the input is zero-padded with zeros. If not given,
        the length of the input along the axis specified by `axis` is used.

    axis : int, default: 0 (1st array axis)
        Axis over which to compute the FFT. If not given, the first axis is used.

    fft_method : str, default: 'torch' (if available)
        Which underlying low-level inverse FFT implementation to use. Options:

        - 'torch' : Torch's FFT on the GPU, using :func:torch.fft.ifft.
            Across a range of tests, this is by far the fastest method.
        - 'fftw' : FFTW's optimized FFT, using :func:pyfftw.interfaces.scipy_fftpack.ifft
        - 'numpy' : Numpy's FFT implementation, using :func:np.fft.ifft

        Depending on what is installed, default order is: torch -> fftw -> numpy

    Returns
    -------
    data : ndarray, shape=(...,n_fft,...)
        1d inverse Fast-Fourier Transformed data, along `axis`. Same shape as `spec`, except `axis`
        (usually corresponding to time) now has length `n_fft`.
    """
    if fft_method is None:
        if HAS_TORCH:   fft_method = 'torch'
        elif HAS_FFTW:  fft_method = 'fftw'
        else:           fft_method = 'numpy'

    if fft_method == 'torch':
        return torch.fft.ifft(torch.from_numpy(spec), n=n_fft, dim=axis).numpy()

    elif fft_method == 'numpy':
        return np.fft.ifft(spec, n=n_fft, axis=axis)

    elif fft_method == 'fftw':
        return fftw.ifft(spec, n=n_fft, axis=axis, **FFTW_KWARGS_DEFAULT)

    elif fft_method == 'scipy':
        return sp.fft.ifft(spec, n=n_fft, axis=axis)

    else:
        raise ValueError("Unsupported value '%s' set for `fft_method`" % fft_method)


# =============================================================================
# Post-processing functions
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
        f0 = interp1(freqs, data, 0, axis=axis, kind='cubic', fill_value='extrapolate')
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
# Data simulation functions
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

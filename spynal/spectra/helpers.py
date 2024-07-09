# -*- coding: utf-8 -*-
""" Helper functions for spectra module for spectral analysis """
from warnings import warn
from math import floor, ceil
import numpy as np
import scipy as sp

from multiprocessing import cpu_count
# from pyfftw.interfaces.scipy_fftpack import fft, ifft # ~ 46/16 s on benchmark

try:
    import pyfftw.interfaces.scipy_fftpack as fftw
    HAS_FFTW = True
    # Set default arguments for pyfftw functions: Fast planning, use all available threads
    _FFTW_KWARGS_DEFAULT = {'planner_effort': 'FFTW_ESTIMATE',
                            'threads': cpu_count()}
except:
    HAS_FFTW = False

try:
    import mkl_fft
    HAS_MKL = True
except:
    HAS_MKL = False

try:
    import torch
    HAS_TORCH = True
except:
    HAS_TORCH = False


# from numpy.fft import fft,ifft        # ~ 15 s on benchmark
# from scipy.fftpack import fft,ifft    # ~ 11 s on benchmark
# from mkl_fft import fft,ifft    # ~ 15.2 s on benchmark
# from pyfftw.interfaces.scipy_fftpack import fft, ifft # ~ 46/16 s on benchmark


def _fft(data, nfft, axis=0, fft_method=None):
    """ FFT """
    if fft_method is None:
        if HAS_TORCH:   fft_method = 'torch'
        elif HAS_FFTW:  fft_method = 'pyfftw'
        elif HAS_MKL:   fft_method = 'mkl_fft'
        else:           fft_method = 'scipy'

    if fft_method == 'torch':
        d = torch.from_numpy(data)
        spec = torch.fft.fft(d.permute(*torch.arange(d.ndim - 1, -1, -1)), n=nfft)
        spec = spec.permute(*torch.arange(spec.ndim - 1, -1, -1))
        spec = spec.numpy()
    elif fft_method == 'pyfftw':
        spec = fftw.fft(data, n=nfft, axis=axis, **_FFTW_KWARGS_DEFAULT)
    elif fft_method == 'mkl_fft':
        spec = mkl_fft.fft(data, n=nfft, axis=axis)
    elif fft_method == 'scipy':
        spec = sp.fftpack.fft(data, n=nfft, axis=axis)
    else:
        spec = np.fft.fft(data, n=nfft, axis=axis)

    return spec


def _ifft(spec, nfft, axis=0, fft_method=None):
    """ Inverse FFT """
    if fft_method is None:
        if HAS_TORCH:   fft_method = 'torch'
        elif HAS_FFTW:  fft_method = 'pyfftw'
        elif HAS_MKL:   fft_method = 'mkl_fft'
        else:           fft_method = 'scipy'

    if fft_method == 'torch':
        s = torch.from_numpy(spec)
        data = torch.fft.ifft(s.permute(*torch.arange(s.ndim - 1, -1, -1)), n=nfft, axis=1)
        data = data.permute(*torch.arange(data.ndim - 1, -1, -1))
        data = data.numpy()
    elif fft_method == 'pyfftw':
        data = fftw.ifft(spec, n=nfft, axis=axis, **_FFTW_KWARGS_DEFAULT)
    elif fft_method == 'mkl_fft':
        data = mkl_fft.ifft(spec, n=nfft, axis=axis)
    elif fft_method == 'scipy':
        data = sp.fftpack.ifft(spec, n=nfft, axis=axis)
    else:
        data = np.fft.ifft(spec, n=nfft, axis=axis)

    return data


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


def _str_to_pool_func(func):
    """ Convert string specifier to callable pooling function """
    # If it's already a callable, return as-is
    if callable(func): return func

    assert isinstance(func,str), \
        TypeError("Unsupported type '%s' for <func>. Use string or function" % type(func))

    func = func.lower()
    if func == 'mean':  return lambda x: np.mean(x, axis=0)
    elif func == 'sum': return lambda x: np.sum(x, axis=0)
    else:
        raise ValueError("Unsupported value '%s' for func. Set='mean'|'sum'" % func)

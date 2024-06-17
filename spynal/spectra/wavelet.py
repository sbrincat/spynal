# -*- coding: utf-8 -*-
""" Continuous-wavelet-based spectral analysis """
from math import sqrt, pi, ceil
import numpy as np

from spynal.utils import standardize_array
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.spectra.preprocess import remove_dc
from spynal.spectra.utils import next_power_of_2, complex_to_spec_type
from spynal.spectra.helpers import fft, ifft, _FFTW_KWARGS_DEFAULT, \
                                   _undo_standardize_array_newaxis
try:
    import torch
except:
    pass

def wavelet_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex', freqs=None,
                     removeDC=True, wavelet='morlet', wavenumber=6, pad=False, buffer=0, **kwargs):
    """
    Compute continuous wavelet transform of data, then averages across timepoints to
    reduce it down to a frequency spectrum.

    Not really the best way to compute 1D frequency spectra, but included for completeness

    Only parameters differing from :func:`.spectrum` are described here.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqs,), default: 2**np.arange(1,7.5,0.25)
        Set of desired wavelet frequencies. Default value logarithmically samples from 2-152
        in 1/4 octaves, but log sampling is not required.

    wavelet : {'morlet'}, default: 'morlet'
        Name of wavelet type. Currently only 'morlet' is supported.

    wavenumber : int, default: 6
        Wavelet wave number parameter ~ number of oscillations in each wavelet.
        Must be >= 6 to meet "admissibility constraint".

    buffer : float, default: 0 (no buffer)
        Time (s) to trim off each end of time dimension of data.
        Removes symmetric buffer previously added (outside of here) to prevent edge effects.

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs,...), dtype=complex or float.
        Wavelet-derived spectrum of data.
        Same shape as data, with frequency axis replacing time axis
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs, ndarray, shape=(n_freqs,)
        List of frequencies in `spec` (in Hz)
    """
    if freqs is None: freqs = 2.0**np.arange(1,7.5,0.25)
    if axis < 0: axis = data.ndim + axis

    spec, freqs, _ = wavelet_spectrogram(data, smp_rate, axis=axis, data_type=data_type,
                                         spec_type=spec_type, freqs=freqs, removeDC=removeDC,
                                         wavelet=wavelet, wavenumber=wavenumber, pad=pad,
                                         buffer=buffer, **kwargs)

    # Take mean across time axis (which is now shifted +1 b/c of frequency axis)
    return spec.mean(axis=axis+1), freqs


def wavelet_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex', freqs=None,
                        removeDC=True, wavelet='morlet', wavenumber=6, pad=False, buffer=0,
                        downsmp=1, use_torch=False, **kwargs):
    """
    Compute continuous time-frequency wavelet transform of data at given frequencies.

    Only parameters differing from :func:`.spectrogram` are described here.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqs,), default: 2**np.arange(1,7.5,0.25)
        Set of desired wavelet frequencies. Default value logarithmically samples from 2-152
        in 1/4 octaves, but log sampling is not required.

    wavelet : {'morlet'}, default: 'morlet'
        Name of wavelet type. Currently only 'morlet' is supported.

    wavenumber : int, default: 6
        Wavelet wave number parameter ~ number of oscillations in each wavelet.
        Must be >= 6 to meet "admissibility constraint".

    buffer : float, default: 0 (no buffer)
        Time (s) to trim off each end of time dimension of data.
        Removes symmetric buffer previously added (outside of here) to prevent edge effects.

    downsmp: int, default: 1 (no downsampling)
        Factor to downsample time sampling by (after spectral analysis).
        eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs,n_timepts_out,...), dtype=complex or float.
        Wavelet time-frequency spectrogram of data, transformed to requested spectral type.
        Same shape as data, with frequency axis prepended before time, and time axis
        possibly reduces via downsampling.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqs,) ndarray
        List of frequencies in `spec` (in Hz)

    timepts : ndarray, shape=(n_timepts_out,)
        List of timepoints in `spec` (in s, referenced to start of data).

    References
    ----------
    Torrence & Compo 1998 https://doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2
    """
    if freqs is None: freqs = 2.0**np.arange(1,7.5,0.25)
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
    else:       n_fft = next_power_of_2(n_timepts_in)

    # Compute set of Fourier-transformed wavelet functions (if not already given)
    if isinstance(wavelet,str):
        wavelets_fft = compute_wavelets(n_fft,smp_rate,freqs=freqs,
                                        wavelet=wavelet,wavenumber=wavenumber,
                                        do_fft=True)
    else:
        wavelets_fft = wavelet

    if removeDC: data = remove_dc(data,axis=0)

    # Compute FFT of data
    if use_torch:
        t = torch.from_numpy(data)
        data = torch.fft.fft(t.permute(*torch.arange(t.ndim - 1, -1, -1)), n=n_fft)
        data = data.permute(*torch.arange(data.ndim - 1, -1, -1))
        data = data.numpy()
    else:
        data = fft(data, n=n_fft,axis=0, **_FFTW_KWARGS_DEFAULT)

    # Reshape data -> (1,n_timepts,n_series) (insert axis 0 for wavelet scales/frequencies)
    # Reshape wavelets -> (n_freqs,n_timepts,1) to broadcast
    #  (except for special case of 1D data with only a single time series)
    data = data[np.newaxis,...]
    if data.ndim == 3: wavelets_fft = wavelets_fft[:,:,np.newaxis]

    # Convolve data with wavelets (multiply in Fourier domain)
    # -> inverse FFT to get wavelet transform
    if use_torch:
        t = torch.from_numpy(data*wavelets_fft)
        spec = torch.fft.ifft(t.permute(*torch.arange(t.ndim - 1, -1, -1)), n=n_fft, axis=1)
        spec = spec.permute(*torch.arange(spec.ndim - 1, -1, -1))
        spec = spec.numpy()[:,time_idxs_out,...]
    else:
        spec = ifft(data*wavelets_fft, n=n_fft,axis=1, **_FFTW_KWARGS_DEFAULT)[:,time_idxs_out,...]

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)

    spec    = _undo_standardize_array_newaxis(spec,data_shape,axis=axis)

    timepts = time_idxs_out.astype(float)/smp_rate  # Convert time sampling from samples -> s

    return spec, freqs, timepts


def compute_wavelets(n, smp_rate, freqs=None, wavelet='morlet', wavenumber=6, do_fft=False):
    """
    Compute set of (Fourier transformed) wavelets for use in wavelet spectral analysis

    Parameters
    ----------
    n : int
        Total number of samples (time points) in analysis, including any padding.

    smp_rate scalar
        Data sampling rate (Hz)

    freqs : array-like, shape=(n_freqs,), default: 2**np.arange(1,7.5,0.25)
        Set of desired wavelet frequencies. Default value logarithmically samples from 2-152
        in 1/4 octaves, but log sampling is not required.

    wavelet : {'morlet'}, default: 'morlet'
        Name of wavelet type. Currently only 'morlet' is supported.

    wavenumber : int, default: 6
        Wavelet wave number parameter ~ number of oscillations in each wavelet.
        Must be >= 6 to meet "admissibility constraint".

    do_fft: bool, default: False
        If True, returns Fourier transform of wavelet functions.
        If False, returns original time-domain functions.

    Returns
    -------
    wavelets : ndarray, shape=(n_freqs,n_timepts)
        Computed set of wavelet functions at multiple frequencies/scales.
        (either the time domain wavelets or theirFourier transform, depending on `do_fft`)

    References
    ----------
    Torrence & Compo 1998 https://doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2
    """
    if freqs is None:   freqs = 2.0**np.arange(1,7.5,0.25)
    else:               freqs = np.asarray(freqs)
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
        raise ValueError("Unsupported value '%s' given for <wavelet>."
                         "Currently only 'Morlet' suppported")

    return wavelets


def wavelet_bandwidth(freqs, wavelet='morlet', wavenumber=6, full=True):
    """
    Return frequency and time bandwidths for set of wavelets at given frequencies

    Parameters
    ----------
    freqs : array-like, shape=(n_freqs,)
        Set of wavelet center frequencies.

    wavelet : {'morlet'}, default: 'morlet'
        Name of wavelet type. Currently only 'morlet' is supported.

    wavenumber : int, default: 6
        Wavelet wave number parameter ~ number of oscillations in each wavelet.

    full : bool, default: True
        If True, return full-bandwidths. If False, return half-bandwidths.

    Returns
    -------
    freq_widths : ndarray, shape=(n_freqs,)
        Frequency bandwidth (Hz) for each given frequency

    time_widths : ndarray, shape=(n_freqs,)
        Time bandwidth (s) for each given frequency
    """
    wavelet = wavelet.lower()
    freqs = np.asarray(freqs)

    if wavelet == 'morlet':
        freq_widths = freqs / wavenumber
        time_widths = 1 / (2*pi*freq_widths)

    else:
        raise ValueError("Unsupported value '%s' given for <wavelet>."
                         "Currently only 'Morlet' suppported")

    # Convert half-bandwidths to full-bandwidths
    if full:
        freq_widths = 2 * freq_widths
        time_widths = 2 * time_widths

    return freq_widths, time_widths


def wavelet_edge_extent(freqs, wavelet='morlet', wavenumber=6):
    """
    Return temporal extent of edge effects for set of wavelets at given frequencies

    Compute time period over which edge effects might effect output of wavelet transform,
    and over which the effects of a single spike-like artifact in data will extend.

    Computed as time for wavelet power to drop by a factor of exp(−2), ensuring that
    edge effects are "negligible" beyond this point.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqs,)
        Set of wavelet center frequencies

    wavelet : {'morlet'}, default: 'morlet'
        Name of wavelet type. Currently only 'morlet' is supported.

    wavenumber : int, default: 6
        Wavelet wave number parameter ~ number of oscillations in each wavelet.

    Returns
    -------
    edge_extent : ndarray, shape=(n_freqs,)
        Time period (s) over which edge effects extend for each given frequency

    References
    ----------
    Torrence & Compo https://doi.org/10.1175/1520-0477(1998)079%3C0061:APGTWA%3E2.0.CO;2 Sxn.3g
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
        raise ValueError("Unsupported value '%s' given for <wavelet>."
                         "Currently only 'Morlet' suppported")

    return edge_extent

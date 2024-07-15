# -*- coding: utf-8 -*-
""" Multitaper spectral analysis """
from math import floor, ceil, sqrt
import numpy as np

from scipy.signal.windows import dpss

from spynal.utils import iarange
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.spectra.preprocess import remove_dc
from spynal.spectra.utils import next_power_of_2, get_freq_sampling, get_freq_length, \
                                 complex_to_spec_type, phase, axis_index_slices
from spynal.spectra.utils import fft
from spynal.spectra.helpers import _extract_triggered_data


def multitaper_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freq_range=None, removeDC=True, freq_width=4, n_tapers=None,
                        keep_tapers=False, tapers=None, pad=True, fft_method=None, **kwargs):
    """
    Multitaper Fourier spectrum computation for continuous (eg LFP) or point process (spike) data

    Multitaper methods project the data onto orthogonal Slepian (DPSS) "taper" functions, which
    increases the data's effective signal-to-noise. It allows a principled tradeoff btwn time
    resolution (data.shape[axis]), frequency resolution (`freq_width`), and the number of taper
    functions (`n_tapers`), which determines the signal-to-noise boost.

    Note: By default, data is zero-padded to the next power of 2 greater than its input length.
    This will change the frequency sampling (number of freqs and exact freqs sampled) from what
    would be obtained from the original raw data, but can be skipped by inputtng pad=False.

    Only parameters differing from :func:`.spectrum` are described here.

    Parameters
    ----------
    freq_range : array-like, shape=(2,) or scalar, default: all frequencies from FFT (0-smp_rate/2)
        Range of frequencies to keep in output, either given as an explicit [low,high]
        range or just a scalar giving the highest frequency to return.

    freq_width : scalar, default: 4 Hz
        Frequency bandwidth 'W' (Hz).

    n_tapers : scalar, default: (2TW-1)
        Number of tapers to compute. Must be <= 2TW-1, as this is the max number of
        spectrally delimited tapers (and is set as default based on set T,W values).
        Note: Time bandwidth 'T' is set to full length of data.

    tapers : ndarray, shape=(n_win_samples,n_tapers), default: (computed from t/f_range,ntapers)
        Precomputed tapers (as computed by :func:`compute_tapers`).

        Alternative method for explictly setting taper functions.
        Input either `time_width`/`freq_width`/`n_tapers` OR `tapers`.
        If tapers not explicitly input, we compute them from `time_width`/`freq_width`/`n_tapers`.
        If tapers *are* explicitly input, `time_width`/`freq_width`/`n_tapers` are ignored.

    keep_tapers : bool, default: False
        If True, tapers axis is retained in output, between frequency axis and time `axis`.
        If False, output is averaged over tapers, and taper axis is removed

    pad : bool, default: True
        If True, zero-pads data to next power of 2 length

    fft_method : str, default: 'torch' (if available)
        Which underlying FFT implementation to use. Options: 'torch', 'fftw', 'numpy'
        Defaults to torch if it's installed, then to FFTW if pyfftw is installed, then to Numpy.

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs,[n_tapers,]...), dtype=complex or float
        Multitaper spectrum of given type of data. Sampling (time) axis is
        replaced by frequency and taper axes (if `keep_tapers` is True), but
        shape is otherwise preserved.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, dtype=(n_freqs,)
        List of frequencies in `spec` (in Hz)

    References
    ----------
    - Mitra & Pesaran 1999 https://doi.org/10.1016/S0006-3495(99)77236-X
    - Jarvis & Mitra 2001 https://doi.org/10.1162/089976601300014312
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
    else:       n_fft = next_power_of_2(n_timepts)
    # Set frequency sampling vector
    freqs,fbool = get_freq_sampling(smp_rate, n_fft, freq_range=freq_range)

    # Compute DPSS taper functions (if not precomputed)
    if tapers is None:
        tapers = compute_tapers(smp_rate, time_width=n_timepts/smp_rate,
                                freq_width=freq_width, n_tapers=n_tapers)

    # Reshape tapers to (n_timepts,n_tapers) (if not already)
    if (tapers.ndim == 2) and (tapers.shape[1] == n_timepts): tapers = tapers.T
    assert tapers.shape[0] == n_timepts, \
        ValueError("tapers must have same length (%d) as number of timepoints in data (%d)"
                   % (tapers.shape[0],n_timepts))

    # Reshape tapers array to pad end of it w/ singleton dims
    taper_shape  = (*tapers.shape,*np.ones((data.ndim-1,),dtype=int))

    # NOTE: In other toolboxes, the overall mean spike rate is subtracted from spike spectra
    #       Results are identical with just subtracting off DC from spike rate data before fft.
    #       So the former is commented out below, and we just do the latter.
    # SKIP Alternate method for normalizing point-process spectra
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
    spec = fft(data, n_fft=n_fft, axis=0, fft_method=fft_method)

    if data_type != 'spike': spec = spec/smp_rate

    # SKIP Alternate method for normalizing point-process spectra
    # Subtract off the DC component (average spike rate) for point process signals
    # if data_type == 'spike' and removeDC: spec -= taper_fft*mean_rate

    # Extract desired set of frequencies
    spec    = spec[fbool,...]

    # Convert to desired output spectral signal type
    spec    = complex_to_spec_type(spec,spec_type)

    # Compute mean across tapers if requested
    if not keep_tapers:
        if spec_type == 'phase':    spec = phase(np.exp(1j*spec).mean(axis=1)) # circular mean
        else:                       spec = spec.mean(axis=1)

    # If observation axis wasn't 0, permute (freq,tapers) back to original position
    if axis != 0:
        if keep_tapers: spec = np.moveaxis(spec,[0,1],[axis,axis+1])
        else:           spec = np.moveaxis(spec,0,axis)

    return spec, freqs


def multitaper_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                           freq_range=None, removeDC=True, time_width=0.5, freq_width=4,
                           n_tapers=None, spacing=None, tapers=None, keep_tapers=False,
                           pad=True, fft_method=None, max_chunk_size=1e9, **kwargs):
    """
    Compute multitaper time-frequency spectrogram for continuous (eg LFP)
    or point process (eg spike) data

    Multitaper methods project the data onto orthogonal Slepian (DPSS) "taper" functions, which
    increases the data's effective signal-to-noise. It allows a principled tradeoff btwn time
    resolution (data.shape[axis]), frequency resolution (freq_width), and the number of taper
    functions (n_tapers), which determines the signal-to-noise increase.

    Note: By default, data is zero-padded to the next power of 2 greater than its input length.
    This will change the frequency sampling (number of freqs and exact freqs sampled) from what
    would be obtained from the original raw data, but can be skipped by inputtng pad=False.

    Only parameters differing from :func:`.spectrogram` are described here.

    Parameters
    ----------
    freq_range : array-like, shape=(2,) or scalar, default: all frequencies from FFT (0-smp_rate/2)
        Range of frequencies to keep in output, either given as an explicit [low,high]
        range or just a scalar giving the highest frequency to return.

    time_width : scalar, default: 0.5 (500 ms)
        Time bandwidth 'T' (s). Width of sliding time window is set equal to this.

    freq_width : scalar, default: 4 Hz
        Frequency bandwidth 'W' (Hz).

    n_tapers : scalar, default: (2TW-1)
        Number of tapers to compute. Must be <= 2TW-1, as this is the max number of
        spectrally delimited tapers (and is set as default based on set T,W values).

    spacing : scalar, default: `time_width` (so each window exactly non-overlapping)
        Spacing between successive sliding time windows (s)

    tapers : ndarray, shape=(n_win_samples,n_tapers), default: (computed from t/f_range,ntapers)
        Precomputed tapers (as computed by :func:`compute_tapers`).

        Alternative method for explicitly setting taper functions.
        Input either `time_width`/`freq_width`/`n_tapers` OR `tapers`.
        If tapers not explicitly input, we compute them from `time_width`/`freq_width`/`n_tapers`.
        If tapers *are* explicitly input, `time_width`/`freq_width`/`n_tapers` are ignored.

    keep_tapers : bool, default: False
        If True, tapers axis is retained in output, between frequency axis and time `axis`.
        If False, output is averaged over tapers, and taper axis is removed

    pad : bool, default: True
        If True, zero-pads data to next power of 2 length

    fft_method : str, default: 'torch' (if available)
        Which underlying FFT implementation to use. Options: 'torch', 'fftw', 'numpy'
        Defaults to torch if it's installed, then to FFTW if pyfftw is installed, then to Numpy.

    max_chunk_size : int, default: TODO

    Returns
    -------
    spec : ndarray, shape=(...,n_freqs[,n_tapers],n_timewins,...), dtype=complex or float
        Multitaper time-frequency spectrogram of data.
        Sampling (time) axis is replaced by frequency, taper (if keep_tapers=True),
        and time window axes but shape is otherwise preserved.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqs,)
        List of frequencies in `spec` (in Hz)

    timepts : ndarray, shape=(n_timewins,...)
        List of timepoints in `spec` (in s, referenced to start of data).
        Timepoints here are centers of each time window.

    References
    ----------
    - Mitra & Pesaran 1999 https://doi.org/10.1016/S0006-3495(99)77236-X
    - Jarvis & Mitra 2001 https://doi.org/10.1162/089976601300014312
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
        tapers = compute_tapers(smp_rate, time_width=time_width, freq_width=freq_width,
                                n_tapers=n_tapers)
    n_tapers = tapers.shape[-1]

    # Set up parameters for data time windows
    # Set window starts to range from time 0 to time n - window width
    win_starts  = iarange(0, n_timepts/smp_rate - window, spacing)
    # Set sampled timepoint vector = center of each window
    timepts     = win_starts + window/2.0
    n_timepts   = len(timepts)

    # Determine size in memory of output spectral data, to see if we need to break into chunks
    n_per_win = np.round(window*smp_rate).astype(int)
    data_size_per_t = n_per_win * np.prod(data.shape[1:]) * data.itemsize
    data_size_total = data_size_per_t * n_timepts

    # If data is small enough, just compute spectrogram of it all in one step
    if data_size_total <= max_chunk_size:
        # Extract time-windowed version of data -> (n_timepts_per_win,n_timewins,n_dataseries)
        data = _extract_triggered_data(data, smp_rate, win_starts, [0,window])

        # Do multitaper analysis on windowed data
        spec, freqs = multitaper_spectrum(data, smp_rate, axis=0, data_type=data_type,
                                          spec_type=spec_type, freq_range=freq_range,
                                          tapers=tapers, pad=pad, removeDC=removeDC,
                                          keep_tapers=keep_tapers, fft_method=fft_method, **kwargs)

    # For larger data that might saturate RAM, break into temporal chunks for spectral analysis
    else:
        # Number of timepoints (windows) who's memory size fits in 1 chunk
        n_timepts_per_chunk = int(floor(max_chunk_size / data_size_per_t))
        # Resulting number of chunks in total data (round up to get final, possibly partial, chunk)
        n_chunks = int(ceil(n_timepts / n_timepts_per_chunk))

        n_fft = np.round(np.asarray(window)*smp_rate).astype(int)
        n_freqs = get_freq_length(smp_rate, n_fft, freq_range=freq_range, pad=pad)
        if keep_tapers:
            spec = np.empty((n_freqs,n_tapers,len(timepts),*data.shape[1:]), dtype=complex)
        else:
            spec = np.empty((n_freqs,len(timepts),*data.shape[1:]), dtype=complex)

        for i_chunk in range(n_chunks):
            # Time indexes for current chunk. Truncate final chunk at end of data.
            if ((i_chunk+1)*n_timepts_per_chunk - 1) < n_timepts:
                idxs = slice(i_chunk*n_timepts_per_chunk, (i_chunk+1)*n_timepts_per_chunk)
            else:
                idxs = slice(i_chunk*n_timepts_per_chunk, n_timepts)
            # Slices to index into time axis of `spec`, with ':' on all other axes    
            slices = axis_index_slices(2 if keep_tapers else 1, idxs, spec.ndim)

            # Extract time-windowed version of data -> (n_timepts_per_win,n_timewins,n_dataseries)
            data_win = _extract_triggered_data(data, smp_rate, win_starts[idxs], [0,window])

            # Do multitaper analysis on windowed data
            spec[slices], freqs = \
                multitaper_spectrum(data_win, smp_rate, axis=0,data_type=data_type,
                                    spec_type=spec_type, freq_range=freq_range, tapers=tapers,
                                    pad=pad, removeDC=removeDC, keep_tapers=keep_tapers,
                                    fft_method=fft_method, **kwargs)

    # If time axis wasn't 0, permute (freq,tapers,timewin) axes back to original position
    if axis != 0:
        if keep_tapers: spec = np.moveaxis(spec, [0,1,2], [axis,axis+1,axis+2])
        else:           spec = np.moveaxis(spec, [0,1], [axis,axis+1])

    return spec, freqs, timepts



def compute_tapers(smp_rate, time_width=0.5, freq_width=4, n_tapers=None):
    """
    Compute Discrete Prolate Spheroidal Sequence (DPSS) tapers for use in
    multitaper spectral analysis.

    Uses scipy.signal.windows.dpss, but arguments are different here

    Parameters
    ----------
    smp_rate : scalar
        Data sampling rate (Hz)

    time_width : scalar, default: 0.5 (500 ms)
        Time bandwidth 'T' (s). Should match data window length.

    freq_width : scalar, default: 4 Hz
        Frequency bandwidth 'W' (Hz)

    n_tapers : scalar, default: (2TW-1)
        Number of tapers to compute. Must be <= 2TW-1, as this is
        the max number of spectrally delimited tapers.

    Returns
    -------
    tapers : ndarray, shape=(n_samples,n_tapers)
        Computed dpss taper functions (n_samples = T*smp_rate)
    """
    # Time-frequency bandwidth product 'TW' (s*Hz)
    time_freq_prod  = time_width*freq_width

    # Up to 2TW-1 tapers are bounded; this is both the default and max value for n_tapers
    n_tapers_max = floor(2*time_freq_prod - 1)
    if n_tapers is None: n_tapers = n_tapers_max

    assert n_tapers <= n_tapers_max, \
        ValueError("For time-freq product = %.1f, %d tapers are tightly bounded in"
                   "frequency (n_tapers set = %d)"
                   % (time_freq_prod,n_tapers_max,n_tapers))

    # Convert time bandwidth from s to window length in number of samples
    n_samples = int(round(time_width*smp_rate))

    # Compute the tapers for given window length and time-freq product
    # Note: dpss() normalizes by sum of squares; x sqrt(smp_rate)
    #       converts this to integral of squares (see Chronux function dpsschk())
    # Note: You might imagine you'd want sym=False, but sym=True gives same values
    #       as Chronux dpsschk() function...
    return dpss(n_samples, time_freq_prod, Kmax=n_tapers, sym=True, norm=2).T * sqrt(smp_rate)

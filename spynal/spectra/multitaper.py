# -*- coding: utf-8 -*-
""" Multitaper spectral analysis """
from math import floor, sqrt
import numpy as np

from scipy.signal.windows import dpss

from spynal.utils import iarange
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.spectra.preprocess import remove_dc
from spynal.spectra.utils import next_power_of_2, get_freq_sampling, complex_to_spec_type, phase
from spynal.spectra.helpers import fft, _extract_triggered_data, _calc_total_data_size, _calc_spec_shape

try:
    import torch
except:
    pass

def multitaper_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freq_range=None, removeDC=True, freq_width=4, n_tapers=None,
                        keep_tapers=False, tapers=None, pad=True, torch_avail=False, 
                        max_bin_size=1e9, **kwargs):
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
    freqs,fbool = get_freq_sampling(smp_rate,n_fft,freq_range=freq_range)

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
    if torch_avail:
        t = torch.from_numpy(data)
        spec = torch.fft.fft(t.permute(*torch.arange(t.ndim - 1, -1, -1)), n=n_fft)
        spec = spec.permute(*torch.arange(spec.ndim - 1, -1, -1))
        spec = spec.numpy()
    else:
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
                           pad=True, torch_avail=False, max_bin_size=1e9, **kwargs):
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

    # Set up parameters for data time windows
    # Set window starts to range from time 0 to time n - window width
    win_starts  = iarange(0, n_timepts/smp_rate - window, spacing)
    # Set sampled timepoint vector = center of each window
    timepts     = win_starts + window/2.0

    # Extract time-windowed version of data -> (n_timepts_per_win,n_wins,n_dataseries)
        
    data_size_tot = _calc_total_data_size(data, smp_rate, win_starts, [0,window])
    if data_size_tot <= max_bin_size:
         
        data = _extract_triggered_data(data, smp_rate, win_starts, [0,window])

        if removeDC: data = remove_dc(data, axis=0)

        # Do multitaper analysis on windowed data
        # Note: Set axis=0 and removeDC=False bc already dealt with above.
        # Note: Input values for `freq_width`,`n_tapers` are implicitly propagated here via `tapers`   
        spec, freqs = multitaper_spectrum(data, smp_rate, axis=0, data_type=data_type,
                                        spec_type=spec_type, freq_range=freq_range, tapers=tapers,
                                        pad=pad, removeDC=False, keep_tapers=keep_tapers, torch_avail=torch_avail, 
                                         **kwargs)
    else:
        nbins = int(np.floor(data_size_tot / max_bin_size))
        s0 = np.round(np.asarray(window)*smp_rate).astype(int)
        s0 = _calc_spec_shape(s0, smp_rate, freq_range, pad)
        n_tapers_max = np.floor(2*time_width*freq_width - 1)
        if n_tapers is None: 
            ntps = n_tapers_max
        else: 
            ntps = n_tapers
            
        if keep_tapers:
            spec = np.zeros((s0,ntps,len(timepts),*data.shape[1:])) * 1j
        else:
            spec = np.zeros((s0,len(timepts),*data.shape[1:])) * 1j

        step = int(np.ceil(win_starts.shape[0]/nbins))
        i = 0
        while (i+1)+step+1 < ntps:
            i_win_starts = win_starts[i*step:(i+1)*step,...]
            d = _extract_triggered_data(data, smp_rate, i_win_starts, [0,window])
            if removeDC: d = remove_dc(d, axis=0)
            spec[...,i*step:(i+1)*step,:], freqs = multitaper_spectrum(d, smp_rate, axis=0, data_type=data_type,
                                        spec_type=spec_type, freq_range=freq_range, tapers=tapers,
                                        pad=pad, removeDC=False, keep_tapers=keep_tapers, torch_avail=torch_avail, 
                                        **kwargs)
            i += 1
            
        i_win_starts = win_starts[i*step:,...]
        d = _extract_triggered_data(data, smp_rate, i_win_starts, [0,window])
        if removeDC: d = remove_dc(d, axis=0)
        spec[...,i*step:,:], freqs = multitaper_spectrum(d, smp_rate, axis=0, data_type=data_type,
                                    spec_type=spec_type, freq_range=freq_range, tapers=tapers,
                                    pad=pad, removeDC=False, keep_tapers=keep_tapers, torch_avail=torch_avail, 
                                    **kwargs)


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

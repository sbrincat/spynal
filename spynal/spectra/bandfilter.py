# -*- coding: utf-8 -*-
""" Band-pass filtering & Hilbert transform-based spectral analysis """
from math import ceil
from collections import OrderedDict
import numpy as np

from scipy.signal import filtfilt, hilbert, zpk2tf, butter, ellip, cheby1, cheby2

from spynal.utils import standardize_array
from spynal.spikes import _spike_data_type, times_to_bool
from spynal.spectra.preprocess import remove_dc
from spynal.spectra.utils import complex_to_spec_type
from spynal.spectra.helpers import _undo_standardize_array_newaxis


def bandfilter_spectrum(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                        freqs=((2,8),(10,32),(40,100)), removeDC=True,
                        filt='butter', order=4, params=None, buffer=0, **kwargs):
    """
    Computes band-filtered and Hilbert-transformed signal from data
    for given frequency band(s), then reduces it to 1D frequency spectra by averaging across time.

    Not really the best way to compute 1D frequency spectra, but included for completeness.

    Only parameters differing from :func:`.spectrum` are described here.

    NOTE: Can specify filter implictly using (`freqs`,`filt`,`order`) OR explicitly using `params`.
          If `params` is input, `freqs`, `filt`, and `order` are ignored.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqbands,2), default: ((2,8),(10,32),(40,100))
        List of (low,high) cut frequencies for each band to use.
        Set low cut = 0 for low-pass, set high cut >= smp_rate/2 for high-pass,
        otherwise assumes band-pass. Default samples ~ theta, alpha/beta, gamma.

    filt : str, default: 'butter' (Butterworth)
        Name of filter to use. See :func:`set_filter_params` for all options

    order : int, default: 4
        Filter order

    params : dict
        Parameters that explicitly define filter for each freq band.

        Alternative method for explicitly setting parameters defining freq band filters,
        which are precomputed with :func:`set_filter_params` (or elsewhere).
        Input either `freqs`/`filt`/`order` OR params.
        If params are not explicitly input, we compute them from `freqs`/`filt`/`order`.
        If params *are* explicitly input, `freqs`/`filt`/`order` are ignored.

        One of two forms: 'ba' or 'zpk', with key/values as follows:

        - b,a : array-like, shape=(n_freqbands,) of array-like (n_params[band,])
            Numerator `b` and denominator `a` polynomials of the filter for each band

        - z,p,k :
            Zeros, poles, and system gain of the IIR filter transfer function

    buffer : float, default: 0 (no buffer)
        Time (s) to trim off each end of time dimension of data.
        Removes symmetric buffer previously added (outside of here) to prevent
        edge effects.

    **kwargs :
        Any other kwargs passed directly to :func:`set_filter_params`

    Returns
    -------
    spec : ndarray, shape=(...,n_freqbands,...), dtype=complex or floats
        Band-filtered, (optionally) Hilbert-transformed data, transformed to requested spectral
        type, and averaged across the time axis to 1D frequency spectra.
        Same shape as input data, but with frequency axis replacing time axis.
        dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqbands,2)
        List of (low,high) cut frequencies (Hz) for each band used
    """
    _ = kwargs.pop('fft_method',None)   # Align API with other *_spectrum() functions
    if axis < 0: axis = data.ndim + axis

    spec, freqs, _ = bandfilter_spectrogram(data, smp_rate, axis=axis, data_type=data_type,
                                            spec_type=spec_type, freqs=freqs, removeDC=removeDC,
                                            filt=filt, order=order, params=params, buffer=buffer,
                                            **kwargs)

    # Take mean across time axis (which is now shifted +1 b/c of frequency axis)
    return spec.mean(axis=axis+1), freqs


def bandfilter_spectrogram(data, smp_rate, axis=0, data_type='lfp', spec_type='complex',
                           freqs=((2,8),(10,32),(40,100)), removeDC=True,
                           filt='butter', order=4, params=None, buffer=0, downsmp=1, **kwargs):
    """
    Computes zero-phase band-filtered and Hilbert-transformed signal from data
    for given frequency band(s).

    Function aliased as bandfilter().

    Only parameters differing from :func:`.spectrogram` are described here.

    NOTE: Can specify filter implictly using (`freqs`,`filt`,`order`) OR explicitly using `params`.
          If `params` is input, `freqs`, `filt`, and `order` are ignored.

    Parameters
    ----------
    freqs : array-like, shape=(n_freqbands,2), default: ((2,8),(10,32),(40,100))
        List of (low,high) cut frequencies for each band to use.
        Set low cut = 0 for low-pass, set high cut >= smp_rate/2 for high-pass,
        otherwise assumes band-pass. Default samples ~ theta, alpha/beta, gamma.

    filt : str, default: 'butter' (Butterworth)
        Name of filter to use. See :func:`set_filter_params` for all options

    order : int, default: 4
        Filter order

    params : dict, default: (computed from `freqs`/`filt`/`order`)
        Parameters that explicitly define filter for each freq band.

        Alternative method for explicitly setting parameters defining freq band filters,
        which are precomputed with :func:`set_filter_params` (or elsewhere).
        Input either `freqs`/`filt`/`order` OR params.
        If params are not explicitly input, we compute them from `freqs`/`filt`/`order`.
        If params *are* explicitly input, `freqs`/`filt`/`order` are ignored.

        One of two forms: 'ba' or 'zpk', with key/values as follows:

        - b,a : array-like, shape=(n_freqbands,) of array-like (n_params[band,])
            Numerator `b` and denominator `a` polynomials of the filter for each band

        - z,p,k :
            Zeros, poles, and system gain of the IIR filter transfer function

    buffer : float, default: 0 (no buffer)
        Time (s) to trim off each end of time dimension of data.
        Removes symmetric buffer previously added (outside of here) to prevent
        edge effects.

    **kwargs :
        Any other kwargs passed directly to :func:`set_filter_params`

    Returns
    -------
    spec : ndarray, shape=(...,n_freqbands,n_timepts_out,...), dtype=complex or float.
        Band-filtered, (optionally) Hilbert-transformed "spectrogram" of data,
        transformed to requested spectral type.
        Same shape as input data, but with frequency axis prepended immediately
        before time `axis`. dtype is complex if `spec_type` is 'complex', float otherwise.

    freqs : ndarray, shape=(n_freqbands,2)
        List of (low,high) cut frequencies (Hz) for each band used in `spec`.

    timepts : ndarray, shape=(n_timepts_out,)
        List of timepoints in `spec` (in s, referenced to start of data).
    """
    _ = kwargs.pop('fft_method',None)   # Align API with other *_spectrogram() functions
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
        # Set any freqs > Nyquist equal to Nyquist
        freqs[freqs > smp_rate/2] = smp_rate/2
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


bandfilter = bandfilter_spectrogram
""" Alias of :func:`bandfilter_spectrogram`. See there for details. """


def set_filter_params(bands, smp_rate, filt='butter', order=4, btypes=None, form='ba',
                      return_dict=False, **kwargs):
    """
    Sets coefficients for desired filter(s) using scipy.signal
    "Matlab-style IIR filter design" functions

    NOTE: If return_dict is False, outputs are returned as a tuple, as described below;
    else, outputs are packaged in a single dict, with param names as keys.

    Parameters
    ----------
    bands : array-like, shape=(n_freqbands,2)
        List of (low,high) cut frequencies for each band to use.
        Set low cut = 0 for low-pass, set high cut >= smp_rate/2 for high-pass,
        otherwise assumes band-pass

    smp_rate : scalar
        Data sampling rate (Hz)

    filt : str, default: 'butter' (Butterworth)
        Name of filter to use. See :func:`set_filter_params` for all options

    order : int, default: 4
        Filter order

    btypes : str or array-like, shape=(n_freqbands,) of str, default: (based on `bands`)
        Passband/stop-type of filter. For multiple bands, can either input a single string
        value that will be used for all bands, or a sequence (list or object array) with
        one string value per band.

        Default values are set per band, depending on the band frequency range (ie, does it
        correspond to a low/high/band-pass filter). Options and default logic:

        - 'lowpass'  : Low-pass (high-cut) filtering. Default for bands[:,0] == 0 or -Inf
        - 'highpass' : High-pass (low-cut) filtering. Default for bands[:,1] == smp_rate/2 or +Inf
        - 'bandpass' : Band-pass filtering. Default if above conditions not met.
        - 'bandstop' : Band-stop filtering (eg for notch filtering). Never assumed as default.

    form : {'ba','zpk'}, default: ‘ba’
        Type of parameters to output:
        - 'ba': numerator(b), denominator (a)
        - 'zpk': Zeros (z), poles (p), and system gain (k) of the IIR filter transfer function

    return_dict : bool, default: False
        If True, params returned in a dict; else as standard series (tuple) of output variables

    **kwargs :
        Any additional kwargs passed directly to filter function

    Returns
    -------
    b,a : list, shape=(n_freqbands,) of list, shape=(n_params[band,])
        Numerator `b` and denominator `a` polynomials of the filter for each band.
        Returned if `form` == 'ba'.

    z,p,k : list, shape=(n_freqbands,) of list, shape=(n_params[band,])
        Zeros, poles, and system gain of IIR transfer function.
        Returned if `form` == 'zpk'.

    Examples
    --------
    params = set_filter_params(bands, smp_rate, form='ba', return_dict=True)

    b,a = set_filter_params(bands, smp_rate, form='ba', return_dict=False)

    z,p,k = set_filter_params(bands, smp_rate, form='zpk', return_dict=False)
    """
    # Convert bands to (n_freqbands,2)
    bands       = np.asarray(bands)
    if bands.ndim == 1: bands = np.reshape(bands,(1,len(bands)))
    n_bands     = bands.shape[0]
    nyquist     = smp_rate/2.0   # Nyquist freq at given sampling freq

    # Set default values for filter pass types based on freq range of each band
    if btypes is None:
        btypes = []
        for i_band,band in enumerate(bands):
            band_norm = band/nyquist  # Convert band to normalized frequency

            # If low-cut freq <= 0, assume low-pass filter
            if band_norm[0] <= 0:   btypes.append('lowpass')
            # If high-cut freq >= Nyquist freq, assume high-pass filter
            elif band_norm[1] >= 1: btypes.append('highpass')
            # Otherwise, assume band-pass filter
            else:                   btypes.append('bandpass')

    # Ensure passed `btypes` is a n_bands-length list of strings
    else:
        if isinstance(btypes,str): btypes = [btypes]
        assert len(btypes) in [1,n_bands], \
            "`btypes` must be given as a single string (used for *all* bands), \
              or a n_bands-length list of strings (%d bands, len(btypes)=%d)" \
            % (n_bands, len(btypes))
        if len(btypes) < n_bands:
            btypes = btypes * n_bands

    # Setup filter-generating function for requested filter type
    # Butterworth filter
    if filt.lower() in ['butter','butterworth']:
        gen_filt = lambda band, btype: butter(order, band, btype=btype, output=form)
    # Elliptic filter
    elif filt.lower() in ['ellip','butterworth']:
        rp = kwargs.pop('rp', 5)
        rs = kwargs.pop('rs', 40)
        gen_filt = lambda band, btype: ellip(order, rp, rs, band, btype=btype, output=form)
    # Chebyshev Type 1 filter
    elif filt.lower() in ['cheby1','cheby','chebyshev1','chebyshev']:
        rp = kwargs.pop('rp', 5)
        gen_filt = lambda band, btype: cheby1(order, rp, band, btype=btype, output=form)
    # Chebyshev Type 2 filter
    elif filt.lower() in ['cheby2','chebyshev2']:
        rs = kwargs.pop('rs', 40)
        gen_filt = lambda band, btype: cheby2(order, rs, band, btype=btype, output=form)
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

    for i_band,(band,btype) in enumerate(zip(bands,btypes)):
        band_norm = band/nyquist  # Convert band to normalized frequency

        # Convert band_norm to format expected by filter functions for different filter types
        if btype == 'lowpass':  band_norm = band_norm[1]
        elif btype == 'highpass': band_norm = band_norm[0]

        if form == 'ba':
            params['b'][i_band],params['a'][i_band] = gen_filt(band_norm, btype)
        else:
            params['z'][i_band],params['p'][i_band],params['k'][i_band] = gen_filt(band_norm, btype)

    if return_dict: return params
    else:           return params.values()

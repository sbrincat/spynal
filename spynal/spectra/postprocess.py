# -*- coding: utf-8 -*-
""" Postprocessing functions for LFP/EEG/continuous data and spectral analysis """
import numpy as np

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False

from spynal.utils import set_random_seed, iarange
from spynal.spectra.helpers import _str_to_pool_func


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


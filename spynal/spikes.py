# -*- coding: utf-8 -*-
"""
Preprocessing, basic analyses, and plotting of neural spiking activity

Overview
--------
Functionality includes computing spike rates (using binning or spike density methods) and their
statistict, inter-spike intervals and their statistics, and spike data preprocessing and plotting.

Most functions expect one of two formats of spiking data:

- **bool**
    Binary spike trains where 1's label times of spikes and 0's = no spike, in a Numpy ndarray
    of dtype=bool, where one axis corresponds to time, but otherwise can have any arbitrary
    dimensionality (including other optional axes corresponding to trials, units, etc.)

- **timestamp**
    Explicit spike timestamps in a Numpy ndarray of dtype=object. This is a "ragged nested
    sequence" (array of lists/sub-arrays of different length), analogous to Matlab cell arrays.
    Each object element is a variable-length list-like 1D subarray of spike timestamps
    (of dtype float or int), and the container object array can have any arbitrary dimensionality
    (including optional axes corresponding to trials, units, etc.).

    Alternatively, for a single spike train, a simple 1D list or ndarray of timestamps
    may be given instead.

Most functions perform operations in a mass-univariate manner. This means that
rather than embedding function calls in for loops over units, trials, etc., like this::

    for unit in units:
        for trial in trials:
            results[trial,unit] = compute_something(data[trial,unit])

You can instead execute a single call on ALL the data, labeling the relevant axis
for the computation (usually time here), and it will run in parallel (vectorized)
across all units, trials, etc. in the data, like this:

``results = compute_something(data, axis)``


Function list
-------------
Spike count/rate and inter-spike interval computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- rate :              Estimate spike rates/counts using given method
- bin_rate :          Compute spike counts/rates in series of time bins (regular or not)
- density :           Compute spike density (smoothed rate) with given kernel
- isi :               Compute inter-spike intervals from spike data

Rate and inter-spike interval stats
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- rate_stats :        Compute given statistic on spike rate data
- isi_stats :         Compute given statistic on inter-spike interval data

Spike waveform-shape stats
^^^^^^^^^^^^^^^^^^^^^^^^^^
- waveform_stats :    Compute given statistic on spike waveform data
- width :             Trough-to-peak temporal width of waveform
- repolarization :    Time for waveform to decay after peak
- trough_width :      Width of waveform trough
- amp_ratio :         Ratio of trough/peak amplitude

Preprocessing
^^^^^^^^^^^^^
- times_to_bool :     Convert spike timestamps to binary spike trains
- bool_to_times :     Convert binary spike train to timestamps
- cut_trials :        Cut spiking data into trials
- realign_data :      Realign data to new within-trial times (new t=0)
- pool_electrode_units : Pool all units on each electrode into a multi-unit

Plotting
^^^^^^^^
- plot_raster :       Generate a raster plot
- plot_mean_waveforms  : Plot mean spike waveforms from one/more units
- plot_waveform_heatmap : Plot heatmap (2d histogram) of spike waveforms

Synthetic data generation
^^^^^^^^^^^^^^^^^^^^^^^^^
- simulate_spike_rates :  Generate synthetic rates from Poisson distribution
- simulate_spike_trains : Generate synthetic spike trains from Poisson process

Function reference
------------------
"""
# Created on Mon Aug 13 14:38:34 2018
#
# @author: sbrincat

from warnings import warn
from math import isclose, ceil
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.signal.windows import hann, gaussian
from scipy.stats import poisson, expon

from spynal.utils import set_random_seed, unsorted_unique, iarange, index_axis, \
                         standardize_array, undo_standardize_array, \
                         setup_sliding_windows, concatenate_object_array, \
                         fano, cv, cv2, lv, gaussian_1d
from spynal.helpers import _isbinary, _merge_dicts, _check_window_lengths, \
                           _enclose_in_object_array
from spynal.plots import plot_line_with_error_fill, plot_heatmap


# =============================================================================
# Spike count/rate and inter-spike interval computation functions
# =============================================================================
def rate(data, method='bin', **kwargs):
    """
    Estimate spike rates (or counts) using given method

    Spiking data can be timestamps or binary (0/1) spike trains

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (each element = (n_spikes,) array) or
        ndarray, shape=(...,n_timepts,...), dtype=bool

        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.
        -or-
        Binary/boolean representation of spike times, for either a single
        or multiple spike trains.

    method : {'bin','density'}, default: 'bin'
        Spike rate estimation method:

        - 'bin' :       Traditional rectangular-binned rate (aka PSTH; see :func:`bin_rate`)
        - 'density' :   Kernel density estimator for smoothed spike rate (see :func:`density`)

    **kwargs :
        Any further arguments passed as-is to specific rate estimation function.

    Returns
    -------
    rates : ndarray, shape=(...,n_timepts_out) or (...,n_timepts_out,...)
        Estimated spike rates in spk/s (or spike counts) using given method
        (and for each trial/unit/etc. in `data`).

        For timestamp data, same dimensionality as `data`, with time axis appended to end.
        For boolean data, same dimensionality as `data`.
        If only a single spike train is input, output is (n_timepts_out,) vector.

        For 'bin' method, n_timepts_out = n_bins.
        For 'density' method, n_timepts_out depends on `lims`, `smp_rate`

        dtype is float (except int for `method` == 'bin AND `output` = 'count')

    timepts : ndarray, shape=(n_timepts_out,) or (n_bins,2)
        For 'density' method: Time sampling vector (in s). shape=(n_timepts_out,).
        For 'bin' method: [start,end] time of each time bin (in s). shape=(n_bins,2).
    """
    if method in ['bin','bins','bin_rate','psth']:      rate_func = bin_rate
    elif method in ['density','spike_density','sdf']:   rate_func = density

    return rate_func(data, **kwargs)


def bin_rate(data, lims=None, width=50e-3, step=None, bins=None, output='rate',
             axis=-1, timepts=None):
    """
    Compute spike rate/count within given sequence of hard-edged time bins

    Spiking data can be timestamps or binary (0/1) spike trains

    Use `lims`/`width`/`step` to set standard-width sliding window bins or
    use `bins` to set any arbitrary custom time bins

    NOTE: Spikes are counted within each bin including the start, but *excluding* the end
    of the bin. That is each bin is defined as the half-open interval [start,end).

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (each element = (n_spikes,) array)
        or ndarray, shape=(...,n_timepts,...), dtype=bool

        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.
        -or-
        Binary/boolean representation of spike times, for either a single
        or multiple spike trains.

    lims : array-like, shape=(2,)
        Full time range of analysis ([start,end] in s).
        Must input a value unless explicitly setting custom `bins`.

    width : scalar, default: 0.050 (50 ms)
        Full width (in s) of each time bin

    step : scalar, default: `width` (each bin starts at end of previous bin)
        Spacing (in s) between successive time bins

    bins : array-like, shape=(n_bins,2), default: setup_sliding_windows(width,lims,step)
        Alternative method for setting bins; overrides `width`/`spacing`/`lims`.
        [start,end] (in s) of each custom time bin. Bins can have any arbitrary width and spacing.
        Default generates bins with `width` and `spacing` from `lims[0]` to `lims[1]`

    output : str, default: 'rate'
        Which type pf spike measure to return:

        - 'rate' :    spike rate in each bin, in spk/s. Float valued.
        - 'count' :   spike count in each bin. Integer valued.
        - 'bool' :    binary presence/absence of any spikes in each bin. Boolean valued.

    axis : int, default: -1 (last axis of array)
        Axis of binary data corresponding to time dimension. Not used for spike timestamp data.

    timepts : ndarray, shape=(n_timepts,)
        Time sampling vector (in s) for binary data.
        Not used for spike timestamp data, but MUST be input for binary data.

    Returns
    -------
    rates : ndarray, shape=(...,n_bins) or (...,n_bins,...), dtype=float or int or bool
        Spike rates (in spk/s) or spike counts in each time bin
        (and for each trial/unit/etc. in `data`).

        For timestamp data, same shape as `data`, with time axis appended to end.
        If only a single spike train is input, output is (n_bins,) vector.

        For boolean data, same shape as `data` with time axis length reduced to n_bins.

        dtype is float for `output` = 'rate', int for 'count', bool for 'bool'.

    bins : ndarray, shape=(n_bins,2)
        [start,end] time (in s) of each time bin
    """
    # Convert boolean spike train data to timestamps for easier computation
    data_type = _spike_data_type(data)
    if data_type == 'bool':
        assert timepts is not None, \
            "For binary spike train data, a time sampling vector <timepts> MUST be given"
        if axis < 0: axis = data.ndim + axis
        data = bool_to_times(data,timepts,axis=axis)

    # If data is not an object array, its assumed to be a single spike train
    single_train = isinstance(data,list) or (data.dtype != object)

    # Enclose in object array to simplify computations; removed at end
    if single_train: data = _enclose_in_object_array(np.asarray(data))

    # If bins not explicitly input, set them based on limits,width,step
    if bins is None:
        assert lims is not None, \
            ValueError("Must input <lims> = full time range of analysis (or set custom <bins>)")
        bins = setup_sliding_windows(width,lims,step=step)
    else:
        bins = np.asarray(bins)
        if (bins.ndim == 1) and (len(bins) == 2): bins = bins[np.newaxis,:]

    assert (bins.ndim == 2) and (bins.shape[1] == 2), \
        ValueError("bins must be given as (n_bins,2) array of bin [start,end] times")

    n_bins  = bins.shape[0]
    widths  = np.diff(bins,axis=1).squeeze()

    # Are bins "standard"? : equal-width, with start of each bin = end of previous bin
    std_bins = (n_bins == 1) or \
               (np.allclose(widths,widths[0]) and np.allclose(bins[1:,0],bins[:-1,1]))

    def _histogram_count(data, bins):
        """ Count spikes in equal-width, disjoint bins """
        return np.histogram(data,bins)[0]


    def _custom_bin_count(data, bins):
        """ Count spikes in any arbitrary custom bins """
        return np.asarray([((start <= data) & (data < end)).sum()
                        for (start,end) in bins], dtype='uint16')

    # For standard bins, can use histogram algorithm
    if std_bins:
        count_spikes = _histogram_count
        # Convert bins to format expected by np.histogram = edges of all bins in 1 series
        bins_ = np.hstack((bins[:,0],bins[-1,-1]))
    # Otherwise, need algorithm to enumerate and count spikes in each bin
    else:
        count_spikes = _custom_bin_count
        bins_ = bins

    # Create 1D flat iterator to iterate over arbitrary-shape data array
    # Note: This always iterates in row-major/C-order regardless of data order, so all good
    data_flat = data.flat

    # Create array to hold counts/rates. Same shape as data, with bin axis appended.
    # dtype is int if computing counts, float if computing rates
    if output == 'rate':    dtype = float
    elif output == 'count': dtype = 'uint16'
    elif output == 'bool':  dtype = bool
    else:
        raise ValueError("Unsupported value '%s' input for <output>")

    rates = np.empty((*data.shape,n_bins),dtype=dtype)

    for _ in range(data.size):
        # Multidim coordinates into data array
        coords = data_flat.coords

        # Count spikes within each bin
        rates[(*coords,slice(None))] = count_spikes(data[coords],bins_)

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(data_flat)

    # Normalize all spike counts by bin widths to get spike rates
    # Note: For output='bool', values are auto-converted to bool when <rates> is set
    if output == 'rate': rates = rates / widths

    # If only a single spike train was input, squeeze out singleton axis 0
    if single_train: rates = rates.squeeze(axis=0)

    # For boolean input data, shift time axis back to its original location in array
    if (data_type == 'bool') and (axis != rates.ndim): rates = np.moveaxis(rates,-1,axis)

    return rates, bins

psth = bin_rate
""" Alias of :func:`bin_rate`. See there for details. """


def density(data, lims=None, width=None, step=1e-3, kernel='gaussian', buffer=None,
            axis=-1, timepts=None, **kwargs):
    """
    Compute spike density function (smoothed rate) via convolution with given kernel

    Spiking data can be timestamps or binary (0/1) spike trains

    NOTE: Spike densities exhibit "edge effects" where rate is biased downward at temporal limits
    of data, due to convolution kernel extending past edge of data and summing in zeros.

    The best way to avoid this is to request `lims` that are well within the temporal limits of
    the data. We also ameliorate this by extending the requested `lims` temporarily by a buffer,
    which is trimmed off to the requested `lims` before returning the computed density.

    Also, for binary spike data, if lims +/- buffer extend beyond the temporal limits of the data,
    we symmetrically reflect the data around its limits (note this can't be done for timestamp
    data as we don't know the data's temporal limits).

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (each element = (n_spikes,) array)
        or ndarray, shape=(...,n_timepts,...), dtype=bool

        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.
        -or-
        Binary/boolean representation of spike times, for either a single
        or multiple spike trains.

    lims : array-like, shape=(2,)
        Desired time range of returned spike density ([start,end] in s).
        For boolean spike data, defaults to (timepts[0],timepts[-1]).
        For spike timestamp data, a value MUST be input.

        NOTE: For either data type, lims should be within (or equal to) the full
        extent of the data itself. We test this for binary spike data, but we
        don't know the actual extent of timestamp data, so that is up to the user!

    width : scalar, default: (kernel-dependent)
        Width parameter (in s) for given convolution kernel.
        Interpretation is specific to value of `kernel`:

        - 'gaussian' : `width` = standard deviation, default: 0.50 (50 ms)
        - 'hanning' : `width` = half-width (~ 2.53x Gaussian SD), default: 0.125 (125 ms)

    step : scalar, default: 0.001 (1 ms)
        Spacing (in s) between successive time points in returned spike density.
        Values > 1 ms imply some downsampling from original spike train time sampling.
        NOTE: Must be an integer multiple of original data sampling (1 ms for timestamp data).
        NOTE: This argument replaces `downsmp` in previous versions.

    kernel : str or ndarray or callable, default: 'gaussian'
        Convolution kernel to use. Can be given as kernel name, with current options:

        - 'gaussian' : Gaussian kernel
        - 'hanning' : Hanning kernel

        Alternatively, can input any arbitrary kernel as an array of float values
        or as a custom function that can take any additional kwargs as arguments
        and return an array.

    buffer : float, default: (kernel-dependent, approximates length of kernel's edge effects)
        Length (in s) of symmetric buffer to add to each end of time dimension
        (and trim off before returning) to avoid edge effects. Kernel defaults:

        - 'gaussian' : 3*`width`  (3x std dev)
        - 'hanning' : `width` (half-width of kernel)

    axis : int, default: -1 (last axis of array)
        Axis of binary data corresponding to time dimension. Not used for spike timestamp data.

    timepts : ndarray, shape=(n_timepts,)
        Time sampling vector (in s) for binary data.
        Not used for spike timestamp data, but MUST be input for binary data.

    **kwargs :
        All other kwargs passed directly to kernel function

    Returns
    -------
    rates : ndarray, shape=(...,n_timepts_out) or (...,n_timepts_out,...), dtype=float
        Spike density function -- smoothed spike rates (in spk/s) estimated at each
        timepoint (and for each trial/unit/etc. in `data`).

        For timestamp data, same dimensionality as `data`, with time axis appended to end.
        For boolean data, same dimensionality as `data`.
        If only a single spike train is input, output is (n_timepts_out,) vector.
        n_timepts_out = iarange(lims[0],lims[1],step)

    timepts_out : ndarray, shape=(n_timepts_out,)
        Time sampling vector (in s) for `rates`
    """
    ### Argument processing ###
    if 'downsmp' in kwargs:
        downsmp = kwargs.pop('downsmp')
        step = 1000/downsmp
        warn("<downsmp> argument has been deprecated. Please use <step> argument instead (see docs).")

    if isinstance(kernel,str): kernel = kernel.lower()
    data_type = _spike_data_type(data)
    if axis < 0: axis = data.ndim + axis

    if data_type == 'bool':
        assert timepts is not None, \
            "For binary spike train data, a time sampling vector <timepts> must be given"
        assert (lims is None) or ((lims[0] >= timepts[0]) and (lims[1] <= timepts[-1])), \
            ValueError("Input <lims> must lie within range of time points in spiking data.")

    # If `lims` not input, set=(1st,last) timepts for boolean data; raise error for timestamp data
    if lims is None:
        if data_type == 'bool':
            lims = (timepts[0],timepts[-1])
        else:
            raise ValueError("For spike timestamp data, analysis time range must be set in <lims>")

    # Set default width specific to convolution kernel
    if width is None:
        if kernel in ['hann','hanning']:        width = 0.125   # half-width = 125 ms
        elif kernel in ['gaussian','normal']:   width = 0.050   # SD = 50 ms

    # Set default buffer based on overlap of convolution kernel used
    if buffer is None:
        if kernel in ['hann','hanning']:        buffer = width
        elif kernel in ['gaussian','normal']:   buffer = 3*width
        else:                                   buffer = 0

    # Compute sampling rate of input data (and thus initial convolution)
    if data_type == 'bool': dt = np.diff(timepts).mean()
    # Hard code sampling of timestamp->bool conveersion (and initial convolution) to 1 kHz (1 ms)
    else:                   dt = 1e-3

    smp_rate = round(1/dt)
    if smp_rate < 1000:
        warn('Sampling of %d Hz may binarize >1 spike/bin to 1' % smp_rate)

    if buffer != 0:
        # Convert buffer from time units -> samples
        n_smps_buffer = int(round(buffer*smp_rate))
        # Extend limits of data sampling (and initial analysis) by +/- buffer
        lims = (lims[0]-buffer, lims[1]+buffer)

    # Downsampling factor necessary to convert initial sampling rate to final desired sampling
    downsmp = int(round(step/dt))
    assert (downsmp >= 1) and np.isclose(downsmp, step/dt), \
        ValueError("<step> must be an integer multiple of %.3f (for %d ms sampling)" % \
                   (dt,int(dt*1000)))


    ### Set up data for spike density computation ###
    # Convert spike timestamps to binary spike trains w/in desired time range -> (...,n_timepts)
    if data_type == 'timestamp':
        data,timepts = times_to_bool(data, lims=lims, width=dt)
        axis = data.ndim

    # Reshape boolean data appropriately for analysis (including buffer)
    else:
        # Reshape boolean data so that time axis is -1 (end of array)
        if axis != data.ndim: data = np.moveaxis(data,axis,-1)

        # If desired limits of data sampling (+ buffer) is more restricted than data range,
        # truncate data to desired limits (no need to do more analysis than necessary!)
        if (lims[0] > timepts[0]) or (lims[1] < timepts[-1]):
            tbool   = (timepts >= lims[0]) & (timepts <= lims[1])
            timepts = timepts[tbool]
            data    = data[...,tbool]

        # If desired limits + buffer extend *beyond* range of data, reflect data at edges
        # Note: This conditional is indpendent of above bc could have both effects on start vs end
        if (lims[0] < timepts[0]) or (lims[1] > timepts[-1]):
            n_samples = data.shape[-1]
            # Number of samples to reflect data around (start,end) to generate desired buffer
            n_smps_reflect = (int(round((timepts[0]-lims[0])*smp_rate)),
                              int(round((lims[1]-timepts[-1])*smp_rate)))

            # Indexes corresponding to any reflection at start,end, with actual data in between
            idxs = np.concatenate((np.flip(iarange(1,n_smps_reflect[0])),
                                   np.arange(n_samples),
                                   iarange(n_samples-2, n_samples-(n_smps_reflect[1]+1), -1)))
            data = data[...,idxs]
            timepts = timepts[idxs]


    ### Set up convolution kernel for spike density computation ###
    n_smps_width = width*smp_rate # convert width to 1 kHz samples

    # Kernel is already a (custom) array of values -- do nothing
    if isinstance(kernel,np.ndarray):
        pass

    # Kernel is a function/callable -- call it to get kernel values
    elif callable(kernel):
        kernel = kernel(**kwargs)

    # Kernel is a string specifier -- call appropriate kernel-generating function
    elif isinstance(kernel,str):
        if kernel in ['hann','hanning']:
            kernel = hann(int(round(n_smps_width*2.0)), **kwargs)
        elif kernel in ['gaussian','normal']:
            kernel = gaussian(int(round(n_smps_width*6.0)), n_smps_width, **kwargs)
        else:
            raise ValueError("Unsupported value '%s' given for kernel. \
                              Should be 'hanning'|'gaussian'" % kernel)

    else:
        raise TypeError("Unsupported type '%s' for <kernel>. Use string, function, \
                         or explicit array of values" % type(kernel))

        # DELETE
        # assert len(kwargs) == 0, \
        #     TypeError("Incorrect or misspelled variable(s) in keyword args: " +
        #               ', '.join(kwargs.keys()))

    # Normalize kernel to integrate to 1
    kernel = kernel / (kernel.sum()/smp_rate)


    ### Compute spike density and reshape data back to desired form ###
    # Ensure kernel broadcasts against data
    slicer = tuple([np.newaxis]*(data.ndim-1) + [slice(None)])

    # Compute density as convolution of spike trains with kernel
    # Note: 1d kernel implies 1d convolution across multi-d array data
    rates = convolve(data, kernel[slicer], mode='same')

    # Remove any time buffer from spike density and time sampling vector
    if buffer != 0:
        rates   = _remove_buffer(rates, n_smps_buffer, axis=-1)
        timepts = _remove_buffer(timepts, n_smps_buffer, axis=-1)
    # Implement any temporal downsampling of rates to final desired <step> size
    if downsmp != 1:
        rates   = rates[...,0::downsmp]
        timepts = timepts[0::downsmp]

    # KLUDGE Sometime trials/neurons/etc. w/ 0 spikes end up with tiny non-0 values
    # due to floating point error in fft routines. Fix by setting = 0.
    # Note: Polling data for any spikes should be done on original data (+buffer, no downsmp)
    no_spike_idxs = ~data.any(axis=-1,keepdims=True)
    shape = tuple([1]*(rates.ndim-1) + [rates.shape[-1]])
    rates[np.tile(no_spike_idxs,shape)] = 0

    # KLUDGE Sometime rates end up with minute negative values due to floating point error,
    # which can mess up things downstream (eg sqrt). Set these = 0.
    rates[rates < 0] = 0

    # Reshape rates so that time axis is in original location
    if (data_type == 'bool') and (axis != data.ndim):
        rates = np.moveaxis(rates,-1,axis)

    return rates, timepts


def isi(data, axis=-1, timepts=None):
    """
    Compute inter-spike intervals of one or more spike trains

    Spiking data can be timestamps or binary (0/1) spike trains

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (each element = (n_spikes,) array)
        or ndarray, shape=(...,n_timepts,...), dtype=bool
        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.
        -or-
        Binary/boolean representation of spike times, for either a single
        or multiple spike trains.

    axis : int, default: -1 (last axis of array)
        Axis of binary data corresponding to time dimension. Not used for spike timestamp data.

    timepts : ndarray, shape=(n_timepts,)
        Time sampling vector (in s) for binary data.
        Not used for spike timestamp data, but MUST be input for binary data.

    Returns
    -------
    ISIs : ndarray, shape=(n_spikes-1,) or ndarray, dtype=object (each elem = (n_spikes-1,) array)
        Time intervals between each successive pair of spikes in data (in same time units as data).
        For boolean data, output is converted to timestamp-like config with time axis removed.
        For timestamp data, same shape as `data` (but with 1 fewer item per array cell).
        Array cells with <= 1 spike returned as empty arrays.
        If only a single spike train is input, output is (n_spikes-1,) vector.
    """
    # Convert boolean spike train data to timestamps for easier computation
    data_type = _spike_data_type(data)
    if data_type == 'bool':
        assert timepts is not None, \
            "For binary spike train data, a time sampling vector <timepts> MUST be given"
        if axis < 0: axis = data.ndim + axis
        data = bool_to_times(data,timepts,axis=axis)

    # If data is not an object array, its assumed to be a single spike train
    single_train = isinstance(data,list) or (data.dtype != object)

    # Enclose in object array to simplify computations; removed at end
    if single_train: data = _enclose_in_object_array(np.asarray(data))

    # Create 1D flat iterator to iterate over arbitrary-shape data array
    # Note: This always iterates in row-major/C-order regardless of data order, so all good
    data_flat = data.flat

    ISIs = np.empty_like(data,dtype=object)

    for _ in range(data.size):
        # Multidim coordinates into data array
        coords = data_flat.coords

        # Compute ISIs for given data cell (unit/trial/etc.)
        ISIs[coords] = np.diff(data[coords])

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(data_flat)

    # If only a single spike train was input, squeeze out singleton axis 0
    if single_train: ISIs = ISIs.squeeze(axis=0)

    return ISIs

interspike_interval = isi
""" Alias of :func:`isi`. See there for details. """


#==============================================================================
# Spike rate and inter-spike interval statistics functions
#==============================================================================
def rate_stats(rates, stat='Fano', axis=None, **kwargs):
    """
    Compute given statistic on spike rates of one or more spike trains

    Input data must be spike rates, eg as computed using :func:`rate`

    Stats may be along one/more array axes (eg trials) or across entire data array

    Parameters
    ----------
    rates : ndarray, shape=(...,n_obs,...)
        Spike rate data. Arbitrary shape.

    stat : {'Fano','CV'}, default: 'Fano'
        Rate statistic to compute. Options:

        - 'Fano' :  Fano factor = var(rate)/mean(rate), using :func:`fano`
        - 'CV' :    Coefficient of Variation = SD(rate)/mean(rate), using :func:`cv`

    axis  : int, default: None
        Array axis to compute rate statistics along (usually corresponding
        to distict trials/observations). If None, computes statistic across
        entire array (analogous to np.mean/var).

    **kwargs :
        Any additional keyword args passed directly to statistic computation function

    Returns
    -------
    stats : float or ndarray, shape=(...,1,...)
        Rate statistic computed on data.
        For 1d data or axis=None, a single scalar value is returned.
        Otherwise, it's an array w/ same shape as `rates`, but with `axis`
        reduced to length 1.
    """
    data_type = _spike_data_type(rates)
    assert data_type not in ['timestamp','bool'], \
        TypeError("Must input spike *rate* data for this function (eg use rate())")

    stat = stat.lower()

    if stat == 'fano':  stat_func = fano
    elif stat == 'cv':  stat_func = cv
    else:
        raise ValueError("Unsupported value '%s' for <stat> (should be 'Fano'|'CV')" % stat)

    return stat_func(rates, axis=axis, **kwargs)


def isi_stats(ISIs, stat='Fano', axis='each', **kwargs):
    """
    Compute given statistic on inter-spike intervals of one or more spike trains

    Input data must be inter-spike intervals, eg as computed using :func:`isi`

    Can request data to be pooled along one/more axes (eg trials) before stats computation

    Parameters
    ----------
    ISIs : ndarray, shape=(n_spikes-1,) or ndarray, dtype=object (each elem = (n_spikes-1,) array)
        List of inter-spike intervals (in s), eg as computed by :func:`isi`. Can be given for either
        a single spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.

    stat : {'Fano','CV','CV2','LV','burst_fract}, default: 'Fano'
        ISI statistic to compute. Options:

        - 'Fano' :    Fano factor = var(ISIs)/mean(ISIs), using :func:`fano`
        - 'CV' :      Coefficient of Variation = SD(ISIs)/mean(ISIs), using :func:`cv`
        - 'CV2' :     Local Coefficient of Variation (Holt 1996), using :func:`cv2`
        - 'LV' :      Local Variation (Shinomoto 2009), using :func:`lv`
        - 'burst_fract' : Measure of burstiness (% spikes in bursts), using :func:`burst_fract`

        CV2 and LV and CV-like measures that reduce influence of changes in spike rate on
        the metric by only measuring local variation (between temporally adjacent ISIs).
        See their specific functions for details.

    axis : int or None or 'each', default: 'each'
        Axis of ISI data to pool ISIs along before computing stat.
        eg, for data that is shape (n_trials,n_units), if you want to compute a stat value
        for each unit, pooled across all trials, you'd set `axis` = 0.
        If axis=None, ISIs are pooled across the *entire* data array.
        If axis='each', stats are computed separately for each spike train in the array.

        NOTE: For locality-sensitive stats ('CV2','LV'), axis MUST = 'each'.
        NOTE: default 'each' is the opposite of default for :func:`rate_stats`

    Returns
    -------
    stats : float or ndarray
        Given ISI stat, computed on ISI data.
        Return as a single scalar if axis=None. Otherwise returns as array of
        same shape as ISIs, but with `axis` reduced to length 1.

    References
    ----------
    - Holt et al. (1996) Journal of Neurophysiology https://doi.org/10.1152/jn.1996.75.5.1806
    - Shinomoto et al. (2009) PLoS Computational Biology https://doi.org/10.1371/journal.pcbi.1000433
    """
    stat = stat.lower()

    if stat in ['cv2','lv']:
        assert axis == 'each', \
            ValueError("No pooling allowed for locality-sensitive stats (CV2,LV). Use axis='each'")

    if stat == 'fano':          stat_func = fano
    elif stat == 'cv':          stat_func = cv
    elif stat == 'cv2':         stat_func = cv2
    elif stat == 'lv':          stat_func = lv
    elif stat == 'burst_fract': stat_func = burst_fract
    else:
        raise ValueError("Unsupported value '%s' for <stat>." % stat)

    # If data is not an object array, its assumed to be a single spike train
    single_train = isinstance(ISIs,list) or (ISIs.dtype != object)

    # Enclose in object array to simplify computations; removed at end
    if single_train:
        ISIs = _enclose_in_object_array(np.asarray(ISIs))

    # Pool (concatenate) along axis if requested (ie not running separately for each spike train)
    elif axis != 'each':
        ISIs = concatenate_object_array(ISIs,axis=axis,sort=False)
        # KLUDGE Function above extracts item from object array. Reenclose it to simplify code.
        if axis is None: ISIs = _enclose_in_object_array(ISIs)

    # Create 1D flat iterator to iterate over arbitrary-shape data array
    # Note: This always iterates in row-major/C-order regardless of data order, so all good
    ISIs_flat = ISIs.flat

    stats = np.empty_like(ISIs,dtype=float)

    for _ in range(ISIs.size):
        # Multidim coordinates into data array
        coords = ISIs_flat.coords

        # Compute ISI stats for given data cell (unit/trial/etc.)
        stats[coords] = stat_func(ISIs[coords], **kwargs)

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(ISIs_flat)

    # If only a single spike train was input, squeeze out singleton axis 0
    if single_train: stats = stats.squeeze(axis=0)

    return stats


def burst_fract(ISIs, crit=0.020):
    """
    Compute measure of burstiness of ISIs of a spike train.

    Burst_fract = fraction of all ISIs that are within a spike burst (ISI < 20 ms by default)

    Parameters
    ----------
    ISIs : array-like, shape=(n_ISIs,)
        List of inter-spike intervals for a single spike train

    crit : float, default: 20 ms
        Criterion ISI value (s) to discriminate burst vs non-burst spikes

    Returns
    -------
    burst : float
        Fraction of all spikes that are within spike bursts
    """
    return (ISIs < crit).sum() / ISIs.size


#==============================================================================
# Spike waveform statistics functions
#==============================================================================
def waveform_stats(spike_waves, stat='width', axis=0, **kwargs):
    """
    Compute given statistic on one or more spike waveforms of one or more spike trains

    Parameters
    ----------
    spike_waves : ndarray, shape=(...,n_timepts,...), dtype=float or
        ndarray, shape=Any, dtype=object (elem's are (...,n_timepts,...) arrays)
        Spike waveform data, given in one of two formats:

        (1) a single ndarray of waveform data with one or more waveforms. Shape is arbitrary,
            but `axis` should correspond to time samples of waveform(s).
        (2) an object ndarray where each element contains waveform data like format (1)
            for one unit, trial, etc. Time axis and time sampling must be the same for all
            elements, but other dimensions need not be (ie there can be different numbers
            of spikes aacross trials, units, etc.)

    stat : str, default: 'width'
        Spike waveform statistic to compute. Options:

        - 'width' :   Temporal width (s) btwn largest depol trough and subsequent hyperpol peak,
                      using :func:`trough_to_peak_width`
        - 'repolarization' : Time from hyperpol peak to subsequent inflection or percent decrease,
                        using :func:`repolarization_time`
        - 'trough_width' : Full with at half-height of depol trough, using :func:`trough_width`
        - 'amp_ratio' : Ratio of amplitude of depol trough / hyperpol peak, using :func:`amp_ratio`

    axis : int, default: 0
        Axis of `spike_waves` to compute stat along, corresponding to waveform timepoints.
        For object array data ((2) above), this should be the axis of each object array *element*
        that corresponds to time.

    Returns
    -------
    stats : float or ndarray, shape=(...,1,...) or
        ndarray, shape=Any, dtype=object (elem's are (n_timepts,n_spikes) arrays)
        Given spike waveform stat, computed on each waveform in `spike_waves`.
        For 1d data (single waveform), a single scalar value is returned.
        Otherwise, it's an array w/ same shape as `spike_waves`, but with `axis`
        reduced to length 1.
    """
    if axis < 0: axis = spike_waves.ndim + axis
    stat = stat.lower()

    if stat in ['width','t2p','trough_to_peak_width']:
        stat_func = trough_to_peak_width
    elif stat in ['repol','repolarization','repolarization_time']:
        stat_func = repolarization_time
    elif stat in ['trough_width','trough']:
        stat_func = trough_width
    elif stat in ['trough_peak_amp_ratio','amp_ratio']:
        stat_func = trough_peak_amp_ratio
    else:
        raise ValueError("Unsupported value '%s' for <stat>." % stat)

    # If data is not an object array, its assumed to be a single spike train
    single_train = isinstance(spike_waves,list) or (spike_waves.dtype != object)

    # Enclose in object array to simplify computations; removed at end
    if single_train: spike_waves = _enclose_in_object_array(np.asarray(spike_waves))

    # Create 1D flat iterator to iterate over arbitrary-shape data array
    # Note: This always iterates in row-major/C-order regardless of data order, so all good
    spike_waves_flat = spike_waves.flat

    stats = np.empty_like(spike_waves,dtype=object)

    for _ in range(spike_waves.size):
        # Multidim coordinates into data array
        coords = spike_waves_flat.coords

        cur_waves = spike_waves[coords]
        multi_dim = cur_waves.ndim > 1
        do_reshape = (axis != 0) or (spike_waves.ndim > 2)

        if do_reshape:
            cur_waves,shape = standardize_array(cur_waves, axis=axis, target_axis=0)

        # Compute waveform stat for all waveform(s) in current data cell (unit/trial/etc.)
        if not multi_dim:
            cur_stats = stat_func(cur_waves, **kwargs)
        else:
            n_spikes = cur_waves.shape[1]
            cur_stats = np.empty((1,n_spikes))
            for i_spike in range(n_spikes):
                cur_stats[0,i_spike] = stat_func(cur_waves[:,i_spike], **kwargs)

        if do_reshape:
            cur_stats = undo_standardize_array(cur_stats, shape, axis=axis, target_axis=0)

        stats[coords] = cur_stats

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(spike_waves_flat)

    # If only a single spike train was input, extract single element from object array
    if single_train: stats = stats[0]

    return stats


def trough_to_peak_width(spike_wave, smp_rate):
    """
    Compute time difference between largest spike waveform trough (depolarization) and
    subsequent peak (after-hyperpolarization) for a single spike waveform

    Parameters
    ----------
    spike_wave : ndarray, shape=(n_timepts,)
        Spike waveform to compute stat on

    smp_rate : float
        Sampling rate of spike waveform (Hz)

    Returns
    -------
    stat : float
        Temporal width of waveform (in s), from trough to after-hyperpolarization peak
    """
    assert (spike_wave.ndim == 1) and not (spike_wave.dtype == object), \
        "This only accepts a single spike waveform. For >1 spikes, use wrapper waveform_stats()"

    # Find largest trough (depolarization) in waveform
    trough_idx = np.argmin(spike_wave)

    # Find largest peak (hyperpolarization) after trough
    peak_idx = trough_idx + np.argmax(spike_wave[trough_idx:])

    return (peak_idx - trough_idx) / smp_rate


def trough_width(spike_wave, smp_rate):
    """
    Compute full width at half-height of spike waveform trough (depolarization phase)

    Parameters
    ----------
    spike_wave : ndarray, shape=(n_timepts,)
        Spike waveform to compute stat on

    smp_rate : float
        Sampling rate of spike waveform (Hz)

    Returns
    -------
    stat : float
        Temporal width (FWHM) of waveform depolarization trough (in s)
    """
    assert (spike_wave.ndim == 1) and not (spike_wave.dtype == object), \
        "This only accepts a single spike waveform. For >1 spikes, use wrapper waveform_stats()"

    # Find amplitude of largest trough (depolarization) in waveform
    trough_idx = np.argmin(spike_wave)
    trough_amp = spike_wave[trough_idx]

    criterion = trough_amp/2
    # Find last point after trough still > half max amplitude
    half_amp_end_idx = trough_idx + np.where(spike_wave[trough_idx:] > criterion)[0][0] - 1
    # Find first point before trough still > half max amplitude
    half_amp_start_idx = trough_idx - np.where(spike_wave[trough_idx::-1] > criterion)[0][0] + 1

    return (half_amp_end_idx - half_amp_start_idx) / smp_rate


def repolarization_time(spike_wave, smp_rate, criterion=0.75):
    """
    Compute time of repolarization of after-hyperpolarization peak of single spike waveform --
    time difference from 1st peak after global trough to when it has decayed

    Parameters
    ----------
    spike_wave : ndarray, shape=(n_timepts,)
        Spike waveform to compute stat on

    smp_rate : float
        Sampling rate of spike waveform data (Hz)

    criterion : float or 'inflection', default: 0.75
        What criterion is used to determine time of repolaration (decay to baseline) of peak:

        - float : Use the point where amplitude has decayed to given proportion of its peak value
            eg, for criterion=0.75, use point where amplitude is <= 75% of peak.
        - 'inflection' : Use the inflection point (change in sign of the 2nd derivative)

    Returns
    -------
    stat : float
        Repolarization time (in s) of spike waveform
    """
    assert (spike_wave.ndim == 1) and not (spike_wave.dtype == object), \
        "This only accepts a single spike waveform. For >1 spikes, use wrapper waveform_stats()"

    # Find largest trough (depolarization) in waveform
    trough_idx = np.argmin(spike_wave)

    # Find largest peak (hyperpolarization) after trough
    peak_idx = trough_idx + np.argmax(spike_wave[trough_idx:])

    # Find timepoint when waveform crosses criterion percentage of peak amplitude
    if isinstance(criterion,float):
        assert (criterion > 0) and (criterion < 1), "Criterion must be in range (0,1)"
        repol_idx = np.where(spike_wave[peak_idx:] <= criterion*spike_wave[peak_idx])[0]
    # Find inflection point (change in sign of 2nd derivative) after peak
    elif criterion == 'inflection':
        repol_idx = np.where(np.diff(np.sign(np.gradient(np.gradient(spike_wave[peak_idx:])))))[0]
    else:
        raise ValueError("Unsupported value '%s' given for <criterion>" % criterion)

    repol_idx = peak_idx + repol_idx[0] if len(repol_idx) > 0 else np.nan

    return (repol_idx - peak_idx) / smp_rate


def trough_peak_amp_ratio(spike_wave):
    """
    Compute ratio of amplitudes (height) of largest spike waveform trough (depolarization)
    and subsequent peak (after-hyperpolarization) for a single spike waveform

    Parameters
    ----------
    spike_wave : ndarray, shape=(n_timepts,)
        Spike waveform to compute stat on

    Returns
    -------
    stat : float
        Amplitude ratio of waveform trough/peak
    """
    assert (spike_wave.ndim == 1) and not (spike_wave.dtype == object), \
        "This only accepts a single spike waveform. For >1 spikes, use wrapper waveform_stats()"

    # Find largest trough (depolarization) in waveform
    trough_idx = np.argmin(spike_wave)

    # Find largest peak (hyperpolarization) after trough
    peak_idx = trough_idx + np.argmax(spike_wave[trough_idx:])

    trough_amp = spike_wave[trough_idx]
    peak_amp =  spike_wave[peak_idx]

    return np.abs(trough_amp) / peak_amp


# =============================================================================
# Preprocessing/Utility functions
# =============================================================================
def bool_to_times(spike_bool, timepts, axis=-1):
    """
    Convert boolean (binary) spike train representaton to spike timestamps

    Inverse function of :func:`times_to_bool`

    Parameters
    ----------
    spike_bool : ndarray, shape=(...,n_timepts,...), dtype=bool
        Binary spike trains, where 1 indicates >= 1 spike in time bin, 0 indicates no spikes

    timepts : ndarray, shape=(n_timepts,)
        Time sampling vector for data (center of each time bin used to compute binary train)

    axis : int, default: -1 (last axis)
        Axis of data corresponding to time dimension

    Returns
    -------
    spike_times : ndarray, dtype=object (each element = (n_spikes,) array)
        or ndarray, shape=(n_spikes,)
        Spike timestamps (in same time units as timepts), for each spike train in input.
        Returns as vector-valued array of timestamps if input is single spike train,
        otherwise as object array of variable-length timestamp vectors.
    """
    spike_bool = np.asarray(spike_bool)
    timepts = np.asarray(timepts)
    if axis < 0: axis = spike_bool.ndim + axis

    # For single-spike-train data, temporarily prepend singleton axis
    single_train = spike_bool.ndim == 1
    if single_train:
        spike_bool = spike_bool[np.newaxis,:]
        axis = 1

    # Reshape input data -> 2d array (n_spike_trains,n_timepts)
    # (where spike trains = trials,units,etc.)
    spike_bool,spike_bool_shape = standardize_array(spike_bool, axis=axis, target_axis=-1)
    n_spike_trains,n_timepts = spike_bool.shape

    spike_times = np.empty((n_spike_trains,), dtype=object)

    # For each spike train, find spikes and convert to timestamps
    for i in range(n_spike_trains):
        spike_times[i] = timepts[spike_bool[i,:]]

    # Reshape output to match shape of input, without time axis
    out_shape = [d for i,d in enumerate(spike_bool_shape) if i != axis]
    spike_times = spike_times.reshape(out_shape)

    # Extract single spike train from nesting array -> (n_spikes,) array
    if single_train: spike_times = spike_times[0]

    return spike_times


def times_to_bool(spike_times, lims=None, width=1e-3, bins=None):
    """
    Convert spike timestamps to boolean (binary) spike train representaton

    Inverse function of :func:`bool_to_times`

    Times bins for computing binary spike trains may be set either implicitly
    via `lims` and `width`, or set explicitly using `bins`.

    Parameters
    ----------
    spike_times : ndarray, shape=Any, dtype=object (each element = (n_spikes,) array)
        or array-like, shape=(n_spikes,)
        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.

    lims : array-like, shape=(2,)
        Full time range of analysis ([start,end] in s).
        Must input a value (unless explicitly setting custom `bins`)

    width : float, default: 0.001 (1 ms)
        Width of bin used to discretize spike times (s). Usually 1 ms.

    bins : array-like, (n_bins,2), default: setup_sliding_windows(width,lims,width)
        Alternative method for setting time bins. Overrides any values set for `lims`, `width`.
        [start,end] time (in s) of each custom time bin. Bins can in theory have any arbitrary
        width and spacing, but really you would always want equal width, and width = spacing,
        so each bin starts at end of last bin

    Returns
    -------
    spike_bool : ndarray, shape=(...,n_bins,...), dtype=bool
        Binary spike trains, where 1 indicates >= 1 spike in time bin, 0 indicates no spikes

    timepts : ndarray, shape(n_bins,)
        Time sampling vector (in s). Center of each time bin used to compute binary spike data
    """
    # If bins is not given explicitly, set it based on width,lims
    if bins is None:
        assert lims is not None, \
            ValueError("Must input <lims> = full time range of analysis (or set custom <bins>)")

        # If width = 1 ms, extend lims by 0.5 ms, so bins end up centered
        # on whole ms values, as we typically want for binary spike trains
        if isclose(width,1e-3): lims = [lims[0] - 0.5e-3, lims[1] + 0.5e-3]
        bins = setup_sliding_windows(width, lims=lims, step=width)

    timepts = bins.mean(axis=1)

    # For each spike train in <spike_times> compute count w/in each hist bin
    # Note: Setting dtype=bool implies any spike counts > 0 will be True
    spike_bool,bins = bin_rate(spike_times, bins=bins, output='bool')

    return spike_bool, timepts


def cut_trials(data, trial_lims, smp_rate=None, axis=None, trial_refs=None):
    """
    Cut time-continuous spiking data into trials

    Spiking data may be in form of spike timestamps or binary spike trains

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (elems=(n_spikes,) arrays)
        or ndarray, shape=(...,n_timepts,...), dtype=bool

        Time-continuous spiking data (not cut into trials). Arbitrary shape, could include
        multiple channels, etc. Given in one of two formats:

        - timestamp: Spike timestamps, usually in seconds referenced to some within-trial event
        - bool: Binary (1/0) spike trains in bool array

        For binary spike data, additional `smp_rate` and `axis` keyword arguments must be
        input to indicate the sampling rate (in Hz) and the array time axis.

    trial_lims : array-like, shape=(n_trials,2)
        List of [start,end] times of each trial (in same timebase as data) to use to cut data

    smp_rate : scalar
        Sampling rate of binary spiking data (Hz).
        Must input a value for binary spiking data; not used for spike timestamp data.

    axis : int, default: 0
        Axis of data array corresponding to time samples. Only used for binary spike data.

    trial_refs : array-like, shape=(n_trials,), default: None
        List giving event time in each trial to re-reference trial's spike timestamps to
        (ie this sets t=0 for each trial). If None, just leave timestamps in original timebase).
        Only used for spike timestamp data.

    Returns
    -------
    cut_data : ndarray, shape=(...,n_trials), dtype=object (elems=(n_trial_spikes,) arrays
        or ndarray, shape=(...,n_trial_timepts,...,n_trials), dtype=bool
        Spiking data segmented into trials. Trial axis is appended to end of all axes in
        input data. Shape is otherwise the same for timestamp data, but for binary data
        time `axis` is reduced to length implied by `trial_lims`.
    """
    trial_lims = np.asarray(trial_lims)

    assert (trial_lims.ndim == 2) and (trial_lims.shape[1] == 2), \
        "trial_lims argument should be a (n_trials,2) array of trial [start,end] times"

    data_type = _spike_data_type(data)

    if data_type == 'timestamp':
        return _cut_trials_spike_times(data, trial_lims, trial_refs=trial_refs)

    elif data_type == 'bool':
        return _cut_trials_spike_bool(data, trial_lims, smp_rate, axis=axis)

    else:
        raise ValueError("Unsupported spike data format. Must be timestamps or binary (0/1)")


def realign_data(data, align_times, trial_axis,
                 time_axis=None, timepts=None, time_range=None):
    """
    Realign timing of trial-cut spiking data on new within-trial event times
    (eg new trial event) so that t=0 on each trial at given event.

    For example, data aligned to a start-of-trial event might
    need to be realigned to the behavioral response.

    Spiking data may be in form of spike timestamps or binary spike trains

    Parameters
    ----------
    data : ndarray, shape=(...,n_trials,...), dtype=object (elems=(n_spikes[trial],) arrays)
        or ndarray, shape=(...,n_timepts,...), dtype=bool

        Spiking data, given in one of two formats:

        - timestamp: Spike timestamps, usually in seconds referenced to some within-trial event
        - bool: Binary (1/0) spike trains in bool array

        Can be any arbitrary shape (including having multiple units), as long as `trial_axis`
        is given (and also `time_axis` for bool data)

    align_times : array-like, shape=(n_trials,)
        New set of times (in old reference frame) to realign spiking data to

    trial_axis : int
        Axis of `data` corresponding to trials. Must input for either data type.

    time_axis : int
        Axis of bool data corresponding to time samples.
        Must input for bool data; not used for spike timestamps.

    timepts : array-like, shape(n_timepts)
        Time sampling vector for bool data (in s).
        Must input for bool data; not used for spike timestamps.

    time_range : array-like, shape=(2,)
        Time range to extract from each trial of bool data around new align time
        ([start,end] time in s relative to `align_times`).
        eg, time_range=(-1,1) -> extract 1 s on either side of align event.
        Must input for bool data; not used for spike timestamps.

    Returns
    -------
    realigned : ndarray, shape=(...,n_trials,...), dtype=object (elems=(n_spikes[trial],) arrays)
        or ndarray, shape=(...,n_timepts_out,...), dtype=bool
        Data realigned to given within-trial times.
        For timestamp data, this has same shape as input data.
        For binary data, time axis is reduced to length implied by `time_range`, but
        otherwise array has same shape as input data.
    """
    data_type = _spike_data_type(data)

    if data_type == 'timestamp':
        return _realign_spike_times(data, align_times, trial_axis=trial_axis)

    elif data_type == 'bool':
        return _realign_spike_bool(data, align_times, trial_axis=trial_axis,
                                   time_axis=time_axis, timepts=timepts, time_range=time_range)

    else:
        raise ValueError("Unsupported spike data format. Must be timestamps or binary (0/1)")


def realign_data_on_event(data, event_data, event, **kwargs):
    """
    Convenience wrapper around `realign_data` for relaligning to a given
    named event within a per-trial dataframe or dict variable.

    Only parameters differing from :func:`realign_data` are described here.

    Parameters
    ----------
    event_data : dict, {str : ndarray, shape=(n_trials,)} or DataFrame, shape=(n_trials,n_events)
        Per-trial event timing data to use to realign spike timestamps.

    event : str
        Dict key or DataFrame column name whose associated values are to be used to realign data
    """
    # Extract vector of times to realign on
    align_times = event_data[event]
    # Compute the realignment and return
    return realign_data(data, align_times, **kwargs)


def pool_electrode_units(data_sua, electrodes, axis=-1, elec_set=None,
                         return_idxs=False, sort=True):
    """
    Pool spiking data across all units (neurons) on each electrode into a single
    multi-unit for each electrode

    Spiking data may be in form of spike timestamps, binary spike trains, or spike rates/counts

    Parameters
    ----------
    data_sua : ndarray, shape=(...,n_units,...), dtype=object (elems= (n_spikes[unit],) arrays)
        or dtype=bool or dtype=float or int

        Spiking data for multiple single units on one or more electrodes, and optionally for
        different trials/conditions/etc. (arbitrary shape), in 1 of 3 formats:

        - timestamp: Lists of spike timestamps (dtype=object)
        - bool: Binary (0/1) spike trains (dtype=bool)
        - rate: Sets of spike rates or counts (dtype=float or int)

    electrodes : array-like, shape=(n_units,)
        List of electrode numbers of each unit in `data_sua`

    axis: int, default: -1 (last axis)
        Axis of `data_sua` corresponding to different units.

    elec_set : array-like, shape=(n_elecs,), default: unsorted_unique(electrodes)
        Set of unique electrodes in `electrodes`. Default uses all unique values in `electrodes`.

    return_idxs : bool, default: False (only return `data_mua`)
        If True, additionally returns list of indexes corresponding to 1st occurrence
        of each electrode in `electrodes`

    sort : bool, default: True
        If True, sort pooled timestamps so they remain in sequential order after concatenation.
        Only used for timestamp data.

    Returns
    -------
    data_mua : ndarray, shape=(...,n_elecs,...), dtype=object (elems= (n_spikes[elec],) arrays)
        or dtype=bool or dtype=float or int
        Spiking data pooled across all single units into a single electrode-level multi-unit
        for each electrode. Same shape as `data_sua`, but with `axis` reduced to length=n_elecs.

        For timestamp data, spike timestamps are concatenated together across units, and optionally
        resorted into sequential order (if `sort` is True).

        For bool data, spike trains are combined via "logical OR" across units (spike is registered
        in output at times when there is a spike in *any* single-unit on electrode).

        For rate data, spike rates/counts are summed across units.

    elec_idxs : ndarray, shape=(n_elecs,), dtype=int, optional
        Indexes of 1st occurrence of each electrode in `elec_set` within `electrodes`.
        Can be used to transform any corresponding metadata appropriately.
        Only returned if `return_idxs` is True.

    Examples
    --------
    data_mua = pool_electrode_units(data_sua, electrodes, return_idxs=False)

    data_mua, elec_idxs = pool_electrode_units(data_sua, electrodes, return_idxs=True)
    """
    # Find set of electrodes in data, if not explicitly input
    if elec_set is None: elec_set = unsorted_unique(electrodes)

    data_type = _spike_data_type(data_sua)

    if data_type == 'timestamp':    pooler_func = _pool_electrode_units_spike_times
    elif data_type == 'bool':       pooler_func = _pool_electrode_units_spike_bool
    else:                           pooler_func = _pool_electrode_units_spike_rate

    extra_args = dict(sort=sort) if data_type == 'timestamp' else {}
    data_mua = pooler_func(data_sua, electrodes, axis=axis, elec_set=elec_set, **extra_args)

    # Generate list of indexes of 1st occurrence of each electrode, if requested
    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return data_mua, elec_idxs
    else:
        return data_mua


#==============================================================================
# Plotting functions
#==============================================================================
def plot_raster(spike_times, ax=None, xlim=None, color='0.25', height=1.0,
                xlabel=None, ylabel=None):
    """
    Generate raster plot of spike times

    Parameters
    ----------
    data : array_like, shape=(n_spikes,) or ndarray, dtype=object (each elem = (n_spikes,) array)
        List(s) of spike timestamps (in s).  Can be given for either a single
        spike train, or for multiple spike trains (eg different trials,
        units, etc.) within an object array of any arbitrary shape.

        NOTE: Unlike other functions, here object array must be 1d and
        boolean spike trains are not supported

    ax : Pyplot Axis object, default: plt.gca()
        Axis to plot into

    xlim : array-like, shape=(2,), default: (auto-set by matlplotlib)
        x-axis limits of plot

    color : Color specifier, default: '0.25' (dark gray)
        Color to plot all spikes in

    height : float, default: 1.0 (each spike height is full range for its row in raster)
        Height of each plotted spike (in fraction of distance btwn spike trains)

    xlabel,ylabel : str, default: (no label)
        x,y-axis labels for plot

    Returns
    -------
    ax : Pyplot Axis object
        Axis for plot
    """
    if ax is None: ax = plt.gca()

    def _plot_raster_line(spike_times, y=0, xlim=None, color='0.25', height=1.0):
        """ Plots single line of raster plot """
        # Extract only spike w/in plotting time window
        if xlim is not None:
            spike_times = spike_times[(spike_times >= xlim[0]) &
                                      (spike_times <= xlim[1])]

        y = np.asarray([[y+height/2.0], [y-height/2.0]])

        plt.plot(spike_times[np.newaxis,:]*np.ones((2,1)),
                 y*np.ones((1,len(spike_times))), '-', color=color,linewidth=1)

    # Plotting multiple spike train (multiple trials or neurons)
    # Plot each spike train as a separate line
    if spike_times.dtype == object:
        n_spike_trains = spike_times.shape[0]
        for i_train,train in enumerate(spike_times):
            if train != []:
                _plot_raster_line(train,i_train,xlim=xlim,
                                  color=color,height=height)

    # Plotting a single spike train
    else:
        n_spike_trains = 1
        _plot_raster_line(spike_times,0,xlim=xlim,color=color,height=height)

    plt.ylim((-0.5,n_spike_trains-0.5))
    if xlim is not None:    plt.xlim(xlim)
    if xlabel is not None:  plt.xlabel(xlabel)
    if ylabel is not None:  plt.ylabel(ylabel)

    plt.show()

    return ax


def plot_mean_waveforms(spike_waves, timepts=None, plot_sd=True, ax=None, **kwargs):
    """
    Plot mean spike waveform for each of one or more units

    Parameters
    ----------
    spike_waves : ndarray, shape=(n_units,), dtype=object (elems=(n_timepts,n_spikes) arrays)
        Spike waveforms for one or more units

    timepts : array-like, shape=(n_timepts,), default: 0:n_timepts
        Common time sampling vector for each spike waveform

    plot_sd: bool, default: True
        If True, also plots standard deviation of waves as fill

    ax : Pyplot Axis object, default: plt.gca()
        Axis to plot into

    **kwargs :
        Any additional keyword args are interpreted as parameters of plt.axes()
        (settable Axes object attributes), plt.plot() (Line2D object attributes),
        or plt.fill() (Polygon object attributes), including the following
        (with given default values):

        xlim : array-like, shape=(2,), default: (timepts[0],timepts[-1])
            x-axis limits

        xticklabels,yticklabels : array-like, default: [] (no labels)
            Labels for x/y ticks

    Returns
    -------
    lines : List of Line2D objects
        ax.plot output. Allows access to line properties of line.

    patches : List of Polygon objects
        ax.fill output. Allows access to patch properties of fill.

    ax : Axis object
        Axis plotted into.
    """
    if spike_waves.dtype != object: spike_waves = _enclose_in_object_array(spike_waves)
    n_units      = len(spike_waves)
    n_timepts    = spike_waves[0].shape[0]

    # If no time sampling vector given, default to 0:n_timepts
    if timepts is None: timepts = np.arange(n_timepts)
    if ax is None: ax = plt.gca()

    # Merge any input parameters with default values
    kwargs = _merge_dicts(dict(xlim=(timepts[0],timepts[-1]),
                               xticklabels=[], yticklabels=[]), kwargs)

    # Compute mean and SD of waveforms for each unit
    mean = np.full((n_units,n_timepts), fill_value=np.nan)
    if plot_sd: sd = np.full((n_units,n_timepts), fill_value=np.nan)
    else:       sd = None

    for unit in range(n_units):
        if spike_waves[unit] is None: continue
        mean[unit,:] = spike_waves[unit].mean(axis=1).T
        if plot_sd: sd[unit,:] = spike_waves[unit].std(axis=1).T

    lines, patches, ax = plot_line_with_error_fill(timepts, mean, err=sd, ax=ax, **kwargs)

    return lines, patches, ax


def plot_waveform_heatmap(spike_waves, timepts=None, ylim=None, n_ybins=20,
                          ax=None, cmap='jet', **kwargs):
    """
    Plot heatmap (2D hist) of all spike waveforms across one or more units

    Parameters
    ----------
    spike_waves : ndarray, shape=(n_units,), dtype=object (elem's are (n_timepts,n_spikes) arrays)
        Spike waveforms for one or more units

    timepts : array-like, shape=(n_timepts,), default: 0:n_timepts
        Common time sampling vector for each spike waveform

    ylim : array-like, shape=(2,), default: [min,max] of given waveforms
        [min,max] waveform amplitude for generating 2D histograms

    n_ybins : int, default: 20
        Number of histogram bins to use for y (amplitude) axis

    ax : Pyplot Axis object, default: plt.gca()
        Axis to plot into

    cmap : str or Colormap object, default: 'jet'
        Colormap to plot heat map in

    Returns
    -------
    ax : Pyplot Axis object
        Axis plotted into
    """
    if spike_waves.dtype != object: spike_waves = _enclose_in_object_array(spike_waves)
    if ax is None: ax = plt.gca()

    # Concatenate waveforms across all units -> (n_timepts,n_spikes_total) ndarray
    ok_idxs = np.asarray([unit_waveforms is not None for unit_waveforms in spike_waves])
    spike_waves = np.concatenate(spike_waves[ok_idxs],axis=1)
    n_timepts,n_spikes = spike_waves.shape

    # If no time sampling vector given, default to 0:n_timepts
    if timepts is None: timepts = np.arange(n_timepts)
    # If no waveform amplitude range given, default to [min,max] of set of waveforms
    if ylim is None: ylim = [np.min(spike_waves),np.max(spike_waves)]

    # Merge any input parameters with default values
    kwargs = _merge_dicts(dict(aspect='auto', xticklabels=[], yticklabels=[]), kwargs)

    # Set histogram bins to sample full range of times,
    dt = np.mean(np.diff(timepts))
    xedges = np.linspace(timepts[0]-dt/2, timepts[-1]+dt/2, n_timepts+1)
    yedges = np.linspace(ylim[0], ylim[1], n_ybins)

    # Compute 2D histogram of all waveforms
    wf_hist = np.histogram2d(np.tile(timepts,(n_spikes,)),
                             spike_waves.T.reshape(-1),
                             bins=(xedges,yedges))[0]
    # Plot heat map image
    y = (yedges[:-1] + yedges[1:])/2
    patch, ax = plot_heatmap(timepts, y, wf_hist.T, ax=ax, **kwargs)

    return patch, ax


# =============================================================================
# Synthetic data generation and testing functions
# =============================================================================
def simulate_spike_rates(gain=5.0, offset=5.0, n_conds=2, n_trials=1000,
                         window=1.0, count=False, seed=None):
    """
    Simulate Poisson spike rates across multiple conditions/groups with given condition effect size

    Parameters
    ----------
    gain : scalar or array-like, shape=(n_conds,), default: 5.0 (5 spk/s btwn-cond diff)
        Spike rate gain (in spk/s) for each condition, which sets effect size.
        If scalar, interpeted as spike rate difference between each successive conditions.
        If array, interpreted as specific spike rate gain over baseline for each condition.
        Set = 0 to simulate no expected difference between conditions.

    offset : scalar, default: 5.0  (5 spk/s baseline)
        Baseline rate added to condition effects

    n_conds : int, default: 2
        Number of distinct conditions/groups to simulate

    n_trials : int, default: 1000
        Number of trials/observations to simulate

    window : scalar, default: 1.0 s
        Time window (in s) to count simulated spikes over.
        Set = 1 if you want spike *counts*, rather than rates.

    count : bool, default: False (compute rates)
        If True, return integer-valued spike counts. If False, return float-valued rates.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Returns
    -------
    rates : ndarray, shape=(n_trials,), dtype=float or int
        Simulated Poisson spike rates

    labels : ndarray, shape=(n_trials,), dtype=int
        Condition/group labels for each trial. Sorted in group order to simplify visualization.
    """
    if seed is not None: set_random_seed(seed)

    # Is gain scalar-valued or array-like?
    scalar_gain = not isinstance(gain, (list, tuple, np.ndarray))

    if not scalar_gain:
        gain = np.asarray(gain)
        assert len(gain) == n_conds, \
            ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                       % (len(gain), n_conds))

    # Create (n_trials,) vector of group labels (ints in set 0-n_conds-1)
    n_reps = ceil(n_trials/n_conds)
    labels = np.tile(np.arange(n_conds),(n_reps,))[0:n_trials]
    # For easy visualization, sort trials by group number
    labels.sort()

    # Per-trial Poisson rate parameters = expected number of spikes in interval
    # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
    if scalar_gain: lambdas = (offset + gain*labels)*window
    # Hand-set gain specific for each condition
    else:           lambdas = (offset + gain[labels])*window

    # Simulate Poisson spike counts, optionally convert to rates
    # Generates Poisson random variables in a way that reproducibly matches output of Matlab
    rates = poisson.ppf(np.random.rand(n_trials), mu=lambdas) / window
    if not count: rates = rates.astype(float) / window

    return rates, labels


def simulate_spike_trains(gain=5.0, offset=5.0, n_conds=2, n_trials=1000, time_range=1.0,
                          refractory=0, seed=None, data_type='timestamp'):
    """
    Simulate Poisson spike trains across multiple conditions/groups
    with given condition effect size

    Parameters
    ----------
    gain : scalar or array-like, shape=(n_conds,), default: 5.0 (5 spk/s btwn-cond diff)
        Spike rate gain (in spk/s) for each condition, which sets effect size.
        If scalar, interpeted as spike rate difference between each successive conditions.
        If array, interpreted as specific spike rate gain over baseline for each condition.
        Set = 0 to simulate no expected difference between conditions.

    offset : scalar, default: 5.0  (5 spk/s baseline)
        Baseline rate added to condition effects

    n_conds : int, default: 2
        Number of distinct conditions/groups to simulate

    n_trials : int, default: 1000
        Number of trials/observations to simulate

    time_range : scalar, default: 1 s
        Full time range (in s) to simulate spike train over

    refractory : scalar, default: 0 (no refractory)
        Absolute refractory period in which a second spike cannot occur after another.
        Set=0 for proper Poisson process with no refractory.

        NOTE: currently implemented by simply deleting spikes < refractory period,
        which affects rates and is thus not optimal

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    data_type : {'timestamp','bool'}, default: 'timestamp'
        Format of output spike trains:

        - 'timestamp' : Spike timestamps in s relative to trial starts
        - 'bool'   : Binary (0/1) vectors flagging spike times

    Returns
    -------
    trains : ndarray, shape=(n_trials,), dtype=object or
        ndarray, shape=(n_trials,n_timepts), dtype=bool
        Simulated Poisson spike trains, returned either as list of timestamps relative
        to trial start or as binary vector for each trial (depending on `data_type`).

    labels : ndarray, shape=(n_trials,), dtype=int
        Condition/group labels for each trial. Sorted in group order to simplify visualization.
    """
    if seed is not None: set_random_seed(seed)

    assert data_type in ['timestamp','bool'], \
        ValueError("Unsupported value '%s' given for <data_type>. Should be 'timestamp' or 'bool'" \
                   % data_type)

    # Is gain scalar-valued or array-like?
    scalar_gain = not isinstance(gain, (list, tuple, np.ndarray))

    if not scalar_gain:
        gain = np.asarray(gain)
        assert len(gain) == n_conds, \
            ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                       % (len(gain), n_conds))

    # Create (n_trials,) vector of group labels (ints in set 0-n_conds-1)
    n_reps = ceil(n_trials/n_conds)
    labels = np.tile(np.arange(n_conds),(n_reps,))[0:n_trials]
    # For easy visualization, sort trials by group number
    labels.sort()

    # Per-trial Poisson rate parameters = expected number of spikes/s
    # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
    if scalar_gain: lambdas = (offset + gain*labels)
    # Hand-set gain specific for each condition
    else:           lambdas = (offset + gain[labels])

    if data_type == 'timestamp':
        trains = np.empty((n_trials,),dtype=object)
    else:
        n_timepts = int(round(time_range*1000))
        trains = np.zeros((n_trials,n_timepts),dtype=bool)

    # Simulate Poisson spike trains with given lambda for each trial
    for i_trial,lam in enumerate(lambdas):
        # Lambda=0 implies no spikes at all, so leave empty
        if lam == 0:
            if data_type == 'timestamp': trains[i_trial] = np.asarray([],dtype=object)
            continue

        # Simulate inter-spike intervals. Poisson process has exponential ISIs,
        # and this is best way to simulate one.
        # HACK Generate 2x expected number of spikes, truncate below
        n_spikes_exp = lam*time_range
        # Generates exponential random variables in a way that reproducibly matches output of Matlab
        ISIs = expon.ppf(np.random.rand(int(round(2*n_spikes_exp))), loc=0, scale=1/lam)

        # HACK Implement absolute refractory period by deleting ISIs < refractory
        # todo More principled way of doing this that doesn't affect rates
        if refractory != 0: ISIs = ISIs[ISIs >= refractory]

        # Integrate ISIs to get actual spike times
        timestamps = np.cumsum(ISIs)
        # Keep only spike times within desired time time_range
        timestamps = timestamps[timestamps < time_range]

        if data_type == 'timestamp':
            trains[i_trial] = timestamps
        # Convert timestamps to boolean spike train
        else:
            idxs = np.floor(timestamps*1000).astype('int')
            trains[i_trial,idxs] = True

    return trains, labels


def simulate_spike_waveforms(trough_time=0.3e-3, peak_time=0.75e-3, trough_amp=0.4, peak_amp=0.25,
                             trough_width=0.2e-3, peak_width=0.4e-3, time_range=(-0.2e-3,1.4e-3),
                             smp_rate=30e3, noise=0.01, n_spikes=100, seed=None):
    """
    Simulate set of spike waveforms with given shape and amplitude parameters

    Parameters
    ----------
    trough_time,peak_time : float, default: 0.5 ms, 1 ms
        Time of waveform depolarization trough, after-hyperpolarization peak
        (in s relative to t=0 at simulated trigger time)

    trough_amp,peak_amp : float, default: 0.4, 0.25
        Absolute amplitude of waveform depolarization trough, after-hyperpolarization peak (in mV)

    trough_width,peak_width : float, default: 0.25 ms, 0.5 ms
        Width (2*Gaussian SD in s) of waveform depolarization trough, after-hyperpolarization peak

    time_range : array-like, shape=(2,), default: (-0.5 ms, 1.5 ms)
        Full time range that spike waveforms are sampled on ((start,end) relative to trigger in s)

    smp_rate : float, default: 30 kHz
        Sampling rate of spike waveforms (in Hz)

    noise : float, default: 0.01
        Amplitude of additive Gaussian noise (in mV)

    n_spikes : int, default: 100
        Total number of spike waveforms to simulate

    seed  : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Returns
    -------
    spike_waves : ndarray, shape=(n_timepts,n_spikes)
        Set of simulated spike waveforms, based on given parameters + noise

    timepts : ndarray, shape=(n_timepts)
        Time sampling vector for all simulated waveforms (in s)
    """
    if seed is not None: set_random_seed(seed)

    dt = 1/smp_rate
    timepts = iarange(time_range[0], time_range[1], dt)
    n_timepts = len(timepts)

    # Depolarization trough
    trough_waveform = gaussian_1d(timepts, center=trough_time, width=trough_width/2, amplitude=trough_amp)
    # After-hyperpolarization peak
    peak_waveform = gaussian_1d(timepts, center=peak_time, width=peak_width/2, amplitude=peak_amp)
    # Mean overall spike waveform = linear superposition of trough + peak
    mean_waveform = -trough_waveform + peak_waveform

    # Replicate mean waveform to desired number of spikes and add in random Gaussian noise
    waveforms = (np.tile(mean_waveform[:,np.newaxis], (1,n_spikes)) +
                 noise * np.random.randn(n_timepts, n_spikes))

    return waveforms, timepts


#==============================================================================
# Other helper functions
#==============================================================================
def _spike_data_type(data):
    """
    Determine what type of spiking data we have:

    - 'timestamp' : spike timestamps (in a Numpy object array or list)
    - 'bool' :      binary spike train (1=spike, 0=no spike at each timepoint)
    - 'rate' :      array of spike rate/counts
    """
    data = np.asarray(data)
    # Data is boolean or contains only 0/1 values -> 'bool'
    if _isbinary(data):
        return 'bool'
    # Data is object array or monotonically-increasing 1D array/list -> 'timestamp'
    elif (data.dtype == 'object') or ((data.ndim == 1) and ((data.sort() == data).all())):
        return 'timestamp'
    # Otherwise (general numeric array) -> 'rate'
    elif np.issubdtype(data.dtype,np.number):
        return 'rate'
    else:
        raise ValueError("Could not identify data type of given data")


def _remove_buffer(data, buffer, axis=-1):
    """
    Removes a temporal buffer (eg zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    Parameters
    ----------
    data : ndarray
        Data array where a buffer has been appended on both ends of time
        dimension. Can be any arbitrary size, typically
        (n_trials,n_units,n_timepts+2*buffer).

    buffer : scalar
        Length (number of samples) of buffer appended to each end.

    axis : int, default: -1
        Array axis to remove buffer from (ie time dim)

    Returns
    -------
    data : ndarray
        Data array with buffer removed, reducing time axis to n_timepts
        (typically shape (n_trials,n_units,n_timepts))
    """
    if axis < 0: axis = data.ndim + axis

    if axis == data.ndim-1:
        return data[...,buffer:-buffer]
    else:
        return (data.swapaxes(-1,axis)[...,buffer:-buffer]
                    .swapaxes(axis,-1))


def _cut_trials_spike_times(data, trial_lims, trial_refs=None):
    """ Cut spike timestamp data into trials """
    n_trials = trial_lims.shape[0]

    do_ref = trial_refs is not None

    # If data is not an object array, its assumed to be a single spike train
    single_train = isinstance(data,list) or (data.dtype != object)

    # Enclose in object array to simplify computations; removed at end
    if single_train: data = _enclose_in_object_array(np.asarray(data))


    # Create 1D flat iterator to iterate over arbitrary-shape data array
    # Note: This always iterates in row-major/C-order regardless of data order, so all good
    data_flat = data.flat

    # Create array to hold trial-cut data. Same shape as data, with trial axis appended.
    cut_data = np.empty((*data.shape,n_trials),dtype=object)

    for _ in range(data.size):
        coords = data_flat.coords   # Multidim coordinates into data array
        data_cell = data[coords]    # Timestamps for current array cell (unit,etc.)

        # Find and extract all spikes in each trial for given element in data
        for trial,lim in enumerate(trial_lims):
            trial_bool = (lim[0] <= data_cell) & (data_cell < lim[1])
            # Note: Returns empty array if no spikes
            trial_spikes = data_cell[trial_bool]
            # Re-reference spike times to within-trial reference time (if requested)
            if do_ref: trial_spikes -= trial_refs[trial]
            cut_data[(*coords,trial)] = trial_spikes

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(data_flat)

    return cut_data


def _cut_trials_spike_bool(data, trial_lims, smp_rate=None, axis=0):
    """ Cut binary spike train data into trials """
    assert smp_rate is not None, "For binary spike train data, must input value for <smp_rate>"

    n_trials = trial_lims.shape[0]

    # Convert trial_lims in s -> indices into continuous data samples
    trial_idxs = np.round(smp_rate*trial_lims).astype(int)
    assert trial_idxs.min() >= 0, \
        ValueError("trial_lims are attempting to index (%d) before start of data (0)" %
                   trial_idxs.min())
    assert trial_idxs.max() < data.shape[axis], \
        ValueError("trial_lims are attempting to index (%d) beyond end of data (%d)" %
                   (trial_idxs.max(),data.shape[axis]))
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


def _realign_spike_times(spike_times, align_times, trial_axis):
    """ Realign trial-cut spike timestamps to new set of within-trial times """
    # Make copy of input data to avoid changing in caller
    spike_times = spike_times.copy()

    # Move trial axis to first axis of array
    if trial_axis != 0: spike_times = np.moveaxis(spike_times, trial_axis, 0)

    # Subtract new reference time from all spike times for each trial
    for trial,align_time in enumerate(align_times):
        spike_times[trial,...] = spike_times[trial,...] - align_time

    # Move trial axis to original location
    if trial_axis != 0: spike_times = np.moveaxis(spike_times, 0, trial_axis)

    return spike_times


def _realign_spike_bool(data, align_times, trial_axis,
                        time_axis=None, timepts=None, time_range=None):
    """ Realign trial-cut binary (1/0) spiking data to new set of within-trial times """
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
    realigned       = np.empty(return_shape,dtype=data.dtype)

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


def _pool_electrode_units_spike_times(data_sua, electrodes, axis, elec_set, sort=True):
    """ Pool (concatenate) spike timestamps across all single units on each electrode """
    n_elecs = len(elec_set)

    # Reshape spike data array -> 2D matrix (n_dataseries,n_units)
    data_sua,data_shape = standardize_array(data_sua, axis=axis, target_axis=-1)
    n_series,_ = data_sua.shape

    data_mua = np.empty((n_series,n_elecs),dtype=object)

    for i_elec,elec in enumerate(elec_set):
        # Find all units on given electrode
        elec_idxs   = electrodes == elec

        for i_series in range(n_series):
            # Concatenate spike_times across all units for current data series
            # -> (n_spikes_total,) ndarray
            data_mua[i_series,i_elec] = \
                np.concatenate([np.reshape(ts, (-1,)) for ts in data_sua[i_series,elec_idxs]])
            # Sort timestamps so they remain in sequential order after concatenation
            if sort: data_mua[i_series,i_elec].sort()

    # Reshape output data array to original shape (now with len(data[axis] = n_elecs)
    return undo_standardize_array(data_mua, data_shape, axis=axis, target_axis=-1)


def _pool_electrode_units_spike_bool(data_sua, electrodes, axis, elec_set):
    """ Pool (OR) boolean spike train data across all units on each electrode """
    n_elecs = len(elec_set)
    data_shape = list(data_sua.shape)
    data_shape[axis] = n_elecs

    data_mua = np.empty(tuple(data_shape),dtype=bool)

    slicer_sua  = [slice(None)]*data_sua.ndim
    slicer_mua  = [slice(None)]*data_sua.ndim

    for i_elec,elec in enumerate(elec_set):
        # Find all units on given electrode
        elec_idxs   = electrodes == elec
        slicer_sua[axis] = elec_idxs    # Extract current electrode units from sua
        slicer_mua[axis] = i_elec       # Save pooled data to current electrode in mua

        data_mua[tuple(slicer_mua)] = data_sua[tuple(slicer_sua)].any(axis=axis,keepdims=False)

    return data_mua


def _pool_electrode_units_spike_rate(data_sua, electrodes, axis, elec_set):
    """ Pool (sum) spike rate/count data across all units on each electrode """
    n_elecs = len(elec_set)
    data_shape = list(data_sua.shape)
    data_shape[axis] = n_elecs

    data_mua = np.empty(tuple(data_shape),dtype=data_sua.dtype)

    slicer_sua  = [slice(None)]*data_sua.ndim
    slicer_mua  = [slice(None)]*data_sua.ndim

    for i_elec,elec in enumerate(elec_set):
        # Find all units on given electrode
        elec_idxs   = electrodes == elec
        slicer_sua[axis] = elec_idxs    # Extract current electrode units from sua
        slicer_mua[axis] = i_elec       # Save pooled data to current electrode in mua

        data_mua[tuple(slicer_mua)] = data_sua[tuple(slicer_sua)].sum(axis=axis,keepdims=True)

    return data_mua

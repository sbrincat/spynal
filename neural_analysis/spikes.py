# -*- coding: utf-8 -*-
"""
spikes  A module for basic analyses of neural spiking activity

FUNCTIONS
### Spike count/rate computation ###
rate                Wrapper around all rate estimation functions
bin_rate            Computes spike counts/rates in series of time bins (regular or not)
density             Computes spike density (smoothed rate) with given kernel

### Rate and inter-spike interval stats ###
rate_stats          Computes given statistic on spike rate data
isi                 Computes inter-spike intervals from spike data
isi_stats           Computes given statistic on inter-spike interval data

### Preprocessing ###
times_to_bool       Converts spike timestamps to binary spike trains
bool_to_times       Converts binary spike train to timestamps

cut_trials          Cuts spiking data into trials
realign_data        Realigns data to new within-trial times (new t=0)
pool_electrode_units Pools all units on each electrode into a multi-unit

### Plotting ###
plot_raster         Generates a raster plot
plot_mean_waveforms Plots mean spike waveforms from one/more units
plot_waveform_heatmap Plots heatmap (2d histogram) of spike waveforms

### Synthetic data generation ###
simulate_spike_rates    Generates sythetic Poisson rates
simulate_spike_trains   Generates sythetic Poisson process spike trains


Created on Mon Aug 13 14:38:34 2018

@author: sbrincat
"""
import os
from math import isclose, ceil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.signal.windows import hann, gaussian
from scipy.stats import poisson, bernoulli, norm, expon

try:
    from .utils import set_random_seed, iarange, unsorted_unique, index_axis, \
                       standardize_array, undo_standardize_array, setup_sliding_windows, \
                       concatenate_object_array, fano, cv, cv2, lv
    from .helpers import _check_window_lengths, _enclose_in_object_array
# TEMP    
except ImportError:
    from neural_analysis.utils import set_random_seed, iarange, unsorted_unique, index_axis, \
                      standardize_array, undo_standardize_array, setup_sliding_windows, \
                      concatenate_object_array, fano, cv, cv2, lv
    from neural_analysis.helpers import _check_window_lengths, _enclose_in_object_array
    

# =============================================================================
# Spike count/rate computation and statistics functions
# =============================================================================
def rate(data, method='bin', **kwargs):
    """
    Wrapper function for computing spike rates using given method
    
    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                -or-
                (...,n_timepts,...) array of bool. Binary/boolean representation of 
                spike times, for either a single or multiple spike trains.
    
    method      String. Spike rate estimation method.
                'bin'       Traditional rectangular-binned rate (aka PSTH) [default]
                'density'   Kernel density estimator for smoothed spike rate
                
    **kwargs    Any further arguments passed as-is to rate computation function.
                See there for details.
                
    RETURNS
    rates       (...,n_timepts) ndarray | (...,n_timepts,...) ndarray.
                Estimated spike rates (in spk/s) using given method.
                    
    timepts     For bin: (n_timepts,) ndarray. Time sampling vector (in s).
                For density: (n_bins,2) ndarray. [start,end] of each time bin (in s).                               
    """
    if method in ['bin','bins','bin_rate','psth']:      rate_func = bin_rate
    elif method in ['density','spike_density','sdf']:   rate_func = density
    
    return rate_func(data, **kwargs)
    
    
def bin_rate(data, lims=None, width=50e-3, step=None, bins=None, count=False,
             axis=-1, timepts=None):
    """
    Computes spike rate/count within given sequence of hard-edged time bins
    
    Spiking data can be timestamps or binary (0/1) spike trains
    
    Use <lims,width,step> to set standard-width sliding window bins or
    use <bins> to set any arbitrary custom time bins

    rates,bins = bin_rate(data,lims=None,width=50e-3,step=<width>,bins=None,count=False,
                          axis=-1,timepts=None)

    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                -or-
                (...,n_timepts,...) array of bool. Binary/boolean representation of 
                spike times, for either a single or multiple spike trains.

    lims        (2,) array-like. Full time range of analysis ([start,end] in s).
                Must input a value (unless explicitly setting custom <bins>)
                
    width       Scalar. Full width of each time bin (s). Default: 50 ms
    
    step        Scalar. Spacing between successive time bins (s). 
                Default: <width> (each bin starts at end of previous bin)
                
    *** Alternatively, any custom time bins may be explicitly input using <bins> arg ***
    
    bins        (n_bins,2) array-like. [start,end] of each custom time bin (in s).
                Bins can have any arbitrary width and spacing.
                Default: bins with given <width,step>, ranging from lims[0] to lims[1]
                
    count       Bool. If True, returns integer-valued spike counts, rather than rates.
                Default: False (computes rates)         
                
    axis        Int. Axis of binary data corresponding to time dimension.
                Not used for spike timestamp data. Default: -1 (last axis of array)           

    timepts     (n_timepts,) ndarray. Time sampling vector (in s) for binary data.
                Not used for spike timestamp data, but MUST be input for binary data.
                                
    RETURNS
    rates       (...,n_bins) ndarray | (...,n_bins,...) ndarray. Spike rates (in spk/s) 
                or spike counts in each time bin (and for each trial/unit/etc. in <data>). 
                For timestamp inputs, same shape as <data>, with time-bin axis appended to end.
                If only a single spike train is input, output is (n_bins,) vector.
                For boolean inputs, rates has same shape as data with time axis length = n_bins.
                dtype is uint16 if <count> is True, float otherwise.
                
    bins        (n_bins,2) ndarray. [start,end] of each time bin (in s).
    
    NOTES
    Spikes are counted within each bin including the start, but *excluding* the end
    of the bin. That is each bin is defined as [start,end).
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
    
    assert (bins.ndim == 2) and (bins.shape[1] == 2), \
        ValueError("bins must be given as (n_bins,2) array of bin [start,end] times")
        
    n_bins  = bins.shape[0]
    widths  = np.diff(bins,axis=1).squeeze()
    
    # Are bins "standard"? : equal-width, with start of each bin = end of previous bin
    std_bins = np.allclose(widths,widths[0]) and np.allclose(bins[1:,0],bins[:-1,1])
    
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
    dtype = 'uint16' if count else float
    rates = np.empty((*data.shape,n_bins),dtype=dtype)
    
    for _ in range(data.size):
        # Multidim coordinates into data array
        coords = data_flat.coords
        
        # Count spikes within each bin
        rates[(*coords,slice(None))] = count_spikes(data[coords],bins_)

        # Iterate to next element (list of spike times for trial/unit/etc.) in data 
        next(data_flat)        
                    
    # Normalize all spike counts by bin widths to get spike rates
    if not count: rates = rates / widths         
        
    # If only a single spike train was input, squeeze out singleton axis 0
    if single_train: rates = rates.squeeze(axis=0)
    
    # For boolean input data, shift time axis back to its original location in array
    if (data_type == 'bool') and (axis != rates.ndim): rates = np.moveaxis(rates,-1,axis)
    
    return rates, bins
    
      
psth = bin_rate
""" Aliases function bin_rate as psth """


def density(data, kernel='gaussian', width=50e-3, lims=None, smp_rate=1000, 
            buffer=None, downsmp=1, axis=-1, timepts=None, **kwargs):
    """
    Computes spike density function via convolution with given kernel/width

    Spiking data can be timestamps or binary (0/1) spike trains

    rates,timepts = density(data,kernel='gaussian',width=50e-3,lims=None,smp_rate=1000,
                      buffer=None,downsmp=1,axis=0,
                      timepts=np.arange(lims[0],lims[1]+1/smp_rate,1/smp_rate),**kwargs)

    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                -or-
                (...,n_timepts,...) ndarray of bool. Binary/boolean representation of 
                spike times, for either a single or multiple spike trains.                
                
    kernel      String. Name of convolution kernel to use:
                'gaussian' [default] | 'hanning'
                Also includes (not well tested) functionality to input the
                kernel itself as an arrray, or a custom function that takes
                a "width" argument (+ any extra kwargs).

    width       Scalar. Width parameter for given kernel. Default: 50 ms
                Interpretation is kernel-specific.
                'gaussian' : <width> = 1 Gaussian standard deviation
                'hanning'  : <width> = kernel half-width (~ 2.53x Gaussian SD)

    lims        (2,) array-like. Full time range of analysis (in s).

    smp_rate    Scalar. Sampling rate (Hz; 1/sample period) for spike density. Default: 1000

    buffer      Float. Length (in s) of symmetric buffer to add to each end
                of time dimension (and trim off before returning) to avoid edge
                effects. Default: (kernel-dependent, approximates length of
                edge effects induced by kernel)
                
    downsmp     Int. Factor to downsample time sampling by (after spike density computation).
                eg, smp_rate=1000 (dt=0.001), downsmp=10 -> smpRateOut=100 (dt=0.01)
                Default: 1 (no downsampling)
                
    axis        Int. Axis of binary data corresponding to time dimension.
                Not used for spike timestamp data. Default: -1 (last axis of array)           

    timepts     (n_timepts,) ndarray. Time sampling vector (in s) for binary input data 
                and/or computed rates. MUST be input for binary data. For timestamp data,
                defaults to np.arange(lims[0], lims[1]+1/smp_rate, 1/smp_rate) (ranging btwn
                lims, in increments of 1/smp_rate)
                                
    **kwargs    All other kwargs passed directly to kernel function

    (any additional kwargs passed directly to kernel function)

    RETURNS
    rates       (...,n_timepts) ndarray | (...,n_timepts,...) ndarray.
                Spike density function -- smoothed spike rates (in spk/s) estimated at each 
                timepoint (and for each trial/unit/etc. in <data>). 
                For timestamp inputs, rates has same shape as data with time axis appended to end.
                If only a single spike train is input, output is (n_timepts,) vector.
                For boolean inputs, rates has same shape as data.
                    
    timepts     (n_timepts,) ndarray. Time sampling vector (in s) for rates
    """
    data_type = _spike_data_type(data)
    if axis < 0: axis = data.ndim + axis
    
    if data_type == 'bool':
        assert timepts is not None, "For binary spike train data, a time sampling vector <timepts> MUST be given"
                
    # Set default buffer based on overlap of kernel used
    if buffer is None:
        if kernel in ['hann','hanning']:        buffer = width
        elif kernel in ['gaussian','normal']:   buffer = 3*width
        else:                                   buffer = 0
    
    # Set time sampling from smp_rate,lims
    if timepts is None: timepts = np.arange(lims[0], lims[1]+1e-12, 1/smp_rate)
        
    # Compute sampling rate from input time sampling vector
    dt = np.diff(timepts).mean()
    smp_rate = round(1/dt)

    if smp_rate < 500: 
        print('Warning: Sampling of %d Hz will likely lead to multiple spikes in bin binarized to 0/1' % round(smp_rate))
        
    # Add buffer to time sampling vector and data to mitigate edge effects
    if buffer != 0:
        n_buffer = int(round(buffer*smp_rate))    # Convert buffer from time units -> samples
        # Extend time sampling by n_buffer samples on either end (for both data types)
        timepts = np.concatenate((np.flip(np.arange(timepts[0]-dt, timepts[0]-n_buffer*dt-1e-12, -dt)),
                                  timepts,
                                  np.arange(timepts[-1]+dt, timepts[-1]+n_buffer*dt+1e-12, dt)))
        # For bool data, reflectively resample edges of data to create buffer
        if data_type == 'bool':
            n_samples = data.shape[axis]
            idxs = np.concatenate((np.flip(np.arange(n_buffer)+1), np.arange(n_samples), 
                                   np.arange(n_samples-2, n_samples-2-n_buffer, -1)))
            data = data.take(idxs,axis=axis)           
                        
    # Convert kernel specifier to actual kernel (window) function 
    width_smps = width*smp_rate # convert width to samples    
    # Kernel is already a (custom) array of values
    if isinstance(kernel,np.ndarray):       kernel_ = kernel
    
    # Kernel is a function/callable -- call it with width in samples
    elif callable(kernel):                  kernel_ = kernel(width_smps,**kwargs)
    
    # Kernel is a string specifier -- call appropriate kernel-generating function
    elif isinstance(kernel,str):
        kernel = kernel.lower()

        if kernel in ['hann','hanning']:
            kernel_ = hann(int(round(width_smps*2.0)))
        elif kernel in ['gaussian','normal']:
            kernel_ = gaussian(int(round(width_smps*6.0)),width_smps)
        else:
            raise ValueError("Unsupported value '%s' given for kernel. \
                            Should be 'hanning'|'gaussian'" % kernel)
    
    # Normalize kernel to integrate to 1
    kernel_ = kernel_ / (kernel_.sum()/smp_rate)
        
    # Convert spike times to binary spike trains -> (...,n_timepts)
    if data_type == 'timestamp':
        bins = np.stack((timepts-dt/2, timepts+dt/2), axis=1)
        data,timepts = times_to_bool(data,bins=bins)
        axis = data.ndim
    else:
        # Reshape data so that time axis is -1 (end of array)
        if axis != data.ndim: data = np.moveaxis(data,axis,-1)

    # Ensure kernel broadcasts against data
    slicer = tuple([np.newaxis]*(data.ndim-1) + [slice(None)])
    
    # Compute density as convolution of spike trains with kernel
    # Note: 1d kernel implies 1d convolution across multi-d array data
    rates = convolve(data,kernel_[slicer],mode='same')

    # Remove any time buffer from spike density and time sampling vector
    # (also do same for data bc of 0-fixing bit below)
    if buffer != 0:
        data        = _remove_buffer(data,n_buffer,axis=-1)
        rates       = _remove_buffer(rates,n_buffer,axis=-1)
        timepts     = _remove_buffer(timepts,n_buffer,axis=-1)
        
    # Implement any temporal downsampling of rates    
    if downsmp != 1:
        rates       = rates[...,0::downsmp]
        timepts     = timepts[0::downsmp]

    # KLUDGE Sometime trials/neurons/etc. w/ 0 spikes end up with tiny non-0 values
    # due to floating point error in fft routines. Fix by setting = 0.
    no_spike_idxs   = ~data.any(axis=-1,keepdims=True)
    shape           = tuple([1]*(rates.ndim-1) + [rates.shape[-1]])    
    rates[np.tile(no_spike_idxs,shape)] = 0    
    
    # Reshape rates so that time axis is in original location
    if (data_type == 'bool') and (axis != data.ndim):
        rates = np.moveaxis(rates,-1,axis)
        
    return rates, timepts


#==============================================================================
# Rate and inter-spike interval statistics functions
#==============================================================================
def rate_stats(rates, stat='Fano', axis=None, **kwargs):
    """
    Computes given statistic on spike rates of one or more spike trains
    
    Input data must be spike rates, eg as computed using rate()
    
    Stats may be along one/more array axes (eg trials) or across entire data array
       
    stats = rate_stats(rates, stat='Fano', axis=None, **kwargs)
           
    ARGS
    rates       (...,n_obs,...) ndarray. Spike rate data. Shape arbitrary.
    
    stat        String. Rate statistic to compute. Options:
                'Fano' :    Fano factor = var(rate)/mean(rate) [default]
                'CV' :      Coefficient of Variation = SD(rate)/mean(rate)
                    
    axis        Int. Array axis to compute rate statistics along (usually corresponding
                to distict trials/observations). If None [default], computes statistic
                across entire array (analogous to np.mean/var).
                
    **kwargs    Any additional keyword args passed directly to statistic computation function
          
    RETURNS
    stats       Float | (...,1,...) ndarray. Rate statistic(s) computed on data.
                For vector data or axis=None, a single scalar value is returned.
                Otherwise, it's an array w/ same shape as <rates>, but with <axis>
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


def rate_fano(rates, axis=None, **kwargs):
    """
    Computes Fano factor (variance/mean) of rates of one or more spike trains    
    Convenience function that calls rate_stats(stat='Fano'). See there for details.
    """
    return rate_stats(rates, stat='Fano', axis=axis, **kwargs)

def rate_cv(rates, axis=None, **kwargs):
    """
    Computes Coefficient of Variation (SD/mean) of rates of one/more spike trains    
    Convenience function that calls rate_stats(stat='CV'). See there for details.
    """
    return rate_stats(rates, stat='CV', axis=axis, **kwargs)


def isi(data, axis=-1, timepts=None):
    """
    Computes inter-spike intervals of one or more spike trains
    
    Spiking data can be timestamps or binary (0/1) spike trains
    
    ISIs = isi(data,axis=-1,timepts=None)

    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                -or-
                (...,n_timepts,...) array of bool. Binary/boolean representation of 
                spike times, for either a single or multiple spike trains.

    axis        Int. Axis of binary data corresponding to time dimension.
                Not used for spike timestamp data. Default: -1 (last axis of array)           

    timepts     (n_timepts,) ndarray. Time sampling vector (in s) for binary data.
                Not used for spike timestamp data, but MUST be input for binary data.
                                
    RETURNS
    ISIs        (n_spikes-1,) array | object array of (n_spikes-1,) arrays. Time intervals
                 between each successive pair of spikes in data (in same time units as data).
                For boolean inputs, output is converted to timestamp-like configuration  
                with time axis removed.                     
                For timestamp inputs, same shape as <data> (but with 1 fewer item per array cell)
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

# Alias isi() as interspike_interval()
interspike_interval = isi


def isi_stats(ISIs, stat='Fano', axis='each', **kwargs):
    """
    Computes given statistic on inter-spike intervals of one or more spike trains
   
    Input data must be inter-spike intervals, eg as computed using isi()
   
    Can request data to be pooled along one/more axes (eg trials) before stats computation
    
    stats = isi_stats(ISIs,stat='Fano',axis='each',**kwargs)

    ARGS
    ISIs        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of inter-spike intervals (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                
    stat        String. ISI statistic to compute. Options:
                'Fano' :    Fano factor = var(ISIs)/mean(ISIs) [default]
                'CV' :      Coefficient of Variation = SD(ISIs)/mean(ISIs)
                'CV2' :     Local Coefficient of Variation (Holt 1996 J Neurophys)
                'LV' :      Local Variation (Shinomoto 2009 PLoS Computational Biology)
                'burst_fract' : Fraction of spikes that are in bursts (measure of burstiness)
                
                CV2 and LV and CV-like measures that reduce influence of changes in 
                spike rate on the metric by only measuring local variation (between
                temporally adjacent ISIs). See their specific functions for details.
                
    axis        Int | String. Axis of ISI data to pool ISIs along before computing stat.
                eg, for data that is shape (n_trials,n_units), if you want to compute
                a stat value for each unit, pooled across all trials, you'd set axis=0.
                If axis=None, ISIs are pooled across the *entire* data array.
                If axis='each', stats are computed separately for each spike train in the array.
                NOTE: For locality-sensitive stats ('CV2','LV'), axis MUST = 'each'.
                Default: 'each' (note this is the opposite of default for rate_stats) 

    RETURNS
    stats       Float | array of floats. Given ISI stat, computed on ISI data.
                Returns as a single scalar if axis=None. Otherwise returns as array of
                same shape as ISIs, but with <axis> reduced to singleton.
    """
    stat = stat.lower()
    
    if stat in ['cv2','lv']:
        assert axis == 'each', \
            ValueError("For locality-sensitive stats (CV2,LV), no pooling is allowed (axis must = 'each')")
    
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


def isi_fano(ISIs, axis=None, **kwargs):
    """ Convenience function that calls isi_stats(stat='Fano'). See there for details. """
    return isi_stats(ISIs, stat='Fano', axis=axis, **kwargs)

def isi_cv(ISIs, axis=None, **kwargs):
    """ Convenience function that calls isi_stats(stat='CV'). See there for details. """
    return isi_stats(ISIs, stat='CV', axis=axis, **kwargs)

def isi_cv2(ISIs, axis=None, **kwargs):
    """ Convenience function that calls isi_stats(stat='CV2'). See there for details. """
    return isi_stats(ISIs, stat='CV2', axis=axis, **kwargs)

def isi_lv(ISIs, axis=None, **kwargs):
    """ Convenience function that calls isi_stats(stat='LV'). See there for details. """
    return isi_stats(ISIs, stat='LV', axis=axis, **kwargs)
        
def isi_burst_fract(ISIs, axis=None, **kwargs):
    """ Convenience function that calls isi_stats(stat='LV'). See there for details. """
    return isi_stats(ISIs, stat='burst_fract', axis=axis, **kwargs)
        
def burst_fract(ISIs, crit=0.020):
    """
    Computes measure of burstiness of ISIs of a spike train = fraction of all
    ISIs that are within a spike burst (ISI < 20 ms by default)
    
    ARGS
    ISIs    Array-like. List of inter-spike intervals for a single spike train
    crit    Float. Criterion ISI value (s) to discriminate burst vs non-burst spikes
            Default: 20 ms
            
    RETURNS
    burst   Float. Fraction of all spikes that are within spike bursts.      
    """
    return (ISIs < crit).sum() / ISIs.size

        
#==============================================================================
# Plotting functions
#==============================================================================
def plot_raster(spike_times, ax=None, xlim=None, color='0.25', height=1.0,
                xlabel=None, ylabel=None):
    """
    Plots rasters.  Currently super slow, needs to be optimized (TODO)
    
    ARGS
    spike_times (n_spikes,) array-like | (n_trains,) object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array. Unlike other functions,
                here object array must be 1d.                
    
    ax          Axes object. Axis to plot into. Default: plt.gca()
    xlim        (2,) array-like. x-axis limits of plot. Default: (auto-set)
    color       Color specifier. Color to plot all spikes in. Default: '0.25' (dark gray)
    height      Float. Height of each plotted spike (in fraction of distance btwn spike trains)
                Default: 1.0 (each spike height is full range for its row in raster)
    x/ylabel    String. x/y-axis labels for plot. Default: (no label)
    
    ACTION      Plots raster plot from spike time data
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


def plot_mean_waveforms(spike_waves, timepts=None, sd=True,
                        ax=None, plot_colors=None):
    """
    Plots mean spike waveform for each of one or more units

    ax = plot_mean_waveforms(spike_waves,timepts=None,sd=True,
                             ax=None,plot_colors=None)

    ARGS
    spike_waves (n_units,) object array of (n_timepts,n_spikes) arrays.
                Spike waveforms for one or more units

    timepts     (n_timepts,) array-like. Common time sampling vector for each
                spike waveform. Default: 0:n_timepts

    sd          Bool. If True, also plots standard deviation of waves as fill.
                Default: True

    plot_colors (n_units_max,3) array. Color to plot each unit in.
                Default: Colors from Blackrock Central Spike grid plots

    ax          Pyplot Axis instance. Axis to plot into. Default: plt.gca()
    """
    # TODO  Setup to handle single array of 1 unit's wfs instead of object array of units
    n_units      = len(spike_waves)
    n_timepts    = spike_waves[0].shape[0]

    # If no time sampling vector given, default to 0:n_timepts
    if timepts is None: timepts = np.arange(n_timepts)
    if plot_colors is None:
        # Default plot colors from Blackrock Central software Spike plots
        plot_colors = np.asarray([[1,1,1], [1,0,1], [0,1,1], [1,1,0]])
    if ax is None: ax = plt.gca()
    plt.sca(ax)

    for unit in range(n_units):
        if spike_waves[unit] is None:  continue

        mean = np.mean(spike_waves[unit], axis=1)

        if sd:
            sd = np.std(spike_waves[unit], axis=1)
            ax.fill(np.hstack((timepts,np.flip(timepts))),
                    np.hstack((mean+sd,np.flip(mean-sd))),
                    facecolor=plot_colors[unit,:], edgecolor=None, alpha=0.25)

        ax.plot(timepts, mean, '-', color=plot_colors[unit,:], linewidth=1)

    ax.set_xlim(timepts[0],timepts[-1])
    ax.set_facecolor('k')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_waveform_heatmap(spike_waves, timepts=None, wf_range=None,
                          ax=None, cmap=None):
    """
    Plots heatmap (2D hist) of all spike waveforms across one or more units

    ax = plot_waveform_heatmap(spike_waves,timepts=None,wf_range=None,
                               ax=None,cmap=None)

    ARGS
    spike_waves (n_units,) object array of (n_timepts,n_spikes) arrays.
                Spike waveforms for one or more units

    timepts     (n_timepts,) array-like. Common time sampling vector for each spike
                waveform. Default: 0:n_timepts

    wf_range    (2,) array-like. [min,max] waveform amplitude for generating
                2D histograms. Default: [min,max] of given waveforms

    ax          Pyplot Axis instance. Axis to plot into. Default: plt.gca()
    cmap        String | Colormap object. Colormap to plot heat map. Default: jet

    RETURNS
    ax          Pyplot Axis instance. Axis for plot
    """
    # TODO  Setup to handle single array of 1 unit's wfs instead of object array of units
    if cmap is None:    cmap = 'jet'
    if ax is None:      ax = plt.gca()
    plt.sca(ax)

    # Concatenate waveforms across all units -> (n_timepts,n_spikes_total) ndarray
    ok_idxs = np.asarray([unitWaveforms != None for unitWaveforms in spike_waves])
    spike_waves = np.concatenate(spike_waves[ok_idxs],axis=1)
    n_timepts,n_spikes = spike_waves.shape

    # If no time sampling vector given, default to 0:n_timepts
    if timepts is None: timepts = np.arange(n_timepts)
    # If no waveform amplitude range given, default to [min,max] of set of waveforms
    if wf_range is None: wf_range = [np.min(spike_waves),np.max(spike_waves)]


    # Set histogram bins to sample full range of times,
    xedges = np.linspace(timepts[0],timepts[-1],n_timepts)
    yedges = np.linspace(wf_range[0],wf_range[1],20)

    # Compute 2D histogram of all waveforms
    wf_hist = np.histogram2d(np.tile(timepts,(n_spikes,)),
                             spike_waves.T.reshape(-1),
                             bins=(xedges,yedges))[0]
    # Plot heat map image
    plt.imshow(wf_hist.T, cmap=cmap, origin='lower',
               extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])

    ax.set_xlim(timepts[0],timepts[-1])
    ax.set_ylim(wf_range[0],wf_range[-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('auto')

    return ax


# =============================================================================
# Preprocessing/Utility functions
# =============================================================================
def bool_to_times(spike_bool, timepts, axis=-1):
    """
    Converts boolean (binary) spike train representaton to spike timestamps
    Inverse function of times_to_bool()

    spike_times = bool_to_times(spike_bool,timepts)

    ARGS
    spike_bool  (...,n_timepts,...) ndarray of bools. Binary spike trains,
                where 1 indicates >= 1 spike in time bin, 0 indicates no spikes.

    timepts     (n_timepts,) ndarray. Time sampling vector for data 
                (center of each time bin used to compute binary representation).
                
    axis        Int. Axis of data corresponding to time dimension. Default: -1 (last axis)              

    RETURNS
    spike_times Object ndarray of (n_spikes[cell],) ndarrays | (n_spikes,) ndarray
                Spike timestamps (in same time units as timepts), for each spike train in input.
                Returns as vector-valued array of timestamps if input is single spike train,
                otherwise as object array of variable-length timestamp vectors
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
    Converts spike timestamps to boolean (binary) spike train representaton
    Inverse function of bool_to_times()

    spike_bool,timepts = times_to_bool(spike_times,lims=None,width=1e-3,bins=None)

    ARGS
    spike_times (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.  
    
    lims        (2,) array-like. Full time range of analysis ([start,end] in s).
                Must input a value (unless explicitly setting custom <bins>)

    width       Float. Width of bin used to discretize spike times (s). 
                Usually 1 ms [default]

    *** Alternatively, time bins may be explicitly input using <bins> arg ***
    
    bins        (n_bins,2) array-like. [start,end] of each custom time bin (in s).
                Bins can have any arbitrary width and spacing, but really you would
                always want width = spacing, so each bin starts at end of last bin
                Default: bins with given <width,step>, ranging from lims[0] to lims[1]

    RETURNS
    spike_bool  (...,n_bins) ndarray of bools. Binary spike trains, where
                 1/True indicates >= 1 spike in time bin, 0/False indicates no spikes.
                    
    timepts     (n_bins,) ndarray. Time sampling vector (in s). Center of each time bin
                used to compute binary spike data.
    """
    # If bins is not given explicitly, set it based on width,lims
    if bins is None:
        assert lims is not None, \
            ValueError("Must input <lims> = full time range of analysis (or set custom <bins>)")

        # If width = 1 ms, extend lims by 0.5 ms, so bins end up centered
        # on whole ms values, as we typically want for binary spike trains
        if isclose(width,1e-3): lims = [lims[0] - 0.5e-3, lims[1] + 0.5e-3]
        bins = setup_sliding_windows(width,lims=lims,step=width)

    timepts = bins.mean(axis=1)
    
    # For each spike train in <spike_times> compute count w/in each hist bin
    # Note: Setting dtype=bool implies any spike counts > 0 will be True
    spike_bool,bins = bin_rate(spike_times,bins=bins)
    spike_bool = spike_bool.astype(bool)

    return spike_bool, timepts


def cut_trials(data, trial_lims, **kwargs):
    """
    Cuts spiking data into trials, for either spike timestamps or binary spike trains
    
    Wrapper around cut_trials_spike_times/cut_trials_spike_bool.  See those for 
    details on data-specific arguments.
    
    cut_data = cut_trials(data, trial_lims, **kwargs)
    
    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays (arbitrary shape) | 
                (...,n_timepts,...) ndarray.
                Spiking data, given either as spike timestamps or binary (1/0) spike trains.
                Can be given for either a single spike train, or for multiple spike trains 
                (eg different trials, units, etc.) within an object array.
                
                For binary spike data, additional <smp_rate> and <axis> keyword arguments must be 
                input to indicate the sampling rate (in Hz) and the array time axis.
                
    trial_lims  (n_trials,2) array-like. List of [start,end] of each trial 
                (in same timebase as data) to use to cut data.
                
    **kwargs    Any additional keyword args passed directly to cut_trials_spike_times/bool()
                
    RETURNS
    cut_data    (...,n_trials) object array of (n_trial_spikes) arrays | 
                (...,n_trial_timepts,...,n_trials) array.
                Data segmented into trials.
                Trial axis is appended to end of all axes in input data.  
    """
    data_type = _spike_data_type(data)
    
    if data_type == 'timestamp':    cut_func = cut_trials_spike_times
    elif data_type == 'bool':       cut_func = cut_trials_spike_bool
    else:
        raise ValueError("Unsupported data format input. Must be spike timestamps or binary (0/1) spike train")
    
    return cut_func(data, trial_lims, **kwargs)
        
        
def cut_trials_spike_times(data, trial_lims, trial_refs=None):
    """
    Cuts spike timestamp data into trials
    
    cut_data = cut_trials_spike_times(data, trial_lims, trial_refs=None)
    
    ARGS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays (arbitrary shape).
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.
                
    trial_lims  (n_trials,2) array-like. List of [start,end] of each trial 
                (in same timebase as data) to use to cut data.
                
    trial_refs  (n_trials,) array-like. List of event time in each trial to re-reference 
                trial's spike timestamps to (ie this sets t=0 for each trial), if desired.
                Default: None (just leave timestamps in original timebase)              
                
    RETURNS
    cut_data    (...,n_trials) object array of (n_trial_spikes) arrays.
                Spike timestamp data segmented into trials.
                Trial axis is appended to end of all axes in input data.          
    """
    trial_lims = np.asarray(trial_lims)    
    assert (trial_lims.ndim == 2) and (trial_lims.shape[1] == 2), \
        "trial_lims argument should be a (n_trials,2) array of trial [start,end] times"
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
    
    
def cut_trials_spike_bool(data, trial_lims, smp_rate=None, axis=0):
    """
    Cuts binary spike train data into trials
    
    cut_data = cut_trials_spike_bool(data, trial_lims, smp_rate, axis=0)
    
    ARGS
    data        (...,n_timepts,...) ndarray. Continuous spike train data unsegmented into trials.
                Arbitrary dimensionality, could include multiple channels, etc.
                
    trial_lims  (n_trials,2) array-like. List of [start,end] of each trial (in s) 
                to use to cut data.
                
    smp_rate    Scalar. Sampling rate of data (Hz). Must input a value for this.
    
    axis        Int. Axis of data array corresponding to time samples. Default: 0                
                
    RETURNS
    cut_data    (...,n_trial_timepts,...,n_trials) array.
                Data segmented into trials.
                Trial axis is appended to end of all axes in input data.          
    """
    assert smp_rate is not None, "For binary spike train data, you must smp_rate = sampling rate in Hz"
        
    trial_lims = np.asarray(trial_lims)    
    assert (trial_lims.ndim == 2) and (trial_lims.shape[1] == 2), \
        "trial_lims argument should be a (n_trials,2) array of trial [start,end] times"
    n_trials = trial_lims.shape[0]
        
    # Convert trial_lims in s -> indices into continuous data samples
    trial_idxs = np.round(smp_rate*trial_lims).astype(int)
    assert trial_idxs.min() >= 0, \
        ValueError("trial_lims are attempting to index (%d) before start of data (0)" % trial_idxs.min())
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
    
    
def realign_data(data, align_times, **kwargs):
    """
    Realigns trial-cut spiking data on new within-trial event times, 
    for either spike timestamps or binary spike trains
    
    Wrapper around realign_spike_times/realign_spike_bool.  See those for details
    on data-specific arguments.
    
    realigned = realign_data(data, align_times, **kwargs)
    
    ARGS
    data        (...,n_trials,...) object ndarray of (n_spikes[trial],) ndarrays | 
                (...,n_timepts,...) ndarray.
                Spiking data, given either as spike timestamps or binary (1/0) spike trains.
                Arbitrary shape, but must give <trial_axis> (and <time_axis> for binary data).

    align_times (n_trials,) array-like. New set of times (in old
                reference frame) to realign spiking data to                
                
    **kwargs    Any additional keyword args passed directly to realign_spike_times/bool()     
                
    RETURNS
    realigned   Data realigned to given within-trial times.
                For timestamp data, this has same shape as input data.
                For binary data, time axis is reduced to length implied by time_range, but 
                otherwise array has same shape as input data.   
    """
    data_type = _spike_data_type(data)
    
    if data_type == 'timestamp':    realign_func = realign_spike_times
    elif data_type == 'bool':       realign_func = realign_spike_bool
    else:
        raise ValueError("Unsupported data format input. Must be spike timestamps or binary (0/1) spike train")
    
    return realign_func(data, align_times, **kwargs)
          

def realign_data_on_event(data, event_data, event, **kwargs):
    """     
    Convenience wrapper around realign_data() for relaligning to a given
    named event within a per-trial dataframe or dict variable.

    realigned = realign_spike_times_on_event(spike_times, event_data, event, **kwargs)

    ARGS
    data        (...,n_trials,...) object ndarray of (n_spikes[trial],) ndarrays | 
                (...,n_timepts,...) ndarray.
                Spiking data, given either as spike timestamps or binary (1/0) spike trains.
                Arbitrary shape, but must give <trial_axis> (and <time_axis> for binary data).

    event_data  {string:(n_trials,) array} dict | (n_trials,n_events) DataFrame.
                Per-trial event timing data to use to realign spike timestamps.

    event       String. Dict key or DataFrame column name whose associated values
                are to be used to realign spike timestamps

    RETURNS
    realigned   Same data struture, but realigned to given event
    """
    # Extract vector of times to realign on
    align_times = event_data[event]
    # Compute the realignment and return
    return realign_data(data, align_times, **kwargs)

                    
def realign_spike_times(spike_times, align_times, trial_axis=0):
    """
    Realigns trial-cut spike timestamps to new set of within-trial times 
    (eg new trial event) so that t=0 on each trial at given event. 
    For example, timestamps aligned to a start-of-trial event might
    need to be relaligned to the behavioral response.

    spike_times = realign_spike_times(spike_times,align_times,trial_axis=0)

    ARGS
    spike_times (...,n_trials,...) object ndarray of (n_spikes[trial],) ndarrays |
                ndarray of bool.
                Spike timestamps, usually in seconds referenced to some
                within-trial event. Can be any arbitrary shape (including having
                multiple units), as long as trial axis is given in <trial_axis>.

    align_times (n_trials,) array-like. New set of times (in old
                reference frame) to realign spike timestamps to
                
    trial_axis  Int. Axis of spike_times corresponding to trials.
                Default: 0 (1st axis of array)       

    RETURNS
    realigned   Same data struture, but with each timestamp realigned to times
    """    
    # Move trial axis to first axis of array
    if trial_axis != 0: spike_times = np.moveaxis(spike_times, trial_axis, 0)
    
    # Subtract new reference time from all spike times for each trial
    for trial,align_time in enumerate(align_times):
        spike_times[trial,...] = spike_times[trial,...] - align_time
        
    # Move trial axis to original location
    if trial_axis != 0: spike_times = np.moveaxis(spike_times, 0, trial_axis)

    return spike_times


def realign_spike_bool(data, align_times, time_range=None, timepts=None, time_axis=0, trial_axis=-1):
    """
    Realigns trial-cut binary (1/0) spiking data to new set of within-trial times 
    (eg new trial event) so that t=0 on each trial at given event. 
    For example, data aligned to a start-of-trial event might
    need to be relaligned to the behavioral response.

    realigned = realign_spike_bool(data,align_times,time_range,timepts,time_axis=0,trial_axis=-1)

    ARGS
    data        ndarray. Binary (1/0) spiking data segmented into trials.
                Arbitrary dimensionality, could include multiple units, etc.
    
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


def pool_electrode_units(data_sua, electrodes, axis=-1, elec_set=None,
                         return_idxs=False):
    """
    Pools spiking data across all units on each electrode, dispatching to
    appropriate function for spike time vs spike rate data

    data_mua = pool_electrode_units(data_sua,electrodes,axis=-1,
                                    elec_set=None,return_idxs=False)

    data_mua,elec_idxs = pool_electrode_units(data_sua,electrodes,axis=-1,
                                              elec_set=None,return_idxs=True)

    ARGS
    data_sua    (...,n_units,...) object ndarray of (n_spikes[unit],) ndarrays.
                Lists of single-unit spike timestamps for different units and 
                trials/conditions/etc. within an object array of any arbitrary shape.
                -or-
                (...,n_units,...) ndarray of bools. Any arbitary shape.
                Binary (0/1) spike trains for different units and trials/conditions/etc.
                -or-
                (...,n_units,...) ndarray. Any arbitary shape.
                Set of spike rates/counts for different units and trials/conditions/etc.                

    electrodes  (n_units,) array-like. List of electrode numbers of each unit
                in <data_sua>

    axis        Int. Axis of <data_sua> corresponding to different units. 
                Default: -1 (last axis)

    elec_set    (n_elecs,) array-like. Set of unique electrodes in <electrodes>.
                Default: unsorted np.unique(electrodes)

    return_idxs Bool. If True, additionally returns list of indexes corresponding
                to 1st occurrence of each electrode in <electrodes>. Default: False

    RETURNS
    data_mua    (...,n_elecs,...) object ndarray of (n_spikes[elecs],) ndarrays.
                Lists of multi-unit spike timestamps pooled across all units 
                on each electrode. Same shape as input, with unit axis reduced
                to len(elec_set).
                -or-
                (...,n_elecs,...) ndarray of bools.
                Binary spike trains pooled across all units on each electrode. 
                Same shape as input, with unit axis reduced to len(elec_set).                
                -or-
                (...,n_elecs,...) ndarray. Spike rates/counts pooled (summed) across
                all units on each electrode. Same shape as input, with unit axis 
                reduced to len(elec_set).                

    elec_idxs   (n_elecs,) ndarray of ints. Indexes of 1st occurrence of each
                electrode in <elec_set> within <electrodes>. Can be used to
                transform any corresponding metadata appropriately.
    """
    data_type = _spike_data_type(data_sua)
    
    if data_type == 'timestamp':    pooler_func = pool_electrode_units_spike_times
    elif data_type == 'bool':       pooler_func = pool_electrode_units_spike_bool
    else:                           pooler_func = pool_electrode_units_spike_rate
    
    return pooler_func(data_sua,electrodes,axis=axis,
                       elec_set=elec_set,return_idxs=return_idxs)


def pool_electrode_units_spike_times(data_sua, electrodes, axis=-1, elec_set=None,
                                     return_idxs=False, sort=True):
    """
    Concatenates spike timestamps across all single units on each electrode, 
    pooling into single "threshold-crossing multi-units" per electrode
    
    data_mua = pool_electrode_units_spike_times(data_sua, electrodes, axis=-1, elec_set=None,
                                                return_idxs=False, sort=True)
                                     
    ARGS
    data_sua    (...,n_units,...) object ndarray of (n_spikes[unit],) ndarrays.
                Lists of single-unit spike timestamps for different units and 
                trials/conditions/etc. within an object array of any arbitrary shape.

    electrodes  (n_units,) array-like. List of electrode numbers of each unit in <data_sua>.

    axis        Int. Axis of <data_sua> corresponding to different units. Default: -1 (last axis)
    
    elec_set    (n_elecs,) array-like. Set of unique electrodes in <electrodes>.
                Default: unsorted np.unique(electrodes) (in original order in <electrodes>)

    return_idxs Bool. If True, additionally returns list of indexes corresponding
                to 1st occurrence of each electrode in <electrodes>. Default: False

    RETURNS
    data_mua    (...,n_elecs,...) object ndarray of (n_spikes[elecs],) ndarrays.
                Lists of multi-unit spike timestamps pooled across all units 
                on each electrode. Same shape as input, with unit axis reduced
                to len(elec_set).

    elec_idxs   (n_elecs,) ndarray of ints. Indexes of 1st occurrence of each
                electrode in <elec_set> within <electrodes>. Can be used to
                transform any corresponding metadata appropriately.
    """
    # Find set of electrodes in data, if not explicitly input
    if elec_set is None: elec_set = unsorted_unique(electrodes)
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
                np.concatenate([ts.reshape((-1,))
                                for ts in data_sua[i_series,elec_idxs]])
            # Sort timestamps so they remain in sequential order after concatenation
            if sort: data_mua[i_series,i_elec].sort()

    # Reshape output data array to original shape (now with len(data[axis] = n_elecs)
    data_mua = undo_standardize_array(data_mua, data_shape, axis=axis, target_axis=-1)
    
    # Generate list of indexes of 1st occurrence of each electrode, if requested
    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return data_mua, elec_idxs
    else:
        return data_mua


def pool_electrode_units_spike_bool(data_sua, electrodes, axis=-1, elec_set=None,
                                    return_idxs=False):
    """
    Pools boolean spike train data across all units on each electrode into
    a single "threshold-crossing multi-units" per electrode.
    A multi-unit spike is registered if there is a spike in any of its 
    constituent single units (logical OR) at each timepoint.
    
    pool_electrode_units_spike_bool(data_sua, electrodes, axis=-1, elec_set=None,
                                    return_idxs=False)
                                    
    ARGS
    data_sua    (...,n_units,...) ndarray of bools. Any arbitary shape.
                Binary (0/1) spike trains for different units and trials/conditions/etc.

    electrodes  (n_units,) array-like. List of electrode numbers of each unit in <data_sua>.

    axis        Int. Axis of <data_sua> corresponding to different units. Default: -1 (last axis)
    
    elec_set    (n_elecs,) array-like. Set of unique electrodes in <electrodes>.
                Default: unsorted np.unique(electrodes) (in original order in <electrodes>)

    return_idxs Bool. If True, additionally returns list of indexes corresponding
                to 1st occurrence of each electrode in <electrodes>. Default: False

    RETURNS
    data_mua    (...,n_elecs,...) ndarray of bools.
                Binary spike trains pooled across all units on each electrode. 
                Same shape as input, with unit axis reduced to len(elec_set).

    elec_idxs   (n_elecs,) ndarray of ints. Indexes of 1st occurrence of each
                electrode in <elec_set> within <electrodes>. Can be used to
                transform any corresponding metadata appropriately.                                      
    """
    if elec_set is None: elec_set = unsorted_unique(electrodes)
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
        # DEL This didn't work for reasons I don't understand
        # data_mua[tuple(slicer_mua)] = data_sua[tuple(slicer_sua)].any(axis=axis,keepdims=True)
    
    # Generate list of indexes of 1st occurrence of each electrode, if requested
    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return data_mua, elec_idxs
    else:
        return data_mua    


def pool_electrode_units_spike_rate(data_sua, electrodes, axis=-1, elec_set=None,
                                    return_idxs=False):
    """
    Pools (sums) spike rate/count data across all units on each electrode into
    a single "threshold-crossing multi-units" per electrode.
    
    pool_electrode_units_spike_rate(data_sua, electrodes, axis=-1, elec_set=None,
                                    return_idxs=False)
                                    
    ARGS
    data_sua    (...,n_units,...) ndarray. Any arbitary shape.
                Set of spike rates/counts for different units and trials/conditions/etc.

    electrodes  (n_units,) array-like. List of electrode numbers of each unit in <data_sua>.

    axis        Int. Axis of <data_sua> corresponding to different units. Default: -1 (last axis)
    
    elec_set    (n_elecs,) array-like. Set of unique electrodes in <electrodes>.
                Default: unsorted np.unique(electrodes) (in original order in <electrodes>)

    return_idxs Bool. If True, additionally returns list of indexes corresponding
                to 1st occurrence of each electrode in <electrodes>. Default: False

    RETURNS
    data_mua    (...,n_elecs,...) ndarray. Spike rates/counts pooled (summed) across
                all units on each electrode. Same shape as input, with unit axis 
                reduced to len(elec_set).

    elec_idxs   (n_elecs,) ndarray of ints. Indexes of 1st occurrence of each
                electrode in <elec_set> within <electrodes>. Can be used to
                transform any corresponding metadata appropriately.                                    
    """
    if elec_set is None: elec_set = unsorted_unique(electrodes)
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
    
    # Generate list of indexes of 1st occurrence of each electrode, if requested
    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return data_mua, elec_idxs
    else:
        return data_mua    


# =============================================================================
# Synthetic data generation and testing functions
# =============================================================================
def simulate_spike_rates(gain=5.0, offset=5.0, n_conds=2, n_trials=1000,
                         window=1.0, count=False, seed=None):
    """
    Simulates Poisson spike rates across multiple conditions/groups
    with given condition effect size

    rates,labels = simulate_spike_rates(gain=5.0,offset=5.0,n_conds=2,
                                        n_trials=1000,window=1.0,seed=None)

    ARGS
    gain    Scalar | (n_conds,) array-like. Spike rate gain (in spk/s) for
            each condition, which sets effect size.
            Scalar : spike rate difference between each successive condition
            (n_conds,) vector : specific spike rate gain for each condition
            Set = 0 to simulate no expected difference between conditions.
            Default: 5.0 (5 spk/s difference btwn each condition)

    offset  Scalar. Baseline rate added to condition effects. Default: 5.0 spk/s

    n_conds Int. Number of distinct conditions/groups to simulate. Default: 2

    n_trials Int. Number of trials/observations to simulate. Default: 1000

    window  Scalar. Time window to count simulated spikes over. Can set = 1
            if you want spike *counts*, rather than rates. Default: 1.0 s

    count   Bool. If True, returns integer-valued spike counts, rather than rates.
            Default: False (computes rates)      
                
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    RETURNS
    rates   (n_trials,). Simulated Poisson spike rates

    labels  (n_trials,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
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
    Simulates Poisson spike trains across multiple conditions/groups
    with given condition effect size

    trains,labels = simulate_spike_trains(gain=5.0,offset=5.0,n_conds=2,
                                          n_trials=1000,time_range=1.0,seed=None,
                                          data_type='timestamp')

    ARGS
    gain    Scalar | (n_conds,) array-like. Spike rate gain (in spk/s) for
            each condition, which sets effect size.
            Scalar : spike rate difference between each successive condition
            (n_conds,) vector : specific spike rate gain for each condition
            Set = 0 to simulate no expected difference between conditions.
            Default: 5.0 (5 spk/s difference btwn each condition)

    offset  Scalar. Baseline rate added to condition effects. Default: 5.0 spk/s

    n_conds Int. Number of distinct conditions/groups to simulate. Default: 2

    n_trials Int. Number of trials/observations to simulate. Default: 1000

    time_range  Scalar. Full time range to simulate spike train over. Default: 1 s

    refractory  Scalar. Absolute refractory period in which a second spike cannot
            occur after another. Set=0 for Poisson with no refractory [default].
            
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    data_type String. Format of output spike trains:
            'timestamp' : Spike timestamps in s relative to trial starts [default]
            'bool'   : Binary (0/1) vectors flagging spike times

    RETURNS
    trains  (n_trials,) of object | (n_trials,n_timepts) of bool.
            Simulated Poisson spike trains, either as list of timestamps relative
            to trial start or as binary vector for each trial (depending on data_type).

    labels  (n_trials,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
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
        # ALT ISIs = np.random.exponential(1/lam, (int(round(2*n_spikes_exp)),))
        
        # HACK Implement absolute refractory period by deleting ISIs < refractory
        # TODO More principled way of doing this that doesn't affect rates
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
            

#==============================================================================
# Other helper functions
#==============================================================================
def _spike_data_type(data):
    """
    Determines what type of spiking data we have:
    'timestamp' :   spike timestamps (in a Numpy object array or list)
    'bool' :        binary spike train (1=spike, 0=no spike at each timepoint)
    'rate' :        array of spike rate/counts
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


def _isbinary(data):
    """
    Tests whether variable contains only binary values (True,False,0,1)
    """
    data = np.asarray(data)
    return (data.dtype == bool) or \
           (np.issubdtype(data.dtype,np.number) and \
            np.all(np.in1d(data,[0,1,0.0,1.0,True,False])))


def _remove_buffer(data, buffer, axis=-1):
    """
    Removes a temporal buffer (eg zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    data = _remove_buffer(data,buffer,axis=-1)

    ARGS
    data    Data array where a buffer has been appended on both ends of time
            dimension. Can be any arbitrary size, typically
            (n_trials,n_units,n_timepts+2*buffer).
    buffer  Scalar. Length (number of samples) of buffer appended to each end.
    axis    Int. Array axis to remove buffer from (ie time dim). Default: -1

    RETURNS
    data    Data array with buffer removed, reducing time axis to n_timepts
            (typically shape (n_trials,n_units,n_timepts))
    """
    if axis < 0: axis = data.ndim + axis
    
    if axis == data.ndim-1:
        return data[...,buffer:-buffer]
    else:
        return (data.swapaxes(-1,axis)[...,buffer:-buffer]
                    .swapaxes(axis,-1))


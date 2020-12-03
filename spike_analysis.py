# -*- coding: utf-8 -*-
"""
A module for basic analyses of neural spiking activity

FUNCTIONS
## Rate analysis ##
bin_rate            Computes spike counts/rates in series of time bins (regular or not)
density             Computes spike density (smoothed rate) with given kernel

## Preprocessing ##
realign_spike_times Realigns spike timestamps to new t=0
times_to_bool       Converts spike timestamps to binary spike trains
bool_to_times       Converts binary spike train to timestamps
pool_electrode_units Pools all units on each electrode into a multi-unit

## Plotting ##
plot_raster         Generates a raster plot
plot_mean_waveforms Plots mean spike waveforms from one/more units
plot_waveform_heatmap Plots heatmap (2d histogram) of spike waveforms

## Synthetic data generation ##
simulate_spike_rates    Generates sythetic Poisson rates
simulate_spike_trains   Generates sythetic Poisson process spike trains


Created on Mon Aug 13 14:38:34 2018

@author: sbrincat
"""
# TODO  Generalize basic functions (2bool,etc.) to arbitrary shape of spike_times array?
# TODO  Add ISI/autocorrelation stats?

from math import isclose, ceil, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import convolve
from scipy.signal.windows import hann, gaussian
from scipy.stats import poisson, bernoulli, norm


# =============================================================================
# Spike count/rate computation functions
# =============================================================================
def bin_rate(data, lims=None, width=50e-3, step=None, bins=None, count=False):
    """
    Computes spike rate/count within given sequence of time bins
    
    Use <lims,width,step> to set standard-width sliding window bins or
    use <bins> to set any arbitrary custom time bins

    rates,bins = bin_rate(data,lims=None,width=50e-3,bins=None,count=False)

    INPUTS
    data        (n_spikes,) array-like | object array of (n_spikes,) arrays.
                List of spike timestamps (in s).  Can be given for either a single
                spike train, or for multiple spike trains (eg different trials,
                units, etc.) within an object array of any arbitrary shape.

    lims        (2,) array-like. Full time range of analysis ([start,end] in s).
                Must input a value (unless explicitly setting custom <bins>)
                
    width       Scalar. Full width of each time bin (s). Default: 50 ms
    
    step        Scalar. Spacing between successive time bins (s). 
                Default: <width> (each bin starts at end of previous bin)
                
    count       Bool. If True, returns integer-valued spike counts, rather than rates.
                Default: False (computes rates)               
                
    *** Alternatively, any custom time bins may be explicitly input using <bins> arg ***
    
    bins        (n_bins,2) array-like. [start,end] of each custom time bin (in s).
                Bins can have any arbitrary width and spacing.
                Default: bins with given <width,step>, ranging from lims[0] to lims[1]

    RETURNS
    rates       (...,n_bins) ndarray. Spike rates (in spk/s) or spike counts 
                in each time bin (and for each trial/unit/etc. in <data>). 
                Same shape as <data>, with time-bin axis appended to end.
                If only a single spike train is input, output is (n_bins,) vector.
                dtype is uint16 if <count> is True, float otherwise.

    bins        (n_bins,2) ndarray. [start,end] of each time bin (in s).
    
    NOTES
    Spikes are counted within each bin including the start, but *excluding* the end
    of the bin. That is each bin is defined as [start,end).
    """    
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
    
    # For standard bins, can use histogram algorithm
    if std_bins:
        # Convert bins to format expected by np.histogram = edges of all bins in 1 series
        bins = np.hstack((bins[:,0],bins[-1,-1]))
        count_spikes = _histogram_count
    # Otherwise, need algorithm to enumerate and count spikes in each bin
    else:        
        count_spikes = _custom_bin_count
        
    # Create 1D flat iterator to iterate over arbitrary-shape data array
    data_flat = data.flat
    
    # Create array to hold counts/rates. Same shape as data, with bin axis appended.
    # dtype is int if computing counts, float if computing rates
    dtype = 'uint16' if count else float
    rates = np.empty((*data.shape,n_bins),dtype=dtype)
    
    for _ in range(data.size):
        # Multidim coordinates into data array
        coords = data_flat.coords
        
        # Count spikes within each bin
        rates[(*coords,slice(None))] = count_spikes(data[coords],bins)

        # Iterate to next element (list of spike times for trial/unit/etc.) in data 
        next(data_flat)        
                    
    # Normalize all spike counts by bin widths to get spike rates
    if not count: rates = rates / widths         
        
    # If only a single spike train was input, squeeze out singleton axis 0
    if single_train: rates = rates.squeeze(axis=0)
    
    return rates, bins
    
    
def _histogram_count(data, bins):
    """ Count spikes in equal-width, disjoint bins """
    return np.histogram(data,bins)[0]


def _custom_bin_count(data, bins):
    """ Count spikes in any arbitrary custom bins """
    return np.asarray([((start <= data) & (data < end)).sum() 
                       for (start,end) in bins], dtype='uint16')
                    
    
psth = bin_rate
""" Aliases function bin_rate as psth """


def density(spike_times, t=None, kernel='gaussian', width=50e-3, smp_rate=1000,
            lims=None, buffer=None, downsmp=1, **kwargs):
    """
    Computes spike density function via convolution with given kernel/width

    spike_density,t = density(spike_times,t=None,kernel='gaussian',width=50e-3,
                              smp_rate=1000,lims=None,buffer=None,downsmp=1,
                              **kwargs)

    INPUTS
    spike_times Arbitrary-shape (...) object ndarray of (n_spikes[row,col],) ndarrays.
                Usually n_rows=n_trials,n_cols=n_units.
                Spike timestamps, usually in seconds referenced to some
                within-trial event. Must be in same time units as epochs.
                Can be given for multiple neurons,trials; output will be same
                size with extra time axis appended.

    kernel      String. Name of convolution kernel to use:
                'gaussian' [default] | 'hanning'
                Also includes (not well tested) functionality to input the
                kernel itself as an arrray, or a custom function that takes
                a "width" argument (+ any extra kwargs).

    width       Scalar. Width parameter for given kernel. Default: 50 ms
                Interpretation is kernel-specific.
                'gaussian' : <width> = 1 Gaussian standard deviation
                'hanning'  : <width> = kernel half-width (~ 2.5x Gaussian SD)

    t           (n_timepts,) array-like. Desired time sampling vector for
                spike density, before any downsampling. Input either <t> OR
                <lims> and <smp_rate> to set time sampling.

    smp_rate    Scalar. Sampling rate (Hz; 1/sample period) for spike density
                Only used if <t> not input. Default: 1000 Hz

    lims        (2,) array-like. Full time range of analysis.
                Only used if <t> not input.

    buffer      Float. Length (in s) of symmetric buffer to add to each end
                of time dimension (and trim off before returning) to avoid edge
                effects. Default: (kernel-dependent, approximates length of
                edge effects induced by kernel)

    **kwargs    All other kwargs passed directly to kernel function

    (any additional kwargs passed directly to kernel function)

    RETURNS
    spike_density (...,n_timepts) ndarray. Same shape as spike_times, with
                time axis (len=n_timepts) appended to end. Spike density function.

    t           (n_timepts,) ndarray. Time sampling vector for spike_density
    """
    # Set default buffer based on overlap of kernel used
    if buffer is None:
        if kernel in ['hann','hanning']:        buffer = width
        elif kernel in ['gaussian','normal']:   buffer = 3*width
        else:                                   buffer = 0

    # Convert string specifier to kernel (window) function (convert width to samples)
    kernel = _str_to_kernel(kernel,width*smp_rate,**kwargs)
    # Normalize kernel to integrate to 1
    kernel = kernel / (kernel.sum()/smp_rate)

    # Set time sampling, either directly from input t, or indirectly from smp_rate,lims
    if t is None:
        assert lims is not None, \
            ValueError("Must input <lims> = full time range of analysis (or set custom time sampling <t>)")
        
        if buffer != 0: lims = [lims[0]-buffer,lims[1]+buffer]
        args = {'lims':lims, 'width':1.0/smp_rate}
    else:
        dt   = np.mean(np.diff(t))
        if buffer != 0:
            t = np.concatenate((np.flip(np.arange(t[0]-dt,t[0]-buffer-1e-12,-dt)),
                                t,
                                np.arange(t[-1]+dt,t[-1]+buffer+1e-12,dt)))
        bins = np.arange(t[0] - dt/2, t[-1] + dt/2 + 1e-12, dt)
        args = {'bins':bins}

    # Convert spike times to binary spike trains -> (n_rows,n_cols,n_timepts)
    spike_bool,t = times_to_bool(spike_times,**args)

    # Compute density as convolution of spike trains with kernel
    spike_density = convolve(spike_bool,kernel[np.newaxis,np.newaxis,:],mode='same')

    # Remove any time buffer from spike density and time sampling veotor
    if buffer != 0:
        dt   = np.mean(np.diff(t))
        buffer = int(round(buffer/dt))  # Convert buffer from time units -> samples
        spike_density = _remove_buffer(spike_density,buffer,axis=-1)
        t = _remove_buffer(t,buffer,axis=-1)
    if downsmp != 1:
        spike_density = spike_density[...,0:-1:downsmp]
        t = t[...,0:-1:downsmp]

    # KLUDGE Sometime trials/neurons w/ 0 spikes end up with tiny non-0 values
    # due to floating point error in fft routines. Fix by setting = 0.
    no_spike_idxs = ~spike_bool.any(axis=2,keepdims=True)
    spike_density[np.tile(no_spike_idxs,(1,1,spike_density.shape[-1]))] = 0

    return spike_density, t


#==============================================================================
# PLOTTING FUNCTIONS
#==============================================================================
def plot_raster(spike_times, ax=None, xlim=None, color='0.25', height=1.0,
                xlabel=None, ylabel=None):
    """
    Plots rasters (TODO more documentation)
    """
    if ax is None: ax = plt.gca()

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


def _plot_raster_line(spike_times, y=0, xlim=None, color='0.25', height=1.0):
    """ PLots single line of raster plot """
    # Extract only spike w/in plotting time window
    if xlim is not None:
        spike_times = spike_times[(spike_times >= xlim[0]) &
                                  (spike_times <= xlim[1])]

    y = np.asarray([[y+height/2.0], [y-height/2.0]])

    plt.plot(spike_times[np.newaxis,:]*np.ones((2,1)),
             y*np.ones((1,len(spike_times))), '-', color=color,linewidth=1)


def plot_mean_waveforms(spike_waves, t=None, sd=True,
                        ax=None, plot_colors=None):
    """
    Plots mean spike waveform for each of one or more units

    ax = plot_mean_waveforms(spike_waves,t=None,sd=True,
                             ax=None,plot_colors=None)

    INPUTS
    spike_waves (n_units,) object array of (n_timepts,n_spikes) arrays.
                Spike waveforms for one or more units

    t           (n_timepts,) array-like. Common time sampling vector for each
                spike waveform. Default: 0:n_timepts

    sd          Bool. If set, also plots standard deviation of waves as fill.
                Default: True

    plot_colors (n_units_max,3) array. Color to plot each unit in.
                Default: Colors from Blackrock Central Spike grid plots

    ax          Pyplot Axis instance. Axis to plot into. Default: plt.gca()
    """
    # TODO  Setup to handle single array of 1 unit's wfs instead of object array of units
    n_units      = len(spike_waves)
    n_timepts    = spike_waves[0].shape[0]

    # If no time sampling vector given, default to 0:n_timepts
    if t is None: t = np.arange(n_timepts)
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
            ax.fill(np.hstack((t,np.flip(t))),
                    np.hstack((mean+sd,np.flip(mean-sd))),
                    facecolor=plot_colors[unit,:], edgecolor=None, alpha=0.25)

        ax.plot(t, mean, '-', color=plot_colors[unit,:], linewidth=1)

    ax.set_xlim(t[0],t[-1])
    ax.set_facecolor('k')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return ax


def plot_waveform_heatmap(spike_waves, t=None, wf_range=None,
                          ax=None, cmap=None):
    """
    Plots heatmap (2D hist) of all spike waveforms across one or more units

    ax = plot_waveform_heatmap(spike_waves,t=None,wf_range=None,
                               ax=None,cmap=None)

    INPUTS
    spike_waves (n_units,) object array of (n_timepts,n_spikes) arrays.
                Spike waveforms for one or more units

    t           (n_timepts,) array-like. Common time sampling vector for each spike
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
    if t is None: t = np.arange(n_timepts)
    # If no waveform amplitude range given, default to [min,max] of set of waveforms
    if wf_range is None: wf_range = [np.min(spike_waves),np.max(spike_waves)]


    # Set histogram bins to sample full range of times,
    xedges = np.linspace(t[0],t[-1],n_timepts)
    yedges = np.linspace(wf_range[0],wf_range[1],20)

    # Compute 2D histogram of all waveforms
    wf_hist = np.histogram2d(np.tile(t,(n_spikes,)),
                             spike_waves.T.reshape(-1),
                             bins=(xedges,yedges))[0]
    # Plot heat map image
    plt.imshow(wf_hist.T, cmap=cmap, origin='lower',
               extent=[xedges[0],xedges[-1],yedges[0],yedges[-1]])

    ax.set_xlim(t[0],t[-1])
    ax.set_ylim(wf_range[0],wf_range[-1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('auto')

    return ax


# =============================================================================
# Preprocessing/Utility functions
# =============================================================================
def realign_spike_times(spike_times, align_times):
    """
    Realigns spike timestamps to new set of within-trial times (eg new trial
    event). For example, timestamps aligned to a start-of-trial event might
    need to be relaligned to the behavioral response.

    spike_times = realign_spike_times(spike_times,align_times)

    INPUTS
    spike_times (n_trials,n_units) object ndarray of (n_spikes[trial,unit],) ndarrays.
                Spike timestamps, usually in seconds referenced to some
                within-trial event.

    align_times (n_trials,) array-like. New set of times (in old
                reference frame) to realign spike timestamps to

    RETURNS
    realigned   Same data struture, but with each timestamp realigned to times

    """
    n_rows,n_cols = spike_times.shape

    realigned = np.empty(spike_times.shape,dtype=object)

    # For each spike train in <spike_times> compute count w/in each hist bin
    for row in range(n_rows):
        for col in range(n_cols):
            realigned[row,col] = spike_times[row,col] - align_times[row]

    return realigned


def realign_spike_times_on_event(spike_times, event_data, event):
    """
    Realigns spike timestamps to new trial event, s.t. t=0 on each trial at
    given event. For example, timestamps aligned to a start-of-trial event
    might need to be relaligned to the behavioral response.

    spike_times = realign_spike_times_on_event(spike_times,event_data,event)

    INPUTS
    spike_times (n_trials,n_units) object ndarray of (n_spikes[trial,unit],) ndarrays.
                Spike timestamps, usually in seconds referenced to some
                within-trial event.

    event_data  {string:(n_trials,) array} dict | (n_trials,n_events) DataFrame.
                Per-trial event timing data to use to realign spike timestamps.

    event       String. Dict key or DataFrame column name whose associated values
                are to be used to realign spike timestamps

    RETURNS
    spike_times Same data struture, but with each timestamp realigned to event

    """
    # Extract vector of times to realign on
    align_times = event_data[event]
    # Compute the realignment and return
    return realign_spike_times(spike_times,align_times)


def bool_to_times(spike_bool, bins):
    """
    Converts boolean (binary) spike train representaton to spike timestamps
    Inverse function of times_to_bool()

    spike_times = bool_to_times(spike_bool,bins)

    INPUTS
    spike_bool  (n_rows,n_cols,n_bins) ndarray of bools. Binary spike trains,
                where 1 indicates >= 1 spike in time bin, 0 indicates no spikes.

    bins        (n_bins,) ndarray. Center of each time bin.

    RETURNS
    spike_times (n_rows,n_cols) object ndarray of (n_spikes[row,col],) ndarrays.
                Usually n_rows=n_trials,n_cols=n_units.
                Spike timestamps, usually in seconds referenced to some
                within-trial event. Must be in same units as bins/width/lims.
    """
    bins = np.asarray(bins)

    n_rows,n_cols,_ = spike_bool.shape

    spike_times = np.empty((n_rows,n_cols), dtype=object)

    # For each spike train, find spikes and convert to timestamps
    for row in range(n_rows):
        for col in range(n_cols):
            spike_times[row,col] = bins[spike_bool[row,col,:]]

    return spike_times


def times_to_bool(spike_times, lims=None, width=1e-3, bins=None):
    """
    Converts spike timestamps to boolean (binary) spike train representaton
    Inverse function of bool_to_times()

    spike_bool,timepts = times_to_bool(spike_times,lims=None,width=1e-3,bins=None)

    INPUTS
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


def pool_electrode_units(spike_data, electrodes, elec_set=None,
                         return_idxs=False):
    """
    Pools spiking data across all units on each electrode, dispatching to
    appropriate function for spike time vs spike rate data

    spike_data = pool_electrode_units(spike_data,electrodes,
                                      elec_set=None,return_idxs=False)

    spike_data,elec_idxs = pool_electrode_units(spike_data,electrodes,
                                                elec_set=None,return_idxs=False)

    INPUTS
    spike_data  (n_trials,n_units) object ndarray of (n_spikes[row,col],) ndarrays
                of spike timestamp data or
                (n_trials,n_units,n_timepts) ndarray of floats/ints of spike
                rate data

    electrodes  (n_units,) array-like. List of electrode numbers of each unit
                in <spike_data>

    elec_set    (n_elecs,) array-like. Set of unique electrodes in <electrodes>.
                Default: unsorted np.unique(electrodes)

    return_idxs Bool. If set, additionally returns list of indexes corresponding
                to 1st occurrence of each electrode in <electrodes>. Default: False

    RETURNS
    spike_data  (n_trials,n_elecs) object ndarray of spike timestamps or
                (n_trials,n_elecs,n_timepts) ndarray of spike rates pooled
                across all units on each electrode

    elec_idxs   (n_elecs,) ndarray of ints. Indexes of 1st occurrence of each
                electrode in <elec_set> within <electrodes>. Can be used to
                filter corresponding metadata (eg "unitInfo" dataframe,
                using unitInfo.iloc[elec_idxs]).
    """
    if elec_set is None:    elec_set = _unsorted_unique(electrodes)

    if _spike_data_type(spike_data) == 'timestamp':
        return pool_electrode_units_spike_times(spike_data,electrodes,elec_set,
                                                return_idxs)
    else:
        return pool_electrode_units_spike_rates(spike_data,electrodes,elec_set,
                                                return_idxs)


def pool_electrode_units_spike_rates(spike_rates, electrodes, elec_set=None,
                                     return_idxs=False):
    """
    Pools (sums) spike rate/count data across all units on each electrode
    See pool_electrode_units() for input/output argument format
    """
    # FIXME Does this work for multi-dim spike_rates?
    if elec_set is None: elec_set = _unsorted_unique(electrodes)

    spike_rates = (pd.DataFrame(spike_rates)# Convert to Pandas DataFrame
                    .groupby(electrodes,sort=False) # Group by electrodes
                    .sum()                  # Sum rates/counts in each electrode
                    .reindex(elec_set)      # Sort to reflect order of <elec_set>
                    .values)                # Convert back to Numpy array

    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return spike_rates, elec_idxs
    else:
        return spike_rates


def pool_electrode_units_spike_times(spike_times_sua, electrodes, elec_set=None,
                                     return_idxs=False, sort=True):
    """
    Pools (concatenates spike timestamps across all units on each electrode
    See pool_electrode_units() for input/output argument format
    """
    if elec_set is None: elec_set = _unsorted_unique(electrodes)

    n_elecs         = len(elec_set)
    n_trials,_      = spike_times_sua.shape
    spike_times_mua = np.empty((n_trials,n_elecs),dtype=object)

    for i_elec,elec in enumerate(elec_set):
        # Find all units on given electrode
        elec_idxs   = electrodes == elec
        for i_trial in range(n_trials):
            # Concatenate spike_times across all units
            # -> (n_spikes_total,) ndarray
            spike_times_mua[i_trial,i_elec] = \
            np.concatenate([ts.reshape((-1,))
                            for ts in spike_times_sua[i_trial,elec_idxs]])
            # Sort timestamps so they remain in order after concatenation
            if sort: spike_times_mua[i_trial,i_elec].sort()

    if return_idxs:
        elec_idxs = [np.nonzero(electrodes == elec)[0][0] for elec in elec_set]
        elec_idxs = np.asarray(elec_idxs,dtype=int)
        return spike_times_mua, elec_idxs
    else:
        return spike_times_mua


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
                Default: None = just start at lims[0]

    force_int   Bool. If True, rounds window starts,ends to integer values.
                Default: False (don't round)

    exclude_end Bool. If True, excludes the endpoint of each (integer-valued)
                sliding win from the definition of that win, to prevent double-sampling
                (eg, the range for a 100 ms window is [1 99], not [1 100])
                Default: True if force_int==True, otherwise default=False

    OUTPUT
    windows     (n_wins,2) ndarray. Sequence of sliding window [start end]'s
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


#==============================================================================
# Synthetic data generation and testing functions
#==============================================================================
def simulate_spike_rates(gain=5.0, offset=5.0, n_conds=2, n_trials=1000,
                         window=1.0, seed=None):
    """
    Simulates Poisson spike rates across multiple conditions/groups
    with given condition effect size

    rates,labels = simulate_spike_rates(gain=5.0,offset=5.0,n_conds=2,
                                        n_trials=1000,window=1.0,seed=None)

    INPUTS
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

    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for actual random numbers.

    RETURNS
    rates   (n_trials,). Simulated Poisson spike rates

    labels  (n_trials,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
    """
    # TODO  Add support for multiple units w/ varying effect probability or size?

    if seed is not None: np.random.seed(seed)

    # Is gain scalar-valued or array-like?
    scalar_gain = not isinstance(gain, (list, tuple, np.ndarray))

    if not scalar_gain:
        gain = np.asarray(gain)
        assert len(gain) == n_conds, \
            ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                       % (len(gain), n_conds))

    # Create (nTrials,) vector of group labels (ints in set 0-n_conds-1)
    n_reps = ceil(n_trials/n_conds)
    labels = np.tile(np.arange(n_conds),(n_reps,))[0:n_trials]
    # For easy visualization, sort trials by group number
    labels.sort()

    # Per-trial Poisson rate parameters = expected number of spikes in interval
    # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
    if scalar_gain: lambdas = (offset + gain*labels)*window
    # Hand-set gain specific for each condition
    else:           lambdas = (offset + gain[labels])*window

    # Simulate Poisson spike counts, convert to rates
    rates = np.random.poisson(lambdas,(n_trials,)) / window

    return rates, labels


def simulate_spike_trains(gain=5.0, offset=5.0, n_conds=2, n_trials=1000,
                          window=1.0, seed=None, out_type='timestamp'):
    """
    Simulates Poisson spike trains across multiple conditions/groups
    with given condition effect size

    trains,labels = simulate_spike_trains(gain=5.0,offset=5.0,n_conds=2,
                                          n_trials=1000,window=1.0,seed=None,
                                          out_type='timestamp')

    INPUTS
    gain    Scalar | (n_conds,) array-like. Spike rate gain (in spk/s) for
            each condition, which sets effect size.
            Scalar : spike rate difference between each successive condition
            (n_conds,) vector : specific spike rate gain for each condition
            Set = 0 to simulate no expected difference between conditions.
            Default: 5.0 (5 spk/s difference btwn each condition)

    offset  Scalar. Baseline rate added to condition effects. Default: 5.0 spk/s

    n_conds Int. Number of distinct conditions/groups to simulate. Default: 2

    n_trials Int. Number of trials/observations to simulate. Default: 1000

    window  Scalar. Time window to simulate spike train over. Default: 1.0 s

    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for actual random numbers.

    out_type String. Format of output spike trains:
            'timestamp' : Spike timestamps in s relative to trial starts [default]
            'boolean'   : Binary (0/1) vectors flagging spike times

    RETURNS
    trains  (n_trials,) of object | (n_trials,n_timepts) of bool.
            Simulated Poisson spike trains, either as list of timestamps relative
            to trial start or as binary vector for each trial (depending on out_type).

    labels  (n_trials,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
    """
    # TODO  Add support for multiple units w/ varying effect probability or size?

    if seed is not None: np.random.seed(seed)

    assert out_type in ['timestamp','boolean'], \
        ValueError("Unsupported value '%s' given for <out_type>. Should be 'timestamp' or 'boolean'" \
                   % out_type)

    # Is gain scalar-valued or array-like?
    scalar_gain = not isinstance(gain, (list, tuple, np.ndarray))

    if not scalar_gain:
        gain = np.asarray(gain)
        assert len(gain) == n_conds, \
            ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                       % (len(gain), n_conds))

    # Create (nTrials,) vector of group labels (ints in set 0-n_conds-1)
    n_reps = ceil(n_trials/n_conds)
    labels = np.tile(np.arange(n_conds),(n_reps,))[0:n_trials]
    # For easy visualization, sort trials by group number
    labels.sort()

    # Per-trial Poisson rate parameters = expected number of spikes/s
    # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
    if scalar_gain: lambdas = (offset + gain*labels)
    # Hand-set gain specific for each condition
    else:           lambdas = (offset + gain[labels])

    if out_type == 'timestamp':
        trains = np.empty((n_trials,),dtype=object)
    else:
        n_timepts = int(round(window*1000))
        trains = np.zeros((n_trials,n_timepts),dtype=bool)

    # Simulate Poisson spike trains with given lambda for each trial
    for i_trial,lam in enumerate(lambdas):
        # Lambda=0 implies no spikes at all, so leave empty
        if lam == 0:
            if out_type == 'timestamp': trains[i_trial] = np.asarray([],dtype=object)
            continue
        
        # Simulate inter-spike intervals. Poisson process has exponential ISIs.
        # HACK Generate 2x expected number of spikes, truncate below
        n_spikes_exp = lam*window
        ISIs = np.random.exponential(1/lam,(int(round(2*n_spikes_exp)),))

        # Integrate ISIs to get actual spike times
        timestamps = np.cumsum(ISIs)
        # Keep only spike times within desired time window
        timestamps = timestamps[timestamps < window]

        if out_type == 'timestamp':
            trains[i_trial] = timestamps
        # Convert timestamps to boolean spike train
        else:
            idxs = np.floor(timestamps*1000).astype('int')
            trains[i_trial,idxs] = True

    return trains, labels


def test_rate(method, plot=False, rates=(5,10,20,40), n_trials=1000, **kwargs):
    """
    Basic testing for functions estimating spike rate over time.
    
    Generates synthetic spike train data with given underlying rates,
    estimates rate using given function, and compares estimated to expected.
    
    means,sems = test_rate(method,plot=False,rates=(5,10,20,40),n_trials=1000, **kwargs)
                              
    INPUTS
    method  String. Name of rate estimation function to test:
            'bin_rate' | 'density'
            
    plot    Bool. Set=True to plot test results. Default: False
    
    rates   (n_rates,) array-like. List of expected spike rates to test
            Default: (5,10,20,40)
            
    n_trials Int. Number of trials to include in simulated data. Default: 1000

    **kwargs All other keyword args passed to rate estimation function
    
    RETURNS
    means   (n_rates,) ndarray. Estimated mean rate for each expected rate
    sems    (n_rates,) ndarray. SEM of mean rate for each expected rate
    
    ACTION
    Throws an error if any estimated rate is too far from expected value
    If <plot> is True, also generates a plot summarizing expected vs estimated rates
    """            
    if method in ['bin','bin_rate']:
        rate_func = bin_rate
        n_timepts = 20
        tbool     = np.ones((n_timepts,),dtype=bool)
    elif method == 'density':
        rate_func = density
        n_timepts = 1001
        # HACK For spike density method, remove edges, which are influenced by boundary artifacts
        t         = np.arange(0,1.001,0.001)
        tbool     = (t > 0.1) & (t < 0.9)        
    else:
        raise ValueError("Unsupported option '%s' given for <method>. \
                         Should be 'bin_rate'|'density'" % method)
       
    rates = np.asarray(rates)
                            
    means = np.empty((len(rates),))
    sems = np.empty((len(rates),))
        
    for i,rate in enumerate(rates):
        # Generate simulated spike train data and compute spike rates
        trains,_ = simulate_spike_trains(gain=0.0,offset=float(rate),
                                         n_conds=1,n_trials=n_trials,seed=0)
        
        spike_rates,t = rate_func(trains, lims=[0,1], **kwargs)
        if rate_func == bin_rate: t = t.mean(axis=1)  # bins -> centers
        # Take average across timepoints -> (n_trials,)
        spike_rates = spike_rates[:,tbool].mean(axis=1)
        
        # Compute mean and SEM across trials
        means[i] = spike_rates.mean(axis=0)
        sems[i]  = spike_rates.std(axis=0,ddof=0) / sqrt(n_trials)
        
    if plot:
        fig = plt.figure()
        fig.add_subplot(111,aspect='equal')        
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.plot([0,1.1*rates[-1]], [0,1.1*rates[-1]], '-', color='k', linewidth=1)
        plt.errorbar(rates, means, 3*sems, marker='o')
        
    # Determine if any estimated rates are outside acceptable range (expected mean +/- 3*SEM)
    errors = np.abs(means - rates) / sems

    # Find estimates that are clearly wrong
    bad_estimates = (errors > 3.3)        
    if bad_estimates.any():         
        if plot: plt.plot(rates[bad_estimates], means[bad_estimates]+0.1, '*', color='k')
        raise AssertionError("%d tested rates failed" % bad_estimates.sum())
        
    return means, sems
                

#==============================================================================
# Helper functions
#==============================================================================
def _spike_data_type(data):
    """
    Determines what type of spiking data we have:
    'timestamp' :   spike timestamps (in a Numpy object array)
    'bool' :        binary spike train (all values 0 or 1)
    'rate' :        array of spike rate/counts
    """
    if data.dtype == 'object':  return 'timestamp'
    elif _isbinary(data):       return 'bool'
    else:                       return 'rate'


def _isbinary(x):
    """
    Tests whether variable contains only binary values (True,False,0,1)
    """
    x = np.asarray(x)
    return (x.dtype == bool) or \
           (np.issubdtype(x.dtype,np.number) and \
            np.all(np.in1d(x,[0,0.0,1,1.0,True,False])))


def _unsorted_unique(x):
    """
    Implements np.unique(x) without sorting, ie maintains original order of
    unique elements as they are found in x.

    SOURCE
    stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    """
    x    = np.asarray(x)
    idxs = np.unique(x,return_index=True)[1]
    return x[np.sort(idxs)]


def _iarange(start=0, stop=0, step=1):
    """
    Implements Numpy arange() with an inclusive endpoint. Same inputs as arange(), same
    output, except ends at stop, not stop - 1 (or more generally stop - step)

    r = _iarange(start=0,stop=0,step=1)
    """
    # TODO  Generalize to deal w/ different input configs (eg iarange(stop))
    if isinstance(step,int):    return np.arange(start,stop+1,step)
    else:                       return np.arange(start,stop+1e-12,step)


def _enclose_in_object_array(data):
    """ Enclose array within an object array """
    out = np.empty((1,),dtype=object)
    out[0] = data
    return out


def _remove_buffer(data, buffer, axis=-1):
    """
    Removes a temporal buffer (eg zeros or additional samples) symmmetrically
    prepended/appended to data to avoid edge effects.

    data = _remove_buffer(data,buffer,axis=-1)

    INPUTS
    data    Data array where a buffer has been appended on both ends of time
            dimension. Can be any arbitrary size, typically
            (n_trials,nNeurons,n_timepts+2*buffer).
    buffer  Scalar. Length (number of samples) of buffer appended to each end.
    axis    Int. Array axis to remove buffer from (ie time dim). Default: -1

    RETURNS
    data    Data array with buffer removed, reducing time axis to n_timepts
            (typically shape (n_trials,nNeurons,n_timepts))
    """
    if axis == -1:
        return data[...,buffer:-buffer]
    else:
        return (data.swapaxes(-1,axis)[...,buffer:-buffer]
                    .swapaxes(axis,-1))


def _str_to_kernel(kernel, width, **kwargs):
    """ Converts string specifier to scipy.signal.windows function """
    if isinstance(kernel,str):  kernel = kernel.lower()

    if isinstance(kernel,np.ndarray):       return kernel
    elif callable(kernel):                  return kernel(width,**kwargs)
    elif kernel in ['hann','hanning']:      return hann(int(round(width*2.0)))
    elif kernel in ['gaussian','normal']:   return gaussian(int(round(width*6.0)),width)
    else:
        raise ValueError("Unsupported value '%s' given for kernel. \
                         Should be 'hanning'|'gaussian'" % kernel)

"""
validity_test_plots.py

Suite of tests to assess "face validity" of plotting functions in plots.py
Usually used to test new or majorly updated functions.

Includes tests that plots come out as expected for typical usage

FUNCTIONS

"""
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt

from neural_analysis.spectra import compute_tapers, simulate_oscillation
from neural_analysis.plots import plot_line_with_error_fill, plot_lineseries, plot_heatmap


def laminar_oscillation():
    """
    Simulates data with bursty oscillation across several electrodes (~ laminar probe)
    Should probably be a fixture, but difficult to call them outside pytest

    RETURNS
    data        (16,1000) ndarray. Simulated multielectrode oscillatory data.
                (eg simulating 1000 timepoints x 16 channels)
    timepts     (1000,) ndarray. Time sampling vector
    channels    (16,) ndarray. Channel sampling vector
    """
    n_chnls = 16
    n_timepts = 1000
    channels = np.arange(n_chnls)
    timepts = np.arange(n_timepts)/1000
    
    frequency = 16
    data = simulate_oscillation(frequency, amplitude=5.0, phase=0, noise=0,
                                n_trials=1, time_range=1.0, smp_rate=1000).T
    
    time_env = np.exp(-0.5*((timepts-200e-3)/50e-3)**2) + \
               0.5*np.exp(-0.5*((timepts-800e-3)/150e-3)**2) # Temporal envelope
    spatial_wgts = np.exp(-0.5*((channels-12)/3)**2) - 0.5*np.exp(-0.5*((channels-4)/4)**2)
    
    data = np.tile(data*time_env,(n_chnls,1)) * spatial_wgts[:,np.newaxis]

    return data, timepts, channels


def test_plot_line_with_error_fill(plot_dir=None):
    """
    Basic testing for plotting function plot_line_with_error_fill()

    ARGS
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.

    ACTIONS Creates a plot and optionally saves it to PNG file
    """
    # Use dpss tapers as data to plot, since they are all very distinct looking
    data = compute_tapers(1000, time_width=1.0, freq_width=4, n_tapers=3).T
    errs = 0.1*np.ptp(data)*np.ones_like(data)
    n_lines, n_timepts = data.shape
    timepts = np.arange(n_timepts)/1000
    events = [(100e-3,300e-3), 500e-3, (880e-3,900e-3,920e-3)]

    # Basic test plot
    plt.figure()
    plot_line_with_error_fill(timepts, data, err=errs, events=events)
    plt.title('Basic test plot')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error_fill.png'))

    # Test w/ different plot colors
    plt.figure()
    plot_line_with_error_fill(timepts, data, err=errs, events=events, color=['C1','C2','C3'])
    plt.title('Different plot colors')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error_fill-colors.png'))

    # Test w/ different x/ylims
    plt.figure()
    plot_line_with_error_fill(timepts, data, err=errs, events=events, xlim=(-0.1,1.1), ylim=(-3,3))
    plt.title('Different x/ylims')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error-lims.png'))

    # Test w/ different x/y indexing
    plt.figure()
    plot_line_with_error_fill(timepts-0.5, data+1, err=errs, events=events)
    plt.title('Changed x/y sampling values')    
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error-xyvalues.png'))

    # Test w/ no error fills
    plt.figure()
    plot_line_with_error_fill(timepts, data, events=events)
    plt.title('No error fills')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error_fill-noerrors.png'))

    # Test w/ x/ylabels and no event markers
    plt.figure()
    plot_line_with_error_fill(timepts, data, err=errs, xlabel='Time (s)', ylabel='Rate (spk/s)')
    plt.title('x/y labels, no event markers')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error_fill-noevents.png'))

    # Test w/ event markers at and beyond plot edge
    events.extend([(950e-3,1050e-3), (1500e-3,1600e-3)])
    plt.figure()
    plot_line_with_error_fill(timepts, data, err=errs, events=events)
    plt.title('Events at the edge')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_line_with_error_fill-edge_events.png'))


def test_plot_lineseries(plot_dir=None):
    """
    Basic testing for plotting function plot_lineseries()

    ARGS
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.

    ACTIONS Creates a plot and optionally saves it to PNG file
    """
    # Simulate 16 channels of laminar-like data with oscillation tapered in time
    # and with a reversal spatial profile across layers
    data, timepts, channels = laminar_oscillation()
    n_chnls, n_timepts = data.shape

    events = [(100e-3,300e-3), 500e-3, (880e-3,900e-3,920e-3)]

    # Basic test plot
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events)
    plt.title('Basic test plot')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries.png'))

    # Test w/ different plot colors
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events,
                    color=['C'+str(j) for j in range(n_chnls)])
    plt.title('Different line colors')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-colors.png'))

    # Test w/ different channel labels
    plt.figure()
    plot_lineseries(timepts, ['channel '+str(j+1) for j in range(n_chnls)], data, events=events)
    plt.title('Different channel labels')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-colors.png'))

    # Test w/ smaller scale
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events, scale=0.75)
    plt.title('Smaller y-axis scale')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-scale75.png'))

    # Test w/ different x/ylims
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events, xlim=(-0.1,1.1), ylim=(-2,n_chnls+1))
    plt.title('Different x/y lims')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-lims.png'))

    # Test w/ different x/y indexing
    plt.figure()
    plot_lineseries(timepts-0.5, channels+1, data, events=events)
    plt.title('Changed x/y sampling values')    
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-xyvalues.png'))

    # Test w/ inverted y-axis
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events, origin='lower')
    plt.title('Inveted y-axis')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-origin_lower.png'))

    # Test w/ x/ylabels and no event markers
    plt.figure()
    plot_lineseries(timepts, channels, data, xlabel='Time (s)', ylabel='Channels')
    plt.title('x/y labels, no event markers')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-noevents.png'))

    # Test w/ event markers at and beyond plot edge
    events.extend([(950e-3,1050e-3), (1500e-3,1600e-3)])
    plt.figure()
    plot_lineseries(timepts, channels, data, events=events)
    plt.title('Events at the edge')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_lineseries-edge_events.png'))
    

def test_plot_heatmap(plot_dir=None):
    """
    Basic testing for plotting function plot_heatmap()

    ARGS
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.

    ACTIONS Creates a plot and optionally saves it to PNG file
    """
    # Simulate 16 channels of laminar-like data with oscillation tapered in time
    # and with a reversal spatial profile across layers
    data, timepts, channels = laminar_oscillation()
    n_chnls, n_timepts = data.shape
    events = [(100e-3,300e-3), 500e-3, (880e-3,900e-3,920e-3)]

    # Basic test plot
    plt.figure()
    plot_heatmap(timepts, channels, data, events=events)
    plt.title('Basic test')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap.png'))

    # Test w/ inverted y-axis
    plt.figure()
    plot_heatmap(timepts, channels, data, events=events, origin='upper')
    plt.title('Inverted y-axis')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-origin_upper.png'))

    # Test w/ different x/ylims
    plt.figure()
    plot_heatmap(timepts, channels, data, events=events, xlim=(-0.1,1.1), ylim=(-1.5,n_chnls+0.5))
    plt.title('Different x/y lims')        
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-lims.png'))

    # Test w/ different x/y indexing
    plt.figure()
    plot_heatmap(timepts-0.5, channels+1, data, events=events)
    plt.title('Changed x/y sampling values')    
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-xyvalues.png'))

    # Test w/ different colormap
    plt.figure()
    plot_heatmap(timepts, channels, data, events=events, cmap='jet')
    plt.title('jet colormap')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-cmap_jet.png'))

    # Test w/ squashed color range
    plt.figure()
    clim = (data.min(), data.max())
    clim = (clim[0]+0.25*np.diff(clim), clim[1]-0.25*np.diff(clim))
    plot_heatmap(timepts, channels, data, events=events, clim=clim)
    plt.title('Squashed color range')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-clim_squash.png'))

    # Test w/ x/ylabels and no event markers
    plt.figure()
    plot_heatmap(timepts, channels, data, xlabel='Time (s)', ylabel='Channels')
    plt.title('x/y labels; no events')        
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-noevents.png'))
    
    # Test w/ event markers at and beyond plot edge
    events.extend([(950e-3,1050e-3), (1500e-3,1600e-3)])
    plt.figure()
    plot_heatmap(timepts, channels, data, events=events)
    plt.title('Edge event markers')
    if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'plot_heatmap-edge_events.png'))
    

"""
validity_test_spike_analysis.py

Suite of tests to assess "face validity" of spiking data analysis functions in spike_analysis.py
Usually used to test new or majorly updated functions to ensure they perform as expected.

Includes tests that parametrically estimate spike rate as a function of the simulated data mean,
number of trials, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_rate           Contains tests of spike rate estimation functions
rate_test_battery   Runs standard battery of tests of rate estimation functions
"""
# TODO  Generalize testing framework cf other validity test functions
# TODO  Code up n_trials test, test battery function
import os
import time
import numpy as np
import matplotlib.pyplot as plt

from spike_analysis import simulate_spike_trains, times_to_bool, rate


def test_rate(method, rates=(5,10,20,40), data_type='timestamp', n_trials=1000,
              plot=False, plot_dir=None, seed=1, **kwargs):
    """
    Basic testing for functions estimating spike rate over time.
    
    Generates synthetic spike train data with given underlying rates,
    estimates rate using given function, and compares estimated to expected.
    
    means,sems = test_rate(method,rates=(5,10,20,40),data_type='timestamp',n_trials=1000,
                           plot=False,plot_dir=None,seed=1, **kwargs)
                              
    ARGS
    method      String. Name of rate estimation function to test: 'bin' | 'density'
                
    rates       (n_rates,) array-like. List of expected spike rates to test
                Default: (5,10,20,40)
            
    data_type   String. Type of spiking data to input into rate functions:           
                'timestamp' [default] | 'bool' (0/1 binary spike train)

    n_trials Int. Number of trials to include in simulated data. Default: 1000

    plot    Bool. Set=True to plot test results. Default: False

    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.

    seed    Int. Random generator seed for repeatable results.
            Set=None for fully random numbers. Default: 1 (reproducible random numbers)
    
    **kwargs All other keyword args passed to rate estimation function
    
    RETURNS
    means   (n_rates,) ndarray. Estimated mean rate for each expected rate
    sems    (n_rates,) ndarray. SEM of mean rate for each expected rate
    
    ACTION
    Throws an error if any estimated rate is too far from expected value
    If <plot> is True, also generates a plot summarizing expected vs estimated rates
    """
    assert data_type in ['timestamp','bool'], \
        ValueError("Unsupported value '%s' given for data_type. Should be 'timestamp' | 'bool" % data_type)
        
    if method == 'bin':
        n_timepts = 20
        tbool     = np.ones((n_timepts,),dtype=bool)
        
    elif method == 'density':
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
    if plot: time_series = np.empty((n_timepts,len(rates)))
        
    for i,rate in enumerate(rates):
        # Generate simulated spike train data
        trains,_ = simulate_spike_trains(gain=0.0,offset=float(rate),data_type='timestamp',
                                         n_conds=1,n_trials=n_trials,seed=seed)
        
        # Convert spike timestamps -> binary 0/1 spike trains (if requested)
        if data_type == 'bool':
            trains,t = times_to_bool(trains,lims=[0,1])
            kwargs.update(t=t)      # Need <t> input for bool data
            
        # Compute spike rate from simulated spike trains -> (n_trials,n_timepts)
        spike_rates,t = rate(trains, method=method, lims=[0,1], **kwargs)
        if method == 'bin': t = t.mean(axis=1)  # bins -> centers
        
        if plot: time_series[:,i] = spike_rates.mean(axis=0)
        # Take average across timepoints -> (n_trials,)
        spike_rates = spike_rates[:,tbool].mean(axis=1)
        
        # Compute mean and SEM across trials
        means[i] = spike_rates.mean(axis=0)
        sems[i]  = spike_rates.std(axis=0,ddof=0) / sqrt(n_trials)
        
    # Optionally plot summary of test results
    if plot:
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot time course of estimated rates
        ax = plt.subplot(1,2,1)
        ylim = (0,1.05*time_series.max())
        for i,rate in enumerate(rates):
            plt.plot(t, time_series[:,i], '-', color=colors[i], linewidth=1.5)
            plt.text(0.99, (0.95-0.05*i)*ylim[1], np.round(rate,decimals=2), 
                     color=colors[i], fontweight='bold', horizontalalignment='right')            
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.ylim(ylim)
        plt.xlabel('Time')
        plt.ylabel('Estimated rate')
        
        # Plot across-time mean rates                
        ax = plt.subplot(1,2,2)
        ax.set_aspect('equal', 'box')
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.plot([0,1.1*rates[-1]], [0,1.1*rates[-1]], '-', color='k', linewidth=1)
        plt.errorbar(rates, means, 3*sems, marker='o')
        plt.xlabel('Simulated rate (spk/s)')
        plt.ylabel('Estimated rate')
        if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'rate-%s-%s.png' % (method,data_type)))
        
    # Determine if any estimated rates are outside acceptable range (expected mean +/- 3*SEM)
    errors = np.abs(means - rates) / sems

    # Find estimates that are clearly wrong
    bad_estimates = (errors > 3.3)        
    if bad_estimates.any():         
        if plot: plt.plot(rates[bad_estimates], means[bad_estimates]+0.1, '*', color='k')
        raise AssertionError("%d tested rates failed" % bad_estimates.sum())
        
    return means, sems
                

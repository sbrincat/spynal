"""
validity_test_spectra.py

Suite of tests to assess "face validity" of spectral analysis functions in spectra.py
Usually used to test new or majorly updated functions.  

Includes tests that parametrically estimate power as a function of frequency, amplitude, phase,
n, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_power              Contains tests of spectral estimation functions
power_test_battery      Runs standard battery of tests of spectral estimation functions
"""
import os
import time
from math import pi, sqrt, ceil, floor, log2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import bernoulli

from ..spectra import simulate_oscillation, power_spectrogram, burst_analysis

# TODO  Run tests of burst_analysis

def test_power(method, test='frequency', test_values=None, plot=False, plot_dir=None, seed=1,
               amp=5.0, freq=32, phi=0, noise=0.5, n=1000, time_range=3.0, smp_rate=1000, 
               burst_rate=0, spikes=False, **kwargs):
    """
    Basic testing for functions estimating time-frequency spectral power 
    
    Generates synthetic LFP data using given network simulation,
    estimates spectrogram using given function, and compares estimated to expected.
    
    means,sems = test_power(method,test='frequency',value=None,plot=False,plot_dir=None,seed=1,
                            amp=5.0,freq=32,phi=0,noise=0.5,n=1000,time_range=3.0,smp_rate=1000,
                            burst_rate=0,**kwargs)
                              
    ARGS
    method  String. Name of time-frequency spectral estimation function to test:
            'wavelet' | 'multitaper' | 'bandfilter' | 'burst'
            
    test    String. Type of test to run. Default: 'frequency'. Options:
            'frequency' Tests multiple simulated oscillatory frequencies
                        Checks for monotonic increase of peak freq
            'amplitude' Tests multiple simulated amplitudes at same freq
                        Checks for monotonic increase of amplitude
            'n'         Tests multiple values of number of trials (n)
                        Checks that power doesn't greatly vary with n.
            'burst_rate' Checks that oscillatory burst rate increases
                        as it's increased in simulated data.

    test_values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'frequency' List of frequencies to test. Default: [4,8,16,32,64]
            'amplitude' List of oscillation amplitudes to test. Default: [1,2,5,10,20]
            'n'         Trial numbers. Default: [25,50,100,200,400,800]

    plot    Bool. Set=True to plot test results. Default: False
          
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
          
    seed    Int. Random generator seed for repeatable results.
            Set=None for fully random numbers. Default: 1 (reproducible random numbers)
                           
    - Following args set param's for simulation, may be overridden by <test_values> depending on test -
    amp     Scalar. Simulated oscillation amplitude (a.u.) if test != 'amplitude'. Default: 5.0
    freq    Scalar. Simulated oscillation frequency (Hz) if test != 'frequency'. Default: 32
    phi     Scalar. Simulated oscillation phase (rad). Default: 0
    noise   Scalar. Additive noise for simulated signal (a.u., same as amp). Default: 0.5
    n       Int. Number of trials to simulate if test != 'n'. Default: 1000
    time_range Scalar. Full time range to simulate oscillation over (s). Default: 1.0
    smp_rate Int. Sampling rate for simulated data (Hz). Default: 1000
    burst_rate Scalar. Oscillatory burst rate (bursts/trial). Default: 0 (non-bursty)
    
    **kwargs All other keyword args passed to spectral estimation function given by <method>.
    
    RETURNS
    means   (n_freqs,n_timepts,n_values) ndarray. Estimated mean spectrogram for each tested value.
    sems    (n_freqs,n_timepts,n_values) ndarray. SEM of mean spectrogram for each tested value.
    
    ACTION
    Throws an error if any estimated power value is too far from expected value
    If <plot> is True, also generates a plot summarizing expected vs estimated power
    """
    method = method.lower()
    test = test.lower()
    
    # Set defaults for tested values and set up rate generator function depending on <test>
    if test in ['frequency','freq']:
        test_values = [4,8,16,32,64] if test_values is None else test_values
        gen_data = lambda freq: simulate_oscillation(freq,amplitude=amp,phase=phi,n_trials=n,noise=noise,
                                                     time_range=time_range,burst_rate=burst_rate,seed=seed)
        
    elif test in ['amplitude','amp']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        gen_data = lambda amp: simulate_oscillation(freq,amplitude=amp,phase=phi,n_trials=n,noise=noise,
                                                     time_range=time_range,burst_rate=burst_rate,seed=seed)
        
    elif test in ['phase','phi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        gen_data = lambda phi: simulate_oscillation(freq,amplitude=amp,phase=phi,n_trials=n,noise=noise,
                                                    time_range=time_range,burst_rate=burst_rate,seed=seed)
        
    elif test in ['n','n_trials']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        gen_data = lambda n: simulate_oscillation(freq,amplitude=amp,phase=phi,n_trials=n,noise=noise,
                                                     time_range=time_range,burst_rate=burst_rate,seed=seed)
        
    elif test in ['burst_rate','burst']:
        test_values = [0.1,0.2,0.4,0.8] if test_values is None else test_values
        gen_data = lambda rate: simulate_oscillation(freq,amplitude=amp,phase=phi,n_trials=n,noise=noise,
                                                     time_range=time_range,burst_rate=rate,seed=seed)        
    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)
    
    # Ensure hand-set values are sorted (ascending), as many tests assume it
    test_values = sorted(test_values)
    n_values = len(test_values)
        
    # Set default parameters for each spectral estimation method
    do_burst = method in ['burst','burst_analysis']
    # Special case: oscillatory burst analysis
    if do_burst:
        # KLUDGE  Reset spectral analysis <method> to 'wavelet' (unless something set explicitly in kwargs)
        if 'bands' not in kwargs:       kwargs['bands'] = ((2,6),(6,10),(10,22),(22,42),(42,86))
            
    elif method == 'multitaper':
        if 'freq_range' not in kwargs:  kwargs['freq_range'] = [1,100]
    
    elif method == 'bandfilter':
        if 'freqs' not in kwargs:       kwargs['freqs'] = ((2,6),(6,10),(10,22),(22,42),(42,86))
            
    if 'buffer' not in kwargs: kwargs['buffer'] = 1.0
                    
    for i,value in enumerate(test_values):
        # print("Running test value %d/%d: %.2f" % (i+1,n_values,value))
        
        # Simulate data with oscillation of given params -> (n_timepts,n_trials)
        data = gen_data(value)
        
        # HACK Convert continuous oscillatory data into spike train (todo find better method)
        if spikes:
            data = (data - data.min()) / data.ptp() # Convert to 0-1 range ~ spike probability
            data = data**2                          # Sparsify probabilies (decrease rates)
            # Use probabilities to generate Bernoulli random variable at each time point
            data = bernoulli.ppf(0.5, data).astype(bool)        
                        
        spec,freqs,timepts = power_spectrogram(data,smp_rate,axis=0,method=method,**kwargs)
        if freqs.ndim == 2:
            bands = freqs            
            freqs = freqs.mean(axis=1)  # Compute center of freq bands
        n_freqs,n_timepts,n_trials = spec.shape
        
        # KLUDGE Initialize output arrays on 1st loop, once spectrogram output shape is known
        if i == 0:
            means = np.empty((n_freqs,n_timepts,n_values))
            sems = np.empty((n_freqs,n_timepts,n_values))
            
        # Compute across-trial mean and SEM of time-frequency data -> (n_freqs,n_timepts,n_values)
        means[:,:,i] = spec.mean(axis=2)
        sems[:,:,i]  = spec.std(axis=2,ddof=0) / sqrt(n_trials)


    # Compute mean across all timepoints -> (n_freqs,n_values) frequency marginal
    marginal_means = means.mean(axis=1)
    marginal_sems = sems.mean(axis=1)    
             
    # For bandfilter, plot frequency bands in categorical fashion
    if do_burst or (method == 'bandfilter'):
        freq_transform  = lambda x: np.argmin(np.abs(x - freqs))  # Index of closest sampled freq
        plot_freqs      = np.arange(n_freqs)
        freq_ticks      = np.arange(n_freqs)
        freq_tick_labels= bands
             
    # For wavelets, evaluate and plot frequency on log scale
    elif method == 'wavelet':
        freq_transform  = np.log2
        plot_freqs      = freq_transform(freqs)
        fmin            = ceil(log2(freqs[0]))
        fmax            = floor(log2(freqs[-1]))    
        freq_ticks      = np.arange(fmin,fmax+1)
        freq_tick_labels= 2**np.arange(fmin,fmax+1)
        
    # For multitaper, evaluate and plot frequency on linear scale        
    elif method == 'multitaper':
        freq_transform  = lambda x: x
        plot_freqs      = freqs
        fmin            = ceil(freqs[0]/10.0)*10.0
        fmax            = floor(freqs[-1]/10.0)*10.0                
        freq_ticks      = np.arange(fmin,fmax+1,10).astype(int)
        freq_tick_labels= freq_ticks        
          
    freqs_transformed   = np.asarray([freq_transform(f) for f in freqs])
            
    # For frequency test, find frequency with maximal power for each test
    if test in ['frequency','freq']:
        idxs = np.argmax(marginal_means,axis=0)
        peak_freqs = freqs[idxs] if not(do_burst or (method == 'bandfilter')) else idxs
        
        # Find frequency in spectrogram closest to each simulated frequency
        test_freq_idxs  = np.asarray([np.argmin(np.abs(freq_transform(f) - freqs_transformed)) for f in test_values])
    else:        
        # Find frequency in spectrogram closest to simulated frequency
        test_freq_idxs  = np.argmin(np.abs(freq_transform(freq) - freqs_transformed))
                
    # Extract mean,SEM of power at tested frequency(s)
    test_freq_means = marginal_means[test_freq_idxs,:]
    test_freq_errs  = marginal_sems[test_freq_idxs,:]
    
    # Plot summary of test results                
    if plot:
        dt      = np.diff(timepts).mean()
        tlim    = [timepts[0]-dt/2, timepts[-1]+dt/2]
        df      = np.diff(plot_freqs).mean()
        flim    = [plot_freqs[0]-df/2, plot_freqs[-1]+df/2]
        
        # # Plot spectrogram for each tested value
        # plt.figure()
        # n_subplots = [floor(n_values/2), ceil(n_values/floor(n_values/2))]
        # for i,value in enumerate(test_values):
        #     ax = plt.subplot(n_subplots[0],n_subplots[1],i+1)
        #     plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        #     target_freq = freq_transform(value) if test in ['frequency','freq'] else freq_transform(freq)
        #     if not (do_burst or (method == 'bandfilter')):
        #         plt.plot(tlim, [target_freq,target_freq], '-', color='r', linewidth=0.5)
        #     plt.imshow(means[:,:,i], extent=[*tlim,*flim], aspect='auto', origin='lower')
        #     if i in [0,n_subplots[1]]:
        #         plt.yticks(freq_ticks,freq_tick_labels)
        #     else:
        #         ax.set_xticklabels([])
        #         plt.yticks(freq_ticks,[])
        #     plt.title(np.round(value,decimal=2))
        #     plt.colorbar()
        # plt.show()
        # if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'power-spectrogram-%s-%s-%s.png' % (method,test)))
        
        # Plot time-averaged spectrum for each tested value
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ylim = [0,1.05*marginal_means.max()]
        for i,value in enumerate(test_values):
            plt.plot(plot_freqs, marginal_means[:,i], '.-', color=colors[i], linewidth=1.5)
            target_freq = freq_transform(value) if test in ['frequency','freq'] else freq_transform(freq)
            if not (do_burst or (method == 'bandfilter')):
                plt.plot([target_freq,target_freq], ylim, '-', color=colors[i], linewidth=0.5)
            plt.text(0.9*flim[1], (0.95-i*0.05)*ylim[1], value, color=colors[i], fontweight='bold')
        plt.xlim(flim)
        plt.ylim(ylim)
        plt.xticks(freq_ticks,freq_tick_labels)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.title("%s %s test" % (method,test))
        plt.show()
        if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'power-spectrum-%s-%s.png' % (method,test)))
            
        # Plot summary curve of power (or peak frequency) vs tested value
        plt.figure()
        ax = plt.subplot(1,1,1)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        if test in ['frequency','freq']:
            lim = (0,1.1*freq_transform(test_values[-1]))
            plt.plot(lim, lim, color='k', linewidth=0.5)
            if do_burst or (method == 'bandfilter'):
                plt.plot([freq_transform(f) for f in test_values], peak_freqs, marker='o')
            else:
                plt.plot([freq_transform(f) for f in test_values], [freq_transform(f) for f in peak_freqs], marker='o')                
            plt.xticks(freq_ticks,freq_tick_labels)
            plt.yticks(freq_ticks,freq_tick_labels)
            plt.xlim(lim)
            plt.ylim(lim)            
            ax.set_aspect('equal', 'box')
        else:
            plt.errorbar(test_values, test_freq_means, 3*test_freq_errs, marker='o')    
        plt.xlabel(test)
        plt.ylabel('frequency' if test in ['frequency','freq'] else 'power')
        plt.title("%s %s test" % (method,test))
        plt.show()
        if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'power-summary-%s-%s.png' % (method,test)))
                
    ## Determine if test actually produced the expected values
    # frequency test: check if frequency of peak power matches simulated target frequency
    if test in ['frequency','freq']:
        assert (np.diff(peak_freqs) >= 0).all(), \
            AssertionError("Estimated peak frequency does not increase monotonically with expected frequency")
            
    # 'amplitude' : Test if power increases monotonically with simulated amplitude            
    elif test in ['amplitude','amp']:
        assert (np.diff(test_freq_means) > 0).all(), \
            AssertionError("Estimated power does not increase monotonically with simulated oscillation amplitude")

    # 'n' : Test if power is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        assert test_freq_means.ptp() < test_freq_errs.max(), \
            AssertionError("Estimated power has larger than expected range across n's (likely biased by n)")
    
    # 'burst_rate': Test if measured burst rate increases monotonically with simulated burst rate
    elif test in ['burst_rate','burst']:
        assert (np.diff(test_freq_means) > 0).all(), \
            AssertionError("Estimated burst rate does not increase monotonically with simulated oscillation burst rate")

    return means,sems
    
    
def power_test_battery(methods=['wavelet','multitaper','bandfilter'],
                       tests=['frequency','amplitude','n','burst_rate'], **kwargs):
    """ 
    Runs a battery of given tests on given oscillatory power computation methods
    
    power_test_battery(methods=['wavelet','multitaper','bandfilter'],
                       tests=['frequency','amplitude','n','burst_rate'], **kwargs)
    
    ARGS
    methods     Array-like. List of power computation methods to test.
                Default: ['wavelet','multitaper','bandfilter'] (all supported methods)
                
    tests       Array-like. List of tests to run.
                Default: ['frequency','amplitude','n','burst_rate'] (all supported tests)
                
    kwargs      Any other kwargs passed directly to test_power()
    
    ACTION
    Throws an error if any estimated power value for any (method,test) is too far from expected value    
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    
    for test in tests:
        for method in methods:
            print("Running %s test on %s spectral analysis" % (test,method))
            extra_args = kwargs
            if (method in ['burst','burst_analysis']) and ('burst_rate' not in kwargs):
                extra_args['burst_rate'] = 0.4
                
            t1 = time.time()
            
            test_power(method, test=test, **extra_args)
            print('PASSED (test ran in %.1f s)' % (time.time()-t1))
            
            # If saving plots to file, let's not leave them all open
            if 'plot_dir' in kwargs: plt.close('all')            
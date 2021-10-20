"""
validity_test_sync.py

Suite of tests to assess "face validity" of synchrony analysis functions in sync.py
Usually used to test new or majorly updated functions.  

Includes tests that parametrically estimate synchrony as a function of frequency, amplitude, phase,
n, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_synchrony          Contains tests of synchrony estimation functions
synchrony_test_battery  Runs standard battery of tests of synchrony estimation functions
spike_field_test_battery Runs standard battery of tests of spike-field sync estimation functions
"""
import os
import time
from warnings import warn
from math import pi, ceil, floor, log2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import bernoulli

from neural_analysis.sync import simulate_multichannel_oscillation, synchrony

                
def test_synchrony(method, pair_type='lfp-lfp', test='frequency', test_values=None, 
                   spec_method='wavelet', do_tests=True, do_plots=False, plot_dir=None,
                   seed=1, phi_sd=pi/4, dphi=0, damp=1, amp=5.0, freq=32, phi=0, noise=0.5,n=1000,
                   time_range=3.0, smp_rate=1000, burst_rate=0, **kwargs):    
    """
    Basic testing for functions estimating bivariate time-frequency synchrony/coherence 
    
    Generates synthetic LFP data using given network simulation,
    estimates t-f synchrony using given function, and compares estimated to expected.
    
    syncs,phases,passed = test_synchrony(method,test='frequency',test_values=None,spec_method='wavelet',
                                         do_tests=True,do_plots=False,plot_dir=None,seed=1,
                                         phi_sd=pi/4,dphi=0,damp=1,amp=5.0,freq=32,phi=0,noise=0.5,n=1000,
                                         time_range=3.0,smp_rate=1000,burst_rate=0,**kwargs)
                              
    ARGS
    method  String. Name of synchrony estimation function to test:
            'PPC' | 'PLV' | 'coherence'

    test    String. Type of test to run. Default: 'frequency'. Options:
            'synchrony' Tests multiple values of strength of synchrony (by manipulating phase SD of one signal)
                        Checks for monotonic increase of synchrony measure
            'frequency' Tests multiple simulated oscillatory frequencies
                        Checks for monotonic increase of peak freq
            'relphase'  Tests multiple simulated btwn-signal relative phases (dphi)
                        Checks that synchrony doesn't vary appreciably
            'ampratio'  Test multiple btwn-signal amplitude ratios (damp)
                        Checks that synchrony doesn't vary appreciably
            'phase'     Tests multiple simulated absolute phases (phi)
                        Checks that synchrony doesn't vary appreciably
            'n'         Tests multiple values of number of trials (n)
                        Checks that synchrony doesn't greatly vary with n.   
                        
    test_values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'synchrony' Relative phase SD's to test (~inverse of synchrony strength).
                        Default: [pi,pi/2,pi/4,pi/8,0]
            'frequency' Frequencies to test. Default: [4,8,16,32,64]                        
            'relphase'  Relative phases to test. Default: [-pi,-pi/2,0,pi/2,pi]
            'ampratio'  Amplitude ratios to test. Default: [1,2,4,8]
            'amplitude' Oscillation amplitudes to test. Default: [1,2,5,10,20]
            'phase'     Absolute phases to test. Default: [-pi,-pi/2,0,pi/2,pi]
            'n'         Trial numbers. Default: [25,50,100,200,400,800]
                        
    spec_method  String. Name of spectral estimation function to use to 
            generate time-frequency representation to input into synchrony function
    
    do_tests Bool. If True, an error is raised if any tests fail to produce expected results.
            If False, only a warning is issued. Default: True
            
    do_plots Bool. Set=True to plot test results. Default: False
    
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
        
    seed    Int. Random generator seed for repeatable results.
            Set=None for fully random numbers. Default: 1 (reproducible random numbers)
          
    - Following args set param's for simulation, may be overridden by <test_values> depending on test - 
    phi_sd  Scalar. Gaussian SD (rad) for phase diff of 2 signals if test != 'synchrony'. Default: pi/2
    dphi    Scalar. Phase difference (rad) of 2 simulated signals if test != 'relphase'. Default: 0
    damp    Scalar. Amplitude ratio of 2 simulated signals. Default: 1 (same amplitude)
    amp     Scalar. Simulated oscillation amplitude (a.u.) if test != 'amplitude'. Default: 5.0
    freq    Scalar. Simulated oscillation frequency (Hz) if test != 'frequency'. Default: 32
    phi     Scalar. Simulated oscillation phase (rad). Default: 0
    noise   Scalar. Additive noise for simulated signal (a.u., same as amp). Default: 0.5    
    n       Int. Number of trials to simulate if test != 'n'. Default: 1000
    time_range Scalar. Full time range to simulate oscillation over (s). Default: 3 s
    smp_rate Int. Sampling rate for simulated data (Hz). Default: 1000
    burst_rate Scalar. Oscillatory burst rate (bursts/trial). Default: 0 (non-bursty)

    **kwargs All other keyword args passed to synchrony estimation function given by <method>.
            Can also include args to time-frequency spectral estimation function given by <spec_method>.
    
    RETURNS
    syncs   (n_freqs,n_timepts,n_values) ndarray. Estimated synchrony strength for each tested value
     
    phases  (n_freqs,n_timepts,n_values) ndarray. Estimated synchrony phase for each tested value
    
    passed  Bool. True if all tests produce expected values; otherwise False.
    
    ACTION
    If do_tests is True, raises an error if any estimated synchrony or phase value 
    is too far from expected value
    
    If do_plots is True, also generates a plot summarizing expected vs estimated synchrony
    """
    method = method.lower()
    test = test.lower()
    
    # Set defaults for tested values and set up rate generator function depending on <test>
    sim_args = dict(amplitude=[amp,amp*damp], phase=[phi+dphi,phi], phase_sd=[0,phi_sd],
                    n_trials=n, noise=noise, time_range=time_range, burst_rate=burst_rate, seed=seed)
    
    if test in ['synchrony','strength','coupling']:
        test_values = [pi, pi/2, pi/4, 0] if test_values is None else test_values
        del sim_args['phase_sd']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi_sd: simulate_multichannel_oscillation(2,freq,**sim_args,phase_sd=[0,phi_sd])
        
    elif test in ['relphase','rel_phase','dphi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']   # Delete preset arg so it uses argument to lambda below
        # Note: Implement dphi in 1st channel only so synchrony phase ends up monotonically *increasing* for tests below
        gen_data = lambda dphi: simulate_multichannel_oscillation(2,freq,**sim_args,phase=[phi+dphi,phi])

    elif test in ['ampratio','amp_ratio','damp']:
        test_values = [1,2,4,8] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda damp: simulate_multichannel_oscillation(2,freq,**sim_args,amplitude=[amp,amp*damp])
    
    elif test in ['frequency','freq']:
        test_values = [4,8,16,32,64] if test_values is None else test_values
        gen_data = lambda freq: simulate_multichannel_oscillation(2,freq,**sim_args)
        
    elif test in ['amplitude','amp']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda amp: simulate_multichannel_oscillation(2,freq,**sim_args,amplitude=[amp,amp*damp])
        
    elif test in ['phase','phi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi: simulate_multichannel_oscillation(2,freq,**sim_args,phase=[phi,phi+dphi])
                
    elif test in ['n','n_trials']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        del sim_args['n_trials']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n: simulate_multichannel_oscillation(2,freq,**sim_args,n_trials=n)        
        
    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)
    
    # Ensure hand-set values are sorted (ascending), as many tests assume it
    test_values = sorted(test_values,reverse=True) if test in ['synchrony','strength','coupling'] else sorted(test_values)
    n_values = len(test_values)
            
    # Set default parameters for each spectral estimation method            
    if spec_method == 'multitaper':
        if 'freq_range' not in kwargs:  kwargs['freq_range'] = [1,100] 
    elif spec_method == 'bandfilter':
        if 'freqs' not in kwargs:       kwargs['freqs'] = ((2,6),(6,10),(10,22),(22,42),(42,86))
                            
    if 'buffer' not in kwargs: kwargs['buffer'] = 1.0
                        
    for i,value in enumerate(test_values):
        # print("Running test value %d/%d: %.2f" % (i+1,n_values,value))
        
        # Simulate data with oscillation of given params in additive noise -> (n_timepts,n_trials,n_chnls=2)
        data = gen_data(value)
        if pair_type == 'spk-lfp':
            data[:,:,0] = _continuous_to_spiking(data[:,:,0])
                 
        # Compute time-frequency/spectrogram representation of data and
        # bivariate measure of synchrony -> (n_freqs,n_timepts)
        sync,freqs,timepts,phase = synchrony(data[:,:,0], data[:,:,1], axis=1, method=method,
                                             spec_method=spec_method, smp_rate=smp_rate,
                                             time_axis=0, return_phase=True, **kwargs)
        n_freqs,n_timepts = sync.shape
        if freqs.ndim == 2: 
            bands = freqs
            freqs = freqs.mean(axis=1)  # Compute center of freq bands
        
        # KLUDGE Initialize output arrays on 1st loop, once spectrogram output shape is known
        if i == 0:
            syncs = np.empty((n_freqs,n_timepts,n_values))
            phases = np.empty((n_freqs,n_timepts,n_values))
        
        syncs[:,:,i] = sync
        phases[:,:,i] = phase
                
    # Compute mean across all timepoints -> (n_freqs,n_values) frequency marginal for each tested value
    marginal_syncs = syncs.mean(axis=1)
    marginal_phases = np.angle(_amp_phase_to_complex(syncs,phases).mean(axis=1)) # weighted circular mean
        
    # For wavelets, evaluate and plot frequency on log scale
    if spec_method == 'wavelet':
        freq_transform  = np.log2
        plot_freqs      = freq_transform(freqs)        
        fmin            = ceil(log2(freqs[0]))
        fmax            = floor(log2(freqs[-1]))    
        freq_ticks      = np.arange(fmin,fmax+1)
        freq_tick_labels= 2**np.arange(fmin,fmax+1)
        
    # For bandfilter, plot frequency bands in categorical fashion
    elif spec_method == 'bandfilter':
        freq_transform  = lambda x: np.argmin(np.abs(x - freqs))  # Index of closest sampled freq
        plot_freqs      = np.arange(len(freqs))
        freq_ticks      = np.arange(len(freqs))
        freq_tick_labels= bands
                     
    # For other spectral analysis, evaluate and plot frequency on linear scale        
    else:
        freq_transform  = lambda x: x
        plot_freqs      = freqs        
        fmin            = ceil(freqs[0]/10.0)*10.0
        fmax            = floor(freqs[-1]/10.0)*10.0                
        freq_ticks      = np.arange(fmin,fmax+1,10).astype(int)
        freq_tick_labels= freq_ticks        
                
    freqs_transformed   = np.asarray([freq_transform(f) for f in freqs])
                             
    # For frequency test, find frequency with maximal synchrony for each test
    if test in ['frequency','freq']:
        idxs = np.argmax(marginal_syncs,axis=0)
        peak_freqs = freqs[idxs] if spec_method != 'bandfilter' else idxs

        # Find frequency in spectrogram closest to each simulated frequency
        test_freq_idxs  = np.asarray([np.argmin(np.abs(freq_transform(f) - freqs_transformed)) for f in test_values])
        
        # Extract synchrony and phase at each tested frequency   
        test_freq_syncs     = marginal_syncs[test_freq_idxs,np.arange(n_values)]
        test_freq_phases    = marginal_phases[test_freq_idxs,np.arange(n_values)]
        
    else:        
        # Find frequency in spectrogram closest to simulated frequency
        test_freq_idx  = np.argmin(np.abs(freq_transform(freq) - freqs_transformed))
        
        # Extract synchrony and phase at simulated frequency   
        test_freq_syncs     = marginal_syncs[test_freq_idx,:]
        test_freq_phases    = marginal_phases[test_freq_idx,:]
                
    if do_plots:
        dt      = np.diff(timepts).mean()
        tlim    = [timepts[0]-dt/2, timepts[-1]+dt/2]
        df      = np.diff(plot_freqs).mean()
        flim    = [plot_freqs[0]-df/2, plot_freqs[-1]+df/2]
                
        # # Plot synchrony/phase spectrogram for each tested value
        # n_subplots = [floor(n_values/2), ceil(n_values/floor(n_values/2))]
        # for i_vbl,variable in enumerate(['sync','phase']):
        #     plot_vals = syncs if variable == 'sync' else phases
        #     cmap = 'viridis' if variable == 'sync' else 'hsv'
        #     plt.figure()
        #     for i,value in enumerate(test_values):
        #         clim = [plot_vals[:,:,i].min(),plot_vals[:,:,i].max()] if variable == 'sync' else [-pi,pi]
        #         ax = plt.subplot(n_subplots[0],n_subplots[1],i+1)
        #         plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        #         target_freq = freq_transform(value) if test in ['frequency','freq'] else freq_transform(freq)
        #         plt.plot(tlim, [target_freq,target_freq], '-', color='r', linewidth=0.5)
        #         plt.imshow(plot_vals[:,:,i], extent=[*tlim,*flim], vmin=clim[0], vmax=clim[1], 
        #                    aspect='auto', origin='lower', cmap=cmap)
        #         if i in [0,n_subplots[1]]:
        #             plt.yticks(freq_ticks,freq_tick_labels)
        #         else:
        #             ax.set_xticklabels([])
        #             plt.yticks(freq_ticks,[])
        #         plt.title(np.round(value,decimal=2))
        #         plt.colorbar()
        #     plt.show()
        # if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'synchrony-spectrogram-%s-%s-%s.png' % (method,test,spec_method)))
        
                
        # Plot time-averaged synchrony/phase spectrum for each tested value
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        threshold_syncs = marginal_syncs > 0.1
        for i_vbl,variable in enumerate(['sync','phase']):
            plt.subplot(1,2,i_vbl+1)
            plot_vals = marginal_syncs if variable == 'sync' else marginal_phases
            ylim = [0,1.05*marginal_syncs.max()] if variable == 'sync' else [-pi,pi]         
            for i,value in enumerate(test_values):
                if variable == 'phase':
                    plt.plot(plot_freqs, plot_vals[:,i], '-', color=colors[i], linewidth=1.5, alpha=0.33)                    
                    plt.plot(plot_freqs[threshold_syncs[:,i]], plot_vals[threshold_syncs[:,i],i], '.-', 
                             color=colors[i], linewidth=1.5)
                else:
                    plt.plot(plot_freqs, plot_vals[:,i], '.-', color=colors[i], linewidth=1.5)                                      
                target_freq = freq_transform(value) if test in ['frequency','freq'] else freq_transform(freq)
                plt.plot([target_freq,target_freq], ylim, '-', color=colors[i], linewidth=0.5)
                plt.text(flim[1]-0.05*np.diff(flim), ylim[1]-(i+1)*0.05*np.diff(ylim), np.round(value,decimals=2),
                         color=colors[i], fontweight='bold', horizontalalignment='right')
            plt.xlim(flim)
            plt.ylim(ylim)
            plt.xticks(freq_ticks,freq_tick_labels)
            plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(variable)
            if i_vbl == 0: plt.title("%s %s %s test" % (spec_method,method,test), horizontalalignment='left')
        plt.show()
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'synchrony-spectrum-%s-%s-%s.png' % (method,test,spec_method)))
        
        # Plot summary curve of synchrony (or peak frequency) vs tested value
        plt.figure()
        for i_vbl,variable in enumerate(['sync','phase']):        
            ax = plt.subplot(1,2,i_vbl+1)
            plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
            if test in ['frequency','freq'] and variable == 'sync':
                lim = (0,1.1*freq_transform(test_values[-1]))
                plt.plot(lim, lim, color='k', linewidth=0.5)
                if spec_method == 'bandfilter':
                    plt.plot([freq_transform(f) for f in test_values], peak_freqs, marker='o')
                else:
                    plt.plot([freq_transform(f) for f in test_values], 
                             [freq_transform(f) for f in peak_freqs], marker='o')                
                plt.xticks(freq_ticks,freq_tick_labels)
                plt.yticks(freq_ticks,freq_tick_labels)
                plt.xlim(lim)
                plt.ylim(lim)            
                ax.set_aspect('equal', 'box')
            else:
                test_freq_results = test_freq_syncs if variable == 'sync' else test_freq_phases
                xplot = [freq_transform(f) for f in test_values] \
                        if spec_method == 'bandfilter' and test in ['frequency','freq'] \
                        else test_values
                ylim = [0,1.05*test_freq_syncs.max()] if variable == 'sync' else [-pi,pi]
                plt.plot(xplot, test_freq_results, marker='o')
                plt.ylim(ylim)
            plt.xlabel('Phase SD' if test == 'synchrony' else test)
            plt.ylabel('frequency' if test in ['frequency','freq'] and variable == 'sync' else variable)
            if i_vbl == 0:
                plt.title("%s %s %s test" % (spec_method,method,test), horizontalalignment='left')            
        plt.show()
        if plot_dir is not None:
            plt.savefig(os.path.join(plot_dir,'synchrony-summary-%s-%s-%s.png' % (method,test,spec_method)))
        
    
    ## Determine if test actually produced the expected values
    # 'synchrony' : Test if synchrony strength increases monotonically with simulated synchrony
    if test in ['synchrony','strength','coupling']:
        evals = [((np.diff(test_freq_syncs) > 0).all(),
                    "Estimated synchrony strength does not increase monotonically with simulated synchrony")]
        
    # 'frequency' : check if frequency of peak synchrony matches simulated target frequency
    elif test in ['frequency','freq']:
        evals = [((np.diff(peak_freqs) > 0).all(),
                    "Estimated peak frequency does not increase monotonically with expected frequency")]
            
    # 'amplitude','phase','ampratio' : Test if synchrony is ~ same for all values      
    elif test in ['amplitude','amp', 'phase','phi', 'ampratio','amp_ratio','damp']:
        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated synchrony has larger than expected range across tested %s value" % test)]

    # 'relphase' : Test if synchrony strength is ~ same for all values, phase increases monotonically      
    elif test in ['relphase','rel_phase','dphi']:
        circ_subtract = lambda data1,data2: np.angle(np.exp(1j*data1) / np.exp(1j*data2))
        circ_diff = lambda data: circ_subtract(data[1:],data[:-1])
        
        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated synchrony has larger than expected range across tested %s value" % test),
                 ((circ_diff(test_freq_phases) > 0).all(),
                    "Estimated synchrony phase does not increase monotonically with simulated reslative phase")]

    # 'n' : Test if synchrony is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated synchrony has larger than expected range across n's (likely biased by n)")]
     
    passed = True
    for cond,message in evals:
        if not cond:    passed = False
        
        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)
            
    return syncs, phases, passed
    
    
def synchrony_test_battery(methods=['PPC','PLV','coherence'],
                           tests=['synchrony','relphase','ampratio','frequency','amplitude','phase','n'],
                           spec_methods=['wavelet','multitaper','bandfilter'], do_tests=True, **kwargs):
    """ 
    Runs a battery of given tests on given oscillatory synchrony computation methods
    
    synchrony_test_battery(methods=['PPC','PLV','coherence'],
                           tests=['synchrony','relphase','ampratio','frequency','amplitude','phase','n'],
                           spec_methods=['wavelet','multitaper','bandfilter'],do_tests=True,**kwargs)
    
    ARGS
    methods     Array-like. List of synchrony computation methods to test.
                Default: ['PPC','PLV','coherence'] (all supported methods)
                
    tests       Array-like. List of tests to run.
                Note: certain combinations of methods,tests are skipped, as they are not expected to pass
                (ie 'ampratio' tests skipped for coherence method)                
                Default: ['synchrony','relphase','ampratio','frequency','amplitude','phase','n'] 
                (all supported tests)
                
    spec_methods Array-like. List of underlying spectral analysis methods to test.                
                Default: ['wavelet','multitaper','bandfilter'] (all supported methods)
                
    do_tests    Bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True
            
    kwargs      Any other keyword passed directly to test_synchrony()
    
    ACTION
    Throws an error if any estimated synchrony or phase value for any (method,test) 
    is too far from expected value    
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    if isinstance(spec_methods,str): spec_methods = [spec_methods]
    tests = [test.lower() for test in tests]
    methods = [method.lower() for method in methods]
    
    for test in tests:
        for method in methods:
            for spec_method in spec_methods:
                t1 = time.time()
                # Skip tests expected to fail due to properties of given info measures
                # (eg ones that are biased/affected by n)
                if (test in ['n','n_trials']) and (method in ['coherence','coh','plv']):
                    do_tests_ = False
                elif (test in ['ampratio','amp_ratio','damp']) and (method in ['coherence','coh']):
                    do_tests_ = False
                else:
                    do_tests_ = do_tests
                    
                print("Running %s test on %s %s" % (test,spec_method,method))
                                
                _,_,passed = test_synchrony(method, pair_type='lfp-lfp', test=test, 
                                            spec_method=spec_method, do_tests=do_tests_, **kwargs)
                
                print('%s (test ran in %.1f s)' % ('PASSED' if passed else 'FAILED', time.time()-t1))
                                
                # If saving plots to file, let's not leave them all open
                if 'plot_dir' in kwargs: plt.close('all')
                
            
def spike_field_test_battery(methods=['PPC','PLV','coherence'],
                           tests=['synchrony','relphase','ampratio','frequency','amplitude','phase','n'],
                           spec_methods=['wavelet','multitaper','bandfilter'],do_tests=True, **kwargs):
    """ 
    Runs a battery of given tests on given oscillatory spike-field coupling computation methods
    
    spike_field_test_battery(methods=['PPC','PLV','coherence'],
                           tests=['synchrony','relphase','frequency','amplitude','phase','n'],
                           spec_methods=['wavelet','multitaper','bandfilter'], **kwargs)
    
    ARGS
    methods     Array-like. List of spike-field coupling computation methods to test.
                Default: ['PPC','PLV','coherence'] (all supported methods)
                
    tests       Array-like. List of tests to run.
                Note: certain combinations of methods,tests are skipped, as they are not expected to pass
                Default: ['synchrony','relphase','frequency','amplitude','phase','n'] (all supported tests)
                
    spec_methods Array-like. List of underlying spectral analysis methods to test.                
                Default: ['wavelet','multitaper','bandfilter'] (all supported methods)
                
    do_tests    Bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True
                                
    kwargs      Any other kwargs passed directly to test_synchrony()
    
    ACTION
    Throws an error if any estimated synchrony or phase value for any (method,test) is too far from
    expected value    
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    if isinstance(spec_methods,str): spec_methods = [spec_methods]
    tests = [test.lower() for test in tests]
    methods = [method.lower() for method in methods]
    
    for test in tests:
        for method in methods:
            for spec_method in spec_methods:
                print("Running %s test on %s %s" % (test,spec_method,method))
                t1 = time.time()
                # Skip tests expected to fail due to properties of given info measures
                # (eg ones that are biased/affected by n)
                if (test in ['n','n_trials']) and (method in ['coherence','coh','plv']):
                    do_tests_ = False
                elif (test in ['ampratio','amp_ratio','damp']) and (method in ['coherence','coh']):
                    do_tests_ = False
                else:
                    do_tests_ = do_tests
                               
                _,_,passed = test_synchrony(method, pair_type='spk-lfp', test=test, spec_method=spec_method, 
                                            do_tests=do_tests_, **kwargs)
                
                print('%s (test ran in %.1f s)' % ('PASSED' if passed else 'FAILED', time.time()-t1))
                
                # If saving plots to file, let's not leave them all open
                if 'plot_dir' in kwargs: plt.close('all')                
 
                
def _amp_phase_to_complex(amp,theta):
    """ Converts amplitude and phase angle to complex variable """
    return amp * np.exp(1j*theta)


def _continuous_to_spiking(data):
    """ Converts continuous data to thresholded spiking response """
    # Convert continuous oscillation to probability (range 0-1)
    data = (data - data.min()) / data.ptp()
    data = data**2  # Sparsen high rates some
    
    # Use probabilities to generate Bernoulli random variable at each time point
    return bernoulli.ppf(0.5, data).astype(bool)
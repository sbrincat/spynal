"""
Suite of tests to assess "face validity" of synchrony analysis functions in sync.py
Usually used to test new or majorly updated functions.

Includes tests that parametrically estimate synchrony as a function of frequency, amplitude, phase,
n, etc. to establish methods produce expected pattern of results.

Plots results and runs assertions that basic expected results are reproduced

Functions
---------
- test_synchrony :          Contains tests of synchrony estimation functions
- synchrony_test_battery :  Runs standard battery of tests of synchrony estimation functions
- spike_field_test_battery :Runs standard battery of tests of spike-field sync estimation functions
"""
import os
import time
from warnings import warn
from copy import deepcopy
from math import pi, ceil, floor, log2
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import bernoulli

from spynal.sync.sync import simulate_multichannel_oscillation, synchrony
from spynal.info import neural_info


# =============================================================================
# Validity test functions
# =============================================================================
def test_synchrony(method, pair_type='lfp-lfp', test='frequency', test_values=None,
                   spec_method='wavelet', single_trial=None,
                   do_tests=True, do_plots=False, plot_dir=None,
                   seed=1, phi_sd=pi/4, dphi=0, damp=1, amp=5.0, freq=32, phi=0, noise=0.5,
                   n=1000, time_range=3.0, smp_rate=1000, burst_rate=0, **kwargs):
    """
    Basic testing for functions estimating bivariate time-frequency oscillatory synchrony/coherence

    Generates synthetic LFP data using given parameters, estimates time-frequency synchrony
    using given function, and compares estimated to expected.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    method : str
        Name of synchrony estimation function to test. Options: 'PPC' | 'PLV' | 'coherence'

    test : str, default: 'frequency'
        Type of test to run. Options:
        - 'synchrony' : Tests multiple values of strength of synchrony (by manipulating phase SD
            of one signal). Checks for monotonic increase of synchrony measure.
        - 'frequency' : Tests multiple simulated oscillatory frequencies.
            Checks for monotonic increase of peak freq.
        - 'relphase' : Tests multiple simulated btwn-signal relative phases (dphi).
            Checks that synchrony doesn't vary appreciably.
        - 'ampratio' : Test multiple btwn-signal amplitude ratios (damp).
            Checks that synchrony doesn't vary appreciably.
        - 'phase' : Tests multiple simulated absolute phases (phi).
            Checks that synchrony doesn't vary appreciably.
        - 'n' : Tests multiple values of number of trials (n).
            Checks that synchrony doesn't greatly vary with n.

    test_values : array-like, shape=(n_values,), dtype=str
        List of values to test. Interpretation and defaults are test-specific:
        
        - 'synchrony' : Relative phase SD's (~inverse of synchrony). Default: [pi,pi/2,pi/4,pi/8,0]
        - 'frequency' : Frequencies to test. Default: [4,8,16,32,64]
        - 'relphase' :  Relative phases to test. Default: [-pi,-pi/2,0,pi/2,pi]
        - 'ampratio' :  Amplitude ratios to test. Default: [1,2,4,8]
        - 'amplitude' : Oscillation amplitudes to test. Default: [1,2,5,10,20]
        - 'phase' :     Absolute phases to test. Default: [-pi,-pi/2,0,pi/2,pi]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    spec_method : str, default: 'wavelet'
        Name of spectral estimation function to use to generate time-frequency representation
        to input into synchrony function

    single_trial : str or None, default: None
        What type of synchrony estimator to compute:
        - None :        standard across-trial estimator [default]
        - 'pseudo' :    single-trial estimates using jackknife pseudovalues
        - 'richter' :   single-trial estimates using actual jackknifes as in Richter & Fries 2015

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.
       
    - Following args set param's for sim, may be overridden by <test_values> depending on test -
    phi_sd  Scalar. SD (rad) for phase diff of 2 signals if test != 'synchrony'. Default: pi/2
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

    **kwargs :
        All other keyword args passed to synchrony estimation function
 
    Returns
    -------
    syncs : ndarray, shape=(n_freqs,n_timepts,n_values)
        Estimated synchrony strength for each tested value

    phases : ndarray=(n_freqs,n_timepts,n_values)
        Estimated synchrony phase for each tested value

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    method = method.lower()
    test = test.lower()

    # Set defaults for tested values and set up rate generator function depending on <test>
    sim_args = dict(amplitude=[amp,amp*damp], phase=[phi+dphi,phi], phase_sd=[0,phi_sd],
                    n_trials=n, noise=noise, time_range=time_range, burst_rate=burst_rate,
                    seed=seed)

    if test in ['synchrony','strength','coupling']:
        test_values = [pi, pi/2, pi/4, 0] if test_values is None else test_values
        del sim_args['phase_sd']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi_sd: simulate_multichannel_oscillation(2,freq,**sim_args,
                                                                    phase_sd=[0,phi_sd])

    elif test in ['relphase','rel_phase','dphi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']   # Delete preset arg so it uses argument to lambda below
        # Note: Implement dphi in 1st channel only so synchrony phase ends up monotonically
        # *increasing* for tests below
        gen_data = lambda dphi: simulate_multichannel_oscillation(2,freq,**sim_args,
                                                                  phase=[phi+dphi,phi])

    elif test in ['ampratio','amp_ratio','damp']:
        test_values = [1,2,4,8] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda damp: simulate_multichannel_oscillation(2,freq,**sim_args,
                                                                  amplitude=[amp,amp*damp])

    elif test in ['frequency','freq']:
        test_values = [4,8,16,32,64] if test_values is None else test_values
        gen_data = lambda freq: simulate_multichannel_oscillation(2,freq,**sim_args)

    elif test in ['amplitude','amp']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda amp: simulate_multichannel_oscillation(2,freq,**sim_args,
                                                                 amplitude=[amp,amp*damp])

    elif test in ['phase','phi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi: simulate_multichannel_oscillation(2,freq,**sim_args,
                                                                 phase=[phi,phi+dphi])

    elif test in ['n','n_trials']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        del sim_args['n_trials']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n: simulate_multichannel_oscillation(2,freq,**sim_args,n_trials=n)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    # Ensure hand-set values are sorted (ascending), as many tests assume it
    test_values = sorted(test_values,reverse=True) if test in ['synchrony','strength','coupling'] \
                  else sorted(test_values)
    n_values = len(test_values)

    # Set default parameters for each spectral estimation method
    if spec_method == 'multitaper':
        if 'freq_range' not in kwargs:  kwargs['freq_range'] = [1,100]
    elif spec_method == 'bandfilter':
        if 'freqs' not in kwargs:       kwargs['freqs'] = ((2,6),(6,10),(10,22),(22,42),(42,86))

    if 'buffer' not in kwargs: kwargs['buffer'] = 1.0

    for i,value in enumerate(test_values):
        # print("Running test value %d/%d: %.2f" % (i+1,n_values,value))

        # Simulate data with oscillation of given params in additive noise
        #  -> (n_timepts,n_trials,n_chnls=2)
        data = gen_data(value)
        if pair_type == 'spk-lfp':
            data[:,:,0] = _continuous_to_spiking(data[:,:,0])

        # Compute time-frequency/spectrogram representation of data and
        # bivariate measure of synchrony -> (n_freqs,n_timepts)
        if single_trial is None:
            sync,freqs,timepts,phase = synchrony(data[:,:,0], data[:,:,1], axis=1, method=method,
                                                spec_method=spec_method, smp_rate=smp_rate,
                                                time_axis=0, return_phase=True,
                                                single_trial=single_trial, **kwargs)
        else:
            sync,freqs,timepts = synchrony(data[:,:,0], data[:,:,1], axis=1, method=method,
                                           spec_method=spec_method, smp_rate=smp_rate,
                                           time_axis=0, return_phase=False,
                                           single_trial=single_trial, **kwargs)
            print(sync.min(), sync.max())
            sync = sync.mean(axis=-1)

        n_freqs,n_timepts = sync.shape
        if freqs.ndim == 2:
            bands = freqs
            freqs = freqs.mean(axis=1)  # Compute center of freq bands

        # KLUDGE Initialize output arrays on 1st loop, once spectrogram output shape is known
        if i == 0:
            syncs = np.empty((n_freqs,n_timepts,n_values))
            if single_trial is None:
                phases = np.empty((n_freqs,n_timepts,n_values))
            else:
                phases = None

        syncs[:,:,i] = sync
        if single_trial is None: phases[:,:,i] = phase

    # Compute mean (weighted circular mean for phase) across all timepoints
    # -> (n_freqs,n_values) frequency marginal for each tested value
    marginal_syncs = syncs.mean(axis=1)
    if single_trial is None:
        marginal_phases = np.angle(_amp_phase_to_complex(syncs,phases).mean(axis=1))

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
        test_freq_idxs  = np.asarray([np.argmin(np.abs(freq_transform(f) - freqs_transformed))
                                      for f in test_values])

        # Extract synchrony and phase at each tested frequency
        test_freq_syncs = marginal_syncs[test_freq_idxs,np.arange(n_values)]
        if single_trial is None:
            test_freq_phases = marginal_phases[test_freq_idxs,np.arange(n_values)]

    else:
        # Find frequency in spectrogram closest to simulated frequency
        test_freq_idx  = np.argmin(np.abs(freq_transform(freq) - freqs_transformed))

        # Extract synchrony and phase at simulated frequency
        test_freq_syncs = marginal_syncs[test_freq_idx,:]
        if single_trial is None:
            test_freq_phases = marginal_phases[test_freq_idx,:]

    if do_plots:
        dt      = np.diff(timepts).mean()
        tlim    = [timepts[0]-dt/2, timepts[-1]+dt/2]
        df      = np.diff(plot_freqs).mean()
        flim    = [plot_freqs[0]-df/2, plot_freqs[-1]+df/2]
        plot_variables = ['sync','phase'] if single_trial is None else ['sync']

        # # Plot synchrony/phase spectrogram for each tested value
        # n_subplots = [floor(n_values/2), ceil(n_values/floor(n_values/2))]
        # for i_vbl,variable in enumerate(plot_variables):
        #     plot_vals = syncs if variable == 'sync' else phases
        #     cmap = 'viridis' if variable == 'sync' else 'hsv'
        #     plt.figure()
        #     for i,value in enumerate(test_values):
        #         clim = [plot_vals[:,:,i].min(),plot_vals[:,:,i].max()] if variable == 'sync' \
        #                else [-pi,pi]
        #         ax = plt.subplot(n_subplots[0],n_subplots[1],i+1)
        #         plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        #         target_freq = freq_transform(value) if test in ['frequency','freq'] else \
        #                       freq_transform(freq)
        #         plt.plot(tlim, [target_freq,target_freq], '-', color='r', linewidth=0.5)
        #         plt.imshow(plot_vals[:,:,i], extent=[*tlim,*flim], vmin=clim[0], vmax=clim[1],
        #                    aspect='auto', origin='lower', cmap=cmap)
        #         if i in [0,n_subplots[1]]:
        #             plt.yticks(freq_ticks,freq_tick_labels)
        #         else:
        #             ax.set_xticklabels([])
        #             plt.yticks(freq_ticks,[])
        #         plt.title(np.round(value,decimals=2))
        #         plt.colorbar()
        #     plt.show()
        # if plot_dir is not None:
        #   plt.savefig(os.path.join(plot_dir,'synchrony-spectrogram-%s-%s-%s.png'
        #               % (method,test,spec_method)))


        # Plot time-averaged synchrony/phase spectrum for each tested value
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        threshold_syncs = marginal_syncs > 0.1

        for i_vbl,variable in enumerate(plot_variables):
            plt.subplot(1,2,i_vbl+1)
            plot_vals = marginal_syncs if variable == 'sync' else marginal_phases
            ylim = [0,1.05*marginal_syncs.max()] if variable == 'sync' else [-pi,pi]
            for i,value in enumerate(test_values):
                if variable == 'phase':
                    plt.plot(plot_freqs, plot_vals[:,i], '-', color=colors[i],
                             linewidth=1.5, alpha=0.33)
                    plt.plot(plot_freqs[threshold_syncs[:,i]], plot_vals[threshold_syncs[:,i],i],
                             '.-', color=colors[i], linewidth=1.5)
                else:
                    plt.plot(plot_freqs, plot_vals[:,i], '.-', color=colors[i], linewidth=1.5)
                target_freq = freq_transform(value) if test in ['frequency','freq'] else \
                              freq_transform(freq)
                plt.plot([target_freq,target_freq], ylim, '-', color=colors[i], linewidth=0.5)
                plt.text(flim[1]-0.05*np.diff(flim), ylim[1]-(i+1)*0.05*np.diff(ylim),
                         np.round(value,decimals=2), color=colors[i], fontweight='bold',
                         horizontalalignment='right')
            plt.xlim(flim)
            plt.ylim(ylim)
            plt.xticks(freq_ticks,freq_tick_labels)
            plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(variable)
            if i_vbl == 0:
                title = "%s %s %s test" % (spec_method,method,test)
                if single_trial is not None: title += " (%s trials)" % single_trial
                plt.title(title, horizontalalignment='left')
        plt.show()
        if plot_dir is not None:
            filename = 'synchrony-spectrum-%s-%s-%s' % (method,test,spec_method)
            if single_trial is not None: filename += '-'+single_trial
            plt.savefig(os.path.join(plot_dir, filename+'.png'))

        # Plot summary curve of synchrony (or peak frequency) vs tested value
        plt.figure()
        for i_vbl,variable in enumerate(plot_variables):
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
            plt.ylabel('frequency' if test in ['frequency','freq'] and variable == 'sync' else \
                        variable)
            if i_vbl == 0:
                plt.title("%s %s %s test" % (spec_method,method,test), horizontalalignment='left')
        plt.show()
        if plot_dir is not None:
            filename = 'synchrony-summary-%s-%s-%s.png' % (method,test,spec_method)
            if single_trial is not None: filename += '-'+single_trial
            plt.savefig(os.path.join(plot_dir, filename+'.png'))

    ## Determine if test actually produced the expected values
    # 'synchrony' : Test if synchrony strength increases monotonically with simulated synchrony
    if test in ['synchrony','strength','coupling']:
        evals = [((np.diff(test_freq_syncs) > 0).all(),
                    "Estimated sync strength does not increase monotonically with simulated sync")]

    # 'frequency' : check if frequency of peak synchrony matches simulated target frequency
    elif test in ['frequency','freq']:
        evals = [((np.diff(peak_freqs) > 0).all(),
                    "Estimated peak freq does not increase monotonically with expected freq")]

    # 'amplitude','phase','ampratio' : Test if synchrony is ~ same for all values
    elif test in ['amplitude','amp', 'phase','phi', 'ampratio','amp_ratio','damp']:
        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated sync has larger than expected range across tested %s value" % test)]

    # 'relphase' : Test if sync strength is ~ same for all values, phase increases monotonically
    elif test in ['relphase','rel_phase','dphi']:
        circ_subtract = lambda data1,data2: np.angle(np.exp(1j*data1) / np.exp(1j*data2))
        circ_diff = lambda data: circ_subtract(data[1:],data[:-1])

        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated sync has larger than expected range across tested %s value" % test)]
        if single_trial is None:
            evals.append(((circ_diff(test_freq_phases) > 0).all(),
                         "Estimated sync phase does not increase monotonic with sim'd relative phase"))

    # 'n' : Test if synchrony is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        evals = [(test_freq_syncs.ptp() < 0.1,
                    "Estimated sync has > expected range across n's (likely biased by n)")]

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return syncs, phases, passed


def test_sync_info(sync_method, pair_type='lfp-lfp', test='sync', test_values=None,
                   spec_method='wavelet', single_trial_method='pseudo', info_method='pev',
                   do_tests=True, do_plots=False, plot_dir=None, seed=1,
                   phi_sd=pi/2, dphi=0, damp=1, amp=5.0, freq=32, phi=0, noise=0.5,
                   n=50, time_range=3.0, smp_rate=1000, burst_rate=0, **kwargs):
    """
    Basic testing for information contained in single-trial synchrony estimates

    Generates synthetic LFP data using given parameters, with some difference across
    stimulated conditions, computes single-trial (jackknife) time-frequency synchrony estimates
    using given function, computes measures of information on those, and compares to expected.

    Parameters
    ----------
    sync_method : str
        Name of synchrony estimation function to test. Options: 'PPC' | 'PLV' | 'coherence'

    test : str, default: 'frequency'
        Type of test to run. Options:
        
        - 'synchrony' : Tests multiple values of btwn-condition difference in synchrony strength
            (by manipulating phase SD of one signal). Checks for monotonic increase of information.
        - 'relphase' : Tests multiple values of btwn-condition difference in relative phase.
            Checks that information doesn't vary appreciably.
        - 'ampratio' : Test multiple values of btwn-condition difference in relative amplitude.
            Checks that information doesn't vary appreciably.

    test_values : array-like, shape=(n_values,), dtype=str
        List of values to test. Interpretation and defaults are test-specific:
        
        - 'synchrony' : Relative phase SDs (~inverse of synchrony strength). Default: [pi,pi/2,pi/4,pi/8,0]
        - 'relphase' : Relative phases to test. Default: [-pi,-pi/2,0,pi/2,pi]
        - 'ampratio' : Amplitude ratios to test. Default: [1,2,4,8]

    spec_method : str, default: 'wavelet'
        Name of spectral estimation function to use to generate time-frequency representation
        to input into synchrony function. Options: 'wavelet' | 'multitaper' | 'bandfilter'

    single_trial_method : str, default: 'pseudo'
        What type of single-trial synchrony estimator to compute:
        
        - 'pseudo' : single-trial estimates using jackknife pseudovalues
        - 'richter' : single-trial estimates using actual jackknifes as in Richter & Fries 2015

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    - Following args set param's for sim, may be overridden by <test_values> depending on test -
    phi_sd  Scalar. SD (rad) for phase diff of 2 signals if test != 'synchrony'. Default: pi/2
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

    **kwargs :
        All other keyword args passed to synchrony estimation function

    Returns
    -------
    infos : ndarray, shape=(n_freqs,n_timepts,n_values)
        Estimated neural information in synchrony.

    syncs : ndarray, shape=(n_freqs,n_timepts,n_trials,n_values)
        Estimated single-trial synchrony for each tested value.

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    test = test.lower()
    sync_method = sync_method.lower()

    # Set defaults for tested values and set up rate generator function depending on <test>
    sim_args = dict(amplitude=[amp,amp*damp], phase=[phi+dphi,phi], phase_sd=[0,phi_sd],
                    n_trials=n, noise=noise, time_range=time_range, burst_rate=burst_rate,
                    seed=seed)

    def simulate_synchrony_information(test_param, base_value, test_value, freq, **sim_args):
        """ Generate synthetic multichannel oscillatory data with diff params across conds """
        base_args = deepcopy(sim_args)
        test_args = deepcopy(sim_args)

        # Set value for manipulated parameter in 'baseline' and 'test' condition trials
        base_args[test_param] = base_value
        test_args[test_param] = test_value

        # Generate simulated oscillations for baseline and test condition trials,
        # concatenate alog trial axis
        return np.concatenate((simulate_multichannel_oscillation(2,freq,**base_args),
                               simulate_multichannel_oscillation(2,freq,**test_args)),
                              axis=1)

    if test in ['sync','synchrony','strength','coupling']:
        base_value = sim_args['phase_sd']
        test_values = [pi/2, pi/4, pi/8, 0] if test_values is None else test_values
        del sim_args['phase_sd']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi_sd: simulate_synchrony_information('phase_sd', base_value, [0,phi_sd],
                                                                 freq, **sim_args)

    elif test in ['relphase','rel_phase','dphi']:
        base_value = sim_args['phase']
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']   # Delete preset arg so it uses argument to lambda below
        # Note: Implement dphi in 1st channel only so synchrony phase ends up monotonically
        # *increasing* for tests below
        gen_data = lambda dphi: simulate_synchrony_information('phase', base_value, [phi+dphi,phi],
                                                               freq, **sim_args)

    elif test in ['ampratio','amp_ratio','damp']:
        base_value = sim_args['amplitude']
        test_values = [1,2,4,8] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda damp: simulate_synchrony_information('amplitude', base_value, [amp,amp*damp],
                                                               freq, **sim_args)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    labels = np.hstack((np.zeros((n,)),np.ones((n,))))

    # Ensure hand-set values are sorted (ascending), as many tests assume it
    test_values = sorted(test_values,reverse=True) if test in ['synchrony','strength','coupling'] \
                  else sorted(test_values)
    n_values = len(test_values)

    # Set default parameters for each spectral estimation method
    if spec_method == 'multitaper':
        if 'freq_range' not in kwargs:  kwargs['freq_range'] = [1,100]
    elif spec_method == 'bandfilter':
        if 'freqs' not in kwargs:       kwargs['freqs'] = ((2,6),(6,10),(10,22),(22,42),(42,86))

    if ('buffer' not in kwargs) and (spec_method != 'multitaper'): kwargs['buffer'] = 1.0

    for i,value in enumerate(test_values):
        # print("Running test value %d/%d: %.2f" % (i+1,n_values,value))

        # Simulate data with oscillation of given params in additive noise
        #  -> (n_timepts,n_trials,n_chnls=2)
        data = gen_data(value)
        if pair_type == 'spk-lfp':
            data[:,:,0] = _continuous_to_spiking(data[:,:,0])

        # Compute single-trial time-frequency/spectrogram representation of data and
        # bivariate measure of synchrony -> (n_freqs,n_timepts,n_trials)
        sync,freqs,timepts = synchrony(data[:,:,0], data[:,:,1], axis=1, method=sync_method,
                                        spec_method=spec_method, smp_rate=smp_rate,
                                        time_axis=0, return_phase=False,
                                        single_trial=single_trial_method, keepdims=False,
                                        **kwargs)

        n_freqs,n_timepts,_ = sync.shape
        if freqs.ndim == 2:
            bands = freqs
            freqs = freqs.mean(axis=1)  # Compute center of freq bands

        # Compute neural information about condition in synchrony
        info = neural_info(labels, sync, axis=-1, method=info_method)

        # KLUDGE Initialize output arrays on 1st loop, once spectrogram output shape is known
        if i == 0:
            syncs = np.empty((n_freqs,n_timepts,n*2,n_values))
            infos = np.empty((n_freqs,n_timepts,n_values))

        syncs[:,:,:,i] = sync
        infos[:,:,i] = info.squeeze(axis=-1)

    # Take marginal means across time -> (freq, [trials], values)
    marginal_syncs = syncs.mean(axis=1)
    marginal_infos = infos.mean(axis=1)

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

    # Find frequency in spectrogram closest to simulated frequency
    test_freq_idx   = np.argmin(np.abs(freq_transform(freq) - freqs_transformed))
    # Extract synchrony and phase at simulated frequency
    test_freq_infos = marginal_infos[test_freq_idx,:]


    if do_plots:
        dt      = np.diff(timepts).mean()
        tlim    = [timepts[0]-dt/2, timepts[-1]+dt/2]
        df      = np.diff(plot_freqs).mean()
        flim    = [plot_freqs[0]-df/2, plot_freqs[-1]+df/2]

        # Plot time-averaged synchrony spectrum and neural information for each tested value
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        # Plot synchrony spectrum (at tested freq) as a function of trials for each test value
        plt.subplot(1,2,1)
        for i,value in enumerate(test_values):
            plt.plot(np.arange(n*2)+1, marginal_syncs[test_freq_idx,:,i], '.-',
                     color=colors[i], linewidth=1.5)
        plt.xlim((1,n*2))
        plt.xlabel('Trials')
        plt.ylabel(single_trial_method + ' ' + sync_method)

        # Plot neural information spectrum for each test value
        plt.subplot(1,2,2)
        ylim = (marginal_infos.min(), marginal_infos.max())
        ylim = (min(0, ylim[0]-0.05*(ylim[1]-ylim[0])), ylim[1]+0.05*(ylim[1]-ylim[0]))
        for i,value in enumerate(test_values):
            plt.plot(plot_freqs, marginal_infos[:,i], '.-', color=colors[i], linewidth=1.5)
            target_freq = freq_transform(value) if test in ['frequency','freq'] else \
                          freq_transform(freq)
            plt.plot([target_freq,target_freq], ylim, '-', color=colors[i], linewidth=0.5)
            plt.text(flim[1]-0.05*np.diff(flim), ylim[1]-(i+1)*0.05*np.diff(ylim),
                        np.round(value,decimals=2), color=colors[i], fontweight='bold',
                        horizontalalignment='right')
        plt.xlim(flim)
        plt.ylim(ylim)
        plt.xticks(freq_ticks,freq_tick_labels)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(info_method)
        title = "%s %s %s %s test" % (spec_method,sync_method,single_trial_method,test)
        plt.title(title, horizontalalignment='left')
        plt.show()
        if plot_dir is not None:
            filename = 'info-spectrum-%s-%s-%s-%s' % (sync_method,test,spec_method,single_trial_method)
            plt.savefig(os.path.join(plot_dir, filename+'.png'))


    ## Determine if test actually produced the expected values
    # 'synchrony' : Test if information increases monotonically with simulated cond synchrony diff
    if test in ['sync','synchrony','strength','coupling']:
        evals = [((np.diff(test_freq_infos) > 0).all(),
                    "Estimated information does not increase monotonically with simulated sync diff")]

    # 'relphase','ampratio' : Test if information is ~ same for all values
    elif test in ['relphase','rel_phase','dphi', 'ampratio','amp_ratio','damp']:
        evals = [(test_freq_infos.ptp() < 0.1,
                    "Estimated information has larger than expected range across tested %s value" % test)]

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return infos, syncs, passed


# =============================================================================
# Functions that loop thru an entire battery of tests
# =============================================================================
def synchrony_test_battery(methods=('PPC','PLV','coherence'),
                           tests=('synchrony','relphase','ampratio','frequency','amplitude','phase','n'),
                           spec_methods=('wavelet','multitaper','bandfilter'),
                           trial_methods=(None,'richter','pseudo'),
                           do_tests=True, **kwargs):
    """
    Run a battery of given tests on given oscillatory synchrony computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('PPC','PLV','coherence') (all supported methods)
        List of synchrony computation methods to test.
                
    tests : array-like of str, default: ('synchrony','relphase','ampratio','frequency','amplitude','phase','n')
        List of tests to run. Note: certain combinations of methods,tests are skipped, 
        as they are not expected to pass (ie 'ampratio' tests skipped for coherence method)
                
    spec_methods : array-like of str, default: ('wavelet','multitaper','bandfilter')
        List of underlying spectral analysis methods to test.                

    trial_methods : array-like of str or None, default: (None,'richter','pseudo')
        List of single-trial estimate methods to test (set=None to run
        usual trial-reduced synchrony methods instead of single-trial).                

    do_tests : bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True

    **kwargs :
        All other keyword args passed to synchrony estimation function
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    if isinstance(spec_methods,str): spec_methods = [spec_methods]
    if isinstance(trial_methods,str): trial_methods = [trial_methods]
    tests = [test.lower() for test in tests]
    methods = [method.lower() for method in methods]

    for test in tests:
        for method in methods:
            for spec_method in spec_methods:
                for trial_method in trial_methods:
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
                                                spec_method=spec_method, single_trial=trial_method,
                                                do_tests=do_tests_, **kwargs)

                    print('%s (test ran in %.1f s)'
                        % ('PASSED' if passed else 'FAILED', time.time()-t1))

                    # If saving plots to file, let's not leave them all open
                    if 'plot_dir' in kwargs: plt.close('all')


def spike_field_test_battery(methods=('PPC','PLV','coherence'),
                           tests=('synchrony','relphase','ampratio','frequency','amplitude','phase','n'),
                           spec_methods=('wavelet','multitaper','bandfilter'),
                           do_tests=True, **kwargs):
    """
    Run a battery of given tests on given oscillatory spike-field coupling computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('PPC','PLV','coherence') (all supported methods)
        List of synchrony computation methods to test.
                
    tests : array-like of str, default: ('synchrony','relphase','frequency','amplitude','phase','n')
        List of tests to run. Note: certain combinations of methods,tests are skipped, 
        as they are not expected to pass (ie 'ampratio' tests skipped for coherence method)
                
    spec_methods : array-like of str, default: ('wavelet','multitaper','bandfilter')
        List of underlying spectral analysis methods to test.                

    do_tests : bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True

    **kwargs :
        All other keyword args passed to test_synchrony() function
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

                _,_,passed = test_synchrony(method, pair_type='spk-lfp', test=test,
                                            spec_method=spec_method, do_tests=do_tests_, **kwargs)

                print('%s (test ran in %.1f s)'
                      % ('PASSED' if passed else 'FAILED', time.time()-t1))

                # If saving plots to file, let's not leave them all open
                if 'plot_dir' in kwargs: plt.close('all')


def sync_info_test_battery(methods=('PPC','PLV','coherence'),
                           tests=('synchrony','relphase','ampratio'),
                           spec_methods=('wavelet','multitaper','bandfilter'),
                           trial_methods=('richter','pseudo'),
                           do_tests=True, **kwargs):
    """
    Run a battery of given tests on given oscillatory synchrony computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('PPC','PLV','coherence') (all supported methods)
        List of synchrony computation methods to test.
                
    tests : array-like of str, default: ('synchrony','relphase','ampratio')
        List of tests to run. Note: certain combinations of methods,tests are skipped, 
        as they are not expected to pass (ie 'ampratio' tests skipped for coherence method)
                
    spec_methods : array-like of str, default: ('wavelet','multitaper','bandfilter')
        List of underlying spectral analysis methods to test.                

    trial_methods : array-like of str or None, default: (None,'richter','pseudo')
        List of single-trial estimate methods to test (set=None to run
        usual trial-reduced synchrony methods instead of single-trial).                

    do_tests : bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True

    **kwargs :
        All other keyword args passed to synchrony estimation function
            
    **kwargs :
        Any other keyword passed directly to test_sync_info()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    if isinstance(spec_methods,str): spec_methods = [spec_methods]
    if isinstance(trial_methods,str): trial_methods = [trial_methods]
    tests = [test.lower() for test in tests]
    methods = [method.lower() for method in methods]

    for test in tests:
        for method in methods:
            for spec_method in spec_methods:
                for trial_method in trial_methods:
                    t1 = time.time()
                    print("Running %s test on %s %s" % (test,spec_method,method))

                    _,_,passed = test_sync_info(method, pair_type='lfp-lfp', test=test,
                                                spec_method=spec_method,
                                                single_trial_method=trial_method,
                                                do_tests=do_tests, **kwargs)

                    print('%s (test ran in %.1f s)'
                        % ('PASSED' if passed else 'FAILED', time.time()-t1))

                    # If saving plots to file, let's not leave them all open
                    if 'plot_dir' in kwargs: plt.close('all')


# =============================================================================
# Helper functions
# =============================================================================
def _amp_phase_to_complex(amp,theta):
    """ Convert amplitude and phase angle to complex variable """
    return amp * np.exp(1j*theta)


def _continuous_to_spiking(data):
    """ Convert continuous data to thresholded spiking response """
    # Convert continuous oscillation to probability (range 0-1)
    data = (data - data.min()) / data.ptp()
    data = data**2  # Sparsen high rates some

    # Use probabilities to generate Bernoulli random variable at each time point
    return bernoulli.ppf(0.5, data).astype(bool)

"""
Suite of tests to assess "face validity" of spectral analysis functions in spectra.py
Usually used to test new or majorly updated functions.

Includes tests that parametrically estimate power as a function of frequency, amplitude, phase,
n, etc. to establish methods produce expected pattern of results.

Plots results and runs assertions that basic expected results are reproduced

Functions
---------
- test_power :            Contains tests of spectral estimation functions
- power_test_battery :    Runs standard battery of tests of spectral estimation functions
- itpc_test_battery :     Runs standard battery of tests of ITPC estimation functions
"""
import os
import time
from warnings import warn
from math import pi, sqrt, ceil, floor, log2
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import bernoulli

from spynal.utils import set_random_seed, iarange
from spynal.spectra.spectra import spectrum, power_spectrogram, itpc
from spynal.spectra.utils import fft, ifft, simulate_oscillation
from spynal.plots import plot_line_with_error_fill


def test_power(method, test='frequency', test_values=None, spec_type='power', fft_method=None,
               do_tests=True, do_plots=False, plot_dir=None, seed=1,
               amp=5.0, freq=32, phi=0, phi_sd=0, noise=0.5, n=1000, burst_rate=0,
               time_range=3.0, smp_rate=1000, spikes=False, **kwargs):
    """
    Basic testing for functions estimating time-frequency spectral power

    Generate synthetic LFP data using given network simulation,
    estimate spectrogram using given function, and compares estimated to expected.

    For test failures, raises an error or warning (depending on value of `do_tests`).
    Optionally plots summary of test results.

    Parameters
    ----------
    method : str
        Name of time-frequency spectral estimation function to test. Options:
        'wavelet' | 'multitaper' | 'bandfilter' | 'burst'

    test : str, default: 'frequency'
        Type of test to run. Options:

        - 'frequency' : Tests multiple simulated oscillatory frequencies
            Checks power for monotonic increase of peak freq
        - 'amplitude' : Tests multiple simulated amplitudes at same freq
            Checks power for monotonic increase of amplitude
        - 'phase_sd': Tests multiple simulated phase std dev's (ie evoked vs induced)
            Checks that power doesn't greatly vary with phase SD.
            Checks that ITPC decreases monotonically with phase SD
        - 'n' : Tests multiple values of number of trials (n)
            Checks that power doesn't greatly vary with n.
        - 'burst_rate' : Checks that oscillatory burst rate increases
            as it's increased in simulated data.

    test_values : array-like, shape=(n_values,), dtype=str
        List of values to test. Interpretation and defaults are test-specific:

        - 'frequency' : List of frequencies to test. Default: [4,8,16,32,64]
        - 'amplitude' : List of oscillation amplitudes to test. Default: [1,2,5,10,20]
        - 'n' :         Trial numbers. Default: [25,50,100,200,400,800]

    spec_type : str, default: 'power'
        Type of spectral signal to return. Options: 'power' | 'itpc' (intertrial phase clustering)

    fft_method : str, default: 'torch' (if available)
        Which underlying FFT implementation to use. Options: 'torch', 'fftw', 'numpy'

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail

    do_plots : bool, default: False
        Set=True to plot test results

    plot_dir : str, default: None (don't save to file)
        Full-path directory to save plots to. Set=None to not save plots.

    seed : int, default: 1 (reproducible random numbers)
        Random generator seed for repeatable results. Set=None for fully random numbers.

    - Following args set param's for sim, may be overridden by <test_values> depending on test -
    amp     Scalar. Simulated oscillation amplitude (a.u.) if test != 'amplitude'. Default: 5.0
    freq    Scalar. Simulated oscillation frequency (Hz) if test != 'frequency'. Default: 32
    phi     Scalar. Simulated oscillation (mean) phase (rad). Default: 0
    phi_sd  Scalar. Simulated oscillation phase std dev (rad). Default: 0
    noise   Scalar. Additive noise for simulated signal (a.u., same as amp). Default: 0.5
    n       Int. Number of trials to simulate if test != 'n'. Default: 1000
    burst_rate Scalar. Oscillatory burst rate (bursts/trial). Default: 0 (non-bursty)
    time_range Scalar. Full time range to simulate oscillation over (s). Default: 1.0
    smp_rate Int. Sampling rate for simulated data (Hz). Default: 1000

    **kwargs :
        All other keyword args passed to spectral estimation function

    Returns
    -------
    means : ndarray, shape=(n_freqs,n_timepts,n_values)
        Estimated mean spectrogram for each tested value.

    sems : ndarray, shape=(n_freqs,n_timepts,n_values)
        SEM of mean spectrogram for each tested value.

    passed : bool
        True if all tests produce expected values; otherwise False.
    """
    method = method.lower()
    test = test.lower()
    spec_type = spec_type.lower()
    fft_method = fft_method.lower()

    # Set defaults for tested values and set up rate generator function depending on <test>
    sim_args = dict(amplitude=amp, phase=phi, phase_sd=phi_sd,
                    n_trials=n, noise=noise, time_range=time_range, burst_rate=burst_rate,
                    seed=seed)

    if test in ['frequency','freq']:
        test_values = [4,8,16,32,64] if test_values is None else test_values
        gen_data = lambda freq: simulate_oscillation(freq,**sim_args)

    elif test in ['amplitude','amp']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['amplitude']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda amp: simulate_oscillation(freq,**sim_args,amplitude=amp)

    elif test in ['phase','phi']:
        test_values = [-pi,-pi/2,0,pi/2,pi] if test_values is None else test_values
        del sim_args['phase']       # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi: simulate_oscillation(freq,**sim_args,phase=phi)

    elif test in ['phase_sd','phi_sd']:
        test_values = [pi, pi/2, pi/4, 0] if test_values is None else test_values
        del sim_args['phase_sd']   # Delete preset arg so it uses argument to lambda below
        gen_data = lambda phi_sd: simulate_oscillation(freq,**sim_args,phase_sd=phi_sd)

    elif test in ['n','n_trials']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        del sim_args['n_trials']    # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n: simulate_oscillation(freq,**sim_args,n_trials=n)

    elif test in ['burst_rate','burst']:
        test_values = [0.1,0.2,0.4,0.8] if test_values is None else test_values
        del sim_args['burst_rate']  # Delete preset arg so it uses argument to lambda below
        gen_data = lambda rate: simulate_oscillation(freq,**sim_args,burst_rate=rate)

    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)

    # Ensure hand-set values are sorted (ascending), as many tests assume it
    test_values = sorted(test_values)
    n_values = len(test_values)

    # Set default parameters for each spectral estimation method
    do_burst = method in ['burst','burst_analysis']
    do_itpc = spec_type == 'itpc'

    # Special case: oscillatory burst analysis
    if do_burst:
        # KLUDGE  Reset spectral analysis <method> to 'wavelet' (unless set explicitly in kwargs)
        if 'bands' not in kwargs:       kwargs['bands'] = ((2,6),(6,10),(10,22),(22,42),(42,86))

    elif method == 'multitaper':
        if 'freq_range' not in kwargs:  kwargs['freq_range'] = [1,100]

    elif method == 'bandfilter':
        if 'freqs' not in kwargs:       kwargs['freqs'] = ((2,6),(6,10),(10,22),(22,42),(42,86))

    if ('buffer' not in kwargs) and (method != 'multitaper'): kwargs['buffer'] = 1.0

    if do_itpc:
        if 'itpc_method' not in kwargs: kwargs['itpc_method'] = 'PLV'
        kwargs['trial_axis'] = 1

    spec_fun = itpc if do_itpc else power_spectrogram

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

        spec,freqs,timepts = spec_fun(data, smp_rate, axis=0, method=method, fft_method=fft_method,
                                      **kwargs)
        if freqs.ndim == 2:
            bands = freqs
            freqs = freqs.mean(axis=1)  # Compute center of freq bands
        if do_itpc: n_freqs,n_timepts = spec.shape
        else:       n_freqs,n_timepts,n_trials = spec.shape

        # KLUDGE Initialize output arrays on 1st loop, once spectrogram output shape is known
        if i == 0:
            means = np.empty((n_freqs,n_timepts,n_values))
            sems = np.empty((n_freqs,n_timepts,n_values))

        # Compute across-trial mean and SEM of time-frequency data -> (n_freqs,n_timepts,n_values)
        if not do_itpc:
            means[:,:,i] = spec.mean(axis=2)
            sems[:,:,i]  = spec.std(axis=2,ddof=0) / sqrt(n_trials)
        # HACK ITPC by definition already reduced across trials, so just copy results into "means"
        else:
            means[:,:,i] = spec
            sems[:,:,i]  = 0

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
        peak_freqs = freqs[idxs] if not (do_burst or (method == 'bandfilter')) else idxs

        # Find frequency in spectrogram closest to each simulated frequency
        test_freq_idxs  = np.asarray([np.argmin(np.abs(freq_transform(f) - freqs_transformed))
                                      for f in test_values])

        # Extract mean,SEM of power at each tested frequency
        test_freq_means = marginal_means[test_freq_idxs,np.arange(n_values)]
        test_freq_errs  = marginal_sems[test_freq_idxs,np.arange(n_values)]

    else:
        # Find frequency in spectrogram closest to simulated frequency
        test_freq_idx   = np.argmin(np.abs(freq_transform(freq) - freqs_transformed))

        # Extract mean,SEM of power at tested frequency
        test_freq_means = marginal_means[test_freq_idx,:]
        test_freq_errs  = marginal_sems[test_freq_idx,:]

    # Plot summary of test results
    if do_plots:
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
        #     target_freq = freq_transform(value) if test in ['frequency','freq'] else \
        #                   freq_transform(freq)
        #     if not (do_burst or (method == 'bandfilter')):
        #         plt.plot(tlim, [target_freq,target_freq], '-', color='r', linewidth=0.5)
        #     plt.imshow(means[:,:,i], extent=[*tlim,*flim], aspect='auto', origin='lower')
        #     if i in [0,n_subplots[1]]:
        #         plt.yticks(freq_ticks,freq_tick_labels)
        #     else:
        #         ax.set_xticklabels([])
        #         plt.yticks(freq_ticks,[])
        #     plt.title(np.round(value,decimals=2))
        #     plt.colorbar()
        # plt.show()
        # if plot_dir is not None:
        #     filename = 'power-spectrogram-%s-%s-%s.png' % (kwargs['itpc_method'],method,test) \
        #                 if do_itpc else 'power-spectrogram-%s-%s.png' % (method,test)
        #     plt.savefig(os.path.join(plot_dir,filename))

        # Plot time-averaged spectrum for each tested value
        plt.figure()
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ylim = [0,1.05*marginal_means.max()]
        for i,value in enumerate(test_values):
            plt.plot(plot_freqs, marginal_means[:,i], '.-', color=colors[i], linewidth=1.5)
            target_freq = freq_transform(value) if test in ['frequency','freq'] else \
                          freq_transform(freq)
            if not (do_burst or (method == 'bandfilter')):
                plt.plot([target_freq,target_freq], ylim, '-', color=colors[i], linewidth=0.5)
            plt.text(0.9*flim[1], (0.95-i*0.05)*ylim[1], value, color=colors[i], fontweight='bold')
        plt.xlim(flim)
        plt.ylim(ylim)
        plt.xticks(freq_ticks,freq_tick_labels)
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(spec_type)
        plt.title("%s %s test" % (method,test))
        plt.show()
        if plot_dir is not None:
            filename = 'power-spectrum-%s-%s-%s-%s.png' % (kwargs['itpc_method'],method,fft_method,test) \
                        if do_itpc else 'power-spectrum-%s-%s-%s.png' % (method,fft_method,test)
            plt.savefig(os.path.join(plot_dir,filename))

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
                plt.plot([freq_transform(f) for f in test_values],
                         [freq_transform(f) for f in peak_freqs], marker='o')
            plt.xticks(freq_ticks,freq_tick_labels)
            plt.yticks(freq_ticks,freq_tick_labels)
            plt.xlim(lim)
            plt.ylim(lim)
            ax.set_aspect('equal', 'box')
        else:
            plt.errorbar(test_values, test_freq_means, 3*test_freq_errs, marker='o')
        plt.xlabel(test)
        plt.ylabel('frequency' if test in ['frequency','freq'] else spec_type)
        plt.title("%s %s test" % (method,test))
        plt.show()
        if plot_dir is not None:
            filename = 'power-summary-%s-%s-%s-%s.png' % (kwargs['itpc_method'],method,fft_method,test) \
                        if do_itpc else 'power-summary-%s-%s-%s.png' % (method,fft_method,test)
            plt.savefig(os.path.join(plot_dir,filename))

    ## Determine if test actually produced the expected values
    # frequency test: check if frequency of peak power matches simulated target frequency
    if test in ['frequency','freq']:
        evals = [((np.diff(peak_freqs) >= 0).all(),
                  "Estimated peak freq does not increase monotonically with expected freq")]

    # 'amplitude' : Test if power increases monotonically with simulated amplitude
    elif test in ['amplitude','amp']:
        if spec_type == 'power':
            evals = [((np.diff(test_freq_means) > 0).all(),
                      "Estimated power doesn't increase monotonically with simulated amplitude")]
        else:
            evals = {}

    # 'phase' : Test if power is ~ constant across phase
    elif test in ['phase','phi']:
        crit = 0.2 if do_itpc else test_freq_errs.max()
        evals = [(test_freq_means.ptp() < crit,
                  "Estimated %s has larger than expected range across different simulated phases"
                  % spec_type)]

    # 'phase_sd' : Test if power is ~ constant across phase SD;
    #              Test if ITPC decreases monotonically with it
    elif test in ['phase_sd','phi_sd']:
        if do_itpc:
            evals = [((np.diff(test_freq_means) < 0).all(),
                      "Estimated ITPC does not decrease monotonically with simulated phase SD")]
        else:
            evals = [(test_freq_means.ptp() < test_freq_errs.max(),
                      "Estimated %s has larger than expected range across simulated phase SDs"
                      % spec_type)]

    # 'n' : Test if power is ~ same for all values of n (unbiased by n)
    elif test in ['n','n_trials']:
        crit = 0.2 if do_itpc else test_freq_errs.max()
        evals = [(test_freq_means.ptp() < crit,
                  "Estimated %s has larger than expected range across n's (likely biased by n)"
                  % spec_type)]

    # 'burst_rate': Test if measured burst rate increases monotonically with simulated burst rate
    elif test in ['burst_rate','burst']:
        evals = [((np.diff(test_freq_means) > 0).all(),
                  "Estimated burst rate does not increase monotonic with simulated burst rate")]

    passed = True
    for cond,message in evals:
        if not cond:    passed = False

        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  warn(message)

    return means, sems, passed


def power_test_battery(methods=('wavelet','multitaper','bandfilter'),
                       fft_methods=('torch','fftw','numpy'),
                       tests=('frequency','amplitude','phase','phase_sd','n','burst_rate'),                       
                       do_tests=True, **kwargs):
    """
    Run a battery of given tests on given oscillatory power computation methods

    Parameters
    ----------
    methods : array-like of str, default: ('wavelet','multitaper','bandfilter') (all supported)
        List of power computation methods to test.

    fft_methods : array-like of str, default: ('torch','fftw','numpy') (all supported)
        List of underlying FFT implementations to test

    tests : array-like of str, default: ('frequency','amplitude','phase','phase_sd','n','burst_rate')
        List of tests to run.

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail.

    **kwargs :
        Any other keyword args passed directly to test_power()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]

    for test in tests:
        for method in methods:
            for fft_method in fft_methods:
                # No FFTs used in bandfilter method, so no need to test them
                if (method == 'bandfilter') and not (fft_method == 'numpy'): continue

                print("Running %s test on %s spectral analysis (%s FFT method)" % (test,method,fft_method))
                extra_args = kwargs
                if (method in ['burst','burst_analysis']) and ('burst_rate' not in kwargs):
                    extra_args['burst_rate'] = 0.4

                t1 = time.time()

                _,_,passed = test_power(method, test=test, do_tests=do_tests, fft_method=fft_method,
                                        **extra_args)
                print('%s (test ran in %.1f s)' % ('PASSED' if passed else 'FAILED', time.time()-t1))

                # If saving plots to file, let's not leave them all open
                if 'plot_dir' in kwargs: plt.close('all')


def itpc_test_battery(methods=('wavelet','multitaper','bandfilter'),
                      fft_methods=('torch','fftw','numpy'),
                      tests=('frequency','amplitude','phase','phase_sd','n'),
                      itpc_methods=('PLV','Z','PPC'), do_tests=True, **kwargs):
    """
    Run a battery of given tests on given intertrial phase clustering computation methods

    Parameters
    ----------
    methods : array-like, default: ('wavelet','multitaper','bandfilter') (all supported methods)
        List of power computation methods to test.

    fft_methods : array-like of str, default: ('torch','fftw','numpy') (all supported)
        List of underlying FFT implementations to test

    tests : array-like, default: ('frequency','amplitude','phase','phase_sd','n') (all supported)
        List of tests to run.

    itpc_method : array-like, default: ('PLV','Z','PPC') (all supported options)
        List of methods to use for computing intertrial phase clustering

    do_tests : bool, default: True
        Set=True to evaluate test results against expected values and raise an error if they fail.

    **kwargs :
        Any other keyword args passed directly to test_power()
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]

    # Default phase SD = 90 deg unless set otherwise
    phi_sd = kwargs.pop('phi_sd',pi/4)

    for test in tests:
        for itpc_method in itpc_methods:
            for method in methods:
                for fft_method in fft_methods:
                    # No FFTs used in bandfilter method, so no need to test them
                    if (method == 'bandfilter') and not (fft_method == 'numpy'): continue

                    print("Running %s test on %s %s" % (test,method,itpc_method))
                    extra_args = kwargs
                    t1 = time.time()

                    _,_,passed = test_power(method, test=test, itpc_method=itpc_method,
                                            spec_type='itpc', fft_method=fft_method, phi_sd=phi_sd,
                                            do_tests=do_tests, **extra_args)
                    print('%s (test ran in %.1f s)'
                          % ('PASSED' if passed else 'FAILED', time.time()-t1))

                    # If saving plots to file, let's not leave them all open
                    if 'plot_dir' in kwargs: plt.close('all')


def test_fft_time(spec_method, fft_methods=('torch','fftw','scipy','numpy'),
                  n_ffts=tuple(2**iarange(10,15)), n_chnls=(1,10,100),
                  n_trials=100, n_reps=5, seed=1):
    """ Time testing for different FFT methods """
    if seed is not None: set_random_seed(seed)

    n_ffts = np.asarray(n_ffts).astype(int)
    n_chnls = np.asarray(n_chnls).astype(int)

    run_times = xr.DataArray(np.empty((len(n_ffts),len(n_chnls),len(fft_methods),n_reps)),
                             dims=['n_fft','n_chnl','method','rep'],
                             coords={'n_fft':list(n_ffts),
                                     'n_chnl':list(n_chnls),
                                     'method':list(fft_methods),
                                     'rep':list(range(n_reps))})
    fft_methods = np.asarray(fft_methods)

    for n_fft in n_ffts:
        for n_chnl in n_chnls:
            for rep in range(n_reps):

                # Reorder methods each loop to avoid any weird order effects
                for fft_method in fft_methods[np.random.permutation(len(fft_methods))]:
                    print(fft_method, n_fft, n_chnl, rep)
                    # Regenerate new data each loop to avoid and weird caching effects
                    set_random_seed(rep)
                    data = np.random.rand(n_fft,n_chnl,n_trials)

                    start_time = time.time()
                    if spec_method == 'fft':
                        fft(data, n_fft=n_fft, axis=0, fft_method=fft_method)
                    elif spec_method == 'fft/ifft':
                        spec = fft(data, n_fft=n_fft, axis=0, fft_method=fft_method)
                        _ = ifft(spec, n_fft=n_fft, axis=0, fft_method=fft_method)
                    else:
                        spectrum(data, 1000, axis=0, method=spec_method, fft_method=fft_method)
                    dt = time.time() - start_time
                    run_times.loc[n_fft,n_chnl,fft_method,rep] = dt

    means = run_times.mean(dim='rep')
    sds = run_times.std(dim='rep')

    fig = plt.figure()
    for j,n_chnl in enumerate(n_chnls):
        plt.subplot(len(n_chnls),1,j+1)
        plot_line_with_error_fill(n_ffts, means.sel(n_chnl=n_chnl).values.T,
                                  sds.sel(n_chnl=n_chnl).values.T)
        plt.legend(fft_methods)
        plt.title("%d channels" % n_chnl)
        plt.grid(axis='y',color=[0.75,0.75,0.75],linestyle=':')
        if n_chnl == n_chnls[-1]:
            plt.xlabel("Number of samples")
            plt.ylabel("Time per call (s)")
            plt.xticks(n_ffts)
        else:
            plt.xticks(n_ffts,[])
    fig.suptitle(spec_method)

"""
validity_test_randomization_stats.py

Suite of tests to assess "face validity" of randomization statistic functions in randomization_stats.py
Usually used to test new or majorly updated functions to ensure they perform as expected.

Includes tests that parametrically estimate statistics as a function of difference in distribution
means, assays of bias, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_randomization_stats    Contains tests of randomization statistic computation functions
stat_test_battery           Runs standard battery of tests of randomization stat functions
"""
# TODO  Update test function (or write new one?) for 2-way tests

import os
import time
from warnings import warn
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, poisson

from randomization_stats import one_sample_test, paired_sample_test, two_sample_test, \
                                one_way_test, two_way_test


def simulate_data(distribution='normal', mean=None, spread=1, n=100, seed=None):
    """
    Simulates random data with given distribution and parameters
    
    data = simulate_data(distribution='normal', mean=None, spread=1, n=100, seed=None)
    
    ARGS
    distribution    String. Name of distribution to simulate data from. 
                    Options: 'normal' [default] | 'poisson'
                    
    mean            Scalar. Mean/rate parameter for distribution.
                    Default: 0 if distribution='normal' else 10
    
    spread          Scalar. Spread parameter for distribution.
                    SD for normal; unused for Poisson
                    Default: 1.0
                    
    n               Int | Tuple. Number/shape of random variable draws to simulate.
                    Can set either single integer value or a tuple for shape of RV array.
                    Default: 100
                              
    seed            Int. Random generator seed for repeatable results.
                    Set=None [default] for unseeded random numbers.
                                                  
    RETURNS
    data            (n,) | (*n) ndarray. Simulated random data.
                    Returns as 1D array if n is an int.
                    Returns with shape given by n if it is a tuple.
    """
    if seed is not None: np.random.seed(seed)
    
    if mean is None:
        if distribution in ['normal','norm','gaussian','gauss']:    mean = 0.0
        else:                                                       mean = 10

    dist_func = _distribution_name_to_func(distribution)
    
    return dist_func(n, mean, spread)
    
    
def simulate_dataset(gain=5.0, offset=5.0, n_conds=2, n=100, distribution='normal',
                     spreads=1.0, seed=None):
    """
    Simulates random data across multiple conditions/groups with given condition effect size,
    distribution and parameters

    data,labels = simulate_dataset(gain=5.0,offset=5.0,n_conds=2,n=100,
                                   distribution='normal',seed=None)

    ARGS
    gain    Scalar | (n_conds,) array-like. Sets the effect size (difference 
            in mean value between each condition).
            If scalar, it's the difference in mean between each successive condition.
            If vector, it's the difference in mean from baseline for each individual condition.
            Set = 0 to simulate no expected difference between conditions.
            Default: 5.0 
            
    offset  Scalar. Additive baseline value added to any condition effects. Default: 5.0

    n_conds Int. Number of distinct conditions/groups to simulate. Default: 2

    n       Int. Number of trials/observations to simulate *per condition*. Default: 100

    distribution    String. Name of distribution to simulation from. 
                    Options: 'normal' [default] | 'poisson'
           
    spreads Scalar | (n_conds,) array-like. Spread of each condition's data distribution.
            (Gaussian SD; not used for other distributions).
            If scalar, the same spread is used for all conditions.
            If vector, one spread value should be given for each condition.
            
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    RETURNS
    data    (n*n_conds,). Simulated data for multiple repetitions of one/more conditions.

    labels  (n*n_conds,) of int. Condition/group labels for each trial.
            Sorted in group order to simplify visualization.
    """
    # TODO Add ability to simulate independent data series, different n for each cond, 
    if seed is not None: np.random.seed(seed)
        
    # For single-condition data, treat gain as scalar increase over baseline response
    if n_conds == 1:
       gains = np.asarray([gain])
       spreads = np.asarray([spreads])
       
    else:     
        # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
        # Convert to vector of gains for each condition
        gains = gain*np.arange(n_conds) if np.isscalar(gain) else gain

        # Convert scalar spread value to vector for each condition
        if np.isscalar(spreads): spreads = spreads*np.ones((n_conds,))
            
    assert len(gains) == n_conds, \
        ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                    % (len(gains), n_conds))

    assert len(spreads) == n_conds, \
        ValueError("Vector-valued <spreads> must have length == n_conds (%d != %d)" \
                    % (len(spreads), n_conds))

    # Final mean value = baseline + condition-specific gain
    means = offset + gains

    # Generate data for each condition and stack together -> (n_trials*n_conds,) array
    data = np.hstack([simulate_data(distribution=distribution, mean=mean, spread=spread,
                                    n=n, seed=None)
                      for mean,spread in zip(means,spreads)])

    # Create (n_trials_total,) vector of group labels (ints in set 0-n_conds-1)
    labels = np.hstack([i_cond*np.ones((n,)) for i_cond in range(n_conds)])

    return data, labels


def test_randomization_stats(stat, method, test='gain', test_values=None, distribution='normal', n_reps=100,
                             alpha=0.05, seed=None, plot=False, plot_dir=None, **kwargs):
    """
    Basic testing for randomization statistic computation functions
    
    Generates synthetic data, computes statistics, p values, and significance using given method,
    and compares computed to expected values.
    
    means,sds = test_randomization_stats(stat,method,test='gain',test_values=None,distribution='normal',
                                         n_reps=100,alpha=0.05,seed=None,n_reps=100,
                                         plot=False,plot_dir=None, **kwargs)
                              
    ARGS
    stat    String. Type of statistical test to evaluate:
            'one_sample' | 'paired_sample' | 'two_sample' | 'one_way' | 'two_way'
            
    method  String. Resampling paradigm to use for test: 'permutation' | 'bootstrap'
            
    test    String. Type of test to run. Default: 'gain'. Options:
            'gain'  Tests multiple values for between-condition response difference (gain)
                    Checks for monotonically increasing stat/decreasing p value
            'spread'Tests multiple values for distribution spread (SD)
                    Checks for monotonically decreasing stat/increasing p value
            'n'     Tests multiple values of number of trials (n)
                    Checks that stat doesn't vary, but p value decreases, with n                
            'bias'  Tests multiple n values with 0 btwn-cond difference
                    Checks that stat doesn't vary and p value remains ~ 0 (unbiased)

    test_values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'gain'      Btwn-condition response differences (gains). Default: [1,2,5,10,20]
            'spread'    Gaussian SDs for each response distribution. Default: [1,2,5,10,20]
            'n'/'bias'  Trial numbers. Default: [25,50,100,200,400,800]
            
    distribution    String. Name of distribution to simulate data from. 
                    Options: 'normal' [default] | 'poisson'
                                
    n_reps  Int. Number of independent repetitions of tests to run. Default: 100
            
    alpha   Float. Significance criterion "alpha". Default: 0.05
                
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    plot    Bool. Set=True to plot test results. Default: False
    
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
        
    **kwargs All other keyword args passed to statistic computation function
    
    RETURNS
    means   {'variable' : (n_values,) ndarray}. Mean results (across independent test runs)
            of variables output from randomization tests for each tested value. Keys:
            'signif'        Binary signficance decision (at criterion <alpha>)
            'p'             p values (negative log-transformed -log10(p) to increase with effect size)
            'stat_obs'      Observed evaluatation statistic values
            'stat_resmp'    Mean resampled statistic values (across all resamples)
            
    sds     {'variable' : (n_values,) ndarray}. Standard deviatio of results (across test runs)
            of variables output from randomization tests for each tested value. Same fields as means.
        
    ACTION
    Throws an error if any computed value is too far from expected value
    If <plot> is True, also generates a plot summarizing computed results
    """
    if seed is not None: np.random.seed(seed)
    
    test = test.lower()
    method = method.lower()    

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    # TODO Should we move some/all of these into function arguments, instead of hard-coding?    
    sim_args = dict(gain=5.0, offset=5.0, spreads=10.0, n_conds=1 if stat == 'one_sample' else 2, 
                    n=100, distribution=distribution, seed=None)
       
    if test == 'gain':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['gain']                        # Delete preset arg so it uses argument to lambda below
        gen_data = lambda gain: simulate_dataset(**sim_args,gain=gain)
        
    elif test in ['spread','spreads','sd']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['spreads']                     # Delete preset arg so it uses argument to lambda below
        gen_data = lambda spreads: simulate_dataset(**sim_args,spreads=spreads)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args['gain'] = 0     # Set gain=0 for bias test
        del sim_args['n']                           # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n_trials: simulate_dataset(**sim_args,n=n_trials)

    elif test == 'n_conds':
        test_values = [2,4,8] if test_values is None else test_values
        del sim_args['n_conds']                     # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n_conds: simulate_dataset(**sim_args,n_conds=n_conds)
        
    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)                    
           
    # Set up function for computing randomization stats
    stat_func = _str_to_stat_func(stat)
    kwargs.update(return_stats=True)
    if 'n_resamples' not in kwargs: kwargs['n_resamples'] = 100     # Default to tractable number of resamples

    results = dict(signif = np.empty((len(test_values),n_reps),dtype=bool),
                   p = np.empty((len(test_values),n_reps)),
                   stat_obs = np.empty((len(test_values),n_reps)),
                   stat_resmp = np.empty((len(test_values),n_reps)))
        
    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)
            # print(i_value, test_value, data[labels==0].mean(), data[labels==1].mean())
                        
            # TODO Need to mod this for 2-way                        
            if stat == 'one_sample':            
                p, stat_obs, stat_resmp = stat_func(data, method=method, **kwargs)
            elif stat in ['paired_sample','two_sample']:
                p, stat_obs, stat_resmp = stat_func(data[labels==1], data[labels==0], method=method, **kwargs)
            else:
                p, stat_obs, stat_resmp = stat_func(data, labels, method=method, **kwargs)
                                               
            # Determine which values are significant (p < alpha criterion)
            results['signif'][i_value,i_rep] = p < alpha
            # Negative log-transform p values so increasing = "better" and less compressed
            results['p'][i_value,i_rep] = -np.log10(p)
            results['stat_obs'][i_value,i_rep] = stat_obs
            # Compute mean resampled stat value across all resamples
            # TODO Need to mod this for 2-way             
            results['stat_resmp'][i_value,i_rep] = stat_resmp.mean(axis=0)
                                      
    # Compute mean and std dev across different reps of simulation            
    means   = {variable : values.mean(axis=-1) for variable,values in results.items()}
    sds     = {variable : values.std(axis=-1,ddof=0) for variable,values in results.items()}
    # TODO Compute sd for binary significance
        
    if plot:
        plt.figure()
        for i,variable in enumerate(means.keys()):
            mean = means[variable]
            sd = sds[variable]
            
            plt.subplot(1,4,i+1)
            plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
            plt.errorbar(test_values, mean, sd, marker='o')
            plt.xlabel('n' if test == 'bias' else test)
            plt.ylabel('-log10(p)' if variable == 'p' else variable)
            plt.title(variable)
            
        if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'stat-summary-%s-%s-%s' % (stat,method,test)))
       
    # Determine if test actually produced the expected values
    # 'gain' : Test if p values decrease and statistic increases monotonically with between-group gain
    if test == 'gain':
        assert (np.diff(means['signif']) >= 0).all(), \
            AssertionError("Significance does not decrease monotonically with between-condition mean difference")
        assert (np.diff(means['p']) >= 0).all(), \
            AssertionError("p values do not decrease monotonically with between-condition mean difference")
        assert (np.diff(means['stat_obs']) > 0).all(), \
            AssertionError("Statistic does not increase monotonically with between-condition mean difference")
        if means['stat_resmp'].ptp() > sds['stat_resmp'].max():
            warn("Resampled statistic has larger than expected range with btwn-condition mean diff")

    # 'spread' : Test if p values increase and statistic decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        assert (np.diff(means['signif']) > 0).all(), \
            AssertionError("Significance does not increase monotonically with within-condition spread increase")
        assert (np.diff(means['p']) > 0).all(), \
            AssertionError("p values do not increase monotonically with within-condition spread increase")
        if not (np.diff(means['stat_obs']) < 0).all():
            warn("Statistic does not decrease monotonically with within-condition spread increase")
        if means['stat_resmp'].ptp() > sds['stat_resmp'].max():
            warn("Resampled statistic has larger than expected range with within-condition spread increase")
                                
    # 'n' : Test if p values decrease, but statistic is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        assert (np.diff(means['signif']) >= 0).all(), \
            AssertionError("Significance does not decrease monotonically with n")
        assert (np.diff(means['p']) >= 0).all(), \
            AssertionError("p values do not decrease monotonically with n")
        if not (np.diff(means['stat_obs']) >= 0).all():
            warn("Statistic does not decrease monotonically with n")
        if means['stat_resmp'].ptp() > sds['stat_resmp'].max():
            warn("Resampled statistic has larger than expected range across n's (likely biased by n)")
        
    # 'bias': Test that statistic is not > 0 and p value ~ alpha if gain = 0, for varying n
    elif test == 'bias':
        if not (alpha/2 < 10**(-means['p'].mean())< 2*alpha):
            warn("Significance is different from expected pct when no mean difference between conditions")
        if not (np.abs(means['stat_obs']) < sds['stat_obs']).all():
            warn("Statistic is above 0 when no mean difference between conditions")
         
    return means, sds


def stat_test_battery(stats=['one_sample','paired_sample','two_sample','one_way','two_way'],
                      methods=['permutation','bootstrap'], tests=['gain','n','bias'], **kwargs):
    """ 
    Runs a battery of given tests on given randomization statistic computation methods
    
    stat_test_battery(stats=['one_sample','paired_sample','two_sample','one_way','two_way'],
                      methods=['permutation','bootstrap'], tests=['gain','n','bias'], **kwargs)
    
    ARGS
    stats       Array-like. List of statistical tests to evaluate.
                Default: ['one_sample','paired_sample','two_sample','one_way','two_way'] 
                (all supported methods)
                
    methods     Array-like. List of resampling paradigms to run.
                Default: ['permutation','bootstrap'] (all supported methods)
                                
    tests       Array-like. List of tests to run.
                Default: ['gain','n','bias'] (all supported tests)
                
    kwargs      Any other kwargs passed directly to test_randomization_stats()
    
    ACTION
    Throws an error if any estimated value for any (stat,method,test) is too far from expected value    
    """
    if isinstance(stats,str): stats = [stats]    
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    
    for stat in stats:
        for test in tests:
            for method in methods:
                print("Running %s test on %s %s" % (test,stat,method))
                # TEMP Bootstrap version of 1-way/2-way tests not coded up yet...
                if (stat in ['one_way','two_way']) and (method != 'permutation'): continue
                
                test_randomization_stats(stat, method, test=test, **kwargs)
                print('PASSED')
                if 'plot_dir' in kwargs: plt.close('all')
            
            
def _distribution_name_to_func(name):
    """ Converts name of distribution to scipy.stats-based function to generate random variables """
    # Random variables generated in a way that reproducibly matches output of Matlab
    name = name.lower()
    
    # Normal RV's : mu = mean, s = SD
    if name in ['normal','norm','gaussian','gauss']:
        return lambda n,mu,s: norm.ppf(np.random.rand(n) if np.isscalar(n) else np.random.rand(*n),
                                       loc=mu, scale=s)

    # Poisson RV's : mu = lamba (aka mean,rate), s is unused
    elif name in ['poisson','poiss']:
        return lambda n,mu,s: poisson.ppf(np.random.rand(n) if np.isscalar(n) else np.random.rand(*n),
                                          mu=mu)

    else:
        raise ValueError("%s distribution is not yet supported. Should be 'normal' | 'poisson'")
            
            
def _str_to_stat_func(stat):
    """ Converts string specifier for statistic to function for computing it """
    stat = stat.lower()
    if stat == 'one_sample':        return one_sample_test
    elif stat == 'paired_sample':   return paired_sample_test
    elif stat == 'two_sample':      return two_sample_test
    elif stat == 'one_way':         return one_way_test
    elif stat == 'two_way':         return two_way_test
    else:
        raise ValueError("Unknown stat type '%s'" % stat)

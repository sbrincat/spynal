"""
validity_test_randstats.py

Suite of tests to assess "face validity" of randomization statistic functions in randstats.py
Usually used to test new or majorly updated functions to ensure they perform as expected.

Includes tests that parametrically estimate statistics as a function of difference in distribution
means, assays of bias, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_randstats      Contains tests of randomization statistic computation functions
stat_test_battery   Runs standard battery of tests of randomization stat functions

test_confints       Contains tests of confidence interval computation functions
confint_test_battery Runs standard battery of tests of confidence interval functions
"""

import os
import time
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from neural_analysis.tests.data_fixtures import simulate_dataset
from neural_analysis.randstats import one_sample_test, paired_sample_test, two_sample_test, \
                                      one_way_test, two_way_test, \
                                      one_sample_confints, paired_sample_confints, two_sample_confints


# =============================================================================
# Tests for functions computing statistics/statistical tests
# =============================================================================
def test_randstats(stat, method, test='gain', test_values=None, term=0, distribution='normal',
                   alpha=0.05, n_reps=100, seed=None, do_tests=True, 
                   do_plots=False, plot_dir=None, **kwargs):
    """
    Basic testing for randomization statistic computation functions
    
    Generates synthetic data, computes statistics, p values, and significance using given method,
    and compares computed to expected values.
    
    means,sds,passed = test_randstats(stat,method,test='gain',test_values=None,term=0,
                                      distribution='normal',alpha=0.05,n_reps=100,seed=None,
                                      do_tests=True,do_plots=False,plot_dir=None, **kwargs)
                              
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
            
    term    Int. Which model term to modify for testing 2-way stats (unused for other stats).
            0,1 = main effects, 2 = interaction. Default: 0
                  
    distribution    String. Name of distribution to simulate data from. 
                    Options: 'normal' [default] | 'poisson'
                                
    n_reps  Int. Number of independent repetitions of tests to run. Default: 100
            
    alpha   Float. Significance criterion "alpha". Default: 0.05
                
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    do_tests Bool. Set=True to evaluate test results against expected values and
            raise an error if they fail. Default: True
            
    do_plots Bool. Set=True to plot test results. Default: False
    
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
        
    **kwargs All other keyword args passed to statistic computation function
    
    RETURNS
    means   {'variable' : (n_values,) ndarray}. Mean results (across independent test runs)
            of variables output from randomization tests for each tested value. Keys:
            'signif'        Binary signficance decision (at criterion <alpha>)
            'p'             p values
            'log_p'         p values negative log-transformed -log10(p) to increase with effect size
            'stat_obs'      Observed evaluatation statistic values
            'stat_resmp'    Mean resampled statistic values (across all resamples)
            
    sds     {'variable' : (n_values,) ndarray}. Standard deviatio of results (across test runs)
            of variables output from randomization tests for each tested value. Same fields as means.
        
    passed  Bool. True if all tests produce expected values; otherwise False.
        
    ACTION
    If do_tests is True, raisers an error if any estimated value is too far from expected value
    If do_plots is True, also generates a plot summarizing expected vs estimated values    
    """
    # Note: Set random seed once here, not for every random data generation loop below    
    if seed is not None: np.random.seed(seed)
    
    test = test.lower()
    method = method.lower()    

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    # todo Should we move some/all of these into function arguments, instead of hard-coding?
    if stat == 'two_way':       n_conds = 4         # Simulate 2-way design w/ 4 conds (2x2)
    elif stat == 'one_sample':  n_conds = 1
    else:                       n_conds = 2
    if stat == 'two_way':
        if term == 0:   gain_pattern = np.asarray([0,0,1,1])    # Set gains for effect on 1st main effect
        elif term == 1: gain_pattern = np.asarray([1,0,1,0])    # Set gains for effect on 1st main effect
        elif term == 2: gain_pattern = np.asarray([0,1,1,0])    # Set gains for effect on interaction effect
    else:
        gain_pattern = 1           
    sim_args = dict(gain=5.0*gain_pattern, offset=0.0, spreads=10.0, n_conds=n_conds, n=100,
                    distribution=distribution, seed=None)
       
    if test == 'gain':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['gain']                        # Delete preset arg so it uses argument to lambda below
        gen_data = lambda gain: simulate_dataset(**sim_args,gain=gain*gain_pattern)
        
    elif test in ['spread','spreads','sd']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['spreads']                     # Delete preset arg so it uses argument to lambda below
        gen_data = lambda spreads: simulate_dataset(**sim_args,spreads=spreads)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args.update(offset=0, gain=0)    # Set gain,offset=0 for bias test
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

    n_terms = 3 if stat == 'two_way' else 1
    results = dict(signif = np.empty((n_terms,len(test_values),n_reps),dtype=bool),
                   p = np.empty((n_terms,len(test_values),n_reps)),
                   log_p = np.empty((n_terms,len(test_values),n_reps)),
                   stat_obs = np.empty((n_terms,len(test_values),n_reps)),
                   stat_resmp = np.empty((n_terms,len(test_values),n_reps)))
        
    resmp_axis = -1 if stat == 'two_way' else 0
        
    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)
                        
            # One-sample tests                     
            if stat == 'one_sample':            
                p, stat_obs, stat_resmp = stat_func(data, method=method, **kwargs)
            
            # 1-way and 2-way ANOVA-like multi-level factorial tests   
            elif stat in ['one_way','two_way']:
                # For 2-way tests, reorg labels from (n,) vector of values in set {0-3}  
                # to (n,3) array where 1st 2 col's are 2 orthogonal factors, 3rd col is interaction
                if stat == 'two_way':
                    labels = np.stack((labels >= 2, (labels == 1) | (labels == 3), labels), axis=1)
                p, stat_obs, stat_resmp = stat_func(data, labels, method=method, **kwargs)
            
            # Paired-sample and two-sample tests    
            else:
                p, stat_obs, stat_resmp = stat_func(data[labels==1], data[labels==0], method=method, **kwargs)
                
                                               
            # Determine which values are significant (p < alpha criterion)
            results['signif'][:,i_value,i_rep]      = p < alpha
            results['p'][:,i_value,i_rep]           = p
            # Negative log-transform p values so increasing = "better" and less compressed
            results['log_p'][:,i_value,i_rep]       = -np.log10(p)
            results['stat_obs'][:,i_value,i_rep]    = stat_obs
            # Compute mean resampled stat value across all resamples
            results['stat_resmp'][:,i_value,i_rep]  = stat_resmp.mean(axis=resmp_axis)
                                 
                                      
    # Compute mean and std dev across different reps of simulation            
    means   = {variable : values.mean(axis=-1) for variable,values in results.items()}
    sds     = {variable : values.std(axis=-1,ddof=0) for variable,values in results.items()}
    variables = ['signif','log_p','stat_obs','stat_resmp']
        
    if do_plots:
        plt.figure()
        for i_vbl,variable in enumerate(variables):            
            for i_term in range(n_terms):
                mean = means[variable][i_term,:]
                sd = sds[variable][i_term,:]
                
                sp = i_term*len(variables) + i_vbl + 1
                plt.subplot(n_terms, len(variables), sp)
                plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
                plt.errorbar(test_values, mean, sd, marker='o')
                if i_term == n_terms-1:
                    plt.xlabel('n' if test == 'bias' else test)
                    plt.ylabel('-log10(p)' if variable == 'log_p' else variable)
                if i_term == 0: plt.title(variable)
                if stat == 'two_way': plt.ylabel("Term %d" % i_term)
            
        if plot_dir is not None: 
            filename = os.path.join(plot_dir,'stat-summary-%s-%s-%s' % (stat,method,test))
            if stat == 'two_way': filename += '-term%d' % term
            plt.savefig(filename)
       
    # Determine if test actually produced the expected values
    # 'gain' : Test if p values decrease and statistic increases monotonically with between-group gain
    if test == 'gain':
        evals = [((np.diff(means['signif'][term,:]) >= 0).all(),
                    "Significance does not increase monotonically with between-condition mean difference"),
                 ((np.diff(means['log_p'][term,:]) >= 0).all(),
                    "p values do not decrease monotonically with between-condition mean difference"),
                 ((np.diff(means['stat_obs'][term,:]) > 0).all(),
                    "Statistic does not increase monotonically with between-condition mean difference"),
                 (means['stat_resmp'][term,:].ptp() <= sds['stat_resmp'][term,:].max(),
                    "Resampled statistic has larger than expected range with btwn-condition mean diff")]

    # 'spread' : Test if p values increase and statistic decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(means['signif'][term,:]) > 0).all(),
                    "Significance does not decrease monotonically with within-condition spread increase"),
                 ((np.diff(means['log_p'][term,:]) >= 0).all(),
                    "p values do not increase monotonically with within-condition spread increase"),
                 ((np.diff(means['stat_obs'][term,:]) < 0).all(),
                    "Statistic does not decrease monotonically with within-condition spread increase"),
                 (means['stat_resmp'][term,:].ptp() <= sds['stat_resmp'].max(),
                    "Resampled statistic has larger than expected range with within-condition spread increase")]
                                
    # 'n' : Test if p values decrease, but statistic is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        evals = [((np.diff(means['signif'][term,:]) >= 0).all(),
                    "Significance does not increase monotonically with n"),
                 ((np.diff(means['log_p'][term,:]) >= 0).all(),
                    "p values do not decrease monotonically with n"),
                 ((np.diff(means['stat_obs'][term,:]) >= 0).all(),
                    "Statistic does not decrease monotonically with n"),
                 (means['stat_resmp'][term,:].ptp() <= sds['stat_resmp'].max(),
                    "Resampled statistic has larger than expected range across n's (likely biased by n)")]
        
    # 'bias': Test that statistic is not > 0 and p value ~ alpha if gain = 0, for varying n
    elif test == 'bias':
        evals = [(((np.abs(means['signif'][term,:].mean()) - alpha) < sds['signif'][term,:]).all(),
                    "Significance different from expected pct (%.1f) when no mean diff between conds" % alpha),
                 (((np.abs(means['p'][term,:].mean()) - 0.5) < sds['p'][term,:]).all(),
                    "p values are different from expected value (0.5) when no mean difference between conditions"),
                 ((np.abs(means['stat_obs'][term,:]) < sds['stat_obs'][term,:]).all(),
                    "Statistic is above 0 when no mean difference between conditions")]

    passed = True
    for cond,message in evals:
        if not cond:    passed = False
        
        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  print("Warning: " + message)
                
    return means, sds, passed
    

def stat_test_battery(stats=['one_sample','paired_sample','two_sample','one_way','two_way'],
                      methods=['permutation','bootstrap'], tests=['gain','n','bias'], 
                      do_tests=True, **kwargs):
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
                
    do_tests    Bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True
                                
    kwargs      Any other kwargs passed directly to test_randstats()
    
    ACTION
    Raises an error or warning if any estimated value for any (stat,method,test)
    is too far from expected value
    """
    if isinstance(stats,str): stats = [stats]    
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    
    for stat in stats:
        for test in tests:
            for method in methods:
                # TEMP Bootstrap version of 1-way/2-way tests not coded up yet...
                if (stat in ['one_way','two_way']) and (method != 'permutation'): continue
                print("Running %s test on %s %s" % (test,stat,method))
                                
                # Run separate tests for each term of 2-way stats (2 main effects and interaction)
                if stat == 'two_way':
                    passed = np.empty((3,))
                    for term in range(3):
                        _,_,passed[term] = test_randstats(stat, method, test=test, term=term,
                                                          do_tests=do_tests, **kwargs)
                    passed = passed.all()
                    
                else:
                    _,_,passed = test_randstats(stat, method, test=test, do_tests=do_tests, **kwargs)
                    
                print('%s' % 'PASSED' if passed else 'FAILED')
                if 'plot_dir' in kwargs: plt.close('all')
            

# =============================================================================
# Tests for functions computing confidence intervals
# =============================================================================
def test_confints(stat, test='gain', test_values=None, distribution='normal', confint=0.95,
                  n_reps=100, seed=None, do_tests=True, do_plots=False, plot_dir=None, **kwargs):
    """
    Basic testing for bootstrap confidence interval computation functions
    
    Generates synthetic data, computes statistics, confints, and significance using given method,
    and compares computed to expected values.
    
    means,sds,passed = test_confints(stat,test='gain',test_values=None,
                                     distribution='normal',confint=0.95,n_reps=100,seed=None,
                                     do_tests=True,do_plots=False,plot_dir=None, **kwargs)
                              
    ARGS
    stat    String. Type of statistical test to evaluate:
            'one_sample' | 'paired_sample' | 'two_sample'
                        
    test    String. Type of test to run. Default: 'gain'. Options:
            'gain'  Tests multiple values for between-condition response difference (gain)
                    Checks for monotonically increasing confints
            'spread'Tests multiple values for distribution spread (SD)
                    Checks for monotonically decreasing confints
            'n'     Tests multiple values of number of trials (n)
                    Checks that confints decrease with n                
            'bias'  Tests multiple n values with 0 btwn-cond difference
                    Checks that stat doesn't vary and p value remains ~ 0 (unbiased)

    test_values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'gain'      Btwn-condition response differences (gains). Default: [1,2,5,10,20]
            'spread'    Gaussian SDs for each response distribution. Default: [1,2,5,10,20]
            'n'/'bias'  Trial numbers. Default: [25,50,100,200,400,800]
            
    distribution    String. Name of distribution to simulate data from. 
                    Options: 'normal' [default] | 'poisson'
            
    confint Float. Confidence interval to compute (1-alpha). Default: 0.95 (95% CI)
                                
    n_reps  Int. Number of independent repetitions of tests to run. Default: 100
                            
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    do_tests Bool. Set=True to evaluate test results against expected values and
            raise an error if they fail. Default: True
            
    do_plots Bool. Set=True to plot test results. Default: False
    
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
        
    **kwargs All other keyword args passed to confint computation function
    
    RETURNS
    means   {'variable' : (n_values,) ndarray}. Mean results (across independent test runs)
            of variables output from randomization tests for each tested value. Keys:
            'signif'        Binary signficance decision (at criterion <alpha>)
            'p'             p values (negative log-transformed -log10(p) to increase with effect size)
            'stat_obs'      Observed evaluatation statistic values
            'stat_resmp'    Mean resampled statistic values (across all resamples)
            
    sds     {'variable' : (n_values,) ndarray}. Standard deviatio of results (across test runs)
            of variables output from randomization tests for each tested value. Same fields as means.
        
    passed  Bool. True if all tests produce expected values; otherwise False.
        
    ACTION
    If do_tests is True, raisers an error if any estimated value is too far from expected value
    If do_plots is True, also generates a plot summarizing expected vs estimated values    
    """
    term = 0    # HACK Hard-code term=0, but keep term for possible future extension to 1-way/2-way stats
    # Note: Set random seed once here, not for every random data generation loop below    
    if seed is not None: np.random.seed(seed)
    
    test = test.lower()

    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    # todo Should we move some/all of these into function arguments, instead of hard-coding?
    if stat == 'two_way':       n_conds = 4         # Simulate 2-way design w/ 4 conds (2x2)
    elif stat == 'one_sample':  n_conds = 1
    else:                       n_conds = 2
    if stat == 'two_way':
        if term == 0:   gain_pattern = np.asarray([0,0,1,1])    # Set gains for effect on 1st main effect
        elif term == 1: gain_pattern = np.asarray([1,0,1,0])    # Set gains for effect on 1st main effect
        elif term == 2: gain_pattern = np.asarray([0,1,1,0])    # Set gains for effect on interaction effect
    else:
        gain_pattern = 1           
    sim_args = dict(gain=5.0*gain_pattern, offset=0.0, spreads=10.0, n_conds=n_conds, n=100,
                    distribution=distribution, seed=None)
       
    if test == 'gain':
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['gain']                        # Delete preset arg so it uses argument to lambda below
        gen_data = lambda gain: simulate_dataset(**sim_args,gain=gain*gain_pattern)
        
    elif test in ['spread','spreads','sd']:
        test_values = [1,2,5,10,20] if test_values is None else test_values
        del sim_args['spreads']                     # Delete preset arg so it uses argument to lambda below
        gen_data = lambda spreads: simulate_dataset(**sim_args,spreads=spreads)

    elif test in ['n','n_trials','bias']:
        test_values = [25,50,100,200,400,800] if test_values is None else test_values
        if test == 'bias': sim_args.update(offset=0, gain=0)    # Set gain,offset=0 for bias test
        del sim_args['n']                           # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n_trials: simulate_dataset(**sim_args,n=n_trials)

    elif test == 'n_conds':
        test_values = [2,4,8] if test_values is None else test_values
        del sim_args['n_conds']                     # Delete preset arg so it uses argument to lambda below
        gen_data = lambda n_conds: simulate_dataset(**sim_args,n_conds=n_conds)
        
    else:
        raise ValueError("Unsupported value '%s' set for <test>" % test)                    
           
    # Set up function for computing randomization stats
    confint_func = _str_to_confint_func(stat)
    kwargs.update(return_stats=True)
    kwargs.update(confint=confint)
    if 'n_resamples' not in kwargs: kwargs['n_resamples'] = 100     # Default to tractable number of resamples
    
    n_terms = 3 if stat == 'two_way' else 1
    results = dict(confints = np.empty((n_terms,2,len(test_values),n_reps)),
                   ci_diff = np.empty((n_terms,len(test_values),n_reps)),
                   signif = np.empty((n_terms,len(test_values),n_reps),dtype=bool),
                   stat_obs = np.empty((n_terms,len(test_values),n_reps)),
                   stat_resmp = np.empty((n_terms,len(test_values),n_reps)))
        
    resmp_axis = -1 if stat == 'two_way' else 0
        
    for i_value,test_value in enumerate(test_values):
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)
                        
            # One-sample tests                     
            if stat == 'one_sample':            
                confints, stat_obs, stat_resmp = confint_func(data, **kwargs)
            
            # 1-way and 2-way ANOVA-like multi-level factorial tests   
            elif stat in ['one_way','two_way']:
                # For 2-way tests, reorg labels from (n,) vector of values in set {0-3}  
                # to (n,3) array where 1st 2 col's are 2 orthogonal factors, 3rd col is interaction
                if stat == 'two_way':
                    labels = np.stack((labels >= 2, (labels == 1) | (labels == 3), labels), axis=1)
                confints, stat_obs, stat_resmp = confint_func(data, labels, **kwargs)
            
            # Paired-sample and two-sample tests    
            else:
                confints, stat_obs, stat_resmp = confint_func(data[labels==1], data[labels==0], **kwargs)
                                                               
            # Determine which values are significant (p < alpha criterion)
            results['signif'][:,i_value,i_rep] = (confints > 0).all()
            # Negative log-transform p values so increasing = "better" and less compressed
            results['confints'][:,:,i_value,i_rep] = confints
            results['stat_obs'][:,i_value,i_rep] = stat_obs
            # Compute mean resampled stat value across all resamples
            results['stat_resmp'][:,i_value,i_rep] = stat_resmp.mean(axis=resmp_axis)
                                 
    # Compute difference btwn each set of confidence intervals (upper - lower)
    results['ci_diff'] = np.diff(results['confints'], axis=1).squeeze(axis=1)
                                        
    # Compute mean and std dev across different reps of simulation            
    means   = {variable : values.mean(axis=-1) for variable,values in results.items()}
    sds     = {variable : values.std(axis=-1,ddof=0) for variable,values in results.items()}
    variables = list(means.keys())
        
    if do_plots:
        plt.figure()
        for i_vbl,variable in enumerate(variables):            
            for i_term in range(n_terms):
                mean = means[variable][i_term,:]
                sd = sds[variable][i_term,:]
                
                sp = i_term*len(variables) + i_vbl + 1
                plt.subplot(n_terms, len(variables), sp)
                plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')
                if variable == 'confints':
                    # Plot lower, upper confints separately
                    plt.errorbar(test_values, mean[0,:], sd[0,:], marker='o')
                    plt.errorbar(test_values, mean[1,:], sd[1,:], marker='o')
                else:
                    plt.errorbar(test_values, mean, sd, marker='o')
                if i_term == n_terms-1:
                    plt.xlabel('n' if test == 'bias' else test)
                    plt.ylabel(variable)
                if i_term == 0: plt.title(variable)
                if stat == 'two_way': plt.ylabel("Term %d" % i_term)
            
        if plot_dir is not None: 
            filename = os.path.join(plot_dir,'stat-summary-%s-%s' % (stat,test))
            if stat == 'two_way': filename += '-term%d' % term
            plt.savefig(filename)
       
    # Determine if test actually produced the expected values    
    # 'gain' : Test if statistic and significance increases monotonically with between-group gain
    if test == 'gain':
        evals = [((np.diff(means['signif'][term,:]) >= 0).all(),
                    "Significance does not increase monotonically with between-condition mean difference"),
                 (means['ci_diff'][term,:].ptp() < sds['ci_diff'][term,:].max(),
                    "Confints have larger than expected range with between-condition mean difference"),
                 ((np.diff(means['stat_obs'][term,:]) > 0).all(),
                    "Statistic does not increase monotonically with between-condition mean difference"),
                 ((np.diff(means['stat_resmp'][term,:]) > 0).all(),
                    "Resampled statistic does not increase monotonically with between-condition mean difference")]
        # Confidence intervals not expected to change with mean for 1-sample data, so remove test
        if stat == 'one_sample': evals.pop(1)
            
    # 'spread' : Test if confints increase and statistic decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        evals = [((np.diff(means['signif'][term,:]) > 0).all(),
                    "Significance does not decrease monotonically with within-condition spread increase"),
                 ((np.diff(means['ci_diff'][term,:]) > 0).all(),
                    "Confints do not increase monotonically with within-condition spread increase"),
                 ((np.diff(means['stat_obs'][term,:]) < 0).all(),
                    "Statistic does not decrease monotonically with within-condition spread increase"),
                 ((np.diff(means['stat_resmp'][term,:]) < 0).all(),
                    "Resampled statistic does not decrease monotonically with within-condition spread increase")]
                                
    # 'n' : Test if confints decrease, but statistic is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        evals = [((np.diff(means['signif'][term,:]) >= 0).all(), 
                    "Significance does not increase monotonically with n"),
                 ((np.diff(means['ci_diff'][term,:]) <= 0).all(),
                    "Confints do not decrease monotonically with n"),
                 (means['stat_obs'][term,:].ptp() <= sds['stat_obs'][term,:].max(),
                    "Statistic has larger than expected range with n"),
                 (means['stat_resmp'][term,:].ptp() <= sds['stat_resmp'][term,:].max(),
                    "Resampled statistic has larger than expected range with n")]
        
    # 'bias': Test that statistic is not > 0 and p value ~ alpha if gain = 0, for varying n
    elif test == 'bias':
        evals = [((means['confints'][term,0,:] < 0).all() & (0 < means['confints'][term,1,:]).all(),
                    "Confints don't overlap with 0 when no mean difference between conditions"),
                 ((np.abs(means['stat_obs'][term,:]) < sds['stat_obs'][term,:].max()).all(),
                    "Statistic is above 0 when no mean difference between conditions"),
                 ((np.abs(means['stat_resmp'][term,:]) < sds['stat_resmp'][term,:].max()).all(),
                    "Resampled statistic is above 0 when no mean difference between conditions")]
         
    passed = True
    for cond,message in evals:
        if not cond:    passed = False
        
        # Raise an error for test fails if do_tests is True
        if do_tests:    assert cond, AssertionError(message)
        # Just issue a warning for test fails if do_tests is False
        elif not cond:  print("Warning: " + message)
                 
    return means, sds, passed


def confint_test_battery(stats=['one_sample','paired_sample','two_sample'],
                         tests=['gain','n','bias'], do_tests=True, **kwargs):
    """ 
    Runs a battery of given tests on given bootstrap confidence interval computation methods
    
    confint_test_battery(stats=['one_sample','paired_sample','two_sample'],
                         tests=['gain','n','bias'], **kwargs)
    
    ARGS
    stats       Array-like. List of statistical tests to evaluate.
                Default: ['one_sample','paired_sample','two_sample'] (all supported methods)
                                                
    tests       Array-like. List of tests to run.
                Default: ['gain','n','bias'] (all supported tests)
                
    do_tests    Bool. Set=True to evaluate test results against expected values and
                raise an error if they fail. Default: True
                                
    kwargs      Any other kwargs passed directly to test_confints()
    
    ACTION
    Raises an error or warning if any estimated value for any (stat,test)
    is too far from expected value
    """
    if isinstance(stats,str): stats = [stats]    
    if isinstance(tests,str): tests = [tests]
    
    for stat in stats:
        for test in tests:
            print("Running %s test on %s" % (test,stat))
            
            # Run separate tests for each term of 2-way stats (2 main effects and interaction)
            if stat == 'two_way':
                passed = np.empty((3,))
                for term in range(3):
                    _,_,passed[term] = test_confints(stat, test=test, term=term,
                                                     do_tests=do_tests, **kwargs)
                passed = passed.all()
                
            else:
                _,_,passed = test_confints(stat, test=test, do_tests=do_tests, **kwargs)
                
            print('%s' % 'PASSED' if passed else 'FAILED')
            if 'plot_dir' in kwargs: plt.close('all')
 
 
# =============================================================================
# Helper functions
# =============================================================================             
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


def _str_to_confint_func(stat):
    """ Converts string specifier for statistic to function for computing its confints """
    stat = stat.lower()
    if stat == 'one_sample':        return one_sample_confints
    elif stat == 'paired_sample':   return paired_sample_confints
    elif stat == 'two_sample':      return two_sample_confints
    else:
        raise ValueError("Unknown stat type '%s'" % stat)
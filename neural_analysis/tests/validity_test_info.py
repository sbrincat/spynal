"""
validity_test_info.py

Suite of tests to assess "face validity" of neural information computation functions in info.py
Usually used to test new or majorly updated functions to ensure they perform as expected.

Includes tests that parametrically estimate information as a function of difference in distribution
means, assays of bias, etc. to establish methods produce expected pattern of results. 

Plots results and runs assertions that basic expected results are reproduced

FUNCTIONS
test_neural_info        Contains tests of neural information computation functions
info_test_battery       Runs standard battery of tests of information computation functions
"""

import os
import time
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, poisson
from patsy import dmatrix

from ..info import neural_info, neural_info_2groups


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
    # todo Add ability to simulate independent data series, different n for each cond, 
    if seed is not None: np.random.seed(seed)
        
    # Single gain = incremental difference btwn cond 0 and 1, 1 and 2, etc.
    # Convert to vector of gains for each condition
    if np.isscalar(gain): gain = gain*np.arange(n_conds)

    # Convert scalar spread value to vector for each condition
    if np.isscalar(spreads): spreads = spreads*np.ones((n_conds,))
            
    assert len(gain) == n_conds, \
        ValueError("Vector-valued <gain> must have length == n_conds (%d != %d)" \
                    % (len(gain), n_conds))

    assert len(spreads) == n_conds, \
        ValueError("Vector-valued <spreads> must have length == n_conds (%d != %d)" \
                    % (len(spreads), n_conds))

    # Final mean value = baseline + condition-specific gain
    means = offset + gain

    # Generate data for each condition and stack together -> (n_trials*n_conds,) array
    data = np.hstack([simulate_data(distribution=distribution, mean=mean, spread=spread,
                                    n=n, seed=None)
                      for mean,spread in zip(means,spreads)])

    # Create (n_trials_total,) vector of group labels (ints in set 0-n_conds-1)
    labels = np.hstack([i_cond*np.ones((n,)) for i_cond in range(n_conds)])

    return data, labels


def test_neural_info(method, test='gain', test_values=None, distribution='normal', n_reps=100,
                     seed=None, arg_type='label', plot=False, plot_dir=None, **kwargs):
    """
    Basic testing for functions estimating neural information.
    
    Generates synthetic data, estimates information using given method,
    and compares estimated to expected values.
    
    info,sd = test_neural_info(method,test='gain',test_values=None,distribution='normal',n_reps=100,
                               seed=None,arg_type='label',n_reps=100,plot=False,plot_dir=None, **kwargs)
                              
    ARGS
    method  String. Name of information function to test:
            'pev' | 'anova1' | 'anova2' | 'regress'
            
    test    String. Type of test to run. Default: 'gain'. Options:
            'gain'  Tests multiple values for between-condition response difference (gain)
                    Checks for monotonically increasing information
            'spread'Tests multiple values for distribution spread (SD)
                    Checks for monotonically decreasing information
            'n'     Tests multiple values of number of trials (n)
                    Checks that information doesn't vary with n.                    
            'bias'  Tests multiple n values with 0 btwn-cond difference
                    Checks that information is not > 0 (unbiased)
            'n_conds' Tests multiple values for number of conditions
                    (no actual checking, just to see behavior of info measure)
            
    test_values  (n_values,) array-like. List of values to test. 
            Interpretation and defaults are test-specific:
            'gain'      Btwn-condition response differences (gains). Default: [1,2,5,10,20]
            'spread'    Gaussian SDs for each response distribution. Default: [1,2,5,10,20]
            'n'/'bias'  Trial numbers. Default: [25,50,100,200,400,800]
            'n_conds'   Number of conditions. Default: [2,4,8]
            
    distribution    String. Name of distribution to simulate data from. 
                    Options: 'normal' [default] | 'poisson'
                                
    n_reps  Int. Number of independent repetitions of tests to run. Default: 100
            
    seed    Int. Random generator seed for repeatable results.
            Set=None [default] for unseeded random numbers.

    arg_type String. Which input-argument version of info computing function to use:
            'label'     : Standard version with labels,data arguments [default]
            '2groups'   : Binary contrast version with data1,data2 arguments
                                    
    plot    Bool. Set=True to plot test results. Default: False
    
    plot_dir String. Full-path directory to save plots to. Set=None [default] to not save plots.
        
    **kwargs All other keyword args passed to information estimation function
    
    RETURNS
    info    (n_values,) ndarray. Estimated information for each tested value
    sd      (n_values,) ndarray. Across-run SD of information for each tested value
    
    ACTION
    Throws an error if any estimated information value is too far from expected value
    If <plot> is True, also generates a plot summarizing estimated information
    """
    if seed is not None: np.random.seed(seed)
    
    test = test.lower()
    method = method.lower()
    arg_type = arg_type.lower()
    
    assert arg_type in ['label','labels','2groups'], \
        ValueError("arg_type value %s not supported. Should be 'label' | '2groups'" % arg_type)
    if arg_type == '2groups':
        assert test != 'n_conds', \
            ValueError("Cannot run 'n_conds' test of condition number with '2groups' arg_type")
    
    # Set defaults for tested values and set up data generator function depending on <test>
    # Note: Only set random seed once above, don't reset in data generator function calls
    # todo Should we move some/all of these into function arguments, instead of hard-coding?
    sim_args = dict(gain=5.0, offset=5.0, spreads=5.0, n_conds=2, n=500, 
                    distribution=distribution, seed=None)
       
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
        
    method_ = method
            
    # Deal with special-case linear models -- funnel into pev function 
    if method in ['pev','regress','anova1','anova2','anovan']:
        method_ = 'pev'
        # For PEV, additional argument to neural_info() specifying linear model to use
        if method == 'pev': kwargs.update({'model':'anova1'})
        else:               kwargs.update({'model':method})
        
    # For these signed binary methods, reverse default grouping bc in stimulated data,
    # group1 > group0, but signed info assumes opposite preference
    if method in ['dprime','d','cohensd', 'auroc','roc','aucroc','auc']:
        groups = [1,0]
        if (arg_type != '2groups') and ('groups' not in kwargs): kwargs.update({'groups':[1,0]})
    else:
        groups = [0,1]
                               
    # Expected baseline value for no btwn-condition = 0.5 for AUROC, 0 for other methods
    baseline = 0.5 if method in ['auroc','roc','aucroc','auc'] else 0
                 
    info = np.empty((len(test_values),n_reps))
        
    for i_value,test_value in enumerate(test_values):        
        for i_rep in range(n_reps):
            # Generate simulated data with current test value
            data,labels = gen_data(test_value)
                        
            # For regression model, convert labels list -> design matrix, append intercept term 
            if method == 'regress': labels = dmatrix('1 + C(vbl1,Sum)',{'vbl1':labels})
                            
            if arg_type == '2groups':
                info[i_value,i_rep] = neural_info_2groups(data[labels==groups[0]], data[labels==groups[1]],
                                                          method=method_, **kwargs)
            else:                            
                info[i_value,i_rep] = neural_info(labels, data, method=method_, **kwargs)
            
    # Compute mean and std dev across different reps of simulation            
    sd      = info.std(axis=1,ddof=0)
    info    = info.mean(axis=1)
    
    if plot:
        plt.figure()
        plt.grid(axis='both',color=[0.75,0.75,0.75],linestyle=':')        
        plt.errorbar(test_values, info, sd, marker='o')
        xlabel = 'n' if test == 'bias' else test
        plt.xlabel(xlabel)
        plt.ylabel("Information (%s)" % method_)
        if plot_dir is not None: plt.savefig(os.path.join(plot_dir,'info-summary-%s-%s' % (method,test)))
       
    # Determine if test actually produced the expected values
    # 'gain' : Test if information increases monotonically with between-group gain
    if test == 'gain':
        assert (np.diff(info) >= 0).all(), \
            AssertionError("Information does not increase monotonically with between-condition mean difference")

    # 'spread' : Test if information decreases monotonically with within-group spread
    elif test in ['spread','spreads','sd']:
        assert (np.diff(info) <= 0).all(), \
            AssertionError("Information does not decrease monotonically with within-condition spread increase")
                                
    # 'n' : Test if information is ~ same for all values of n (unbiased by n)      
    elif test in ['n','n_trials']:
        assert info.ptp() < sd.max(), \
            AssertionError("Information has larger than expected range across n's (likely biased by n)")
        
    # 'bias': Test if information is not > baseline if gain = 0, for varying n
    elif test == 'bias':
        assert ((info - baseline) < sd).all(), \
            AssertionError("Information is above baseline for no mean difference between conditions")
         
    return info, sd


def info_test_battery(methods=['pev','dprime','auroc','mutual_information'], 
                      tests=['gain','spread','n','bias'], **kwargs):
    """ 
    Runs a battery of given tests on given neural information computation methods
    
    info_test_battery(methods=['pev','dprime','auroc','mutual_information'],
                      tests=['gain','spread','n','bias'], **kwargs)
    
    ARGS
    methods     Array-like. List of neural information methods to test.
                Default: ['pev','dprime','auroc','mutual_information'] (all supported methods)
                
    tests       Array-like. List of tests to run.
                Note: certain combinations of methods,tests are skipped, as they are not expected to pass
                (ie 'n_trials','bias' tests skipped for biased metric 'mutual_information')
                Default: ['gain','n','bias'] (all supported tests)
                
    kwargs      Any other kwargs passed directly to test_neural_info()
    
    ACTION
    Throws an error if any estimated information value for any (method,test) is too far from expected value    
    """
    if isinstance(methods,str): methods = [methods]
    if isinstance(tests,str): tests = [tests]
    
    for test in tests:
        for method in methods:
            print("Running %s test on %s" % (test,method))
            # Skip tests expected to fail due to properties of given info measures (ie ones that are biased/affected by n)
            if (test in ['n','n_trials','bias']) and (method in ['mutual_information','mutual_info']): continue
            
            test_neural_info(method, test=test, **kwargs)
            print('PASSED')
            # If saving plots to file, let's not leave them all open
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
            
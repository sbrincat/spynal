# -*- coding: utf-8 -*-
"""
randstats     A module for nonparametric randomization, permutation, and bootstrap statistics

FUNCTIONS
### One-sample tests (analogs of 1-sample t-test) ###
one_sample_test                 Wrapper for all 1-sample tests (~ 1-sample t-test)
one_sample_randomization_test   Randomized-sign 1-sample test
one_sample_bootstrap_test       Bootstrap 1-sample test

### Paired-sample tests (analogs of paired-sample t-test)  ###
paired_sample_test              Wrapper for all paired-sample tests (~ paired-sample t-test)
paired_sample_test_labels       Same, but with (data,labels) arg format instead of (data1,data2)
paired_sample_permutation_test  Permutation paired-sample test
paired_sample_bootstrap_test    Bootstrap paired-sample test

### Two-sample tests (analogs of 2-sample t-test) ###
two_sample_test                 Wrapper for all 2-sample tests (~ 2-sample t-test)
two_sample_test_labels          Same, but with (data,labels) arg format instead of (data1,data2)
two_sample_permutation_test     Permutation 2-sample test
two_sample_bootstrap_test       Bootstrap 2-sample test

### One-way/Two-way multi-level tests (analogs of 1-way/2-way ANOVA) ###
one_way_test                    Wrapper for all 1-way test (~ 1-way ANOVA/F-test)
one_way_permutation_test        Permutation 1-way multi-level test
two_way_permutation_test        Permutation 2-way multi-level/multi-factor test

### Confidence intervals ###
bootstrap_confints              Bootstrap confidence intervals for any one-sample stat

### Functions to compute frequently-used stats ###
one_sample_tstat                Mass univariate 1-sample t-statistic
paired_tstat                    Mass univariate paired-sample t-statistic
two_sample_tstat                Mass univariate 2-sample t-statistic
one_way_fstat                   Mass univariate 1-way F-statistic
two_way_fstat                   Mass univariate 2-way (with interaction) F-statistic

### Random-sample generators ###
permutations        Generates random permutations (resampling w/o replacement)
bootstraps          Generates random bootstrap samples (resampling w/ replacement)
signs               Generates random binary variables (eg for sign tests)


Created on Tue Jul 30 16:28:12 2019

@author: sbrincat
"""
# TODO  Parallelize resampling loops! (using joblib?)
# TODO  Add option to return resamples as actual sequences (not iterators)?
# TODO  How to pass custom function to paired_sample tests, when they're actually evaluated as 1-samples?
# TODO  How to enforce dimensionality of custom functions (esp re: keepdims)?

from math import sqrt
from warnings import warn
import numpy as np

try:
    from .utils import set_random_seed
# TEMP    
except ImportError:
    from utils import set_random_seed


# =============================================================================
# One-sample randomization tests
# =============================================================================
def one_sample_test(data, axis=0, method='randomization', **kwargs):
    """
    Mass univariate 1-sample test of whether any arbitrary 1-sample stat (eg mean)
    is different from a given value <mu>, often 0 (analogous to 1-sample t-test).
    
    Wrapper around functions for specific 1-sample tests. See those for details.
    
    ARGS
    data        (...,n,...) ndarray. Data to run test on. Arbitrary shape.
    
    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)

    method      String. Resampling paradigm to use for test:
                'randomization' : Randomization sign test in one_sample_randomization_test [default]
                'bootstrap'     : Bootstrap test in one_sample_bootstrap_test
            
    See specific test functions for further arguments.
    
    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for <axis> reduced to a singleton. 
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.    
    """
    method = method.lower()
    
    if method in ['randomization','permutation','sign']:    test_func = one_sample_randomization_test
    elif method == 'bootstrap':                             test_func = one_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Should be 'randomization' or 'bootstrap'" % method)
    
    return test_func(data, axis=axis, **kwargs)

    
def one_sample_randomization_test(data, axis=0, mu=0, stat='t', tail='both',
                                  n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate 1-sample randomization test of whether any arbitrary
    1-sample stat (eg mean) is different from a given value <mu>, often 0
    (analogous to 1-sample t-test).

    For each random resample, each observation is randomly assigned a sign
    (+ or -), similar to a Fisher sign test.  The same stat is then computed on
    the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    ARGS
    data        (...,n,...) ndarray. Data to run test on.  Arbitrary shape.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    mu          Scalar. Theoretical value under the null hypothesis, to compare
                distibution of data to. Default: 0

    stat        String | Callable. String specifier for statistic to resample:
                't'     : 1-sample t-statistic [default]
                'mean'  : mean across observations

                -or- Custom function to generate resampled statistic of interest.
                Should take single array argument (data) with <axis> corresponding
                to trials/observations, and return a scalar value (for each independent 
                data series if multiple given).

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.
            
    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    - Any additional kwargs passed directly to stat function -

    RETURNS
    p           Float | (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for (...,n_resamples,...) reduced to a singleton. 
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    Float | (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>. 

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but (...,n_resamples,...) has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.6.2
    """
    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat)    # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    # Copy data variable to avoid changing values in caller
    data = data.copy()
    
    # Reshape data so axis 0 is observations
    if axis != 0: data = np.moveaxis(data, axis, 0)
    n = data.shape[0]

    # Subtract hypothetical mean(s) 'mu' from data to center them there
    if mu != 0:  data -= mu

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, **kwargs)

    # Create generators with n_resamples-1 length-n independent Bernoulli RV's
    resamples = signs(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_resmp = np.empty((n_resamples-1,*data.shape[1:]))
        for i_resmp,resample in enumerate(resamples):
            # Randomly flip signs of obervations flagged by random variables above
            data[resample,...] *= -1
            # Compute statistic on resampled data
            stat_resmp[i_resmp,...] = stat_func(data, **kwargs)

        # p value = proportion of permutation-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, 0, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((1,*data.shape[1:]))
        for resample in resamples:
            # Randomly flip signs of obervations flagged by random variables above
            data[resample,...] *= -1
            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs, stat_func(data, **kwargs))

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples


    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: stat_obs = stat_obs.item()
        
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


one_sample_permutation_test = one_sample_randomization_test
""" Alias one_sample_randomization_test as one_sample_permutation_test """


def one_sample_bootstrap_test(data, axis=0, mu=0, stat='t', tail='both',
                              n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate 1-sample bootstrap test of whether any arbitrary
    1-sample stat (eg mean) is different from a given value <mu>, often 0
    (analogous to 1-sample t-test).

    Computes stat on each bootstrap resample, and subtracts off stat computed on
    observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.

    p = one_sample_bootstrap_test(data,axis=0,mu=0,stat='t',tail='both',
                                  n_resamples=10000,return_stats=False,**kwargs)

    p,stat_obs,stat_resmp = one_sample_bootstrap_test(data,mu=0,stat='t',tail='both',
                                                      n_resamples=10000,return_stats=True,
                                                      **kwargs)
                              
    ARGS
    data        (...,n,...) ndarray. Data to run test on.  Arbitrary shape.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    mu          Scalar. Theoretical value under the null hypothesis, to compare
                distibution of data to. Default: 0

    stat        String | Callable. String specifier for statistic to resample:
                't'     : 1-sample t-statistic [default]
                'mean'  : mean across observations

                -or- Custom function to generate resampled statistic of interest.
                Should take single array argument (data) with <axis> corresponding
                to trials/observations, and return a scalar value (for each independent 
                data series if multiple given).

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    - Any additional kwargs passed directly to stat function -

    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for <axis> reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch. 3.10
    """    
    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat)    # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    # Copy data variable to avoid changing values in caller
    if (axis != 0) or (mu != 0): data = data.copy()

    # Reshape data so axis 0 is observations
    if axis != 0: data = np.moveaxis(data, axis, 0)
    n = data.shape[0]

    # Subtract hypothetical mean(s) 'mu' from data to center them there
    if mu != 0: data -= mu

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, **kwargs)

    # Create generators with n_resamples-1 length-n independent Bernoulli RV's
    resamples = bootstraps(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_resmp = np.empty((n_resamples-1,*data.shape[1:]))
        for i_resmp,resample in enumerate(resamples):
            # Compute statistic on resampled data
            stat_resmp[i_resmp,...] = stat_func(data[resample,...], **kwargs)                      

        # Subtract observed statistic from distribution of resampled statistics
        stat_resmp -= stat_obs
        
        # p value = proportion of bootstrap-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, 0, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((1,*data.shape[1:]))
        for resample in resamples:
            # Compute statistic on resampled data
            stat_resmp = stat_func(data[resample,...])
                                   
            # Subtract observed statistic from distribution of resampled statistics
            stat_resmp -= stat_obs
                        
            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs, stat_resmp, **kwargs)

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples
    
    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: stat_obs = stat_obs.item()
        
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p
    
    
# =============================================================================
# Paired-sample randomization tests
# =============================================================================
def paired_sample_test(data1, data2, axis=0, method='permutation', **kwargs):
    """
    Mass univariate paired-sample test of whether any arbitrary statistic
    differs between paired samples (analogous to paired-sample t-test)
    
    Wrapper around functions for specific paired-sample tests. See those for details.
    
    ARGS
    data1       (...,n,...) ndarray. Data from one group to compare.
    data2       (...,n,...) ndarray. Data from a second group to compare.
                Must have same number of observations (trials) n as data1, but 
                otherwise arbitrary shape.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    method      String. Resampling paradigm to use for test:
                'permutation'   : Permutation test in paired_sample_permutation_test [default]
                'bootstrap'     : Bootstrap test in paired_sample_bootstrap_test
                
    See specific test functions for further arguments.
    
    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for (...,n_resamples,...) reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.    
    """
    method = method.lower()
    
    if method in ['randomization','permutation','sign']:    test_func = paired_sample_permutation_test
    elif method == 'bootstrap':                             test_func = paired_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Should be 'permutation' or 'bootstrap'" % method)
    
    return test_func(data1, data2, axis=axis, **kwargs)


def paired_sample_test_labels(data, labels, axis=0, method='permutation', groups=None, **kwargs):
    """
    Mass univariate paired-sample test of whether any arbitrary statistic
    differs between paired samples (analogous to paired-sample t-test)
    
    Wrapper around functions for specific paired-sample tests. See those for details.
    
    This version is an alternative interface to paired_sample_test() with (data,labels) 
    arguments instead of (data1,data2)
    
    ARGS
    data        (...,N,...) ndarray. Data from both groups to run test on. 
                Arbitrary shape, but both groups must have the same n (n1 = n2 = N/2).
    
    labels      (N,) array-like. Group labels for each observation (trial),
                identifying which group/condition each observation belongs to.
    
    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    method      String. Resampling paradigm to use for test:
                'permutation'   : Permutation test in paired_sample_permutation_test [default]
                'bootstrap'     : Bootstrap test in paired_sample_bootstrap_test
                
    groups      (n_groups,) array-like. List of labels for each group (condition).
                Default: set of unique values in <labels> (np.unique(labels))
                                
    See specific test functions for further arguments.
    
    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for (...,n_resamples,...) reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.    
    """
    labels = np.asarray(labels)
    if groups is None: groups = np.unique(labels)
    
    n1 = (labels == groups[0]).sum()
    n2 = (labels == groups[1]).sum()
    
    assert len(groups) == 2, "Must have only 2 groups (conditions) in labels"
    assert n1 == n2, \
        "Two groups must have same number of observations/trials (%d != %d)" % (n1,n2)
    
    return paired_sample_test(data.compress(labels == groups[0], axis=axis),
                              data.compress(labels == groups[1], axis=axis),
                              axis=axis, method=method, **kwargs)
    
    
def paired_sample_permutation_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                   n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate permutation test of whether any arbitrary statistic
    differs between paired samples (analogous to paired-sample t-test)

    For each random resample, each paired-difference observation is randomly
    assigned a sign (+ or -), which is equivalent to randomly permuting each
    pair between samples (data1 vs data2).  The same stat is then computed on
    the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    ARGS
    data1       (...,n,...) ndarray. Data from one group to compare.
    data2       (...,n,...) ndarray. Data from a second group to compare.
                Must have same number of observations (trials) n as data1, but 
                otherwise arbitrary shape.
    
    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    d           Float. Hypothetical difference in means for null distribution.
                Default: 0

    stat        String | Callable. String specifier for statistic to resample:
                't'         : paired t-statistic [default]
                'mean'/'meandiff'  : mean of pair differences

                -or- Custom function to generate resampled statistic of interest.
                Should take single array argument (equal to differences between
                paired samples, with <axis> corresponding to trials/observations) 
                and return a scalar value for each independent data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,1,...) ndarray. p values from permutation test. Same size as
                data1/data2, except for <axis> reduced to a singleton.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data1/data2, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.6.1
    """
    if isinstance(stat,str) and (stat.lower == 'meandiff'): stat = 'mean'

    return one_sample_randomization_test(data1 - data2, axis=axis, mu=d, stat=stat,
                                         tail=tail, n_resamples=n_resamples,
                                         return_stats=return_stats, seed=seed, **kwargs)


def paired_sample_bootstrap_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                 n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate bootstrap test of whether any arbitrary statistic
    differs between paired samples (analogous to paired-sample t-test)

    Computes stat on each bootstrap resample, and subtracts off stat computed on
    observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.
    
    ARGS
    data1       (...,n,...) ndarray. Data from one group to compare.
    data2       (...,n,...) ndarray. Data from a second group to compare.
                Must have same number of observations (trials) n as data1

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    d           Float. Hypothetical difference in means for null distribution.
                Default: 0

    stat        String | Callable. String specifier for statistic to resample:
                't'         : paired t-statistic [default]
                'mean'/'meandiff'  : mean of pair differences

                -or- Custom function to generate resampled statistic of interest.
                Should take single array argument (equal to differences between
                paired samples with <axis> corresponding to trials/observations) 
                and return a scalar value for each independent data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,1,...) ndarray. p values from bootstrap test. Same size as
                data1/data2, except for <axis> reduced to a singleton.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data1/data2, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.6.1
    """
    if isinstance(stat,str) and (stat.lower == 'meandiff'): stat = 'mean'

    return one_sample_bootstrap_test(data1 - data2, axis=axis, mu=d, stat=stat,
                                     tail=tail, n_resamples=n_resamples,
                                     return_stats=return_stats, seed=seed, **kwargs)


# =============================================================================
# Two-sample randomization tests
# =============================================================================
def two_sample_test(data1, data2, axis=0, method='permutation', **kwargs):
    """
    Mass univariate two-sample test of whether any arbitrary statistic
    differs between two non-paired samples (analogous to 2-sample t-test)
    
    Wrapper around functions for specific two-sample tests. See those for details.
    
    ARGS
    data        (...,n,...) ndarray. Data to run test on.  Arbitrary shape.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    method      String. Resampling paradigm to use for test:
                'permutation'   : Permutation test in two_sample_permutation_test [default]
                'bootstrap'     : Bootstrap test in two_sample_bootstrap_test
                
    See specific test functions for further arguments.
    
    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for <axis> reduced to a singleton.
                For vector data, returns as a single float.
                
    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.    
    """
    method = method.lower()
    
    if method in ['randomization','permutation']:   test_func = two_sample_permutation_test
    elif method == 'bootstrap':                     test_func = two_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Should be 'permutation' or 'bootstrap'" % method)
    
    return test_func(data1, data2, axis=axis, **kwargs)


def two_sample_test_labels(data, labels, axis=0, method='permutation', groups=None, **kwargs):
    """
    Mass univariate two-sample test of whether any arbitrary statistic
    differs between two non-paired samples (analogous to 2-sample t-test)
    
    Wrapper around functions for specific two-sample tests. See those for details.
    
    This version is an alternative interface to two_sample_test() with (data,labels) 
    arguments instead of (data1,data2)
    
    ARGS
    data        (...,N,...) ndarray. Data from both groups to run test on. Arbitrary shape.
    
    labels      (N,) array-like. Group labels for each observation (trial),
                identifying which group/condition each observation belongs to.
    
    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    method      String. Resampling paradigm to use for test:
                'permutation'   : Permutation test in paired_sample_permutation_test [default]
                'bootstrap'     : Bootstrap test in paired_sample_bootstrap_test
                
    groups      (n_groups,) array-like. List of labels for each group (condition).
                Default: set of unique values in <labels> (np.unique(labels))
                                
    See specific test functions for further arguments.
    
    RETURNS
    p           (...,1,...) ndarray. p values from randomization test. Same size as data,
                except for (...,n_resamples,...) reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.    
    """
    labels = np.asarray(labels)
    if groups is None: groups = np.unique(labels)

    assert len(groups) == 2, "Must have only 2 groups (conditions) in labels"
    
    return two_sample_test(data.compress(labels == groups[0], axis=axis),
                           data.compress(labels == groups[1], axis=axis),
                           axis=axis, method=method, **kwargs)
    
    
def two_sample_permutation_test(data1, data2, axis=0, stat='t', tail='both',
                                n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate permutation test of whether any arbitrary statistic
    differs between two non-paired samples (analogous to 2-sample t-test)
    
    Observations are permuted across the two samples (data1 vs data2).
    The data is pooled across data1 and and data2, and for each random resample,
    n1 and n2 observations are extracted from the pooled and form the resampled
    versions of data1 and data2, respectively.  The same stat is then computed
    on the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    ARGS
    data1       (...,n1,...) ndarray. Data from one group to compare.
    data2       (...,n2,...) ndarray. Data from a second group to compare.
                Need not have the same n as data1, but all other dim's must be
                same size/shape

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    stat        String | Callable. String specifier for statistic to resample:
                't'         : 2-sample t-statistic [default]
                'meandiff'  : group difference in across-observation means

                -or- Custom function to generate resampled statistic of interest.
                Should take two array arguments (data1,data2) with <axis> corresponding
                to trials/observations and return a scalar value for each independent 
                data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,1,...) ndarray. p values from permutation test. Same size as
                data1/data2, except for <axis> reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs     (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data1/data2, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.6.3
    """    
    # Convert string specifiers to callable functions
    stat_func    = _str_to_two_sample_stat(stat)    # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    # Reshape data so axis 0 is observations
    if axis != 0: 
        data1 = np.moveaxis(data1.copy(), axis, 0)
        data2 = np.moveaxis(data2.copy(), axis, 0)
    
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    N  = n1+n2

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1,data2, **kwargs)

    # Pool two samples together into combined data
    data_pool = np.concatenate((data1,data2),axis=0)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_resmp = np.empty((n_resamples-1,*data1.shape[1:]))
        for i_resmp,resample in enumerate(resamples):
            # First n1 permuted indexes are resampled "data1"
            data1_resmp  = data_pool[resample[0:n1],...]
            # Remaining n2 permuted indexes are resampled "data2"
            data2_resmp  = data_pool[resample[n1:],...]
            # Compute statistic on resampled data1,data2
            stat_resmp[i_resmp,...] = stat_func(data1_resmp,data2_resmp, **kwargs)

        # p value = proportion of permutation-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, 0, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((1,*data1.shape[1:]))
        for resample in resamples:
            # First n1 permuted indexes are resampled "data1"
            data1_resmp   = data_pool[resample[0:n1],...]
            # Remaining n2 permuted indexes are resampled "data2"
            data2_resmp   = data_pool[resample[n1:],...]
            # Compute statistic on resampled data1,data2
            stat_resmp    = stat_func(data1_resmp,data2_resmp, **kwargs)
            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples

    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: stat_obs = stat_obs.item()
        
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


def two_sample_bootstrap_test(data1, data2, axis=0, stat='t', tail='both',
                              n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate bootstrap test of whether any arbitrary statistic
    differs between two non-paired samples (analogous to 2-sample t-test)

    Computes stat on each pair of bootstrap resamples, and subtracts off stat computed
    on observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.
    
    ARGS
    data1       (...,n1,...) ndarray. Data from one group to compare.
    data2       (...,n2,...) ndarray. Data from a second group to compare.
                Need not have the same n as data1, but all other dim's must be
                same size/shape

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    stat        String | Callable. String specifier for statistic to resample:
                't'         : 2-sample t-statistic [default]
                'meandiff'  : group difference in across-observation means

                -or- Custom function to generate resampled statistic of interest.
                Should take two array arguments (data1,data2) with <axis> corresponding
                to trials/observations and return a scalar value for each independent 
                data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Default: 'both' (2-tailed test)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function

    RETURNS
    p           (...,1,...) ndarray. p values from bootstrap test. Same size as
                data1/data2, except for <axis> reduced to a singleton.
                For vector data, returns as a single float.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data1/data2, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.3.10, 6.3
    """
    # Convert string specifiers to callable functions
    stat_func    = _str_to_two_sample_stat(stat) # Statistic (includes default)
    compare_func = _tail_to_compare(tail)        # Tail-specific comparator

    # Reshape data so axis 0 is observations
    if axis != 0: 
        data1 = np.moveaxis(data1.copy(), axis, 0)
        data2 = np.moveaxis(data2.copy(), axis, 0)
    
    n1 = data1.shape[0]
    n2 = data2.shape[0]

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1,data2, **kwargs)

    # Create generators with n_resamples-1 random resamplings with replacement
    # of ints 0:n1-1 and 0:n2-1
    # Note: Seed random number generator only *once* before generating both random samples
    if seed is not None: set_random_seed(seed)    
    resamples1 = bootstraps(n1,n_resamples-1)
    resamples2 = bootstraps(n2,n_resamples-1)

    if return_stats:
        stat_resmp = np.empty((n_resamples-1,*data1.shape[1:]))

        # Iterate thru <n_resamples> random resamplings, recomputing statistic on each
        for i_resmp,(resample1,resample2) in enumerate(zip(resamples1,resamples2)):
            stat_resmp[i_resmp,...] = stat_func(data1[resample1,...],
                                                data2[resample2,...], **kwargs)

        # Subtract observed statistic from distribution of resampled statistics
        stat_resmp -= stat_obs

        # p value = proportion of permutation-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, 0, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((1,*data1.shape[1:]))
        for resample1,resample2 in zip(resamples1,resamples2):
            # Compute statistic on resampled data
            stat_resmp = stat_func(data1[resample1,...],
                                   data2[resample2,...], **kwargs)
            # Subtract observed statistic from distribution of resampled statistics
            stat_resmp -= stat_obs

            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples

    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: stat_obs = stat_obs.item()
        
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


# =============================================================================
# One-way/Two-way randomization tests
# =============================================================================
def one_way_test(data, labels, axis=0, method='permutation', **kwargs):
    """
    Mass univariate permutation test on any arbitrary 1-way statistic with
    multiple groups/levels (analogous to F-test in a 1-way ANOVA)

    Wrapper around functions for specific one-way tests. See those for details.

    ARGS
    data        (...,n,...) ndarray. Data to run test on. Arbitrary shape.

    labels      (n,) array-like. Group labels for each observation (trial),
                identifying which group/factor level each observation belongs to.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    method      String. Type of test to run. Currently only 'permutation' supported.
                
    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,1,...) ndarray. p values from permutation test. Same size as data,
                except for <axis> reduced to a singleton.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.
    """
    method = method.lower()
    
    if method in ['randomization','permutation','sign']:    test_func = one_way_permutation_test
    # elif method == 'bootstrap':                             test_func = one_way_sample_bootstrap_test
    else:
        raise ValueError("Only 'permutation' method currently supported")
    
    return test_func(data, labels, axis=axis, **kwargs)
    
    
def one_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate permutation test on any arbitrary 1-way statistic with
    multiple groups/levels (analogous to F-test in a 1-way ANOVA)

    Observation labels are randomly shuffled and allocated to groups
    with the same n's as in the actual observed data.

    ARGS
    data        (...,n,...) ndarray. Data to run test on.
                Axis 0 should correspond to distinct observations/trials.

    labels      (n,) array-like. Group labels for each observation (trial),
                identifying which group/factor level each observation belongs to.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    stat        String | Callable. String specifier for statistic to resample:
                'F'         : F-statistic (as in 1-way ANOVA) [default]

                -or- Custom function to generate resampled statistic of interest.
                Should take data array (data) with <axis> corresponding to
                trials/observations and labels arguments (labels) and return
                a scalar value for each independent data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Note: For F-test, only right-tailed test really makes sense bc F distn
                only has positive values and right-sided taile
                Default: 'right' (1-tailed test)

    groups      (n_groups,) array-like. List of labels for each group/level.
                Default: set of unique values in <labels> (np.unique(labels))

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,1,...) ndarray. p values from permutation test. Same size as data,
                except for <axis> reduced to a singleton.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_resamples-1,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but <axis> has length n_resamples-1.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.7.1
    """
    if (stat == 'F') and (tail != 'right'):
        warn("For F-test, only right-tailed tests make sense (tail set = %s in args)" % tail)
        
    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_way_stat(stat)     # Statistic (includes default)
    compare_func = _tail_to_compare(tail)       # Tail-specific comparator

    labels = np.asarray(labels)
    # Reshape data so axis 0 is observations
    if axis != 0: data = np.moveaxis(data.copy(), axis, 0)
    N = data.shape[0]

    # Find set of unique group labels in list of labels (if not given)
    if groups is None:
        groups = np.unique(labels)

    # If groups set in args, remove any observations not represented in <groups>
    else:
        idxs = np.in1d(labels,groups)
        if idxs.sum() != N:
            labels  = labels[idxs]
            data    = data[idxs,...]
            N       = data.shape[0]
            
    # Append <groups> to stat_func args        
    if stat_func == one_way_fstat: kwargs.update({'groups':groups})     
        
    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data,labels, **kwargs)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_resmp = np.empty((n_resamples-1,*data.shape[1:]))
        for i_resmp,resample in enumerate(resamples):
            # Compute statistic on data and resampled labels
            # Note: Only need to resample labels
            stat_resmp[i_resmp,...] = stat_func(data,labels[resample], **kwargs)

        # p value = proportion of permutation-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, 0, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((1,*data.shape[1:]))
        for resample in resamples:
            # Compute statistic on data and resampled labels
            # Note: Only need to resample labels
            stat_resmp = stat_func(data,labels[resample], **kwargs)
            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples

    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: stat_obs = stat_obs.item()
        
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


def two_way_test(data, labels, axis=0, method='permutation', **kwargs):
    """
    Mass univariate permutation test on any arbitrary 2-way statistic with
    multiple groups/levels (analogous to F-test in a 2-way ANOVA)

    Wrapper around functions for specific two-way tests. See those for details.

    ARGS
    data        (...,n,...) ndarray. Data to run test on.
                Axis 0 should correspond to distinct observations/trials.

    labels      (n,n_terms=2|3) array-like. Group labels for each observation,
                identifying which group/factor level each observation belongs to,
                for each model term. First two columns correspond to model main
                effects; optional third column corresponds to interaction term.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
 
    method      String. Type of test to run. Currently only 'permutation' supported.

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,n_terms,...) ndarray. p values from permutation test. Same size
                as data, except for <axis> reduced to length <n_terms> (2 or 3).

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,n_terms,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_terms,...,n_resamples-1) ndarray. Distribution of statistic
                values for all resamplings of data. Same size as data, but <axis> 
                has length <n_terms>, and a new axis of length n_resamples-1 is 
                appended to end of array.
                NOTE: axis for resamples is different from all other functions bc
                we need to accomodate both resample and terms dimensions here.

    """
    method = method.lower()
    
    if method in ['randomization','permutation','sign']:    test_func = two_way_permutation_test
    # elif method == 'bootstrap':                             test_func = two_way_sample_bootstrap_test
    else:
        raise ValueError("Only 'permutation' method currently supported")
    
    return test_func(data, labels, axis=axis, **kwargs)
    

def two_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False, **kwargs):
    """
    Mass univariate permutation test on any arbitrary 2-way statistic with
    multiple groups/levels (analogous to F-test in a 2-way ANOVA)

    Observation labels are randomly shuffled and allocated to groups (ANOVA cells)
    with the same n's as in the actual observed data.

    ARGS
    data        (...,n,...) ndarray. Data to run test on.
                Axis 0 should correspond to distinct observations/trials.

    labels      (n,n_terms=2|3) array-like. Group labels for each observation,
                identifying which group/factor level each observation belongs to,
                for each model term. First two columns correspond to model main
                effects; optional third column corresponds to interaction term.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    stat        String | Callable. String specifier for statistic to resample:
                'F'         : F-statistic (as in 2-way ANOVA) [default]

                -or- Custom function to generate resampled statistic of interest.
                Should take data array (data) with <axis> corresponding to
                trials/observations and labels arguments (labels) and return
                a scalar value for each independent data series.

    tail        String. Specifies tail of test to perform:
                'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
                'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
                'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp
                Note: For F-test, only right-tailed test really makes sense bc F distn
                only has positive values and right-sided taile                
                Default: 'right' (1-tailed test)

    groups      [(n_groups(term),),] array-like. List of labels for each group/level,
                for each model term.
                Default: set of unique values in <labels> (np.unique(labels))

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns p values. If True, returns p values,
                observed stats, and resampled stats. Default: False

    Any additional kwargs passed directly to stat function.

    RETURNS
    p           (...,n_terms,...) ndarray. p values from permutation test. Same size
                as data, except for <axis> reduced to length <n_terms> (2 or 3).

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,n_terms,...) ndarray. Statistic values for actual observed data.
                Same size as <p>.

    stat_resmp  (...,n_terms,...,n_resamples-1) ndarray. Distribution of statistic
                values for all resamplings of data. Same size as data, but <axis> 
                has length <n_terms>, and a new axis of length n_resamples-1 is 
                appended to end of array.
                NOTE: axis for resamples is different from all other functions bc
                we need to accomodate both resample and terms dimensions here.

    REFERENCE
    Manly _Randomization, Bootstrap and Monte Carlo Methods in Biology_ ch.7.4

    Note: We resample the entire row of labels for each observation, which
        effectively shuffles observations between specific combinations
        of groups/factor levels (ANOVA "cells") cf. recommendation in Manly book
    """
    # TODO  Add mechanism to auto-generate interaction term from factors (if interact arg==True)
    if (stat == 'F') and (tail != 'right'):
        warn("For F-test, only right-tailed tests make sense (tail set = %s in args)" % tail)
    
    # Convert string specifier to callable function
    stat_func = _str_to_two_way_stat(stat)       # Statistic (includes default))
    compare_func = _tail_to_compare(tail)   # Tail-specific comparator

    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = len(data.shape) + axis
    
    labels  = np.asarray(labels)
    n_terms = labels.shape[1]
    # Reshape data so axis 0 is observations
    if axis != 0: data = np.moveaxis(data.copy(), axis, 0)    
    N       = data.shape[0]

    # Find all groups/levels in list of labels (if not given in inputs)
    if groups is None:        
        groups = [np.unique(labels[:,term]) for term in range(n_terms)] 
        
    # Append <groups> to stat_func args        
    if stat_func == two_way_fstat: kwargs.update({'groups':groups})

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data,labels, **kwargs)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # DEL? stat_obs = np.moveaxis(stat_obs, 0,-1)[np.newaxis,...]

        # Compute statistic under <n_resamples> random resamplings
        # DEL stat_resmp = np.empty((n_resamples-1,*data.shape[1:],n_terms))
        stat_resmp = np.empty((n_terms, *data.shape[1:],n_resamples-1))
        for i_resmp,resample in enumerate(resamples):
            # Compute statistic on data and resampled label rows
            stat_resmp[...,i_resmp] = stat_func(data,labels[resample,:], **kwargs)
            
            # DEL
            # # HACK Move terms to axis=-1 so gets treated as another independent
            # #  data series in resamples_to_pvalue() below
            # tmp = stat_func(data,labels[resample,:], **kwargs)
            # stat_resmp[i_resmp,...] = np.moveaxis(tmp, 0,-1)[np.newaxis,...]

        # p value = proportion of permutation-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs[...,np.newaxis], stat_resmp, -1, compare_func).squeeze(axis=-1)

        # DEL
        # p = np.squeeze(np.swapaxes(p, 0,-1), -1)
        # stat_obs    = np.squeeze(np.swapaxes(stat_obs, 0,-1), -1)
        # stat_resmp  = np.swapaxes(stat_resmp, 0,-1)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        count = np.ones((n_terms,*data.shape[1:]))
        for resample in resamples:
            # Compute statistic on data and resampled label rows
            stat_resmp = stat_func(data,labels[resample,:], **kwargs)
            # Compute statistic on resampled data and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of permutation-resampled test statistic values
        p = count / n_resamples

    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        p = np.moveaxis(p, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)            
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)
            
    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p
    
    
# =============================================================================
# Confidence intervals
# =============================================================================
def bootstrap_confints(data, axis=0, stat='mean', confint=0.95, n_resamples=10000, 
                       seed=None, return_stats=False, return_sorted=True, **kwargs):
    """
    Mass univariate bootstrap confidence intervals of any arbitrary 1-sample stat 
    (eg mean).  Analogous to SEM/parametric confidence intervals.
        
    ARGS
    data        (...,n,...) ndarray. Data to run test on.  Axis 0 should correspond
                to distinct observations/trials.

    axis        Int. Axis of data corresponding to distinct trials/observations.
                Default: 0 (1st axis)
                
    stat        String | Callable. String specifier for statistic to resample:
                'mean'  : mean across observations [default]

                -or- Custom function to generate resampled statistic of interest.
                Should take single array argument (data) with <axis> corresponding
                to trials/observations and return a scalar value for each independent 
                data series.

    confint     Float. Confidence interval to compute, expressed as decimal value in range 0-1.
                Typical values are 0.68 (to approximate SEM), 0.95 (95% confint), and 0.99 (99%)
                Default: 0.95 (95% confidence interval)

    n_resamples Int. Number of random resamplings to perform for test
                (should usually be >= 10000 if feasible). Default: 10000

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.

    return_stats Bool. If False, only returns confidence intervals. If True, returns confints,
                statistic computed on observed data, and full distribution of resample statistic.
                Default: False
    
    return_sorted   Bool. If True [default], returns stat_resmp sorted by value. If False, returns 
                stat_resmp unsorted (ordered by resample number), which is useful if you want 
                to keep each resampling for all mass-univariate data series's together.
                
    - Any additional kwargs passed directly to stat function -
    
    RETURNS
    confints    (...,2,...) ndarray. Computed bootstrap confidence intervals. Same size as data,
                except for axis 0 reduced to length 2 = [lower,upper] confidence interval.

    - Following variables are only returned if return_stats is True -
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data.
                Same size as data, but axis 0 has length 1.        

    stat_resmp (...,n_resamples,...) ndarray. Distribution of statistic values
                for all resamplings of data.
                Same size as data, but axis 0 has length n_resamples.        
    """    
    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat)
    
    # Indexes into set of bootstrap resamples corresponding to given confint
    conf_indexes = confint_to_indexes(confint, n_resamples)

    # Reshape data so axis 0 is observations
    if axis != 0: data = np.moveaxis(data, axis, 0)
    n = data.shape[0]
    
    # Compute statistic of interest on actual observed data
    if return_stats: stat_obs = stat_func(data, **kwargs)    

    # Create generators with n_resamples length-n independent Bernoulli RV's
    resamples = bootstraps(n, n_resamples, seed)

    # Compute statistic under <n_resamples> random resamplings
    stat_resmp = np.empty((n_resamples,*data.shape[1:]))
    for i_resmp,resample in enumerate(resamples):
        # Compute statistic on resampled data
        stat_resmp[i_resmp,...] = stat_func(data[resample,...], **kwargs)
       
    if return_stats and not return_sorted:
        # Sort copy of stat_resmp, to return original unsorted version
        stat_resmp_sorted = stat_resmp.sort(axis=0)
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp_sorted[conf_indexes,...]
        
    else:
        stat_resmp.sort(axis=0)     # Sort resample stats in place
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp[conf_indexes,...]
        
    # Reshape output so observation axis is in appropriate location
    if axis != 0:
        confints = np.moveaxis(confints, 0, axis)
        if return_stats:
            stat_obs    = np.moveaxis(stat_obs, 0, axis)
            stat_resmp  = np.moveaxis(stat_resmp, 0, axis)
        
    # For vector-valued data, extract value from scalar array -> float for output
    if return_stats and (stat_obs.size == 1): stat_obs = stat_obs.item()
                
    if return_stats:    return confints, stat_obs, stat_resmp
    else:               return confints
        
                                                  
#==============================================================================
# Functions to compute statistics
#==============================================================================
def one_sample_tstat(data, axis=0, mu=0):
    """
    Mass univariate 1-sample t-statistic, relative to null mean <mu> on data

    ARGS
    data    (...,n,...) ndarray. Data to compute stat on.  Axis <axis>
            should correspond to distinct observations/trials; other axes
            are treated as independent data series, and stat is computed
            separately for each

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    mu      Float. Expected mean under the null hypothesis. Default: 0

    RETURNS
    t       (...,1,...) ndarray. 1-sample t-statistic for given data.
            t = (mean(data) - mu) / SEM(data)

    Note: In timeit tests, ran ~2x as fast as scipy.stats.ttest_1samp
    """
    data = np.asarray(data)

    n   = data.shape[axis]

    if mu != 0:  data = data - mu

    # Compute mean and unbiased standard deviation of data
    mu  = data.mean(axis=axis,keepdims=True)
    sd  = data.std(axis=axis,ddof=1,keepdims=True)

    # Return t statistic = mean/SEM
    return mu / (sd/sqrt(n))


def paired_tstat(data1, data2, axis=0, d=0):
    """
    Mass univariate paired-sample t-statistic, relative to null mean difference <d> on data

    ARGS
    data1   (...,n,...) ndarray. Data from one group to compare.
    data2   (...,n,...) ndarray. Data from a second group to compare.
            Must have same number of observations (trials) n as data1

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    d       Float. Hypothetical difference in means for null distribution.
            Default: 0

    RETURNS
    t       (...,1,...) ndarray. Paired-sample t-statistic for given data.
            dData = data1 - data2
            t = (mean(dData) - d) / SEM(dData)
    """
    return one_sample_tstat(data1 - data2, axis=axis, mu=d)


def two_sample_tstat(data1, data2, axis=0, equal_var=True, d=0):
    """
    Mass univariate 2-sample t-statistic, relative to null mean difference <d> on data

    ARGS
    data1   (...,n1,...) ndarray. Data from one group to compare.
    data2   (...,n2,...) ndarray. Data from a second group to compare.
            Need not have the same n as data1, but all other dim's must be
            same size/shape. For both, axis <axis> should correspond to
            distinct observations/trials; other axes are treated as
            independent data series, and stat is computed separately for each.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    equal_var Bool. If True [default], performs a standard independent
            2-sample test that assumes equal population variances for 2 groups.
            If False, perform Welch’s t-test, which does not assume equal
            population variances.

    d       Float. Hypothetical difference in means for null distribution.
            Default: 0

    RETURNS
    t       (...,1,...) ndarray. 2-sample t-statistic for given data.
            t = (mean(data1) - mean(data2) - mu) / pooledSE(data1,data2)
            (where the formula for pooled SE differs depending on <equal_var>)

    REFERENCES
    Indep t-test : en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test
    Welch's test : en.wikipedia.org/wiki/Welch%27s_t-test

    Note: In timeit tests, ran ~2x as fast as scipy.stats.ttest_ind
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    n1  = data1.shape[axis]
    n2  = data2.shape[axis]

    # Compute mean of each group and their difference (offset by null mean)
    d   = data1.mean(axis=axis,keepdims=True) - \
          data2.mean(axis=axis,keepdims=True) - d

    # Compute variance of each group
    var1 = data1.var(axis=axis,ddof=1,keepdims=True)
    var2 = data2.var(axis=axis,ddof=1,keepdims=True)

    # Standard independent 2-sample t-test (assumes homoscedasticity)
    if equal_var:
        # Compute pooled standard deviation across data1,data2 -> standard error
        df1         = n1 - 1
        df2         = n2 - 1
        sd_pooled   = np.sqrt((var1*df1 + var2*df2) / (df1+df2))
        se          = sd_pooled * sqrt(1/n1 + 1/n2)

    # Welch's test (no homoscedasticity assumption)
    else:
        se      = np.sqrt(var1/n1 + var2/n2)

    # Return t statistic = difference in means / pooled standard error
    return d / se


def one_way_fstat(data, labels, axis=0, groups=None):
    """
    Mass univariate 1-way F-statistic on given data and labels

    ARGS
    data    (...,n,...) ndarray. Data to compute stat on.  Axis <axis>
            should correspond to distinct observations/trials; other axes
            are treated as independent data series, and stat is computed
            separately for each

    labels  (n,) array-like. Group labels for each observation (trial),
            identifying which group/factor level each observation belongs to.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    groups  (n_groups,) array-like. List of labels for each group/level.
            Default: set of unique values in <labels> (np.unique(labels))

    RETURNS
    F       (...,1,...) ndarray. F-statistic for given data.
            F = var(between groups) / var(within groups)

    Note: In timeit tests, this code ran slightly faster than scipy.stats.f_oneway
    and can handle arrays of arbitrary shape.
    """
    labels  = np.asarray(labels)
    # Find all groups/levels in list of labels (if not given)
    if groups is None: groups = np.unique(labels)

    data_shape = data.shape
    n = data_shape[axis]

    SS_shape = list(data_shape)
    SS_shape[axis] = 1

    # Compute grand mean across all observations (for each data series)
    grand_mean = data.mean(axis=axis,keepdims=True)

    # Total Sums of Squares
    SS_total = ((data - grand_mean)**2).sum(axis=axis,keepdims=True)

    # Groups (between-group) Sums of Squares
    SS_groups = np.zeros(SS_shape)
    for group in groups:
        group_bool = labels == group
        # Number of observations for given group
        n = group_bool.sum()
        # Group mean for given group
        group_mean = data.compress(group_bool,axis=axis).mean(axis=axis,keepdims=True)
        # Groups Sums of Squares for given group
        SS_groups += n*(group_mean - grand_mean)**2

    # Error (within-group) Sums of Squares
    SS_error = SS_total - SS_groups

    df_groups   = len(groups) - 1   # Groups degrees of freedom
    df_error    = n-1 - df_groups   # Error degrees of freedom

    return  (SS_groups/df_groups) / (SS_error/df_error)    # F statistic

    # DELETE
    # Reformat data for scipy function -- list with each group data,
    # observation axis=0
    # return f_oneway(*[(data.compress(labels == group,axis=axis)
    #                        .swapaxes(0,axis))
    #                   for group in groups])[0].swapaxes(0,axis)


def two_way_fstat(data, labels, axis=0, groups=None):
    """
    Mass univariate 2-way (with interaction) F-statistic on given data and labels

    ARGS
    data    (...,n,...) ndarray. Data to compute stat on.  Axis <axis>
            should correspond to distinct observations/trials; other axes
            are treated as independent data series, and stat is computed
            separately for each

    labels  (n,n_terms=2|3) array-like. Group labels for each model term and
            observation (trial), identifying which group/factor level each
            observation belongs to for each term. First 2 columns should reflect
            main effects, and optional third column should be their interaction.

    axis    Int. Data axis corresponding to distinct observations. Default: 0

    groups  [(n_groups(term),),] array-like. List of labels for each group/level,
            for each model term.
            Default: set of unique values in <labels> (np.unique(labels))

    RETURNS
    F       (...,n_terms,...) ndarray. F-statistics for given data and terms.
            F = var(between groups) / var(within groups)

    REFERENCE
    Zar _Biostatistical Analysis_ ch.12

    Note:   In timeit tests, this code ran much faster than ols and anova_lm
            from statsmodels, and can run multiple data series at once
    """
    labels  = np.asarray(labels)
    n_terms  = labels.shape[1]
    doInteract = n_terms == 3
    # Find all groups/levels in list of labels (if not given)
    if groups is None:
        groups = [np.unique(labels[:,term]) for term in range(n_terms)] 
    n_groups = np.asarray([len(termGroups) for termGroups in groups])

    data_shape = data.shape
    n = data_shape[axis]

    SS_shape = list(data_shape)
    SS_shape[axis] = 1

    # Compute grand mean across all observations (for each data series)
    grand_mean = data.mean(axis=axis,keepdims=True)

    # Total Sums of Squares
    SS_total = ((data - grand_mean)**2).sum(axis=axis,keepdims=True)

    # Groups (between-group) Sums of Squares for each term
    SS_groups = []
    for term in range(n_terms):
        SS_groups.append( np.zeros(SS_shape) )

        for group in groups[term]:
            group_bool = labels[:,term] == group
            # Number of observations for given group
            n = group_bool.sum()
            # Group mean for given group
            group_mean = data.compress(group_bool,axis=axis).mean(axis=axis,keepdims=True)
            # Groups Sums of Squares for given group
            SS_groups[term] += n*(group_mean - grand_mean)**2

        # For interaction term, calculations above give Cells Sum of Squares (Zar eqn. 12.18)
        # Interaction term Sum of Squares = SScells - SS1 - SS2 (Zar eqn. 12.12)
        if term == 2:
          SS_groups[term] -= (SS_groups[0] + SS_groups[1])

    SS_groups = np.concatenate(SS_groups,axis=axis)

    # Error (within-cells) Sums of Squares
    SS_error = SS_total - SS_groups.sum(axis=axis,keepdims=True)

    # Groups degrees of freedom (Zar eqn. 12.9)
    df_groups= n_groups - 1
    dfCells = df_groups[-1]      # Cells degrees of freedom (Zar eqn. 12.4)
    if doInteract:
        # Interaction term degrees of freedom = dfCells - dfMain1 - dfMain2 (Zar eqn. 12.13)
        df_groups[2] -= (df_groups[0] + df_groups[1])

    # Error degrees of freedom = dfTotal - dfCells (Zar eqn. 12.7)
    df_error = n - 1 - dfCells

    if axis != -1:
        df_groups = df_groups.reshape((*np.ones((axis,),dtype=int),
                                     n_terms,
                                     *np.ones((SS_groups.ndim-axis-1,),dtype=int)))

    return  (SS_groups/df_groups) / (SS_error/df_error)    # F statistic


# =============================================================================
# Random sample generators
# =============================================================================
def permutations(n, n_resamples=9999, seed=None):
    """
    Yields generator with a set of <n_resamples> random permutations of integers
    0:n-1, as would be needed for permutation/randomization tests

    resamples = permutations(n,n_resamples=9999,seed=None)
    
    ARGS
    n           Int. Number of items to randomly resample from.
                Will usually correspond to number of observations/trials

    n_resamples Int. Number of independent resamples to generate.
                Default: 9999 (appropriate number for test w/ 10,000 samples)

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.
                
    YIELDS
    resamples   (n_resamples,) generator of (n,) vector of ints. Each iteration
                contains a distinct random permutation of integers 0:n-1
    """
    if seed is not None: set_random_seed(seed)
    
    for _ in range(n_resamples):
        yield np.random.permutation(n)


def bootstraps(n, n_resamples=9999, seed=None):
    """
    Yields generator with a set of <n_resamples> random resamplings with
    replacement of integers 0:n-1, as would be needed for bootstrap tests or
    confidence intervals

    resamples = bootstraps(n,n_resamples=9999,seed=None)
    
    ARGS
    n           Int. Number of items to randomly resample from.
                Will usually correspond to number of observations/trials

    n_resamples Int. Number of independent resamples to generate.
                Default: 9999 (appropriate number for test w/ 10,000 samples)

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.
                
    YIELDS
    resamples   (n_resamples,) generator of (n,) vector of ints. Each iteration
                contains a distinct random resampling with replacemnt from
                integers 0:n-1
    """
    if seed is not None: set_random_seed(seed)
    
    for _ in range(n_resamples):
        yield np.random.randint(n, size=(n,))


def signs(n, n_resamples=9999, seed=None):
    """
    Yields generator with a set of <n_resamples> random Bernoulli(p=0.5)
    variables (ie binary 0/1 w/ probability of 0.5), each of length <n>,
    as would ne needed to set the signs of stats in a sign test.

    resamples = signs(n,n_resamples=9999,seed=None)
    
    ARGS
    n           Int. Number of items to randomly resample from.
                Will usually correspond to number of observations/trials

    n_resamples Int. Number of independent resamples to generate.
                Default: 9999 (appropriate number for test w/ 10,000 samples)

    seed        Int. Random generator seed for repeatable results.
                Set=None [default] for unseeded random numbers.
                
    YIELDS
    resamples   (n_resamples,) generator of (n,) vector of bool. Each iteration
                contains a distinct random resampling of n Bernoulli binary RVs
    """
    if seed is not None: set_random_seed(seed)
    
    for _ in range(n_resamples):
        yield np.random.binomial(1,0.5, size=(n,)).astype(bool)

 
#==============================================================================
# Utility functions
#==============================================================================
def resamples_to_pvalue(stat_obs, stat_resmp, axis=0, tail='both'):
    """
    Computes p value from observed and resampled values of a statistic

    ARGS
    stat_obs    (...,1,...) ndarray. Statistic values for actual observed data

    stat_resmp  (...,n_resamples,...) ndarray. Statistic values for
                randomly resampled data

    axis        Int. Axis in stat_resmp corresponding to distinct resamples
                (should correspond to a singleton axis in stat_obs)
                Default: 0

    tail        String. Specifies tail of test to perform: 'both'|'right'|'left'

    RETURNS
    p           (...,1,...) ndarray. p values from permutation test. Same size as
                data1/data2, except for <axis> reduced to a singleton.
    """
    if callable(tail):  compare_func = tail
    else:               compare_func = _tail_to_compare(tail)

    n_resamples = stat_resmp.shape[axis]
    
    # Count number of resampled stat values more extreme than observed value
    p = np.sum(compare_func(stat_obs,stat_resmp), axis=axis, keepdims=True)

    # p value is proportion of samples failing criterion (+1 for observed stat)
    return (p + 1) / (n_resamples + 1)


def confint_to_indexes(confint, n_resamples):
    """
    Returns indexes into set of bootstrap resamples corresponding 
    to given confidence interval
    
    conf_indexes = confint_to_indexes(confint, n_resamples)
    
    ARGS
    confint         Float. Desired confidence interval, in range 0-1.
                    eg, for 99% confidence interval, input 0.99
                    
    n_resamples     Int. Number of bootstrap resamples.
    
    RETURNS
    conf_indexes    (2,) list of int. Indexes into sorted bootstrap
                    resamples corresponding to [lower,upper] confidence
                    interval.
    """
    max_interval = 1 - 2.0/n_resamples
    assert (confint <= max_interval) or np.isclose(confint,max_interval), \
        ValueError("Requested confidence interval is too large for given number of resamples (max = %.3f)" % max_interval)
        
    return [round(n_resamples * (1-confint)/2) - 1,
            round(n_resamples - (n_resamples * (1-confint)/2)) - 1]


#==============================================================================
# Helper functions
#==============================================================================
def _tail_to_compare(tail):
    """ Convert string specifier to callable function implementing it """

    assert isinstance(tail,str), \
        TypeError("Unsupported type '%s' for <tail>. Must be string specifier or function" % type(tail))
    assert tail in ['both','right','left'], \
        ValueError("Unsupported value '%s' for <tail>. Must be 'both', 'right', or 'left'" % tail)

    # 2-tailed test: hypothesis ~ stat_obs ~= statShuf
    if tail == 'both':
        return lambda stat_obs,stat_resmp: np.abs(stat_resmp) >= np.abs(stat_obs)
    
    # 1-tailed rightward test: hypothesis ~ stat_obs > statShuf
    elif tail == 'right':
        return lambda stat_obs,stat_resmp: stat_resmp >= stat_obs
    
    # 1-tailed leftward test: hypothesis ~ stat_obs < statShuf
    else: # tail == 'left':
        return lambda stat_obs,stat_resmp: stat_resmp <= stat_obs


def _str_to_one_sample_stat(stat):
    """ Convert string specifier to function to compute 1-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):                  return stat
    elif stat in ['t','tstat','t1']:    return one_sample_tstat
    elif stat == 'mean':                return lambda data: data.mean(axis=0,keepdims=True)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_sample_stat(stat):
    """ Convert string specifier to function to compute 2-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):                  return stat
    elif stat in ['t','tstat','t1']:    return two_sample_tstat
    elif stat == ['meandiff','mean']:
        return lambda data1,data2: data1.mean(axis=0,keepdims=True) - data2.mean(axis=0,keepdims=True)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_one_way_stat(stat):
    """ Convert string specifier to function to compute 1-way multi-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):                  return stat
    elif stat in ['f','fstat','f1']:    return one_way_fstat
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_way_stat(stat):
    """ Convert string specifier to function to compute 2-way multi-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):                  return stat
    elif stat in ['f','fstat','f2']:    return two_way_fstat
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)
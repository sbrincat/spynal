# -*- coding: utf-8 -*-
"""
Nonparametric randomization, permutation, and bootstrap statistics

Overview
--------
Functionality for hypothesis significance testing and confidence interval computation based on
random resampling of the observed data. This uses the data itself to generate an expected
distribution for the null hypothesis, and does not rely on any specific assumptions about
the form of the data distribution(s).

Users interested in *non-randomization* (eg rank-based) nonparametric methods are encouraged
to look at the statsmodels package (https://www.statsmodels.org/). Users interested in traditional
parametric methods are encouraged to look at the scipy.stats submodule
(https://docs.scipy.org/doc/scipy/reference/stats.html).

Includes tests/confints for several common data schemes:

- one-sample (are data values different from 0 or baseline?)
- paired-sample difference (are paired data observations different?)
- paired-sample association (are paired data observations correlated?)
- two-sample difference (are two groups/conditions of data different?)
- one-way difference (is there some difference between multiple groups/conditions of data?)
- two-way analysis (for data varying along 2 dims, are there diff's along each and/or interaction?)

Most significance tests include options for either permutation (shuffle) or bootstrap methods.

All functions can compute tests/confints based on random resampling of a default statistic typical
for given data scheme (eg t-statistic, F-statistic) or on any custom user-input statistic.

For data not conforming to the above schemes, there is also direct access to low-level functions
for generating samples for permutation or bootstraps in your own code.

Most functions perform operations in a mass-univariate manner. This means that
rather than embedding function calls in for loops over channels, timepoints, etc., like this::

    for channel in channels:
        for timepoint in timepoints:
            results[timepoint,channel] = compute_something(data[timepoint,channel])

You can instead execute a single call on ALL the data, labeling the relevant axis
for the computation (usually trials/observations here), and it will run in parallel (vectorized)
across all channels, timepoints, etc. in the data, like this:

``results = compute_something(data, axis)``


Function list
-------------
Hypothesis tests
^^^^^^^^^^^^^^^^
- one_sample_test :             Random-sign/bootstrap 1-sample tests (~ 1-sample t-test)
- paired_sample_test :          Permutation/bstrap paired-sample difference tests (~ paired t-test)
- paired_sample_test_labels :   Same, but with (data,labels) arg format instead of (data1,data2)
- paired_sample_association_test : Perm/bstrap paired-sample association tests (~ correlation)
- paired_sample_association_test_labels : Same, but with (data,labels) arg format
- two_sample_test :             Permutation/bootstrap for all 2-sample tests (~ 2-sample t-test)
- two_sample_test_labels :      Same, but with (data,labels) arg format instead of (data1,data2)
- one_way_test :                Permutation 1-way multi-level test (~ 1-way ANOVA/F-test)
- two_way_test :                Perm 2-way multi-level/multi-factor test (~ 2-way ANOVA/F-test)

Confidence intervals
^^^^^^^^^^^^^^^^^^^^
- one_sample_confints :         Bootstrap confidence intervals for any one-sample stat
- paired_sample_confints :      Bootstrap confidence intervals for any paired-sample stat
- two_sample_confints :         Bootstrap confidence intervals for any two-sample stat

Random-sample generators
^^^^^^^^^^^^^^^^^^^^^^^^
- permutations :                Generate random permutations (resampling w/o replacement)
- bootstraps :                  Generate random bootstrap samples (resampling w/ replacement)
- signs :                       Generate random binary variables (eg for sign tests)
- jackknifes :                  Generate jackknife samples (exclude each observation in turn)

Function reference
------------------
"""
# Created on Tue Jul 30 16:28:12 2019
#
# @author: sbrincat

# TODO  Parallelize resampling loops! (using joblib?)
# TODO  How to pass custom function to paired_sample tests (eval'd as 1-samples)?
# TODO  How to enforce dimensionality of custom functions (esp re: keepdims)?

from math import sqrt
from warnings import warn
import numpy as np

from spynal.utils import set_random_seed, axis_index_slices, data_labels_to_data_groups, \
                         one_sample_tstat, paired_tstat, two_sample_tstat, \
                         one_way_fstat, two_way_fstat, correlation, rank_correlation, isarraylike


# =============================================================================
# One-sample randomization tests
# =============================================================================
def one_sample_test(data, axis=0, method='randomization', mu=0, stat='t', tail='both',
                    n_resamples=10000, seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate 1-sample test of whether any arbitrary 1-sample stat (eg mean)
    is different from a given value `mu`, often 0 (analogous to 1-sample t-test).

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data to run test on

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str
        Resampling paradigm to use for test:

        - 'randomization' : Randomization sign test in :func:`one_sample_randomization_test`
        - 'bootstrap'     : Bootstrap test in :func:`one_sample_bootstrap_test`

    mu : float, default: 0
        Expected value of `stat` under the null hypothesis (usually 0)

    stat : str or callable, default: 't'
        Statistic to compute and resample. Can be given as a string specifier:

        - 't'     : 1-sample t-statistic
        - 'mean'  : mean across observations

        Or as a custom function to generate resampled statistic of interest.
        Should take single array argument (data) with `axis` corresponding
        to trials/observations, and return a scalar value (for each independent
        data series if multiple given).

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from output.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_resamples-1,...), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_resamples-1.

    Examples
    --------
    p = one_sample_test(data, return_stats=False)

    p, stat_obs, stat_resmp = one_sample_test(data, return_stats=True)
    """
    method = method.lower()

    if method in ['randomization','permutation','sign']:
        test_func = one_sample_randomization_test
    elif method == 'bootstrap':
        test_func = one_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Use 'randomization' or 'bootstrap'" % method)

    return test_func(data, axis=axis, mu=mu, stat=stat, tail=tail,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                     keepdims=keepdims, **kwargs)


def one_sample_randomization_test(data, axis=0, mu=0, stat='t', tail='both', n_resamples=10000,
                                  seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate 1-sample randomization test

    Parameters and returns are same as :func:`one_sample_test`

    For each random resample, each observation is randomly assigned a sign
    (+ or -), similar to a Fisher sign test.  The same stat is then computed on
    the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.6.2
    """
    # Copy data variable to avoid changing values in caller
    data = data.copy()

    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data.ndim + axis

    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat,axis)   # Statistic (includes default)
    compare_func = _tail_to_compare(tail)               # Tail-specific comparator

    n = data.shape[axis]
    ndim = data.ndim

    # Subtract hypothetical mean(s) 'mu' from data to center them there
    if mu != 0:  data -= mu

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, **kwargs)

    # Create generators with n_resamples-1 length-n independent Bernoulli RV's
    resamples = signs(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)
            # Randomly flip signs of obervations flagged by random variables above
            data[data_slices] *= -1
            # Compute statistic on resampled data
            stat_resmp[stat_slices] = stat_func(data, **kwargs)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            # Randomly flip signs of obervations flagged by random variables above
            data[data_slices] *= -1
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs, stat_func(data, **kwargs))

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isarraylike(stat_obs): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p

one_sample_permutation_test = one_sample_randomization_test
""" Alias of :func:`one_sample_randomization_test`. See there for details. """


def one_sample_bootstrap_test(data, axis=0, mu=0, stat='t', tail='both', n_resamples=10000,
                              seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate 1-sample bootstrap test

    Parameters and returns are same as :func:`one_sample_test`

    Computes stat on each bootstrap resample, and subtracts off stat computed on
    observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch. 3.10
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data.ndim + axis

    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat,axis)   # Statistic (includes default)
    compare_func = _tail_to_compare(tail)               # Tail-specific comparator

    ndim = data.ndim
    n = data.shape[axis]

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, **kwargs)

    # Create generators with n_resamples-1 random bootstrap resamplings with replacement
    resamples = bootstraps(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)
            # Compute statistic on resampled data
            stat_resmp[stat_slices] = stat_func(data[data_slices], **kwargs)

        # Center each resampled statistic on their mean to approximate null disttribution
        # TODO Need to determine which of these is more apropriate (and align ~return_stats option)
        stat_resmp -= stat_obs
        # stat_resmp -= stat_resmp.mean(axis=axis,keepdims=True)

        # p value = proportion of bootstrap-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs - mu, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            # Compute statistic on resampled data
            stat_resmp = stat_func(data[data_slices])

            # Subtract observed statistic from distribution of resampled statistics
            stat_resmp -= stat_obs

            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs - mu, stat_resmp, **kwargs)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isarraylike(stat_obs): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


# =============================================================================
# Paired-sample difference randomization tests
# =============================================================================
def paired_sample_test(data1, data2, axis=0, method='permutation', d=0, stat='t', tail='both',
                       n_resamples=10000, seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate paired-sample test of whether any arbitrary statistic (eg mean)
    differs between paired samples (analogous to paired-sample t-test)

    Parameters
    ----------
    data1,data2 : ndarray, shape=(...,n,...)
        Data from two groups to compare. Shape is arbitrary, but must be same for data1,2.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str
        Resampling paradigm to use for test:

        - 'permutation' : Permutation test in :func:`paired_sample_permutation_test`
        - 'bootstrap' : Bootstrap test in :func:`paired_sample_bootstrap_test`

    d : float, Default: 0
        Expected value of `stat` under null distribution (usually 0)

    stat : str or callable, default: 't'
        Statistic to compute and resample. Can be given as a string specifier:

        - 't' : paired t-statistic
        - 'mean'/'meandiff' : mean of pair differences

        Or as a custom function to generate resampled statistic of interest.
        Should take single array argument (equal to differences between
        paired samples, with `axis` corresponding to trials/observations)
        and return a scalar value for each independent data series.

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from output.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.
        
    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_resamples-1,...), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_resamples-1.
    """
    method = method.lower()

    if method in ['randomization','permutation','sign']:
        test_func = paired_sample_permutation_test
    elif method == 'bootstrap':
        test_func = paired_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Should be 'permutation' or 'bootstrap'"
                         % method)

    return test_func(data1, data2, axis=axis, d=d, stat=stat, tail=tail,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                     keepdims=keepdims, **kwargs)


def paired_sample_test_labels(data, labels, axis=0, method='permutation', groups=None, **kwargs):
    """
    Alternative interface to paired_sample_test() that allows arguments of form (data,labels)
    instead of (data1,data2)

    Only parameters differing from :func:`paired_sample_test` are described here.

    Parameters
    ----------
    data : ndarray, shape=(...,N,...).
        Data from *both* groups to run test on.
        Arbitrary shape, but both groups must have the same n (n1 = n2 = N/2).

    labels : array-like, shape=(N,)
        Group labels for each observation (trial), identifying which group/condition
        each observation belongs to.

    groups : array-like, shape=(n_groups,), optional, default: np.unique(labels)
        List of labels for each group (condition). Used to test only a subset (pair)
        of multi-value labels.
    """
    data1, data2 = data_labels_to_data_groups(data, labels, axis=axis, groups=groups, max_groups=2)
    return paired_sample_test(data1, data2, axis=axis, method=method, **kwargs)


def paired_sample_permutation_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                   n_resamples=10000, seed=None, return_stats=False,
                                   keepdims=True, **kwargs):
    """
    Mass univariate paired-sample permutation test

    Parameters and returns are same as :func:`paired_sample_test`

    For each random resample, each paired-difference observation is randomly
    assigned a sign (+ or -), which is equivalent to randomly permuting each
    pair between samples (data1 vs data2).  The same stat is then computed on
    the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.6.1
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _paired_sample_data_checks(data1, data2)

    if isinstance(stat,str) and (stat.lower == 'meandiff'): stat = 'mean'

    return one_sample_randomization_test(data1 - data2, axis=axis, mu=d, stat=stat,
                                         tail=tail, n_resamples=n_resamples,
                                         return_stats=return_stats, seed=seed,
                                         keepdims=keepdims, **kwargs)


def paired_sample_bootstrap_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                 n_resamples=10000, seed=None, return_stats=False,
                                 keepdims=True, **kwargs):
    """
    Mass univariate paired-sample bootstrap test

    Parameters and returns are same as :func:`paired_sample_test`

    Computes stat on each bootstrap resample, and subtracts off stat computed on
    observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.6.1
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _paired_sample_data_checks(data1, data2)

    if isinstance(stat,str) and (stat.lower == 'meandiff'): stat = 'mean'

    return one_sample_bootstrap_test(data1 - data2, axis=axis, mu=d, stat=stat,
                                     tail=tail, n_resamples=n_resamples,
                                     return_stats=return_stats, seed=seed,
                                     keepdims=keepdims, **kwargs)


# =============================================================================
# Paired-sample association (correlation) randomization tests
# =============================================================================
def paired_sample_association_test(data1, data2, axis=0, method='permutation', stat='r',
                                   tail='both', n_resamples=10000, seed=None, return_stats=False,
                                   keepdims=True, **kwargs):
    """
    Mass bivariate test of association (eg correlation) between two paired samples

    Parameters
    ----------
    data1,data2 : ndarray, shape=(...,n,...)
        Data from two groups to compare. Shape is arbitrary, but must be same for data1,2.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str, default: 'permutation'
        Resampling paradigm to use for test:

        - 'permutation'   : Permutation test in :func:`paired_sample_association_permutation_test`
        - 'bootstrap'     : Bootstrap test in :func:`paired_sample_association_bootstrap_test`

    stat : str or callable, default: 'r'
        Statistic to compute and resample. Can be given as a string specifier:

        - 'r'/'pearson'   : Standard Pearson product-moment correlation
        - 'rho'/'spearman': Spearman rank correlation

        Or as a custom function to generate resampled statistic of interest.
        Should take two array arguments (data1,data2) with `axis` corresponding
        to trials/observations and return a scalar value for each independent
        data series.

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from output.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.
        
    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_resamples-1,...), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_resamples-1.
    """
    method = method.lower()

    if method in ['randomization','permutation','sign']:
        test_func = paired_sample_association_permutation_test
    elif method == 'bootstrap':
        test_func = paired_sample_association_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Should be 'permutation' or 'bootstrap'"
                         % method)

    return test_func(data1, data2, axis=axis, stat=stat, tail=tail,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                     keepdims=keepdims, **kwargs)


def paired_sample_association_test_labels(data, labels, axis=0, method='permutation', groups=None,
                                          **kwargs):
    """
    Alternative interface to paired_sample_association_test() that allows arguments of form
    (data,labels) instead of (data1,data2)

    Only parameters differing from :func:`paired_sample_association_test` are described here.

    Parameters
    ----------
    data : ndarray, shape=(...,N,...).
        Data from *both* groups to run test on.
        Arbitrary shape, but both groups must have the same n (n1 = n2 = N/2).

    labels : array-like, shape=(N,)
        Group labels for each observation (trial), identifying which group/condition
        each observation belongs to.

    groups : array-like, shape=(n_groups,), optional, default: np.unique(labels)
        List of labels for each group (condition). Used to test only a subset (pair)
        of multi-value labels.
    """
    data1, data2 = data_labels_to_data_groups(data, labels, axis=axis, groups=groups, max_groups=2)
    return paired_sample_association_test(data1, data2, axis=axis, method=method, **kwargs)


def paired_sample_association_permutation_test(data1, data2, axis=0, stat='r', tail='both',
                                               n_resamples=10000, seed=None, return_stats=False,
                                               keepdims=True, **kwargs):
    """
    Mass bivariate permutation test of association (eg correlation) between two paired samples

    Parameters and returns are same as :func:`paired_sample_association_test`

    Observations are randomly permuted across one of the paired samples (data1 vs data2) relative
    to the other, to eliminate any association between them while preserving the marginal
    distributions of each sample.  The same stat is then computed on the resampled data to estimate
    the distrubution under the null hypothesis, and the stat value for the actual observed data
    is compared to this.
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _paired_sample_data_checks(data1, data2)

    # Convert string specifiers to callable functions
    stat_func    = _str_to_assoc_stat(stat,axis)    # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    ndim = data1.ndim
    n = data1.shape[axis]

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1, data2, **kwargs)

    # Create generators with n_resamples-1 random permutations of ints 0:n-1
    resamples = permutations(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data1.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)
            # Compute statistic on resampled data. Only need to permute one sample
            stat_resmp[stat_slices] = stat_func(data1, data2[data_slices], **kwargs)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data1.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Index into <axis> of data, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            # Compute statistic on resampled data1,data2
            stat_resmp  = stat_func(data1, data2[data_slices], **kwargs)
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs, stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isinstance(stat_obs,np.ndarray): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


def paired_sample_association_bootstrap_test(data1, data2, axis=0, stat='r', tail='both',
                                             n_resamples=10000, seed=None, return_stats=False,
                                             keepdims=True, **kwargs):
    """
    Mass bivariate boostrap test of association (eg correlation) between two paired samples

    Parameters and returns are same as :func:`paired_sample_association_test`

    Observations are bootstrap resampled in pairs, and stat is recomputed on each.
    Stat computed on observed data is subtracted off resamples, to center them at 0 and estimate
    null distribution. p value is proportion of centered resampled values exceeding observed value.
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _paired_sample_data_checks(data1, data2)

    # Convert string specifiers to callable functions
    stat_func    = _str_to_assoc_stat(stat,axis)    # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    ndim = data1.ndim
    n = data1.shape[axis]

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1,data2, **kwargs)

    # Create generators with n_resamples random bootstrap resamplings with replacement
    resamples = bootstraps(n, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data1.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)
            # Compute statistic on resampled data. Only need to permute one sample
            stat_resmp[stat_slices] = stat_func(data1[data_slices], data2[data_slices], **kwargs)

        # Center each resampled statistic on their mean to approximate null disttribution
        # TODO Need to determine which of these is more apropriate (and align ~return_stats option)
        stat_resmp -= stat_obs
        # stat_resmp -= stat_resmp.mean(axis=axis,keepdims=True)

        # p value = proportion of bootstrap-resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data1.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Index into <axis> of data, with ':' for all other axes
            data_slices = axis_index_slices(axis, resample, ndim)
            # Compute statistic on resampled data1,data2
            stat_resmp  = stat_func(data1[data_slices], data2[data_slices], **kwargs)
            # Subtract observed statistic from resampled statistics
            stat_resmp -= stat_obs
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs, stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isinstance(stat_obs,np.ndarray): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


# =============================================================================
# Two-sample randomization tests
# =============================================================================
def two_sample_test(data1, data2, axis=0, method='permutation', stat='t', tail='both',
                    n_resamples=10000, seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate two-sample test of whether any arbitrary statistic
    differs between two non-paired samples (analogous to 2-sample t-test)

    Parameters
    ----------
    data1 : ndarray, shape=(...,n1,...)
        Data from one group to compare

    data2 : ndarray, shape=(...,n2,...)
        Data from a second group to compare.
        Need not have the same n as data1, but all other dim's must be same size/shape.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str
        Resampling paradigm to use for test:

        - 'permutation' : Permutation test in :func:`two_sample_permutation_test`
        - 'bootstrap' : Bootstrap test in :func:`two_sample_bootstrap_test`

    d : float, Default: 0
        Expected value of `stat` under null distribution (usually 0)

    stat : str or callable, default: 't'
        Statistic to compute and resample. Can be given as a string specifier:

        - 't' : 2-sample t-statistic
        - 'meandiff' : group difference in means

        Or as a custom function to generate resampled statistic of interest.
        Should take single array argument (equal to differences between
        paired samples, with `axis` corresponding to trials/observations)
        and return a scalar value for each independent data series.

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from output.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.
        
    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_resamples-1,...), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_resamples-1.
    """
    method = method.lower()

    if method in ['randomization','permutation']:   test_func = two_sample_permutation_test
    elif method == 'bootstrap':                     test_func = two_sample_bootstrap_test
    else:
        raise ValueError("Unsupported test type '%s'. Use 'permutation' or 'bootstrap'" % method)

    return test_func(data1, data2, axis=axis, stat=stat, tail=tail,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                     keepdims=keepdims, **kwargs)


def two_sample_test_labels(data, labels, axis=0, method='permutation', groups=None, **kwargs):
    """
    Altenative interface to two_sample_test() that allows arguments of form (data,labels)
    instead of (data1,data2)

    Only parameters differing from :func:`two_sample_test` are described here.

    Parameters
    ----------
    data : ndarray, shape=(...,N,...).
        Data from *both* groups to run test on.
        Arbitrary shape, but both groups must have the same n (n1 = n2 = N/2).

    labels : array-like, shape=(N,)
        Group labels for each observation (trial), identifying which group/condition
        each observation belongs to.

    groups : array-like, shape=(n_groups,), optional, default: np.unique(labels)
        List of labels for each group (condition). Used to test only a subset (pair)
        of multi-value labels.
    """
    data1, data2 = data_labels_to_data_groups(data, labels, axis=axis, groups=groups, max_groups=2)
    return two_sample_test(data1, data2, axis=axis, method=method, **kwargs)


def two_sample_permutation_test(data1, data2, axis=0, stat='t', tail='both', n_resamples=10000,
                                 seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate permutation two-sample test

    Parameters and returns are same as :func:`two_sample_test`

    Observations are permuted across the two samples (data1 vs data2).
    The data is pooled across data1 and and data2, and for each random resample,
    n1 and n2 observations are extracted from the pooled and form the resampled
    versions of data1 and data2, respectively.  The same stat is then computed
    on the resampled data to estimate the distrubution under the null hypothesis,
    and the stat value for the actual observed data is compared to this.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.6.3
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _two_sample_data_checks(data1, data2, axis)

    # Convert string specifiers to callable functions
    stat_func    = _str_to_two_sample_stat(stat,axis)   # Statistic (includes default)
    compare_func = _tail_to_compare(tail)               # Tail-specific comparator

    ndim = data1.ndim
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]
    N  = n1+n2

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1,data2, **kwargs)

    # Pool two samples together into combined data
    data_pool = np.concatenate((data1,data2),axis=axis)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data1.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of data and stat, with ':' for all other axes
            # First n1 permuted indexes are resampled "data1"
            # Remaining n2 permuted indexes are resampled "data2"
            data1_slices = axis_index_slices(axis, resample[0:n1], ndim)
            data2_slices = axis_index_slices(axis, resample[n1:], ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)

            data1_resmp  = data_pool[data1_slices]
            data2_resmp  = data_pool[data2_slices]
            # Compute statistic on resampled data1,data2
            stat_resmp[stat_slices] = stat_func(data1_resmp, data2_resmp, **kwargs)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data1.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Index into <axis> of data, with ':' for all other axes
            # First n1 permuted indexes are resampled "data1"
            # Remaining n2 permuted indexes are resampled "data2"
            data1_slices = axis_index_slices(axis, resample[0:n1], ndim)
            data2_slices = axis_index_slices(axis, resample[n1:], ndim)

            data1_resmp   = data_pool[data1_slices]
            data2_resmp   = data_pool[data2_slices]
            # Compute statistic on resampled data1,data2
            stat_resmp    = stat_func(data1_resmp, data2_resmp, **kwargs)
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs, stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isarraylike(stat_obs): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


def two_sample_bootstrap_test(data1, data2, axis=0, stat='t', tail='both',
                              n_resamples=10000, seed=None, return_stats=False,
                              keepdims=True, **kwargs):
    """
    Mass univariate bootstrap two-sample test

    Parameters and returns are same as :func:`two_sample_test`

    Computes stat on each pair of bootstrap resamples, and subtracts off stat computed
    on observed data to center resamples at 0 (mu) to estimate null distribution.
    p value is proportion of centered resampled values exceeding observed value.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.3.10, 6.3
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _two_sample_data_checks(data1, data2, axis)

    # Convert string specifiers to callable functions
    stat_func    = _str_to_two_sample_stat(stat,axis)   # Statistic (includes default)
    compare_func = _tail_to_compare(tail)               # Tail-specific comparator

    ndim = data1.ndim
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data1, data2, **kwargs)

    # Create generators with n_resamples-1 random resamplings with replacement
    # of ints 0:n1-1 and 0:n2-1
    # Note: Seed random number generator only *once* before generating both random samples
    if seed is not None: set_random_seed(seed)
    resamples1 = bootstraps(n1,n_resamples-1)
    resamples2 = bootstraps(n2,n_resamples-1)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data1.shape)]
        stat_resmp = np.empty(stat_shape)

        # Iterate thru <n_resamples> random resamplings, recomputing statistic on each
        for i_resmp,(resample1,resample2) in enumerate(zip(resamples1,resamples2)):
            # Index into <axis> of data and stat, with ':' for all other axes
            data1_slices = axis_index_slices(axis, resample1, ndim)
            data2_slices = axis_index_slices(axis, resample2, ndim)
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)
            stat_resmp[stat_slices] = stat_func(data1[data1_slices], data2[data2_slices], **kwargs)

        # Center each resampled statistic on their mean to approximate null disttribution
        # TODO Need to determine which of these is more apropriate (and align ~return_stats option)
        stat_resmp -= stat_obs
        # stat_resmp -= stat_resmp.mean(axis=axis,keepdims=True)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data1.shape)]
        count = np.ones(stat_shape)
        for resample1,resample2 in zip(resamples1,resamples2):
            # Index into <axis> of data and stat, with ':' for all other axes
            data1_slices = axis_index_slices(axis, resample1, ndim)
            data2_slices = axis_index_slices(axis, resample2, ndim)

            # Compute statistic on resampled data
            stat_resmp = stat_func(data1[data1_slices], data2[data2_slices], **kwargs)
            # Subtract observed statistic from distribution of resampled statistics
            stat_resmp -= stat_obs

            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isarraylike(stat_obs): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


# =============================================================================
# One-way/Two-way randomization tests
# =============================================================================
def one_way_test(data, labels, axis=0, method='permutation', stat='F', tail='right',
                 groups=None, n_resamples=10000, seed=None, return_stats=False,
                 keepdims=True, **kwargs):
    """
    Mass univariate test on any arbitrary 1-way statistic with
    multiple groups/levels (analogous to F-test in a 1-way ANOVA)

    Wrapper around functions for specific one-way tests. See those for details.

    Parameters
    ----------
    data : ndarray, shape=(...,N,...)
        Data to run test on

    labels : array-like, shape=(N,)
        Group labels for each observation (trial), identifying which group (factor level)
        each observation belongs to.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str
        Resampling paradigm to use for test. Currently only 'permutation' implemented.

    stat : str or callable, default: 'F'
        Statistic to compute and resample. Can be given as a string specifier:
        'F' : F-statistic

        Or as a custom function to generate resampled statistic of interest.
        Should take data array (data) with <axis> corresponding to
        trials/observations and labels arguments (labels) and return
        a scalar value for each independent data series.

    tail : {'both','right','left'}, default: 'right' (1-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

        Note: For F-test, only right-tailed test really makes sense bc F distn
        only has positive values and right-sided tailed

    groups : array-like, shape=(n_groups,), optional, default: np.unique(labels)
        List of labels for each group (condition). Used to test only a subset of labels.

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from output.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_resamples-1,...), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_resamples-1.
    """
    method = method.lower()

    if method in ['randomization','permutation','sign']:    test_func = one_way_permutation_test
    else:
        raise ValueError("Only 'permutation' method currently supported")

    return test_func(data, labels, axis=axis, stat=stat, tail=tail, groups=groups,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                     keepdims=keepdims, **kwargs)


def one_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False,
                             keepdims=True, **kwargs):
    """
    Mass univariate one-way permutation test

    Parameters and returns are same as :func:`one_way_test`

    Observation labels are randomly shuffled and allocated to groups
    with the same n's as in the actual observed data.

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.7.1
    """
    if (stat == 'F') and (tail != 'right'):
        warn("For F-test, only right-tailed tests make sense (tail set = %s in args)" % tail)

    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data.ndim + axis

    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_way_stat(stat,axis)  # Statistic (includes default)
    compare_func = _tail_to_compare(tail)           # Tail-specific comparator

    labels  = np.asarray(labels)
    ndim    = data.ndim
    N       = data.shape[axis]

    # Find set of unique group labels in list of labels (if not given)
    if groups is None:
        groups = np.unique(labels)

    # If groups set in args, remove any observations not represented in <groups>
    else:
        idxs = np.in1d(labels, groups)
        if idxs.sum() != N:
            labels  = labels[idxs]
            data    = data[axis_index_slices(axis, idxs, ndim)]
            N       = data.shape[axis]

    # Append <groups> to stat_func args
    if stat_func is one_way_fstat: kwargs.update({'groups':groups})

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, labels, **kwargs)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_resamples-1 for ax,axlen in enumerate(data.shape)]
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Index into <axis> of stat, with ':' for all other axes
            stat_slices = axis_index_slices(axis, [i_resmp], ndim)

            # Compute statistic on data and resampled labels
            # Note: Only need to resample labels
            stat_resmp[stat_slices] = stat_func(data, labels[resample], **kwargs)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs, stat_resmp, axis, compare_func)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else 1 for ax,axlen in enumerate(data.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Compute statistic on data and resampled labels
            # Note: Only need to resample labels
            stat_resmp = stat_func(data,labels[resample], **kwargs)
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats and isarraylike(stat_obs): stat_obs = stat_obs.item()
    elif not keepdims:
        p = p.squeeze(axis=axis)
        if return_stats: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


def two_way_test(data, labels, axis=0, method='permutation', stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate test on any arbitrary 2-way statistic with
    multiple groups/levels (analogous to F-test in a 2-way ANOVA)

    Parameters
    ----------
    data : ndarray, shape=(...,N,...)
        Data to run test on

    labels : array-like, shape=(n,n_terms=2|3) array-like
        Group labels for each observation, identifying which group (factor level)
        each observation belongs to, for each model term. First two columns correspond
        to model main effects; optional third column corresponds to interaction term.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    method : str
        Resampling paradigm to use for test. Currently only 'permutation' implemented.

    stat : str or callable, default: 'F'
        Statistic to compute and resample. Can be given as a string specifier:
        'F' : F-statistic

        Or as a custom function to generate resampled statistic of interest.
        Should take data array (data) with <axis> corresponding to
        trials/observations and labels arguments (labels) and return
        a scalar value for each independent data series.

    tail : {'both','right','left'}, default: 'right' (1-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

        Note: For F-test, only right-tailed test really makes sense bc F distn
        only has positive values and right-sided tailed

    groups : array_like, shape=(n_terms,) of [array-like, shape=(n_groups(term),)], default: all
        List of group labels to use for each for each model term.
        Used to test only a subset of labels. Default to using all values in `labels`.

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If True, returns p values, observed stats, and resampled stats.
        If False, only returns p values.

    keepdims : True
        NOTE: This arg not used here; only here to maintain same API with other stat func's.
        
    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    p : ndarray, shape=(...,n_terms,...)
        p values from test. Same shape as data, with `axis` reduced to length n_terms.

    stat_obs : ndarray, shape=(...,n_terms,...), optional
        Statistic values for actual observed data. Same shape as `p`.

    stat_resmp : ndarray, shape=(...,n_terms,...,n_resamples-1), optional
        Distribution of statistic values for all resamplings of data.
        Same size as data, but `axis` has length n_terms and a new axis of
        length n_resamples-1 is appended to end of array.
        NOTE: axis for resamples is different from all other functions bc
        we need to accomodate both resample and terms dimensions here.
    """
    method = method.lower()

    if method in ['randomization','permutation','sign']:    test_func = two_way_permutation_test
    else:
        raise ValueError("Only 'permutation' method currently supported")

    return test_func(data, labels, axis=axis, stat=stat, tail=tail, groups=groups,
                     n_resamples=n_resamples, seed=seed, return_stats=return_stats, **kwargs)


def two_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False,
                             keepdims=True, **kwargs):
    """
    Mass univariate permutation 2-way test

    Parameters and returns are same as :func:`two_way_test`

    Observation labels are randomly shuffled and allocated to groups (ANOVA cells)
    with the same n's as in the actual observed data.

    We resample the entire row of labels for each observation, which
    effectively shuffles observations between specific combinations
    of groups/factor levels (ANOVA "cells") cf. recommendation in Manly book

    References
    ----------
    Manly (1997) "Randomization, Bootstrap and Monte Carlo Methods in Biology" ch.7.4
    """
    # todo  Add mechanism to auto-generate interaction term from factors (if interact arg==True)
    if (stat == 'F') and (tail != 'right'):
        warn("For F-test, only right-tailed tests make sense (tail set = %s in args)" % tail)

    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data.ndim + axis

    # Convert string specifier to callable function
    stat_func = _str_to_two_way_stat(stat,axis) # Statistic (includes default))
    compare_func = _tail_to_compare(tail)       # Tail-specific comparator

    labels  = np.asarray(labels)
    n_terms = labels.shape[1]
    N       = data.shape[axis]

    # Find all groups/levels in list of labels (if not given in inputs)
    if groups is None:
        groups = [np.unique(labels[:,term]) for term in range(n_terms)]

    # Append <groups> to stat_func args
    if stat_func is two_way_fstat: kwargs.update({'groups':groups})

    # Compute statistic of interest on actual observed data
    stat_obs = stat_func(data, labels, **kwargs)

    # Create generators with n_resamples-1 random permutations of ints 0:N-1
    resamples = permutations(N, n_resamples-1, seed)

    if return_stats:
        # Compute statistic under <n_resamples> random resamplings
        stat_shape = [axlen if ax != axis else n_terms for ax,axlen in enumerate(data.shape)]
        stat_shape.append(n_resamples-1)
        stat_resmp = np.empty(stat_shape)
        for i_resmp,resample in enumerate(resamples):
            # Compute statistic on data and resampled label rows
            stat_resmp[...,i_resmp] = stat_func(data,labels[resample,:], **kwargs)

        # p value = proportion of resampled test statistic values
        #  >= observed value (+ observed value itself)
        p = resamples_to_pvalue(stat_obs[...,np.newaxis],
                                stat_resmp, -1, compare_func).squeeze(axis=-1)

    else:
        # Compute statistic under <n_resamples> random resamplings and
        # tally values exceeding given criterion (more extreme than observed)
        # Note: Init count to 1's to account for observed value itself
        stat_shape = [axlen if ax != axis else n_terms for ax,axlen in enumerate(data.shape)]
        count = np.ones(stat_shape)
        for resample in resamples:
            # Compute statistic on data and resampled label rows
            stat_resmp = stat_func(data,labels[resample,:], **kwargs)
            # Compare observed and resampled stats, and tally values passing criterion
            count += compare_func(stat_obs,stat_resmp)

        # p value = proportion of resampled test statistic values passing criterion
        p = count / n_resamples

    if return_stats:    return p, stat_obs, stat_resmp
    else:               return p


# =============================================================================
# Confidence intervals
# =============================================================================
def one_sample_confints(data, axis=0, stat='mean', confint=0.95, n_resamples=10000, seed=None,
                        return_stats=False, return_sorted=True, keepdims=True, **kwargs):
    """
    Mass univariate bootstrap confidence intervals of any arbitrary 1-sample stat
    (eg mean).  Analogous to SEM/parametric confidence intervals.

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data to compute confints on. Arbitrary shape.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    stat : str or callable, default: 'mean'
        Statistic to compute and resample. Can be given as a string specifier:
        mean'  : mean across observations

        Or as a custom function to generate resampled statistic of interest.
        Should take single array argument (data) with `axis` corresponding
        to trials/observations, and return a scalar value (for each independent
        data series if multiple given).

    confint : float, default: 0.95 (95% confidence interval)
        Confidence interval to compute, expressed as decimal value in range 0-1.
        Typical values are 0.68 (to approximate SEM), 0.95 (95% confint), and 0.99 (99%)

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If False, only return confidence intervals. If True, also return statistic computed
        on observed data, and full distribution of resample statistic.

    return_sorted : bool, default: True
        If True, return stat_resmp sorted by value. If False, return stat_resmp unsorted
        (ordered by resample number), which is useful if you want to keep each resampling
        for all mass-univariate data series's together.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in `stat_obs`.
        If False, removes reduced observations `axis` from `stat_obs`.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    confints : ndarray, shape=(...,2,...)
        Computed bootstrap confidence intervals. Same size as data, with `axis` reduced
        to length 2 = [lower,upper] confidence interval.

    stat_obs : float or ndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    stat_resmp : ndarray, shape=(...,n_resamples,...)
        Distribution of statistic values for all resamplings of data.
        Same size as data, with `axis` now having length=n_resamples.
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data.ndim + axis

    # Convert string specifiers to callable functions
    stat_func    = _str_to_one_sample_stat(stat,axis)

    # Indexes into set of bootstrap resamples corresponding to given confint
    conf_indexes = confint_to_indexes(confint, n_resamples)

    ndim = data.ndim
    n = data.shape[axis]

    # Compute statistic of interest on actual observed data
    if return_stats: stat_obs = stat_func(data, **kwargs)

    # Create generators with n_resamples random bootstrap resamplings with replacement
    resamples = bootstraps(n, n_resamples, seed)

    # Compute statistic under <n_resamples> random resamplings
    stat_shape = [axlen if ax != axis else n_resamples for ax,axlen in enumerate(data.shape)]
    stat_resmp = np.empty(stat_shape)
    for i_resmp,resample in enumerate(resamples):
        # Index into <axis> of data and stat, with ':' for all other axes
        data_slices = axis_index_slices(axis, resample, ndim)
        stat_slices = axis_index_slices(axis, [i_resmp], ndim)
        # Compute statistic on resampled data
        stat_resmp[stat_slices] = stat_func(data[data_slices], **kwargs)

    # Sort boostrap resampled stats and extract confints from them
    stat_slices = axis_index_slices(axis, conf_indexes, ndim)
    if return_stats and not return_sorted:
        # Sort copy of stat_resmp, so we can return original unsorted version
        stat_resmp_sorted = np.sort(stat_resmp, axis=axis)
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp_sorted[stat_slices]
    else:
        stat_resmp.sort(axis=axis)     # Sort resample stats in place
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp[stat_slices]

    # For vector-valued data, extract value from scalar array -> float for output
    if return_stats and (stat_obs.size == 1): stat_obs = stat_obs.item()
    elif not keepdims: stat_obs = stat_obs.squeeze(axis=axis)
        
    if return_stats:    return confints, stat_obs, stat_resmp
    else:               return confints


def paired_sample_confints(data1, data2, axis=0, stat='mean', confint=0.95, n_resamples=10000,
                           seed=None, return_stats=False, return_sorted=True,
                           keepdims=True, **kwargs):
    """
    Mass univariate bootstrap confidence intervals of any arbitrary paired-sample stat
    (eg mean difference). Analogous to SEM/parametric confidence intervals.

    Parameters
    ----------
    data1,data2 : ndarray, shape=(...,n,...)
        Data from two paired groups. Shape is arbitrary, but must be same for data1,2.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    stat : str or callable, default: 'mean'
        Statistic to compute and resample. Can be given as a string specifier:
        'mean'  : mean difference between paired observations

        Or as a custom function to generate resampled statistic of interest.
        Should take single array argument (equal to differences between
        paired samples, with `axis` corresponding to trials/observations)
        and return a scalar value for each independent data series.

    confint : float, default: 0.95 (95% confidence interval)
        Confidence interval to compute, expressed as decimal value in range 0-1.
        Typical values are 0.68 (to approximate SEM), 0.95 (95% confint), and 0.99 (99%)

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If False, only return confidence intervals. If True, also return statistic computed
        on observed data, and full distribution of resample statistic.

    return_sorted : bool, default: True
        If True, return stat_resmp sorted by value. If False, return stat_resmp unsorted
        (ordered by resample number), which is useful if you want to keep each resampling
        for all mass-univariate data series's together.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in `stat_obs`.
        If False, removes reduced observations `axis` from `stat_obs`.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    confints : ndarray, shape=(...,2,...)
        Computed bootstrap confidence intervals. Same size as data, with `axis` reduced
        to length 2 = [lower,upper] confidence interval.

    stat_obs : ndarray, shape=(...,1,...), optional
        Statistic values for actual observed data.
        Same size as data, with `axis` reduced to length 1.

    stat_resmp : ndarray, shape=(...,n_resamples,...)
        Distribution of statistic values for all resamplings of data.
        Same size as data, with `axis` now having length=n_resamples.
    """
    return one_sample_confints(data1 - data2, axis=axis, stat=stat, confint=confint,
                               n_resamples=n_resamples, seed=seed, return_stats=return_stats,
                               return_sorted=return_sorted, keepdims=keepdims, **kwargs)


def two_sample_confints(data1, data2, axis=0, stat='meandiff', confint=0.95, n_resamples=10000,
                        seed=None, return_stats=False, return_sorted=True,
                        keepdims=True, **kwargs):
    """
    Mass univariate bootstrap confidence intervals of any arbitrary 2-sample stat
    (eg difference in group means).  Analogous to SEM/parametric confidence intervals.

    Parameters
    ----------
    data1 : ndarray, shape=(...,n1,...)
        Data from one group to compare

    data2 : ndarray, shape=(...,n2,...)
        Data from a second group to compare.
        Need not have the same n as data1, but all other dim's must be same size/shape.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    stat : str or callable, default: 'meandiff'
        Statistic to compute and resample. Can be given as a string specifier:
        'meandiff'  : difference between group means

        Or as custom function to generate resampled statistic of interest.
        Should take two array arguments (data1,2) and `axis` corresponding
        to trials/observations and return a scalar value for each independent
        data series.

    confint : float, default: 0.95 (95% confidence interval)
        Confidence interval to compute, expressed as decimal value in range 0-1.
        Typical values are 0.68 (to approximate SEM), 0.95 (95% confint), and 0.99 (99%)

    n_resamples : int, default: 10000
        Number of random resamplings to perform for test (should usually be >= 10000 if feasible)

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    return_stats : bool, default: False
        If False, only return confidence intervals. If True, also return statistic computed
        on observed data, and full distribution of resample statistic.

    return_sorted : bool, default: True
        If True, return stat_resmp sorted by value. If False, return stat_resmp unsorted
        (ordered by resample number), which is useful if you want to keep each resampling
        for all mass-univariate data series's together.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in `stat_obs`.
        If False, removes reduced observations `axis` from `stat_obs`.

    **kwargs
        All other kwargs passed directly to callable `stat` function

    Returns
    -------
    confints : ndarray, shape=(...,2,...)
        Computed bootstrap confidence intervals. Same size as data, with `axis` reduced
        to length 2 = [lower,upper] confidence interval.

    stat_obs : float orndarray, shape=(...,[1,]...), optional
        Statistic values for actual observed data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    stat_resmp : ndarray, shape=(...,n_resamples,...)
        Distribution of statistic values for all resamplings of data.
        Same size as data, with `axis` now having length=n_resamples.
    """
    # Wrap negative axis back into 0 to ndim-1
    if axis < 0: axis = data1.ndim + axis

    _two_sample_data_checks(data1, data2, axis)


    # Convert string specifiers to callable functions
    stat_func    = _str_to_two_sample_stat(stat,axis)

    # Indexes into set of bootstrap resamples corresponding to given confint
    conf_indexes = confint_to_indexes(confint, n_resamples)

    ndim = data1.ndim
    n1 = data1.shape[axis]
    n2 = data2.shape[axis]

    # Compute statistic of interest on actual observed data
    if return_stats: stat_obs = stat_func(data1, data2, **kwargs)

    # Create generators with n_resamples random bootstrap resamplings with replacement
    # of ints 0:n1-1 and 0:n2-1
    # Note: Seed random number generator only *once* before generating both random samples
    if seed is not None: set_random_seed(seed)
    resamples1 = bootstraps(n1,n_resamples)
    resamples2 = bootstraps(n2,n_resamples)

    # Compute statistic under <n_resamples> random resamplings
    stat_shape = [axlen if ax != axis else n_resamples for ax,axlen in enumerate(data1.shape)]
    stat_resmp = np.empty(stat_shape)
    for i_resmp,(resample1,resample2) in enumerate(zip(resamples1,resamples2)):
        # Index into <axis> of data and stat, with ':' for all other axes
        data1_slices = axis_index_slices(axis, resample1, ndim)
        data2_slices = axis_index_slices(axis, resample2, ndim)
        stat_slices = axis_index_slices(axis, [i_resmp], ndim)
        # Compute statistic on resampled data
        stat_resmp[stat_slices] = stat_func(data1[data1_slices], data2[data2_slices], **kwargs)

    # Sort boostrap resampled stats and extract confints from them
    stat_slices = axis_index_slices(axis, conf_indexes, ndim)
    if return_stats and not return_sorted:
        # Sort copy of stat_resmp, so we can return original unsorted version
        stat_resmp_sorted = np.sort(stat_resmp, axis=axis)
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp_sorted[stat_slices]
    else:
        stat_resmp.sort(axis=axis)     # Sort resample stats in place
        # Extract lower,upper confints from resampled and sorted stats
        confints = stat_resmp[stat_slices]

    # For vector-valued data, extract value from scalar array -> float for output
    if return_stats and (stat_obs.size == 1): stat_obs = stat_obs.item()
    elif not keepdims: stat_obs = stat_obs.squeeze(axis=axis)

    if return_stats:    return confints, stat_obs, stat_resmp
    else:               return confints


# =============================================================================
# Random sample generators
# =============================================================================
def permutations(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random permutations of integers
    0:n-1, as would be needed for permutation/randomization tests

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=int]
        Generator to iterate over for permutation test.
        Each iteration contains a distinct random permutation of integers 0:n-1.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.permutation(n)


def bootstraps(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random resamplings with
    replacement of integers 0:n-1, as would be needed for bootstrap tests or
    confidence intervals

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=int]
        Generator to iterate over for boostrap test or confidence interval computation.
        Each iteration contains a distinct random resampling with replacement from integers 0:n-1.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.randint(n, size=(n,))


def signs(n, n_resamples=9999, seed=None):
    """
    Yield generator with a set of `n_resamples` random Bernoulli(p=0.5)
    variables (ie binary 0/1 w/ probability of 0.5), each of length <n>,
    as would be needed to set the signs of stats in a sign test.

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int, default: 9999 (appropriate number for test w/ 10,000 samples)
        Number of independent resamples to generate.

    seed : int, default: None
        Random generator seed for repeatable results. Set=None for unseeded random numbers.

    Yields
    ------
    resamples : generator, shape=(n_resamples,) of [ndarray, shape=(n,), dtype=bool]
        Generator to iterate over for random sign test.
        Each iteration contains a distinct random resampling of n Bernoulli random variables.
    """
    if seed is not None: set_random_seed(seed)

    for _ in range(n_resamples):
        yield np.random.binomial(1,0.5, size=(n,)).astype(bool)


def jackknifes(n, n_resamples=None, seed=None):
    """
    Yield generator with a set of n_resamples = n boolean variables,
    each of length n, and each of which excludes one observation/trial in turn,
    as would be needed for a jackknife or leave-one-out test.

    Parameters
    ----------
    n : int
        Number of items to randomly resample from.
        Will usually correspond to number of observations/trials

    n_resamples : int
        Automatically set=n here. Only included for consistent interface.

    seed : None
        Not used. Only included for consistent interface with other functions.

    Yields
    ------
    resamples : generator, shape=(n,) of [ndarray, shape=(n,), dtype=bool]
        Generator to iterate over for jackknife test.
        Each iteration is all 1's except for a single 0, the observation (trial) excluded
        in that iteration. For the ith resample, the ith trial is excluded.
    """
    assert (n_resamples is None) or (n_resamples == n), \
        ValueError("For jackknife/leave-one-out, n_resamples MUST = n")

    trials = np.arange(n)
    for trial in range(n):
        yield trials != trial


#==============================================================================
# Utility functions
#==============================================================================
def resamples_to_pvalue(stat_obs, stat_resmp, axis=0, tail='both'):
    """
    Compute p value from observed and resampled values of a statistic

    Parameters
    ----------
    stat_obs : ndarray, shape=(...,1,...)
        Statistic values for actual observed data

    stat_resmp : ndarray, shape=(...,n_resamples,...)
        Statistic values for randomly resampled data

    axis : int, default: 0
        Axis in `stat_resmp` corresponding to distinct resamples
        (should correspond to a length=1 axis in `stat_obs`)

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    Returns
    -------
    p : ndarray, shape=(...,1,...)
        p values from resampling test. Same size as `stat_obs`.
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
    Return indexes into set of bootstrap resamples corresponding
    to given confidence interval

    Parameters
    ----------
    confint : float
        Desired confidence interval, in range 0-1. eg, for 99% confidence interval, input 0.99

    n_resamples : int
        Number of bootstrap resamples

    Returns
    -------
    conf_indexes : list[int], shape=(2,)
        Indexes into sorted bootstrap resamples corresponding to [lower,upper] confidence interval
    """
    max_interval = 1 - 2.0/n_resamples
    assert (confint <= max_interval) or np.isclose(confint,max_interval), \
        ValueError("Requested confint too large for given number of resamples (max = %.3f)" \
                    % max_interval)

    return [round(n_resamples * (1-confint)/2) - 1,
            round(n_resamples - (n_resamples * (1-confint)/2)) - 1]


#==============================================================================
# Helper functions
#==============================================================================
def _tail_to_compare(tail):
    """ Convert string specifier to callable function implementing it """

    assert isinstance(tail,str), \
        TypeError("Unsupported type '%s' for <tail>. Use string or function" % type(tail))
    assert tail in ['both','right','left'], \
        ValueError("Unsupported value '%s' for <tail>. Use 'both', 'right', or 'left'" % tail)

    # 2-tailed test: hypothesis ~ stat_obs ~= statShuf
    if tail == 'both':
        return lambda stat_obs,stat_resmp: np.abs(stat_resmp) >= np.abs(stat_obs)

    # 1-tailed rightward test: hypothesis ~ stat_obs > statShuf
    elif tail == 'right':
        return lambda stat_obs,stat_resmp: stat_resmp >= stat_obs

    # 1-tailed leftward test: hypothesis ~ stat_obs < statShuf
    else: # tail == 'left':
        return lambda stat_obs,stat_resmp: stat_resmp <= stat_obs


def _str_to_one_sample_stat(stat,axis):
    """ Convert string specifier to function to compute 1-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):                  return stat
    elif stat in ['t','tstat','t1']:    return lambda data: one_sample_tstat(data,axis=axis)
    elif stat == 'mean':                return lambda data: data.mean(axis=axis,keepdims=True)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_assoc_stat(stat,axis):
    """ Convert string specifier to function to compute paired-sample association statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):
        return stat
    elif stat in ['r','pearson','pearsonr']:
        return lambda data1,data2: correlation(data1, data2, axis=axis)
    elif stat in ['r','pearson','pearsonr']:
        return lambda data1,data2: rank_correlation(data1, data2, axis=axis)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_sample_stat(stat,axis):
    """ Convert string specifier to function to compute 2-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):
        return stat
    elif stat in ['t','tstat','t1']:
        return lambda data1,data2: two_sample_tstat(data1, data2, axis=axis)
    elif stat in ['meandiff','mean']:
        return lambda data1,data2: (data1.mean(axis=axis,keepdims=True) -
                                    data2.mean(axis=axis,keepdims=True))
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_one_way_stat(stat,axis):
    """ Convert string specifier to function to compute 1-way multi-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):
        return stat
    elif stat in ['f','fstat','f1']:
        return lambda data, labels: one_way_fstat(data, labels, axis=axis)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_way_stat(stat,axis):
    """ Convert string specifier to function to compute 2-way multi-sample statistic """
    if isinstance(stat,str):  stat = stat.lower()

    if callable(stat):
        return stat
    elif stat in ['f','fstat','f2']:
        return lambda data, labels: two_way_fstat(data, labels, axis=axis)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _paired_sample_data_checks(data1, data2):
    """ Check data format requirements for paired-sample data """

    assert np.array_equal(data1.shape, data2.shape), \
        ValueError("data1 and data2 must have same shape for paired-sample tests. \
                    Use two-sample tests to compare non-paired data with different n's.")


def _two_sample_data_checks(data1, data2, axis):
    """ Check data format requirements for two-sample data """

    assert (data1.ndim == data2.ndim), \
        "data1 and data2 must have same shape except for observation/trial axis (<axis>)"

    if data1.ndim > 1:
        assert np.array_equal([data1.shape[ax] for ax in range(data1.ndim) if ax != axis],
                              [data2.shape[ax] for ax in range(data2.ndim) if ax != axis]), \
            "data1 and data2 must have same shape except for observation/trial axis (<axis>)"

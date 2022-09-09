# -*- coding: utf-8 -*-
"""
Nonparametric randomization, permutation (shuffle), and bootstrap statistics

Overview
--------
Functionality for hypothesis significance testing and confidence interval computation based on
random resampling of the observed data. This uses the data itself to generate an expected
distribution for the null hypothesis, and does not rely on any specific assumptions about
the form of the data distribution(s).

Users interested in *non-randomization* (eg rank-based) nonparametric methods are encouraged
to look at the `statsmodels <https://www.statsmodels.org/>`_ package. Users interested in traditional
parametric methods are encouraged to look at the `scipy.stats
<https://docs.scipy.org/doc/scipy/reference/stats.html>`_ submodule.

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
for generating samples for permutations or bootstraps in your own code.

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

Function reference
------------------
"""
# Created on Tue Jul 30 16:28:12 2019
#
# @author: sbrincat

# TODO  Parallelize resampling loops! (using joblib?)

from math import sqrt
import numpy as np

from spynal.utils import set_random_seed, axis_index_slices, data_labels_to_data_groups, \
                         correlation
from spynal.randstats.sampling import bootstraps
from spynal.randstats.permutation import one_sample_randomization_test, \
                                         paired_sample_permutation_test, \
                                         paired_sample_association_permutation_test, \
                                         two_sample_permutation_test, \
                                         one_way_permutation_test, two_way_permutation_test
from spynal.randstats.bootstrap import one_sample_bootstrap_test, paired_sample_bootstrap_test, \
                                       paired_sample_association_bootstrap_test, \
                                       two_sample_bootstrap_test
from spynal.randstats.utils import confint_to_indexes
from spynal.randstats.helpers import _tail_to_compare, _two_sample_data_checks, \
                                     _str_to_one_sample_stat, _str_to_two_sample_stat


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

        - 'randomization' : Randomization sign test in :func:`.one_sample_randomization_test`
        - 'bootstrap'     : Bootstrap test in :func:`.one_sample_bootstrap_test`

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

        - 'permutation' : Permutation test in :func:`.paired_sample_permutation_test`
        - 'bootstrap' : Bootstrap test in :func:`.paired_sample_bootstrap_test`

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
        NOTE: Custom function should take *single array* = data1 - data2

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

        - 'permutation'   : Permutation test in :func:`.paired_sample_association_permutation_test`
        - 'bootstrap'     : Bootstrap test in :func:`.paired_sample_association_bootstrap_test`

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

        - 'permutation' : Permutation test in :func:`.two_sample_permutation_test`
        - 'bootstrap' : Bootstrap test in :func:`.two_sample_bootstrap_test`

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


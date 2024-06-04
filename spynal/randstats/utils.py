# -*- coding: utf-8 -*-
"""
Utility functions for randomization statistics

Function list
-------------
- resamples_to_pvalue : Compute p value from observed and resampled statistic values
- confint_to_indexes : Returns indexes into resampled stats corresponding to given confints
- jackknife_to_pseudoval : Compute single-trial pseudovalues from leave-one-out jackknife estimates

Function reference
------------------
"""
import numpy as np


def tail_to_compare(tail):
    """
    Convert string specifier for randomization test tail-type to callable function implementing it
    
    Parameters
    ----------
    tail : {'left','right','both'}, default: 'both'
        Type of statistical test ("tail") to perform on x vs x_rsmp:
        
        - 'left' : HO: x_rsmp >= x; H1: x < x_rsmp
        - 'right' : HO: x_rsmp <= x; H1: x > x_rsmp
        - 'both' : HO: x_rsmp == x; H1: x != x_rsmp
        
    Returns
    -------
    compare_func : lambda, args:stat_obs,stat_resmp
        Lambda function that implements comparison to evaluate randomization
        statistical tests of given type
    """
    # If input value is already a callable function, just return it
    if callable(tail): return tail

    assert isinstance(tail,str), \
        TypeError("Unsupported type '%s' for <tail>. Use string or function" % type(tail))

    tail = tail.lower()

    # 2-tailed test: hypothesis ~ stat_obs ~= stat_resmp
    if tail == 'both':
        return lambda stat_obs,stat_resmp: np.abs(stat_resmp) >= np.abs(stat_obs)

    # 1-tailed rightward test: hypothesis ~ stat_obs > stat_resmp
    elif tail == 'right':
        return lambda stat_obs,stat_resmp: stat_resmp >= stat_obs

    # 1-tailed leftward test: hypothesis ~ stat_obs < stat_resmp
    elif tail == 'left':
        return lambda stat_obs,stat_resmp: stat_resmp <= stat_obs

    else:
        ValueError("Unsupported value '%s' for <tail>. Use 'both', 'right', or 'left'" % tail)


def resamples_to_pvalue(stat_obs, stat_resmp, axis=0, tail='both'):
    """
    Compute p value with given tail from observed and resampled values of a statistic

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
    compare_func = tail_to_compare(tail)           

    n_resamples = stat_resmp.shape[axis]

    # Count number of resampled stat values more extreme than observed value
    p = np.sum(compare_func(stat_obs,stat_resmp), axis=axis, keepdims=True)

    # p value is proportion of samples failing criterion (+1 for observed stat)
    return (p + 1) / (n_resamples + 1)


def confint_to_indexes(confint, n_resamples):
    """
    Return indexes into set of resamples corresponding to given confidence interval

    Typically used for bootstrap resampled confidence intervals. Could be used for jackknifes.

    Parameters
    ----------
    confint : float
        Desired confidence interval, in range 0-1. eg, for 99% confidence interval, input 0.99

    n_resamples : int
        Number of resamples (eg bootstraps)

    Returns
    -------
    conf_indexes : list[int], shape=(2,)
        Indexes into sorted resamples corresponding to [lower,upper] confidence interval
    """
    max_interval = 1 - 2.0/n_resamples
    assert (confint <= max_interval) or np.isclose(confint,max_interval), \
        ValueError("Requested confint too large for given number of resamples (max = %.3f)" \
                    % max_interval)

    return [round(n_resamples * (1-confint)/2) - 1,
            round(n_resamples - (n_resamples * (1-confint)/2)) - 1]


def jackknife_to_pseudoval(x, xjack, n):
    """
    Compute single-trial pseudovalues from leave-one-out jackknife estimates

    Parameters
    ----------
    x : ndarray, shape=Any
        Statistic computed on full observed data. Any arbitrary shape.

    xjack : ndarray, shape=Any
        Statistic computed on jackknife resampled data

    n : int
        Number of observations (trials) used to compute x

    Returns
    -------
    pseudo : ndarray, shape=Any
        Single-trial jackknifed pseudovalues. Same shape as xjack.
    """
    return n*x - (n-1)*xjack

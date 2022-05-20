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

from spynal.randstats.helpers import _tail_to_compare


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
    if callable(tail):  compare_func = tail
    else:               compare_func = _tail_to_compare(tail)

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

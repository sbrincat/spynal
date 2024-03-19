# -*- coding: utf-8 -*-
""" Nonparametric permutation/shuffle/randomization statistics """
from warnings import warn
import numpy as np

from spynal.utils import axis_index_slices, isarraylike, one_way_fstat, two_way_fstat
from spynal.randstats.sampling import signs, permutations
from spynal.randstats.utils import tail_to_compare, resamples_to_pvalue
from spynal.randstats.helpers import _str_to_one_sample_stat, _str_to_assoc_stat, \
                                     _str_to_two_sample_stat, _str_to_one_way_stat, \
                                     _str_to_two_way_stat, \
                                     _paired_sample_data_checks, _two_sample_data_checks


def one_sample_randomization_test(data, axis=0, mu=0, stat='t', tail='both', n_resamples=10000,
                                  seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate 1-sample randomization test

    Parameters and returns are same as :func:`.one_sample_test`

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
    compare_func = tail_to_compare(tail)                # Tail-specific comparator

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


def paired_sample_permutation_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                   n_resamples=10000, seed=None, return_stats=False,
                                   keepdims=True, **kwargs):
    """
    Mass univariate paired-sample permutation test

    Parameters and returns are same as :func:`.paired_sample_test`

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


def paired_sample_association_permutation_test(data1, data2, axis=0, stat='r', tail='both',
                                               n_resamples=10000, seed=None, return_stats=False,
                                               keepdims=True, **kwargs):
    """
    Mass bivariate permutation test of association (eg correlation) between two paired samples

    Parameters and returns are same as :func:`.paired_sample_association_test`

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
    compare_func = tail_to_compare(tail)            # Tail-specific comparator

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


def two_sample_permutation_test(data1, data2, axis=0, stat='t', tail='both', n_resamples=10000,
                                 seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate permutation two-sample test

    Parameters and returns are same as :func:`.two_sample_test`

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
    compare_func = tail_to_compare(tail)                # Tail-specific comparator

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


def one_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False,
                             keepdims=True, **kwargs):
    """
    Mass univariate one-way permutation test

    Parameters and returns are same as :func:`.one_way_test`

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
    compare_func = tail_to_compare(tail)            # Tail-specific comparator

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


def two_way_permutation_test(data, labels, axis=0, stat='F', tail='right', groups=None,
                             n_resamples=10000, seed=None, return_stats=False,
                             keepdims=True, **kwargs):
    """
    Mass univariate permutation 2-way test

    Parameters and returns are same as :func:`.two_way_test`

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
    compare_func = tail_to_compare(tail)        # Tail-specific comparator

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


# -*- coding: utf-8 -*-
""" Nonparametric bootstrap statistics """
import numpy as np

from spynal.utils import axis_index_slices, isarraylike, set_random_seed
from spynal.randstats.sampling import bootstraps
from spynal.randstats.utils import tail_to_compare, resamples_to_pvalue
from spynal.randstats.helpers import _str_to_one_sample_stat, _str_to_assoc_stat, \
                                     _str_to_two_sample_stat, \
                                     _paired_sample_data_checks, _two_sample_data_checks


def one_sample_bootstrap_test(data, axis=0, mu=0, stat='t', tail='both', n_resamples=10000,
                              seed=None, return_stats=False, keepdims=True, **kwargs):
    """
    Mass univariate 1-sample bootstrap test

    Parameters and returns are same as :func:`.one_sample_test`

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
    compare_func = tail_to_compare(tail)                # Tail-specific comparator

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


def paired_sample_bootstrap_test(data1, data2, axis=0, d=0, stat='t', tail='both',
                                 n_resamples=10000, seed=None, return_stats=False,
                                 keepdims=True, **kwargs):
    """
    Mass univariate paired-sample bootstrap test

    Parameters and returns are same as :func:`.paired_sample_test`

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


def paired_sample_association_bootstrap_test(data1, data2, axis=0, stat='r', tail='both',
                                             n_resamples=10000, seed=None, return_stats=False,
                                             keepdims=True, **kwargs):
    """
    Mass bivariate boostrap test of association (eg correlation) between two paired samples

    Parameters and returns are same as :func:`.paired_sample_association_test`

    Observations are bootstrap resampled in pairs, and stat is recomputed on each.
    Stat computed on observed data is subtracted off resamples, to center them at 0 and estimate
    null distribution. p value is proportion of centered resampled values exceeding observed value.
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


def two_sample_bootstrap_test(data1, data2, axis=0, stat='t', tail='both',
                              n_resamples=10000, seed=None, return_stats=False,
                              keepdims=True, **kwargs):
    """
    Mass univariate bootstrap two-sample test

    Parameters and returns are same as :func:`.two_sample_test`

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
    compare_func = tail_to_compare(tail)                # Tail-specific comparator

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


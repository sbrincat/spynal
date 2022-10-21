# -*- coding: utf-8 -*-
""" Helper functions for `randstats` module for randomization statistics """
import numpy as np

from spynal.utils import one_sample_tstat, two_sample_tstat, one_way_fstat, two_way_fstat, \
                         correlation, rank_correlation

#==============================================================================
# Utility functions
#==============================================================================

#==============================================================================
# Private helper functions
#==============================================================================
def _tail_to_compare(tail):
    """ Convert string specifier to callable function implementing it """
    # If input value is already a callable function, just return it
    if callable(tail): return tail

    assert isinstance(tail,str), \
        TypeError("Unsupported type '%s' for <tail>. Use string or function" % type(tail))

    tail = tail.lower()

    # 2-tailed test: hypothesis ~ stat_obs ~= statShuf
    if tail == 'both':
        return lambda stat_obs,stat_resmp: np.abs(stat_resmp) >= np.abs(stat_obs)

    # 1-tailed rightward test: hypothesis ~ stat_obs > statShuf
    elif tail == 'right':
        return lambda stat_obs,stat_resmp: stat_resmp >= stat_obs

    # 1-tailed leftward test: hypothesis ~ stat_obs < statShuf
    elif tail == 'left':
        return lambda stat_obs,stat_resmp: stat_resmp <= stat_obs

    else:
        ValueError("Unsupported value '%s' for <tail>. Use 'both', 'right', or 'left'" % tail)


def _str_to_one_sample_stat(stat, axis):
    """ Convert string specifier to function to compute 1-sample statistic """
    if callable(stat): return stat

    assert isinstance(stat,str), \
        TypeError("Unsupported type '%s' for <stat>. Use string or function" % type(stat))

    stat = stat.lower()
    if stat in ['t','tstat','t1']:      return lambda data: one_sample_tstat(data, axis=axis)
    elif stat == 'mean':                return lambda data: data.mean(axis=axis, keepdims=True)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_assoc_stat(stat, axis):
    """ Convert string specifier to function to compute paired-sample association statistic """
    if callable(stat): return stat

    assert isinstance(stat,str), \
        TypeError("Unsupported type '%s' for <stat>. Use string or function" % type(stat))

    stat = stat.lower()
    if stat in ['r','pearson','pearsonr']:
        return lambda data1,data2: correlation(data1, data2, axis=axis)
    elif stat in ['r','pearson','pearsonr']:
        return lambda data1,data2: rank_correlation(data1, data2, axis=axis)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_sample_stat(stat, axis):
    """ Convert string specifier to function to compute 2-sample statistic """
    if callable(stat): return stat

    assert isinstance(stat,str), \
        TypeError("Unsupported type '%s' for <stat>. Use string or function" % type(stat))

    stat = stat.lower()
    if stat in ['t','tstat','t1']:
        return lambda data1,data2: two_sample_tstat(data1, data2, axis=axis)
    elif stat in ['meandiff','mean']:
        return lambda data1,data2: (data1.mean(axis=axis, keepdims=True) -
                                    data2.mean(axis=axis, keepdims=True))
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_one_way_stat(stat, axis):
    """ Convert string specifier to function to compute 1-way multi-sample statistic """
    if callable(stat): return stat

    assert isinstance(stat,str), \
        TypeError("Unsupported type '%s' for <stat>. Use string or function" % type(stat))

    stat = stat.lower()
    if stat in ['f','fstat','f1']:
        return lambda data, labels: one_way_fstat(data, labels, axis=axis)
    else:
        raise ValueError('Unsupported option ''%s'' given for <stat>' % stat)


def _str_to_two_way_stat(stat, axis):
    """ Convert string specifier to function to compute 2-way multi-sample statistic """
    if callable(stat): return stat

    assert isinstance(stat,str), \
        TypeError("Unsupported type '%s' for <stat>. Use string or function" % type(stat))

    stat = stat.lower()
    if stat in ['f','fstat','f2']:
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

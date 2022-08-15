# -*- coding: utf-8 -*-
"""
General-purpose python utilities for data preprocessing and analysis

Overview
--------
Functionality includes:

- basic statistics: z-scoring, t/F-stats, SNR measures (Fano,CV,etc.), correlation
- numerical methods: interpolation, setting random seed
- functions to reshape data arrays and dynamically index into specific array axes
- functions for dealing w/ Numpy "object" arrays (similar to Matlab cell arrays)
- various other useful little utilities

Function list
-------------
Basic statistics
^^^^^^^^^^^^^^^^
- zscore :            Mass univariate Z-score data along given axis (or whole array)
- fano :              Fano factor (variance/mean) of data
- cv :                Coefficient of Variation (SD/mean) of data
- cv2 :               Local Coefficient of Variation (Holt 1996) of data
- lv :                Local Variation (Shinomoto 2009) of data

- one_sample_tstat :  Mass univariate 1-sample t-statistic
- paired_tstat :      Mass univariate paired-sample t-statistic
- two_sample_tstat :  Mass univariate 2-sample t-statistic
- one_way_fstat :     Mass univariate 1-way F-statistic
- two_way_fstat :     Mass univariate 2-way (with interaction) F-statistic

- correlation :       Pearson product-moment correlation btwn two variables
- rank_correlation :  Spearman rank correlation btwn two variables

Numerical utility functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- set_random_seed :     Seed Python/Numpy random number generators with given seed
- interp1 :             Interpolate 1d data vector at given index values
- gaussian :            Evaluate parameterized 1D Gaussian function at given datapoint(s)
- gaussian_2d :         Evaluate parameterized 2D Gaussian function at given datapoint(s)
- gaussian_nd :         Evaluate parameterized N-D Gaussian function at given datapoint(s)
- is_symmetric :        Test if matrix is symmetric
- is_positive_definite : Test if matrix is symmetric positive (semi)definite
- setup_sliding_windows : Generates set of sliding windows using given parameters

Data indexing and reshaping functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- index_axis :          Dynamically index into arbitrary axis of ndarray
- axis_index_slices :   Generates list of slices for dynamic axis indexing
- standardize_array :   Reshapes array to 2D w/ axis relevant for analysis at start or end
- undo_standardize_array : Undoes effect of standardize_array after analysis
- data_labels_to_data_groups : Convert (data,labels) pair to tuple of (data_1,data_2,...,data_k)
- data_groups_to_data_labels : Convert tuple of (data_1,data_2,...,data_k) to (data,labels) pair


Other utilities
^^^^^^^^^^^^^^^
- iarange :             np.arange(), but with an inclusive endpoint
- unsorted_unique :     np.unique(), but without sorting values
- isarraylike :         Tests if variable is "array-like" (ndarray, list, or tuple)
- isnumeric :           Tests if array dtype is numeric (int, float, or complex)
- ispc:                 Tests if running on Windows OS
- ismac:                Tests if running on MacOS
- isunix:               Tests if running on Linux/UNIX (but not Mac OS)
- object_array_equal :          Determine if two object arrays are equal
- object_array_compare :        Compare each object within an object ndarray
- concatenate_object_array :    Concatenates objects across one/more axes of object ndarray

Function reference
------------------
"""
# Created on Fri Apr  9 13:28:15 2021
#
# @author: sbrincat
import os
import platform
import time
import random
from math import cos, sin, sqrt
import numpy as np

from scipy.interpolate import interp1d
from scipy.stats import rankdata

from spynal.helpers import _standardize_to_axis_0, _undo_standardize_to_axis_0, \
                           _standardize_to_axis_end, _undo_standardize_to_axis_end

# =============================================================================
# Basic statistics
# =============================================================================
def zscore(data, axis=None, time_range=None, time_axis=None, timepts=None,
           ddof=0, zerotol=1e-6, return_stats=False):
    """
    Z-score data along given axis (or over entire array)

    Optionally also returns mean,SD (eg, to compute on training set and apply to test set)

    Parameters
    ----------
    data : array-like, shape=(...,n_obs,...)
        Data to z-score. Arbitrary dimensionality.

    axis : int, default: None (compute z-score across entire data array)
        Array axis to compute mean/SD along for z-scoring (usually corresponding to
        distict trials/observations). If None, computes mean/SD across entire
        array (analogous to np.mean/std).

    time_range : array-like, shape=(2,), default: None (compute mean/SD over all time points)
        Optionally allows for computing mean/SD within a given time window, then using
        these to z-score ALL timepoints (eg compute mean/SD within a "baseline" window,
        then use to z-score all timepoints). Set=[start,end] of time window. If set, MUST
        also provide values for `time_axis` and `timepts`.

    time_axis : int, optional
        Axis corresponding to timepoints. Only necessary if `time_range` is set.

    timepts : array-like, shape=(n_timepts,), optional
        Time sampling vector for data. Only necessary if `time_range` is set, unused otherwise.

    ddof : int, default: 0
        Sets divisor for computing SD = N - ddof. Set=0 for max likelihood estimate,
        set=1 for unbiased (N-1 denominator) estimate

    zerotol : float, default: 1e-6
        Any SD values < `zerotol` are treated as 0, and corresponding z-scores set = np.nan

    return_stats : bool, default: False
        If True, also returns computed mean, SD. If False, only returns z-scored data.

    Returns
    -------
    data : ndarray, shape=(...,n_obs,...)
        Z-scored data. Same shape as input `data`.

    mean : ndarray, shape=(...,n_obs,...), optional
        Computed means for z-score. Only returned if `return_stats` is True.
        Same as input `data` with 'axis`reduced to length 1.

    sd : ndarray, shape=(...,n_obs,...), optional
        Computed standard deviations for z-score. Only returned if `return_stats` is True.
        Same as input `data` with 'axis`reduced to length 1.

    Examples
    --------
    data = zscore(data, return_stats=False)

    data, mu, sd = zscore(data, return_stats=True)
    """
    # Compute mean/SD separately for each timepoint (axis not None) or across all array (axis=None)
    if time_range is None:
        # Compute mean and standard deviation of data along <axis> (or entire array)
        mu = data.mean(axis=axis, keepdims=True)
        sd = data.std(axis=axis, ddof=ddof, keepdims=True)

    # Compute mean/SD within given time range, then apply to all timepoints
    else:
        assert (len(time_range) == 2) and (time_range[1] > time_range[0]), \
            "time_range must be given as [start,end] time of desired time window"
        assert timepts is not None, "If time_range is set, must also input value for timepts"
        assert time_axis is not None, "If time_range is set, must also input value for time_axis"

        # Compute mean data value across all timepoints within window = "baseline" for z-score
        win_bool = (timepts >= time_range[0]) & (timepts <= time_range[1])
        win_data = index_axis(data, time_axis, win_bool).mean(axis=time_axis, keepdims=True)

        # Compute mean and standard deviation of data along <axis>
        mu = win_data.mean(axis=axis, keepdims=True)
        sd = win_data.std(axis=axis, ddof=ddof, keepdims=True)

    # Compute z-score -- Subtract mean and normalize by SD
    data = (data - mu) / sd

    # Find any data values w/ sd ~ 0 and set data = NaN for those points
    if axis is None:
        if np.isclose(sd,0,rtol=zerotol): data = np.nan
    else:
        zero_points = np.isclose(sd,0,rtol=zerotol)
        tiling = [1]*data.ndim
        tiling[axis] = data.shape[axis]
        if time_range is not None: tiling[time_axis] = data.shape[time_axis]
        data[np.tile(zero_points,tiling)] = np.nan

    if return_stats:    return data, mu, sd
    else:               return data


def fano(data, axis=None, ddof=0, keepdims=True):
    """
    Computes Fano factor of data along a given array axis or across entire array

    np.nan is returned for cases where the mean ~ 0

    Fano factor = variance/mean

    Fano factor has an expected value of 1 for a Poisson distribution/process.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Data of arbitrary shape

    axis : int, default: None (compute across entire array)
        Array axis to compute Fano factor on (usually corresponding to distict
        trials/observations). If None, computes Fano factor across entire array
        (analogous to np.mean/var).

    ddof : int, default: 0
        Sets divisor for computing variance = N - ddof. Set=0 for max likelihood
        estimate, set=1 for unbiased (N-1 denominator) estimate.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    fano : float or ndarray, shape=(...,[1,] ...)
        Fano factor of data.
        For 1d data or axis=None, a single scalar value is returned.
        Otherwise, it's an array w/ same shape as data, but with `axis` reduced to length 1
        if `keepdims` is True, and with `axis` removed if `keepdims` is False.
    """
    mean    = data.mean(axis=axis, keepdims=keepdims)
    var     = data.var(axis=axis, keepdims=keepdims, ddof=ddof)
    fano_   = var/mean
    # Find any data values w/ mean ~ 0 and set output = NaN for those points
    fano_[np.isclose(mean,0)] = np.nan

    if fano_.size == 1: fano_ = fano_.item()
    return fano_

fano_factor = fano
""" Alias of :func:`fano`. See there for details """


def cv(data, axis=None, ddof=0, keepdims=True):
    """
    Compute Coefficient of Variation of data, along a given array axis or across entire array

    np.nan is returned for cases where the mean ~ 0

    CV = standard deviation/mean

    CV has an expected value of 1 for a Poisson distribution or Poisson process.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Data of arbitrary shape

    axis : int, default: None (compute across entire array)
        Array axis to compute Fano factor on (usually corresponding to distict
        trials/observations). If None, computes Fano factor across entire array
        (analogous to np.mean/var).

    ddof : int, default: 0
        Sets divisor for computing variance = N - ddof. Set=0 for max likelihood
        estimate, set=1 for unbiased (N-1 denominator) estimate.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    CV : float or ndarray, shape=(...,[1,] ...)
        CV (SD/mean) of data.
        For 1d data or axis=None, a single scalar value is returned.
        Otherwise, it's an array w/ same shape as data, but with `axis` reduced to length 1
        if `keepdims` is True, and with `axis` removed if `keepdims` is False.
    """
    mean    = data.mean(axis=axis, keepdims=keepdims)
    sd      = data.std(axis=axis, keepdims=keepdims, ddof=ddof)
    CV      = sd/mean
    # Find any data values w/ mean ~ 0 and set output = NaN for those points
    CV[np.isclose(mean,0)] = np.nan

    if CV.size == 1: CV = CV.item()
    return CV

coefficient_of_variation = cv
""" Alias of :func:`cv`. See there for details """


def cv2(data, axis=0, keepdims=True):
    """
    Compute local Coefficient of Variation (CV2) of data, along a given array axis

    CV2 reduces effects of slow changes in data (eg changes in spike rate) on
    measure of variation by only comparing adjacent data values (eg adjacent ISIs).

    CV2 has an expected value of 1 for a Poisson process.

    Typically used as measure of local variation in inter-spike intervals.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Data of arbitrary shape

    axis : int, default: 0
        Array axis to compute CV2 on. Unlike CV, computing CV2 over entire array is
        not permitted and will raise an error, as it is a locally-defined measure.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    CV2 : float or ndarray, shape=(...,[1,] ...)
        CV2 of data.
        For 1d data, a single scalar value is returned.
        Otherwise, it's an array w/ same shape as data, but with `axis` reduced to length 1
        if `keepdims` is True, and with `axis` removed if `keepdims` is False.

    References
    ----------
    Holt et al. (1996) Journal of Neurophysiology https://doi.org/10.1152/jn.1996.75.5.1806
    """
    assert axis is not None, \
        ValueError("Must input int value for `axis`. \
                    CV2 is locally-defined and not computable across entire array")

    # Difference between adjacent values in array, along <axis>
    diff    = np.diff(data,axis=axis)
    # Sum between adjacent values in array, along <axis>
    denom   = index_axis(data, axis, range(0,data.shape[axis]-1)) + \
              index_axis(data, axis, range(1,data.shape[axis]))

    # CV2 formula (Holt 1996 eqn. 4)
    CV2     = (2*np.abs(diff) / denom).mean(axis=axis, keepdims=keepdims)

    if CV2.size == 1:   CV2 = CV2.item()
    return CV2


def lv(data, axis=0, keepdims=True):
    """
    Compute Local Variation (LV) of data along a given array axis

    LV reduces effects of slow changes in data (eg changes in spike rate) on
    measure of variation by only comparing adjacent data values (eg adjacent ISIs).

    LV has an expected value of 1 for a Poisson process.

    Typically used as measure of local variation in inter-spike intervals.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Data of arbitrary shape

    axis : int, default: 0
        Array axis to compute LV on. Unlike CV, computing LV over entire array is
        not permitted and will raise an error, as it is a locally-defined measure.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    LV : float or ndarray, shape=(...,[1,]...)
        LV of data.
        For 1d data, a single scalar value is returned.
        Otherwise, it's an array w/ same shape as data, but with `axis` reduced to length 1
        if `keepdims` is True, and with `axis` removed if `keepdims` is False.

    References
    ----------
    Shinomoto et al. (2009) PLoS Computational Biology https://doi.org/10.1371/journal.pcbi.1000433
    """
    assert axis is not None, \
        ValueError("Must input int value for `axis`. \
                    LV is locally-defined and not computable across entire array")

    # Difference between adjacent values in array, along <axis>
    diff    = np.diff(data,axis=axis)
    # Sum between adjacent values in array, along <axis>
    denom   = index_axis(data, axis, range(0,data.shape[axis]-1)) + \
              index_axis(data, axis, range(1,data.shape[axis]))
    n       = data.shape[axis]

    # LV formula (Shinomoto 2009 eqn. 2)
    # Note: np.diff() reverses sign from original formula, but it gets squared anyway
    LV     = (((diff/denom)**2) * (3/(n-1))).sum(axis=axis, keepdims=keepdims)

    if LV.size == 1:    LV = LV.item()
    return LV


# Note: In timeit tests, ran ~2x as fast as scipy.stats.ttest_1samp
def one_sample_tstat(data, axis=0, mu=0, keepdims=True):
    """
    Mass univariate 1-sample t-statistic, relative to expected mean under null `mu`

    t = (mean(data) - mu) / SEM(data)

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data to compute stat on. `axis` should correspond to distinct observations/trials;
        other axes Z treated as independent data series, and stat is computed separately for each

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    mu : float, default: 0
        Expected mean under the null hypothesis.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    t : float or ndarray, shape=(...[,1,]...)
        1-sample t-statistic for data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True or removed if `keepdims` is False.
    """
    data = np.asarray(data)

    n   = data.shape[axis]

    if mu != 0:  data = data - mu

    # Compute mean and unbiased standard deviation of data
    mu  = data.mean(axis=axis,keepdims=keepdims)
    sd  = data.std(axis=axis,ddof=1,keepdims=keepdims)

    # t statistic = mean/SEM
    t = mu / (sd/sqrt(n))

    if t.size == 1: t = t.item()
    return t


def paired_tstat(data1, data2, axis=0, d=0, keepdims=True):
    """
    Mass univariate paired-sample t-statistic, relative to mean difference under null `d`

    d_obs = data1 - data2

    t = (mean(d_obs) - d) / SEM(d_obs)

    Parameters
    ----------
    data1/data2 : ndarray, shape=(...,n,...)
        Data from two groups to compare.
        Shape is arbitrary, but must be same for data1,2.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    d : float, default: 0
        Hypothetical difference in means under null hypothesis

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    t : float or ndarray, shape=(...[,1,]...)
        Paired-sample t-statistic for given data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True or removed if `keepdims` is False.
    """
    return one_sample_tstat(data1 - data2, axis=axis, mu=d, keepdims=keepdims)


# Note: In timeit tests, ran ~2x as fast as scipy.stats.ttest_ind
def two_sample_tstat(data1, data2, axis=0, equal_var=True, d=0, keepdims=True):
    """
    Mass univariate 2-sample t-statistic, relative to mean difference under null `d`

    t = (mean(data1) - mean(data2) - mu) / pooledSE(data1,data2)

    (where the formula for pooled SE differs depending on `equal_var`)

    Parameters
    ----------
    data1 : ndarray, shape=(...,n1,...)
        Data from one group to compare.

    data2 : ndarray, shape=(...,n2,...)
        Data from a second group to compare.
        Need not have the same n as data1, but all other dim's must be
        same size/shape. For both, `axis` should correspond to
        distinct observations/trials; other axes are treated as
        independent data series, and stat is computed separately for each.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    equal_var : bool, default: True
        If True, compute standard t-stat assuming equal population variances for 2 groups.
        If False, compute Welch’s t-stat, which does not assume equal population variances.

    d : float, default: 0
        Hypothetical difference in means under null hypothesis

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    t : float or ndarray, shape=(...[,1,]...)
        2-sample t-statistic for data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True or removed if `keepdims` is False.

    References
    ----------
    - Indep t-test : https://en.wikipedia.org/wiki/Student%27s_t-test#Independent_two-sample_t-test
    - Welch's test : https://en.wikipedia.org/wiki/Welch%27s_t-test
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    n1  = data1.shape[axis]
    n2  = data2.shape[axis]

    # Compute mean of each group and their difference (offset by null mean)
    d   = data1.mean(axis=axis,keepdims=keepdims) - \
          data2.mean(axis=axis,keepdims=keepdims) - d

    # Compute variance of each group
    var1 = data1.var(axis=axis,ddof=1,keepdims=keepdims)
    var2 = data2.var(axis=axis,ddof=1,keepdims=keepdims)

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

    # t statistic = difference in means / pooled standard error
    t = d / se

    if t.size == 1: t = t.item()
    return t


# Note: In timeit tests, this code ran slightly faster than scipy.stats.f_oneway
def one_way_fstat(data, labels, axis=0, groups=None, keepdims=True):
    """
    Mass univariate 1-way F-statistic on given data and labels

    F = var(between groups) / var(within groups)

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data to compute stat on. `axis` should correspond to distinct observations/trials;
        other axes are treated as independent data series, and stat is computed separately for each

    labels : array-like, shape=(n,)
        Group labels for each observation (trial), identifying which group (factor level)
        each observation belongs to.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    groups : array-like, shape=(n_groups,), optional, default: np.unique(labels)
        List of labels for each group (condition). Used to test only a subset of labels.

    keepdims : bool, default: True
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    F : float or ndarray, shape=(...[,1,]...)
        F-statistic for data. For 1d data, returned as scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True or removed if `keepdims` is False.
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

    F = (SS_groups/df_groups) / (SS_error/df_error)    # F statistic

    if F.size == 1:     F = F.item()
    elif not keepdims:  F = F.squeeze(axis=axis)
    return F


# Note: In timeit tests, this code ran much faster than ols and statsmodels.anova_lm
def two_way_fstat(data, labels, axis=0, groups=None):
    """
    Mass univariate 2-way (with interaction) F-statistic on given data and labels

    F = var(between groups) / var(within groups)

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data to compute stat on. `axis` should correspond to distinct observations/trials;
        other axes are treated as independent data series, and stat is computed separately for each

    labels : array-like, shape=(n,n_terms=2|3)
        Group labels for each model term and observation (trial), identifying which group
        (factor level) each observation belongs to for each term. First 2 columns should reflect
        main effects, and optional third column should be their interaction.

    axis : int, default: 0 (1st axis)
        Axis of data corresponding to distinct trials/observations.

    groups : array_like, shape=(n_terms,) of [array-like, shape=(n_groups(term),)], default: all
        List of group labels to use for each for each model term.
        Used to test only a subset of labels. Default to using all values in `labels`.

    Returns
    -------
    F : ndarray, shape=(...,n_terms,...)
        F-statistic for given data. Same shape as data, with `axis` reduced to length=n_terms.

    References
    ----------
    Zar "Biostatistical Analysis" ch.12
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


def correlation(data1, data2, axis=None, keepdims=True):
    """
    Compute Pearson product-moment (standard) correlation between two variables,
    in mass-bivariate fashion

    `axis` is treated as observations (eg trials), which correlation is computed over.
    Correlations are computed separately across all other array dims (eg, timepoints, freqs, etc).
    If axis is None, correlations are computed across entire 1-d flattened (unrolled) arrays.

    Correlations range from -1 (perfect anti-correlation) to +1 (perfect positive correlation),
    with 0 indicating a lack of correlation.

    Pearson correlation only identifies linear relationships between variables.
    If a nonlinear (monotonic) relationship is suspected, consider using rank_correlation instead.

    Parameters
    ----------
    data1,data2 : ndarray, shape=(n,) or (...,n,...)
        Paired data to compute correlations between.
        Can be 1d vectors or multi-dim arrays, but must have same shape.

    axis : int or None, default: None (compute across entire flattened array)
        Array axis to treat as observations and compute correlations over.
        Correlations are computed in mass-bivariate fashion across all other array axes.
        If axis=None, correlation is computed across entire 1d flattened arrays.

    keepdims : bool, default: True
        If False, correlation `axis` is removed (squeezed out) from output.
        If True, `axis` is kept in output as singleton (length 1) axis.

    Returns
    -------
    r : float or ndarray, shape=(...,[1,]...)
        Correlation between data1 & data2.
        For 1d data, r is a float. For multi-d data, r is same shape as data, but with
        `axis` reduced to length 1 (if `keepdims` is True) or removed (if `keepdims` is False).
    """
    assert data1.shape == data2.shape, ValueError("data1 and data2 must have same shape")

    # Center each data array around its mean (along given axis or across entire array)
    mean1 = data1.mean(axis=axis,keepdims=True)
    mean2 = data2.mean(axis=axis,keepdims=True)

    data1_c = data1 - mean1
    data2_c = data2 - mean2

    # Compute normalization terms for each data array
    norm1 = (data1_c**2).sum(axis=axis,keepdims=keepdims)
    norm2 = (data2_c**2).sum(axis=axis,keepdims=keepdims)

    # Compute correlation r = cross-product / sqrt(each auto-product)
    r = (data1_c*data2_c).sum(axis=axis,keepdims=keepdims) / np.sqrt(norm1*norm2)

    # Deal with any possible floating point errors that push r outside [-1,1]
    r = np.maximum(np.minimum(r,1.0), -1.0)
    # For vector-valued data, extract value from scalar array -> float for output
    if r.size == 1: r = r.item()

    return r


def rank_correlation(data1, data2, axis=None, keepdims=True):
    """
    Computes Spearman rank correlation between two variables, in mass-bivariate fashion

    Input `axis` is treated as observations (eg trials), which correlation is computed over.
    Correlations are computed separately across all other array dims (eg, timepoints, freqs, etc).
    If axis is None, correlations are computed across entire 1d flattened (unrolled) arrays.

    Each data is sorted into rank-order separately, and the resulting ranks are entered
    into a standard (Pearson) correlation. This identifies any monotonic relationship
    between variables, and thus should be favored when a nonlinear relationship is suspected.

    Correlations range from -1 (perfect anti-correlation) to +1 (perfect positive correlation),
    with 0 indicating a lack of correlation.

    Parameters
    ----------
    data1,data2 : ndarray, shape=(n,) or (...,n,...)
        Paired data to compute correlations between.
        Can be 1d vectors or multi-dim arrays, but must have same shape.

    axis : int or None, default: None (compute across entire flattened array)
        Array axis to treat as observations and compute correlations over.
        Correlations are computed in mass-bivariate fashion across all other array axes.
        If axis=None, correlation is computed across entire 1d flattened arrays.

    keepdims : bool, default: True
        If False, correlation `axis` is removed (squeezed out) from output.
        If True, `axis` is kept in output as singleton (length 1) axis.

    Returns
    -------
    rho : float or ndarray, shape=(...,[1,]...)
        Correlation between data1 & data2.
        For 1d data, rho is a float. For multi-d data, rho is same shape as data, but with
        `axis` reduced to length 1 (if `keepdims` is True) or removed (if `keepdims` is False).
    """
    # Rank data in each data array, either along entire flattened array or along axis
    if axis is None:
        data1_ranks = rankdata(data1)
        data2_ranks = rankdata(data2)
    else:
        data1_ranks = np.apply_along_axis(rankdata, axis, data1)
        data2_ranks = np.apply_along_axis(rankdata, axis, data2)

    # Compute Spearman rho = standard Pearson correlation on data ranks
    return correlation(data1_ranks, data2_ranks, axis=axis, keepdims=keepdims)


# =============================================================================
# Numerical utility functions
# =============================================================================
def set_random_seed(seed=None):
    """
    Seed built-in Python and Numpy random number generators with given value

    Parameters
    ----------
    seed : int or str, default: (use current clock time)
        Seed to use. If string given, converts each char to ascii and sums the
        resulting values. If no seed given, seeds based on current clock time.

    Returns
    -------
    seed : int
        Actual integer seed used
    """
    if seed is None:            seed = int(time.time()*1000.0) % (2**32 - 1)
    # Convert string seeds to int's (convert each char->ascii and sum them)
    elif isinstance(seed,str):  seed = np.sum([ord(c) for c in seed])

    # Set Numpy random number generator
    np.random.seed(seed)
    # Set built-in random number generator in Python random module
    random.seed(seed)

    return seed


def interp1(x, y, xinterp, axis=0, **kwargs):
    """
    Interpolate data over one dimension to new sampling vector

    Convenience wrapper around :func:`scipy.interpolate.interp1d` w/o weird call structure

    Parameters
    ----------
    x : array-like, shape=(n_orig,)
        Original 1d sampling vector

    y : array-like, shape=(...,n_orig,...)
        Original data sampled at values in `x`. May contain multiple data vectors sampled
        along same sampling vector `x`. The length of `y` along the interpolation axis
        `axis` must be equal to the length of `x`.  

    xinterp : array-like, shape=(n_interp,)
        Desired interpolated sampling vector. Typically `n_interp` > `n_orig`.

    axis : int, default: 0
        Specifies the axis of `y` along which to interpolate. Defaults to 1st axis.
 
    **kwargs
        Any additional keyword args are passed as-is to scipy.interpolate.interp1d

    Returns
    -------
    yinterp : ndarray, shape=(n_interp,)
        Data in `y` interpolated to sampling in `xinterp`
    """
    return interp1d(x, y, axis=axis, **kwargs).__call__(xinterp)


def gaussian(points, center=0.0, width=1.0, amplitude=1.0, baseline=0.0):
    """
    Evaluate a 1D Gaussian function with given parameters at given datapoint(s)

    Parameter values can be set for Gaussian center (mean), width (SD), amplitude,
    and additive baseline/offset. Defaults are set to generate standard normal function
    (mean=0, sd=1, amp=1, baseline-0).

    Parameters
    ----------
    points : float or ndarray, shape=(n_datapoints,)
        Datapoints to evaluate Gaussian function at

    center : float, default: 0.0
        Center (mean) of Gaussian function

    width : float, default: 1.0
        Width (standard deviation) of Gaussian function

    amplitude : float, default: 1.0
        Gaussian amplitude (multiplicative gain)

    baseline : float, default: 0.0 (no offset)
        Additive baseline value for Gaussian function

    Returns
    -------
    f_x : float or ndarray, shape=(n_datapoints,)
        Gaussian function with given parameters evaluated at each given datapoint.
        Returned as float for single datapoint input, as array for multiple datapoints.
    """
    if not np.isscalar(points): points = np.asarray(points)

    assert np.isscalar(points) or (points.ndim == 1), \
        ValueError("points must be 1d array with shape (n_datapoints,)")

    # Compute Gaussian function = exp(-(z**2)/2), where z = (x-mu)/sd
    f_x = np.exp(-0.5*((points - center)/width)**2)

    # Scale by amplitude and add in any baseline
    f_x = amplitude*f_x + baseline

    if f_x.size == 1: f_x = f_x.item()
    return f_x


# Alias gaussian() to gaussian_1d() to match format of other gaussian_*d() functions
gaussian_1d = gaussian
""" Alias of :func:`gaussian`. See there for details """


def gaussian_2d(points, center_x=0.0, center_y=0.0, width_x=1.0, width_y=1.0,
                amplitude=1.0, baseline=0.0, orientation=0.0):
    """
    Evaluate an 2D Gaussian function with given parameters at given datapoint(s)

    Parameter values can be set for Gaussian centers (means), widths (SDs), amplitude,
    additive baseline, and rotation. Defaults are set to generate unrotated 2D standard normal
    function (mean=(0,0), sd=(1,1), amp=1, no baseline offset, no rotation).

    Parameters
    ----------
    points : ndarray, shape=(n_datapoints,2=[x,y])
        Datapoints to evaluate 2D Gaussian function at.
        Each row is a distinct datapoint x to evaluate f(x) at, and the 2 columns
        correspond to the 2 dimensions (x and y) of the 2D Gaussian function.

    center_x/y : float, default: 0.0
        Center (mean) of Gaussian function along x and y dims

    width_x/y : float, default: 1.0
        Width (standard deviation) of Gaussian function along x and y dims

    amplitude : float, default: 1.0
        Gaussian amplitude (multiplicative gain)

    baseline : float, default: 0.0 (no offset)
        Additive baseline value for Gaussian function

    orientation : float, default: 0.0 (axis-aligned)
        Orientation (radians CCW from + x-axis) of 2D Gaussian. 0=oriented along
        standard x/y axes (non-rotated); 45=oriented along positive diagonal

    Returns
    -------
    f_x : float or ndarray, shape=(n_datapoints,)
        2D Gaussian function with given parameters evaluated at each given datapoint.
        Returned as float for single datapoint input, as array for multiple datapoints.
    """
    # Expand datapoints to (n_datapoints,n_dimensions=2)
    if (points.ndim == 1) and (len(points) == 2): points = points[np.newaxis,:]

    assert (points.ndim == 2) and (points.shape[1] == 2), \
        ValueError("points must have shape (n_datapoints,2=[x,y]), not: ", points.shape)

    # Compute difference d btwn datapoints and corresponding Gaussian means (x - mu_x, y - mu_y)
    d = points - np.asarray([center_x,center_y])[np.newaxis,:]

    # Rotate position (x-mu)'s by (negative of) orientation
    # Note that this is done AFTER subtracting off mu's, so mu's are NOT fitted in rotated coords
    # (ie x,y mu's = center positions in original reference frame, not rotated reference frame)
    if orientation != 0:
        theta = -orientation
        # Create rotation matrix
        rot_mx = np.asarray([[cos(theta), sin(theta)],
                             [-sin(theta), cos(theta)]])
        # Rotate mean-referenced data with rotation matrix
        d = np.matmul(d, rot_mx)

    # Compute 2D Gaussian function = exp(-(z_x**2 + z_y**2)/2), where z = (x-mu)/sd
    f_x = np.exp(-((d[:,0]/width_x)**2 + (d[:,1]/width_y)**2)/2)

    # Scale by amplitude and add in any baseline
    f_x = amplitude*f_x + baseline

    if f_x.size == 1: f_x = f_x.item()
    return f_x


def gaussian_nd(points, center=None, width=None, covariance=None, amplitude=1.0, baseline=0.0,
                check=True):
    """
    Evaluate an N-D Gaussian function with given parameters at given datapoint(s).

    Parameter values can be set for Gaussian center (mean), width (SD) or covariance,
    amplitude, and additive baseline.

    Gaussian shape can be set in one of two ways (but NOT using both):

    - `width` : computes an axis-aligned (0 off-diagonal covariance) Gaussian with SD's = `width`
    - 'covariance' : computes an N-D Gaussian with full variance/covariance matrix = `covariance`

    Defaults are set to generate N-D standard normal function (mean=0's, sd=1's,
    no off-diagonal covariance, amp=1, no baseline offset).

    Parameters
    ----------
    points : ndarray, shape=(n_datapoints,n_dims)
        Datapoints to evaluate N-D Gaussian function at.
        Each row is a distinct datapoint x to evaluate f(x) at, and each column is
        a distinct dimension of the N-dimensional Gaussian function.

    center : ndarray, shape=(n_dims,) or scalar, default: (0.0,...,0.0) (0 for all dims)
        Center (mean) of Gaussian function along each dim.
        Scalar value expanded to `n_dims`.

    width : ndarray, shape=(n_dims,) or scalar, default: (1.0,...,1.0) (1 for all dims)
        Width (standard deviation) of Gaussian function along each dim.
        Scalar value expanded to `n_dims`.
        NOTE: Can input values for either `width` OR `covariance`. Setting both raises an error.

    covariance : ndarray, shape=(n_dims,n_dims), default: (identity matrix: var's=1, covar's=0)
        Variance/covariance matrix for N-D Gaussian. Diagonals are variances for each dim, off-
        diagonals are covariances btwn corrresponding dims.
        Must be symmetric symmetric, positive semi-definite matrix
        Alternative method for setting function width/shape, allowing non-axis-aligned Gaussian.
        NOTE: Can input values for either `width` OR `covariance`. Setting both raises an error.

    amplitude : scalar, default: 1.0
        Gaussian amplitude (multiplicative gain)

    baseline : scalar, default: 0.0
        Additive baseline value for Gaussian function

    check : bool, default: True
        If True, checks if covariance is symmetric positive semidefinite; else skips slow check

    Returns
    -------
    f_x : float or ndarray, shape=(n_datapoints,)
        N-D Gaussian function evaluated at each given datapoint.
        Returned as float for single datapoint input, as array for multiple datapoints.
    """
    if (width is not None) and (covariance is not None):
        assert np.allclose(width, np.sqrt(np.diag(covariance))), \
            ValueError("Inconsistent values for `width` and `covariance`. \
                        Set one or the other, but not both")

    # Expand datapoints to (n_datapoints,n_dimensions) even if n_dims = 1
    if points.ndim == 1: points = points[np.newaxis,:]
    n_datapoints,n_dims = points.shape

    # Set defaults for center and width/covariance, expand scalars to n_dims
    if center is None:
        center = np.zeros((n_dims,))
    elif np.isscalar(center) and (n_dims > 1):
        center = np.tile(center, (n_dims,))
    else:
        center = np.asarray(center).squeeze()
        assert len(center) == n_dims, \
            ValueError("points is %d-dimensional, but center is %d-dim" % (n_dims,len(center)))

    # If `covariance` is input, use that (with some checks)
    if covariance is not None:
        assert covariance.shape == (n_dims,n_dims), \
            ValueError("`covariance` must have shape (n_dims,n_dims): (%d,%d) not (%d,%d)" % \
                        (n_dims,n_dims,*covariance.shape))
        if check:
            assert is_positive_definite(covariance, semi=True), \
                ValueError("`covariance` must be symmetric, positive semi-definite matrix")

    # If `width` is input, use to generate covariance (with var = width**2)
    elif width is not None:
        if np.isscalar(width) and (n_dims > 1):
            width = np.tile(width, (n_dims,))
        else:
            width = np.asarray(width).squeeze()
            assert len(width) == n_dims, \
                ValueError("points is %d-dimensional, but width is %d-dim" % (n_dims,len(width)))

    # If neither is input, default covariance = identity matrix (variances=1, covariances=0)
    else:
        width = np.ones((n_dims,))

    if covariance is None:
        f_x = np.zeros((n_datapoints,))
        # Step thru each dimension in N-D Gaussian
        for dim in range(n_dims):
            # Compute and cumulate z**2 = ((x-mu)/sd)**2 for current dimension
            f_x += ((points[:,dim] - center[dim]) / width[dim])**2

    else:
        d = points - center[np.newaxis,:]

        # Note: empirically, this algorithm is faster for small n, other is faster for large n
        # todo Must be more efficient way to do this???
        if n_datapoints < 10000:
            f_x = np.diagonal(d @ np.linalg.pinv(covariance) @ d.T)
        else:
            f_x = np.empty((n_datapoints,))
            for j in range(n_datapoints):
                d_j = d[j,:]
                f_x[j] = d_j @ np.linalg.pinv(covariance) @ d_j.T

    # Compute exp(-1/2*z**2), scale by amplitude and add in any baseline
    f_x = amplitude*np.exp(-f_x/2) + baseline

    if f_x.size == 1: f_x = f_x.item()
    return f_x


def is_symmetric(X):
    """
    Test if matrix is symmetric

    Parameters
    ----------
    X : ndarray or Numpy matrix, shape=Any
        Matrix to test

    Returns
    -------
    symmetric : bool
        True only if `X` is square and symmetric

    References
    ----------
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    """
    # Test for square, symmetric matrix
    if (X.ndim == 2) and (X.shape[0] == X.shape[1]) and np.array_equal(X, X.T):
        return True
    else:
        return False


def is_positive_definite(X, semi=False):
    """
    Test if matrix is symmetric positive (semi)definite

    Parameters
    ----------
    X : ndarray or Numpy matrix, shape=Any
        Matrix to test

    semi : bool, default: False
        If True, tests if positive *semi*-definite. If False, tests if positive definite.

    Returns
    -------
    pos_def : bool
        True only if `X` is square and symmetric positive (semi)definite

    References
    ----------
    https://stackoverflow.com/questions/16266720/find-out-if-matrix-is-positive-definite-with-numpy
    """
    if not is_symmetric(X):
        return False

    try:
        if semi:    np.linalg.cholesky(X + np.eye(X.shape[0]) * 1e-14)
        else:       np.linalg.cholesky(X)
        return True

    except np.linalg.LinAlgError:
        return False


def setup_sliding_windows(width, lims, step=None, reference=None,
                          force_int=False, exclude_end=None):
    """
    Generate set of sliding windows using given parameters

    Parameters
    ----------
    width : scalar
        Full width of each window

    lims : array-like, shape=(2,)
        [start end] of full range of domain you want windows to sample

    step : scalar, default: step = `width` (ie, perfectly non-overlapping windows)
        Spacing between start of adjacent windows

    reference : bool, dfault: None (just start at lim[0])
        Optionally sets a reference value at which one window starts and the
        rest of windows will be determined from there.
        eg, set = 0 to have a window start at x=0, or set = -width/2 to have a
        window centered at x=0

    force_int : bool, default: False (don't round)
        If True, rounds window starts,ends to integer values.

    exclude_end : bool, default: True if force_int==True, otherwise False
        If True, excludes the endpoint of each (integer-valued) sliding win from
        the definition of that win, to prevent double-sampling
        (eg, the range for a 100 ms window is [1,99], not [1,100])

    Returns
    -------
    windows : ndarray, shape=(n_wins,2)
        Sequence of sliding window [start,end]'s
    """
    # Default: step is same as window width (ie windows perfectly disjoint)
    if step is None: step = width
    # Default: Excluding win endpoint is default for integer-valued win's,
    #  but not for continuous wins
    if exclude_end is None:  exclude_end = True if force_int else False

    if exclude_end:
        # Determine if window params (and thus windows) are integer or float-valued
        params = np.concatenate((lims,width,step))
        is_int = np.allclose(np.round(params), params)
        # Set window-end offset appropriately--1 for int, otherwise small float value
        offset = 1 if is_int else 1e-12

    # Standard sliding window generation
    if reference is None:
        if exclude_end: win_starts = iarange(lims[0], lims[-1]-width+offset, step)
        else:           win_starts = iarange(lims[0], lims[-1]-width, step)

    # Origin-anchored sliding window generation
    #  One window set to start at given 'reference', position of rest of windows
    #  is set around that window
    else:
        if exclude_end:
            # Series of windows going backwards from ref point (flipped to proper order),
            # followed by Series of windows going forwards from ref point
            win_starts = np.concatenate((np.flip(iarange(reference, lims[0], -1*step)),
                                         iarange(reference+step, lims[-1]-width+offset, step)))

        else:
            win_starts = np.concatenate((np.flip(iarange(reference, lims[0], -1*step)),
                                         iarange(reference+step, lims[-1]-width, step)))

    # Set end of each window
    if exclude_end: win_ends = win_starts + width - offset
    else:           win_ends = win_starts + width

    # Round window starts,ends to nearest integer
    if force_int:
        win_starts = np.round(win_starts)
        win_ends   = np.round(win_ends)

    return np.stack((win_starts,win_ends),axis=1)


# =============================================================================
# Data indexing and reshaping functions
# =============================================================================
def index_axis(data, axis, idxs):
    """
    Utility to dynamically index into a arbitrary axis of an ndarray

    Similar to function of Numpy take and compress functions, but this can take either
    integer indexes, boolean indexes, or a slice object. And this is generally much faster.

    Parameters
    ----------
    data : ndarray, shape=Any
        Array of arbitrary shape, to index into given axis of.

    axis : int
        Axis of ndarray to index into

    idxs :  array-like, shape=(n_selected,), dtype=int or array-like, shape=(axis_len,), dtype=bool or Slice object
        Indexing into given axis of array to perform, given as list of integer indexes,
        as boolean vector, or as Slice object

    Returns
    -------
    data : ndarray
        Input array with indexed values selected from given axis.
    """
    # Generate list of slices, with ':' for all axes except <idxs> for <axis>
    slices = axis_index_slices(axis, idxs, data.ndim)

    # Use slices to index into data, and return sliced data
    return data[slices]


def axis_index_slices(axis, idxs, ndim):
    """
    Generate list of slices, with ':' for all axes except `idxs` for `axis`,
    to use for dynamic indexing into an arbitary axis of an ndarray

    Parameters
    ----------
    axis : int
        Axis of ndarray to index into

    idxs : array_like, shape=(n_selected,), dtype=int or array-like, shape=(axis_len,), dtrype=bool or Slice object
        Indexing into given axis of array to perform, given as list of integer indexes,
        as boolean vector, or as Slice object

    ndim : int
        Number of dimensions in ndarray to index into

    Returns
    -------
    slices : tuple of slices
        Indexing tuple to use to index into given axis of ndarray as:
        selected_values = array[slices]
    """
    # Initialize list of null slices, equivalent to [:,:,:,...]
    slices = [slice(None)] * ndim

    # Set slice for <axis> to desired indexing
    slices[axis] = idxs

    # Convert to tuple bc indexing arrays w/ a list is deprecated
    return tuple(slices)


def standardize_array(data, axis=0, target_axis=0):
    """
    Reshape multi-dimensional data array to standardized 2D array (matrix-like) form,
    with `axis` shifted to `target_axis` for analysis

    Parameters
    ----------
    data : ndarray, shape=(...,n,...)
        Data array of arbitrary shape.

    axis : int, default: 0
        Axis of data to move to `target_axis` for subsequent analysis

    target_axis : int, default: 0
        Array axis to move `axis` to for subsequent analysis.
        MUST be 0 (first axis) or -1 (last axis).

    Returns
    -------
    data  : ndarray, shape=(n,m) or (m,n)
        Data array w/ `axis` moved to `target_axis`, and all other axes unwrapped
        into single dimension, where m = prod(shape[axes != axis])

        NOTE: Even 1d (vector) data is expanded into 2d (n,1) | (1,n) array to
        standardize for calling code.

    data_shape : tuple, shape=(data.ndim,)
        Original shape of input data array
    """
    data = np.asarray(data)
    if axis < 0: axis = data.ndim + axis
    if target_axis < 0: target_axis = data.ndim + target_axis

    assert target_axis in [0,data.ndim-1], \
        ValueError("target_axis set = %d. Must be 0 (first axis) or -1 (last axis)" % target_axis)

    if target_axis == 0:    return _standardize_to_axis_0(data, axis=axis)
    else:                   return _standardize_to_axis_end(data, axis=axis)


def undo_standardize_array(data, data_shape, axis=0, target_axis=0):
    """
    Undo effect of standardize_array() -- reshapes data array from unwrapped
    2D (matrix-like) form back to ~ original multi-dimensional form, with `axis`
    shifted back to original location (but allowing that `data.shape[axis]` may have changed)

    Parameters
    ----------
    data : ndarray, shape=(axis_len,m) or (m,axis_len)
        Standardized data array -- with `axis` moved to `target_axis`, and all
        axes != `target_axis` unwrapped into single dimension, where
        m = prod(shape[axes != axis])

    data_shape : tuple, shape=(data_orig.ndim,)
        Original shape of data array. Second output of standardize_array.

    axis : int, default: 0
        Axis of original data moved to `target_axis`, which will be shifted
        back to original axis

    target_axis : int, default: 0
        Array axis `axis` was moved to for subsequent analysis
        MUST be 0 (first axis) or -1 (last axis)

    Returns
    -------
    data : ndarray,. shape=(...,axis_len,...)
        Data array reshaped back to original shape
    """
    data = np.asarray(data)
    data_shape  = np.asarray(data_shape)
    if axis < 0: axis = len(data_shape) + axis
    if target_axis < 0: target_axis = len(data_shape) + target_axis

    assert target_axis in [0,len(data_shape)-1], \
        ValueError("target_axis set = %d. Must be 0 (first axis) or -1 (last axis)" % target_axis)

    if target_axis == 0:    return _undo_standardize_to_axis_0(data, data_shape, axis=axis)
    else:                   return _undo_standardize_to_axis_end(data, data_shape, axis=axis)


def data_labels_to_data_groups(data, labels, axis=0, groups=None, max_groups=None):
    """
    Convert (data,labels) pair to tuple of (data_1,data_2,...,data_k) where each `data_j`
    corresponds to all datapoints in input data associated with a given label value
    (eg group/condition/etc.).

    Parameters
    ----------
    data : ndarray, shape=(...,N,...)
        Array of multi-class data. Arbitrary shape, but `axis` must correspond to
        observations/trials and have same length as `labels`.

    labels : array-like, shape=(N,)
        List of labels corresponding to each observation in data.

    axis : int, default: 0
        Axis of data array corresponding to observations/trials in labels

    groups : array-like, shape=(n_groups,), default: np.unique(labels) (all unique values)
        Which group labels from `labels` to include. Useful to ensure a specific group order
        in outputs or to retain only subset of groups in labels.

    max_groups : int, default: None
        Maximum number of allowed groups in data. Raises an error if len(groups) > max_groups.
        Set=None to allow any number of groups.

    Returns
    -------
    data_1,...,data_k : ndarray (...,n_j,...)
        `n_groups` arrays of data corresponding to each group in groups,
        each returned in a separate variable. Shape is same as input data on all axes
        except `axis`, which is reduced to the n for each group.
    """
    labels = np.asarray(labels).squeeze()

    assert labels.ndim == 1, ValueError("labels must be 1d array-like variable")
    assert len(labels) == data.shape[axis], \
        ValueError("Data must have same length (%d) as labels (%d) along <axis>"
                   % (data.shape[axis], len(labels)))

    # Find all unique values in labels (if not input)
    if groups is None: groups = np.unique(labels)
    n_groups = len(groups)

    assert (max_groups is None) or (n_groups <= max_groups), \
        ValueError("Input labels have %d distinct values (only %d allowed)"
                   % (n_groups,max_groups))

    return tuple([data.compress(labels == group, axis=axis) for group in groups])


def data_groups_to_data_labels(*data, axis=0, groups=None):
    """
    Convert tuple of (data_1,data_2,...,data_k) to (data,labels) pair, where a unique label
    is associated with all datapoints in each data group `data_j` (eg group/condition/etc.).

    Parameters
    ----------
    data_1,...,data_k : ndarray (...,n_j,...)
        n_groups arrays of data corresponding to each group in groups, each input in a
        separate variable. Shape is arbitrary, but `axis` must correspond to observations/trials
        and all axes but `axis` must have same length across all data arrays.

    axis : int, default: 0
        Axis of data arrays corresponding to observations/trials in labels

    groups : array_like, shape=(n_groups,), default: integers from 0 - n_groups-1
        List of names of each group in input data to use in labels.

    Returns
    -------
    data : ndarray, shape=(...,N,...)
        Array of multi-class data. Shape is same as input data on all axes
        except `axis`, which expands to the sum of all group n's.

    labels : array-like, shape=(N,)
        List of labels corresponding to each observation in data.
    """
    if isinstance(data,tuple) and len(data) == 1:
        raise TypeError("Seems you are missing dereferencer '*' for argument <data>")
    
    n_groups = len(data)
    if groups is None: groups = np.arange(n_groups)
    n_per_group = [x.shape[axis] for x in data]

    labels = np.hstack([np.tile(group, (n,)) for n,group in zip(n_per_group,groups)])
    return np.concatenate(data, axis=axis), labels


# =============================================================================
# Other utility functions
# =============================================================================
def iarange(*args, **kwargs):
    """
    Implements :func:`np.arange` with an inclusive endpoint. Same inputs as np.arange(),
    same output, except ends at stop, not stop - 1 (or more generally stop - step)

    Like np.arange, iarange can be called with a varying number of positional arguments:

    - iarange(stop) : Values are generated within the closed interval [0,stop]
        (in other words, the interval including both start AND stop).

    - iarange(start,stop) : Values are generated within the closed interval [start,stop].

    - iarange(start,stop,step) : Values are generated within the closed interval [start,stop],
        with spacing between values given by step.

    Parameters
    ----------
    start : int, default: 0
        Starting index for range

    stop : int, default: 0
        *Inclusive* ending index for range

    step : int, default: 1
        Stepping value for range

    **kwargs :
        Any other kwargs passed directly to :func:`np.arange` function

    Returns
    -------
    range : ndarray
        Array of evenly spaced values from `start` to `stop` (*inclusive*) in length `step` steps
    """
    # Parse different argument formats
    if len(args) == 1:
        start = 0
        stop = args[0]
        step = 1
    elif len(args) == 2:
        start = args[0]
        stop = args[1]
        step = 1
    elif len(args) == 3:
        start = args[0]
        stop = args[1]
        step = args[2]

    # Offset to get final value in sequence is 1 for int-valued step, small float otherwise
    offset = 1 if isinstance(step,int) else 1e-12
    # Make offset negative for a negative step
    if step < 0: offset = -offset

    return np.arange(start, stop+offset, step, **kwargs)


def unsorted_unique(x, axis=None, **kwargs):
    """
    Implements :func:`np.unique` without sorting, ie maintaining original order of unique
    elements as they are found in `x`.

    Parameters
    ----------
    x : ndarray, shape:Any
        Array to find unique values in

    axis : int, default: None (unique values over entire array)
        Axis of array to find unique values on. If None, finds unique values in entire array.

    **kwargs
        All other keyword passed directly to np.unique

    Returns
    -------
    unique: ndarray
        Unique values in `x`, in order in which they appear in `x`

    References
    ----------
    https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    """
    x = np.asarray(x)
    if axis is not None:
        idxs = np.unique(x, return_index=True, axis=axis, **kwargs)[1]
        return index_axis(x, axis, np.sort(idxs))
    else:
        x = x.flatten()
        idxs = np.unique(x, return_index=True, axis=axis, **kwargs)[1]
        return x[np.sort(idxs)]


def isarraylike(x):
    """
    Test if variable `x` is "array-like": np.ndarray, list, or tuple

    Returns True if x is array-like, False otherwise
    """
    return isinstance(x, (list, tuple, np.ndarray))


def isnumeric(x):
    """
    Test if dtype of ndarray `x` is numeric (some subtype of int,float,complex)

    Returns True if x.dtype is numeric, False otherwise
    """
    return np.issubdtype(np.asarray(x).dtype, np.number)


def isunix():
    """ Return true iff current system OS is Linux/UNIX (but not Mac OS) """
    return (os.name == 'posix') and (platform.system() == 'Linux')


def ismac():
    """ Return true iff current system OS is Mac OS """
    return (os.name == 'posix') and (platform.system() == 'Darwin')


def ispc():
    """ Return true iff current system OS is PC Windows """
    return platform.system() == 'Windows'


def object_array_equal(data1, data2, comp_func=np.array_equal, reduce_func=np.all):
    """
    Determine if each object element within two object arrays is equal

    Parameters
    ----------
    data1,data2 : ndarray, shape= Any
        Two arrays to determine elementwise equality of. Must have same shape if using
        anything other than defaults for `comp_func`, `reduce_func` (bc we have no way of
        knowing how to deal with this).

    comp_func : callable, default: np.array_equal (True iff elements have same shape and values)
        Comparison function used to determine equality of each element
        If None, no reduction of the comparison results is performed.

    reduce_func : callable, default: np.all (True iff ALL objects in array are elementwise True)
        Optional function to reduce equality results for each element across entire array

    Returns
    -------
    equal : bool or ndarray, shape=data.shape, dtype=bool
        Reflects equality of each object element in data1,data2.
        If `reduce_func` is None, this is the elementwise equality of each object,
        and has same shape as data1,2.
        Otherwise, elementwise equality is reduced across the array using `reduce_func`,
        and this returns as a single scalar bool.

        If data1,2 have different shapes: we return False if `comp_func` is array_equal
        and `reduce_func` is np.all; otherwise an error is raised (don't know how to compare).
    """
    if data1.shape != data2.shape:
        # For vanilla array_equal comparison, different shapes imply equality is False
        if (comp_func is np.array_equal) and (reduce_func is np.all):
            return False
        # General case: we don't really know how to deal with different shapes
        # w/o knowing comp_func and reduce_func
        else:
            raise ValueError("Unsure how to compare data1 and data2 with different shapes")

    equal = np.empty_like(data1)

    # Create 1D flat iterator to iterate over arbitrary-shape data arrays
    data_flat = data1.flat
    for _ in range(data1.size):
        # Multidim coordinates into data array
        coords = data_flat.coords

        # Run `comp_func` on current array element
        equal[coords] = comp_func(data1[coords], data2[coords])

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(data_flat)

    # Perform any reduction operation across output array
    if reduce_func is not None: equal = reduce_func(equal)

    return equal


def object_array_compare(data1, data2, comp_func=np.equal, reduce_func=None):
    """
    Compares object elements within two object arrays using given comparison function

    Parameters
    ----------
    data1,data2 : ndarray, shape=Any (but data1.shape = data2.shape)
        Two arrays to determine elementwise equality of

    comp_func : callable, default: np.equal (True/False for each value w/in each object element)
        Comparison function used to compare each object element.

    reduce_func : callable, Default: None (don't perform any reduction on result)
        Optional function to reduce comparison results for each element across entire array
        If None, no reduction of the comparison results is performed.

    Returns
    -------
    equal : ndarray | bool
        Reflects comparison of each object element in data1,data2.
        If `reduce_func` is None, this is the elementwise comparison of each object,
        and has same shape as data1,2.
        Otherwise, elementwise comparison is reduced across the array using `reduce_func`,
        and this returns as a single scalar bool.
    """
    assert data1.shape == data2.shape, \
        ValueError("data1 and data2 must have same shape for comparison")

    out = np.empty_like(data1)

    # Create 1D flat iterator to iterate over arbitrary-shape data arrays
    data_flat = data1.flat
    for _ in range(data1.size):
        # Multidim coordinates into data array
        coords = data_flat.coords

        # Run `comp_func` on current array element
        out[coords] = comp_func(data1[coords], data2[coords])

        # Perform any reduction operation across values in current element
        if reduce_func is not None: out[coords] = reduce_func(out[coords])

        # Iterate to next element (list of spike times for trial/unit/etc.) in data
        next(data_flat)

    return out


def concatenate_object_array(data, axis=None, sort=False):
    """
    Concatenate objects across one or more axes of an object array.
    Useful for concatenating spike timestamps across trials, units, etc.

    Parameters
    ----------
    data : ndarray, shape=Any, dtype=object (containing 1d lists/arrays)

    axis : int or list of int or None, default: None
        Axis(s) to concatenate object array across.
        Set = list of ints to concatenate across multiple axes.
        Set = None to concatenate across *all* axes in data.

    sort : bool, default: False
        If True, sorts items in concatenated list objects

    Returns
    -------
    data : list or ndarray, dtype=object
        Concatenated object(s).
        If axis is None, returns as single list extracted from object array.
        Otherwise, returns as object ndarray with all concatenated axes
        reduced to singletons.

    Examples
    --------
    data = [[[1, 2],    [3, 4, 5]],
            [[6, 7, 8], [9, 10]  ]]

    concatenate_object_array(data,axis=0)
    >> [[1,2,6,7,8], [3,4,5,9,10]]

    concatenate_object_array(data,axis=1)
    >> [[1,2,3,4,5], [6,7,8,9,10]]
    """
    if ~isinstance(data,np.ndarray): data = np.asarray(data, dtype=object)
    assert data.dtype == object, \
        ValueError("data is not an object array. Use np.concatenate() instead.")

    # Convert axis=None to list of *all* axes in data
    if axis is None:    axis_ = 0 if data.ndim == 1 else list(range(data.ndim))
    else:               axis_ = axis

    # If <axis> is a list of multiple axes, iterate thru each axis,
    # recursively calling this function on each axis in list
    if not np.isscalar(axis_):
        for ax in axis_:
            # Only need to sort for final concatenation axis
            sort_ = sort if ax == axis_[-1] else False
            data = concatenate_object_array(data,axis=ax,sort=sort_)

        # Extract single object if we concatenated across all axes (axis=None)
        if axis is None: data = data.item()

        return data

    # If concatenation axis is already a singleton, we are done, return as-is
    if data.shape[axis_] == 1: return data


    # Reshape data so concatenation axis is axis 0, all other axes are unwrapped to 2d array
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_series = data.shape[1]

    data_concat = np.empty((1,n_series),dtype=object)

    for j in range(n_series):
        # Concatenate objects across all entries
        data_concat[0,j] = np.concatenate([values for values in data[:,j]])
        # Sort items in concatenated object, if requested
        if sort: data_concat[0,j].sort()

    # Reshape data back to original axis order and shape
    data_concat = undo_standardize_array(data_concat, data_shape, axis=axis, target_axis=0)

    return data_concat

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic descriptive and inferential statistics for circular/angular/directional data

Overview
--------
Most standard statistics, defined for linear variables, aren't appropriate for variables that are
circular-valued, such as angles, times of day, neural oscillation phase, etc. This module provides
appropriate analogous statistics for circular variables. It includes descriptive stats (mean,
variance/SD), inferential stats (t-test, ANOVA, correlation, regression), as well as several
useful utilities for dealing with circular data.

For all functions, angles may be given in either radians [default] or degrees (by setting the
`degrees` parameter=True). Axially-symmetric data whose values only range over pi radians (180 deg)
(ie where 0 and 180 deg "mean" the same thing) can also be input and analyzed appropriately by
setting the `axial` parameter=True.

Most functions perform operations in a mass-univariate (or mass-bivariate) manner. This means
that rather than embedding function calls in for loops over channels, timepoints, etc., like this::

    for channel in channels:
        for timepoint in timepoints:
            results[timepoint,channel] = compute_something(data[timepoint,channel])

You can instead execute a single call on ALL the data, labeling the relevant axis
for the computation (usually trials/observations here), and it will run in parallel (vectorized)
across all channels, timepoints, etc. in the data, like this:

``results = compute_something(data, axis)``

Many functions also have the option of setting axis=None, in which case the relevant computation
is performed across the entire flattened (unraveled) array(s), as in Numpy reduction functions,
(np.mean, np.var, etc.).

Function list
-------------
Utilities
^^^^^^^^^
- wrap :                Wraps set of angles into given range
- circ_distance :       Absolute/unsigned circular distance between angles
- circ_subtract :       Circular subtraction (signed circular difference) between angles
- circ_subtract_complex : Circular subtraction for angles expressed as complex numbers
- amp_phase_to_complex : Combines amplitudes and phases into complex numbers
- complex_to_amp_phase : Decomposes complex numbers into amplitudes and phases

Descriptive stats
^^^^^^^^^^^^^^^^^
- circ_mean :           Circular mean of set of angles
- circ_average :        Circular weighted mean of set of angles
- circ_r :              Raw resultant length (R) of set of angles
- circ_rbar :           Mean resultant length (Rbar) of set of angles
- circ_rbar2_unbiased : Unbiased estimate of squared resultant length (Rbar) of set of angles
- circ_var :            Circular variance of set of angles (multiple definitions to choose from)
- circ_std :            Circular standard deviation of set of angles (multiple definitions)
- circ_dispersion :     Circular dispersion of set of angles
- circ_sem :            Circular standard error of the mean of set of angles
- von_mises_kappa :     Estimate VonMises dist'n concentration parameter kappa (~circ normal SD)
- circ_hist :           Computes histogram on circular data

Inferential stats
^^^^^^^^^^^^^^^^^
- circ_mean_test :      Test circular mean angle against specific value (~circ 1-sample t-test)
- circ_ANOVA1 :         Watson-Williams test for diff of circ means btwn data groups (~circ ANOVA1)
- circ_circ_correlation : Circular-circular correlation btwn pairs of circular observations
- circ_linear_correlation : Circular-linear correlation between paired circular and linear data
- circ_linear_regression : Circular-linear multiple regression between paired circ and linear data

Acknowledgements
^^^^^^^^^^^^^^^^
Much of this was adapted from Fisher's 1993 textbook "Statistical Analysis of Circular Data",
which is a useful reference for circular stats. Also recommended is ch. 26 of Zar's 1999
textbook "Biostatistical Analysis" and Jammalamadaka & SenGupta's 2001 book
"Topics in Circular Statistics"

Many functions here were inspired by analogous ones in Philipp Berens's CircStat Toolbox for MATLAB.
For more info about that toolbox, see:
www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics

P. Berens (2009) "CircStat: A Matlab Toolbox for Circular Statistics" J Stat Software
http://www.jstatsoft.org/v31/i10
"""
# Created on Mon Nov 16 16:38:46 2020
# @author: sbrincat

from math import pi, inf
from warnings import warn
from itertools import product
from collections.abc import Iterator
import numpy as np

from scipy.stats import norm, f, chi2
from scipy.optimize import minimize

from spynal.utils import standardize_array, undo_standardize_array, correlation


# =============================================================================
# Utility functions
# =============================================================================
def wrap(data, limits=None, degrees=False, axial=False):
    """
    Wrap circular data into specified range

    Parameters
    ----------
    data : float or ndarray, shape=Any
        Set of angles (in radians or degrees, see `degrees`).

    limits : array-like, shape=(2,), default: (0,2*pi) or (0,360)
        Range to wrap angles into, given as [min,max] angle. If angles are given in radians,
        this defaults to (0,2*pi) for circular data, (0,pi) for axial data.
        If given in degrees it defaults to (0,360) and (0,180), respectively.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.
        Only used to set default for `limits` if not input.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.
        Only used to set default for `limits` if not input.

    Returns
    -------
    data : float or ndarray, shape=Any
        Data wrapped into given range. Same type/shape as input.
    """
    if limits is None:
        if axial:   limits = (0,180) if degrees else (0,pi)
        else:       limits = (0,360) if degrees else (0,2*pi)
    assert limits[1] > limits[0], ValueError("limits[1] must be > limits[0]")

    diff = limits[1] - limits[0]

    return np.mod(data - limits[0], diff) + limits[0]


def circ_distance(data1, data2, degrees=False, axial=False):
    """
    Elementwise absolute/unsigned circular distance between pair(s) of angles

    Parameters
    ----------
    data1/data2 : float or ndarray, shape=Any
        Pair of angles or arrays of angles (in radians or degrees, see `degrees`).
        Shape is arbitrary, but must be same for both.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.
        Only used to set default for `limits` if not input.

    Returns
    -------
    d : float or ndarray, shape=Any
        Unsigned circular distance betweeen angles. Same type, shape, and units as input.
        Ranges from [0,pi]/[0,180] if axial is not True; [0,pi/2]/[0,90] if it is True.
    """
    if axial:   limits = (0,180) if degrees else (0,pi)
    else:       limits = (0,360) if degrees else (0,2*pi)

    data1 = wrap(data1, limits=limits)
    data2 = wrap(data2, limits=limits)

    d = np.abs(data1 - data2)
    return np.minimum(d, limits[1] - d)


def circ_subtract(data1, data2, degrees=False, axial=False):
    """
    Elementwise signed dircular difference (circular subtraction): data1 - data2

    Parameters
    ----------
    data1/data2 : float or ndarray, shape=Any
        Pair of angles or arrays of angles (in radians or degrees, see `degrees`).
        Shape is arbitrary, but must be same for both.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    Returns
    -------
    d : float or ndarray, shape=Any
        Signed circular difference betweeen angles. Same type, shape, and units as input.
        Ranges from [-pi,pi]/[-180,180] if axial is not True; [-pi/2,pi/2]/[-90,90] if it is True.
    """
    data1 = _check_and_process_data(data1, degrees, axial)
    data2 = _check_and_process_data(data2, degrees, axial)

    d = data1 - data2
    d = np.arctan2(np.sin(d), np.cos(d))

    # Note: Alternative formula based on complex representation -- ~2x - 3x slower
    # d = np.angle(np.exp(1j*data1) / np.exp(1j*data2))

    if degrees: d = np.rad2deg(d)
    if axial: d = d/2

    return d


def amp_phase_to_complex(amp, theta, degrees=False):
    """
    Convert vector amplitude and phase angle of to complex variable

    Parameters
    ----------
    amp : float or ndarray, shape=Any
        Amplitude(s) of one or more vectors to convert

    theta : float or ndarray, shape=Any
        Angles of one or more vector(s) (in radians or degrees, see `degrees`).
        Shape is arbitrary, but must be same as `amp`.

    degrees : bool, default: False
        Set=True if `theta` is given in degrees, False if given in radians.

    Returns
    -------
    c : float or ndarray, shape=Any
        Complex value(s) corresponding to given vector amplitude(s) and angle(s).
        Same type and shape as input.
    """
    if degrees:
        theta = np.deg2rad(theta)

    return amp * np.exp(1j*theta)


def complex_to_amp_phase(c, degrees=False):
    """
    Convert complex variable to vector amplitude (magnitude) and phase angle

    Parameters
    ----------
    c : float or ndarray, shape=Any
        Complex value representation of one or more vectors

    degrees : bool, default: False
        If True, `theta` is returned in degrees; if False, returned in radians.

    Returns
    -------
    amp : float or ndarray, shape=Any
        Amplitude(s) of input complex vector(s). Same type and shape as input.

    theta : float or ndarray, shape=Any
        Angle(s) of input complex vector(s) (in radians or degrees, see `degrees`).
    """
    if degrees: return np.abs(c), np.rad2deg(np.angle(c))
    else:       return np.abs(c), np.angle(c)


# =============================================================================
# Descriptive statistics functions
# =============================================================================
def circ_mean(data, axis=None, degrees=False, axial=False, keepdims=False):
    """
    Circular mean of array of angle data along given axis. Circular analog of :func:`np.mean`.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute mean along.
        If None, compute mean across entire flattened array (as in :func:`np.mean`).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    mu : float or ndarray, shape=(...,[1,],...)
        Circular mean of given set of angles. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    Notes
    -----
    Algorithm using sines/cosines is ~ an order of magnitude than one
    using complex vector mean: np.angle(np.exp(1j*data).mean(axis=axis))
    """
    data = _check_and_process_data(data, degrees, axial)

    sines = np.sin(data).sum(axis=axis,keepdims=keepdims)
    cosines = np.cos(data).sum(axis=axis,keepdims=keepdims)
    mu = np.arctan2(sines, cosines)

    if axial: mu = mu/2
    if degrees: mu = np.rad2deg(mu)

    return mu


def circ_average(data, axis=None, weights=None, degrees=False, axial=False, keepdims=False):
    """
    Circular weighted mean of array of angle data along given axis.

    Circular analog of :func:`np.average`.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute mean along.
        If None, compute mean across entire flattened array (as in :func:`np.average`).

    weights : ndarray, shape=(n_obs,) or (...,n_obs,...)
        Array of weights for weighted average.
        Should be vector with length data.shape[axis] or array of same shape as data.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    mu : float or ndarray, shape=(...,[1,],...)
        Circular weighted average of given set of angles. Returns a float for 1D data or
        `axis` is None. Otherwise, returns array the same shape as data, but with `axis`
        reduced to length 1 (if keepdims=True) or removed (if keepdims=False).
    """
    if weights is None:
        return circ_mean(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    else:
        data = _check_and_process_data(data, degrees, axial)

        # Expand vector-valued weights to same dimensionality as data, so it will broadcast
        if (data.ndim > 1) and (weights.ndim == 1):
            assert axis is not None, "For axis=None, `weights` and `data` must have same shape"
            assert len(weights) == data.shape[axis], \
                "`weights` (%d) must have same length as `data` (%d) along observation `axis`" % \
                (len(weights),data.shape[axis])

            slices = [None] * data.ndim
            slices[axis] = slice(None)
            weights = weights[tuple(slices)]
        else:
            assert np.array_equal(weights.shape,data.shape), \
                "`weights` must have same shape as `data`"

        denom   = weights.sum(axis=axis,keepdims=keepdims)
        sines   = (np.sin(data)*weights).sum(axis=axis,keepdims=keepdims) / denom
        cosines = (np.cos(data)*weights).sum(axis=axis,keepdims=keepdims) / denom

        mu = np.arctan2(sines, cosines)

        if axial: mu = mu/2
        if degrees: mu = np.rad2deg(mu)

        return mu


circ_weighted_mean = circ_average
""" Alias for circ_average, which is a dumb name (thanks, Numpy) """


def circ_r(data, axis=None, degrees=False, axial=False, keepdims=False):
    """
    Raw resultant vector length (Rbar) of array of angle data along given axis

    Note that this is NOT the mean resultant length (as in :func:`circ_rbar`),
    but the raw, non-normalized resultant, with range [0,+Inf].

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians or degrees, see `degrees`). Arbitrary shape.

    axis : int, default: None
        Axis to compute Rbar along. If None, compute across entire flattened array (as in np.mean).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    rbar : float or ndarray, shape=(...,[1,],...)
        Resultant vector length of given set of angles, in range [0,Inf].
        Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.7-2.10
    """
    data = _check_and_process_data(data, degrees, axial)

    # Compute sum of sines and cosines of data
    # Note: This algorithm is ~ an order of magnitude than the one
    # using complex vector mean: np.abs(np.exp(1j*data).mean(axis=axis))
    sines = np.sin(data).sum(axis=axis,keepdims=keepdims)
    cosines = np.cos(data).sum(axis=axis,keepdims=keepdims)

    # Compute resultant length and ensure it's non-negative (due to float point error)
    return np.sqrt(cosines**2 + sines**2).clip(0,np.inf)


def circ_rbar(data, axis=None, degrees=False, axial=False, keepdims=False):
    """
    Mean resultant vector length (Rbar) of array of angle data along given axis

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians or degrees, see `degrees`). Arbitrary shape.

    axis : int, default: None
        Axis to compute Rbar along. If None, compute across entire flattened array (as in np.mean).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    rbar : float or ndarray, shape=(...,[1,],...)
        Mean resultant of given set of angles, in range [0,1].
        Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.7-2.10
    """
    R = circ_r(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    # Number of observations contributing to computation (all elem's or along given axis)
    n = data.size if axis is None else data.shape[axis]

    # Compute mean resultant length and ensure it's in range [0,1] (due to float point error)
    return (R / n).clip(0,1)


def circ_rbar2_unbiased(data, axis=None, degrees=False, axial=False, keepdims=False):
    """
    Unbiased estimate of squared resultant vector length (Rbar) of array of angle data
    along given axis

    In the phase synchrony literature, this estimator is called the Pairwise Phase Consistency
    (PPC), while the original biased Rbar is called the Phase Locking Value (PLV).

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians or degrees, see `degrees`). Arbitrary shape.

    axis : int, default: None
        Axis to compute Rbar along. If None, compute across entire flattened array (as in np.mean).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    rbar2 : float or ndarray, shape=(...,[1,],...)
        Unbiased estimated of squared circular Rbar of given set of angles.
        Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    - Kutil (2012) Statistic https://doi.org/10.1080/02331888.2010.543463
    - Kornblith, Buschman, Miller (2015) https://doi.org/10.1093/cercor/bhv182
    """
    Rbar = circ_rbar(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)
    n = data.size if axis is None else data.shape[axis]
    return (n*Rbar**2 - 1) / (n - 1)


def circ_var(data, axis=None, method='Fisher_Mardia', degrees=False, axial=False, keepdims=False):
    """
    Circular variance of array of angle data along given axis

    Uses one of the many alternative definitions for "circular variance" depending on `method`.
    By default, uses Fisher/Mardia circular variance (1 - Rbar), which is the most common.
    However, arguably, 'circvar2' is most well-behaved and closely-aligned to variance of
    linear data (range is 0-Inf, approximates linear for small variance).
    See below for formulas and sources for each.

    NOTE: sqrt(circ_var) is in general NOT an appropriate measure of
    standard deviation for circular data. Use :func:`circ_std` instead.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute var along. If None, compute across entire flattened array (as in np.var).

    method : str, default: 'Fisher_Mardia'
        Specifies which of several alternative definitions of "circular variance" to use:

        - 'circ_variance'/'Fisher_Mardia': Fisher/Mardia "circular variance"
            - var = 1 - Rbar
            - Range: 0-1 rad**2
            - Refs: Zar p.604 eq. 26.17, Mardia'72 p.45

        - 'angular_variance'/'Batschelet': Batschelet's "angular variance"
            - var = 2*(1 - Rbar)
            - Range: 0-2 rad**2
            - Refs: Zar p.604 eq. 26.18, Batschelet p.34

        - 'circvar2': Mardia's circvar2 estimator
            - var = -2*log(Rbar)
            - Range: 0-Inf rad**2
            - Refs: Zar p.604 eq. 26.19, Mardia'72 p.24, Fisher p.32 eq. 2.12

        - 'dispersion': Fisher Circular dispersion
            - rho2 = sum(cos(2*(theta-mu)))/ n
            - var = (1 - rho2) / (2 * rbar**2)
            - Range: 0-Inf rad**2
            - Refs: Fisher p.76, eq.4.21

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    var : float or ndarray, shape=(...,[1,],...)
        Circular variance of given set of angles. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    - Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.3.1
    - Zar (1999) "Biostatistical Analysis" 4th ed. sxn. 26.5
    - Mardia (1972) J Royal Stat Soc B https://www.jstor.org/stable/2984782
    - Batschelet (1981) "Circular Statistics in Biology"
    """
    method = method.lower()

    if method != 'dispersion':
        rbar = circ_rbar(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    # Fisher/Mardia "circular variance"
    if method in ['circ_variance','fisher_mardia']:
        return 1 - rbar

    # Batschelet "angular variance"
    elif method in ['angular_variance','batschelet']:
        return 2*(1 - rbar)

    # Mardia circvar2
    elif method == 'circvar2':
        return -2*np.log(rbar)

    # Fisher Circular dispersion
    elif method == 'dispersion':
        return circ_dispersion(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    else:
        raise ValueError("Unsupported value '%s' set for <method>" % method)


circ_variance = circ_var
""" Alias circ_var as circ_variance """


def circ_std(data, axis=None, method='Fisher_Mardia', degrees=False, axial=False, keepdims=False):
    """
    Circular standard deviation of array of angle data along given axis

    Uses one of the many alternative definitions for "circular standard deviation" depending
    on `method`. See below for formulas and sources for each.
    By default, uses Fisher/Mardia "circvar2" estimator, which ranges from
    0 - Inf, and produces similar values to linear SD for small angles.

    NOTE: sqrt(circ_var) is in general NOT an appropriate measure of
    standard deviation for circular data. Use this instead.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute var along. If None, compute across entire flattened array (as in np.var).

    method : str, default: 'Fisher_Mardia'
        Specifies which of several alternative definitions of "circular standard deviation" to use.

        -'Fisher_Mardia'/'circ_std': circular standard deviation = sqrt(Mardia's circvar2)
            - sd = sqrt(-2*log(Rbar))
            - Range: 0-Inf rad
            - Refs: Zar p.604 eq. 26.21, Mardia'72 pp.24,74, Fisher p.32 eq. 2.12

        - 'angular_deviation'/'Batschelet': Mean angular deviation
            - sd = sqrt(2*(1 - Rbar))
            - Range: 0-sqrt(2) rad
            - Refs: Zar p.604 eq. 26.20

        - 'dispersion': Square root of Fisher circular dispersion
            - rho2 = sum(cos(2*(theta-mu)))/ n
            - d = (1 - rho2) / (2 * rbar**2)
            - sd = sqrt(d)
            - Range: 0-Inf rad
            - Refs: Fisher p.76, eq.4.21

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    sd : float or ndarray, shape=(...,[1,],...)
        Circular std dev of given set of angles. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    - Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.3.1
    - Zar (1999) "Biostatistical Analysis" 4th ed. sxn. 26.5
    - Mardia (1972) J Royal Stat Soc B https://www.jstor.org/stable/2984782
    """
    method = method.lower()

    if method != 'dispersion':
        rbar = circ_rbar(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    # Fisher/Mardia "circular standard deviation"
    if method in ['circ_std','fisher_mardia']:
        return np.sqrt(-2*np.log(rbar))

    # Mean angular deviation
    elif method in ['angular_deviation','batschelet']:
        return np.sqrt(2*(1 - rbar))

    # Square root of Fisher Circular dispersion
    elif method == 'dispersion':
        return np.sqrt(circ_dispersion(data, axis=axis, degrees=degrees, axial=axial,
                                       keepdims=keepdims))

    else:
        raise ValueError("Unsupported value '%s' set for <method>" % method)


def circ_dispersion(data, axis=None, degrees=False, axial=False, keepdims=False):
    """
    Circular dispersion of array of angular data along given axis.

    A measure of circular standard error can be obtained as sqrt(dispersion/n) (Fisher eqn. 4.21).

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute dispersion along. If None, compute across entire flattened array (as in np.var).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    d : float or ndarray, shape=(...,[1,],...)
        Circular dispersion of given set of angles. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.3.1.
    """
    if degrees: data = np.deg2rad(data)
    if axial: data = 2*data

    # Compute circular mean and mean resultant length (Rbar)
    # Note: Need to keepdims here to ensure circ_subtract below works
    #       If keepdims is False, axis dim is removed below in sum()
    mu = circ_mean(data, axis=axis, keepdims=True)
    rbar = circ_rbar(data, axis=axis, keepdims=keepdims)

    # Number of elements contributing to dispersion (all elem's or along given axis)
    n = data.size if axis is None else data.shape[axis]

    # Angular difference btwn each data point and mean (Note: NOT absolute circ distance)
    dtheta = circ_subtract(data, mu)

    # Second centered circular moment (Fisher eqn. 2.27)
    rho2 = np.cos(2*dtheta).sum(axis=axis,keepdims=keepdims) / n

    # Circular dispersion (Fisher p.34, eqn. 2.28)
    return (1 - rho2) / (2 * rbar**2)


def circ_sem(data, axis=None, degrees=False, axial=False, is_stat=False, keepdims=False):
    """
    Compute a measure of circular standard error of array of angular data along given axis.
    Circular dispersion of array of angle data along given axis.

    Circular standard error = sqrt(circ_dispersion/n) (Fisher eqn. 4.21)

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute SEM along. If None, compute across entire flattened array (as in np.std).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    is_stat : bool, default: False
        Set=True only if input `data` actually reflects a statistic computed from raw data
        (eg as generated by a bootstrap resampling procedure), rather than raw data itself.
        In this case, the *standard deviation* of the statistic values is the estimate of the
        SEM of the original data. Therefore, if this is set, the computed SEM will *not* be
        normalized by sqrt(n).

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    sem : float or ndarray, shape=(...,[1,],...)
        Circular SEM of given set of angles. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    Fisher (1993) "Statistical Analysis of Circular Data" sxn. 2.3.1.
    """
    # Circular dispersion (Fisher p.34, eqn. 2.28)
    d = circ_dispersion(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)

    # Circular standard error = sqrt(circDispersion/n) (Fisher eqn. 4.21)
    sem = np.sqrt(d)

    # Normalize by sqrt(n) for raw data, but not if `data` is actually a derived stat (eg means)
    if not is_stat:
        # Number of elements contributing to dispersion (all elem's or along given axis)
        n = data.size if axis is None else data.shape[axis]
        sem = sem / np.sqrt(n)

    return sem


def von_mises_kappa(data=None, Rbar=None, n=None, axis=None, degrees=False, axial=False,
                    keepdims=False):
    """
    Compute an approximation to the maximum likelihood estimate of the concentration
    parameter kappa of the von Mises distribution (roughly, circular analog of inverse
    standard deviation of normal distribution)

    Can compute kappa either from raw data or from pre-computed mean resultant length Rbar
    and number of contributing observations n. Must input EITHER `data` OR `Rbar` and `n`,
    but not both.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    Rbar : float or ndarray
        Pre-computed mean resultant length. `Rbar` and `n` are alternative arguments to `data`.
        Do not input a value for `Rbar` if inputting raw `data`.
        Arbitary shape (if array, we assume each value is one Rbar, compute a kappa for each).

    n : float or ndarray
        Number of observtations going into each pre-computed Rbar. Must be same shape as `Rbar`.
        Do not input a value for `n` if inputting raw `data`.

    axis : int, default: None
        Axis to estimate kappa along.
        If None, estimate kappa across entire flattened array (as in :func:`np.mean`).

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    kappa : float or ndarray, shape=(...,[1,],...)
        Estimated value(s) of von Mises kappa. Returns a float for 1D data or `axis` is None.
        Otherwise, returns array the same shape as data, but with `axis` reduced to length 1
        (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    - Fisher (1993) "Statistical Analysis of Circular Data" p. 88
    - Vectorized version of circ_kappa.m from Circular Statistics Toolbox for Matlab
    """
    if Rbar is not None:
        assert data is None, \
            ValueError("Input either raw `data` OR precomputed `Rbar` and `n`, but not both")
        assert n is not None, \
            ValueError("If inputting precomputed `Rbar`, must also provide value for `n`")

    # Calculate mean resultant length Rbar (if not input)
    else:
        Rbar = circ_rbar(data, axis=axis, degrees=degrees, axial=axial, keepdims=keepdims)
        n = data.shape[axis] if axis is not None else data.size
    if isinstance(Rbar,float): Rbar = np.asarray([Rbar])
    if isinstance(n,float): n = n*np.ones_like(Rbar)

    # Calculate estimate of von Mises kappa. Formula to use depends on values of Rbar, n.
    kappa = np.empty_like(Rbar)

    idx = Rbar < 0.53
    if idx.any():
        kappa[idx] = 2*Rbar[idx] + Rbar[idx]**3 + (5/6)*Rbar[idx]**5

    idx = (Rbar >= 0.53) & (Rbar < 0.85)
    if idx.any():
        kappa[idx] = -0.4 + 1.39*Rbar[idx] + 0.43 / (1-Rbar[idx])

    idx = Rbar >= 0.85
    if idx.any():
        kappa[idx] = 1 / (Rbar[idx]**3 - 4*Rbar[idx]**2 + 3*Rbar[idx])

    # Small n corrections
    idx = (n < 15) & (kappa < 2)
    if idx.any():
        kappa[idx] = np.max(kappa[idx] - 2/(n[idx]*kappa[idx]), 0)

    idx = (n < 15) & (kappa >= 2)
    if idx.any():
        kappa[idx] = (n[idx]-1)**3 * kappa[idx] / (n[idx]**3 + n[idx])

    # For vector-valued data, extract value from scalar array -> float for output
    if kappa.size == 1: kappa = kappa.item()

    return kappa


def circ_hist(data, n_bins=8, bins=None, degrees=False, axial=False):
    """
    Computes histogram count of array of angular data

    Histogram bins can be set explicitly using `bins` or implicitly by setting the number
    of evenly-spaced, equal-width bins using `n_bins`.

    Parameters
    ----------
    data : ndarray, shape=(n_obs,)
        Array of angular data (in radians, unless `degrees` is True). Currently only set up for a 1D list of angles.

    n_bins : int, default: 8
        Number of equally-spaced, equal-width bins to use for histogram.
        Note: Only used if `bins` not set explicitly.

    bins : array_like, shape=(n_bins,2), default: (width=2pi/n_bins, starting bin center at 0)
        [start,end] angle to use for each bin in histogram

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    Returns
    -------
    counts : ndarray, shape=(n_bins,), dtype=int
        Histogram counts of number of angles falling into each histogram bin

    bins : ndarray, shape=(n_bins,2)
        Actual bins used for histogram
    """
    # TODO  Option to actually generate plot, cf. https://stackoverflow.com/a/55067613
    # TODO  Set up ability to compute histogram along given axis of arbitrary-shape array
    # todo  Better algorithm (shift bins and data so min=0, then np.hist()?)

    # If bins not set explicitly, set bins from n_bins, with 1st bin centered at 0
    if bins is None:
        if axial:   range = 180 if degrees else pi
        else:       range = 360 if degrees else 2*pi

        width = range/n_bins
        bins = np.stack((np.arange(0,range,width) - width/2,
                         np.arange(0,range,width) + width/2), axis=1)
    else:
        assert (bins.ndim == 2) and (bins.shape[1] == 2), \
            ValueError("bins should be (n_bins,2) array with each row giving bin [start,end]")

    # Wrap all data and bins into range (0,2pi)
    data = wrap(data, degrees=degrees, axial=axial)
    bins = wrap(bins, degrees=degrees, axial=axial)

    n_bins = bins.shape[0]

    counts = np.empty((n_bins,),dtype=int)

    for i_bin,(start,end) in enumerate(bins):
        # Wrapping angular bin
        if start > end: counts[i_bin] = ((data >= start) | (data < end)).sum()
        # Non-Wrapping angular bin
        else:           counts[i_bin] = ((data >= start) & (data < end)).sum()

    return counts, bins


# =============================================================================
# Inferential statistics functions
# =============================================================================
def rayleigh_test(data, axis=None, degrees=False, axial=False, return_stats=False,
                  keepdims=False):
    """
    Rayleigh Test to test for significant deviation of a distribution of from circular uniform.

    Assumes data is sampled from a von Mises distribution (circular analog of normal), and is
    uniform or unimodal (does not work well for bi/multimodal data).

    Uses test from Fisher sxn. 4.3, eqn. 4.17.  See also Zar sxn 27.1.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute var along. If None, compute across entire flattened array.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    return_stats : bool, default: False
        If True, returns p value(s) and S statistic(s). If False, only returns p values.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values indicating probability that data arises from a circular uniform distribution.
        For 1d data, returned as single scalar value. For n-d data, it has same shape as data,
        with `axis` reduced to length 1 if `keepdims` is True, or with `axis` removed
        if `keepdims` is False.

    Z : float or ndarray, shape=(...,[1,]...), optional
        'Z' statistic values used in Rayleigh test. Sometimes Z, or log(Z), is reported as a
        continuous measure of non-circular-uniformity of data. Same shape as `p`.

        Note: the Pairwise Phase Consistency (PPC) metric used in analysis of phase synchrony
        (Vinck Neuroimage 2010) is closely related to Z: Z = PPC*(n-1) + 1

    References
    ----------
    - Fisher (1993) "Statistical Analysis of Circular Data" sxn. 4.3, eqn. 4.17
    - Zar (1999) "Biostatistical Analysis" sxn. 27.1
    """
    data = _check_and_process_data(data, degrees, axial)
    n = data.shape[axis] if axis is not None else data.size

    # Compute mean resultant length Rbar
    Rbar = circ_rbar(data, axis=axis, keepdims=keepdims)

    # Rayleigh test Z statistic
    Z = n * Rbar**2
    # Resulting p value is this complicated formula from Fisher eqn. 4.17
    p = np.exp(-Z) * (1 + (2*Z - Z**2)/(4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4)/(288*n**2))

    if return_stats:    return p, Z
    else:               return p


def circ_mean_test(data, axis=None, mu=0, tail='both', degrees=False, axial=False,
                   is_stat=False, return_stats=False, keepdims=False):
    """
    Test of a circular mean direction against a specific value. Tests null hypothesis that
    observed circular mean = mu, against alternative hypothesis that the observed mean != mu.

    Fisher recommends using this only when n >= 25. We issue a warning if n < 25.

    Run in mass-univariate fashion along given array axis (or over entire array if no axis given).

    Uses test given in Fisher sxn. 4.4.5(b), originally from Watson (1983)

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True). Arbitrary shape.

    axis : int, default: None
        Axis to compute var along. If None, compute across entire flattened array.

    mu : float, default: 0
        Expected value of circular mean under the null hypothesis.

    tail : {'both','right','left'}, default: 'both' (2-tailed test)
        Specifies tail of test to perform:

        - 'both'  : 2-tail test -- test for abs(stat_obs) > abs(stat_resmp)
        - 'right' : right-sided 1-tail test -- tests for stat_obs > stat_resmp
        - 'left'  : left-sided 1-tail test -- tests for stat_obs < stat_resmp

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    is_stat : bool, default: False
        Set=True only if input `data` actually reflects a statistic computed from raw data
        (eg as generated by a bootstrap resampling procedure), rather than raw data itself.
        In this case, the *standard deviation* of the statistic values is the estimate of the
        SEM of the original data. Therefore, if this is set, the computed SEM will *not* be
        normalized by sqrt(n).

    return_stats : bool, default: False
        If True, returns p value(s) and S statistic(s). If False, only returns p values.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as single scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    S : float or ndarray, shape=(...,[1,]...), optional
        'S' statistic values for actual observed data. Same shape as `p`.

    References
    ----------
    Fisher (1993) "Statistical Analysis of Circular Data" sxn. 4.4.5(b)
    """
    data = _check_and_process_data(data, degrees, axial)
    mu = _check_and_process_data(mu, degrees, axial)
    n = data.size if axis is None else data.shape[axis]
    if n < 25:
        warn("This test is recommended only for n>=25 (here n=%d). See Fisher sxn. 4.4.5" % n)

    # Compute circular mean and standard error of mean
    mu_hat = circ_mean(data, axis=axis, keepdims=keepdims)
    sem = circ_sem(data, axis=axis, is_stat=is_stat, keepdims=keepdims)

    # Compute test statistic (Fisher eqn. 4.23)
    S = np.sin(mu_hat - mu) / sem

    tail = tail.lower()

    ## Convert test statistic to p value using normal(0,1) distribution
    # 2-tailed test: hypothesis ~ mu_hat != mu (Note: 2x p-value to account for 2-tailed test)
    if tail == 'both':      p = 2*norm.sf(np.abs(S))
    # 1-tailed rightward test: hypothesis ~ mu_hat > mu (Note: sf = 1 - cdf, but "more accurate")
    elif tail == 'right':   p = norm.sf(S)
    # 1-tailed leftward test: hypothesis ~ mu_hat < mu
    elif tail == 'left':    p = norm.cdf(S)
    else:
        raise ValueError("Unsupported value '%s' for `tail`. Use 'both', 'right', or 'left'" % tail)

    # For vector-valued data, extract value from scalar array -> float for output
    if p.size == 1:
        p = p.item()
        if return_stats: S = S.item()

    if return_stats:    return p, S
    else:               return p


def circ_ANOVA1(data, labels, axis=0, groups=None, omega=True, degrees=False, axial=False,
                return_stats=False, keepdims=False):
    """
    Parametric Watson-Williams multi-sample test for equal means of circular data groups.

    Analog of a one-way ANOVA test for circular data.  Like ANOVA, assumes each group is
    distributed as a VonMises on the circle with common concentration parameter k
    (circular analog of normal distributions with same variance).

    Based on method described in Zar (1999), Jammalamadaka & SenGupta (2001), and on function
    circ_wwtest() in Matlab Circular Analysis Toolbox.

    Run in mass-univariate fashion along given array axis.

    Parameters
    ----------
    data : ndarray, shape=(...,n_obs,...)
        Array of angular data to analyze (in radians, unless `degrees` is set).
        Array `axis` should correspond to data observations (eg trials) to compute test across,
        and all other dimensions are treated as independent data, and test is performed separately
        in mass-univariate for each.

    labels : array-like, shape=(n_obs,)
        Array of group (factor level) labels for each observation (eg, 0,1,2, etc.;
        but can be any types/values). Must be same length as data.shape[axis]

    axis : int, default: 0
        Dimension to compute test along, corresponding to data observations (eg trials).
        Defaults to 1st axis of data array.

    groups : array-like, shape=(n_groups,), default: np.unique(labels)
        List of all expected group labels in `labels`. Only used to run a subset of groups in
        `labels`; otherwise, defaults to all unique values in `labels`.

    omega : bool, default: True
        If True, uses bias-corrected omega-squared formula for proportion of explained variance.
        If False uses eta-squared/R-squared formula, which is positively biased.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    return_stats : bool, default: False
        If True, returns p value(s) and dict with additional statistics (see below).
        If False, only returns p values.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    p : float or ndarray, shape=(...,[1,]...)
        p values from test. For 1d data, returned as single scalar value.
        For n-d data, it has same shape as data, with `axis` reduced to length 1
        if `keepdims` is True, or with `axis` removed  if `keepdims` is False.

    stats : dict, optional
        If `return_stats` set, statistics on each fit also returned:

        - pev : float or ndarray, shape=(...,[1,]...)
            Percent of data variance explained by groups for each data series. Same shape as `p`.
        - F : float or ndarray, shape=(...,[1,]...)
            F-statistic for each data series. Same shape as `p`.
        - R : ndarray, shape=(...,n_groups,...)
            Group resultant length R for each group/level. Note different shape.
        - n : ndarray, shape=(n_groups,)
            Number of observations (eg trials) in each group/level. 1D array.

    References
    ----------
    - Zar (1999) "Biostatistical Analysis" sxn. 27.5 (see eqn. 27.14)
    - Jammalamadaka & SenGupta (2001) "Topics in Circular Statistics" sxn. 5.3.1
    - Snyder & Lawson (1993) https://doi.org/10.1080/00220973.1993.10806594 (omega-squared)
    - https://en.wikipedia.org/wiki/Effect_size (omega-squared)
    """
    data = _check_and_process_data(data, degrees, axial)
    labels = np.asarray(labels)
    assert labels.ndim == 1, \
        ValueError("`labels` must be 1 dimensional (same labels for all datapoints)")
    assert len(labels) == data.shape[axis], \
        ValueError("`labels` length (%d) must be same as `data` along observation `axis` (%d)"
                   % (len(labels), data.shape[axis]))
    assert axis is not None, \
        TypeError("axis cannot be set=None. This analysis must be done along an array axis.")

    # If not explicitly input, find all unique values in `labels`
    if groups is None: groups = np.unique(labels)
    n_groups = len(groups)
    assert n_groups > 1, ValueError("Cannot compute ANOVA with only a single group (level)")

    # Reshape data so observation is axis 0, all other dims are unwrapped into axis 1
    data, data_shape = standardize_array(data, axis=axis, target_axis=0)
    n_data_series = data.shape[1]

    # Calculate resultant lengths for each group (factor level) in dataset
    #  (~analogous to group means for linear ANOVA)
    n = np.empty((n_groups,))
    R = np.empty((n_groups,n_data_series))
    for i_group,group, in enumerate(groups):
        group_bool = labels == group
        n[i_group] = group_bool.sum()
        # Resultant length R for given group
        R[i_group,:] = circ_r(data[group_bool,:], axis=0)

    # # Sum over groups of all group (R's * n's)
    # R_sum = (n[:,np.newaxis] * R).sum(axis=0, keepdims=True)
    R_sum = R.sum(axis=0, keepdims=True)        # Sum of all groupwise R's
    N = n.sum()                                 # Total number of observations

    # Grand-Resultant length R for all observations (analogous to grand mean for linear ANOVA)
    # Note: R of group-mean-angles is equivalent, so 2 methods of spynal.info.anova1 are identical
    R_grand = circ_r(data, axis=0)

    ## Calculate explained variance for all data points
    # Compute estimate of concentration parameter kappa of the von Mises distribution
    kappa       = von_mises_kappa(Rbar=R_sum/N, n=N)
    # Correction factor for bias in pev/F statistic calculation
    K           = 1 + (3/(8*kappa))
    # Compute Sums of Squares, df's, and Mean Squares used for stats
    # SS_groups   = R_sum - N*R_grand             # Groups Sum of Squares
    SS_groups   = R_sum - R_grand               # Groups Sum of Squares
    SS_error    = N - R_sum                     # Error Sum of Squares
    SS_total    = SS_groups + SS_error          # Total Sum of Squares
    df_groups   = n_groups - 1                  # Groups degrees of freedom
    df_error    = N - n_groups                  # Error degrees of freedom
    MS_error    = SS_error / df_error           # Error mean square
    MS_groups   = SS_groups / df_groups         # Groups mean square
    F           = K * MS_groups / MS_error      # F-statistic;  see Zar eqn. 27.14
    p           = f.sf(F, df_groups, df_error)  # p value. Note: sf = 1 - cdf, but "more accurate"

    # Calculate PEV and package PEV, R, n, and F in a dict to return
    if return_stats:
        # Compute percent of variance explained
        # Omega-squared stat = bias-corrected explained variance
        if omega:
            pev = 100 * K * (SS_groups - df_groups*MS_error) / (SS_total + MS_error)
        # Standard (eta-squared) formula for explained variance
        else:
            pev = 100 * K * SS_groups / SS_total

        # Rearrange output variable arrays so in format/size expected based on original input arg's
        R = undo_standardize_array(R, data_shape, axis=axis, target_axis=0)
        if pev.size == 1:
            F = F.item()
            pev = pev.item()
        else:
            F = undo_standardize_array(F, data_shape, axis=axis, target_axis=0)
            pev = undo_standardize_array(pev, data_shape, axis=axis, target_axis=0)
            if not keepdims:
                F = F.squeeze(axis=axis)
                pev = pev.squeeze(axis=axis)

        stats = {'pev':pev, 'F':F, 'R':R, 'n':n}

    # Rearrange output variable arrays so in format/size expected based on original input arg's
    if p.size == 1:
        p = p.item()
    else:
        p = undo_standardize_array(p, data_shape, axis=axis, target_axis=0)
        if not keepdims: p = p.squeeze(axis=axis)

    if return_stats:    return p, stats
    else:               return p


# Alias circ_ANOVA1 as watson_williams_test (actual name of test is Watson-Williams test)
watson_williams_test = circ_ANOVA1
""" Alias of :func:`circ_ANOVA1`. See there for details. """


def circ_circ_correlation(data1, data2, axis=None, degrees=False, axial=False,
                          method='js', keepdims=False):
    """
    Compute a measure of circular-circular correlation between two arrays of angular
    random variables, analogous to the standard Pearson product-moment for linear variables.

    Optionally computes either the "T-linear association" of Fisher & Lee (1983) or the
    circular correlation measure "rho_c" of Jammalamadaka & Sarma (1988). Both range from
    -1 (perfect anti-correlation) to +1 (perfect correlation), with 0 indicating no correlation.
    Both measures produce very similar results, though J & S is a bit faster and has an arguably
    more easily interpretable formula.

    Parameters
    ----------
    data1/data2 : ndarray, shape=(...,n_obs,...)
        Arrays of angular data (in radians, unless `degrees` is True).
        Arbitrary shape, but both must have same shape.

    axis : int, default: None
        Axis to compute correlation along. If None, compute across entire flattened array.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    method : str, default: 'js'
        Method used to compute circular correlation. Both give similar results, though
        'js' is slightly faster, arguably a bit more straightforward.

        - 'js' : Jammalamadaka-Sarma (1988) method (as described in J & SenGupta 2001 book).
            Circular analog of Pearson correlation formula comparing sine's of
            deviations of each data point from it's mean.
        - 'fl' : Fisher-Lee (1983) T-linear association method. Similar formula to J-S,
            but evaluated on all *pairs* of data points (though implemented here with more
            computation-friendly equivalent from Fisher book).

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    rho : float or ndarray, shape=(...,[1,]...)
        Circular correlation between given sets of angles.
        Returns a single float if data1/2 are 1-dimensional or `axis` is None.
        Otherwise, returns array the same shape as data1/2, but with `axes`
        reduced to length 1 (if keepdims=True) or removed (if keepdims=False).

    References
    ----------
    - Jammalamadaka & Sarma (1988) "A correlation coefficient for angular variables"
    - Jammalamadaka & SenGupta (2001) "Topics in circular statistics" sxn. 8.2
    - Fisher & Lee (1983) Biometrika https://doi.org/10.1093/biomet/70.2.327
    - Fisher (1993) "Statistical Analysis of Circular Data" sxn. 6.3.3
    """
    method = method.lower()
    data1 = _check_and_process_data(data1, degrees, axial)
    data2 = _check_and_process_data(data2, degrees, axial)
    assert data1.shape == data2.shape, "Data arrays must have same size/shape"
    assert method in ['js','fl'], \
        ValueError("Unsupported value '%s' set for `method`. Should be 'js' or 'fl'." % method)

    # Jammalamadaka & Sarma (1988) method. Formula from J & S book eqn. 8.2.5.
    # (NOTE error in missing summation in 2nd term of denom of formula in J & S)
    if method == 'js':
        mean1 = circ_mean(data1, axis=axis, keepdims=True)
        mean2 = circ_mean(data2, axis=axis, keepdims=True)

        sin1 = np.sin(data1 - mean1)
        sin2 = np.sin(data2 - mean2)

        num = np.sum(sin1 * sin2, axis=axis, keepdims=keepdims)
        den = np.sqrt(np.sum(sin1**2, axis=axis, keepdims=keepdims) *
                      np.sum(sin2**2, axis=axis, keepdims=keepdims))
        # NOTE: This is the formula from J & S eqn. 8.2.5, which produces incorrect results
        # den = np.sqrt(np.sum((sin1**2)*(sin2**2), axis=axis, keepdims=keepdims))
        rho = num / den

    # Fisher & Lee (1983) method. Formulas from Fisher (1993) eqn's 6.36-6.37
    else:
        cos1 = np.cos(data1)
        cos2 = np.cos(data2)
        sin1 = np.sin(data1)
        sin2 = np.sin(data2)

        A = np.sum(cos1*cos2, axis=axis, keepdims=keepdims)
        B = np.sum(sin1*sin2, axis=axis, keepdims=keepdims)
        C = np.sum(cos1*sin2, axis=axis, keepdims=keepdims)
        D = np.sum(sin1*cos2, axis=axis, keepdims=keepdims)
        E = np.sum(np.cos(2*data1), axis=axis, keepdims=keepdims)
        F = np.sum(np.sin(2*data1), axis=axis, keepdims=keepdims)
        G = np.sum(np.cos(2*data2), axis=axis, keepdims=keepdims)
        H = np.sum(np.sin(2*data2), axis=axis, keepdims=keepdims)
        n = data1.shape[axis] if axis is not None else data1.size

        num = 4*(A*B - C*D)
        den = np.sqrt((n**2 - E**2 - F**2) * (n**2 - G**2 - H**2))
        rho = num / den

    return rho


def circ_linear_correlation(circ_data, linear_data, axis=None, degrees=False, axial=False,
                            method='js', return_stats=False, keepdims=False):
    """
    Compute a measure of circular-linear correlation between one array of angular
    random variables and one (array of) linear random variables.

    Two very different measured are available:

    - Mardia (1976) "C-linear association" : Unsigned (non-negative) correlation that ranges
        from 0 (no correlation) to 1 (perfect postive correlation or anti-correlation).
        Allows for computation of p-value under parametric assumptions, and is widely
        accepted and used in the statistical literature

    - Circular-linear adaptation of Jammalamadaka-Sarma (1988) : Signed correlation, which
        is a straightforward analog of standard Pearson correlation. Does not have parametric
        p-values, and does not appear much in statistical lit, only in applied literature.
        Basically a bit of a Frankenstein hack of J & S formula for two circular variables and
        standard Pearson formula for two linear variables...but seems to work well. YMMV.

    For Mardia method, optionally returns p-value based on chi-squared (df=2) approx. (cf Zar).

    Mardia method adapted from Fisher book sxn. 6.2.3, Zar book sxn. 27.16,
    and Circular Statistics Toolbox for Matlab function circ_corrcl().

    Parameters
    ----------
    circ_data : ndarray, shape=(...,n_obs,...)
        Array of angular data (in radians, unless `degrees` is True).

    linear_data : ndarray, shape=(...,n_obs,...)
        Array of linear data. Arbitrary shape, but must have same shape as `circ_data`.

    axis : int, default: None
        Axis to compute correlation along. If None, compute across entire flattened array.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    method : str, default: 'js'
        Method used to compute circular-linear correlation:

        - 'js' : Circular-linear adaptation of Jammalamadaka-Sarma (1988) circ-circ method.
            Circular-linear analog of Pearson correlation formula comparing sine's of
            deviations of each circular data point from it's mean. Signed correlation
            measure ranging from -1 to 1. Though I can't find an authoritative statistical
            reference for it, it appears in the applied (neuroscience) literature,
            and it seems to work well...
        - 'mardia' : Mardia (1976) C-linear association method. Unlike standard correlation
            (and J-S), it is unsigned, ranging from 0 to 1. But it is more widely used and
            accepted in the statistical literature.

    return_stats : bool, default: False
        Set=True to also return p value based on chi-squared approximation.
        Set=False to only return correlation coefficient.

    keepdims : bool, default: False
        If True, retains reduced observations `axis` as length-one axes in output.
        If False, removes reduced observations `axis` from outputs.

    Returns
    -------
    rho : float or ndarray, shape=(...,[1,]...)
        Circular-linear correlation between given sets of angular and linear data.
        Returns a single float if input data are 1-dimensional or `axis` is None.
        Otherwise, returns array the same shape as data1/2, but with `axes`
        reduced to length 1 (if keepdims=True) or removed (if keepdims=False).

    p : float or ndarray, shape=(...,[1,]...), optional
        p value based on chi-squared approximation. Only returned if `return_stats` is True.
        Same shape as `rho`.

    References
    ----------
    - Mardia (1976) Biometrika https://doi.org/10.2307/2335637
    - Fisher (1993) "Statistical Analysis of Circular Data" sxn. 6.2.3.
    - Zar (1999) "Biostatistical Analysis" 4th ed. sxn. 27.16 (eq. 27.47)
    """
    method = method.lower()
    circ_data = _check_and_process_data(circ_data, degrees, axial)
    linear_data = np.asarray(linear_data)
    assert circ_data.shape == linear_data.shape, \
        ValueError("Data arrays must have same size/shape")
    assert method in ['js','mardia'], \
        ValueError("Unsupported value '%s' set for `method`. Should be 'js' or 'mardia'." % method)
    if return_stats:
        assert method == 'mardia', \
            ValueError("p values are only defined for Mardia method (`method`='mardia')")

    if method == 'js':
        mean1 = circ_mean(circ_data, axis=axis, keepdims=True)
        mean2 = np.mean(linear_data, axis=axis, keepdims=True)

        d1 = np.sin(circ_data - mean1)
        d2 = linear_data - mean2

        num = np.sum(d1 * d2, axis=axis, keepdims=keepdims)
        den = np.sqrt(np.sum(d1**2, axis=axis, keepdims=keepdims) *
                      np.sum(d2**2, axis=axis, keepdims=keepdims))
        rho = num / den

    else:
        C = np.cos(circ_data)
        S = np.sin(circ_data)
        n = circ_data.shape[axis] if axis is not None else circ_data.size

        # Compute correlation coefficient separately for sine and cosine components of angles
        r_XC = correlation(linear_data, C, axis=axis, keepdims=keepdims)
        r_XS = correlation(linear_data, S, axis=axis, keepdims=keepdims)
        r_CS = correlation(C, S, axis=axis, keepdims=keepdims)

        # Compute angular-linear correlation (Zar eqn. 27.47)
        rho = np.sqrt((r_XC**2 + r_XS**2 - 2*r_XC*r_XS*r_CS) / (1 - r_CS**2))

    # Compute p value based on chi^2 (df=2) approximation (see Zar).
    if return_stats:
        p = chi2.sf(n*rho**2, 2) # Note: sf = 1 - cdf, but "more accurate"
        return rho, p
    else:
        return rho


def circ_linear_regression(circ_data, linear_data, degrees=False, axial=False, fit_constant=True,
                           fit_method='hybrid', error='1-R', grid=None, return_stats=False,
                           return_all_fits=False, verbose=False, **kwargs):
    """
    Circular-linear regression with a single target variable that is circular/angular, and one
    or more predictor variables that are linear.

    Fits circular variable with a "barber-pole"-type model where increasing values of the
    weighted sum of the predictors result in increasing values of the predicted circular variable
    within the range [0,2pi] radians ([0,360] deg), but then the predicted variable *wraps back
    to 0* (like a circular variable!). Based on model proposed by Gould (1969). Fits model using
    grid-search or nonlinear optimization methods, finding model fit with minimal error between
    predicted and actual circular data.

    This is a model often used to describe oscillatory traveling activity waves in cortex, where
    the target variable is the wave phase at each recorded site (or difference in phase btwn
    site pairs) and the predictors are the physical locations of the recorded site (or differences
    in location btwn site pairs). It's a generalization of the model used in Das...Jacobs (2023)
    The specific fitting setup used in that paper is implemented in jacobs_regression(), which is
    currently in an in-progress module for traveling wave analysis (ask Scott B for it).

    NOTE: This is NOT the circular-linear model described in section 6.4.2 of the Fisher
    "Statistical Analysis of Circular Data" book or in Fisher & Lee (1992). In that model,
    increasing values of the weighted sum of the predictors result in a monotonically increasing
    predicted circular variable that approaches an asymptote at +/- 1, due to the sigmoidal link
    function used there. This may be a good model for some phenomena, but seems inappropriate for
    most analyses of oscillatory data (eg brain rhythms).

    NOTE: Unlike most other functions here -- which allow some flexibility in input data array
    shapes and are set up to perform analysis in a mass-univariate fashion across multiple
    independent data series -- this function currently accepts only a single data series and has
    stricter requirements on data shape (see below). Correspondingly, there are no `axis` or
    `keepdims` parameters here.

    Parameters
    ----------
    circ_data : ndarray, shape=(n_obs,)
        Array of angular data (in radians, unless `degrees` is True) to use as regression targets

    linear_data : ndarray, shape=(n_obs,n_predictors)
        Array of linear data to use as regressors to predict `circ_data`

    axis : int, default: None
        Axis to compute correlation along. If None, compute across entire flattened array.

    degrees : bool, default: False
        Set=True if data is given in degrees, False if given in radians.

    axial : bool, default: False
        Set=True if data is axially symmetric, ie ranges over pi rad or 180 degrees.
        Set=False if data is circular/angular, ie ranges over 2*pi rad or 360 degrees.

    fit_constant : bool, default: True
        If True, fits an overall mean angle (constant angular offset, analogous to intercept
        in linear regression). A constant column will be appended to `linear_data` for the final
        regression design matrix (so don't add one yourself). If False, doesn't fit a mean angle,
        and returns 0 for its estimate.

        You will generally want to fit a mean angle for SSE or deviance errors, which are
        sensitive to the mean angle (you will get a warning if you don't). It is not so necessary
        for 1-R error, which is invariant to the mean angle.

    fit_method : str, default: 'hybrid'
        Method to use for fitting regression model: 'gridsearch' | 'optimization' | 'hybrid'

        - 'gridsearch' : Evaluates model only at a set grid of parameter values, and picks the
            parameter values (`beta`s) minimizing the fit error. The mean/offset angle `mu`
            is set to the empirical circular mean of `y`. Fast, but accuracy depends on
            alignment btwn grid and actual parameters.
        - 'optimization' : Runs nonlinear optimization using :func:`scipy.optimize.minimize`
            (which uses the BFGS quasi-Newton method by default) to find the optimal (mimimal
            error) parameters, starting the fit from each of a grid of starting points
            (to minimize chance of finding only a local, rather than the global, error minimum).
            The mean/offset angle `mu` is fitted to the data, along with the coefficients.
            Much slower, but very accurate.
        - 'hybrid' : Runs grid search, then a single optimization run on the best-fit
            coefficients obtained across the grid search. About as fast as gridsearch, but
            much more accurate.

    error : str, default: '1-R'
        Measure to use to quantify (circular) residual error of model fits. In practice, they
        tend to be highly correlated, except in some edge cases. Options:

        - '1-R' :  1 - mean resultant length (Rbar) of residual errors. This option is more
            commonly used in the stats (and neuroscience) literature, but it only quantifies error
            precision, it is completely invariant to error bias/accuracy. So, in some edge cases
            you can have large errors, but all with the same phase offset from the true values,
            and that will be labeled a "good fit".
        - 'SSE' : Sum of Squared (circular) Errors. This is not very commonly used, though it is
            the most straightforward circular analog of the standard linear SSE. It does take into
            account the full error, both precision and bias/accuracy.
        - 'deviance' : Deviance from a Von Mises maximum likelihood model ~ -sum(cos(errors)).
            Like SSE, also takes both precision and bias/accuracy into account.

    grid : int or array-like or Iterator, default: (see below)
        Determines the grid of parameter values used for model evaluation (gridsearch method) or
        for model fit starting points (optimization method). Can be specified in one of several
        ways, with increasing granularity of user control:

        - int : Sets the number of values sampled for each parameter, ranging linearly from
            -1.5 to +1.5 * the ratio of SD(y)/SD(x) (for each predictor x)
        - array-like : (n_predictors,) array-like of (n_grid_pts[predictor],) lists of
            grid/starting points to sample for each predictor in `linear_data`. Final grid is the
            cross-product of all these value lists across all predictors.
        - Iterator : Iterator object corresponding to cross-product of all parameter lists as
            described above. Obtained, eg, as `itertools.product(*predictor_grid_values)`.

        If no values are input, defaults to an int value that depends on the `fit_method` and
        parameter type (beta coefficient vs constant/mean, if `fit_constant` is True):
        - beta : 24/12/8 for gridsearch/hybrid/optimization
        - constant : 15/5/5 for gridsearch/hybrid/optimization

    return_stats : bool, default: False
        If False, only returns fitted 'beta' coefficients and mean target value offset 'mu'.
        If True, also return error and predicted target values for fitted parameter set.

    return_all_fits : bool, default: False
        If return_stats and return_all_fits are True, also returns fitted parameters and stats
        for all gridsearch points / fit starting points.

    verbose: bool, default: False
        If True, print to stdout an error message every time an optimization fit fails

    **kwargs :
        Any other kwargs passed directly to :func:`scipy.optimize.minimize` function

    Returns
    -------
    beta : ndarray, shape=(n_predictors,)
        Fitted coefficients for each predictor, for best model fit

    mu : float
        Fitted mean target value offset for best model fit (lowercase theta in Das 2023 model).
        Set=0 if `fit_constant` is False.

    stats : dict [optional]
        Only returned if `return_stats` is True. Dictionary with additional statistics on
        best model fit (and on all model fits, if `return_all_fits` is True). Fields:

        - 'predicted' : ndarray, shape=(n_obs,)
            Predicted values for all observations, based on best model fit
        - 'error' : float
            Error (1 - mean resultant length) for best model fit
        - 'all_fits' : dict [optional] of (n_fits,) lists
            Grid search/fit start 'sample' point, fitted parameters 'beta' and 'mu',
            and 'error' for each fit (all `grid` values).
            Only returned if `return_all_fits` is True.

    References
    ----------
    - Gould (1969) Biometrics https://doi.org/10.2307/2528567
    - Das ... Jacobs (2023) In "Intracranial EEG" https://doi.org/10.1007/978-3-031-20910-9_30
    """
    # todo Actually make function deal with arbitrary data shape and multiple data series?
    circ_data = _check_and_process_data(circ_data, degrees, axial)
    linear_data = np.asarray(linear_data)
    error = error.lower()
    fit_method = fit_method.lower()
    assert circ_data.shape[0] == linear_data.shape[0], \
        "Data arrays must have same number of rows (shape[0])"
    assert fit_method in ['optimization','gridsearch','hybrid'], \
        "Unsupported value '%s' set for `fit_method`. Must be 'optimization' or 'gridsearch'." \
        % fit_method
    assert error in ['1-r','sse','deviance'], \
        "Unsupported value '%s' set for `error`. Must be '1-R', 'SSE', or 'deviance'." \
        % error
    if fit_method == 'gridsearch':
        assert len(kwargs) == 0, "Should not pass any extra kwargs for `fit_method`='gridsearch"
    if (error != '1-r') and not fit_constant:
        warn("%s error is sensitive to mean angle; you should fit one (set `fit_constant`=True)"
             % error)

    X = linear_data.copy()
    if X.ndim == 1: X = X[:,np.newaxis]
    n_obs,n_predictors = X.shape

    # Center each regressor, so mu parameter can be interpreted as data mean
    X -= X.mean(axis=0, keepdims=True)

    # Append column of ones, reflecting mean/constant term, to end of design matrix X
    if fit_constant:
        X = np.column_stack((X, np.ones((n_obs,1))))
    # Wrap target circular data into range [-pi,pi] (not strictly necessary?)
    y = wrap(circ_data, limits=(-pi,pi))

    # If no grid for grid search or for optimization starting points was input, generate one
    default_grid = grid is None
    if default_grid:
        if fit_method == 'gridsearch':      grid = 24
        elif fit_method == 'hybrid':        grid = 12
        elif fit_method == 'optimization':  grid = 8

    # Convert number of grid points to sample into actual grid (in not already done)
    if np.isscalar(grid):
        n_grid_pts = grid
        sd_y = y.std()

        grid = []
        for dim in range(n_predictors):
            # Compute ratio of std(y)/std(x)
            sd_x = X[:,dim].std()
            sd_ratio = sd_y / sd_x
            # Sample linearly from -1.5x to +1.5x sd_ratio
            grid.append(np.linspace(-1.5*sd_ratio, 1.5*sd_ratio, n_grid_pts))
        # Also include grid points for the mean angle/constant offset term
        if fit_constant:
            if default_grid:
                n_constant_pts = 15 if fit_method == 'gridsearch' else 5
            else:
                n_constant_pts = n_grid_pts
            grid.append(wrap(np.arange(0,2*pi,2*pi/n_constant_pts), limits=(-pi,pi)))

    # Compute cross-product of all grid values across all predictors (if not already done in input)
    grid_product = product(*grid) if not isinstance(grid, Iterator) else grid

    # Set function to compute error based on requested error type
    # 1 - mean resultant length (Rbar)
    if error == '1-r':
        error_func = lambda x: (1 - circ_rbar(x))
    # Circular squared error (errors input to this take circularity into account)
    elif error == 'sse':
        error_func = lambda x: np.sum(x**2)
    # Deviance (negative log-likelihood) for VonMises errors.
    # Formula is computational simplification of sum(1 - cos(x)).
    else:
        error_func = lambda x: (x.size - np.sum(np.cos(x)))


    def eval_model_prediction(B, X):
        """
        Evaluate (compute model-estimated angles) of circular-linear regression model
        at given coefficient values.

        Model-estimated angles = predictors * current values of coefficients (including any
        constant/mean term), wrapped into range [-pi,+pi]
        """
        return wrap(X @ B, limits=(-pi,pi))

    def compute_model_residuals(B, X, theta):
        """
        Compute residuals (observed - predicted angles) of circular-linear regression model
        at given coefficient values
        """
        # Compute model-estimated angles
        theta_hat = eval_model_prediction(B, X)

        # Model residuals = observed - estimated thetas, wrapped into range [-pi,+pi]
        return circ_subtract(theta, theta_hat)

    def compute_model_error(B, X, theta):
        """
        Compute measure of  error of circular-linear regression model at given coefficients
        """
        # Compute residuals (observed - predicted angles) of model fit
        res = compute_model_residuals(B, X, theta)

        # Compute error from residuals based on requested formula
        return error_func(res)


    if fit_method == 'optimization':
        fun = lambda B: compute_model_error(B, X, y)

    best_error = inf
    best_coeffs = np.full((n_predictors+fit_constant,), fill_value=np.nan)
    if return_stats and return_all_fits:
        all_fits = {'sample':[], 'beta':[], 'mu':[], 'error':[]}

    # For each sample in grid of search/starting points...
    for sample in grid_product:
        sample = np.asarray(sample)

        # For optimization method, use sample as starting point for nonlinear fit of coeffs/mu
        if fit_method == 'optimization':
            res = minimize(fun, sample, **kwargs)
            if res.success:
                fit_coeffs = res.x
                fit_error = compute_model_error(fit_coeffs, X, y)
            else:
                fit_coeffs = np.full_like(sample, fill_value=np.nan)
                fit_error = np.nan
                if verbose: print(np.round(sample,2), "Fit failed with message:", res.message)

        # For grid search method, just evaluate model goodness of fit (~circular error)
        # at given grid sample point (coefficient set)
        else:
            fit_coeffs = sample
            fit_error = compute_model_error(fit_coeffs, X, y)

        # If current fit is better than previous best, save it and its associated error
        if fit_error < best_error:
            best_error = fit_error
            best_coeffs = fit_coeffs

        if return_stats and return_all_fits:
            beta = fit_coeffs[:-1] if fit_constant else fit_coeffs
            if beta.size == 1: beta = beta.item()
            mu = fit_coeffs[-1] if fit_constant else 0
            all_fits['sample'].append(sample)
            all_fits['beta'].append(beta)
            all_fits['mu'].append(mu)
            all_fits['error'].append(fit_error)

    # For 'hybrid' fitting, run optimization on best-fit coefficients from gridsearch
    if fit_method == 'hybrid':
        start = iter([best_coeffs])  # Hack best-fit coefs/offset into an iterator
        # Note: Data has already been converted to radians/non-axial, so don't set those here
        beta, mu = circ_linear_regression(circ_data, linear_data, degrees=False, axial=False,
                                          fit_constant=fit_constant, fit_method='optimization',
                                          error=error, grid=start,
                                          return_stats=False, return_all_fits=False, **kwargs)
        if fit_constant:
            best_coeffs[:-1] = beta
            best_coeffs[-1] = mu
        else:
            best_coeffs = beta
    else:
        beta = best_coeffs[:-1] if fit_constant else best_coeffs
        if beta.size == 1: beta = beta.item()
        mu = best_coeffs[-1] if fit_constant else 0

    # Return fitted coefficents/mean angle
    if return_stats:
        stats = {'predicted' : eval_model_prediction(best_coeffs, X),
                 'error' : compute_model_error(best_coeffs, X, y)}

        if return_all_fits: stats['all_fits'] = all_fits

        return beta, mu, stats

    else:
        return beta, mu


# =============================================================================
# Helper functions
# =============================================================================
def _check_and_process_data(data, degrees, axial):
    """
    Preprocess data for circular stats functions, as need: convert deg->rad, axial->circular
    """
    data = np.asarray(data)
    assert np.isrealobj(data), ValueError("Data is complex -- should be input as angles!!!")

    if degrees: data = np.deg2rad(data)
    if axial: data = 2*data

    return data

""" Unit tests for circstats.py module """
import pytest
from math import pi
import numpy as np

from spynal.tests.data_fixtures import MISSING_ARG_ERRS
from spynal.utils import set_random_seed
from spynal.circstats import wrap, circ_distance, circ_subtract, \
                             amp_phase_to_complex, complex_to_amp_phase, \
                             circ_mean, circ_average, circ_rbar, \
                             circ_rbar2_unbiased, circ_var, circ_std, \
                             circ_sem, von_mises_kappa, \
                             rayleigh_test, circ_mean_test, circ_ANOVA1, \
                             circ_circ_correlation, circ_linear_correlation, circ_linear_regression

# TODO Tests for: circ_hist

# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('units', ['radians', 'degrees'])
def test_complex_to_amp_phase(one_sample_circ_data_parametered, units):
    """ Unit tests for amp_phase_to_complex() and complex_to_amp_phase() """
    degrees = units == 'degrees'

    theta, _ = one_sample_circ_data_parametered
    theta = theta[units+'_circle']
    amp = np.random.rand(*theta.shape)
    amp_orig = amp.copy()
    theta_orig = theta.copy()

    c = amp_phase_to_complex(amp, theta, degrees=degrees)
    assert np.array_equal(amp, amp_orig)     # Ensure input data not altered by func
    assert np.array_equal(theta, theta_orig)
    c_orig = c.copy()

    amp_, theta_ = complex_to_amp_phase(c, degrees=degrees)
    theta_ = wrap(theta_, degrees=degrees)
    assert np.array_equal(c, c_orig)     # Ensure input data not altered by func
    assert np.allclose(amp_, amp)
    assert np.allclose(theta_, theta)


@pytest.mark.parametrize('units, range', [('radians', 'circle'),
                                          ('degrees', 'circle'),
                                          ('radians', 'axial'),
                                          ('degrees', 'axial')])
def test_wrap(one_sample_circ_data_parametered, units, range):
    """ Unit tests for wrap() """
    degrees = units == 'degrees'
    axial = range == 'axial'
    if axial:   limits = (0,180) if degrees else (0,pi)
    else:       limits = (0,360) if degrees else (0,2*pi)

    data, _ = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data_orig = data.copy()

    # Basic test of shape, value of output
    wrapped = wrap(data, degrees=degrees, axial=axial)
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert np.allclose(wrapped, data, rtol=1e-4, atol=1e-4)

    # Test explicit input of limits
    wrapped = wrap(data, limits=limits, degrees=degrees, axial=axial)
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert np.allclose(wrapped, data, rtol=1e-4, atol=1e-4)

    # Test wrapping of +/- 360
    wrapped = wrap(data+limits[1], degrees=degrees, axial=axial)
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert np.allclose(wrapped, data, rtol=1e-4, atol=1e-4)

    wrapped = wrap(data-limits[1], degrees=degrees, axial=axial)
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert np.allclose(wrapped, data, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        wrapped = wrap(data, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('func, units, range, result', [('distance',  'radians',  'circle', 2.0706),
                                                        ('distance',  'radians',  'axial',  2.0706),
                                                        ('subtract',  'degrees',  'circle', 2.0706),
                                                        ('subtract',  'degrees',  'axial',  2.0706),
                                                        ('distance',  'radians',  'circle', 2.0706),
                                                        ('distance',  'radians',  'axial',  2.0706),
                                                        ('subtract',  'degrees',  'circle', 2.0706),
                                                        ('subtract',  'degrees',  'axial',  2.0706)])
def test_circ_diff(paired_circ_data_parametered, func, units, range, result):
    """ Unit tests for circ_distance() and circ_subtract() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data1, data2, data2rad = paired_circ_data_parametered
    data1 = data1[units+'_'+range]
    data2 = data2[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    n_trials, n_chnls = data1.shape
    data1_orig = data1.copy()
    data2_orig = data2.copy()

    circ_diff = circ_distance if func == 'distance' else circ_subtract

    # Basic test of shape, value of output
    d = data2rad(circ_diff(data1, data2, degrees=degrees, axial=axial))
    print(units, range, d.shape, d[0,0])
    assert np.array_equal(data1, data1_orig)     # Ensure input data not altered by func
    assert np.array_equal(data2, data2_orig)
    assert d.shape == (n_trials,n_chnls)
    assert np.isclose(d[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    d = data2rad(circ_diff(data1.T, data2.T, degrees=degrees, axial=axial))
    assert d.shape == (n_chnls,n_trials)
    assert np.isclose(d[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    d = data2rad(circ_diff(data1[:,0], data2[:,0], degrees=degrees, axial=axial))
    assert d.shape == (n_trials,)
    assert np.isclose(d[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with args swapped
    d = data2rad(circ_diff(data2, data1, degrees=degrees, axial=axial))
    assert np.isclose(d[0,0], -result if func == 'subtract' else result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        d = circ_diff(data1, data2, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    1.1663),
                                                  ('degrees', 'circle',    1.1663),
                                                  ('radians', 'axial',     1.1663),
                                                  ('degrees', 'axial',     1.1663)])
def test_circ_mean(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for circ_mean() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    mu = data2rad(circ_mean(data, axis=0, degrees=degrees, axial=axial))
    print(units, range, mu.shape, mu[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert mu.shape == (n_chnls,)
    assert np.isclose(mu[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    mu = data2rad(circ_mean(data, axis=0, degrees=degrees, axial=axial, keepdims=True))
    assert mu.shape == (1,n_chnls)
    assert np.isclose(mu[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    mu = data2rad(circ_mean(data.T, axis=-1, degrees=degrees, axial=axial))
    assert mu.shape == (n_chnls,)
    assert np.isclose(mu[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    mu = data2rad(circ_mean(data[:,0], axis=None, degrees=degrees, axial=axial))
    assert np.isscalar(mu)
    assert np.isclose(mu, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        mu = circ_mean(data, axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',   -1.0809),
                                                  ('degrees', 'circle',   -1.0809),
                                                  ('radians', 'axial',    -1.0809),
                                                  ('degrees', 'axial',    -1.0809)])
def test_circ_average(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for circ_average() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    n_obs, n_chnls = data.shape
    data_orig = data.copy()
    set_random_seed(1)
    weights = np.random.rand(n_obs, n_chnls)

    # Basic test of shape, value of output
    mu = data2rad(circ_average(data, axis=0, weights=weights, degrees=degrees, axial=axial))
    print(units, range, mu.shape, mu[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert mu.shape == (n_chnls,)
    assert np.isclose(mu[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    mu = data2rad(circ_average(data, axis=0, weights=weights, degrees=degrees, axial=axial,
                               keepdims=True))
    assert mu.shape == (1,n_chnls)
    assert np.isclose(mu[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    mu = data2rad(circ_average(data.T, axis=-1, weights=weights.T, degrees=degrees, axial=axial))
    assert mu.shape == (n_chnls,)
    assert np.isclose(mu[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    mu = data2rad(circ_average(data[:,0], axis=None, weights=weights[:,0], degrees=degrees, axial=axial))
    assert np.isscalar(mu)
    assert np.isclose(mu, result, rtol=1e-4, atol=1e-4)

    # Test for broadcasting of weights
    mu = data2rad(circ_average(data, axis=0, weights=weights[:,0], degrees=degrees, axial=axial))
    assert mu.shape == (n_chnls,)
    assert np.isclose(mu[0], result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        mu = circ_average(data, axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('algorithm, units, range, result', [('biased',    'radians', 'circle',    0.2026),
                                                             ('biased',    'degrees', 'circle',    0.2026),
                                                             ('biased',    'radians', 'axial',     0.2026),
                                                             ('biased',    'degrees', 'axial',     0.2026),
                                                             ('unbiased',  'radians', 'circle',    0.0215),
                                                             ('unbiased',  'degrees', 'circle',    0.0215),
                                                             ('unbiased',  'radians', 'axial',     0.0215),
                                                             ('unbiased',  'degrees', 'axial',     0.0215)])
def test_circ_rbar(one_sample_circ_data_parametered, algorithm, units, range, result):
    """ Unit tests for circ_rbar() and circ_rbar2_unbiased() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    func = circ_rbar if algorithm == 'biased' else circ_rbar2_unbiased

    # Basic test of shape, value of output
    rbar = func(data, axis=0, degrees=degrees, axial=axial)
    print(units, range, rbar.shape, rbar[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert rbar.shape == (n_chnls,)
    assert np.isclose(rbar[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    rbar = func(data, axis=0, degrees=degrees, axial=axial, keepdims=True)
    assert rbar.shape == (1,n_chnls)
    assert np.isclose(rbar[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    rbar = func(data.T, axis=-1, degrees=degrees, axial=axial)
    assert rbar.shape == (n_chnls,)
    assert np.isclose(rbar[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    rbar = func(data[:,0], axis=None, degrees=degrees, axial=axial)
    assert np.isscalar(rbar)
    assert np.isclose(rbar, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        rbar = func(data, axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('method, units, range, result', [('Fisher_Mardia', 'radians', 'circle',    0.7974),
                                                          ('Fisher_Mardia', 'degrees', 'circle',    0.7974),
                                                          ('Fisher_Mardia', 'radians', 'axial',     0.7974),
                                                          ('Fisher_Mardia', 'degrees', 'axial',     0.7974),
                                                          ('Batschelet',    'radians', 'circle',    1.5947),
                                                          ('Batschelet',    'degrees', 'circle',    1.5947),
                                                          ('Batschelet',    'radians', 'axial',     1.5947),
                                                          ('Batschelet',    'degrees', 'axial',     1.5947),
                                                          ('circvar2',      'radians', 'circle',    3.1927),
                                                          ('circvar2',      'degrees', 'circle',    3.1927),
                                                          ('circvar2',      'radians', 'axial',     3.1927),
                                                          ('circvar2',      'degrees', 'axial',     3.1927),
                                                          ('dispersion',    'radians', 'circle',    13.6616),
                                                          ('dispersion',    'degrees', 'circle',    13.6616),
                                                          ('dispersion',    'radians', 'axial',     13.6616),
                                                          ('dispersion',    'degrees', 'axial',     13.6616)])
def test_circ_var(one_sample_circ_data_parametered, method, units, range, result):
    """ Unit tests for circ_var() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    var = circ_var(data, axis=0, method=method, degrees=degrees, axial=axial)
    print(units, range, var.shape, var[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by circ_var
    assert var.shape == (n_chnls,)
    assert np.isclose(var[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    var = circ_var(data, axis=0, method=method, degrees=degrees, axial=axial, keepdims=True)
    assert var.shape == (1,n_chnls)
    assert np.isclose(var[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    var = circ_var(data.T, axis=-1, method=method, degrees=degrees, axial=axial)
    assert var.shape == (n_chnls,)
    assert np.isclose(var[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    var = circ_var(data[:,0], axis=None, method=method, degrees=degrees, axial=axial)
    assert np.isscalar(var)
    assert np.isclose(var, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        var = circ_var(data, axis=None, method=method, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('method, units, range, result', [('Fisher_Mardia', 'radians', 'circle',    1.7868),
                                                          ('Fisher_Mardia', 'degrees', 'circle',    1.7868),
                                                          ('Fisher_Mardia', 'radians', 'axial',     1.7868),
                                                          ('Fisher_Mardia', 'degrees', 'axial',     1.7868),
                                                          ('Batschelet',    'radians', 'circle',    1.2628),
                                                          ('Batschelet',    'degrees', 'circle',    1.2628),
                                                          ('Batschelet',    'radians', 'axial',     1.2628),
                                                          ('Batschelet',    'degrees', 'axial',     1.2628),
                                                          ('dispersion',    'radians', 'circle',    3.6962),
                                                          ('dispersion',    'degrees', 'circle',    3.6962),
                                                          ('dispersion',    'radians', 'axial',     3.6962),
                                                          ('dispersion',    'degrees', 'axial',     3.6962)])
def test_circ_std(one_sample_circ_data_parametered, method, units, range, result):
    """ Unit tests for circ_std() and (implicitly) circ_dispersion() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    std = circ_std(data, axis=0, method=method, degrees=degrees, axial=axial)
    print(units, range, std.shape, std[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by circ_std
    assert std.shape == (n_chnls,)
    assert np.isclose(std[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    std = circ_std(data, axis=0, method=method, degrees=degrees, axial=axial, keepdims=True)
    assert std.shape == (1,n_chnls)
    assert np.isclose(std[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    std = circ_std(data.T, axis=-1, method=method, degrees=degrees, axial=axial)
    assert std.shape == (n_chnls,)
    assert np.isclose(std[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    std = circ_std(data[:,0], axis=None, method=method, degrees=degrees, axial=axial)
    assert np.isscalar(std)
    assert np.isclose(std, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        std = circ_std(data, axis=None, method=method, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    0.5227),
                                                  ('degrees', 'circle',    0.5227),
                                                  ('radians', 'axial',     0.5227),
                                                  ('degrees', 'axial',     0.5227)])
def test_circ_sem(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for circ_sem() and (implicitly) circ_dispersion() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    sem = circ_sem(data, axis=0, degrees=degrees, axial=axial)
    print(units, range, sem.shape, sem[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by circ_sem
    assert sem.shape == (n_chnls,)
    assert np.isclose(sem[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    sem = circ_sem(data, axis=0, degrees=degrees, axial=axial, keepdims=True)
    assert sem.shape == (1,n_chnls)
    assert np.isclose(sem[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    sem = circ_sem(data.T, axis=-1, degrees=degrees, axial=axial)
    assert sem.shape == (n_chnls,)
    assert np.isclose(sem[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    sem = circ_sem(data[:,0], axis=None, degrees=degrees, axial=axial)
    assert np.isscalar(sem)
    assert np.isclose(sem, result, rtol=1e-4, atol=1e-4)

    # Test is_stat arg
    sem = circ_sem(data, axis=0, degrees=degrees, axial=axial, is_stat=True)
    print(units, range, sem.shape, sem[0])
    assert np.isclose(sem[0], result*np.sqrt(data.shape[0]), rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        sem = circ_sem(data[:,0], axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    0.4139),
                                                  ('degrees', 'circle',    0.4139),
                                                  ('radians', 'axial',     0.4139),
                                                  ('degrees', 'axial',     0.4139)])
def test_von_mises_kappa(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for von_mises_kappa() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    n, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    kappa = von_mises_kappa(data, axis=0, degrees=degrees, axial=axial)
    print(units, range, kappa.shape, kappa[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by von_mises_kappa
    assert kappa.shape == (n_chnls,)
    assert np.isclose(kappa[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    kappa = von_mises_kappa(data, axis=0, degrees=degrees, axial=axial, keepdims=True)
    assert kappa.shape == (1,n_chnls)
    assert np.isclose(kappa[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    kappa = von_mises_kappa(data.T, axis=-1, degrees=degrees, axial=axial)
    assert kappa.shape == (n_chnls,)
    assert np.isclose(kappa[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    kappa = von_mises_kappa(data[:,0], axis=None, degrees=degrees, axial=axial)
    assert np.isscalar(kappa)
    assert np.isclose(kappa, result, rtol=1e-4, atol=1e-4)

    # Test with Rbar,n args
    Rbar = circ_rbar(data, axis=0, degrees=degrees, axial=axial)
    Rbar_orig = Rbar.copy()
    kappa = von_mises_kappa(Rbar=Rbar, n=n, axis=0)
    assert np.array_equal(Rbar,Rbar_orig)
    assert kappa.shape == (n_chnls,)
    assert np.isclose(kappa[0], result, rtol=1e-4, atol=1e-4)

    kappa = von_mises_kappa(Rbar=Rbar[0], n=n, axis=0)
    assert np.isscalar(kappa)
    assert np.isclose(kappa, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        kappa = von_mises_kappa(data[:,0], axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    (0.1283, 2.0530)),
                                                  ('degrees', 'circle',    (0.1283, 2.0530)),
                                                  ('radians', 'axial',     (0.1283, 2.0530)),
                                                  ('degrees', 'axial',     (0.1283, 2.0530))])
def test_rayleigh_test(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for rayleigh_test() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, _ = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    p, Z = rayleigh_test(data, axis=0, return_stats=True, degrees=degrees, axial=axial)
    print(units, range, p.shape, p[0], Z.shape, Z[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert p.shape == (n_chnls,)
    assert Z.shape == (n_chnls,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(Z[0], result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    p, Z = rayleigh_test(data, axis=0, return_stats=True, degrees=degrees, axial=axial, keepdims=True)
    assert p.shape == (1,n_chnls)
    assert Z.shape == (1,n_chnls)
    assert np.isclose(p[0,0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(Z[0,0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    p, Z = rayleigh_test(data.T, axis=-1, return_stats=True, degrees=degrees, axial=axial)
    assert p.shape == (n_chnls,)
    assert Z.shape == (n_chnls,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(Z[0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    p, Z = rayleigh_test(data[:,0], axis=None, return_stats=True, degrees=degrees, axial=axial)
    assert np.isscalar(p)
    assert np.isscalar(Z)
    assert np.isclose(p, result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(Z, result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with default return_stats=False call
    p = rayleigh_test(data, axis=0, degrees=degrees, axial=axial, keepdims=True)
    assert p.shape == (1,n_chnls)
    assert np.isclose(p[0,0], result[0], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        p = rayleigh_test(data[:,0], axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    (0.0786, 1.7587)),
                                                  ('degrees', 'circle',    (0.0786, 1.7587)),
                                                  ('radians', 'axial',     (0.0786, 1.7587)),
                                                  ('degrees', 'axial',     (0.0786, 1.7587))])
def test_circ_mean_test(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for circ_mean_test() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    _, n_chnls = data.shape
    data_orig = data.copy()

    # Basic test of shape, value of output
    p, S = circ_mean_test(data, axis=0, return_stats=True, degrees=degrees, axial=axial)
    print(units, range, p.shape, p[0], S.shape, S[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert p.shape == (n_chnls,)
    assert S.shape == (n_chnls,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(S[0], result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    p, S = circ_mean_test(data, axis=0, return_stats=True, degrees=degrees, axial=axial,
                          keepdims=True)
    assert p.shape == (1,n_chnls)
    assert S.shape == (1,n_chnls)
    assert np.isclose(p[0,0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(S[0,0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    p, S = circ_mean_test(data.T, axis=-1, return_stats=True, degrees=degrees, axial=axial)
    assert p.shape == (n_chnls,)
    assert S.shape == (n_chnls,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(S[0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    p, S = circ_mean_test(data[:,0], axis=None, return_stats=True, degrees=degrees, axial=axial)
    assert np.isscalar(p)
    assert np.isscalar(S)
    assert np.isclose(p, result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(S, result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with default mu!=0
    mu = 45 if degrees else pi/4
    p, S = circ_mean_test(data+mu, axis=0, return_stats=True, mu=mu, degrees=degrees, axial=axial)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(S[0], result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with default return_stats=False call
    p = circ_mean_test(data, axis=0, degrees=degrees, axial=axial, keepdims=True)
    assert p.shape == (1,n_chnls)
    assert np.isclose(p[0,0], result[0], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        p = circ_mean_test(data[:,0], axis=None, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('units, range, result', [('radians', 'circle',    (0.1254, 2.0424)),
                                                  ('degrees', 'circle',    (0.1254, 2.0424)),
                                                  ('radians', 'axial',     (0.1254, 2.0424)),
                                                  ('degrees', 'axial',     (0.1254, 2.0424))])
def test_circ_ANOVA1(one_sample_circ_data_parametered, units, range, result):
    """ Unit tests for circ_ANOVA1() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data, data2rad = one_sample_circ_data_parametered
    data = data[units+'_'+range]
    n, n_chnls = data.shape
    data_orig = data.copy()
    n_groups = 5
    n_per_group = int(round(n/n_groups))
    labels = np.ravel(np.arange(n_groups)[:,np.newaxis] * np.ones((1,n_per_group)))

    # Basic test of shape, value of output
    p, stats = circ_ANOVA1(data, labels, axis=0, return_stats=True, degrees=degrees, axial=axial)
    print(units, range, p.shape, p[0], stats['R'].shape, stats['pev'][0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert p.shape == (n_chnls,)
    assert stats['R'].shape == (n_groups,n_chnls)
    assert stats['pev'].shape == (n_chnls,)
    assert stats['F'].shape == (n_chnls,)
    assert stats['n'].shape == (n_groups,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(stats['pev'][0], result[1], rtol=1e-4, atol=1e-4)
    assert np.array_equal(stats['n'], n_per_group*np.ones((n_groups,)))

    # Test for expected output with keepdims=True call
    p, stats = circ_ANOVA1(data, labels, axis=0, return_stats=True,
                           degrees=degrees, axial=axial, keepdims=True)
    assert p.shape == (1,n_chnls)
    assert stats['R'].shape == (n_groups,n_chnls)
    assert stats['pev'].shape == (1,n_chnls)
    assert stats['F'].shape == (1,n_chnls)
    assert stats['n'].shape == (n_groups,)
    assert np.isclose(p[0,0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(stats['pev'][0,0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    p, stats = circ_ANOVA1(data.T, labels, axis=-1, return_stats=True,
                           degrees=degrees, axial=axial)
    assert p.shape == (n_chnls,)
    assert stats['R'].shape == (n_chnls,n_groups)
    assert stats['pev'].shape == (n_chnls,)
    assert stats['F'].shape == (n_chnls,)
    assert stats['n'].shape == (n_groups,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(stats['pev'][0], result[1], rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    p, stats = circ_ANOVA1(data[:,0], labels, axis=0, return_stats=True,
                           degrees=degrees, axial=axial)
    assert np.isscalar(p)
    assert stats['R'].shape == (n_groups,)
    assert np.isscalar(stats['pev'])
    assert np.isscalar(stats['F'])
    assert stats['n'].shape == (n_groups,)
    assert np.isclose(p, result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(stats['pev'], result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with return_stats=False call
    p = circ_ANOVA1(data, labels, axis=0, return_stats=False, degrees=degrees, axial=axial)
    assert p.shape == (n_chnls,)
    assert np.isclose(p[0], result[0], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        p,stats = circ_ANOVA1(data, labels, axis=0, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('method, units, range, result',
                         [('js',    'radians', 'circle',    -0.0557),
                          ('js',    'degrees', 'circle',    -0.0557),
                          ('js',    'radians', 'axial',     -0.0557),
                          ('js',    'degrees', 'axial',     -0.0557),
                          ('fl',    'radians', 'circle',    0.0103),
                          ('fl',    'degrees', 'circle',    0.0103),
                          ('fl',    'radians', 'axial',     0.0103),
                          ('fl',    'degrees', 'axial',     0.0103)])
def test_circ_circ_correlation(paired_circ_data_parametered, method, units, range, result):
    """ Unit tests for circ_circ_correlation() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    data1, data2, data2rad = paired_circ_data_parametered
    data1 = data1[units+'_'+range]
    data2 = data2[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = data1.shape
    data1_orig = data1.copy()
    data2_orig = data2.copy()

    # Basic test of shape, value of output
    r = circ_circ_correlation(data1, data2, axis=0, method=method, degrees=degrees, axial=axial)
    print(method, units, range, r.shape, r[0])
    assert np.array_equal(data1,data1_orig)     # Ensure input data not altered by circ_circ_correlation
    assert np.array_equal(data2,data2_orig)
    assert r.shape == (n_chnls,)
    assert np.isclose(r[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    r = circ_circ_correlation(data1, data2, axis=0, method=method, degrees=degrees, axial=axial, keepdims=True)
    assert r.shape == (1,n_chnls)
    assert np.isclose(r[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    r = circ_circ_correlation(data1.T, data2.T, axis=-1, method=method, degrees=degrees, axial=axial)
    assert r.shape == (n_chnls,)
    assert np.isclose(r[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    r = circ_circ_correlation(data1[:,0], data2[:,0], axis=None, method=method, degrees=degrees, axial=axial)
    assert np.isscalar(r)
    assert np.isclose(r, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        r = circ_circ_correlation(data1[:,0], data2[:,0], axis=None, method=method, degrees=degrees, axial=axial, foo=None)


@pytest.mark.parametrize('method, units, range, result',
                         [('js',    'radians', 'circle',    (-0.0016, np.nan)),
                          ('js',    'degrees', 'circle',    (-0.0016, np.nan)),
                          ('js',    'radians', 'axial',     (-0.0016, np.nan)),
                          ('js',    'degrees', 'axial',     (-0.0016, np.nan)),
                          ('mardia','radians', 'circle',    (0.0389, 0.9628)),
                          ('mardia','degrees', 'circle',    (0.0389, 0.9628)),
                          ('mardia','radians', 'axial',     (0.0389, 0.9628)),
                          ('mardia','degrees', 'axial',     (0.0389, 0.9628))])
def test_circ_linear_correlation(paired_circ_linear_data_parametered, method, units, range, result):
    """ Unit tests for circ_linear_correlation() """
    degrees = units == 'degrees'
    axial = range == 'axial'

    circ_data, linear_data, data2rad = paired_circ_linear_data_parametered
    circ_data = circ_data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    _, n_chnls = circ_data.shape
    circ_data_orig = circ_data.copy()
    linear_data_orig = linear_data.copy()

    # Basic test of shape, value of output
    if method == 'mardia':
        r,p = circ_linear_correlation(circ_data, linear_data, axis=0, method=method,
                                      degrees=degrees, axial=axial, return_stats=True)
        print(units, range, r.shape, r[0], p[0])
        assert np.isclose(p[0], result[1], rtol=1e-4, atol=1e-4)
        assert p.shape == (n_chnls,)
    else:
        r = circ_linear_correlation(circ_data, linear_data, axis=0, method=method,
                                    degrees=degrees, axial=axial, return_stats=False)
        print(units, range, r.shape, r[0])

    assert np.array_equal(circ_data,circ_data_orig)     # Ensure input data not altered by circ_linear_correlation
    assert np.array_equal(linear_data,linear_data_orig)
    assert r.shape == (n_chnls,)
    assert np.isclose(r[0], result[0], rtol=1e-4, atol=1e-4)

    # Test for expected output with keepdims=True call
    if method == 'mardia':
        r,p = circ_linear_correlation(circ_data, linear_data, axis=0, method=method,
                                      degrees=degrees, axial=axial, keepdims=True, return_stats=True)
        assert p.shape == (1,n_chnls)
        assert np.isclose(p[0,0], result[1], rtol=1e-4, atol=1e-4)
    else:
        r = circ_linear_correlation(circ_data, linear_data, axis=0, method=method,
                                    degrees=degrees, axial=axial, keepdims=True, return_stats=False)

    assert r.shape == (1,n_chnls)
    assert np.isclose(r[0,0], result[0], rtol=1e-4, atol=1e-4)

    # Test for consistent output with transposed data dimensionality
    if method == 'mardia':
        r,p = circ_linear_correlation(circ_data.T, linear_data.T, axis=-1, method=method,
                                      degrees=degrees, axial=axial, return_stats=True)
        assert p.shape == (n_chnls,)
        assert np.isclose(p[0], result[1], rtol=1e-4, atol=1e-4)
    else:
        r = circ_linear_correlation(circ_data.T, linear_data.T, axis=-1, method=method,
                                    degrees=degrees, axial=axial, return_stats=False)

    assert r.shape == (n_chnls,)
    assert np.isclose(r[0], result[0], rtol=1e-4, atol=1e-4)

    # Test for consistent output with vector-valued data
    if method == 'mardia':
        r,p = circ_linear_correlation(circ_data[:,0], linear_data[:,0], axis=None, method=method,
                                      degrees=degrees, axial=axial, return_stats=True)
        assert np.isscalar(p)
        assert np.isclose(p, result[1], rtol=1e-4, atol=1e-4)
    else:
        r = circ_linear_correlation(circ_data[:,0], linear_data[:,0], axis=None, method=method,
                                    degrees=degrees, axial=axial, return_stats=False)
    assert np.isscalar(r)
    assert np.isclose(r, result[0], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        r = circ_linear_correlation(circ_data[:,0], linear_data[:,0], axis=None, method=method,
                                    foo=None)


# Set this to ignore expected warnings on using non-1-R error loss w/o fitting constant
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize('fit_method, error, result',
                         [('gridsearch',    '1-R',      (0.0418, 0.0000, 0.6485)),
                          ('gridsearch',    'SSE',      (0.0418, 2.0944, 95.3874)),
                          ('gridsearch',    'deviance', (0.0418, 2.0944, 32.7265)),
                          ('optimization',  '1-R',      (0.5027, 0.0000, 0.6002)),
                          ('optimization',  'SSE',      (-0.0991, -2.3428, 93.1774)),
                          ('optimization',  'deviance', (-0.5175, 0.0607, 28.2641)),
                          ('hybrid',        '1-R',      (0.0412, 0.0000, 0.6484)),
                          ('hybrid',        'SSE',      (0.0419, 2.0554, 95.1299)),
                          ('hybrid',        'deviance', (0.0412, 1.9126, 32.4203))])
def test_circ_linear_regression(paired_circ_linear_data_parametered, fit_method, error, result):
    """ Unit tests for circ_linear_regression() """
    units = 'radians'
    range = 'circle'
    degrees = units == 'degrees'
    axial = range == 'axial'

    circ_data, linear_data, data2rad = paired_circ_linear_data_parametered
    circ_data = circ_data[units+'_'+range]
    data2rad = data2rad[units+'_'+range]
    # Extract a single (n_obs,) data series from circular_data, (n_obs,2) subarray from linear_data
    circ_data = circ_data[:,0]
    linear_data = linear_data[:,:2]
    n_obs, n_predictors = linear_data.shape
    if fit_method == 'gridsearch':      n_fits = 24*24*15
    elif fit_method == 'hybrid':        n_fits = 12*12*5
    elif fit_method == 'optimization':  n_fits = 8*8*5
    n_fits_noconstant = n_fits/15 if fit_method == 'gridsearch' else n_fits/5
    circ_data_orig = circ_data.copy()
    linear_data_orig = linear_data.copy()

    # Basic test of shape, value of output
    beta,mu,stats = circ_linear_regression(circ_data, linear_data, degrees=degrees, axial=axial,
                                           fit_method=fit_method, error=error,
                                           return_stats=True, return_all_fits=True)
    print(fit_method, error, beta.shape, beta, mu, stats['error'])
    assert beta.shape == (n_predictors,)
    assert np.isscalar(mu)
    assert np.isclose(beta[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(mu, result[1], rtol=1e-4, atol=1e-4)
    assert stats['predicted'].shape == (n_obs,)
    assert np.isscalar(stats['error'])

    assert len(stats['all_fits']) == 4
    assert (len(stats['all_fits']['sample']) == n_fits) and (stats['all_fits']['sample'][0].shape == (n_predictors+1,))
    assert (len(stats['all_fits']['beta']) == n_fits) and (stats['all_fits']['beta'][0].shape == (n_predictors,))
    assert (len(stats['all_fits']['mu']) == n_fits) and np.isscalar(stats['all_fits']['mu'][0])
    assert (len(stats['all_fits']['error']) == n_fits) and np.isscalar(stats['all_fits']['error'][0])

    # Test for consistent output with return_stats=False
    beta,mu = circ_linear_regression(circ_data, linear_data, degrees=degrees, axial=axial,
                                     fit_method=fit_method, error=error, return_stats=False)
    assert np.array_equal(circ_data,circ_data_orig)     # Ensure input data not altered by circ_linear_regression
    assert np.array_equal(linear_data,linear_data_orig)
    assert beta.shape == (n_predictors,)
    assert np.isscalar(mu)
    assert np.isclose(beta[0], result[0], rtol=1e-4, atol=1e-4)
    assert np.isclose(mu, result[1], rtol=1e-4, atol=1e-4)

    # Test for expected output with single regressor
    beta,mu,stats = circ_linear_regression(circ_data, linear_data[:,0], degrees=degrees, axial=axial,
                                           fit_method=fit_method, error=error, return_stats=True)
    assert np.isscalar(beta)
    assert np.isscalar(mu)
    assert not np.isclose(beta, result[0], rtol=1e-4, atol=1e-4)

    # Test for expected output with no constant (mean/offset angle) term
    beta,mu,stats = circ_linear_regression(circ_data, linear_data, degrees=degrees, axial=axial,
                                           fit_method=fit_method, error=error, fit_constant=False,
                                           return_stats=True, return_all_fits=True)
    assert beta.shape == (n_predictors,)
    assert np.isscalar(mu)
    assert (len(stats['all_fits']['sample']) == n_fits_noconstant) and \
           (stats['all_fits']['sample'][0].shape == (n_predictors,))
    assert mu == 0

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        beta,mu = circ_linear_regression(circ_data, linear_data, fit_method=fit_method,
                                         error=error, return_stats=False, foo=None)


def test_imports():
    """ Test different import methods for circstats module """
    # Import entire package
    import spynal
    spynal.circstats.circ_distance
    # Import module
    import spynal.circstats as circ
    circ.circ_distance
    # Import specific function from module
    from spynal.circstats import circ_distance
    circ_distance

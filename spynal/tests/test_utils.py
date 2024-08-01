""" Unit tests for utils.py module """
import pytest
import random
from math import pi
import numpy as np

from spynal.tests.data_fixtures import one_sample_data, two_sample_data, one_way_data, \
                                       two_way_data, simulate_dataset, MISSING_ARG_ERRS
from spynal.utils import zscore, one_sample_tstat, paired_tstat, two_sample_tstat, \
                         one_way_fstat, two_way_fstat, fano, cv, cv2, lv, \
                         set_random_seed, randperm, interp1, setup_sliding_windows, \
                         correlation, rank_correlation, condition_mean, condition_apply, \
                         gaussian, gaussian_2d, gaussian_nd, is_symmetric, is_positive_definite, \
                         index_axis, standardize_array, undo_standardize_array, \
                         data_labels_to_data_groups, data_groups_to_data_labels, \
                         iarange, unsorted_unique, isarraylike, isnumeric, isunix, ismac, ispc, \
                         object_array_equal, object_array_compare, concatenate_object_array


# =============================================================================
# Unit tests for statistics functions
# =============================================================================
def test_zscore(one_sample_data):
    """ Unit tests for zscore() function """
    data = one_sample_data
    data_orig = data.copy()
    n_trials,n_chnls = data.shape

    # Basic test of shape, value of output. Test that z-scored data has mean ~ 0, sd ~ 1
    z = zscore(data[:,0])
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert z.shape == (n_trials,)
    assert np.isclose(z.mean(), 0, rtol=1e-2, atol=1e-2)
    assert np.isclose(z.std(), 1, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    z = zscore(data, axis=0)
    assert z.shape == (n_trials,n_chnls)
    assert np.allclose(z.mean(axis=0), np.zeros((1,n_chnls)), rtol=1e-2, atol=1e-2)
    assert np.allclose(z.std(axis=0), np.ones((1,n_chnls)), rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    z = zscore(data.T, axis=1)
    assert z.shape == (n_chnls,n_trials)
    assert np.allclose(z.mean(axis=1), np.zeros((n_chnls,1)), rtol=1e-2, atol=1e-2)
    assert np.allclose(z.std(axis=1), np.ones((n_chnls,1)), rtol=1e-2, atol=1e-2)

    # Test for proper function with zscore across entire array
    z = zscore(data, axis=None)
    assert z.shape == (n_trials,n_chnls)
    assert np.isclose(z.mean(), 0, rtol=1e-2, atol=1e-2)
    assert np.isclose(z.std(), 1, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        z = zscore(data, foo=None)


@pytest.mark.parametrize('stat_func,    result',
                         [(fano,        1.95),
                          (cv,          0.43),
                          (cv2,         0.57),
                          (lv,          0.31)])
def test_snr_stats(one_sample_data, stat_func, result):
    """ Unit tests for signal-to-noise-type stats (Fano, CV, etc.) """
    data = one_sample_data
    data_orig = data.copy()
    n_datapoints, n_chnls = data.shape

    # Basic test of shape, value of output
    stat = stat_func(data[:,0], axis=0)
    print(np.round(stat,2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isscalar(stat)
    assert np.isclose(stat, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    stat = stat_func(data[:,:], axis=0)
    assert stat.shape == (1,n_chnls)
    assert np.isclose(stat[:,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    stat = stat_func(data[:,:].T, axis=1)
    assert stat.shape == (n_chnls,1)
    assert np.isclose(stat[0,:], result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        stat = stat_func(data[:,0], axis=0, foo=None)


@pytest.mark.parametrize('stat_func,    result',
                         [(one_sample_tstat,    6.91),
                          (paired_tstat,        -2.97),
                          (two_sample_tstat,    -3.39),
                          (one_way_fstat,        1.79),
                          (two_way_fstat,       (3.10,24.91,0.18))])
def test_parametric_stats(two_way_data, stat_func, result):
    data, labels = two_way_data
    if stat_func is one_sample_tstat:
        data = data[:10,:]
    elif stat_func in [paired_tstat,two_sample_tstat]:
        data = data[:20,:]
        labels = labels[:20,0]
    elif stat_func is one_way_fstat:
        labels = labels[:,0]

    data_orig = data.copy()
    labels_orig = labels.copy()
    n_datapoints, n_chnls = data.shape

    extra_args = dict(axis=0)
    if stat_func is not two_way_fstat: extra_args.update(keepdims=True)

    # Basic test of shape, value of output
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,0], **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,0], data[labels==1,0], **extra_args)
    else:
        stat = stat_func(data[:,0], labels, **extra_args)
    print(np.round(stat,2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(labels,labels_orig)
    if stat_func is two_way_fstat:
        assert stat.shape == (3,)
        assert np.allclose(stat, result, rtol=1e-2, atol=1e-2)
    else:
        assert np.isscalar(stat)
        assert np.isclose(stat, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims = True
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,:], **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,:], data[labels==1,:], **extra_args)
    else:
        stat = stat_func(data, labels, **extra_args)
    if stat_func is two_way_fstat:
        assert stat.shape == (3,n_chnls)
        assert np.allclose(stat[:,0], result, rtol=1e-2, atol=1e-2)
    else:
        assert stat.shape == (1,n_chnls)
        assert np.isclose(stat[0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims = False
    if stat_func is not two_way_fstat:
        if stat_func is one_sample_tstat:
            stat = stat_func(data[:,:], axis=0, keepdims=False)
        elif stat_func in [paired_tstat,two_sample_tstat]:
            stat = stat_func(data[labels==0,:], data[labels==1,:], axis=0, keepdims=False)
        else:
            stat = stat_func(data, labels, axis=0, keepdims=False)
        assert stat.shape == (n_chnls,)
        assert np.isclose(stat[0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    extra_args['axis'] = 1
    if stat_func is one_sample_tstat:
        stat = stat_func(data[:,:].T, **extra_args)
    elif stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==0,:].T, data[labels==1,:].T, **extra_args)
    else:
        stat = stat_func(data.T, labels, **extra_args)
    if stat_func is two_way_fstat:
        assert stat.shape == (n_chnls,3)
        assert np.allclose(stat[0,:], result, rtol=1e-2, atol=1e-2)
    else:
        assert stat.shape == (n_chnls,1)
        assert np.isclose(stat[0,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with arguments swapped.
    # Stat should have same magnitude, opposite sign for 2-sample stats.
    if stat_func in [paired_tstat,two_sample_tstat]:
        stat = stat_func(data[labels==1,0], data[labels==0,0], axis=0)
        assert np.isscalar(stat)
        assert np.isclose(stat, -result, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        if stat_func is one_sample_tstat:
            stat = stat_func(data[:,0], axis=0, foo=None)
        elif stat_func in [paired_tstat,two_sample_tstat]:
            stat = stat_func(data[labels==0,0], data[labels==1,0], axis=0, foo=None)
        else:
            stat = stat_func(data[:,0], labels, axis=0, foo=None)


@pytest.mark.parametrize('corr_type, result, result2',
                         [('pearson',   -0.33, -0.20),
                          ('spearman',  -0.36, -0.26)])
def test_correlation(two_sample_data, corr_type, result, result2):
    """ Unit tests for correlation() and rank_correlation() functions """
    data, labels = two_sample_data
    data_orig = data.copy()

    corr_func = correlation if corr_type == 'pearson' else rank_correlation

    # Basic test of shape, value of output
    r = corr_func(data[labels==0,0], data[labels==1,0], keepdims=False)
    print(np.round(r,2), type(r))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.isscalar(r)
    assert np.isclose(r, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with arguments swapped
    r = corr_func(data[labels==1,0], data[labels==0,0], keepdims=False)
    assert np.isclose(r, result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with multi-dim array inputs and keepdims
    r = corr_func(data[labels==0,:], data[labels==1,:], axis=0, keepdims=True)
    assert r.shape == (1,4)
    assert np.isclose(r[:,0], result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed inputs
    r = corr_func(data[labels==0,:].T, data[labels==1,:].T, axis=1, keepdims=True)
    assert r.shape == (4,1)
    assert np.isclose(r[0,:], result, rtol=1e-2, atol=1e-2)

    # Test for proper function with correlation across entire array
    r = corr_func(data[labels==0,:], data[labels==1,:], axis=None)
    assert np.isscalar(r)
    assert np.isclose(r, result2, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        r = corr_func(data[labels==0,:], data[labels==1,:], foo=None)


@pytest.mark.parametrize('method, funcstr, result',
                         [('mean',  'mean', 4.6184),
                          ('apply', 'mean', 4.6184),
                          ('apply', 'std',  0.7157)])
def test_condition_apply(method, funcstr, result):
    """ Unit tests for condition_mean()/condition_apply() """
    n_trials, n_conds, n_chnls = 10, 4, 6
    data, labels = simulate_dataset(n=n_trials, n_chnls=n_chnls, n_conds=n_conds, seed=1)
    data_orig = data.copy()

    if method == 'apply':
        func = np.mean if funcstr == 'mean' else np.std
        cfunc = lambda data,labels,**kwargs: condition_apply(data, labels, function=func, **kwargs)
    else:
        cfunc = lambda data,labels,**kwargs: condition_mean(data, labels, **kwargs)

    # Basic test of shape, value of output
    kwargs = {}
    out,conds_out = cfunc(data, labels, **kwargs)
    print(method, funcstr, out[0,0])
    assert np.array_equal(data, data_orig)
    assert out.shape == (n_conds,n_chnls)
    assert (conds_out == np.arange(n_conds)).all()
    assert np.isclose(out[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistency with multi-d data
    data_stack = np.reshape(data, (-1,int(n_chnls/2),2))
    out,conds_out = cfunc(data_stack, labels, **kwargs)
    assert out.shape == (n_conds,n_chnls/2,2)
    assert np.isclose(out[0,0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistency with transposed data
    kwargs = {'axis':-1}
    out,conds_out = cfunc(data.T, labels, **kwargs)
    assert out.shape == (n_chnls,n_conds)
    assert np.isclose(out[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistency with explicit condition subset selection
    kwargs = {'axis':0, 'conditions':[0,1]}
    out,conds_out = cfunc(data, labels, **kwargs)
    assert out.shape == (2,n_chnls)
    assert (conds_out == [0,1]).all()
    assert np.isclose(out[0,0], result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        out = cfunc(data, labels, foo=None)


# =============================================================================
# Unit tests for numerical functions
# =============================================================================
@pytest.mark.parametrize('rand_func', [np.random.rand, random.random])
def test_set_random_seed(rand_func):
    """ Unit tests for test_random_seed() """
    # Generate unseeded random number using given random number generator
    set_random_seed()
    unseeded = rand_func()

    # Seed random number generators and generate seeded random number
    seed = 1
    set_random_seed(seed)
    seeded = rand_func()

    # Generate seeded random number with same seed
    set_random_seed(seed)
    seeded2 = rand_func()

    # Check that both seeded values are identical and unseeded value is not
    assert seeded == seeded2
    assert seeded != unseeded

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        rand_func(foo=None)


def test_randperm():
    """ Unit test for randperm() function """
    n,k = 4,4
    set_random_seed(1)
    out = randperm(n,k)
    assert np.array_equal(out, [3,2,0,1])
    
    n,k = 4,2
    set_random_seed(1)
    out = randperm(n,k)
    assert np.array_equal(out, [3,2])
    
    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        randperm(n, k, foo=None)
            
def test_interp1():
    """ Unit tests for interp1() function """
    x = iarange(0,1,0.1)
    y = np.cos(2*pi*4*x)
    xinterp = iarange(0,1,0.01)
    n_interp = len(xinterp)

    y_orig = y.copy()
    result = 0.0099

    # Test basic function
    yinterp = interp1(x, y, xinterp)
    print(yinterp.shape, np.round(yinterp.mean(),4))
    assert np.array_equal(y,y_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(yinterp.shape, (n_interp,))
    assert np.isclose(yinterp.mean(), result, rtol=1e-4, atol=1e-4)

    # Test function w/ 2d array data
    Y = np.tile(y[:,np.newaxis], (1,4))
    Yinterp = interp1(x, Y, xinterp, axis=0)
    assert np.array_equal(Yinterp.shape, (n_interp,4))
    assert np.allclose(Yinterp.mean(axis=0), result, rtol=1e-4, atol=1e-4)

    # Test function w/ transposed data
    print(x.shape, Y.T.shape)
    Yinterp = interp1(x, Y.T, xinterp, axis=-1)
    assert np.array_equal(Yinterp.shape, (4,n_interp))
    assert np.allclose(Yinterp.mean(axis=-1), result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        yinterp = interp1(x, y, xinterp, foo=None)


@pytest.mark.parametrize('func, result, result2',
                         [(gaussian,    0.9783, 2.3780),
                          (gaussian_2d, 0.8250, 1.3873),
                          (gaussian_nd, 0.0009, 0.6169)])
def test_gaussian(one_sample_data, func, result, result2):
    """ Unit tests for gaussian/gaussian_2d/gausssian_nd functions """
    data = (one_sample_data - 10)/5
    if func is gaussian:        data = data[:,0]
    elif func is gaussian_2d:   data = data[:,:2]
    elif func is gaussian_nd:   data = data[:,:3]
    data_orig = data.copy()
    n = data.shape[0]

    # Basic test of shape, value of output
    f_x = func(data)
    print(np.round(f_x[0],4))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert f_x.shape == (n,)
    assert np.isclose(f_x[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent results with scalar input
    if func is gaussian:    f_x = func(data[0])
    else:                   f_x = func(data[0,:])
    assert np.isscalar(f_x)
    assert np.isclose(f_x, result, rtol=1e-4, atol=1e-4)

    # Test for consistent results with hand-set parameters
    params = dict(amplitude=1, baseline=0)
    if func is gaussian:
        params.update(dict(center=0, width=1))
    elif func is gaussian_2d:
        params.update(dict(center_x=0, center_y=0, width_x=1, width_y=1, orientation=0))
    elif func is gaussian_nd:
        params.update(dict(center=np.zeros((3,)), width=np.ones((3,))))
    f_x = func(data, **params)
    assert np.isclose(f_x[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected results with another set of hand-set parameters
    params = dict(amplitude=2, baseline=0.5)
    if func is gaussian:
        params.update(dict(center=0.5, width=2))
    elif func is gaussian_2d:
        params.update(dict(center_x=0.5, center_y=-0.5, width_x=2, width_y=1, orientation=pi/4))
    elif func is gaussian_nd:
        params.update(dict(center=[0.5,-0.5,0.5], width=[2,1,2]))
    f_x = func(data, **params)
    print(np.round(f_x[0],4))
    assert np.isclose(f_x[0], result2, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        f_x = func(data, foo=None)


def test_matrix_tests():
    """ Unit tests for is_symmetric, is_positive_definite """
    # Test that matrix tests pass for Identity matrix
    X = np.eye(2)
    assert is_symmetric(X)
    assert is_positive_definite(X, semi=False)

    # Test that matrix tests fail for assymmetric, non-square matrices
    X[0,1] = 2
    assert ~is_symmetric(X)
    assert ~is_positive_definite(X, semi=False)

    X = np.concatenate((np.eye(2), np.eye(2)), axis=0)
    assert ~is_symmetric(X)
    assert ~is_positive_definite(X, semi=False)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = is_symmetric(X, foo=None)
    with pytest.raises(MISSING_ARG_ERRS):
        _ = is_positive_definite(X, semi=False, foo=None)


# =============================================================================
# Unit tests for data indexing and reshaping functions
# =============================================================================
def test_index_axis():
    data = np.reshape(np.arange(3,dtype=int), (3,1,1))
    idxs = np.arange(3,dtype=int)
    result = np.arange(3,dtype=int)

    print(data.shape)
    # Test that correct data is extracted for each of several axes
    for axis in range(3):
        print(axis, np.moveaxis(data,0,axis).shape)
        assert np.array_equal(index_axis(np.moveaxis(data,0,axis), axis, idxs).squeeze(), result)
        assert np.array_equal(index_axis(np.moveaxis(data,0,axis), axis-3, idxs).squeeze(), result)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = index_axis(data, axis, idxs, foo=None)


@pytest.mark.parametrize('axis, target_axis', [(0,0), (1,0), (2,0), (-1,0),
                                               (0,2), (1,2), (2,2), (-1,2),
                                               (0,-1), (1,-1), (2,-1), (-1,-1)])
def test_standardize_array(axis, target_axis):
    """ Unit tests for standardize_array/undo_standardize_array """
    # Test whether standardize->unstandardize returns original array
    data = np.random.rand(3,4,5)
    data_orig = data.copy()

    data2, data_shape = standardize_array(data, axis=axis, target_axis=target_axis)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function

    data2 = undo_standardize_array(data2, data_shape, axis=axis, target_axis=target_axis)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data,data2)

    # Same test w/ 2D data array
    axis = min(axis,1)  # Reset axes from 2 -> 1
    target_axis = min(target_axis,1)

    data = np.random.rand(3,4)
    data_orig = data.copy()

    data2, data_shape = standardize_array(data, axis=axis, target_axis=target_axis)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function

    data2 = undo_standardize_array(data2, data_shape, axis=axis, target_axis=target_axis)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert np.array_equal(data,data2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _,_ = standardize_array(data, axis=axis, target_axis=target_axis, foo=None)
    with pytest.raises(MISSING_ARG_ERRS):
        _,_ = undo_standardize_array(data2, data_shape, axis=axis, target_axis=target_axis, foo=None)


def test_data_groups_to_data_labels(one_way_data):
    data, labels = one_way_data
    data_orig = data.copy()
    labels_orig = labels.copy()

    # Test basic function
    data1, data2, data3 = data_labels_to_data_groups(data, labels, axis=0)
    data1_orig = data1.copy()
    assert np.array_equal(data, data_orig)      # Ensure input data isn't altered by function
    assert np.array_equal(labels, labels_orig)

    data_check, labels_check = data_groups_to_data_labels(data1, data2, data3, axis=0)
    assert np.array_equal(data1, data1_orig)    # Ensure input data isn't altered by function
    assert np.array_equal(data_check, data)
    assert np.array_equal(labels_check, labels)

    # Test for consistency with transposed data
    data1, data2, data3 = data_labels_to_data_groups(data.T, labels, axis=-1)
    data_check, labels_check = data_groups_to_data_labels(data1, data2, data3, axis=-1)
    assert np.array_equal(data_check, data.T)
    assert np.array_equal(labels_check, labels)

    # Test for consistency with subgroup selection from data
    data1, data2 = data_labels_to_data_groups(data, labels, axis=0, groups=(0,1))
    data_check, labels_check = data_groups_to_data_labels(data1, data2, axis=0)
    assert np.array_equal(data_check, data[:20,:])
    assert np.array_equal(labels_check, labels[:20])

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _,_,_ = data_labels_to_data_groups(data, labels, axis=0, foo=None)
    with pytest.raises(MISSING_ARG_ERRS):
        _,_ = data_groups_to_data_labels(data1, data2, data3, axis=0, foo=None)


# =============================================================================
# Other utility functions
# =============================================================================
@pytest.mark.parametrize('args', [(4,), (0,4), (0,4,2), (8,4,-2)])
def test_iarange(args):
    """ Unit tests for iarange() function """
    # Test that each set of arguments produces an output ending in same value
    assert iarange(*args)[-1] == 4

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = iarange(*args, foo=None)


def test_unsorted_unique():
    """ Unit tests for unsorted_unique() function """
    # Test for expected output for different arguments
    x = np.asarray((3,3,2,2,1,1))
    x_orig = x.copy()
    unique = unsorted_unique(x)
    assert np.array_equal(x,x_orig)
    assert np.array_equal(unique.shape, (3,))
    assert np.array_equal(unique, (3,2,1))

    X = np.vstack((x,x))
    X_orig = X.copy()
    unique = unsorted_unique(X,axis=1)
    assert np.array_equal(X,X_orig)
    assert np.array_equal(unique.shape, (2,3))
    assert np.array_equal(unique[0,:], (3,2,1))

    unique = unsorted_unique(X.T,axis=0)
    assert np.array_equal(unique.shape, (3,2))
    assert np.array_equal(unique[:,0], (3,2,1))

    unique = unsorted_unique(X)
    assert np.array_equal(unique.shape, (3,))
    assert np.array_equal(unique, (3,2,1))

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = unsorted_unique(x, foo=None)


@pytest.mark.parametrize('x, result',
                         [((1,2), True), ([1,2], True), (np.array((1,2)), True),
                          (1.2, False), (1, False), (True, False)])
def test_isarraylike(x, result):
    """ Unit tests for isarraylike() function """
    # Test that each variable type produces expected result
    assert isarraylike(x) == result

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = isarraylike(x, foo=None)


@pytest.mark.parametrize('dtype, result',
                         [(np.uint16, True), (np.float64, True), (np.complex128, True),
                          (bool, False), (object, False)])
def test_isnumeric(dtype, result):
    """ Unit tests for isnumeric() function """
    # Test that each dtype produces expected result
    x = np.ones((2,), dtype=dtype)
    assert isnumeric(x) == result

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = isnumeric(x, foo=None)


def test_isplatform():
    """ Unit tests for isunix(), ismac(), ispc() functions """
    # Ensure we identify 1 and only 1 platform/OS
    assert np.sum([isunix(), ismac(), ispc()]) == 1


def test_setup_sliding_windows():
    """ Unit tests for setup_sliding_windows() function """
    # Test for expected output with different argument sets
    width = 0.2
    lims = (0,1)
    wins = setup_sliding_windows(width, lims)
    assert np.array_equal(wins.shape, (5,2))
    assert np.array_equal(wins[-1,:], (0.8,1.0))

    wins = setup_sliding_windows(width, lims, step=0.1)
    assert np.array_equal(wins.shape, (9,2))
    assert np.array_equal(wins[-1,:], (0.8,1.0))

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = setup_sliding_windows(width, lims, foo=None)


@pytest.mark.parametrize('axis', [0, 1])
def test_object_array_functions(axis):
    """ Unit tests for concatenate_object_array, object_array_equal, object_array_compare """
    data = [[[1, 2],    [3, 4, 5]],
            [[6, 7, 8], [9, 10]  ]]
    data = np.asarray(data, dtype=object)
    data_orig = data.copy()

    # HACK To get Numpy to make object arrays w/o being too smart and making them (2,5) arrays
    shape = (1,2) if axis == 0 else (2,1)
    result = np.empty(shape, dtype=object)
    result_true = np.empty(shape, dtype=object)
    result_false = np.empty(shape, dtype=object)
    result_true[0,0] = [True]*5
    result_false[0,0] = [False]*5
    if axis == 0:
        result[0,0] = [1,2,6,7,8]
        result[0,1] = [3,4,5,9,10]
        result_true[0,1] = [True]*5
        result_false[0,1] = [False]*5
    elif axis == 1:
        result = np.empty((2,1), dtype=object)
        result[0,0] = [1,2,3,4,5]
        result[1,0] = [6,7,8,9,10]
        result_true[1,0] = [True]*5
        result_false[1,0] = [False]*5

    data_cat = concatenate_object_array(data, axis=axis)
    print(data_cat)
    print(result)
    assert np.array_equal(data, data_orig)  # Ensure input data isn't altered by function
    assert data_cat.shape == shape

    # Test basic `object_array_equal` call (comp_func=np.array_equal, reduce_func=np.all)
    assert object_array_equal(data_cat, result)
    # Test `object_array_equal` with alternative `comp_func`
    assert object_array_equal(data_cat, result, comp_func=np.allclose)
    # Test `object_array_equal` with alternative `reduce_func`
    assert object_array_equal(data_cat, result, reduce_func=np.sum) == 2
    # Test `object_array_equal` with `reduce_func` = None
    assert np.all(object_array_equal(data_cat, result, reduce_func=None) == [True,True])

    # Test basic `object_array_compare` call (comp_func=np.equal, reduce_func=None)
    print(object_array_compare(data_cat, result), object_array_compare(data_cat, result).shape)
    print(result_true, result_true.shape)
    assert object_array_equal(object_array_compare(data_cat, result), result_true)
    # Test `object_array_compare` with alternative `comp_func`
    assert object_array_equal(object_array_compare(data_cat, result, comp_func=np.less), result_false)
    # Test `object_array_compare` with alternative `reduce_func`
    assert np.all(object_array_compare(data_cat, result, reduce_func=np.all) == [True, True])
    assert np.all(object_array_compare(data_cat, result, reduce_func=np.sum) == [5, 5])

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        _ = concatenate_object_array(data, axis=axis, foo=None)
    with pytest.raises(MISSING_ARG_ERRS):
        _ = object_array_equal(data_cat, result, foo=None)
    with pytest.raises(MISSING_ARG_ERRS):
        _ = object_array_compare(data_cat, result, foo=None)


def test_imports():
    """ Test different import methods for utils module """
    # Import entire package
    import spynal
    spynal.utils.zscore
    # Import module
    import spynal.utils as utils
    utils.zscore
    # Import specific function from module
    from spynal.utils import zscore
    zscore

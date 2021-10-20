""" Unit tests for info.py module """
import pytest
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from patsy import dmatrix


from neural_analysis.tests.data_fixtures import two_sample_data, one_way_data, two_way_data
from neural_analysis.info import neural_info, neural_info_2groups, neural_info_ngroups


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('method, params, result',
                         [('pev',           {'omega':True},         (34.43,35.91,71.03,60.72)),
                          ('pev',           {'omega':False},        (38.99,40.41,73.55,63.94)),
                          ('dprime',        {'signed':True},        (-1.52,-1.56,-3.16,-2.53)),
                          ('dprime',        {'signed':False},       (1.52,1.56,3.16,2.53)),
                          ('auroc',         {'signed':True},        (0.14,0.13,0.01,0.04)),
                          ('auroc',         {'signed':False},       (0.86,0.87,0.99,0.96)),
                          ('mutual_info',   {},                     (0.5,0.42,0.9,0.82)),
                          ('decode',        {'decoder':'LDA'},      0.90),
                          ('decode',        {'decoder':'logistic'}, 0.70),
                          ('decode',        {'decoder':'SVM'},      0.95),
                          ('decode',        {'decoder':'custom'},   0.90)])
def test_two_sample_info(two_sample_data, method, params, result):
    """
    Unit tests for neural_info function for computing neural information
    for two-sample (two-condition) data
    """
    data, labels = two_sample_data
    data_orig = data.copy()

    n = int(10)
    n_chnls = int(4)
    info_shape = (1,n_chnls)
    # Set any extra parameters for info function
    extra_args = params
    if method == 'decode':
        extra_args['seed'] = 1
        # Test for consistency when passing custom decoder function
        if extra_args['decoder'] == 'custom':
            extra_args['decoder'] = LinearDiscriminantAnalysis(priors=(1/2)*np.ones((2,)))

    # Basic test of shape, value of output
    info = neural_info(labels, data, axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':
        print(round(info,2))
        assert isinstance(info,float)
    else:
        print(np.round(info.squeeze(),2))
        assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with 2-group form of neural computation function
    info = neural_info_2groups(data[labels==0,:], data[labels==1,:], axis=0, method=method,
                               **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with n-group form of neural computation function
    info = neural_info_ngroups(data[labels==0,:], data[labels==1,:], axis=0, method=method,
                               **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape (3rd axis)
    info = neural_info(labels, data.reshape((n*2,int(n_chnls/2),int(n_chnls/2))), axis=0,
                       method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert info.shape == (1,1,n_chnls/2)
    else:                   assert info.shape == (1,n_chnls/2,n_chnls/2)
    # Skip decoding bc different results here due to different # of channels (decoding features)
    if method != 'decode':
        assert np.allclose(info.flatten().squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    info = neural_info(labels, data.T, axis=-1, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape[::-1]
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with string-valued labels
    groups = np.asarray(['cond1','cond2'])
    info = neural_info(groups[labels], data, axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output using groups argument to subset data
    info = neural_info(np.hstack((labels,labels+2)), np.concatenate((data,data),axis=0),
                       axis=0, method=method, groups=[2,3], **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    # For mutual info, computing bins over all data != over 1 channel,
    # so this ensures we have the same binning for both
    if method == 'mutual_info':
        bins = np.histogram_bin_edges(data, bins='fd')
        bins = np.stack((bins[:-1],bins[1:]),axis=1)
        extra_args['bins'] = bins
    info = neural_info(labels, data[:,0], axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert isinstance(info,float)
    # Skip decoding bc results are different when using only single channel/feature
    if method != 'decode': assert np.isclose(info, result[0], rtol=1e-2, atol=1e-2)
    if method == 'mutual_info': extra_args.pop('bins')

    if method == 'pev':
        # Test for consistent output using regression model (instead of ANOVA1)
        # Convert list of labels into a regression design matrix
        df = pd.DataFrame(labels,columns=['cond1'])
        model_formula= '1 + C(cond1, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with constant column auto-appended in code
        df = pd.DataFrame(labels,columns=['cond1'])
        model_formula= 'C(cond1, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with returning linear model stats (and check those)
        info, stats = neural_info(labels, data, axis=0, method=method, model='anova1',
                                  return_stats=True, **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert stats['F'].shape == info_shape
        assert stats['p'].shape == info_shape
        assert stats['mu'].shape == (2,n_chnls)
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
        assert np.allclose(stats['F'].squeeze(), [11.5,12.21,50.04,31.91], rtol=1e-2, atol=1e-2)
        assert np.allclose(stats['p'].squeeze(), [3.25e-3,2.59e-3,1.35e-6,2.33e-5],
                           rtol=1e-5, atol=1e-5)
        mu = [[10.35,13.00, 3.74, 8.95], [19.45,19.15,20.77,19.00]]
        assert np.allclose(stats['mu'].squeeze(), mu, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        info = neural_info(labels, data, axis=0, method=method, foo=None, **extra_args)


@pytest.mark.parametrize('method, params, result',
                         [('pev',       {'omega':True},         (74.30,81.87,73.69,79.15)),
                          ('pev',       {'omega':False},        (76.67,83.58,76.11,81.10)),
                          ('decode',    {'decoder':'LDA'},      1.00),
                          ('decode',    {'decoder':'logistic'}, 0.80),
                          ('decode',    {'decoder':'SVM'},      0.95),
                          ('decode',    {'decoder':'custom'},   1.00)])
def test_one_way_info(one_way_data, method, params, result):
    """
    Unit tests for neural_info function for computing neural information
    for one-way/multi-class (1-factor, 3 conditions/levels) data
    """
    data, labels = one_way_data
    data_orig = data.copy()

    n = int(10)
    n_chnls = int(4)
    info_shape = (1,n_chnls)
    # Set any extra parameters for info function
    extra_args = params
    if method == 'decode':
        extra_args['seed'] = 1
        # Test for consistency when passing custom decoder function
        if extra_args['decoder'] == 'custom':
            extra_args['decoder'] = LinearDiscriminantAnalysis(priors=(1/3)*np.ones((3,)))

    # Basic test of shape, value of output
    info = neural_info(labels, data, axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':
        print(round(info,2))
        assert isinstance(info,float)
    else:
        print(np.round(info.squeeze(),2))
        assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with n-group form of neural computation function
    info = neural_info_ngroups(data[labels==0,:], data[labels==1,:], data[labels==2,:],
                               axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape (3rd axis)
    info = neural_info(labels, data.reshape((n*3,int(n_chnls/2),int(n_chnls/2))), axis=0,
                       method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    new_shape = (1,1,n_chnls/2) if method == 'decode' else (1,n_chnls/2,n_chnls/2)
    assert info.shape == new_shape
    # Skip decoding bc different results here due to different # of channels (decoding features)
    if method != 'decode':
        assert np.allclose(info.flatten().squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    info = neural_info(labels, data.T, axis=-1, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape[::-1]
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with string-valued labels
    groups = np.asarray(['cond1','cond2','cond3'])
    info = neural_info(groups[labels], data, axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output using groups argument to subset data
    info = neural_info(np.hstack((labels,labels+3)), np.concatenate((data,data),axis=0),
                       axis=0, method=method, groups=[3,4,5], **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    if method == 'decode':  assert isinstance(info,float)
    else:                   assert info.shape == info_shape
    assert np.allclose(np.asarray(info).squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    # For mutual info, computing bins over all data != over 1 channel,
    # so this ensures we have the same binning for both
    if method == 'mutual_info':
        bins = np.histogram_bin_edges(data, bins='fd')
        bins = np.stack((bins[:-1],bins[1:]),axis=1)
        extra_args ['bins'] = bins
    info = neural_info(labels, data[:,0], axis=0, method=method, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert isinstance(info,float)
    # Skip decoding bc results are different when using only single channel/feature
    if method != 'decode': assert np.isclose(info, result[0], rtol=1e-2, atol=1e-2)
    if method == 'mutual_info': extra_args.pop('bins')

    # For PEV method, test for consistent output using regression model (instead of ANOVA1)
    if method == 'pev':
        # Convert list of labels into a regression design matrix
        df = pd.DataFrame(labels,columns=['cond1'])
        model_formula= '1 + C(cond1, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with constant column auto-appended in code
        df = pd.DataFrame(labels,columns=['cond1'])
        model_formula= 'C(cond1, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with returning linear model stats (and check those)
        info, stats = neural_info(labels, data, axis=0, method=method, model='anova1',
                                  return_stats=True, **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == info_shape
        assert stats['F'].shape == info_shape
        assert stats['p'].shape == info_shape
        assert stats['mu'].shape == (3,n_chnls)
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
        assert np.allclose(stats['F'].squeeze(), [44.36,68.72,43.00,57.93], rtol=1e-2, atol=1e-2)
        assert np.allclose(stats['p'].squeeze(), [2.93e-9,2.55e-11,4.04e-9,1.71e-10],
                           rtol=1e-11, atol=1e-11)
        mu = [[10.35,13.00,3.74,8.95], [19.45,19.15,20.77,19.00], [32.87,33.69,26.73,31.10]]
        assert np.allclose(stats['mu'].squeeze(), mu, rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        info = neural_info(labels, data, axis=0, method=method, foo=None, **extra_args)


@pytest.mark.parametrize('method, interact, params, result',
                         [('pev', False,    {'omega':True},     (( 8.55, 2.91,27.23,15.67),
                                                                 (72.03,79.56,54.45,71.96))),
                          ('pev', False,    {'omega':False},    (( 9.07, 3.36,27.80,16.02),
                                                                 (72.84,80.33,55.14,72.48))),
                          ('pev', True,     {'omega':True},     (( 8.54, 2.93,27.23,15.65),
                                                                 (72.01,79.59,54.44,71.94),
                                                                 ( 0.05, 0.99, 0.31,-0.25))),
                          ('pev', True,     {'omega':False},    (( 9.07, 3.36,27.80,16.02),
                                                                 (72.84,80.33,55.14,72.48),
                                                                 ( 0.54, 1.41, 0.77, 0.07)))])
def test_two_way_info(two_way_data, method, interact, params, result):
    """
    Unit tests for neural_info function for computing neural information
    for two-way (2-factor) data
    """
    data, labels = two_way_data
    data_orig = data.copy()
    labels = labels[:,0:2]
    result = np.asarray(result)
    # Set any extra parameters for info function
    extra_args = params

    n = int(10)
    n_chnls = int(4)
    n_terms = 2 + interact

    # Basic test of shape, value of output
    info = neural_info(labels, data, axis=0, method=method, model='anova2',
                       interact=interact, **extra_args)
    print(np.round(info,2))
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert info.shape == (n_terms,n_chnls)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

    # Test for consistent output with different data array shape (3rd axis)
    info = neural_info(labels, data.reshape((n*4,int(n_chnls/2),int(n_chnls/2))), axis=0,
                       method=method, model='anova2', interact=interact, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert info.shape == (n_terms,n_chnls/2,n_chnls/2)
    assert np.allclose(info.flatten().squeeze(), result.flatten().squeeze(), rtol=1e-2, atol=1e-2)

    # Test for consistent output with transposed data dimensionality
    info = neural_info(labels, data.T, axis=-1, method=method, model='anova2',
                       interact=interact, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert info.shape == (n_chnls,n_terms)
    assert np.allclose(info.squeeze(), result.T, rtol=1e-2, atol=1e-2)

    # Test for consistent output with vector-valued data
    info = neural_info(labels, data[:,0], axis=0, method=method, model='anova2',
                       interact=interact, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert info.shape == (n_terms,)
    assert np.allclose(info, result[:,0], rtol=1e-2, atol=1e-2)

    # Test for consistent output with string-valued labels
    groups1 = np.asarray(['cond1.1','cond1.2'])
    groups2 = np.asarray(['cond2.1','cond2.2'])
    string_labels = np.stack((groups1[labels[:,0]],groups2[labels[:,1]]), axis=1)
    info = neural_info(string_labels, data, axis=0, method=method, model='anova2',
                       interact=interact, **extra_args)
    assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
    assert info.shape == (n_terms,n_chnls)
    assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

    # For PEV method, test for consistent output using regression model (instead of ANOVA1)
    if method == 'pev':
        # Convert list of labels into a regression design matrix
        df = pd.DataFrame(labels,columns=['cond1','cond2'])
        if interact:    model_formula = '1 + C(cond1, Sum)*C(cond2, Sum)'
        else:           model_formula = '1 + C(cond1, Sum) + C(cond2, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == (n_terms,n_chnls)
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with constant column auto-appended in code
        df = pd.DataFrame(labels,columns=['cond1','cond2'])
        if interact:    model_formula = 'C(cond1, Sum)*C(cond2, Sum)'
        else:           model_formula = 'C(cond1, Sum) + C(cond2, Sum)'
        design = dmatrix(model_formula,df)
        info = neural_info(design, data, axis=0, method=method, model='regress', **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == (n_terms,n_chnls)
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)

        # Test for consistent output with returning linear model stats (and check those)
        info, stats = neural_info(labels, data, axis=0, method=method, model='anova2',
                                  interact=interact, return_stats=True, **extra_args)
        assert np.array_equal(data,data_orig)     # Ensure input data isn't altered by function
        assert info.shape == (n_terms,n_chnls)
        assert stats['F'].shape == (n_terms,n_chnls)
        assert stats['p'].shape == (n_terms,n_chnls)
        assert np.allclose(info.squeeze(), result, rtol=1e-2, atol=1e-2)
        if not interact:
            F = [[19.05, 7.82,61.94,52.94], [153.03,187.17,122.84,239.53]]
            p = [[9.44e-5,8.07e-3,1.69e-9,1.04e-8], [6.75e-15,2.92e-16,1.81e-13,5.40e-18]]
            mu = [[[21.61,23.34,15.24,20.02], [28.92,27.08,29.84,30.76]],
                  [[14.90,16.08,12.26,13.97], [35.63,34.34,32.82,36.81]]]
        else:
            F = [[18.60, 8.11,61.44,50.44], [149.45,194.08,121.84,228.22], [ 1.11, 3.40, 1.69, 0.21]]
            p = [[1.20e-4,7.24e-3,2.70e-9,2.41e-8],
                 [2.24e-14,4.52e-16,14.16e-13,3.70e-17],
                 [2.99e-1,7.33e-2,2.02e-1,6.52e-1]]
            mu = [[[21.61,23.34,15.24,20.02], [28.92,27.08,29.84,30.76]],
                  [[14.90,16.08,12.26,13.97], [35.63,34.34,32.82,36.81]],
                  [[10.35,13.00, 3.74,8.95], [19.45,19.15,20.77,19.00],
                   [32.87,33.69,26.73,31.10], [38.40,35.00,38.91,42.52]]]
        assert np.allclose(stats['F'].squeeze(), F, rtol=1e-2, atol=1e-2)
        assert np.allclose(stats['p'].squeeze(), p, rtol=1e-3, atol=1e-3)
        for term in range(len(mu)):
            print("MU", term, stats['mu'][term].shape)
            assert np.allclose(stats['mu'][term].squeeze(), mu[term], rtol=1e-2, atol=1e-2)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises((TypeError,AssertionError)):
        info = neural_info(labels, data, axis=0, method=method, model='anova2', interact=interact,
                           foo=None)
        info = neural_info(design, data, axis=0, method=method, model='regress', foo=None)

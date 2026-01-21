""" Unit tests for multi.py module """
import pytest
import numpy as np

from spynal.tests.data_fixtures import MISSING_ARG_ERRS
from spynal.tests.validity_test_multi import create_subspace, simulate_subspace_dataset
from spynal.utils import set_random_seed
from spynal.multi import covariance_matrix, vector_cosine, orthogonalize_matrix, \
                         dimensionality, pc_noise_dim, pc_expvar_dim, participation_ratio, \
                         subspace_reconstruction_var, subspace_projection_var, \
                         subspace_reconstruction_index, subspace_projection_index, \
                         subspace_error_index, subspace_principal_angles
# TODO shatter_dim



# =============================================================================
# Data fixtures TEMP TODO Move to data_fixtures.py
# =============================================================================
@pytest.fixture(scope='session')
def one_sample_multivariate_data():
    """ Generate 1-sample MVN data for unit tests: basis~(3, 2), data~(200, 3), cov~(3, 3) """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    basis = create_subspace()
    data,labels = simulate_subspace_dataset(basis=basis)
    data -= data.mean(axis=0, keepdims=True)
    cov = covariance_matrix(data)

    return basis, data, cov, labels

@pytest.fixture(scope='session')
def two_sample_multivariate_data():
    """ Generate 2-sample MVN data for unit tests: basis~(3, 2), data~(200, 3), cov~(3, 3) """
    set_random_seed(1) # Note: seed=1 makes data reproducibly match output of Matlab

    basis = create_subspace()
    data,labels = simulate_subspace_dataset(basis=basis)
    data -= data.mean(axis=0, keepdims=True)
    cov = covariance_matrix(data)

    # Tweak args to create_subspace() to get diff basis for 2nd set of data
    basis2 = create_subspace(angle=30)
    data2,labels = simulate_subspace_dataset(basis=basis2)
    data2 -= data2.mean(axis=0, keepdims=True)
    cov2 = covariance_matrix(data2)

    return basis, data, cov, basis2, data2, cov2, labels


# =============================================================================
# Unit tests
# =============================================================================
def test_vector_cosine():
    """ Unit tests for vector_cosine() """
    set_random_seed(1)
    x1 = np.random.randn(20)
    x2 = np.random.randn(20)
    x1_orig = x1.copy()

    result = -0.0434

    # Basic test of shape, value of output
    c = vector_cosine(x1, x2)
    # print(c)
    assert np.array_equal(x1,x1_orig)
    assert isinstance(c,float)
    assert np.isclose(c, result, rtol=1e-4, atol=1e-4)

    # Test for transitivity
    c = vector_cosine(x2, x1)
    assert np.isclose(c, result, rtol=1e-4, atol=1e-4)

    # Test for consistency with stacked-vector array inputs
    X1 = np.column_stack((x1,x1))
    X2 = np.column_stack((x2,x2))
    X1_orig = X1.copy()
    c = vector_cosine(X1, X2, axis=0, keepdims=False)
    assert np.array_equal(X1,X1_orig)
    assert c.shape == (2,)
    assert np.isclose(c[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected results with keepdims=True
    c = vector_cosine(X1, X2, axis=0, keepdims=True)
    assert np.array_equal(X1,X1_orig)
    assert c.shape == (1,2)
    assert np.isclose(c[0,0], result, rtol=1e-4, atol=1e-4)

    # Test for consistency with transposed data
    c = vector_cosine(X1.T, X2.T, axis=1, keepdims=False)
    assert np.array_equal(X1,X1_orig)
    assert c.shape == (2,)
    assert np.isclose(c[0], result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        c = vector_cosine(x1, x2, foo=None)


@pytest.mark.parametrize('method, result', [('QR',              0.4147),
                                            ('gram-schmidt',    0.4147),
                                            ('symmetric',       0.4146)])
def test_orthogonalize(method, result):
    """ Unit tests for orthogonalize_matrix() """
    set_random_seed(1)
    n_elems,n_vects = 20,3
    X = np.random.randn(n_elems,n_vects)
    X_orig = X.copy()

    # Basic test of shape, value of output
    X_ortho = orthogonalize_matrix(X, method=method)
    print(1, method, (X_ortho @ X_ortho.T).sum()), print(np.round(X_ortho.T @ X_ortho,2))
    print(np.linalg.norm(X_ortho,axis=0))
    assert np.array_equal(X,X_orig)
    assert X_ortho.shape == (n_elems,n_vects)
    assert np.allclose(np.linalg.norm(X_ortho,axis=0), np.ones((n_vects,)))
    assert np.allclose(X_ortho.T @ X_ortho, np.eye(n_vects), rtol=1e-4, atol=1e-4)
    assert np.allclose((X_ortho @ X_ortho.T).sum(), result, rtol=1e-4, atol=1e-4)

    # Test for transitivity
    X_ortho = orthogonalize_matrix(np.flip(X,axis=1), method=method)
    assert np.allclose(np.linalg.norm(X_ortho,axis=0), np.ones((n_vects,)))
    assert np.allclose(X_ortho.T @ X_ortho, np.eye(n_vects), rtol=1e-4, atol=1e-4)
    assert np.allclose((X_ortho @ X_ortho.T).sum(), result, rtol=1e-4, atol=1e-4)

    # # TODO Need to debug behavior in near-ill-conditioned regime
    # # Test for expected results with rank < #columns
    # # X_aug = np.concatenate((X,X),axis=1) # This version (perfect collinearity) fails on symmetric, against eye()
    # X_aug = np.concatenate((X,X+1e-12*np.random.randn(n_elems,n_vects)),axis=1) # This version (approx collinearity) fails on G-S, on shape test (6 cols not 3)
    # X_ortho = orthogonalize_matrix(X_aug, method=method, rankerr=False)
    # print(3, method, (X_ortho @ X_ortho.T).sum()), print(np.round(X_ortho.T @ X_ortho,2))
    # shape = (n_elems,n_vects*2) if method == 'symmetric' else (n_elems,n_vects)
    # assert X_ortho.shape == shape
    # assert np.allclose(np.linalg.norm(X_ortho,axis=0), np.ones((shape[1],)))
    # assert np.allclose(X_ortho.T @ X_ortho, np.eye(shape[1]),rtol=1e-4, atol=1e-4)
    # # assert np.allclose((X_ortho @ X_ortho.T).sum(), result, rtol=1e-4, atol=1e-4)

    if method != 'QR':
        with pytest.raises(ValueError):
            X_ortho = orthogonalize_matrix(np.concatenate((X,X),axis=1), method=method, rankerr=True)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        X_ortho = orthogonalize_matrix(np.flip(X,axis=1), method=method, foo=None)


# ('shatter', 'Perceptron', 1)])
# ('shatter',  True,   'LDA', 1)
@pytest.mark.parametrize('method, supervised, param, result',
                         [('pc_noise', True,   'expvar', 1),
                          ('pc_noise', True,   'quantile', 1),
                          ('pc_noise', True,   'sd', 1),
                          ('pc_expvar',True,   None, 1),
                          ('pc_expvar',False,  None, 2),
                          ('pr',       True,   None, 1.0),
                          ('pr',       False,  None, 1.1842)])
def test_dimensionality(one_sample_multivariate_data, method, supervised, param, result):
    """ Unit tests for functions computing data variance explained by a subspace """
    _, data, _, labels = one_sample_multivariate_data
    data_orig = data.copy()

    if method == 'pc_noise':    extra_args = dict(noise_method=param)
    elif method == 'shatter':   extra_args = dict(decoder=param)
    else:                       extra_args = {}

    if method == 'pc_noise':    dim_func = pc_noise_dim
    elif method == 'pc_expvar': dim_func = pc_expvar_dim
    elif method == 'pr':        dim_func = participation_ratio
    # TODO elif method == 'shatter':   dim_func = shatter_dim

    labels_ = labels if supervised else None

    # Basic test of shape, value of output for single data vector / covariance matrix
    dim = dimensionality(data, labels_, method=method, **extra_args)
    print(method.upper(), dim)
    assert np.array_equal(data,data_orig) # Ensure input data not altered by func
    assert np.isscalar(dim)
    assert np.isclose(dim, result, rtol=1e-4, atol=1e-4)

    # Test for consistency with direct call to specific method function
    dim = dim_func(data, labels_, **extra_args)
    assert np.array_equal(data,data_orig) # Ensure input data not altered by func
    assert np.isscalar(dim)
    assert np.isclose(dim, result, rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        dim = dimensionality(data, labels_, method=method, foo=None, **extra_args)


@pytest.mark.parametrize('method, result', [('captured', 0.9780), ('explained', 0.9586)])
def test_subspace_variance(one_sample_multivariate_data, method, result):
    """ Unit tests for functions computing data variance explained by a subspace """
    basis, data, cov, _ = one_sample_multivariate_data
    cov = np.tile(cov[np.newaxis,:,:], (2,1,1)) # HACK: copy this to properly test non-mutation
    basis_orig = basis.copy()
    data_orig = data.copy()
    cov_orig = cov.copy()

    if method == 'captured':
        func = lambda data,cov,basis,**kwargs: subspace_reconstruction_var(data, basis, **kwargs)
        shape = (200,1)
    elif method == 'explained':
        func = lambda data,cov,basis,**kwargs: subspace_projection_var(cov, basis, **kwargs)
        shape = (2,1,1)

    # Basic test of shape, value of output for single data vector / covariance matrix
    expvar = func(data[0,:], cov[0,:,:], basis)
    print(method.upper(), expvar)
    assert np.array_equal(basis,basis_orig) # Ensure input data not altered by func
    assert np.array_equal(data,data_orig)
    assert np.array_equal(cov,cov_orig)
    assert isinstance(expvar,float)
    assert np.isclose(expvar, result, rtol=1e-4, atol=1e-4)

    # Test for consistent output for multiple stacked data vectors / covariance matrixes
    expvar = func(data, cov, basis)
    # print(method.upper(), expvar.shape, expvar[0])
    assert np.array_equal(data,data_orig)     # Ensure input data not altered by func
    assert np.array_equal(cov,cov_orig)
    assert np.array_equal(basis,basis_orig)
    assert expvar.shape == shape
    assert np.isclose(expvar[0], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output for transposed data
    feature_axis = 0 if method == 'captured' else (0,1)
    expvar = func(data.T, cov.T, basis, feature_axis=feature_axis)
    # print(method.upper(), expvar.shape, expvar.squeeze()[0])
    assert np.array_equal(basis,basis_orig) # Ensure input data not altered by func
    assert np.array_equal(data,data_orig)
    assert np.array_equal(cov,cov_orig)
    assert expvar.shape == shape[::-1]
    assert np.isclose(expvar.squeeze()[0], result, rtol=1e-4, atol=1e-4)

    # Test for expected output for keepdims=False
    expvar = func(data, cov, basis, keepdims=False)
    assert expvar.shape == (shape[0],)
    assert np.isclose(expvar[0], result, rtol=1e-4, atol=1e-4)

    if method == 'captured':
        # Test for expected output with sum_axis argument
        expvar = func(data, cov, basis, sum_axis=0)
        assert isinstance(expvar,float)
        assert np.isclose(expvar, 0.9586, rtol=1e-4, atol=1e-4)

    elif method == 'explained':
        # Test for expected output with different normalization
        expvar = func(data[0,:], cov[0,:,:], basis, normalization='none', keepdims=False)
        assert np.isclose(expvar, 109.5676, rtol=1e-4, atol=1e-4)

        expvar = func(data[0,:], cov[0,:,:], basis, normalization='feature', keepdims=False)
        assert np.isclose(expvar, 36.5225, rtol=1e-4, atol=1e-4)

        # Test for expected output with keep_components=True
        expvar = func(data[0,:], cov[0,:,:], basis, keep_components=True, keepdims=False)
        assert np.allclose(expvar, [11.5169,0.5227], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        expvar = func(data[0,:], cov[0,:,:], basis, foo=None)


@pytest.mark.parametrize('method, result', [('reconstruction',  0.6770),
                                            ('projection',      0.7707),
                                            ('error',           0.6464),
                                            ('angles',          (0,0.5236))])
def test_subspace_comparison(two_sample_multivariate_data, method, result):
    """ Unit tests for functions comparing subspaces """
    basis1, data1, cov1, basis2, data2, cov2, _ = two_sample_multivariate_data
    cov1 = np.tile(cov1[np.newaxis,:,:], (2,1,1)) # HACK: copy this to properly test non-mutation
    cov2 = np.tile(cov2[np.newaxis,:,:], (2,1,1))
    basis1_orig = basis1.copy()
    data1_orig = data1.copy()
    cov1_orig = cov1.copy()
    basis2_orig = basis2.copy()
    data2_orig = data2.copy()
    cov2_orig = cov2.copy()

    if method == 'reconstruction':
        func = lambda data1,cov1,basis1, data2,cov2,basis2, **kwargs: \
            subspace_reconstruction_index(data2, basis1, basis2, **kwargs)
        shape = (200,1)
    elif method == 'projection':
        func = lambda data1,cov1,basis1, data2,cov2,basis2, **kwargs: \
            subspace_projection_index(cov2, basis1, basis2, **kwargs)
        shape = (2,1,1)
    elif method == 'error':
        func = lambda data1,cov1,basis1, data2,cov2,basis2, **kwargs: \
            subspace_error_index(basis1, basis2, **kwargs)
        shape = (2,1,1)
    elif method == 'angles':
        func = lambda data1,cov1,basis1, data2,cov2,basis2, **kwargs: \
            subspace_principal_angles(basis1, basis2, **kwargs)
        shape = (2,1,2)

    # Basic test of shape, value of output
    comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2)
    print(method.upper(), comp)
    assert np.array_equal(basis1,basis1_orig) # Ensure input data not altered by func
    assert np.array_equal(data1,data1_orig)
    assert np.array_equal(cov1,cov1_orig)
    assert np.array_equal(basis2,basis2_orig)
    assert np.array_equal(data2,data2_orig)
    assert np.array_equal(cov2,cov2_orig)
    assert comp.shape == (2,) if method == 'angles' else isinstance(comp,float)
    assert np.allclose(comp, result, rtol=1e-4, atol=1e-4)

    # Test for consistent output for 3D array inputs
    basis1_ = np.tile(basis1[np.newaxis,:,:], (2,1,1)) if method in ['angles','error'] else basis1
    basis2_ = np.tile(basis2[np.newaxis,:,:], (2,1,1)) if method in ['angles','error'] else basis2
    comp = func(data1, cov1, basis1_, data2, cov2, basis2_)
    assert (np.array_equal(basis1_[0,:,:],basis1_orig) if method in ['angles','error']
            else np.array_equal(basis1,basis1_orig))
    assert (np.array_equal(basis2_[0,:,:],basis2_orig) if method in ['angles','error']
            else np.array_equal(basis2,basis2_orig))
    assert np.array_equal(data1,data1_orig)
    assert np.array_equal(cov1,cov1_orig)
    assert np.array_equal(data2,data2_orig)
    assert np.array_equal(cov2,cov2_orig)
    assert comp.shape == shape
    assert np.allclose(comp[0,:], result, rtol=1e-4, atol=1e-4)

    # Test for consistent output for transposed data
    if method == 'reconstruction':      extra_args = {'feature_axis': 0}
    elif method == 'projection':        extra_args = {'feature_axis': (0,1)}
    elif method in ['error','angles']:  extra_args = {'feature_axis': 1, 'component_axis': 0}
    if method in ['error','angles']:    basis1_T, basis2_T = basis1_.T, basis2_.T
    else:                               basis1_T, basis2_T = basis1_, basis2_
    basis1_T_orig, basis2_T_orig = basis1_T.copy(), basis2_T.copy()
    comp = func(data1.T, cov1.T, basis1_T, data2.T, cov2.T, basis2_T, **extra_args)
    assert np.array_equal(basis1_T,basis1_T_orig) # Ensure input data not altered by func
    assert np.array_equal(data1,data1_orig)
    assert np.array_equal(cov1,cov1_orig)
    assert np.array_equal(basis2_T,basis2_T_orig)
    assert np.array_equal(data2,data2_orig)
    assert np.array_equal(cov2,cov2_orig)
    assert comp.shape == shape[::-1]
    assert np.allclose(comp.T[0,:], result, rtol=1e-4, atol=1e-4) # comp.squeeze()[0]

    # Test for expected output for keepdims=False
    comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2, keepdims=False)
    assert len(comp) == 2 if method == 'angles' else np.isscalar(comp)
    assert np.allclose(comp, result, rtol=1e-4, atol=1e-4)

    if method == 'reconstruction':
        # Test for expected output with sum_axis argument
        comp = func(data1, cov1, basis1, data2, cov2, basis2, sum_axis=0)
        assert isinstance(comp,float)
        assert np.isclose(comp, 0.7707, rtol=1e-4, atol=1e-4)

    elif method == 'projection':
        # Test for expected output with different normalization
        comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2,
                    normalization='none', keepdims=False)
        assert np.isclose(comp, 0.7707, rtol=1e-4, atol=1e-4)

        comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2,
                    normalization='feature', keepdims=False)
        assert np.isclose(comp, 0.7707, rtol=1e-4, atol=1e-4)

        # Test for expected output with keep_components=True
        comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2,
                    keep_components=True, keepdims=False)
        assert np.allclose(comp, [0.7707,0.7706], rtol=1e-4, atol=1e-4)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        comp = func(data1[0,:], cov1[0,:,:], basis1, data2[0,:], cov2[0,:,:], basis2, foo=None)

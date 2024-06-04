""" Unit tests for matIO.py module """
import pytest
import os
import numpy as np
import pandas as pd

import tempfile

from spynal.tests.data_fixtures import MISSING_ARG_ERRS
from spynal.matIO import loadmat, load, whomat, who, savemat, save


# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('version', [('v7'), ('v73')])
def test_loadmat(version):
    """ Unit tests for loadmat function in matIO module """
    filename = _set_filename(version)

    variables = ['integer','floating','boolean','string', 'num_array','cell_array',
                 'gen_struct','table_struct']
    # scipy.io returns v7 Matlab logicals as ints (not bools) and not much we can do about it
    bool_type = int if version == 'v7' else bool

    extra_args = dict(asdict=True, extract_items=True, verbose=False)

    # Test data is loaded correctly when all variables loaded into a dict
    data = loadmat(filename, **extra_args)
    # _print_data_summary(data)
    _variable_tests(data, table_type=dict, bool_type=bool_type, extract_item=True)

    # Test function alias
    data = load(filename, **extra_args)
    _variable_tests(data, table_type=dict, extract_item=True, bool_type=bool_type)

    # Test extract_item=False
    extra_args['extract_items'] = False
    data = load(filename, **extra_args)
    _variable_tests(data, table_type=dict, extract_item=False, bool_type=bool_type)
    extra_args['extract_items'] = True

    # Test data is loaded correctly when all variables loaded into separate variables
    extra_args['asdict'] = False
    integer, floating, boolean, string, num_array, cell_array, gen_struct, table_struct = \
        loadmat(filename, variables=variables, **extra_args)
    data = {'integer':integer, 'floating':floating, 'boolean':boolean, 'string':string,
            'num_array':num_array, 'cell_array':cell_array,
            'gen_struct':gen_struct, 'table_struct':table_struct}
    _variable_tests(data, dict, bool_type=bool_type, extract_item=True)
    extra_args['asdict'] = True

    # Test for correct loading when variables to load are specified (as list or single name)
    data = loadmat(filename, variables=['integer','string'], **extra_args)
    assert list(data.keys()) == ['integer','string']
    assert isinstance(data['integer'], int)
    assert data['integer'] == 1
    assert isinstance(data['string'], str)
    assert data['string'] == 'abc'

    data = loadmat(filename, variables='integer', **extra_args)
    assert list(data.keys()) == ['integer']
    assert isinstance(data['integer'], int)
    assert data['integer'] == 1

    # Test for correct loading of table-like struct into Pandas DataFrame
    data = loadmat(filename, typemap={'table_struct':'DataFrame'}, **extra_args)
    _variable_tests(data, pd.DataFrame, bool_type)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        data = loadmat(filename, **extra_args, foo=None)


@pytest.mark.parametrize('version', [('v7'), ('v73')])
def test_whomat(version):
    """ Unit tests for whomat function in matIO module """
    filename = _set_filename(version)

    variables = ['boolean', 'cell_array', 'floating', 'gen_struct', 'integer',
                 'num_array', 'string', 'table_struct']

    # Basic test of functionality
    test_vbls = sorted(whomat(filename, verbose=False))
    assert test_vbls == variables

    # Test function alias
    test_vbls = sorted(who(filename, verbose=False))
    assert test_vbls == variables

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with pytest.raises(MISSING_ARG_ERRS):
        data = whomat(filename, verbose=False, foo=None)


@pytest.mark.parametrize('version', [('v7'), ('v73')])
def test_savemat(version):
    """ Unit tests for savemat function in matIO module """
    # TODO Add tests where variables are appended to existing file
    _version = 7 if version == 'v7' else 7.3

    # Load variables from test file
    filename = _set_filename(version)
    variables = loadmat(filename, asdict=True, verbose=False)

    # Basic test of function -- Save variables out to temporary file
    with tempfile.TemporaryDirectory() as temp_folder:
        filename = os.path.join(temp_folder, 'test_matfile_'+version+'.mat')
        savemat(filename, variables, version=_version)

    # Test function alias
    with tempfile.TemporaryDirectory() as temp_folder:
        filename = os.path.join(temp_folder, 'test_matfile_'+version+'.mat')
        save(filename, variables, version=_version)

    # Ensure that passing a nonexistent/misspelled kwarg raises an error
    with tempfile.TemporaryDirectory() as temp_folder:
        filename = os.path.join(temp_folder, 'test_matfile_'+version+'.mat')
        with pytest.raises(MISSING_ARG_ERRS):
            savemat(filename, variables, version=_version, foo=None)


def test_imports():
    """ Test different import methods for matIO module """
    # Import entire package
    import spynal
    spynal.matIO.matIO.load
    spynal.matIO.load
    # Import module
    import spynal.matIO as matIO
    matIO.matIO.load
    matIO.load
    # Import specific function from module
    from spynal.matIO import load
    load
    # Import specific function from module
    from spynal.matIO.matIO import load
    load


# =============================================================================
# Helper functions for unit tests
# =============================================================================
def _set_filename(version):
    """ Set filename for test file independent of current dir """
    # HACK Make this work when run from top-level Python dir in VS Code or from tests dir in terminal
    cwd = os.getcwd()
    if cwd.endswith('tests'):   load_dir = r'./'
    else:                       load_dir = r'./spynal/spynal/tests'
    return os.path.join(load_dir, 'testing_datafile_' + version + '.mat')


def _variable_tests(data, table_type=dict, bool_type=bool, extract_item=True):
    """ Set of tests to run on loadmat-loaded variables """
    assert isinstance(data, dict)

    # Test scalar and array variables
    _scalar_tests(data, bool_type=bool_type, extract_item=extract_item)
    _array_tests(data)

    # Test generic struct variable -- loaded as dict
    _scalar_tests(data['gen_struct'], bool_type=bool_type, extract_item=extract_item)
    _array_tests(data['gen_struct'])

    # Test table-like struct variable -- loaded as dict or Pandas DataFrame
    assert isinstance(data['table_struct'], table_type)
    col_type = np.ndarray if isinstance(data['table_struct'],dict) else pd.Series

    assert isinstance(data['table_struct']['num_array'], col_type)
    assert np.issubdtype(data['table_struct']['num_array'].dtype, float)
    assert data['table_struct']['num_array'].shape == (4,)
    assert (data['table_struct']['num_array'] == np.arange(1,5)+0.1).all()

    assert isinstance(data['table_struct']['cell_array'], col_type)
    assert np.issubdtype(data['table_struct']['cell_array'].dtype, object)
    assert data['table_struct']['cell_array'].shape == (4,)
    assert (data['table_struct']['cell_array'] == np.asarray(['abc','def','gh','ij'])).all()


def _scalar_tests(data, bool_type=bool, extract_item=True):
    """ Tests for scalar variables """
    if extract_item:
        assert isinstance(data['integer'], int)
        assert data['integer'] == 1

        assert isinstance(data['floating'], float)
        assert data['floating'] == 1.1

        assert isinstance(data['boolean'], bool_type)
        assert data['boolean'] == True

        assert isinstance(data['string'], str)
        assert data['string'] == 'abc'

    else:
        assert isinstance(data['integer'], np.ndarray) and np.issubdtype(data['integer'].dtype, np.integer)
        assert np.array_equal(data['integer'], np.array(1, dtype=np.int16))

        assert isinstance(data['floating'], np.ndarray) and np.issubdtype(data['floating'].dtype, float)
        assert np.array_equal(data['floating'], np.array(1.1, dtype=float))

        if bool_type == int: bool_type = np.integer
        assert isinstance(data['boolean'], np.ndarray) and np.issubdtype(data['boolean'].dtype, bool_type)
        assert np.array_equal(data['boolean'], np.array(True, dtype=bool))

        assert isinstance(data['string'], np.ndarray) and np.issubdtype(data['string'].dtype, object)
        assert np.array_equal(data['string'], np.array('abc', dtype=object))


def _array_tests(data):
    """ Tests for array variables """
    assert isinstance(data['num_array'], np.ndarray)
    assert np.issubdtype(data['num_array'].dtype, float)
    assert data['num_array'].shape == (4,3,2)
    assert (data['num_array'] == np.arange(1,25).reshape((4,3,2),order='F')+0.1).all()

    assert isinstance(data['cell_array'], np.ndarray)
    assert np.issubdtype(data['cell_array'].dtype, object)
    assert data['cell_array'].shape == (4,6)
    assert (data['cell_array'] == np.tile(np.asarray(['abc','def','gh','ij'])[:,np.newaxis], (1,6))).all()


def _print_data_summary(data):
    """ Print summary of key/value pairs in data struct """
    print(data.keys())
    for key in data:
        if np.isscalar(data[key]):              size = 1
        elif isinstance(data[key],np.ndarray):  size = data[key].shape
        elif hasattr(data[key], '__len__'):     size = len(data[key])
        else:                                   size = '???'
        print(key, type(data[key]), size)
        print(data[key])

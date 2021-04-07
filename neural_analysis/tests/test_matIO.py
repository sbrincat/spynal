""" Unit tests for matIO.py module """
import pytest
import string
import numpy as np
import pandas as pd

from ..matIO import loadmat

# todo Should we embed mat file generation in a fixture? Would require matlab engine for Python.

# =============================================================================
# Unit tests
# =============================================================================
@pytest.mark.parametrize('version', [('v7'), ('v73')])
def test_loadmat(version):
    """ Unit tests for loadmat function in matIO module """
    # TODO This works when run in terminal, but in VS Code can't find file bc runs in my ~/Code/Python. Fix.
    filename = r'./testing_datafile_' + version + '.mat'

    variables = ['integer','floating','boolean','string', 'num_array','cell_array', 
                 'gen_struct','table_struct']
    # scipy.io returns v7 Matlab logicals as ints (not bools) and not much we can do about it
    bool_type = int if version == 'v7' else bool
    
    # Test data is loaded correctly when all variables loaded into a dict
    data = loadmat(filename, asdict=True, verbose=False)
    # _print_data_summary(data)
    _variable_tests(data, dict, bool_type)
    
    # Test data is loaded correctly when all variables loaded into separate variables    
    integer, floating, boolean, string, num_array, cell_array, gen_struct, table_struct = \
        loadmat(filename, variables=variables, asdict=False, verbose=False)
    data = {'integer':integer, 'floating':floating, 'boolean':boolean, 'string':string, 
            'num_array':num_array, 'cell_array':cell_array, 
            'gen_struct':gen_struct, 'table_struct':table_struct}        
    _variable_tests(data, dict, bool_type)
    
    # Test for correct loading when variables to load are specified (as list or single name)
    data = loadmat(filename, variables=['integer','string'], asdict=True, verbose=False)
    assert list(data.keys()) == ['integer','string']
    assert isinstance(data['integer'], int)
    assert data['integer'] == 1
    assert isinstance(data['string'], str)
    assert data['string'] == 'abc'

    data = loadmat(filename, variables='integer', asdict=True, verbose=False)
    assert list(data.keys()) == ['integer']
    assert isinstance(data['integer'], int)
    assert data['integer'] == 1
    
    # Test for correct loading of table-like struct into Pandas DataFrame
    data = loadmat(filename, typemap={'table_struct':'DataFrame'}, asdict=True, verbose=False)
    _variable_tests(data, pd.DataFrame, bool_type)
    
    
def _variable_tests(data, table_type=dict, bool_type=bool):
    """ Set of tests to run on loadmat-loaded variables """
    assert isinstance(data, dict)

    # Test scalar and array variables
    _scalar_tests(data, bool_type)    
    _array_tests(data)
 
    # Test generic struct variable -- loaded as dict        
    _scalar_tests(data['gen_struct'], bool_type)
    _array_tests(data['gen_struct'])
    
    # Test table-like struct variable -- loaded as dict or Pandas DataFrame
    assert isinstance(data['table_struct'], table_type)    
    col_type = np.ndarray if isinstance(data['table_struct'],dict) else pd.Series
    
    assert isinstance(data['table_struct']['num_array'], col_type)
    assert np.issubdtype(data['table_struct']['num_array'].dtype, np.float) 
    assert data['table_struct']['num_array'].shape == (4,)
    assert (data['table_struct']['num_array'] == np.arange(1,5)+0.1).all()

    assert isinstance(data['table_struct']['cell_array'], col_type)
    assert np.issubdtype(data['table_struct']['cell_array'].dtype, object)     
    assert data['table_struct']['cell_array'].shape == (4,)
    assert (data['table_struct']['cell_array'] == np.asarray(['abc','def','gh','ij'])).all()           

    
def _scalar_tests(data, bool_type=bool):
    """ Tests for scalar variables """
    assert isinstance(data['integer'], int)
    assert data['integer'] == 1

    assert isinstance(data['floating'], float)
    assert data['floating'] == 1.1

    assert isinstance(data['boolean'], bool_type)
    assert data['boolean'] == True
    
    assert isinstance(data['string'], str)
    assert data['string'] == 'abc'
      
            
def _array_tests(data):
    """ Tests for array variables """
    assert isinstance(data['num_array'], np.ndarray)
    assert np.issubdtype(data['num_array'].dtype, np.float) 
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
               
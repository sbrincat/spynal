#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Helper functions for matIO package """
from types import SimpleNamespace
import numpy as np
import pandas as pd

import scipy.io

# We check for xarray variables in _variables_to_mat, but don't require as dependency
try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False
    # print("xarray module is not installed. No support for loading/saving xarray format")

from spynal.helpers import _isbinary, _enclose_in_object_array
# from spynal.matIO.matIO_7 import _v7_matlab_type, _process_v7_object

# Set=True to print out detailed debugging statements in loading functions
DEBUG = False


# =============================================================================
# General helper functions
# =============================================================================
def _structuredarray_to_dict(sarray, typemap=None, extract_item=True, extract_items=None,
                             transpose=False, level=0):
    """
    Convert Numpy structured array (which Matlab structs are initially
    imported as, using scipy.io.loadmat) to a dict, with type names as keys

    Parameters
    ----------
    sarray :  Numpy structured array

    Returns
    -------
    dic : dict, {sarray.dtype.names : sarray[key]}
    """
    # Convert structured array to dict
    dic = {name:sarray[name] for name in sarray.dtype.names}

    for name in dic.keys():
        if DEBUG: print('\t'*level, "'%s'" % name)
        # Infer Matlab variable type of struct field
        matlab_vbl_type = _v7_matlab_type(dic[name])

        # If specific field name is listed in <typemap>/<extract_items>, use given value
        python_vbl_type = typemap[name] if name in typemap else None
        extract_item = extract_items[name] if name in extract_items else extract_item

        dic[name] = _process_v7_object(dic[name], matlab_vbl_type=matlab_vbl_type,
                                       python_vbl_type=python_vbl_type, extract_item=extract_item,
                                       typemap=typemap, extract_items=extract_items,
                                       transpose=transpose, level=level)

    return dic


def _structuredarray_to_dataframe(sarray, typemap=None, extract_item=None, extract_items=None,
                                  transpose=False, level=0):
    """
    Convert Numpy structured array (which Matlab structs are initially imported
    as, using scipy.io.loadmat) to a Pandas DataFrame, with type names as keys.
    Intended for table-like Matlab scalar structs, which naturally fit into a
    DataFrame (though Matlab table variables are not currently loadable in Python).

    Parameters
    ----------
    sarray : Numpy structured array

    Returns
    -------
    df : Pandas DataFrame
        Columns are names and values of all the sarray.dtype.names
        whose associated values are vector-valued with length = modal length
        of all names. Rows are each entry from these vectors

    References
    ----------
    http://poquitopicante.blogspot.com/2014/05/loading-matlab-mat-file-into-pandas.html
    """
    # Convert structured array to dict {keys = sarray.dtype.names : values = sarray[key]}
    dic = _structuredarray_to_dict(sarray, typemap=typemap, extract_item=extract_item,
                                   extract_items=extract_items, transpose=transpose, level=level)

    # Convert dict to a Pandas DataFrame
    df = _dict_to_dataframe(dic)

    return df


def _get_matfile_version(filename):
    """
    Determines version of given MAT file (7.3 or 7/older) by reading its header

    Parameters
    ----------
    filename : str
        Full-path name of MAT-file to load from

    Returns
    -------
    version : float
        Version of MAT file: 7 (or 4,5 = older) | 7.3 (newer/HDF5)
    """
    # Read in first 10 bytes of MAT file header and convert to string ("with" auto-closes file)
    with open(filename, 'rb') as file:
        hdr = file.read(10).decode("utf-8")

    # Check for version 7 MAT file header (no clue why it says "5.0", but this is really the case)
    if 'MATLAB 5.0' in hdr:
        version = 7.0

    # Check for version 7.3 MAT file header
    elif 'MATLAB 7.3' in hdr:
        version = 7.3

    else:
        version = None
        print('WARNING: Unable to determine version of MAT file ''%s''' % filename)

    return version


def _extract_strings(x):
    """
    Deal with weird way Matlab cell-array-of-strings data structures are loaded.

    Convert them from Numpy array of objects, which are each trivial
    1-length arrays of strings, to just an array of strings themselves.
    """
    return np.asarray([(x[j][0] if len(x[j]) > 0 else '') for j in range(len(x))])


def _is_structured_array(array):
    """ Return True if input array is a Numpy structured array, False otherwise """
    # For v7.3 mat files loaded with h5py, structured arrays are ndarrays with dtype np.void
    # For v7 mat files loaded with scipy.io, structured arrays are an "np.void" type
    # Test for either version
    return (isinstance(array,np.ndarray) and (array.dtype.type is np.void)) \
         or isinstance(array,np.void)


def _dict_to_dataframe(dic):
    """
    Convert dict (which Matlab structs are initially imported as, using h5py)
    to a Pandas DataFrame, with type names as keys. Intended for table-like
    Matlab scalar structs, which naturally fit into a DataFrame
    (though Matlab tables themselves are not currently loadable in Python AKAIK).

    Parameters
    ----------
    dic : dict, {str : array-like, shape=(n_rows,)}
        Dictionary with keys to be used as column names and values that are same-length
        vectors. Values not fitting this criterion are not retained in output.

    Returns
    -------
    df : Pandas DataFrame, shape=(n_rows,n_columns)
        Columns are dict keys, for all associated values are vector-valued with
        length = modal length of all keys. Rows are each entry from these vectors.

    References
    ----------
    http://poquitopicante.blogspot.com/2014/05/loading-matlab-mat-file-into-pandas.html
    """
    # Special case: Matlab structs converted from Matlab tables may retain the
    # 'Properties' metadata attribute as a struct field. If so, we extract and
    #  remove it, convert it to dict, and below tack it onto df as an attribute.
    if 'Properties' in dic:
        metadata = dic.pop('Properties')

        # If metadata is a structured array, convert it to dict
        if _is_structured_array(metadata):
            metadata = _structuredarray_to_dict(metadata, typemap={'struct':'dict'},
                                                extract_item=True, transpose=False)
    else:
        metadata = None

    # Convert any scalar dict values -> ndarrays, and ensure ndarrays are
    # at least 1D (0-dimensional arrays mess up DF conversion and are weird)
    for key in dic.keys():
        if not isinstance(dic[key],np.ndarray) or (dic[key].ndim == 0):
            dic[key] = np.atleast_1d(dic[key])

    # Find length of each value in dict <dic> (to become DataFrame columns)
    lengths = [value.shape[0] if value.ndim > 0 else 1 for value in dic.values()]
    # Find modal value of column lengths
    # From: https://stackoverflow.com/questions/10797819/finding-the-mode-of-a-list
    height = max(set(lengths), key=lengths.count)

    # Reshape any (n,k>1) ndarrays -> (n,) vectors of k-tuples
    for key in dic.keys():
        if (dic[key].ndim > 1) and \
           (dic[key].shape[0] == height) and (dic[key].shape[1] > 1):
            dic[key] = _array_to_tuple_vector(dic[key])

    # Special case: height=1 DF with some tuple-valued columns--convert to (1,) object arrays
    if height == 1:
        for key in dic.keys():
            if dic[key].shape[0] > height: dic[key] = _enclose_in_object_array(dic[key])
            # HACK Convert empty (0-length) arrays to (1,) array with value = nan
            elif dic[key].shape[0] == 0:    dic[key] = np.atleast_1d(np.nan)

    columns = list(dic.keys())

    # Create a DataFrame with length-consistent key/value pairs as columns
    df = pd.DataFrame(dic,columns=columns,index=range(height))

    # Tack Matlab table-derived 'Properties' metadata onto df as attribute.
    if metadata is not None:
        # Setting a DF attribute with a dict throws an error.
        # Workaround here from https://stackoverflow.com/a/54137536
        df.metadata = SimpleNamespace()
        df.metadata = metadata
        # If 'Properties' had a 'RowNames' field, use that as DataFrame row index.
        if ('RowNames' in metadata):
            metadata['RowNames'] = np.atleast_1d(metadata['RowNames'])
            if metadata['RowNames'].ndim > 1: metadata['RowNames'] = metadata['RowNames'].squeeze()
            # Ensure RowNames was not empty, has expected length
            if not np.array_equal(metadata['RowNames'],[0,0]) and \
                (len(metadata['RowNames']) == df.shape[0]):
                df.index = metadata['RowNames']

    return df


def _array_to_tuple_vector(array):
    """
    Convert ndarray vector of shape (n_rows,n_cols) to (n_rows,) vector of
    n_cols length tuples, which can more easily be assembled into a DataFrame

    Parameters
    ----------
    array : ndarray, shape=(n_rows,n_cols)

    Returns
    -------
    tuples : ndarray, shape=(n_rows,) of tuples, shape=(n_cols,)
    """
    n = array.shape[0]
    # Create vector of size (n,) of same type as <array>
    tuples = np.empty((n,), dtype=object)
    for row in range(n):
        tuples[row] = tuple(array[row,:])

    return tuples


def _xarray_to_array(array):
    """
    Convert xarray DataArray -> Numpy array + dict with metadata attributes

    Parameters
    ----------
    array : xarray DataArray, shape=Any

    Returns
    -------
    array : ndarray, shape=Any
        Data values from DataArray `array`. Same shape and dtype as `array`.

    attrs : dict, {str : *}
        Metatdata attributes extracted from DataArray:

        - 'dims' :  tuple, shape=(n_dims,), of str. Dimension names.
        - 'coords' : tuple, shape=(n_dims,), of [len(dim),]. Coordinate indexes for each dimension.
        - 'name' :  str. Name of variable ('' if none set).
        - 'attrs':  OrderedDict. Any additional metadata ([] if none set).
    """
    # Extract metadata from variable into dict
    attrs = {'dims':   array.dims,
             'coords': tuple([array.coords[dim].values for dim in array.dims]),
             'name':   array.name if (array.name is not None) else '',
             'attrs':  array.attrs if bool(array.attrs) else []    # Tests if dict is empty
            }

    # Extract data from variable as numpy ndarray
    array = array.values

    return array,attrs


def _parse_typemap(typemap_in=None):
    """
    Combine any input values for typemap (which maps loaded Matlab variable
    types to Python variable types) with defaults

    Parameters
    ----------
    typemap_in : dict, {str:str}
        Maps names of Matlab variable types or specific Matlab variable names
        to Python variable types.
        Matlab types: 'array' = numerical array, 'cell' = cell array, 'struct' = structure).
        Python type: 'array' = Numpy ndarray, 'dataframe' = Pandas DataFrame, 'dict' = dictionary.

    Returns
    -------
    typemap_out : dict, {str:str}, default: {'array':'array', 'cell':'array', 'struct':'dict'}
        Default mapping, overwritten with any input values.
    """
    # Set of Matlab data types to deal with here
    vbl_types = {'array', 'cell', 'struct', 'structarray'}
    # Set default values
    typemap_out = {'array':'array', 'cell':'array',
                   'struct':'dict', 'structarray':'dict'}

    # Modify defaults with any input values
    if isinstance(typemap_in,dict):
        for key in typemap_in.keys():
            # If key is in list of Matlab data types, standardize to lowercase
            # (otherwise keep as is, to match specific variable names)
            if key.lower() in vbl_types:  key = key.lower()
            typemap_out[key] = typemap_in[key].lower()

        assert typemap_out['struct'] in ['dict','dataframe'], \
            "'struct' must be mapped to 'dict' or 'dataframe' (set to %s)" \
            % typemap_out['struct']

    else:
        assert typemap_in is None, TypeError("If <typemap> is input, it must be given as a dict")

    return typemap_out


def _parse_extract_items(extract_items_in=None):
    """
    Combine any input values for extract_items (which determines whether single items in
    loaded Matlab variables are extracted from containing arrays or left as length-1 arrays)
    with defaults

    Parameters
    ----------
    extract_items_in : bool or dict, {str:bool}
        Determines whether scalar-valued Matlab variables are loaded/returned as length-1
        arrays (False) or are extracted from the containing arrays as their specific dtype (True).
        Given as a single bool value to be used for *all* loaded variables or as a
        dict that maps names of Matlab variable types or specific Matlab variable names to bools.
        Matlab types: 'array' = numerical array, 'cell' = cell array, 'struct' = structure).

    Returns
    -------
    extract_items_out : dict, {str:bool}, default: {'array':True, 'cell':False, 'struct':True}
        Default mapping, overwritten with any input values.
    """
    # Set of Matlab data types to deal with here
    vbl_types = {'array', 'cell', 'struct', 'structarray'}
    # Set default values
    extract_items_out = {'array':True, 'cell':False, 'struct':True, 'structarray':False}

    # Modify defaults with any input values
    # If input given as single bool value, set that value for all variable types
    if _isbinary(extract_items_in):
        extract_items_out = {vbl_type : extract_items_in for vbl_type in extract_items_out}

    # If input given as dict, merge with default dict
    elif isinstance(extract_items_in,dict):
        for key in extract_items_in.keys():
            assert _isbinary(extract_items_in[key]), \
                ValueError("Values for <extract_items> must be boolean (0/1)")

            # If key is in list of Matlab data types, standardize to lowercase
            # (otherwise keep as is, to match specific variable names)
            if key.lower() in vbl_types:  key = key.lower()
            extract_items_out[key] = bool(extract_items_in[key])

    else:
        assert extract_items_in is None, \
            TypeError("If <extract_items> is input, it must be given as a bool or str:bool dict")

    return extract_items_out


def _variables_to_mat(variables):
    """
    Does any necessary conversions to get all variables into matfile-compatible
    variable types format

    variables = _variables_to_mat(variables)

    Parameters
    ----------
    variables : dict, {str : <vbl>}
        Names and values of variables to convert

    Returns
    -------
    variables : dict, {str : <vbl>}
       Same as input, but with variables converted to mat compatible types:

        - xarray.DataArray -> Numpy ndarray + {string:*} dict with metadata
        attributes (stored in separate variable named <variable>_attr)

        - strings, lists of strings -> Numpy object ndarray
    """
    new_vbl_dict = {}     # In case we need to create new variables in loop below

    def _size_general(x):
        if isinstance(x,np.ndarray):        return x.size
        elif isinstance(x, (list,tuple)):   return len(x)
        else:                               return 1

    for variable,value in variables.items():
        # Call function recursively with dictionary variables
        if isinstance(value,dict):
            variables[variable] = _variables_to_mat(value)

        elif isinstance(value,list):
            # Convert lists with any strings or with unequal-length entries -> object arrays
            # (same-length numerical lists are auto-converted to numerical arrays)
            # todo should we recurse here in case of nested lists?
            is_str = np.any([isinstance(item,str) for item in value])
            sizes = [_size_general(item) for item in value]
            is_unequal_size = np.any(np.diff(sizes) != 0)
            if is_str or is_unequal_size:
                variables[variable] = np.asarray(value,dtype=object)

        # Convert strings to Numpy object arrays
        elif isinstance(value,str):
            variables[variable] = np.asarray([value],dtype=object)

        # Convert Pandas DataFrame to dict and call this func recursively on items (cols)
        elif isinstance(value,pd.DataFrame):
            variables[variable] = _variables_to_mat(value.to_dict(orient='list'))

        # Conversion from xarray -> numpy array + dict (skip if xarray not installed)
        # Extract metadata attributes as new dict variable called <variable>_attr
        elif _HAS_XARRAY and isinstance(value,xr.DataArray):
            variables[variable],new_vbl_dict[variable+'_attr'] = \
                _xarray_to_array(value)


    # Merge any newly-created variables into existing variable dict
    return {**variables, **new_vbl_dict}


# =============================================================================
# v7-specific helper functions
# =============================================================================
# NOTE: This function can't be nested in _load7 b/c _dict_to_dataframe() needs access
def _process_v7_object(obj, matlab_vbl_type=None, python_vbl_type=None, extract_item=None,
                       typemap=None, extract_items=None, transpose=False, level=0, vbl=None):
    """ Properly handles arbitrary objects loaded from v7 mat files in recursive fashion """
    level += 1

    # Matlab class objects are returned as weird objects that are not parsable. Return as None.
    if matlab_vbl_type == 'class':
        converted = None

    # Matlab scalar structs are returned as (1,1) Numpy structured arrays
    #  Convert to dict or Pandas DataFrame (or leave as structured array)
    elif matlab_vbl_type == 'struct':
        if DEBUG: print('\t'*level, "struct")
        if python_vbl_type is None: python_vbl_type = typemap['struct']
        if extract_item is None: extract_item = extract_items['struct']

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj[0,0], typemap=typemap, transpose=transpose,
                                                 extract_item=extract_item,
                                                 extract_items=extract_items, level=level)

        elif python_vbl_type.lower() == 'dataframe':
            converted = _structuredarray_to_dataframe(obj[0,0], typemap=typemap, transpose=transpose,
                                                      extract_item=extract_item,
                                                      extract_items=extract_items, level=level)

        else:
            converted = obj[0,0]
            assert python_vbl_type.lower() != 'structuredarray', \
                ValueError("%s is an unsupported output type for Matlab structs. \n \
                            Must be 'dict'|'dataFrame'|'structuredArray'" % python_vbl_type)

    # Matlab struct arrays are returned as (m,n) Numpy structured arrays
    #  Convert to dict (or leave as structured array)
    elif matlab_vbl_type == 'structarray':
        if DEBUG: print('\t'*level, "structarray")
        if python_vbl_type is None: python_vbl_type = typemap['structarray']
        if extract_item is None: extract_item = extract_items['structarray']

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj, typemap=typemap, transpose=transpose,
                                                 extract_item=extract_item,
                                                 extract_items=extract_items, level=level)

        else:
            converted = obj
            assert python_vbl_type.lower() != 'structuredarray', \
                ValueError("%s is an unsupported output type for Matlab struct arrays. \n \
                            Must be 'dict'|'structuredArray'" % python_vbl_type)

    # Cell arrays are returned as object arrays. Squeeze out any singleton axes and
    # Permute axes of each cell's contents if 'PYTHON' order requested
    elif matlab_vbl_type == 'cell':
        if DEBUG: print('\t'*level, "cell")
        if extract_item is None: extract_item = extract_items['cell']

        flatiter = obj.flat             # 1D iterator to iterate over arbitrary-shape array
        converted = np.ndarray(shape=obj.shape,dtype=object) # Create empty object array
        for _ in range(obj.size):
            coords = flatiter.coords    # Multidim coordinates into array
            # Infer Matlab variable type of cell array element
            matlab_elem_type = _v7_matlab_type(obj[coords])
            elem_c = _process_v7_object(obj[coords], typemap=typemap,
                                        matlab_vbl_type=matlab_elem_type,
                                        extract_item=extract_item, extract_items=extract_items,
                                        transpose=transpose, level=level)
            if not extract_item: elem_c = np.atleast_1d(elem_c)
            converted[coords] = elem_c
            next(flatiter)              # Iterate to next element

    # General numerical/logical and cell arrays
    elif isinstance(obj,np.ndarray):
        if DEBUG: print('\t'*level, "array")
        # Logicals (binary-valued variables): Convert to bool type
        if matlab_vbl_type == 'logical':
            if DEBUG: print('\t'*level, "logical")
            converted = obj.astype(bool)

        # Strings: Extract string (dealing with empty strings appropriately)
        elif obj.dtype.type == np.str_:
            if DEBUG: print('\t'*level, "string")
            # Empty strings
            if len(obj) == 0:   converted = str()
            # General strings: Convert to string type (from np.str_)
            else:               converted = str(obj[0])
            if not extract_item: converted = np.asarray(converted, dtype=object)

        # General numerical array: Copy as-is
        else:
            if DEBUG: print('\t'*level, "numerical")
            converted = obj

    # Note: Only do the following for objects that remain output as ndarrays
    if isinstance(converted,np.ndarray):
        if extract_item is None: extract_item = extract_items['array']

        # Length-1 arrays: Extract single item as type <dtype> (if <extract_item> is True)
        if ((converted.shape == ()) or (converted.size == 1)) and extract_item:
            converted = converted.item()

        else:
            # Squeeze out any singleton axes, eg: Reshape (1,n) ndarrays -> (n,) vectors
            converted = converted.squeeze()

            # Permute array axes if 'PYTHON'/'C'/'ROWMAJOR' order requested
            if transpose: converted = converted.T

    return converted


def _v7_matlab_type(obj):
    """ Infer Matlab variable type of objects loaded from v7 mat files """
    # Matlab class objects are returned as this weird object (which is unparsable)
    if isinstance(obj,scipy.io.matlab.mio5_params.MatlabOpaque):    return 'class'
    # Matlab scalar structs are returned as (1,1) Numpy structured arrays
    elif _is_structured_array(obj) and (obj.size == 1):             return 'struct'
    # Matlab struct arrays are returned as (m,n) Numpy structured arrays
    elif _is_structured_array(obj) and (obj.size > 1):              return 'structarray'
    elif isinstance(obj,np.ndarray):
        # Cell arrays are returned as object arrays
        if obj.dtype == object:                                         return 'cell'
        # Heuristically assume arrays with only 0/1 (but not all 0's) are logicals
        elif _isbinary(obj) and len(obj) > 1 and not (obj == 0).all():  return 'logical'
        # General numeric array
        else:                                                           return 'array'
    else:
        raise TypeError("Undetermined type of variable:", obj)


# =============================================================================
# v7.3-specific helper functions
# =============================================================================
def _h5py_matlab_type(obj):
    """ Return variable type of Matlab variable encoded in h5py object """
    assert 'MATLAB_class' in obj.attrs, \
        AttributeError("Can't determine Matlab variable type. " \
                       "No 'MATLAB_class' attribute in h5py object '%s'" % obj)

    # Extract attribute with Matlab variable type, convert bytes -> string
    return obj.attrs['MATLAB_class'].decode('UTF-8')


def _convert_string(value, encoding='UTF-16'):
    """ Convert integer-encoded strings in HDF5 files to strings """
    return ''.join(value[:].tobytes().decode(encoding))    # time test ~ 700 ms

    # Note: Alternative methods that tested much slower:
    # return ''.join([chr(c) for c in value])               # time test ~ 7.2 s
    # return ''.join(map(chr,value))                        # time test ~ 7.3 s

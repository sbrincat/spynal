#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions for loading from and saving to Matlab MAT files

Function list
-------------
- loadmat : Loads variables from any Matlab MAT file. Also aliased as 'load'.
- savemat : Saves given variables into a MAT file. Also aliased as 'save'.
- whomat :  Lists all variables in any MAT file. Also aliased as 'who'.

Dependencies
------------
- h5py :    Python interface to the HDF5 binary data format (used for mat v7.3 files)

Function reference
------------------
"""
# Created on Mon Mar 12 17:20:26 2018
#
# @author: sbrincat
import sys
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd

try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False
    # print("xarray module is not installed. No support for loading/saving xarray format")

from spynal.helpers import _enclose_in_object_array
from spynal.matIO.matIO_7 import _load7, _who7, _save7, _process_v7_object, _v7_matlab_type
from spynal.matIO.matIO_73 import _load73, _who73

# Set=True to print out detailed debugging statements in loading functions
DEBUG = False


# =============================================================================
# Matfile loading/introspection functions
# =============================================================================
def loadmat(filename, variables=None, typemap=None, asdict=False, order='Matlab', verbose=True):
    """
    Load variables from a given MAT file and return them in appropriate Python types

    Handles both older (v4-v7) and newer (v7.3) versions of MAT files,
    transparently to the user.

    Variables returned individually or in a dict, where each variable maps to key/value pair

    Returned variable types are logical Python equivalents of Matlab types:
    ======              ======
    MATLAB              PYTHON
    ======              ======
    double/single       float
    int                 int
    char,string         str
    logical             bool
    array               Numpy ndarray of appropriate dtype
    cell array          Numpy ndarray of object dtype
    struct              dict or Pandas Dataframe (for table-like structs; depends on typemap)
    ======              ======

    Single-element Matlab arrays are converted to the contained item type (eg float/int/str)

    NOTE: Some proprietary or custom Matlab variables cannot be loaded, including:
    table/timetable, datetime, categorical, function_handle, map container, any custom object class

    Parameters
    ----------
    filename : str
        Full-path name of MAT file to load from

    variables : list of str, default: <all variables in file>
        Names of all variables to load

    typemap : dict {str:str}, default: {'array':'array', 'cell':'array', 'struct':'dict'}
        Maps names of Matlab variables or variable types to returned Python variable types.
        Currently alternative options only supported for table-like structs, which can
        return either as 'dict' or 'dataframe' (Pandas DataFrame).

    asdict : bool, default: False
        If True, returns variables in a {'variable_name':value} dict.
        If False, returns variables separately (as tuple).

    order : str, default: 'Matlab'
        Dimension order of returned arrays. Determines how values are arranged when reshaped.
        Options:
        - 'Matlab'/'F'  : Use Matlab/Fortran dimensional ordering (column-major-compatible)
        - 'Python'/'C'  : Use Python/C dimensional ordering (row-major compatible)

    verbose : bool, default: True
        If True, prints names and shapes of all loaded variables to stdout

    Returns
    -------
    data_dict : dict {str:<variable>}
        Dictionary holding all loaded variables, mapping variable name to its value

    -or-

    vbl1,vbl2,... :
        Variables returned individually, as in 2nd example above

    Examples
    --------
    data_dict = loadmat(filename, variable, asdict=True)

    variable1, variable2, ... = loadmat(filename, variable, asdict=False)
    """
    # If variables input as string, convert to list
    if isinstance(variables,str):  variables = [variables]

    # Combine any input values for typemap with defaults
    typemap = _parse_typemap(typemap)

    assert order.upper() in ['MATLAB','F','COL','COLMAJOR','PYTHON','C','ROW','ROWMAJOR'], \
        "<order> must be 'Matlab' or 'Python' (%s given)" % order

    version = _get_matfile_version(filename)

    # Use h5py to load v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:  data = _load73(filename,variables,typemap,order)

    # Use scipy.io.loadmat() to load v7 and older MAT-files
    else:               data = _load7(filename,variables,typemap,order)

    if variables is None: variables = list(data.keys())

    if verbose:
        for vbl in variables:
            if isinstance(data[vbl], np.ndarray):       # Numpy array variables
                vblstr = vbl + ' : numpy.array(' + \
                        ''.join('%3d,' % x for x in data[vbl].shape) + ')' + \
                        (' of type %s' % data[vbl].dtype)
            elif isinstance(data[vbl], pd.DataFrame):   # Pandas DataFrame variables
                vblstr = vbl + ' : pandas.DataFrame with columns[' + \
                        ''.join('%s,' % x for x in data[vbl].keys()) + ']'
            elif isinstance(data[vbl], dict):           # dict variables (Matlab structs)
                vblstr = vbl + ' : dict with keys[' + \
                        ''.join('%s,' % x for x in data[vbl].keys()) + ']'
            else:                                       # Scalar variables (float/int/string)
                vblstr = vbl
            vblstr = vblstr + '\n'
            print(vblstr)

    if asdict:                  return data
    elif len(variables) == 1:   return data[variables[0]]
    else:                       return tuple(data[vbl] for vbl in variables)


load = loadmat
""" Alias of :func:`loadmat`. See there for details. """


def whomat(filename, verbose=True):
    """
    Return list of variables in a given MAT file and/or print them to stdout

    Parameters
    ----------
    filename : str
        Full-path name of MAT-file to examine

    verbose : bool, default: True
        If True, prints names of all file variables to stdout

    Returns
    -------
    variables : list of str
        Names of variables in file
    """
    version = _get_matfile_version(filename)

    # Use h5py to load v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:  variables = _who73(filename)

    # Use scipy.io.loadmat() to load v7 and older MAT-files
    else:               variables = _who7(filename)

    if verbose: print(variables)

    return variables


who = whomat
""" Alias of :func:`whomat`. See there for details. """


# =============================================================================
# Matfile saving functions
# =============================================================================
def savemat(filename, variables, version=None, **kwargs):
    """
    Save data variables to a Matlab MAT file

    NOTE: Currently can only save older (v7), not newer (v7.3), versions of MAT files

    Parameters
    ----------
    filename : str
        Full-path name of MAT file to save to

    variables : dict {str:<variable>}
        Names and values of variables to save

    version : float, default: (7.3 if any variable is > 2 GB; 7 otherwise)
        Version of MAT file: 7 (older) | 7.3 (newer/HDF5).

    **kwargs
        All other keyword args passed to scipy.io.savemat()
    """
    assert (version is None) or (version in [7,7.3]), ValueError("version must be 7 or 7.3")

    # Do any necessary conversions to get all variables into matfile-compatible format
    # Note: Use deepcopy to create copy of all variables to avoid changing in caller
    variables = _variables_to_mat(deepcopy(variables))

    # If version is not set or set=7, check to make sure no variables are > 2 GB
    if version != 7.3:
        # Compute max size in memory of any variable (in GB)
        sizes = [sys.getsizeof(variables[vbl]) for vbl in variables.keys()]
        max_size = np.max(np.asarray(sizes))/(1024.0**3)
        # If any veriable is > 2GB, must use v7.3, otherwise default to v7
        if max_size >= 2:
            if version == 7: print('WARNING: Variable > 2 GB, switching to MAT file v7.3')
            version = 7.3
        else:
            version = 7

    # Use hdf5storage to save v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:
        raise ValueError('Saving MAT file version 7.3 not coded up yet')

    # Use scipy.io.savemat() to save v7 MAT-files
    else:
        _save7(filename, variables, **kwargs)

save = savemat
""" Alias of :func:`savemat`. See there for details. """


# =============================================================================
# Helper functions
# =============================================================================
def _structuredarray_to_dict(sarray, typemap=None, transpose=False, level=0):
    """
    Converts Numpy structured array (which Matlab structs are initially
    imported as, using scipy.io.loadmat) to a dict, with type names as keys

    INPUT
    sarray   Numpy structured array

    OUTPUT
    dic      Dict. {keys = sarray.dtype.names : values = sarray[key]}
    """
    # Convert structured array to dict
    dic = {name:sarray[name] for name in sarray.dtype.names}

    for name in dic.keys():
        if DEBUG: print('\t'*level, "'%s'" % name)
        # Infer Matlab variable type of struct field
        matlab_vbl_type = _v7_matlab_type(dic[name])
        # If specific field name is listed in <typemap>, use given mapping
        python_vbl_type = typemap[name] if name in typemap else None

        dic[name] = _process_v7_object(dic[name], matlab_vbl_type=matlab_vbl_type,
                                       python_vbl_type=python_vbl_type,
                                       typemap=typemap, transpose=transpose, level=level)

    return dic


def _structuredarray_to_dataframe(sarray, typemap=None, transpose=False, level=0):
    """
    Convert Numpy structured array (which Matlab structs are initially imported
    as, using scipy.io.loadmat) to a Pandas DataFrame, with type names as keys.
    Intended for table-like Matlab scalar structs, which naturally fit into a
    DataFrame (though Matlab table variables are not currently loadable in Python).

    df = _structuredarray_to_dataframe(sarray)

    INPUT
    sarray  Numpy structured array

    OUTPUT
    df      Pandas DataFrame. Columns are names and values of all the sarray.dtype.names
            whose associated values are vector-valued with length = modal length
            of all names. Rows are each entry from these vectors

    REFERENCE Based on code obtained from:
    http://poquitopicante.blogspot.com/2014/05/loading-matlab-mat-file-into-pandas.html
    """
    # Convert structured array to dict {keys = sarray.dtype.names : values = sarray[key]}
    dic = _structuredarray_to_dict(sarray, typemap=typemap, transpose=transpose, level=level)

    # Convert dict to a Pandas DataFrame
    df = _dict_to_dataframe(dic)

    return df


def _get_matfile_version(filename):
    """
    Determines version of given MAT file (7.3 or 7/older) by reading its header

    INPUT
    filename    String. Full-path name of MAT-file to load from

    OUTPUT
    version     Float. Version of MAT file: 7 (older) | 7.3 (newer/HDF5)
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
    Deals with weird way Matlab cell-array-of-strings data structures are loaded.

    Converts them from Numpy array of objects, which are each trivial
    1-length arrays of strings, to just an array of strings themselves.
    """
    return np.asarray([(x[j][0] if len(x[j]) > 0 else '') for j in range(len(x))])


def _is_structured_array(array):
    """ Returns True if input array is a Numpy structured array, False otherwise """
    # For v7.3 mat files loaded with h5py, structured arrays are ndarrays with dtype np.void
    # For v7 mat files loaded with scipy.io, structured arrays are an "np.void" type
    # Test for either version
    return (isinstance(array,np.ndarray) and (array.dtype.type is np.void)) \
         or isinstance(array,np.void)


def _dict_to_dataframe(dic):
    """
    Converts dict (which Matlab structs are initially imported as, using h5py)
    to a Pandas DataFrame, with type names as keys. Intended for table-like
    Matlab scalar structs, which naturally fit into a DataFrame
    (though Matlab tables themselves are not currently loadable in Python AKAIK).

    df = _dict_to_dataframe(dic)

    INPUT
    dic {string:(nRow,) array-like} dict. Dictionary with keys to be used as
        column names and values that are same-length vectors. Values not
        fitting this criterion are not retained in output.

    OUTPUT
    df  Pandas DataFrame. Columns are dict keys, for all associated values are
        vector-valued with length = modal length of all keys.
        Rows are each entry from these vectors.

    REFERENCE Based on code obtained from:
    http://poquitopicante.blogspot.com/2014/05/loading-matlab-mat-file-into-pandas.html
    """
    # Special case: Matlab structs converted from Matlab tables may retain the
    # 'Properties' metadata attribute as a struct field. If so, we extract and
    #  remove it, convert it to dict, and below tack it onto df as an attribute.
    if 'Properties' in dic:
        metadata = dic.pop('Properties')

        # If metadata is a structured array, convert it to dict
        if _is_structured_array(metadata):
            metadata = _structuredarray_to_dict(metadata,
                                                typemap={'struct':'dict'}, transpose=False)
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
    Converts ndarray vector of shape (n_rows,n_cols) to (n_rows,) vector of
    n_cols length tuples, which can more easily be assembled into a DataFrame

    tuples = _array_to_tuple_vector(array)

    INPUT
    array   (n_rows,n_cols) Numpy ndarray

    OUTPUT
    tuples  (n_rows,) Numpy vector of nCol-tuples
    """
    n = array.shape[0]
    # Create vector of size (n,) of same type as <array>
    tuples = np.empty((n,), dtype=object)
    for row in range(n):
        tuples[row] = tuple(array[row,:])

    return tuples


def _xarray_to_array(array):
    """
    Converts xarray DataArray -> Numpy array + dict with metadata attributes

    array,attrs = _xarray_to_array(array,attrDict)

    INPUT
    array       xarray DataArray (any shape and dtype)

    OUTPUT
    array       Numpy ndarray (same shape and dtype). Data values from DataArray.
    attrs       {String:*} dict. Metatdata attributes extracted from DataArray:
                'dims' :    (n_dims,) tuple of strings. Dimension names.
                'coords' :  (n_dims,) tuple of [len(dim),] array.
                            Coordinate indexes for each dimension
                'name' :    String. Name of variable ('' if none set)
                'attrs':    OrderedDict. Any additional metadata ([] if none set)
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
    Combines any input values for typemap (which maps loaded Matlab variable
    types to Python variable types) with defaults

    INPUT
    typemap_in  {string:string} Dict. Maps names of Matlab variable types or
                specific Matlab variable names to Python variable types.
                Matlab types: 'array' = numerical array, 'cell' = cell array,
                    'struct' = structure).
                Python type: 'array' = Numpy ndarray, 'dataframe' = Pandas
                    DataFrame, 'dict' = dictionary

    OUTPUT
    typemap_out {string:string} Dict. Default mapping, overwritten with any
                input values.
                Default: {'array':'array', 'cell':'array', 'struct':'dict'}

    """
    # Set of Matlab data types to deal with here
    vbl_types = {'array', 'cell', 'struct', 'structarray'}
    # Set default values
    typemap_out = {'array':'array', 'cell':'array',
                   'struct':'dict', 'structarray':'dict'}

    # Modify defaults with any input values
    if typemap_in is not None:
        for key in typemap_in.keys():
            # If key is in list of Matlab data types, standardize to lowercase
            # (otherwise keep as is, to match specific variable names)
            if key.lower() in vbl_types:  key = key.lower()
            typemap_out[key] = typemap_in[key].lower()

        assert typemap_out['struct'] in ['dict','dataframe'], \
            "'struct' must be mapped to 'dict' or 'dataframe' (set to %s)" \
            % typemap_out['struct']

    return typemap_out


def _variables_to_mat(variables):
    """
    Does any necessary conversions to get all variables into matfile-compatible
    variable types format

    variables = _variables_to_mat(variables)

    INPUT
    variables   {String:<vbl>} dict. Names and values of variables to convert

    OUTPUT
    variables   Same, but with variables converted to mat compatible types:
                xarray.DataArray -> Numpy ndarray + {string:*} dict with metadata
                attributes (stored in separate variable named <variable>_attr)
                strings, lists of strings -> Numpy object ndarray
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



# Setup so module can be run from command line using:  python matIO.py <arguments>
if __name__ == "__main__":
    loadmat(sys.argv[1])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matIO   Functions for loading from and saving to Matlab MAT files

FUNCTIONS
loadmat Loads variables from any Matlab MAT file. Also aliased as 'load'.

whomat  Lists all variables in any MAT file. Also aliased as 'who'.

savemat Saves given variables into a MAT file. Also aliased as 'save'.
        NOTE: Currently only version 7 MAT files are supported, and thus you
        cannot save variables > 2GB.

variables_to_mat Converts given Python variables into MATfile-compatible
        variable types

DEPENDENCIES
h5py    Python interface to the HDF5 binary data format (used for mat v7.3 files)


Created on Mon Mar 12 17:20:26 2018

@author: sbrincat
"""
import time
import sys
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
import pandas as pd

import scipy.io
import h5py

try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False
    # print("xarray module is not installed. No support for loading/saving xarray format")

from neural_analysis.helpers import _enclose_in_object_array

# Uncomment to print out detailed debugging statements in loading functions
DEBUG = False


# =============================================================================
# Matfile loading functions
# =============================================================================
def loadmat(filename, variables=None, typemap=None, asdict=False, extract_elem=True,
            order='Matlab', version=None, verbose=True):
    """
    Loads variables from a given MAT file (of any version), and returns them
    either individually or in a dict, where each variable maps to a key/value pair,
    and value types are logical Python equivalents of Matlab types:
    Matlab      ->      Python
    ------              ------
    double/single       float
    int                 int
    char,string         str
    logical             bool (returns as int for v7, complain to scipy)
    array               Numpy ndarray of appropriate dtype
    cell array          Numpy ndarray of object dtype
    struct              dict or Pandas Dataframe (for table-like structs; depends on typemap)

    NOTE: Some proprietary or custom Matlab variables CANNOT be loaded, including:
    table/timetable, datetime, categorical, function_handle, map container, any custom object class

    Handles both older (v4-v7) and newer (v7.3) versions of MAT files,
    transparently to the user.

    loadmat(filename,variables=None,typemap=None,asdict=False,extract_elem=True,
            order='Matlab',version=None,verbose=True)

    USAGE
    data_dict = loadmat(filename,variable,asdict=True)
    variable1,variable2,... = loadmat(filename,variable,asdict=False)

    INPUT
    filename    String. Full-path name of MAT-file to load from

    variables   List of strings. Names of variables to load. Default: all file variables

    typemap     {string:string} Dict. Maps names of Matlab variable types or
                specific Matlab variable names to Python variable types.
                Matlab types: 'array' = numerical array, 'cell' = cell array,
                    'struct' = structure).
                Python types: 'array' = Numpy ndarray, 'dataframe' = Pandas
                    DataFrame, 'dict' = dictionary
                Default: {'array':'array', 'cell':'array', 'struct':'dict'}

    asdict      Bool. If True, returns variables in a {'variable_name':value} dict.
                If False [default], returns variables separately in tuple.

    extract_elem Bool. If False, 1-item ndarrays are returned as such.
                If True [default], the element value is extracted and returned as
                the type implied by its array dtype (eg float/int/string).

    order       String. Dimension order of loaded/returned arrays (Default: Matlab):
                  'Matlab'/'F'  : Use Matlab dimensional ordering (column-major-compatible;
                                   default behavior for scipy.io.loadmat)
                  'Python'/'C'  : Use Python dimensional ordering (row-major compatible;
                                   default behavior for h5py)

    version     Float. Version of MAT file: 7 (older) | 7.3 (newer/HDF5).
                Default: finds it from mat file header

    verbose     Logical. Set to print names and shapes of all loaded variables to stdout
                Default: True

    OUTPUT
    data_dict   {String:<variable>} dict. Dictionary holding all loaded variables,
                mapping variable name to its value.
    -or-
    vbl1,vbl2,... Tuple of variables.

    For both output types, Matlab arrays are generally returned as Numpy arrays;
    Matlab structs are returned as (sub)dicts or Pandas DataFrames (with fieldnames as keys)
    """
    # If variables input as string, convert to list
    if isinstance(variables,str):  variables = [variables]

    # Combine any input values for typemap with defaults
    typemap = _parse_typemap(typemap)

    assert order.upper() in ['MATLAB','F','COL','COLMAJOR','PYTHON','C','ROW','ROWMAJOR'], \
        "<order> must be 'Matlab' or 'Python' (%s given)" % order

    if version is None: version = _get_matfile_version(filename)

    # Use h5py to load v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:  data = _load73(filename,variables,typemap,order)

    # Use scipy.io.loadmat() to load v7 and older MAT-files
    else:               data = _load7(filename,variables,typemap,order)

    if variables is None: variables = list(data.keys())

    # Extract values from 1-item arrays as their given dtype
    if extract_elem:
        for vbl in variables:
            if isinstance(data[vbl], np.ndarray) and (data[vbl].size == 1):
                data[vbl] = data[vbl].item()

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
""" Also alias loadmat() as load() """


def whomat(filename, version=None, verbose=True):
    """
    Returns list of variables in a given MAT file (of any version)
    and/or prints them to stdout

    variables = whomat(filename,version=None,verbose=True)

    Handles both older (v4-v7) and newer (v7.3) versions of MAT files,
    transparently to the user.

    INPUT
    filename    String. Full-path name of MAT-file to examine

    version     Float. Version of MAT file: 7 (older) | 7.3 (newer/HDF5).
                Default: finds it from mat file header

    verbose     Logical. Set to print names of all file variables to stdout
                Default: True

    OUTPUT
    variables   List of strings. Names of variables in file
    """
    if version is None: version = _get_matfile_version(filename)

    # Use h5py to load v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:  variables = _who73(filename)

    # Use scipy.io.loadmat() to load v7 and older MAT-files
    else:               variables = _who7(filename)

    if verbose: print(variables)

    return variables

who = whomat
""" Also alias whomat() as who() """

# =============================================================================
# Matfile saving functions
# =============================================================================
def savemat(filename, variables, version=None, **kwargs):
    """
    Saves data variables to a Matlab MAT file

    savemat(filename,variables,version=None,do_compression=False)

    Currently handles only older (v7), not newer (v7.3), versions of MAT files

    INPUT
    filename    String. Full-path name of MAT-file to save to

    variables   {String:<vbl>} dict. Names and values of variables to save.

    version     Float. Version of MAT file: 7 (older) | 7.3 (newer/HDF5).
                Default: 7.3 if any variable is > 2 GB; 7 otherwise

    **kwargs    All other keyword args passed to scipy.io.savema()

    ACTION
    Saves given variables to a MAT file of given (or internally set) version
    """
    # Do any necessary conversions to get all variables into matfile-compatible format
    # Note: Use deepcopy to create copy of all variables to avoid changing in caller
    variables = variables_to_mat(deepcopy(variables))

    # If version is not set or set=7, check to make sure no variables are > 2 GB
    if version != 7.3:
        # Compute max size in memory of any variable (in GB)
        sizes = [sys.getsizeof(variables[vbl]) for vbl in variables.keys()]
        max_size = np.max(np.asarray(sizes))/(1024.0**3)
        # If any veriable is > 2GB, must use v7.3, otherwise default to v7
        if max_size >= 2:
            if version == 7.3: print('WARNING: Variable > 2 GB, switching to MAT file v7.3')
            version = 7.3
        else:
            version = 7

    # Use hdf5storage to save v7.3 MAT-files (which are a type of hdf5 file)
    if version == 7.3:
        raise ValueError('Saving MAT file version 7.3 not coded up yet')

    # Use scipy.io.savemat() to save v7 MAT-files
    else:
        scipy.io.savemat(filename,variables,**kwargs)


save = savemat
""" Also alias savemat() as save() """


def variables_to_mat(variables):
    """
    Does any necessary conversions to get all variables into matfile-compatible
    variable types format

    variables = variables_to_mat(variables)

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
            variables[variable] = variables_to_mat(value)

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
            variables[variable] = variables_to_mat(value.to_dict(orient='list'))

        # Conversion from xarray -> numpy array + dict (skip if xarray not installed)
        # Extract metadata attributes as new dict variable called <variable>_attr
        elif _HAS_XARRAY and isinstance(value,xr.DataArray):
            variables[variable],new_vbl_dict[variable+'_attr'] = \
                _xarray_to_array(value)


    # Merge any newly-created variables into existing variable dict
    return {**variables, **new_vbl_dict}


# =============================================================================
# Functions for v7.3 (HDF5) mat files (using h5py)
# =============================================================================
def _load73(filename, variables=None, typemap=None, order='C'):
    """
    Loads data variables from a version 7.3 Matlab MAT file

    Uses h5py to load data, as v7.3 MAT files are a type of HDF5 file
    """
    typemap = _parse_typemap(typemap)

    # h5py returns arrays in row-major (Python/C) order by default
    # Transpose/permute array axes if Matlab/Fortran/column-major requested
    transpose = order.upper() in ['MATLAB','F','COL','COLMAJOR']

    # Open datafile for reading
    # For newer versions, can specify to maintain original object attribute order
    #  (eg original order of struct fields)
    if h5py.__version__ >= '2.9.0':
        file = h5py.File(filename,'r',track_order=True)
    else:
        file = h5py.File(filename,'r')

    # If <variables> not set, load all variables from datafile (keys for File object)
    # Note: Get rid of header info ('#refs#' variable)
    if variables is None:
        variables = [vbl for vbl in file.keys() if vbl[0] != '#']

    # If <variables> list was input, ensure file actually contains all requested variables
    else:
        for vbl in variables:
            assert vbl in file.keys(), \
                ValueError("Variable '%s' not present in file %s" % (vbl,filename))


    def _process_h5py_object(obj, file, matlab_vbl_type=None, python_vbl_type=None, level=0):
        """ Properly handles arbitrary objects loaded from HDF5 files in recursive fashion """
        level += 1
        # For h5py Dataset (contains Matlab array), extract and process array data
        if isinstance(obj, h5py.Dataset):
            if DEBUG: print('\t'*level, "Dataset", matlab_vbl_type)
            if matlab_vbl_type is None: matlab_vbl_type = _h5py_matlab_type(obj)
            return _process_h5py_object(obj[()], file, matlab_vbl_type=matlab_vbl_type, level=level)

        # For h5py Group (Matlab struct), recurse thru fields and return as dict or DataFrame
        elif isinstance(obj, h5py.Group):
            if DEBUG: print('\t'*level, "Group", matlab_vbl_type)
            converted = {}
            for key in obj.keys():
                if DEBUG: print('\t'*level, "'%s'" % key)
                matlab_elem_type = _h5py_matlab_type(obj[key])
                converted[key] = _process_h5py_object(obj[key], file,
                                                      matlab_vbl_type=matlab_elem_type, level=level)

            # If no specific output type requested for variable, default to type for structs
            if python_vbl_type is None: python_vbl_type = typemap['struct']
            # Convert entire (former Matlab struct) variable to a Pandas DataFrame
            if python_vbl_type.lower() == 'dataframe':
                converted = _dict_to_dataframe(converted)

            return converted

        # For a HDF5 Reference, get the name of the referenced object and read directly from file
        # stackoverflow.com/questions/28541847/how-convert-this-type-of-data-hdf5-object-reference-to-something-more-readable
        elif isinstance(obj, h5py.h5r.Reference):
            if DEBUG: print('\t'*level, "Reference", matlab_vbl_type)
            # For int-encoded string, convert to string
            if _h5py_matlab_type(file[obj]) == 'char':  return _convert_string(file[obj][()])
            else:                                       return file[obj][()]

        # For an ndarray (Matlab array)
        elif isinstance(obj,np.ndarray):
            if DEBUG: print('\t'*level, "ndarray", matlab_vbl_type)

            # BUGFIX For some reason, h5py seems to output empty cells in cell arrays as
            #        (2,) ndarrays == [n,0] (where n is n_rows of other cells).
            #        Fix that by properly setting = []
            if ((obj.shape == (2,)) and (obj[1] == [0]).all()):
                if DEBUG: print('\t'*level, "empty")
                # Empty strings
                if matlab_vbl_type == 'char':
                    converted = str()
                # General empty arrays
                else:
                    converted = np.ndarray(shape=(0,1),dtype='uint64')

            # For length-1 ndarray, extract and process its single array element
            # (but don't do this for Matlab char arrays, which are treated as strings)
            elif (obj.size == 1) and not (matlab_vbl_type == 'char'):
                if DEBUG: print('\t'*level, "size 1")
                converted = _process_h5py_object(obj.item(), file,
                                                 matlab_vbl_type=matlab_vbl_type, level=level)

            # Matlab char arrays (strings) -- convert to Python string
            elif matlab_vbl_type == 'char':
                if DEBUG: print('\t'*level, "char")
                converted = _convert_string(obj)

            # Matlab logical arrays -- convert to Numpy ndarray of dtype bool
            elif matlab_vbl_type == 'logical':
                if DEBUG: print('\t'*level, "logical")
                converted = obj.astype(bool)

            # Matlab cell arrays -- convert to Numpy ndarray of dtype object
            # Iterate thru and process each array element individually
            elif matlab_vbl_type == 'cell':
                if DEBUG: print('\t'*level, "cell array")
                assert obj.ndim <= 2, "Cell arrays with > 2 dimensions not (yet) supported"

                converted = np.ndarray(shape=obj.shape,dtype=object) # Create empty object array
                for row,elem in enumerate(obj):
                    for col in range(len(elem)):
                        # print(row, col, len(elem))
                        matlab_elem_type = _h5py_matlab_type(file[elem[col]])

                        converted[row,col] = \
                            _process_h5py_object(file[elem[col]][()], file,
                                                 matlab_vbl_type=matlab_elem_type, level=level)

            # Matlab numerical arrays -- straight copy to Numpy ndarray of appropriate dtype
            else:
                if DEBUG: print('\t'*level, "numerical")
                converted = obj

            # Note: Only do the following for variables output as arrays (ie not strings/scalars)
            if isinstance(converted,np.ndarray):
                # Squeeze out any singleton axes, eg: Reshape (1,n) ndarrays -> (n,) vectors
                converted = converted.squeeze()

                # Permute array axes if 'MATLAB'/column-major order requested
                if transpose: converted = converted.T

            return converted

        # Scalar values
        else:
            # Convert logical scalar -> boolean
            if matlab_vbl_type == 'logical':
                if DEBUG: print('\t'*level, "bool scalar")
                return bool(obj)

            # Convert chars -> string
            # Note: we shouldn't get here, but if we do, have to re-package obj in list
            elif matlab_vbl_type == 'char':
                if DEBUG: print('\t'*level, "char scalar")
                return _convert_string([obj])

            # Everything else (numerical types) -- return as-is
            else:
                if DEBUG: print('\t'*level, "numerical scalar")
                return obj


    # Load each requested variable and save as appropriate Python variable (based on typemap)
    # into a {variable_name:variable_value} dict
    data = {}
    for vbl in variables:
        # Extract Matlab variable type from h5py object attributes
        matlab_vbl_type = _h5py_matlab_type(file[vbl])
        if DEBUG: print("'%s'" % vbl, matlab_vbl_type)

        # If specific variable name is listed in <typemap>, use given mapping
        python_vbl_type = typemap[vbl] if vbl in typemap else None

        # Process h5py object -- extract data, convert to appropriate Python type,
        # traversing down into object (cell elements/struct fields) with recursive calls as needed
        data[vbl] = _process_h5py_object(file[vbl], file, matlab_vbl_type=matlab_vbl_type,
                                         python_vbl_type=python_vbl_type, level=0)

    file.close()

    return data


def _who73(filename):
    """ Lists data variables from a version 7.3 Matlab MAT file """
    # Open datafile for reading
    if h5py.__version__ >= '2.9.0':
        file = h5py.File(filename,'r',track_order=True)
    else:
        file = h5py.File(filename,'r')

    # Find all variables in file, save into list
    # Note: Get rid of header info ('#refs#' variable)
    variables = [vbl for vbl in file.keys() if vbl[0] != '#']
    file.close()
    return variables


def _h5py_matlab_type(obj):
    """ Returns variable type of Matlab variable encoded in h5py object """
    assert 'MATLAB_class' in obj.attrs, \
        AttributeError("Can't determine Matlab variable type. " \
                       "No 'MATLAB_class' attribute in h5py object '%s'" % obj)

    # Extract attribute with Matlab variable type, convert bytes -> string
    return obj.attrs['MATLAB_class'].decode('UTF-8')


def _convert_string(value, encoding='UTF-16'):
    """ Converts integer-encoded strings in HDF5 files to strings """
    return ''.join(value[:].tostring().decode(encoding))    # time test ~ 700 ms

    # Note: Alternative methods that tested much slower:
    # return ''.join([chr(c) for c in value])               # time test ~ 7.2 s
    # return ''.join(map(chr,value))                        # time test ~ 7.3 s


# =============================================================================
# Functions for v7 (and earlier) mat files (using scipy.io)
# =============================================================================
def _load7(filename, variables=None, typemap=None, order='Matlab'):
    """
    Loads data variables from a version 7 (or older) Matlab MAT file
    Uses scipy.io.loadmat to load data
    """
    typemap = _parse_typemap(typemap)

    # scipy.io returns arrays in column-major (Matlab/Fortran) order by default
    # Transpose/permute array axes if Python/C/row-major requested
    transpose = order.upper() in ['PYTHON','C','ROW','ROWMAJOR']

    # If <variables> is set, convert to {variable_name:None} dict
    # (otherwise leave = None, so it loads all variables from datafile)
    if variables is not None:
        variables = {vbl:None for vbl in variables}

    # Load each requested variable and save each as Numpy array in a {variable_name:array} dict
    data = scipy.io.loadmat(filename,variable_names=variables)

    # Get rid of header info returned by loadmat() (variables starting with '_...')
    data = {vbl:value for vbl,value in data.items() if vbl[0] != '_'}

    # Ensure we actually loaded all requested variables
    if variables is not None:
        for vbl in variables:
            assert vbl in data.keys(), \
                ValueError("Variable '%s' not present in file %s" % (vbl,filename))

    # If <variables> not set, load all variables in datafile
    else:
        variables = list(data.keys())

    # Get data into desired output format
    for vbl in variables:
        # Infer Matlab variable type from object structure
        matlab_vbl_type = _v7_matlab_type(data[vbl])
        # If specific variable name is listed in <typemap>, use given mapping
        python_vbl_type = typemap[vbl] if vbl in typemap else None

        if DEBUG: print("'%s'" % vbl, matlab_vbl_type, python_vbl_type)

        # Process loaded object -- extract data, convert to appropriate Python type,
        # traversing down into object (cell elements/struct fields) with recursive calls as needed
        data[vbl] = _process_v7_object(data[vbl], matlab_vbl_type=matlab_vbl_type,
                                       python_vbl_type=python_vbl_type, typemap=typemap,
                                       transpose=transpose, level=0)

    return data


def _who7(filename, **kwargs):
    """  Lists data variables from a version 7 (or older) Matlab MAT file """
    # Load list of 3-tuples of (variable,size,type) for each variable in file
    variables = scipy.io.whosmat(filename, appendmat=True, **kwargs)
    # Extract and return just the variable names
    return [vbl[0] for vbl in variables]


# NOTE: This function can't be nested in _load7 b/c _dict_to_dataframe() needs access
def _process_v7_object(obj, matlab_vbl_type=None, python_vbl_type=None,
                       typemap=None, transpose=False, level=0):
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

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj[0,0], typemap=typemap,
                                                 transpose=transpose, level=level)

        elif python_vbl_type.lower() == 'dataframe':
            converted = _structuredarray_to_dataframe(obj[0,0], typemap=typemap,
                                                      transpose=transpose, level=level)

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

        if python_vbl_type.lower() == 'dict':
            converted = _structuredarray_to_dict(obj, typemap=typemap,
                                                 transpose=transpose, level=level)

        else:
            converted = obj
            assert python_vbl_type.lower() != 'structuredarray', \
                ValueError("%s is an unsupported output type for Matlab struct arrays. \n \
                            Must be 'dict'|'structuredArray'" % python_vbl_type)

    # Cell arrays are returned as object arrays. Squeeze out any singleton axes and
    # Permute axes of each cell's contents if 'PYTHON' order requested
    elif matlab_vbl_type == 'cellarray':
        if DEBUG: print('\t'*level, "cellarray")
        flatiter = obj.flat             # 1D iterator to iterate over arbitrary-shape array
        converted = np.ndarray(shape=obj.shape,dtype=object) # Create empty object array
        for _ in range(obj.size):
            coords = flatiter.coords    # Multidim coordinates into array
            # Infer Matlab variable type of cell array element
            matlab_elem_type = _v7_matlab_type(obj[coords])            
            converted[coords] = _process_v7_object(obj[coords], typemap=typemap,
                                                   matlab_vbl_type=matlab_elem_type,
                                                   transpose=transpose, level=level)
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

        # General numerical array: Copy as-is
        else:
            if DEBUG: print('\t'*level, "numerical")
            converted = obj

    # Note: Only do the following for objects that remain output as ndarrays
    if isinstance(converted,np.ndarray):
        # Length-1 arrays: Extract single item as type <dtype>
        if (converted.shape == ()) or (converted.size == 1):
            converted = converted.item()

        else:
            # Squeeze out any singleton axes, eg: Reshape (1,n) ndarrays -> (n,) vectors
            converted = converted.squeeze()

            # Permute array axes if 'PYTHON'/'C'/'ROWMAJOR' order requested
            if transpose: converted = converted.T

    return converted


def _structuredarray_to_dict(sarray, typemap=None, transpose=False, level=0):
    """
    Converts Numpy structured array (which Matlab structs are initially
    imported as, using scipy.io.loadmat) to a dict, with type names as keys

    dic = _structuredarray_to_dict(sarray)

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
        if obj.dtype == object:                                     return 'cellarray'
        # Heuristically assume arrays with only 0/1 (but not all 0's) are logicals
        elif _isbinary(obj) and not (obj == 0).all():               return 'logical'
        # General numeric array
        else:                                                       return 'array'
    else:
        raise TypeError("Undetermined type of variable:", obj)


def _isbinary(x):
    """
    Tests whether variable contains only binary values (True,False,0,1)
    """
    x = np.asarray(x)
    return (x.dtype == bool) or \
           (np.issubdtype(x.dtype,np.number) and \
            np.all(np.in1d(x,[0,0.0,1,1.0,True,False])))


# =============================================================================
# Other helper functions
# =============================================================================
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


# Setup so module can be run from command line using:  python matIO.py <arguments>
if __name__ == "__main__":
    loadmat(sys.argv[1])

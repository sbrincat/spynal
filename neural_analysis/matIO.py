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
# TODO  Empty strings return garbage (eg '\x00\x00...') (maybe from mixed string/other cell arrays). Fix?

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

try:
    from .helpers import _enclose_in_object_array
# TEMP    
except ImportError:
    from helpers import _enclose_in_object_array

    
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

    asdict      Bool. If True, returns variables in a {'variableName':value} dict.
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
# Loading helper functions
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


def _load7(filename, variables=None, typemap=None, order='Matlab'):
    """
    Loads data variables from a version 7 (or older) Matlab MAT file
    Uses scipy.io.loadmat to load data
    """
    typemap = _parse_typemap(typemap)

    # scipy.io returns arrays in column-major (Matlab/Fortran) order by default
    # Transpose/permute array dim's if Python/C/row-major requested
    transpose = order.upper() in ['PYTHON','C','ROW','ROWMAJOR']

    # If <variables> is set, convert to {variableName:None} dict
    # (otherwise leave = None, so it loads all variables from datafile)
    if variables is not None:
        variables = {vbl:None for vbl in variables}

    # Load each requested variable and save as np array in a {variableName:array} dict
    data = scipy.io.loadmat(filename,variable_names=variables)

    # Ensure we actually loaded all requested variables
    if variables is not None:
        for vbl in variables:
            if vbl not in data.keys():
                raise ValueError('Variable "%s" not present in file %s' %
                                 (vbl,filename))

    # Get rid of header info returned by loadmat() (variables starting with '_...')
    data = {vbl:value for vbl,value in data.items() if vbl[0] != '_'}

    # Get data into desired output format
    for vbl in data.keys():
        # If specific variable name is listed in <typemap>, use given mapping
        if vbl in typemap:  typeout = typemap[vbl]
        else:               typeout = None

        # Matlab scalar structs are returned as (1,1) Numpy structured arrays.
        #  Convert to dict or Pandas DataFrame.
        # Note: structured arrays are indicated by dtype type "np.void"
        if (data[vbl].dtype.type is np.void) and (data[vbl].size == 1):
            if typeout is None: typeout = typemap['struct']
            if typeout == 'dict':
                data[vbl] = _structuredarray_to_dict(data[vbl][0,0])
            elif typeout.lower() == 'dataframe':
                data[vbl] = _structuredarray_to_dataframe(data[vbl][0,0])
            elif typeout.lower() != 'structuredarray':
                raise ValueError("%s is an unsupported output type for Matlab structs. \n \
                                 Must be 'dict'|'dataFrame'|'structuredArray'" % typeout)

        # Matlab struct arrays are returned as (m,n) Numpy structured arrays.
        #  Convert to dict.
        elif (data[vbl].dtype.type is np.void) and (data[vbl].size > 1):
            if typeout is None: typeout = typemap['structarray']
            if typeout == 'dict':
                data[vbl] = _structuredarray_to_dict(data[vbl])
            elif typeout.lower() != 'structuredarray':
                raise ValueError("%s is an unsupported output type for Matlab struct arrays. \n \
                                 Must be 'dict'|'structuredArray'" % typeout)

        # Cell arrays are returned as object arrays. Squeeze out any singleton dims and
        # Permute dim's of each cell's contents if 'PYTHON' order requested
        elif data[vbl].dtype == object:
            flatiter = data[vbl].flat       # 1D iterator to iterate over arbitrary-shape array
            for _ in range(data[vbl].size):
                coords = flatiter.coords    # Multidim coordinates into array
                if isinstance(data[vbl][coords],np.ndarray):
                    data[vbl][coords] = data[vbl][coords].squeeze()
                    if transpose: data[vbl][coords] = data[vbl][coords].T
                next(flatiter)              # Iterate to next element

        # DELETE? No way to know whether variable should be bool or not
        # # Convert binary-valued variables to bool type
        # elif isinstance(data[vbl],np.ndarray) and _isbinary(data[vbl]) and (data[vbl] != bool):
        #     data[vbl] = data[vbl].astype(bool)

        # Permute array axes if 'PYTHON' order requested
        if transpose and isinstance(data[vbl],np.ndarray):
            data[vbl] = data[vbl].T

    return data


def _load73(filename, variables=None, typemap=None, order='C'):
    """
    Loads data variables from a version 7.3 Matlab MAT file
    Uses h5py to load data, as v7.3 MAT files are a type of HDF5 file
    """
    typemap = _parse_typemap(typemap)

    # h5py returns arrays in row-major (Python/C) order by default
    # Transpose/permute array dim's if Matlab/Fortran/column-major requested
    transpose = order.upper() in ['MATLAB','F','COL','COLMAJOR']

    # Open datafile for reading
    # For newer versions, can specify to maintain original object attribute  order
    #  (eg original order of struct fields)
    if h5py.__version__ >= '2.9.0':
        file = h5py.File(filename,'r',track_order=True)
    else:
        file = h5py.File(filename,'r')

    # If <variables> not set, load all variables from datafile (keys for File object)
    # Note: Get rid of header info ('#refs#' variable)
    if variables is None:
        variables = [vbl for vbl in file.keys() if vbl[0] != '#']

    # Load each requested variable and save as np array in a {variableName:array} dict
    # Access datafile variables using dict-like syntax
    data = {}
    for vbl in variables:
        # Ensure we actually loaded all requested variables
        assert vbl in file.keys(), \
            ValueError("Variable '%s' not present in file %s" % (vbl,filename))

        # If specific variable name is listed in <typemap>, use given mapping
        typeout = typemap[vbl] if vbl in typemap else None

        # Matlab arrays and cell arrays are loaded as h5py Datasets
        if isinstance(file[vbl], h5py.Dataset):
            # Matlab arrays -- Just extract their value and
            #  save into np.ndarray w/in data dict
            if file[vbl].dtype != object:
                data[vbl] = file[vbl][()]

                # Convert integer-encoded strings to strings
                if ('MATLAB_int_decode' in file[vbl].attrs) and \
                   (file[vbl].attrs['MATLAB_int_decode'] > 1):
                    data[vbl] = _convert_string(data[vbl])
                    
                # Convert binary-valued variables to bool type
                elif ('MATLAB_class' in file[vbl].attrs) and \
                     (file[vbl].attrs['MATLAB_class'] == b'logical') and (data[vbl].dtype != bool):
                    data[vbl] = data[vbl].astype(bool)

            # Cell arrays -- Need to work a bit to extract values
            else:
                # Create empty object array
                data[vbl] = np.ndarray(shape=file[vbl].shape,dtype=object)

                for col,hdf5obj in enumerate(file[vbl]):
                    for row in range(len(hdf5obj)):
                        # BUGFIX For some reason, h5py seems to output empty cells as
                        #       (2,) ndarrays == [n,0] (where n is nRows of other cells).
                        #        Fix that by properly setting = []
                        if ((file[hdf5obj[row]][()].shape == (2,)) and
                            (file[hdf5obj[row]][()][1] == [0]).all()):
                            data[vbl][col,row] = np.ndarray(shape=(0,1),dtype='uint64')
                           
                        # Convert integer-encoded strings to strings                            
                        elif ('MATLAB_int_decode' in file[hdf5obj[row]].attrs) and \
                             (file[hdf5obj[row]].attrs['MATLAB_int_decode'] > 1):
                            data[vbl][col,row] = _convert_string(file[hdf5obj[row]][()])
                                 
                        # Permute array axes if 'MATLAB'/column-major order requested
                        elif transpose:
                            data[vbl][col,row] = file[hdf5obj[row]][()].T
                            
                        else:
                            data[vbl][col,row] = file[hdf5obj[row]][()]
                    
            # Permute array axes if 'MATLAB'/column-major order requested
            if transpose and isinstance(data[vbl],np.ndarray):
                data[vbl] = data[vbl].T


        # Matlab structs are loaded as h5py Groups. Iterate thru their Datasets, convert to dict.
        elif isinstance(file[vbl], h5py.Group):
            if typeout is None: typeout = typemap['struct']

            data[vbl] = {}
            # Iterate thru each Dataset in Group = each field in Struct
            for key in file[vbl].keys():
                # Nested struct
                if isinstance(file[vbl][key], h5py.Group):
                    data[vbl][key] = _process_h5py_object(file[vbl][key],file)

                # Generic object (non-numeric) dtypes (Matlab struct fields)
                elif file[vbl][key].dtype == object:
                    n_elems = len(file[vbl][key])

                    # Determine if objects are strings or something else
                    # This is indicated by this 'MATLAB_int_decode' attribute in each h5py ref
                    isstring = False
                    for elem in range(n_elems):
                        for ref in file[vbl][key][elem]:
                            if 'MATLAB_int_decode' in file[ref].attrs:
                                isstring = True
                                break
                        if isstring: break

                    # Load each value, convert as needed, concatenate into a list
                    # DEL tmp_list = []
                    # String (Matlab char arrays) are loaded in h5py as ascii.
                    # Convert ascii coded strings -> actual string, append to list
                    if isstring:
                        tmp_list = []
                        for elem in range(n_elems):                            
                            for ref in file[vbl][key][elem]:
                                tmp_list.append(_convert_string(file[ref]))
                                
                        data[vbl][key] = np.asarray(tmp_list,dtype=object).reshape(file[vbl][key].shape)
                        
                    # Otherwise (other element types): Just append values to list
                    else:
                        data[vbl][key] = np.empty((file[vbl][key].size,), dtype=object)                        
                        for elem in range(n_elems):
                            tmp_list = []
                            for ref in file[vbl][key][elem]:
                                tmp_list.append(_process_h5py_object(file[ref],file))
                            data[vbl][key][elem] = np.asarray(tmp_list,dtype=object).squeeze() # tmp_list # DEL np.asarray(tmp_list)

                    # Reshape to original object array shape
                    data[vbl][key] = data[vbl][key].reshape(file[vbl][key].shape)
                    # Convert list -> Numpy array (of objects) and reshape to original shape
                    # DEL data[vbl][key] = np.asarray(tmp_list,dtype=object).reshape(file[vbl][key].shape)                                            

                # 1-item arrays -- extract item as array dtype
                elif file[vbl][key].size == 1:
                    data[vbl][key] = file[vbl][key][()].item()
                    # Convert binary-valued variables to bool type
                    if ('MATLAB_class' in file[vbl][key].attrs) and \
                       (file[vbl][key].attrs['MATLAB_class'] == b'logical'):
                        data[vbl][key] = bool(data[vbl][key])
                    
                # Numeric dtypes -- just load as is
                else:
                    data[vbl][key] = file[vbl][key][()]


                # Convert binary-valued variables to bool type
                if ('MATLAB_class' in file[vbl][key].attrs) and \
                    (file[vbl][key].attrs['MATLAB_class'] == b'logical') and \
                    isinstance(data[vbl][key],np.ndarray) and (data[vbl][key].dtype != bool):
                    data[vbl][key] = data[vbl][key].astype(bool)

                # Convert strings that are loaded as uint16 ascii vectors to strings
                # (this is apparently indicated by this 'MATLAB_int_decode' attribute)
                elif ('MATLAB_int_decode' in file[vbl][key].attrs) and \
                     (file[vbl][key].attrs['MATLAB_int_decode'] > 1):
                    # HACK Convert ascii from original HDF5 dataset bc scalar extract above messes it up
                    data[vbl][key] = _convert_string(file[vbl][key][()])
                    # DEL  _convert_string(file[vbl][key])

                if isinstance(data[vbl][key],np.ndarray):
                    # Single-items arrays: extract single value as type <dtype>
                    if data[vbl][key].size == 1:
                        data[vbl][key] = data[vbl][key].item()

                    # 2d arrays: remove singleton dim and/or transpose if requested
                    elif data[vbl][key].ndim > 1:
                        # Transpose (k>1,n) -> (n,k>1) if requested,
                        # or if putting into a DataFrame
                        if (transpose or typeout == 'dataframe'):
                           # DELETE: and (file[vbl][key].dtype != object):
                            data[vbl][key] = data[vbl][key].T

                        # Squeeze out any singleton axes
                        # eg: Reshape (1,n) ndarrays -> (n,) vectors
                        data[vbl][key] = data[vbl][key].squeeze()

            # Convert entire (former Matlab struct) variable to a Pandas DataFrame
            if typeout == 'dataframe':
                data[vbl] = _dict_to_dataframe(data[vbl])

    file.close()

    return data


def _convert_string(value, encoding='UTF-16'):
    """ Converts integer-encoded strings in HDF5 files to strings """
    # TODO can we always assume UTF-16 encoding?  How to know this?
    return ''.join(value[:].tostring().decode(encoding))    # time test ~ 700 ms
    # return ''.join([chr(c) for c in value])               # time test ~ 7.2 s
    # return ''.join(map(chr,value))                        # time test ~ 7.3 s


def _process_h5py_object(value,file):
    """ Properly handles arbitrary objects loaded from HDF5 files in recursive fashion """
    # For h5py Dataset (contains Matlab array), extract and process data
    if isinstance(value, h5py.Dataset):
        return _process_h5py_object(value[()],file)

    # For h5py Group (Matlab struct), recurse thru fields and return as dict
    elif isinstance(value, h5py.Group):
        return {key : _process_h5py_object(value[key],file) for key in value.keys()}

    # For a HDF5 Reference, get the name of the referenced object and read directly from file
    # stackoverflow.com/questions/28541847/how-convert-this-type-of-data-hdf5-object-reference-to-something-more-readable
    elif isinstance(value, h5py.h5r.Reference):
        isstring = ('MATLAB_int_decode' in file[value].attrs) and (file[value].attrs['MATLAB_int_decode'] > 1)
        # For int-encoded string, convert to string
        if isstring:    return _convert_string(file[value][()])
        else:           return file[value][()]

        # Note: This is *several orders of magnitude* slower
        # name = h5py.h5r.get_name(value,file.id)
        # if 'MATLAB_int_decode' in file[name].attrs: return _convert_string(file[name][()])
        # else:                                       return file[name][()]

    # For an ndarray (Matlab array)
    elif isinstance(value,np.ndarray):
        # For length-1 ndarray, extract and process its single array element
        if value.size == 1:
            return _process_h5py_object(value.item(),file)

        # For object ndarray, iterate thru and process each array element individually
        elif value.dtype == object:
            obj = [_process_h5py_object(elem,file) for elem in value]
            try:
                return np.asarray(obj,dtype=object).reshape(value.shape).squeeze()
            # HACK When each element of input has same length j, asarray creates an
            # (i,j) array instead of (i,) array of (j,) objects (see lfpSchema.userData.removedFreqs).
            # TEMP Workaround til I can find better way to deal with this
            except:
                out = np.empty(value.squeeze().shape,dtype=object)
                for j in range(value.size): out[j] = obj[j]

        # For general numerical ndarray, return array with any singleton axes removed
        else:
            return value.squeeze()

    # Otherwise (scalar value), we assume object is OK to return as is
    else:
        return value


def _who7(filename, **kwargs):
    """  Lists data variables from a version 7 (or older) Matlab MAT file """
    # Load list of 3-tuples of (variable,size,type) for each variable in file
    variables = scipy.io.whosmat(filename, appendmat=True, **kwargs)
    # Extract and return just the variable names
    return [vbl[0] for vbl in variables]


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


# =============================================================================
# Other helper functions
# =============================================================================
def _isbinary(x):
    """
    Tests whether variable contains only binary values (True,False,0,1)
    """
    x = np.asarray(x)
    return (x.dtype == bool) or \
           (np.issubdtype(x.dtype,np.number) and \
            np.all(np.in1d(x,[0,0.0,1,1.0,True,False])))


def _structuredarray_to_dict(sarray):
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
        # For ndarrays, first remove redundant axes
        if isinstance(dic[name],np.ndarray):
            # 2d arrays: Squeeze out any singleton axes
            if dic[name].ndim > 1:
                dic[name] = dic[name].squeeze()

            # Also squeeze out extra singleton axes in object arrays
            if isinstance(dic[name].dtype.type,object):
                flatiter = dic[name].flat       # 1D iterator to iterate arbitrary-shape array
                for _ in range(dic[name].size):
                    coords = flatiter.coords    # Multidim coordinates into array
                    if isinstance(dic[name][coords],np.ndarray):
                        dic[name][coords] = dic[name][coords].squeeze()
                    next(flatiter)              # Iterate to next element

        # Convert dict entries that are themselves structured arrays
        # (ie Matlab sub-structs) also to dict via recursive call of this function
        # Note: structured arrays in v7.3 return as ndarrays with dtype "np.void",
        # but in v7 return as "np.void" type
        if (isinstance(dic[name],np.ndarray) and (dic[name].dtype.type is np.void)) or \
            isinstance(dic[name],np.void):
            dic[name] = _structuredarray_to_dict(dic[name])

        # If variable is any other ndarray, convert special case data types
        elif isinstance(dic[name],np.ndarray):
            # For single-item arrays: extract single value as type <dtype>
            if dic[name].size == 1: dic[name] = dic[name].item()
            
            # Heuristically convert binary-valued variables to bool type
            elif (dic[name].dtype != bool) and _isbinary(dic[name]) and not (dic[name] == 0).all():
                dic[name] = dic[name].astype(bool)

            # Cell-string array fields import as Numpy arrays of objects, where
            # each object is a trivial (1-length) array containing a string.
            # Just extract the strings, converting to array of string types
            # TODO Need to deal with possibility of mixed-type cell arrays (some string, some other)
            elif isinstance(dic[name].dtype.type,object) and (dic[name].size != 0) \
               and (dic[name][0].dtype.type == np.str_):
                # print(name, dic[name].shape, dic[name].size, dic[name].dtype)
                # print([(dic[name][j].shape, dic[name][j].size, dic[name][j].dtype) for j in range(dic[name].size)])
                # print(dic[name])
                # dic[name] = _extract_strings(dic[name])   

                # Empty strings in cell-string arrays are imported as empty array,
                # so need this extra step to test for that
                for elem in range(len(dic[name])):
                    # Extract item from single-element arrays
                    if dic[name][elem].shape == ():
                        dic[name][elem] = dic[name][elem].item()
                    # Deal with empty strings
                    elif len(dic[name][elem]) == 0:
                        dic[name][elem] = ''
                    # General strings
                    # TODO Feel like we need a recursion here?
                    else:
                        dic[name][elem] = dic[name][elem][0]

                # Empty strings in cell-string arrays are imported as empty array,
                # so need this extra step to test for that
                # isstring = False
                # for elem in range(len(dic[name])):
                #     # If elem is itself a string, convert it
                #     if dic[name][elem].shape == () and isinstance(dic[name][elem].item(),str):
                #         dic[name][elem] = dic[name][elem].item()
                #     # This skips empty arrays, only tests for string type on non-emptys
                #     elif (len(dic[name][elem]) != 0) and isinstance(dic[name][elem][0],str):
                #         isstring = True
                #         break
                # if isstring: dic[name] = _extract_strings(dic[name])

    return dic


def _extract_strings(x):
    """
    Deals with weird way Matlab cell-array-of-strings data structures are loaded.

    Converts them from Numpy array of objects, which are each trivial
    1-length arrays of strings, to just an array of strings themselves.
    """
    return np.asarray([(x[j][0] if len(x[j]) > 0 else '') for j in range(len(x))])


def _structuredarray_to_dataframe(sarray):
    """
    Converts Numpy structured array (which Matlab structs are initially imported
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
    dic = _structuredarray_to_dict(sarray)

    # Convert dict to a Pandas DataFrame
    df = _dict_to_dataframe(dic)

    return df


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

        # If metadata is a structured array (which in v7.3 looks like an ndarray ,
        # with dtype np.void,  but in v7 looks like a "np.void" type), convert it to dict
        if (isinstance(metadata,np.ndarray) and (metadata.dtype.type is np.void)) or \
            isinstance(metadata,np.void):
            metadata = _structuredarray_to_dict(metadata)
    else:
        metadata = None

    # Convert any scalar dict values -> ndarrays, and ensure ndarrays are 
    # at least 1D (0-dimensional array mess up DF conversion and are weird)
    for key in dic.keys():
        if not isinstance(dic[key],np.ndarray) or (dic[key].ndim == 0):
            dic[key] = np.atleast_1d(dic[key])
        # DEL if not isinstance(dic[key],np.ndarray): dic[key] = np.asarray(dic[key])

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
    Converts ndarray vector of shape (nRows,nCols) to (nRows,) vector of
    nCols length tuples, which can more easily be assembled into a DataFrame

    tuples = _array_to_tuple_vector(array)

    INPUT
    array   (nRows,nCols) Numpy ndarray

    OUTPUT
    tuples  (nRows,) Numpy vector of nCol-tuples
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
                'dims' :    (nDims,) tuple of strings. Dimension names.
                'coords' :  (nDims,) tuple of [len(dim),] array.
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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Functions for loading from and saving to Matlab v7.3 MAT files using h5py & hdf5storage """
import numpy as np

import h5py
import hdf5storage

from spynal.matIO.helpers import _dict_to_dataframe, _h5py_matlab_type, _convert_string, DEBUG


def _load73(filename, variables=None, typemap=None, extract_items=None, order='C'):
    """
    Load data variables from a version 7.3 Matlab MAT file

    Uses h5py to load data, as v7.3 MAT files are a type of HDF5 file
    """
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


    def _process_h5py_object(obj, file, matlab_vbl_type=None, python_vbl_type=None,
                             extract_item=None, extract_items=None, level=0):
        """ Properly handles arbitrary objects loaded from HDF5 files in recursive fashion """
        level += 1
        # For h5py Dataset (contains Matlab array), extract and process array data
        if isinstance(obj, h5py.Dataset):
            if DEBUG: print('\t'*level, "Dataset", matlab_vbl_type, extract_item)
            if matlab_vbl_type is None: matlab_vbl_type = _h5py_matlab_type(obj)
            return _process_h5py_object(obj[()], file, matlab_vbl_type=matlab_vbl_type,
                                        extract_item=extract_item, extract_items=extract_items,
                                        level=level)

        # For h5py Group (Matlab struct), recurse thru fields and return as dict or DataFrame
        elif isinstance(obj, h5py.Group):
            if DEBUG: print('\t'*level, "Group", matlab_vbl_type)
            converted = {}
            for key in obj.keys():
                if DEBUG: print('\t'*level, "'%s'" % key)
                matlab_elem_type = _h5py_matlab_type(obj[key])
                if key in extract_items: extract_item = extract_items[key]
                converted[key] = _process_h5py_object(obj[key], file,
                                                      matlab_vbl_type=matlab_elem_type,
                                                      extract_item=extract_item,
                                                      extract_items=extract_items, level=level)

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
            if DEBUG: print('\t'*level, "ndarray", matlab_vbl_type, extract_item)

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
            elif (obj.size <= 1) and extract_item and not (matlab_vbl_type == 'char'):
                if DEBUG: print('\t'*level, "size 1", extract_item)
                converted = _process_h5py_object(obj.item(), file,
                                                 matlab_vbl_type=matlab_vbl_type,
                                                 extract_item=extract_item, level=level)

            # Matlab char arrays (strings) -- convert to Python string
            elif matlab_vbl_type == 'char':
                if DEBUG: print('\t'*level, "char")
                converted = _convert_string(obj)
                if not extract_item: converted = np.array(converted, dtype=object)

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

                        elem_c = _process_h5py_object(file[elem[col]][()], file,
                                                      matlab_vbl_type=matlab_elem_type,
                                                      extract_item=extract_item,
                                                      extract_items=extract_items, level=level)
                        if not extract_item: elem_c = np.atleast_1d(elem_c)
                        converted[row,col] = elem_c    

            # Matlab numerical arrays -- straight copy to Numpy ndarray of appropriate dtype
            else:
                if DEBUG: print('\t'*level, "numerical", obj.size, extract_item)
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

        # If specific variable name is listed in <typemap>/<extract_items>, use given value
        python_vbl_type = typemap[vbl] if vbl in typemap else None

        # If specific variable name is listed in <extract_items>, use associated value
        if vbl in extract_items:                extract_item = extract_items[vbl]
        # Otherwise, if variable type is listed in <extract_items>, use associated value
        elif matlab_vbl_type in extract_items:  extract_item = extract_items[matlab_vbl_type]
        # Otherwise, just use value for 'array' as generic value for all other (eg scalar) types
        else:                                   extract_item = extract_items['array']
        if DEBUG: print("'%s'" % vbl, matlab_vbl_type, python_vbl_type, extract_item)

        # Process h5py object -- extract data, convert to appropriate Python type,
        # traversing down into object (cell elements/struct fields) with recursive calls as needed
        data[vbl] = _process_h5py_object(file[vbl], file, matlab_vbl_type=matlab_vbl_type,
                                         python_vbl_type=python_vbl_type, extract_item=extract_item,
                                         extract_items=extract_items, level=0)

    file.close()

    return data


def _who73(filename):
    """ List data variables from a version 7.3 Matlab MAT file """
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


def _save73(filename, variables, appendmat=True, oned_as='column', store_python_metadata=True):
    """
    Save data variables to version 7.3 Matlab MAT file (HDF5 file with specific header).
    Uses hdf5storage.savemat to save data
    """
    hdf5storage.savemat(filename, variables, format='7.3', appendmat=appendmat, oned_as=oned_as,
                        store_python_metadata=store_python_metadata)

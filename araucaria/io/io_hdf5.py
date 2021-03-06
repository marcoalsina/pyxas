#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.io.io_hdf5` submodule offers the following functions to read, write and 
manipulate data in the Hierarchical Data Format ``HDF5``:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Function
     - Description
   * - :func:`read_hdf5`
     - Reads a group dataset from an HDF5 file.
   * - :func:`write_hdf5`
     - Writes a group dataset into an HDF5 file.
   * - :func:`rename_dataset_hdf5`
     - Renames a group dataset in an HDF5 file.
   * - :func:`delete_dataset_hdf5`
     - Deletes a group dataset in an HDF5 file.
   * - :func:`summary_hdf5`
     - Returns a summary of datasets in an HDF5 file.
"""
from os.path import isfile
from ast import literal_eval
from pathlib import Path
from typing import Optional
from numpy import ndarray, inf
from h5py import File, Dataset
from .. import Group, Report
from ..xas import pre_edge

def read_hdf5(fpath: Path, name: str)-> Group:
    """Reads a group dataset from an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Dataset name to retrieve from the HDF5 file.

    Returns
    -------
    :
        Group containing the requested dataset.

    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` does not exist in the HDF5 file.
    
    Example
    -------
    >>> from araucaria import Group
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.utils import check_objattrs
    >>> from araucaria.io import read_hdf5
    >>> fpath = get_testpath('test_database.h5')
    >>> # extracting dnd_testfile
    >>> group_mu = read_hdf5(fpath, name='dnd_testfile')
    >>> check_objattrs(group_mu, Group, attrlist=['mu', 'mu_ref'])
    [True, True]
    """    
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "r")
    else:
        raise IOError("file %s does not exists." % fpath)

    # convereted types for reading datasets
    converted_types = (dict, list)
    
    if name in hdf5:
        data = {}
        for key, record in hdf5.get(name).items():
            if isinstance(record, Dataset):
                # verifying strings saved as bytes
                if isinstance(record[()], bytes):
                    # converting to string with asstr
                    try:
                        # evaluating the string with literal_eval
                        eval_record = literal_eval( record.asstr()[()] )
                    except:
                        # if conversion fails then keeping as str
                        eval_record = record.asstr()[()]
                    if isinstance(eval_record, converted_types):
                        data[key] = eval_record
                else:
                    data[key]=record[()]
            
            #print '%s dataset: %s <%s>' % (flag, key, vtype)
    else:
        raise ValueError("%s does not exists in %s!" % (name, fpath))
    
    hdf5.close()
    group = Group(**data)
    group.name = name
    return group

def write_hdf5(fpath: Path, group: Group, name: str='dataset1', 
               replace: bool=False) -> None:
    """Writes a group dataset into an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    group
        Group to write in the HDF5 file.
    name
        Name for the dataset in the HDF5 file. The default is 'dataset1'.
    replace
        Replace previous dataset. The default is False.

    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If dataset cannot be written to the HDF5 file.
    TypeError
        If ``group`` is not a valid Group instance.
    ValueError
        If ``name`` dataset already exists in the HDF5 file and ``replace=False``.

    Notes
    -----
    If the file specified by ``fpath`` does not exists, it will be automatically created.
    If the file already exists then the dataset will be appended.
    
    By default the write operation will be cancelled if ``name`` already exists in the HDF5 file.
    The previous dataset can be overwritten with the option ``replace=True``. 
    
    Example
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    """    
    # testing that the group exists 
    if type(group) is not Group:
        raise TypeError('%s is not a valid Group instance.' % group)
    
    # verifying existence of path:
    # (a)ppend to existing file
    # (w)rite to new file.
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        hdf5 = File(fpath, "w")
    
    # testing name of the dataset
    if name in hdf5:
        # dataset present in the file
            if replace:
                hdf5.__delitem__(name)
            else:
                hdf5.close()
                raise ValueError("%s already exists in %s." % (name, fpath))
    
    dataset = hdf5.create_group(name)
    try:
        write_recursive_hdf5(dataset, group)
        print("%s written to %s." % (name, fpath))
    except:
        hdf5.close()
        raise IOError("%s couldn't be written to %s." % (name, fpath))
    
    hdf5.close()
    return
    
def write_recursive_hdf5(dataset: Dataset, group: Group) -> None:
    """Utility function to write a Group recursively into an HDF5 file.
    
    Parameters
    ----------
    dataset
        Dataset in the HDF5 file.
    group
        Group to write in the HDF5 file.

    Returns
    -------
    :
    
    Warning
    -------
    Only :class:`str`, :class:`float`, :class:`int` and :class:`~numpy.ndarray` 
    types are currently supported for recursive writting into an HDF5 :class:`~h5py.Dataset`.
    
    :class:`dict` and :class:`list` types will be convertet to :class:`~numpy.str`, which is in
    turn saved as :class:`bytes` in the HDF5 database.
    If read with :func:`read_hdf5`, such records will be automatically converted to their
    original type in the group.
    
    """
    # accepted type variables for recursive writting
    accepted_types  = (str, float, int, ndarray)
    converted_types = (dict, list)
    
    for key in dir(group):
        if '__' not in key:
            record =getattr(group,key)
            vtype = type(record).__name__
        
            if isinstance(record, accepted_types):
                dataset.create_dataset(key, data=record)
            
            elif isinstance(record, converted_types):
                dataset.create_dataset(key, data=str(record))
    return

def rename_dataset_hdf5(fpath: Path, name: str, newname: str) -> None:
    """Renames a dataset in an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Name of Group dataset.
    newname
        New name for Group dataset.

    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` dataset does not exist in the HDF5 file.
    ValueError
        If ``newname`` dataset already exists in the HDF5 file.
    
    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5, rename_dataset_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    >>> # renaming dataset
    >>> rename_dataset_hdf5('database.h5', 'xmu_testfile', 'xmu_renamed')
    xmu_testfile renamed to xmu_renamed in database.h5.
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        raise IOError("file %s does not exists." % fpath)
    
    if newname in hdf5:
        hdf5.close()
        raise ValueError('%s already exists in %s' % (newname, fpath))
    
    # verifying existence of datagroup
    if name in hdf5:
        hdf5[newname] = hdf5[name]
    else:
        hdf5.close()
        raise ValueError("%s does not exists in %s." % (name, fpath))
       
    hdf5.__delitem__(name)
    hdf5.close()
    print ("%s renamed to %s in %s." % (name, newname, fpath))
    return
    
def delete_dataset_hdf5(fpath: Path, name: str) -> None:
    """Deletes a dataset from an HDF5 file.
    
    Parameters
    ----------
    fpath
        Path to HDF5 file.
    name
        Name of dataset to delete.
    
    Returns
    -------
    :
    
    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If ``name`` dataset does not exist in the HDF5 file.
    
    Example
    -------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import read_xmu, write_hdf5, rename_dataset_hdf5
    >>> fpath = get_testpath('xmu_testfile.xmu')
    >>> # extracting mu and mu_ref scans
    >>> group_mu = read_xmu(fpath, scan='mu')
    >>> # saving a new hdf5 file
    >>> write_hdf5('database.h5', group_mu, name='xmu_testfile', replace=True)
    xmu_testfile written to database.h5.
    >>> # deleting dataset
    >>> delete_dataset_hdf5('database.h5', 'xmu_testfile')
    xmu_testfile deleted from database.h5.
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "a")
    else:
        hdf5.close()
        raise IOError("File %s does not exists." % fpath)
        
    # verifying existence of datagroup
    if name in hdf5:
        hdf5.__delitem__(name)
        hdf5.close()
        print ("%s deleted from %s." % (name, fpath))
    else:
        hdf5.close()
        raise ValueError ("%s does not exists in %s." % (name, fpath))
    return

def summary_hdf5(fpath: Path, optional: Optional[list]=None, 
                 **pre_edge_kws:dict) -> Report:
    """Returns a summary report of datasets in an HDF5 file.

    Parameters
    ----------
    fpath
        Path to HDF5 file.
    optional
        List with optional parameters. See Notes for details.
    pre_edge_kws
        Dictionary with arguments for :func:`pre_edge`.

    Returns
    -------
    :
        Report for datasets in the HDF5 file.

    Raises
    ------
    IOError
        If the HDF5 file does not exist in the specified path.
    ValueError
        If any requested parameter in ``optional`` is not recognized.        

    Notes
    -----
    Summary data includes the following:
    
    1. Dataset index.
    2. Dataset name.
    3. Measurement mode.
    4. Numbers of scans.
    5. absorption edge step :math:`\Delta\mu(E_0)`, if ``optional=['edge_step']``.
    6. absorption threshold energy :math:`E_0`, if ``optional=['e0']``.
    7. Merged scans, if ``optional=['merged_scans']``.
    
    The number of scans and names of merged files are retrieved from the 
    ``merged_scans`` attribute of the HDF5 dataset.
    
    The absorption threshold and the edge step are retrieved by calling the function
    :func:`~araucaria.xas.normalize.pre_edge`.

    See also
    --------
    :func:`read_hdf5`
    :class:`~araucaria.main.repot.Report`
    
    Examples
    --------
    >>> from araucaria.testdata import get_testpath
    >>> from araucaria.io import summary_hdf5
    >>> fpath = get_testpath('test_database.h5')
    >>> # printing default summary
    >>> report = summary_hdf5(fpath)
    >>> report.show()
    ===========================
    id  dataset       mode  n  
    ===========================
    1   dnd_testfile  mu    3  
    2   p65_testfile  mu    2  
    3   xmu_testfile  mu    1  
    ===========================
    
    >>> # printing summary with merged scans
    >>> report = summary_hdf5(fpath, optional=['e0', 'merged_scans'])
    >>> report.show()
    ====================================================
    id  dataset       mode  n  e0     merged_scans      
    ====================================================
    1   dnd_testfile  mu    3  29203  dnd_test_001.dat  
                                      dnd_test_002.dat  
                                      dnd_test_003.dat  
    ----------------------------------------------------
    2   p65_testfile  mu    2  18011  p65_test_001.xdi  
                                      p65_test_002.xdi  
    ----------------------------------------------------
    3   xmu_testfile  mu    1  11873  None              
    ====================================================
    """
    # verifying existence of path:
    if isfile(fpath):
        hdf5 = File(fpath, "r")
    else:
        raise IOError("file %s does not exists." % fpath)

    # list with parameter names
    field_names = ['id', 'dataset', 'mode', 'n']
    opt_list    = ['merged_scans', 'edge_step', 'e0']

    if pre_edge_kws == {}:
        # default values
        pre_edge_kws={'pre_range':[-150,-50], 'nnorm':3, 'post_range':[150, inf]}

    # verifying optional values
    if optional is not None:
        for opt_val in optional:
            if opt_val not in opt_list:
                hdf5.close()
                raise ValueError("optional parameter '%s' not recognized." % opt_val)
            else:
                field_names.append(opt_val)

    # instanciating report class
    report   = Report()
    report.set_columns(field_names)

    for i, key in enumerate(hdf5.keys()):
        data    = read_hdf5(fpath, str(key))
        scanval = data.get_mode()
        extra_content = False  # aux variable for 'merged_scans'
        try:
            # merged_scans is saved as string, so we count the number of commas
            nscans = hdf5[key]['merged_scans'].asstr()[()].count(',') + 1
        except:
            nscans = 1

        field_vals = [i+1, key, scanval, nscans]
        if optional is not None:
            for j, opt_val in enumerate(optional):
                if opt_val == 'merged_scans':
                    if i == 0:
                        # storing the col merge_index
                        merge_index = len(field_vals)
                    try:
                        list_scans = literal_eval(hdf5[key]['merged_scans'].asstr()[()] )
                        field_vals.append(list_scans[0])
                        extra_content = True
                    except:
                        field_vals.append('None')

                elif opt_val in opt_list[1:]:
                    out = pre_edge(data, **pre_edge_kws)
                    field_vals.append(out[opt_val])

        report.add_row(field_vals)
        
        if extra_content:
            for item in list_scans[1:]:
                field_vals = []
                for j,index in enumerate(field_names):
                    if j !=  merge_index:
                        field_vals.append('')
                    else:
                        field_vals.append(item)
                report.add_row(field_vals)
            report.add_midrule()
    
    hdf5.close()
    return report

if __name__ == '__main__':
    import os
    import doctest
    doctest.testmod()

    # removing temp files    
    for fpath in ['database.h5',]:
        if os.path.exists(fpath):
            os.remove(fpath)

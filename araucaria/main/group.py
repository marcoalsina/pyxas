#!/usr/bin/python
# -*- coding: utf-8 -*-
class Group(object):
    """Group storage class.

    This class stores a single XAFS dataset.
    
    Parameters
    ----------
    name
        Name for the group. The default is None.
    kwargs
        Dictionary with content for the group.

    Example
    -------
    >>> from araucaria import Group
    >>> group = Group()
    >>> type(group)
    <class 'araucaria.main.group.Group'>
    """
    def __init__(self, name: str=None, **kwargs:dict):
        if name is None:
            name = hex(id(self))
        self.__name__ = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        if self.__name__ is not None:
            return '<Group %s>' % self.__name__
        else:
            return '<Group>'

    def add_content(self, content: dict) -> None:
        """Adds content to the Group.
        
        Parameters
        ----------
        content
            Dictionary with content to add to the Group.
        
        Returns
        -------
        :
        
        Raises
        ------
        TypeError
            If ``content`` is not a dictionary.

        Example
        -------
        >>> from araucaria import Group
        >>> from araucaria.utils import check_objattrs
        >>> content = {'name': 'group1'}
        >>> group   = Group()
        >>> group.add_content(content)
        >>> check_objattrs(group, Group, attrlist=['name',])
        [True]
        """
        if not isinstance(content, dict):
            raise TypeError('conent is not a valid dictionary.')
        else:
            for key, val in content.items():
                setattr(self, key, val)

    def get_mode(self) -> str:
        """Returns scan type of mu(E) for the group.

        Parameters
        ----------
        None

        Returns
        -------
        :
            Scan type of mu(E). Either 'fluo', 'mu', or 'mu_ref'.
    
        Raises
        ------
        ValueError
            If the scan type is unavailable or not recognized.
    
        Important
        ---------
        The scan type of mu(E) is assigned during reading of a file, 
        and should adhere to the following convention:
        
        - ``mu`` corresponds to a transmision mode scan.
        - ``fluo`` corresponds to a fluorescence mode scan.
        - ``mu_ref`` corresponds to a reference scan.
        
        Examples
        --------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_xmu
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> group_mu.get_mode()
        'mu'
    
        >>> # extracting only fluo scan
        >>> group_fluo = read_xmu(fpath, scan='fluo', ref=False)
        >>> group_fluo.get_mode()
        'fluo'
    
        >>> # extracting only mu_ref scan
        >>> group_ref = read_xmu(fpath, scan=None, ref=True)
        >>> group_ref.get_mode()
        'mu_ref'
        """
        scanlist = ['mu', 'fluo', 'mu_ref']
        scan = None

        for scantype in scanlist:
            if scantype in dir(self):
                scan = scantype
                break

        if scan is None:
            raise ValueError('scan type unavailable or not recognized.')

        return scan

    def has_ref(self) -> bool:
        """Tests if a reference scan for mu(E) is present in the group.

        Parameters
        ----------
        None

        Returns
        -------
        :
            True if attribute ``mu_ref`` exists in the group. False otherwise.
        
        Examples
        --------
        >>> from araucaria.testdata import get_testpath
        >>> from araucaria.io import read_xmu
        >>> fpath = get_testpath('xmu_testfile.xmu')
        >>> # extracting mu and mu_ref scans
        >>> group_mu = read_xmu(fpath, scan='mu')
        >>> group_mu.has_ref()
        True
    
        >>> # extracting only fluo scan
        >>> group_fluo = read_xmu(fpath, scan='fluo', ref=False)
        >>> group_fluo.has_ref()
        False
    
        >>> # extracting only mu_ref scan
        >>> group_ref = read_xmu(fpath, scan=None, ref=True)
        >>> group_ref.has_ref()
        True
        """
        if 'mu_ref' in dir(self):
            return True
        else:
            return False

class FitGroup(object):
    """Fit Group storage class.

    This class stores a fitted XAFS dataset.
    
    Parameters
    ----------
    name
        Name for the FitGroup. The default is None.
    kwargs
        Dictionary with content for the FitGroup.

    Example
    -------
    >>> from araucaria import FitGroup
    >>> group = FitGroup()
    >>> type(group)
    <class 'araucaria.main.group.FitGroup'>
    """
    def __init__(self, name: str=None, **kwargs:dict):
        if name is None:
            name = hex(id(self))
        self.__name__ = name
        for key, val in kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        if self.__name__ is not None:
            return '<FitGroup %s>' % self.__name__
        else:
            return '<FitGroup>'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
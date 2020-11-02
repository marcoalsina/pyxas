#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The :mod:`~araucaria.xas.xasutils` module offers the following XAFS utility functions :

.. list-table::
   :widths: auto
   :header-rows: 1

   * - Function
     - Description
   * - :func:`etok`
     - Converts energies to wavenumbers.
   * - :func:`ktoe`
     - Converts wavenumbers to energies.
"""

from numpy import ndarray
from scipy.constants import hbar    # reduced planck constant
from scipy.constants import m_e, e  # electron mass and and elementary charge

# constants
k2e = 1.e20 * hbar**2 / (2 * m_e * e)
e2k = 1/k2e

def etok(energy: ndarray) -> ndarray:
    """Converts photo-electron energies to wavenumbers.
    
    Parameters
    ----------
    energy
        Array of photo-electron energies.
    
    Returns
    -------
    :
        Arary of photo-electron wavenumbers.
    
    Example
    -------
    >>> from araucaria.xas import etok
    >>> e = 400      # eV
    >>> k = etok(e)  # A^{-1}
    >>> print('%1.5f' % k)
    10.24633
    """
    from numpy import sqrt
    return sqrt(energy/k2e)

def ktoe(k: ndarray) -> ndarray:
    """Converts photo-electron wavenumbers to energies.
    
    Parameters
    ----------
    k
        Array with photo-electron wavenumbers.
    
    Returns
    -------
    :
        Array with photo-electron energies.

    Example
    -------
    >>> from araucaria.xas import ktoe
    >>> k = 10      # A^{-1}
    >>> e = ktoe(k)  # eV
    >>> print('%1.5f' % e)
    380.99821
    """
    return k**2*k2e

def xftf_phase(group, path, kmult):
    """Phase-corrected magnitude of FFT EXAFS dataset.
    
    This function writes the phase corrected 
    magnitude of the forward XAFS fourier-transform 
    for the data and, if available, the FEFFIT model.
    
    Parameters
    ----------
    group: ndarray group
        Group containing the FFT EXAFS data.
    path: ndarray group
        FEFF group path to extract the phase shift
    kmult int
        k-multiplier of the EXAFS data.

    Returns
    -------
    None
    
    Notes
    -----
    The following data is appended to the parsed Group:
    
    - group.data.chir_pha_mag.
    - group.model.chir_pha_mag (optional).
    
    Warning
    -------
    This function depends on :func:`larch.xafs.xftf_fast`.
    
    """
    from numpy import interp, sqrt, exp
    import larch
    from larch.xafs import xftf_fast

    nrpts = len(group.data.r)
    feff_pha = interp(group.data.k, path._feffdat.k, path._feffdat.pha)
    
    # phase-corrected Fourier Transform
    data_chir_pha = xftf_fast(group.data.chi * exp(0-1j*feff_pha) *
                    group.data.kwin * group.data.k**kmult)[:nrpts] 
    group.data.chir_pha_mag = sqrt(data_chir_pha.real**2 + data_chir_pha.imag**2)

    try:
        model_chir_pha = xftf_fast(group.model.chi * exp(0-1j*feff_pha) *
                         group.model.kwin * group.model.k**kmult)[:nrpts]
        group.model.chir_pha_mag = sqrt(model_chir_pha.real**2 + model_chir_pha.imag**2)
    except:
        pass
    return
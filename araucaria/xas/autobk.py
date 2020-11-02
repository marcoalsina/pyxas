#!/usr/bin/python
# -*- coding: utf-8 -*-
from warnings import warn
from numpy import (pi, sign, sqrt, ceil, copy, ptp, 
                   arange, zeros, inf, ndarray, 
                   array, concatenate)
from scipy.interpolate import interp1d, splrep, splev
from lmfit import Parameters, minimize
from araucaria import Group
from .normalize import pre_edge
from .xasft import ftwindow, xftf_kwin
from .xasutils import e2k
from ..utils import index_nearest, check_objattrs, check_xrange

fmt_coef = 'coef_%2.2i'  # formated coefficient

def realimag(arr):
    #return real array of real/imag pairs from complex array
    return array([(i.real, i.imag) for i in arr]).flatten()

def spline_eval(kraw, mu, knots, coefs, order, kout):
    #eval bkg(kraw) and chi(k) for knots, coefs, order
    bkg = splev(kraw, [knots, coefs, order])
    chi = interp1d(kraw, (mu-bkg), kind='cubic')(kout)
    return bkg, chi

def resid(pars, ncoefs=1, knots=None, order=3, irbkg=1, nfft=2048,
            kraw=None, mu=None, kout=None, ftwin=1, kweight=1, 
            chi_std=None, nclamp=0, clamp_lo=1, clamp_hi=1):
    # residuals function
    coefs    = [pars[fmt_coef % i].value for i in range(ncoefs)]
    bkg, chi = spline_eval(kraw, mu, knots, coefs, order, kout)
    if chi_std is not None:
        chi = chi - chi_std
    out =  realimag(xftf_kwin(chi*ftwin, nfft=nfft)[:irbkg])
    if nclamp == 0:
        return out
    # spline clamps:
    scale       = (1.0 + 100*(out*out).sum())/(len(out)*nclamp)
    scaled_chik = scale * chi * kout**kweight
    return concatenate((out,
                        abs(clamp_lo)*scaled_chik[:nclamp],
                        abs(clamp_hi)*scaled_chik[-nclamp:]))


def autobk(group: Group, rbkg: float=1.0, k_range: list=[0,inf], 
           kweight: int=1, win: str='hanning',  dk:float=0.1, 
           nfft: int=2048, kstep: float=0.05, k_std: ndarray=None, 
           chi_std: ndarray=None, nclamp: int=2, clamp_lo: int=1, 
           clamp_hi: int=1, update:bool = False) -> dict:
    """Autobk algorithm to remove background of a XAFS scan.

    Parameters
    ----------
    group
        Group containing the spectrum for background removal.
    rbkg
        Distance (Å) for :math:`\chi(R)` above which the signal is ignored.
        The default is 1.0.
    k_range
        Wavenumber range (:math:`Å^{-1}`).The default is [0, :data:`~numpy.inf`].
    kweight
        Exponent for weighting chi(k) by k**kweight.
        The default is 1.
    win
        Name of the the FT window type. The default is 'hanning'.
    dk
        Tapering parameter for the FT window. The default is 0.1.
    nfft
        Array size for the FT.  The default is 2048.
    kstep
        Wavenumber step size for the FT (:math:`Å^{-1}`).  The default is 0.05.
    k_std
        Optional k array for standard :math:`\chi(k)`.
    chi_std
        Optional array for standard :math:`\chi(k)`.
    nclamp
        Number of energy end-points for clamp. The default is 2.
    clamp_lo
        Weight of low-energy clamp. The default is 1.
    clamp_hi
        Weight of high-energy clamp. The default is 1.
    update
        Indicates if the group should be updated with the autobk attributes.
        The default is False.
      
    Returns
    -------
    :
        Dictionary with the following arguments:

        - ``bkg``         : array with background signal :math:`\mu_0(E)`.
        - ``chie``        : array with :math:`\chi(E)`.
        - ``chi``         : array with :math:`\chi(k)`.
        - ``k``           : array with wavenumbers.
        - ``autobk_pars`` : dictionary with autobk parameters.

    Raises
    ------
    TypeError
        If ``group`` is not a valid Group instance.
    AttributeError
        If attribute ``e0`` or ``edge_step`` does not exist in ``group``.

    Warning
    -------
    ``rbkg`` cannot be lower than 2 x grid resolution of :math:`\chi(R)`.

    Notes
    -----
    The Autobk algorithm approximates a XAFS bakground signal by 
    fitting a cubic spline to chi(R) below the ``rbkg`` value.
    This spline is then removed from the original signal.

    If ``update=True`` the contents of the returned dictionary 
    will be included as attributes of ``group``.
    
    Example
    -------
    .. plot::
        :context: reset

        >>> from araucaria.testdata import get_testpath
        >>> from araucaria import Group
        >>> from araucaria.io import read_dnd
        >>> from araucaria.xas import pre_edge, autobk
        >>> from araucaria.utils import check_objattrs
        >>> fpath    = get_testpath('dnd_testfile.dat')
        >>> group    = read_dnd(fpath, scan='mu')  # extracting mu and mu_ref scans
        >>> pre      = pre_edge(group, update=True)
        >>> attrs    = ['e0', 'edge_step', 'bkg', 'chie', 'chi', 'k']
        >>> autbk    = autobk(group, update=True)
        >>> check_objattrs(group, Group, attrs)
        [True, True, True, True, True, True]

        >>> # plotting original and background spectrum
        >>> import matplotlib.pyplot as plt
        >>> from araucaria.plot import fig_xas_template
        >>> fig, ax = fig_xas_template(panels='xe')
        >>> line = ax[0].plot(group.energy, group.mu, label='mu')
        >>> line = ax[0].plot(group.energy, group.bkg, label='bkg', zorder=-1)
        >>> text = ax[0].set_ylabel('Absorbance')
        >>> leg  = ax[0].legend()
        >>> line = ax[1].plot(group.k, group.k**2 * group.chi, label='k^2 chi')
        >>> leg  = ax[1].legend()
        >>> fig.tight_layout()
        >>> plt.show(block=False)
    """
    # checking class and attributes
    check_objattrs(group, Group, attrlist=['e0', 'edge_step'], exceptions=True)

    #extracting data and mu as independent arrays
    energy    = group.energy
    mu        = getattr(group, group.get_mode())
    e0        = group.e0
    edge_step = group.edge_step

    # get array index for e0 (ie0)
    ie0   = index_nearest(energy, e0)
    e0    = energy[ie0]

    # get array index for rbkg (irbkg)
    rgrid = pi / (kstep * nfft)
    if rbkg < 2*rgrid:
        warn('rbkg is lower than 2 x grid resolution of chi(R). Resetting tbkg to this limit.')
        rbkg = 2*rgrid
    irbkg = int(ceil(rbkg/rgrid))

    # save ungridded k (kraw) and grided k (kout)
    # and ftwin (*k-weight) for FT in residual
    enpe = energy[ie0:] - e0
    kraw = sign(enpe) * sqrt(e2k*abs(enpe))

    # calculating krange
    krange = check_xrange(k_range, kraw)
    kout   = kstep * arange(ceil(krange[1]/kstep) )
    iemax  = min(len(energy), 2 + index_nearest(energy, e0 + (krange[1]**2)/e2k)) - 1

    # interpolate provided chi(k) onto the kout grid
    if chi_std is not None and k_std is not None:
        chi_std = interp1d(kout, k_std, chi_std, kind='cubic')(kout)
        
    # pre-load FT window
    ftwin = kout**kweight * ftwindow(kout, x_range=krange, win=win, dx1=dk)

    # calc k-value and initial guess for y-values of spline params
    # a minimum of 5 knots and a maximum of 64 knots are considered for the spline.
    nspl = max(5, min(64, int(2*rbkg*ptp(krange)/pi) + 2))
    spl_y, spl_k, spl_e  = [], [], []

    for i in range(nspl):
        # looping through the spline points
        # spline window in kraw is [ik-5, ik + 5], except at the extremes of the array
        # a weighted average for mu is calculated in the extremes and center of this window.
        q        = krange[0] + i*ptp(krange)/(nspl - 1)
        ik       = index_nearest(kraw, q)
        i1       = min(len(kraw)-1, ik + 5)
        i2       = max(0, ik - 5)
        spl_k.append(kraw[ik])
        spl_e.append(energy[ik+ie0])
        spl_y.append( (mu[i1+ie0] + 2*mu[ik+ie0]  + mu[i2+ie0]) / 4.0 )

    # get B-spline represention: knots, coefs, order=3
    # coefs will be fitted
    knots, coefs, order = splrep(spl_k, spl_y , k=3, s=0)

    # set fit parameters from initial coefficients
    params = Parameters()
    tol    = 1.e-5
    for i in range(len(coefs)):
        params.add(name = fmt_coef % i, value=coefs[i], vary=i<len(spl_y))

    initbkg, initchi = spline_eval(kraw[:iemax-ie0+1], mu[ie0:iemax+1],
                                   knots, coefs, order, kout)

    result = minimize(resid, params, method='leastsq',
                      gtol=tol, ftol=tol, xtol=tol, epsfcn=tol,
                      kws = dict(ncoefs  =len(coefs),
                                 chi_std =chi_std,
                                 knots=knots, order=order,
                                 kraw=kraw[:iemax-ie0+1],
                                 mu=mu[ie0:iemax+1], irbkg=irbkg, kout=kout,
                                 ftwin=ftwin, kweight=kweight,
                                 nfft=nfft, nclamp=nclamp,
                                 clamp_lo=clamp_lo, clamp_hi=clamp_hi))

    # write final results
    coefs = [result.params[fmt_coef % i].value for i in range(len(coefs))]
    bkg, chi = spline_eval(kraw[:iemax-ie0+1], mu[ie0:iemax+1],
                           knots, coefs, order, kout)
    obkg = copy(mu)
    obkg[ie0:ie0+len(bkg)] = bkg

    # output dictionaries
    init_bkg = copy(mu)
    init_bkg[ie0:ie0+len(bkg)] = initbkg
    
    autobk_pars = {'init_bkg'     : init_bkg,
                   'init_chi'     : initchi/edge_step,
                   'knots_e'      : spl_e,
                   'knots_y'      : [coefs[i] for i in range(nspl)],
                   'init_knots_y' : spl_y,
                   'nfev'         : result.nfev,
                   'k_range'      : krange,
                  }

    content = {'bkg'         : obkg,
               'chie'        : (mu-obkg)/edge_step,
               'k'           : kout,
               'chi'         : chi/edge_step,
               'edge_step'   : edge_step,
               'e0'          : e0,
               'autobk_pars' : autobk_pars
              }
    if update:
        group.add_content(content)

    return content
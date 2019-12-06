# -*- coding: utf-8 -*-
"""Collection of functions to deglitch XAS spectra.
"""

##Implemented functions
##---------------------
##    genesd
##    find_ri
##    find_critval
##    rollstd
##    deglitch

def genesd(data, max_outliers, alpha):
    """Routine to identify outliers from normally-distributed data set.

    Utilizes the generalized extreme Studentized deviate test for outliers to identify indices in an array that correspond to outliers.

    Parameters
    ----------
    data : array
        Array containing the data to perform the genESD routine.
    max_outliers : int
        Maximum number of outliers to remove.
    alpha : float
        alpha value for statistical test.
    
    Returns
    -------
    indexOutliers : group
        indices of outliers in the data.
    """
    import numpy as np
    
    # copy of original data
    cpdata       = np.copy(data)
    
    # containers for data
    rivals         = []    # Ri values
    critvals       = []    # critical values
    outliers       = []    # outliers
    
    for i in range(max_outliers):
        ri, outlier = find_ri(cpdata)
        
        outliers.append(cpdata[outlier])
        #removing outlier before calculating critical values
        cpdata    = np.delete(cpdata, outlier)
        critval   = find_critval(cpdata, alpha)

        # appending values to containers
        rivals.append(ri)
        critvals.append(critval)

    # at the highest value where Ri > critical value, that is the number of outliers
    j = 0
    i = 0
    while j < len(rivals):
        if rivals[j] > critvals[j]:
            i = j + 1
        j += 1
    outliers = outliers[:i]
    
    # returning outliers indices in the original data
    outliers_index = [i for i,elem in enumerate(data) if elem in outliers]

    return (np.array(outliers_index))


def find_ri(data):
    """Calculates test statistic for genesd.

    This function finds the value furthest from the mean in a dataset.
    Ri is given in terms of sample standard deviations from the mean.

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    
    Returns
    -------
    ri : float
        Test statistic for the generalized extreme Studentized deviate test.

    max_index : float
        The index corresponding to the data point furthest from the mean.
    """
    import numpy as np

    # calculating mean and std of data
    mean = np.mean(data)
    std  = np.std(data, ddof=1)
    
    # obtaining index for residual maximum
    residuals = np.absolute(data - mean)
    max_index = np.argmax(residuals)
    max_obs   = residuals[max_index]
    ri        = max_obs/std
    
    return (ri, max_index)


def find_critval(data, alpha):
    """Finds critical values for the genesd function.

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    alpha : float
        Significance level.
    
    Returns
    -------
    critval : float
        Returns the critical value for comparison with the test statistic Ri.
    """
    from scipy.stats import t
    
    n    = len(data)
    p    = 1 - ( alpha / ( 2 * (n + 1) ) )
    
    # finds t value corresponding to probability that 
    # sample within data set is itself an outlying point
    tval    = t.ppf(p, n-1) 
    critval = (n * tval) / ( ( (n - 1 + (tval**2)) * (n + 1) )**(1/2) )
    return (critval)


def rollstd(data, window, min_samples=2, edgemethod='nan'):
    """Rolling standard deviation calculation.

    Ignores nan values and calculates the standard deviation for a moving window.
    Results are returned in the index corresponding to the center of the window.

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    window : odd int
        Size of the rolling window for analysis.
    min_samples: int
        Minimum samples needed to calculate standard deviation. If the number of datapoints
        in the window is less than min_samples, np.nan is given as the standard deviation at
        that index.
    edgemethod : {'nan','calc','extend'}
        Dictates how standard deviation at the edge of the dataset is calculated
        'nan' inserts np.nan values for each point where the window cannot be centered on the analyzed point. 
        'calc' calculates standard deviation with an abbreviated window at the edges (e.g. the first sample will have (window/2)+1 points in the calculation).
        'extend' uses the nearest calculated value for the points at the edge of the data.
    Returns
    -------
    stddev : array
        Array with standard deviation found for each point centered in the window.
    """
    import numpy as np

    if window%2 == 0:
        raise ValueError('Please choose an odd value for the window length.')
    elif window < 3 or type(window)!=int:
        raise ValueError('Please select an odd integer value of at least 3 for the window length.')

    validEdgeMethods = ['nan', 'extend', 'calc'] 
    
    if edgemethod not in validEdgeMethods:
        raise ValueError('Please choose a valid edge method: '+ validEdgeMethods)

    movement=int((window - 1) / 2) #how many points on either side of the point of interest are included in the window?
    stddev=np.array([np.nan for point in data])
    for i, point in enumerate(data[ : -movement]):
        if i>=movement:
            if np.count_nonzero(np.isnan(data[i - movement : i + 1 + movement]) == False) >= min_samples:
                stddev[i]  =   np.nanstd(data[i - movement : i + 1 + movement], ddof=1)
    if edgemethod == 'nan':
        return stddev
    for i, point in enumerate(data[ : movement]):
        if edgemethod == 'calc':
            if np.count_nonzero(np.isnan(data[0 : i + 1 + movement]) == False) >= min_samples:
                stddev[i]  =   np.nanstd(data[0 : i + 1 + movement], ddof=1)
        if edgemethod == 'extend':
            stddev[i] = stddev[movement]
    for i, point in enumerate(data[-movement : ]):
        if edgemethod == 'calc':
            if np.count_nonzero(np.isnan(data[(-2 * movement) + i : ]) == False)>=min_samples:
                stddev[-movement + i] = np.nanstd(data[(-2 * movement) + i : ],ddof=1)
        if edgemethod == 'extend':
            stddev[-movement + i] = stddev[-movement - 1]
   
    return stddev

def deglitch(data, e_window='xas', sg_window_length=7, sg_polyorder=3, 
             alpha=.001, max_glitches='Default'):
    """Routine to deglitch a XAS spectrum.

    This function deglitches points in XAS data through two-step 
    fitting with Savitzky-Golay filter and outlier identification 
    with generalized extreme student deviate test.

    This code requires the data group to have at least an energy 
    and normalized absorption channel.

    Parameters
    ----------
    data : Larch Group
        Larch Group containing normalized XAS data.
    e_window : {'xas', 'xanes', 'exafs', (float, float)}
        'xas' scans the full spectrum.
        'xanes' looks from the beginning up to the edge + 150eV.
        'exafs' looks at the edge + 150eV to the end.
        (float, float) provides start and end energies in eV for analysis
    sg_window_length : odd int, default: 7
        Window length to build Savitzky-Golay filter from normalized data
    sg_polyorder : int, default: 3
        Polynomial order to build Savitzky-Golay filter from normalized data
    alpha : float, default: .001
        Alpha value for generalized ESD test for outliers.
    max_glitches : int, default: len(data)//10
         Maximum number of outliers to remove.
    
    Returns
    -------
    None
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    from larch_plugins.utils import group2dict
    from copy import deepcopy
    
    # computing the energy window to perform the deglitch:
    e_val     = 150  # energy limit to separate xanes from exafs [eV]
    e_windows = ['xas', 'xanes', 'exafs']
    if e_window in e_windows:
        if e_window   =='xas':
            deglitch(data, e_window='xanes', sg_window_length=sg_window_length,
                     sg_polyorder=sg_polyorder, alpha=alpha, max_glitches=max_glitches)
            deglitch(data, e_window='exafs', sg_window_length=sg_window_length,
                     sg_polyorder=sg_polyorder, alpha=alpha, max_glitches=max_glitches)
            return
        elif e_window =='xanes':
            e_window  = [data.energy[0], data.e0+e_val]
        elif e_window =='exafs':
            e_window  = [data.e0+e_val, data.energy[-1]]
    
    index = np.where((data.energy >= e_window[0]) & (data.energy <= e_window[1]))
    index=index[0]
    
    # creating copies of original data
    mu      = np.copy(data.norm)   # interpolated values for posterior analysis will be inserted in this 
    muNan   = np.copy(data.norm)   # copy to insert nan values at potential glitches to run the rolling standard deviation
    ener    = np.copy(data.energy) # copy of energy to create interp1d function without the potential glitches

    # not limited to start:end to ensure data at edges gets best possible fit
    sg_init = savgol_filter(data.norm, sg_window_length, sg_polyorder) 

    # computing the difference between normalized spectrum and the savitsky-golay filter
    res1    = data.norm[index]-sg_init[index]

    #If the max is not set to an int, the max will be set to the default of the length of the analyzed data//10
    if type(max_glitches) != int:
        maxGlitches = len(res1)//10
    out1 = genesd(res1, maxGlitches, alpha) #finds outliers in residuals between data and Savitzky-Golay filter
    if index[0] != 0: #compensates for nonzero starting index
        out1 = out1 + index[0]
    if len(out1) == 0: #deglitching ends here if no outliers are found in this first round of analysis
        return
    
    e2        = np.delete(ener, out1) #removes points that are poorly fitted by the S-G filter
    n2        = np.delete(mu, out1)
    f         = interp1d(e2, n2, kind='cubic') 
    interpPts = f(data.energy[out1]) #interpolates for normalized mu at the removed energies
    
    for i, point in enumerate(out1):
        mu[point] = interpPts[i] #inserts interpolated points into normalized data
        muNan[point] = np.nan #inserts nan values at the outlying points to calculate rolling standard deviation 
            #without influence from the outlying poitns
    
    SG2 = savgol_filter(mu, sg_window_length, sg_polyorder) #fits the normalized absorption with the interpolated points
        #in the place of the outlying points
    resDev = rollstd(muNan-SG2, sg_window_length, edgemethod='calc') #not limited to start:end ensure best calculation for edge values
    
    if True in np.isnan(resDev):
        devcopy = np.copy(resDev)
        nans = np.where(np.isnan(devcopy))
        print('Warning: there are nan values in your residuals standard deviation! Values will be estimated based on nearest values. Try a larger window length to avoid this.')

        for nanpoint in nans:
            newval = np.nanmedian(resDev[nanpoint - sg_window_length : nanpoint + sg_window_length + 1])
            devcopy[nanpoint] = newval #new median doesn't impact calculations
        resDev = devcopy
    
    res2 = (data.norm[index] - SG2[index]) / resDev[index] #residuals normalized to rolling standard deviation
    glitches = genesd(res2, maxGlitches, alpha) #by normalizing the standard deviation to the same window as our S-G calculation, 
        #we can tackle the full spectrum, accounting for the noise we expect in the data;
        #as a bonus, with the S-G filter, we ideally have a near-normal distribution of residuals
        #(which makes the generalized ESD a robust method for finding the outliers)
        
    if index[0] != 0:
        glitches = glitches + index[0]
    
    dataFilt = deepcopy(data) #non-destructive copy for comparison
    groupDict = group2dict(dataFilt) #transfers data copy to a dictionary (easier to work with)
    
    
    if len(glitches) == 0:
        glitches=None
    
    else:
        glitchDict={data.energy[glitch]:{} for glitch in glitches}
        for number in sorted(glitches, reverse=True):
            targetLength=len(data.energy) #everything that is of the same length as the energy array will have the indices
                                            #corresponding to glitches removed
            for key in dir(data):
                if type(getattr(data, key)) == np.ndarray or type(getattr(data, key)) == list:
                    if len(getattr(data, key)) == targetLength and key!='energy': #deletes the energy last
                        glitchDict[data.energy[number]].update({key : groupDict[key][number]})
                        groupDict[key] = np.delete(groupDict[key], number) #replaces the array with one that removes glitch points
                        #numpy arrays require extra steps to delete an element (which is why this takes this structure)
                        #removed indices is reversed to avoid changing the length ahead of the removal of points
            groupDict['energy'] = np.delete(groupDict['energy'], number)
    
    if glitches is not None:
        if hasattr(data,'glitches'):
            groupDict['glitches'].update(glitchDict)
        else:
            setattr(data,'glitches', glitchDict)
    
    dataKeys = list(groupDict.keys())
    for item in dataKeys:
        setattr(data, item, groupDict[item])
                
    return
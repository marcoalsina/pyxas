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
##    replacedata

def genesd(data, max_outliers, alpha):
    """One-line summary.

    Extended summary...

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

        #removing outlier before calculating critical values
        cpdata    = np.delete(cpdata, outlier)
        critval   = find_critval(cpdata, alpha)

        # appending values to containers
        rivals.append(ri)
        critvals.append(critval)
        outliers.append(cpdata[outlier])

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
    """One-line summary.

    Extended summary...

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    
    Returns
    -------
    ri : float
        Description here.

    max_index : float
        Description here.
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
    """One-line summary.

    Extended summary...

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    alpha : float
        Description here.
    
    Returns
    -------
    critval : float
        Description here.
    """
    from scipy.stats import t
    
    n    = len(data)
    p    = 1 - ( alpha / ( 2 * (n + 1) ) )
    
    # finds t value corresponding to probability that 
    # sample within data set is itself an outlying point
    tval    = t.ppf(p, n-1) 
    critval = (n * tval) / ( ( (n - 1 + (tval**2)) * (n + 1) )**(1/2) )
    return (critval)


def rollstd(data, window, minSamples=2, edgemethod='nan'):
    """One-line summary.

    Extended sumamry...

    Parameters
    ----------
    data : array
        Array containing the data to perform the analysis.
    window : int
        Description here.
    
    Returns
    -------
    Ri : float
        Description here.

    maxIndex : float
        Description here.
    """
    import numpy as np

    # checking the value of window
    if window%2 == 0:
        raise ValueError('Please choose an odd value for the window length.')
    elif window < 3 or type(window)!=int:
        raise ValueError('Please select an odd integer value of at least 3 for the window length.')

    validEdgeMethods=['nan','extend', 'calc'] #edge method - how do we handle the points at the beginning/end of the data
    #where one does not yet have a full window to calculate standard deviation?
    #   nan - just plug in nan for these values
    #   extend - bring out the values for the first full calculation
    #   calc - calculate over an incomplete window, e.g. the first sample will have (window/2)+1 points in the calculation
    if edgemethod not in validEdgeMethods:
        raise valueError('Please choose a valid edge method: '+ validEdgeMethods)

    movement=int((window-1)/2) #how many points on either side of the point of interest are included in the window?
    loopIndex = 0
    result=[]
    while loopIndex<len(data):
        if loopIndex<movement:
            if edgemethod!='calc': #this can be better optimized
                result.append(np.nan) #will not calculate standard deviation if it does not have a full window
            elif np.count_nonzero(np.isnan(data[0:loopIndex+movement])==False)>=minSamples: #number of nan values from start to end
                #                                                                           greater than the minimum values needed?
                result.append(np.nanstd(data[0:loopIndex+movement],ddof=1))
            else:
                result.append(np.nan)
        elif len(data)-1-loopIndex<movement:
            if edgemethod!='calc':
                result.append(np.nan)
            elif np.count_nonzero(np.isnan(data[loopIndex-movement:-1])==False)>=minSamples:
                result.append(np.nanstd(data[loopIndex-movement:-1],ddof=1))
            else:
                result.append(np.nan)
        else:
            if np.count_nonzero(np.isnan(data[loopIndex-movement:loopIndex+movement])==False)>=minSamples:
                result.append(np.nanstd(data[loopIndex-movement:loopIndex+movement],ddof=1))
            else:
                result.append(np.nan)
        loopIndex=loopIndex+1

    if edgemethod=='extend': #replaces values at beginning with the nearest value
        for number in result:
            if np.isnan(number)==False:
                beginningValue=number
                break
        for number in result[::-1]:
            if np.isnan(number)==False:
                endValue=number
                break
        for reach in list(range(movement)):
            result[reach]=beginningValue
            result[(-1*reach)-1]=endValue
    return result

def deglitch(data, e_window='xas', sg_window_length=7, sg_polyorder=3, 
             alpha=.0025, max_glitches='Default', replace=False):
    """Routine to deglitch a XAS spectrum.

    This function deglitches points in XAS data through two-step 
    fitting with Savitzky-Golay filter and outlier identification 
    with generalized extreme student deviate test.

    This code requires the data group to have at least an energy 
    and normalized absorption channel.

    Parameters
    ----------
    data : array
        Array containing the data to perform the genESD routine.
    e_window : {'xsa', 'xanes', 'exafs'}  
        'xas' scans the full spectrum.
        'xanes' looks from the beginning up to the edge + 150eV.
        'exafs' looks at the edge + 150eV to the end.
    e_range: list
        Description here.
    sg_window_length : 
        Description here.
    sg_polyorder : 
        Description here.
    alpha : float
        Alpha value for statistical test.
    max_glitches : int
         Maximum number of outliers to remove.
    replace : bool
        Description here.
    
    Returns
    -------
    indexOutliers : group
        indices of outliers in the data.
    """
    import numpy as np
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    from larch_plugins.utils import group2dict
    
    # computing the energy window to perform the deglitch:
    e_val     = 150  # energy limit to separate xanes from exafs [eV]
    e_windows = ['xas', 'xanes', 'exafs']
    if e_window in e_windows:
        if e_window   =='xas':
            e_window = [data.energy[0], data.energy[-1]]
        elif e_window =='xanes':
            e_window  = [data.energy[0], data.e0+150]
        elif e_window =='exafs':
            e_window  = [data.e0+150, data.energy[-1]]
    
    index = where(data.energy >= e_window[0]) & (data.energy <= e_window[1]))
    
    # creating copies of original data
    mu    = np.copy(data.norm)   # interpolated values for posterior analysis will be inserted in this 
    muNan = np.copy(data.norm)   # copy to insert nan values at potential glitches to run the rolling standard deviation
    ener  = np.copy(data.energy) # copy of energy to create interp1d function without the potential glitches

    # not limited to start:end to ensure data at edges gets best possible fit
    sg_init = savgol_filter(data.norm, sg_window_length, sg_polyorder) 

    # computing the difference between normalized spectrum and the savitsky-golay filter
    res1=data.norm[index]-sg_init[index]

    #If the max is not set to an int, the max will be set to the default of the length of the analyzed data//10
    if type(max_glitches) != int:
        maxGlitches=len(res1)//10
    out1 = genesd(res1, maxGlitches, alpha) #finds outliers in residuals between data and Savitzky-Golay filter
    if startIndex!=None: #compensates for nonzero starting index
        out1=out1+startIndex
    if len(out1)==0: #deglitching ends here if no outliers are found in this first round of analysis
        return
    
    e2 = np.delete(eCopy, out1) #removes points that are poorly fitted by the S-G filter
    n2 = np.delete(muCopy, out1)
    f = interp1d(e2, n2, kind='cubic') 
    interpPts = f(data.energy[out1]) #interpolates for normalized mu at the removed energies
    
    for i, point in enumerate(out1):
        muCopy[point]=interpPts[i] #inserts interpolated points into normalized data
        muNanCopy[point]=np.nan #inserts nan values at the outlying points to calculate rolling standard deviation 
            #without influence from the outlying poitns
    
    SG2 = savgol_filter(muCopy, SGwindow, SGpolyorder) #fits the normalized absorption with the interpolated points
        #in the place of the outlying points
    resDev=rollingStDev(muNanCopy-SG2, SGwindow, edgemethod='calc') #not limited to start:end ensure best calculation for edge values
    
    if True in np.isnan(resDev): #this next bit of code does a very rough interpolation for standard deviation
        #when the window is too small to accurately calculate it; 
        #it simply splits the difference between the two adjactent values for standard deviation (not including previously interpolated values)
        print('Warning: there are nan values in your residuals standard deviation! Values will be estimated based on nearest values. Try a larger window length to avoid this.')
        loopIndex=0
        wasNan=[]
        while loopIndex<len(resDev):
            if np.isnan(resDev[loopIndex]):
                wasNan.append(loopIndex)
                miniLoopIndex=loopIndex+1
                while np.isnan(resDev[miniLoopIndex]):
                    miniLoopIndex=miniLoopIndex+1
                for number in range(1,len(resDev)):
                    if loopIndex-number not in wasNan: #will not base this interpolation on already-interpolated points
                        resDev[loopIndex]=(resDev[loopIndex-number]+resDev[miniLoopIndex])/2
                        break
            loopIndex=loopIndex+1
    
    res2=(data.norm[startIndex:endIndex]-SG2[startIndex:endIndex])/resDev[startIndex:endIndex] #residuals normalized to rolling standard deviation
    glitches = genESD(res2, maxGlitches, alpha) #by normalizing the standard deviation to the same window as our S-G calculation, 
        #we can tackle the full spectrum, accounting for the noise we expect in the data;
        #as a bonus, with the S-G filter, we ideally have a near-normal distribution of residuals
        #(which makes the generalized ESD a robust method for finding the outliers)
        
    if startIndex!=None:
        glitches=glitches+startIndex
    
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
                        glitchDict[data.energy[number]].update({key:groupDict[key][number]})
                        groupDict[key]=np.delete(groupDict[key],number) #replaces the array with one that removes glitch points
                        #numpy arrays require extra steps to delete an element (which is why this takes this structure)
                        #removed indices is reversed to avoid changing the length ahead of the removal of points
            groupDict['energy']=np.delete(groupDict['energy'],number)
    if replace == True: #replaces the original data with the deglitched data
        if glitches is not None:
            if hasattr(data,'glitches'):
                groupDict['glitches'].update(glitchDict)
            else:
                setattr(data,'glitches',glitchDict)
            replaceData(data, groupDict)
    else:
        if glitches is not None:
            setattr(data, 'glitches_id', glitchDict)
                
    return


def replaceData(data, deglitchedDictionary):
    dataKeys = list(deglitchedDictionary.keys())
    for item in dataKeys:
        setattr(data,item,deglitchedDictionary[item])
    #replaces each subgroup in data with the equivalent subgroup in the filtered data dictionary
    return

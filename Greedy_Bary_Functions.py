#Functions for use with Barycentring Codes


import numpy as np
import greedy
from get_baryf import get_bary_in_loop
from scipy.interpolate import interp1d


def greedy_bary(detector, tssize, wl, source, tGPS, Eephem, Sephem, dt):
    """
    Generates emitdt data for sources and detector input. Creates Reduced Basis 
    for this using modified Gram-Schmidt process.  
    
    detector should be case insensitive string matching one of the below abbreviations
    GREEN BANK = 'GB'
    NARRABRI CS08 = 'NA'
    ARECIBO XYZ (JPL) = 'AO'
    Hobart, Tasmania = 'HO'
    DSS 43 XYZ = 'TD'
    PARKES  XYZ (JER) = 'PK'
    JODRELL BANK = 'JB'
    GB 300FT = 'G3'
    GB 140FT = 'G1RAD' (changed to distinguish it from GEO)
    VLA XYZ = 'VL'
    Nancay = 'NC'
    Effelsberg = 'EF'
    GW telescopes):
    LIGO Hanford = 'LHO', 'H1', 'H2'
    LIGO Livingston = 'LLO', 'L1'
    GEO 600 = 'GEO', 'G1'
    VIRGO = 'V1', 'VIRGO'
    TAMA = 'T1', 'TAMA'
    Earth geocentre = 'GEOCENTER'
    
    tssize is number of training sets, integer
    wl is length of training set, integer
    source = [sourcealpha, sourcedelta], 
    sourcealpha 1d array of right ascension, length wl
    sourcedelta 1d array of declination, length wl
    tGPS is array of linearly spaced time values within 630720013 and 1261872018
    of length wl
    Eephem & Sephem contain ephemeris data for the Earth and Sun respectively
    dt = tGPS[1]-tGPS[0]
    
    """
    #initialises training sets
    TS = np.zeros((tssize, wl))
    for i in range(tssize):
        
        
        #performs barycentring functions for source i
        emit = get_bary_in_loop(tGPS, detector,[source[0][i], source[1][i]], Eephem, Sephem) 
    
        #creates training vectors of time difference
        [emitdt, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
        TS[i]= np.reshape(emitdt, wl)
    
    
        ## normalises training vectors
        TS[i] /= np.sqrt(np.abs(greedy.dot_product(dt, TS[i], TS[i])))
            
    # tolerance for stopping algorithm
    tol = 1e-12
    
    
    
    #forms normalised basis vectors
    RB = greedy.greedy(TS, dt, tol)
    return RB
   
def lin_com(csize, no_v, RB, Edt):
    """
    extracts location of points along tGPS to create arrays of corresponding points
    along RB to form tRB. Also extracts similar points on Edt. Then inverts tRB 
    to assist in solving the matrix equation x = (AB)tRB
    
    Input:
    csize = check set size
    no_v = number of basis vectors in RB
    Edt = emitdt data for different source locations than those used to create RB
    
    Output:
    newEdt = fitted time delay vectors of Edt
    AB = Coefficient of matrix equation
    p = array of linearly spaced points along tGPS (not incl endpoints)
    x = array of Edt points corresponding to p locations
    tRB = array of RB points corresponding to p locations
    """
    
    
    #initialises arrays for use solving matrix equation x = (AB)tRB
    x= np.zeros((csize,no_v))
    tRB = np.zeros((no_v,no_v))
    AB = np.zeros((csize,no_v))
    
    
    #creates array of specific integer locations along Edt, missing end points
    l = np.linspace(csize/(2*no_v),csize,no_v,False)
    p = l.astype(int)
    for i in range(no_v):
        for j in range(no_v):
            #creates vectors of reduced bases corresponding to time location of points used 
            tRB[i][j] = RB[i][p[j]]
        
    #inverts tRB for use solving x = (AB)tRB
    C = np.linalg.inv((tRB))
    ##creates Edt to be tested with RB
    for i in range(csize):
        
        #takes points from basis vectors at corresponding tGPS entries
        for j in range(no_v):
            x[i][j] = Edt[i][p[j]]    
            
    #solves x=(AB)(tRB)
    AB = np.dot(x,C)
    newEdt=np.dot(AB,RB)

    return newEdt, AB, p, x, tRB
    


def interpnewEdt(newEdt, tGPS, step, wl):
    """
    Interpolation of emitdt vector created by lin com of RB. 
    
    Input:
        newEdt, tGPS & wl as above
        step = period between new time points. preferably integer
    
    Output:
       interEdt = ndarray of linear fit of points along each newEdt vector 
    """
    #creates time vector finer than tGPS but with same period
    newtime = np.arange(tGPS[0], tGPS[wl-1], step)
    
    #defines function f of form newEdt = f(tGPS) 
    f =interp1d(tGPS, newEdt)
    
    #applies f defined above to newtime vector, returning linear fit of points
    interEdt = f(newtime)
    return interEdt

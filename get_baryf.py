import numpy as np
from math import floor
from barycentre_earthf import barycentre_earth
from barycentref import barycentre

def get_bary_in_loop(tGPS, detector, source, Eephem, Sephem):
 
    """   
    This function is a driver for the solar system barycentring codes:
    barycenter_earth and barycenter. It takes in a detector
    name (for a variety of GW and radio telescopes [mainly consistent with
    TEMPO naming convensions]) - case insensitive:
    Radio telescopes (from TEMPO2 observatories.dat file):
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
    It also takes in a vector of GPS times (tGPS). Ephemeris data for Sun and Earth loaded in init_barycentre. The source must be specified with:
    sourcealpha = right ascension in rads
    sourcedelta = declination in rads
 
    The output will be vectors of time differences between the arrival time
    at the detector and SSB (emitdt), pulse emission time (emitte), and time
    derivatives (emitdd), and the individual elements making up the time
    delay - the Roemer delay (emitR), the Earth rotation delay (emitER), the
    Einstein delay (emitE) and the Shapiro delay (emitS).
    
    These have been tested against the equivalent LAL functions and the
    results are given at https://wiki.ligo.org/CW/MatlabBarycentring
    """  
    
    
    
    
    # set speed of light in vacuum (m/s)
    C_SI = 299792458
    
    # set the detector x, y and z positions on the Earth surface. For radio
    # telescopes use values from the TEMPO2 observatories.dat file, and for GW
    # telescopes use values from LAL.
    binsloc = np.zeros((3,1))  
    
    
    
    if detector.upper() == 'GB': # GREEN BANK #case insensitive string compare
        binsloc[0] = 882589.65
        binsloc[1] = -4924872.32
        binsloc[2] = 3943729.348
    
    elif detector.upper() == 'NA': # NARRABRI
        binsloc[0] = -4752329.7000
        binsloc[1] = 2790505.9340
        binsloc[2] = -3200483.7470
    elif detector.upper() == 'AO': # ARECIBO
        binsloc[0] = 2390490.0
        binsloc[1] = -5564764.0
        binsloc[2] = 1994727.0
    elif detector.upper() == 'HO': # Hobart
        binsloc[0] = -3950077.96
        binsloc[1] = 2522377.31
        binsloc[2] = -4311667.52
    elif detector.upper() == 'TD': # DSS 43
        binsloc[0] = -4460892.6
        binsloc[1] = 2682358.9
        binsloc[2] = -3674756.0
    elif detector.upper() == 'PK': # PARKES
        binsloc[0] = -4554231.5
        binsloc[1] = 2816759.1
        binsloc[2] = -3454036.3
    elif detector.upper() == 'JB': # JODRELL BANK
        binsloc[0] = 3822252.643
        binsloc[1] = -153995.683
        binsloc[2] = 5086051.443
    elif detector.upper() == 'G3': # GB 300FT
        binsloc[0] = 881856.58
        binsloc[1] = -4925311.86
        binsloc[2] = 3943459.70
    elif detector.upper() == 'G1RAD': # GB 140FT
        binsloc[0] = 882872.57
        binsloc[1] = -4924552.73
        binsloc[2] = 3944154.92
    elif detector.upper() == 'VL': # VLA
        binsloc[0] = -1601192.0
        binsloc[1] = -5041981.4
        binsloc[2] = 3554871.4
    elif detector.upper() == 'NC': # NANCAY
        binsloc[0] = 4324165.81
        binsloc[1] = 165927.11
        binsloc[2] = 4670132.83
    elif detector.upper() == 'EF': # Effelsberg
        binsloc[0] = 4033949.5
        binsloc[1] = 486989.4
        binsloc[2] = 4900430.8
    elif detector.upper() in ['H1','H2', 'LHO']: # LIGO Hanford
        binsloc[0] = -2161414.92636
        binsloc[1] = -3834695.17889
        binsloc[2] = 4600350.22664
    elif detector.upper() ==  'LLO' or detector.upper() ==  'L1':# LIGO Livingston
        binsloc[0] = -74276.04472380
        binsloc[1] = -5496283.71971000
        binsloc[2] = 3224257.01744000
    elif detector.upper() ==  'GEO' or detector.upper() ==  'G1': # GEO600
        binsloc[0] = 3856309.94926000
        binsloc[1] = 666598.95631700
        binsloc[2] = 5019641.41725000
    elif detector.upper() ==  'V1' or detector.upper() ==  'VIRGO': # Virgo
        binsloc[0] = 4546374.09900000
        binsloc[1] = 842989.69762600
        binsloc[2] = 4378576.96241000
    elif detector.upper() ==  'TAMA' or detector.upper() ==  'T1': # TAMA300
        binsloc[0] = -3946408.99111000
        binsloc[1] = 3366259.02802000
        binsloc[2] = 3699150.69233000
    elif detector.upper() ==  'GEOCENTER' or detector.upper() ==  'GEOCENTRE': # the geocentre
        binsloc[0] = 0
        binsloc[1] = 0
        binsloc[2] = 0 
        
    # set positions in light seconds    
    binsloc[:] = [x / C_SI for x in binsloc]
    
    # Set source info
    [sourcealpha, sourcedelta] = np.array(source)
    binalpha = sourcealpha # right ascension in radians
    bindelta = sourcedelta # declination in radians
    bindInv = 0 # inverse distance (assumption is that source is very distant)
    baryinsource = np.array([binalpha, bindelta, bindInv])
    
    
    # length of time vector
    length = len(tGPS)
    
    #initialise arrays for barycente output
    emitdt = np.zeros([length,1])
    emitte =np.zeros([length,1])
    emitdd =np.zeros([length,1])
    emitR =np.zeros([length,1])
    emitER =np.zeros([length,1])
    emitE =np.zeros([length,1])
    emitS =np.zeros([length,1])
    
   
    
    for i in range(length):
        # split time into seconds and nanoseconds
        tts = floor(tGPS[i])
        ttns = np.multiply((tGPS[i]-tts),1e9) 
        tt= np.array([tts, ttns])
        
            ### perform earth barycentring
        earthstruct = barycentre_earth(Eephem, Sephem, tt)
        
        #unpack earthstruct
        [[earthposNow, earthvelNow, earthgmstRad],
        [earthtzeA, earthzA, earththetaA],
        [earthdelpsi, earthdeleps, earthgastRad],
        [eartheinstein, earthdeinstein],
        [earthse, earthdse, earthdrse, earthrse]] = np.array(earthstruct)
        bingpss = tts
        bingpsns = ttns
        bingps = np.array([bingpss, bingpsns])
        baryinput = np.array([binsloc, bingps, baryinsource])   
    
  
        #perform barycentring   
        emit = barycentre(baryinput, earthstruct)
        [emitdeltaT, emittDot, emittes, emittens, emitroemer,  emiterot, emiteinstein, 
        emitshapiro] = emit
    
    
        emitdt[i] = emitdeltaT
        emitte[i] = [emittens*1e-9 + emittes]
        emitdd[i] = emittDot
        emitR[i] = emitroemer
        emitER[i] = emiterot
        emitE[i] = emiteinstein
        emitS[i] = emitshapiro
    
    emit = np.array([[emitdt], [emitte], [emitdd], [emitR], [emitER], [emitE], [emitS]]) ## final output!!!
    return emit
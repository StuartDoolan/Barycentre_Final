import numpy as np
from math import floor
from math import atan2
from math import log

def barycentre(baryinput, earthstruct):
    """
    This function takes in a structure, baryinput, which contains information
    about the detector and source for which the barycentring is required:
    Detector:
      binloc - a three element vector containing the x, y, z
          location of a detector position on the Earth's surface
      bintgps - a structure containing the GPS time in seconds (s)
          and nanoseconds (ns) at the detector
    Source:
      binalpha - the right ascension of the source in rads
    #  bindelta - the declination of the source in rads
      bindInv - inverse distance to source (generally set this to zero
          unless the source is very close)
    
    The function also takes in the earth structure produced by
    barycenter_earth

    The function transforms a detector arrival time (ta) to pulse emission
    time (te) in # TDB (plus the constant light-travel-time from source to
    SSB). Also returned is the time derivative dte/dta, and the time
    difference te(ta) - ta. This is contained in the arrays:
      emitte - pulse emission time
      emittDot - time derivative
      emitdeltaT - time difference
      emitroemer - Roemer delay
      emiterot - delay due to Earth's rotation
      emiteinstein - Einstein delay
      emitshapiro - Shapiro delay

    This function is a translation of Matthew Pitkin's Matlab-ified version of Curt Cutler's LAL function
    LALBarycenter.
    """
    # ang. vel. of Earth (rad/sec)
    OMEGA = 7.29211510e-5;
    
    #splitting input, definitely not the best way of doing this
    [binsloc, bingps, baryinsource]= baryinput 
    [bingpss, bingpsns] = bingps
    [binalpha, bindelta, bindInv] = baryinsource
    
    [[earthposNow, earthvelNow, earthgmstRad],
    [earthtzeA, earthzA, earththetaA],
    [earthdelpsi, earthdeleps, earthgastRad],
    [eartheinstein, earthdeinstein],
    [earthse, earthdse, earthdrse, earthrse]] = np.array(earthstruct)
    
    
    
    s = np.zeros([3,1]); # unit vector pointing at source, in J2000 Cartesian coords
    
    tgps = np.zeros([2,1]);
    
    tgps[0] = bingpss;
    tgps[1] = bingpsns;
    alpha = binalpha;
    delta = bindelta;
    
    # check that alpha and delta are in reasonable range
    if abs(alpha) > 2*np.pi or abs(delta) > 0.5*np.pi:
        print('Source position is not in reasonable range');
        emit = 0
        raise SystemExit()
        
    
    
    sinTheta=np.sin(np.pi/2.0-delta);
    s[2]=np.cos(np.pi/2.0-delta);    # s is vector that points towards source
    s[1]=sinTheta*np.sin(alpha);  # in Cartesian coords based on J2000
    s[0]=sinTheta*np.cos(alpha);  # 0=x,1=y,2=z
    
    rd = np.sqrt( binsloc[0]*binsloc[0] 
        + binsloc[1]*binsloc[1] 
        + binsloc[2]*binsloc[2])
    if rd == 0.0:
        latitude = np.pi/2;
    else:
        latitude = np.pi/2 - np.arccos(binsloc[2]/rd);
    
    
    longitude = atan2(binsloc[1], binsloc[0]);
    
    # ********************************************************************
    # Calucate Roemer delay for detector at center of earth
    # We extrapolate from a table produced using JPL DE405 ephemeris.
    # ---------------------------------------------------------------------
    
    roemer = 0;
    droemer = 0;
    
    for j in range(3):
        roemer = roemer + s[j]*earthposNow[j]
        droemer = droemer + s[j]*earthvelNow[j]
    
    
    # ********************************************************************
    # Now including Earth's rotation
    # ---------------------------------------------------------------------
    
    # obliquity of ecliptic at JD 245145.0, in radians. NOT! to be confused
    # with permittivity of free space; value from Explan. Supp. to Astronom.
    # Almanac:
    # eps0 = (23 + 26/60 + 21.448/3.6e3)*pi/180;
    eps0 = 0.40909280422232891;
    
    cosDeltaSinAlphaMinusZA = np.sin(alpha + earthtzeA[0])*np.cos(delta);
    
    cosDeltaCosAlphaMinusZA = (np.cos(alpha + earthtzeA[0])*np.cos(earththetaA[0])*np.cos(delta) 
    - np.sin(earththetaA[0])*np.sin(delta))
    
    sinDelta = (np.cos(alpha + earthtzeA[0])*np.sin(earththetaA[0])*np.cos(delta) 
        + np.cos(earththetaA[0])*np.sin(delta));
    
    # now taking NdotD, including lunisolar precession, using Eqs. 3.212-2 of
    # Explan. Supp. Basic idea for incorporating luni-solar precession is to
    # change the (alpha,delta) of source to compensate for Earth's
    # time-changing spin axis.
    
    NdotD = np.sin(latitude)*sinDelta +np.cos(latitude)*( 
        np.cos(earthgastRad+longitude-earthzA)*cosDeltaCosAlphaMinusZA 
        + np.sin(earthgastRad+longitude-earthzA)*cosDeltaSinAlphaMinusZA )
    
    erot = rd*NdotD;
    
    derot = OMEGA*rd*np.cos(latitude)*( 
        - np.sin(earthgastRad+longitude-earthzA[0])*cosDeltaCosAlphaMinusZA 
        + np.cos(earthgastRad+longitude-earthzA[0])*cosDeltaSinAlphaMinusZA )
    
    
    # --------------------------------------------------------------------------
    # Now adding approx nutation (= short-period,forced motion, by definition).
    # These two dominant terms, with periods 18.6 yrs (big term) and
    # 0.500 yrs (small term),resp., give nutation to around 1 arc sec; see
    # p. 120 of Explan. Supp. The forced nutation amplitude
    #  is around 17 arcsec.
    #
    # Note the unforced motion or Chandler wobble (called ``polar motion''
    # in Explanatory Supp) is not included here. However its amplitude is
    # order of (and a somewhat less than) 1 arcsec; see plot on p. 270 of
    # Explanatory Supplement to Ast. Alm.
    #
    # Below correction for nutation from Eq.3.225-2 of Explan. Supp.
    # Basic idea is to change the (alpha,delta) of source to
    # compensate for Earth's time-changing spin axis.
    #--------------------------------------------------------------------------
    
    delXNut = (-(earthdelpsi[0])*(np.cos(delta)*np.sin(alpha)*np.cos(eps0) 
        + np.sin(delta)*np.sin(eps0)))
    
    delYNut = (np.cos(delta)*np.cos(alpha)*np.cos(eps0)*(earthdelpsi[0]) 
        - np.sin(delta)*(earthdeleps[0]))
    
    delZNut = ((np.cos(delta)*np.cos(alpha)*np.sin(eps0)*(earthdelpsi[0]) 
        + np.cos(delta)*np.sin(alpha)*(earthdeleps[0])))
    
    NdotDNut = (np.sin(latitude)*delZNut 
        + np.cos(latitude)*np.cos(earthgastRad+longitude)*delXNut 
        + np.cos(latitude)*np.sin(earthgastRad+longitude)*delYNut)
    
    erot = erot + rd*NdotDNut
    
    derot = derot + OMEGA*rd*( 
        - np.cos(latitude)*np.sin(earthgastRad+longitude)*delXNut 
        + np.cos(latitude)*np.cos(earthgastRad+longitude)*delYNut )
    
    # Note erot has a periodic piece (P=one day) AND a constant piece,
    # since z-component (parallel to North pole) of vector from
    # Earth-center to detector is constant
    
    # ********************************************************************
    # Now adding Shapiro delay. Note according to J. Taylor review article
    # on pulsar timing, max value of Shapiro delay (when rays just graze sun)
    # is 120 microsec.
    #
    # Here we calculate Shapiro delay
    # for a detector at the center of the earth
    # Causes errors of order 10^{-4}sec * 4 * 10^{-5} = 4*10^{-9} sec
    # --------------------------------------------------------------------
    
    rsun = 2.322; # radius of sun in sec
    seDotN = (earthse[2]*np.sin(delta)+ (earthse[0]*np.cos(alpha) 
        + earthse[1]*np.sin(alpha))*np.cos(delta))
    
    dseDotN = (earthdse[2]*np.sin(delta)+(earthdse[0]*np.cos(alpha) 
        + earthdse[1]*np.sin(alpha))*np.cos(delta))
    
    b = np.sqrt(earthrse*earthrse-seDotN*seDotN)
    db = (earthrse*earthdrse-seDotN*dseDotN)/b
    
    AU_SI = 1.4959787066e11; # AU in m
    C_SI = 299792458; # speed of light in vacuum in m/s
    
    if b < rsun and seDotN < 0: # if gw travels thru interior of Sun
        shapiro  = (9.852e-6*log( (AU_SI/C_SI) / 
            (seDotN + np.sqrt(rsun*rsun + seDotN*seDotN))) 
            + 19.704e-6*(1 - b/rsun))
        dshapiro = - 19.704e-6*db/rsun
    else:  #else the usual expression
        shapiro  = 9.852e-6*log( (AU_SI/C_SI)/(earthrse + seDotN));
        dshapiro = -9.852e-6*(earthdrse + dseDotN)/(earthrse + seDotN);
    
    
    # ********************************************************************
    # Now correcting Roemer delay for finite distance to source.
    # Timing corrections are order 10 microsec
    # for sources closer than about 100 pc = 10^10 sec.
    # --------------------------------------------------------------------
    
    r2 = 0.; # squared dist from SSB to center of earth, in sec^2
    dr2 = 0.; # time deriv of r2
    
    if bindInv > 1.0e-11: #implement if corr.  > 1 microsec
        for j in range(0,2):
            r2 = r2 + earthposNow[j]*earthposNow[j];
            dr2 = dr2 + 2*earthposNow[j]*earthvelNow[j];
        
    
        finiteDistCorr = -0.5e0*(r2 - roemer*roemer)*bindInv;
        dfiniteDistCorr = -(0.5e0*dr2 - roemer*droemer)*bindInv;
    
    else:
   	finiteDistCorr = 0
        dfiniteDistCorr = 0
    
    
    # -----------------------------------------------------------------------
    # Now adding it all up.
    # emit.te is pulse emission time in TDB coords
    # (up to a constant roughly equal to ligh travel time from source to SSB).
    # emit->deltaT = emit.te - tgps.
    # -----------------------------------------------------------------------
    
    emitdeltaT = roemer + erot + eartheinstein - shapiro + finiteDistCorr
    emittDot = (1.e0 + droemer + derot + earthdeinstein 
        - dshapiro + dfiniteDistCorr)
    
    deltaTint = floor(emitdeltaT);
    
    if 1.e-9*tgps[1] + emitdeltaT - deltaTint >= 1:
        emittes = (bingpss+float(deltaTint)+1)
        emittens = floor(1.e9*(bingpsns*1.e-9 + emitdeltaT - deltaTint - 1));
    else:
        emittes =(float(bingpss)+float(deltaTint))
        emittens = floor(1.e9*(bingpsns*1.e-9 + emitdeltaT - deltaTint));
    
    emitroemer = roemer;
    emiterot = erot;
    emiteinstein = eartheinstein;
    emitshapiro = -shapiro;
    
    emit = np.array([emitdeltaT, emittDot, emittes, emittens, emitroemer,  emiterot, emiteinstein, 
    emitshapiro])
    return emit
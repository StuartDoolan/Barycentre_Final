import numpy as np
from math import floor

def barycentre_earth(Eephem, Sephem, tGPS):

    """
     This function takes in ephemeris data for the Earth (Eephem)
     and Sun (Sephem), containing the following information:
       ephemgps - GPS times of the data within the ephemeris
       ephempos - a three by N array containing the x, y, z position
       ephemvel - a three by N array containing the x, y, z velocity
       ephemacc - a three by N array containing the x, y, z acceleration
       ephemnentries - number of entries in file (N)
       ephemdttable - time steps in table
    
     It also takes in a GPS time to calculate informtion about the Earth's
     state at that time. The tGPS time is a
     structure holding the GPS time in seconds (tGPS.s) and nanoseconds
     (tGPS.ns). The earth nd array holds the following information:
     Position information
       earthposNow - a three element array of the x, y and z position
           of the Earth
       earthvelNow - a three element array of the x, y and z velocity
           of the Earth
       earthgmstRad - the Greenwich Mean Sidereal Time in radians
    
     Lunisolar precession terms
        earthtzeA
        earthzA
        earththetaA
    
    Nutation terms
       earthdelpsi
       earthdeleps
       earthgastRad
    
     Einstein Delay terms
       eartheinstein
       earthdeinstein
    
     Shanp.piro Delay terms
       earthse - three element vector
       earthdse - three element vector
       earthdrse
       earthrse
    
     This function is translated from Matthew Pitkin's Matlab-ified version of Curt Culter's LAL
     function LALBarycenterEarth, which itself uses functions from TEMPO.
     """
    #unpack Eephem & Sephem data
    [ephemEgps, Edttables, Eentries, ephemEpos, ephemEvel, ephemEacc] = np.array(Eephem)
    [ephemSgps, Sdttables, Sentries, ephemSpos, ephemSvel, ephemSacc] = np.array(Sephem)
    
    #set leap second table
    leapstable = np.array([[ 2444239.5,    -43200, 19], # 1980-Jan-01
    [2444786.5,  46828800, 20],  # 1981-Jul-01
    [2445151.5,  78364801, 21],  # 1982-Jul-01
    [2445516.5, 109900802, 22],  # 1983-Jul-01
    [2446247.5, 173059203, 23],  # 1985-Jul-01
    [2447161.5, 252028804, 24],  # 1988-Jan-01
    [2447892.5, 315187205, 25],  # 1990-Jan-01
    [2448257.5, 346723206, 26],  # 1991-Jan-01
    [2448804.5, 393984007, 27],  # 1992-Jul-01
    [2449169.5, 425520008, 28],  # 1993-Jul-01
    [2449534.5, 457056009, 29],  # 1994-Jul-01
    [2450083.5, 504489610, 30],  # 1996-Jan-01
    [2450630.5, 551750411, 31],  # 1997-Jul-01
    [2451179.5, 599184012, 32],  # 1999-Jan-01
    [2453736.5, 820108813, 33],  # 2006-Jan-01
    [2454832.5, 914803214, 34],  # 2009-Jan-01
    [2456109.5, 1025136015, 35],  # 2012-Jul-01
    [2457204.5, 1119744016, 36],  # 2015-Jul-01
    [2457754.5, 1167264017, 37]])    # 2017-Jan-01
    
    
  
    tgps = np.zeros((2,1))
    
    tgps[0] = tGPS[0]
    tgps[1] = tGPS[01]
    
    # set GPS times of the first data points in the Earth and Sun ephemeris
    # files
    tinitE = ephemEgps[0] 
    tinitS = ephemSgps[0]
    
    t0e=tgps[0]-tinitE 
    
    
    # finding Earth table entry closest to arrival time
    ientryE = int(floor(float(t0e)/float(Edttables)+0.5))
    
    
    t0s=tgps[0]-tinitS
    
    # finding Sun table entry closest to arrival time
    ientryS = int(floor(float(t0e)/float(Sdttables)+0.5))
    
    ## Making sure tgps is within earth and sun ephemeris arrays
    ##ie between 630720013 and 1261872018
    
    if ientryE < 0 or ientryE >=  int(Eentries):
        print('Problem with Earth ephemeris data. Time out of ephemeris file bounds.')
        raise SystemExit()
    
    

    if ientryS < 0 or ientryS >=  int(Sentries):
        print('Problem with Sun ephemeris data. Time out of ephemeris file bounds')
        raise SystemExit()
        
    # tdiff is arrival time minus closest Earth table entry tdiff can be pos.
    # or neg.
    tdiffE = t0e - float(Edttables)*float(ientryE) + tgps[1]*1.e-9
    tdiff2E = tdiffE*tdiffE
    
    # same for Sun
    tdiffS = t0s - float(Sdttables)*float(ientryS)  + tgps[1]*1.e-9
    tdiff2S = tdiffS*tdiffS
    
    # ********************************************************************
    # Calucate position and vel. of center of earth
    
    pos = np.array([ephemEpos[0][ientryE], ephemEpos[1][ientryE], ephemEpos[2][ientryE]]) 
    vel = np.array([ephemEvel[0][ientryE], ephemEvel[1][ientryE], ephemEvel[2][ientryE]])
    acc = np.array([ephemEacc[0][ientryE], ephemEacc[1][ientryE], ephemEacc[2][ientryE]])
    
    earthposNow = np.zeros((3,1))
    earthvelNow = np.zeros((3,1))
    
    for j in range(3):
            earthposNow[j] = pos[j] + vel[j]*tdiffE + 0.5*acc[j]*tdiff2E 
            earthvelNow[j] = vel[j] + acc[j]*tdiffE
    
    # ********************************************************************
    # Now calculating Earth's rotational state.
    
    eps0 = 0.40909280422232891 # obliquity of ecliptic at JD 245145.0  
    
    # number of leap secs added to UTC calendar since Jan 1, 2000 00:00:00 UTC
    # leap is a NEGATIVE number for dates before Jan 1, 2000.
    
    # get number of leap seconds
    # check we're not before the leap second table
  
    if tGPS[0] < leapstable[0,1]:
        print('GPS time is prior to start of data table')
        raise SystemExit()
        
    #account for 0 indexing ######### check
    numleaps =len(leapstable)
    
    # if after time is after end of table then add all leap second
    if tGPS[0] > leapstable[numleaps-1,1]:
        leaps = leapstable[numleaps-1,2] - 19
    else:
        for leap in range(2,numleaps+1):
            if tGPS[0] < leapstable[leap-1,1]:
                break
        # number of leap seconds since start of GPS
        leaps = leapstable[leap-2,2] - 19
    # number of leap seconds since 2000
    leapsSince2000 = leaps - 13 
    
    
    # number of full seconds (integer) on elapsed on UT1 clock from Jan 1, 2000
    # 00:00:00 UTC to present
    # first subtract off value of gps clock at Jan 1, 2000 00:00:00 UTC
    tuInt = tGPS[0] - 630720013
    
    # next subtract off # leap secs added to UTC calendar since Jan 1, 2000
    # 00:00:00 UTC
    ut1secSince1Jan2000 = tuInt - leapsSince2000
    
    # time elapsed on UT1 clock (in Julian centuries) between Jan 1.5 2000
    # (= J2000 epoch) and the present time (tGPS)
    tuJC = (ut1secSince1Jan2000 + tgps[1]*1.e-9 - 43200)/(8.64e4*36525) 
    
    # UT1 time elapsed, in Julian centuries, since  Jan 1.5 2000 (= J2000
    # epoch) and pulse arrival time
    
    # full days on ut1 clock since Jan 1, 2000 00:00:00 UTC
    fullUt1days = floor(ut1secSince1Jan2000/8.64e4)
    
    # UT1 time elapsed, in Julian centuries between Jan 1.5 2000 (= J2000
    # epoch) and midnight before gps(1)
    tu0JC = (fullUt1days - 0.5)/36525.0
    
    # time, in Jul. centuries, between pulse arrival time and previous midnight
    # (UT1)
    dtu= tuJC - tu0JC
    
    # days (not an integer) on GPS (or TAI,etc.) clock since J2000
    daysSinceJ2000 = (tuInt -43200)/8.64e4 
    
    # ----------------------------------------------------------------------
    # in below formula for gmst0 & gmst, on rhs we SHOULD use time elapsed
    # in UT1, but instead we insert elapsed in UTC. This will lead to
    # errors in tdb of order 1 microsec.
    # ----------------------------------------------------------------------
    
    # formula for Greenwich mean sidereal time on p.50 of Explanatory Supp. to
    # Ast Alm. gmst unit is sec, where 2np.pi rad = 24*60*60sec
    gmst0 = 24110.54841e0 + tu0JC*(8640184.812866 + tu0JC*(0.093104 
        - tu0JC*6.2e-6))
    
    # -----------------------------------------------------------------------
    # Explan. of tu0JC*8640184.812866e0 term: when tu0JC0=0.01 Julian centuries
    #(= 1 year), Earth has spun an extra 86401.84 sec (approx one revolution)
    # with respect to the stars.
    #
    # Note: this way of organizing calc of gmst was stolen from TEMPO
    # see subroutine lmst (= local mean sidereal time) .
    # ------------------------------------------------------------------------
    
    gmst = gmst0+ dtu*(8.64e4*36525 + 8640184.812866 + 
        0.093104*(tuJC + tu0JC) - 6.2e-6*(tuJC*tuJC + tuJC*tu0JC + 
        tu0JC*tu0JC))
    
    gmstRad=gmst*np.pi/43200 
    
    earthgmstRad = gmstRad 
    
    # ------------------------------------------------------------------------
    # Now calculating 3 terms, describing  lunisolar precession
    # see Eqs. 3.212 on pp. 104-5 in Explanatory Supp.
    # -------------------------------------------------------------------------
    
    earthtzeA = tuJC*(2306.2181 + (0.30188 + 0.017998*tuJC)*tuJC ) *np.pi/6.48e5
    
    earthzA = tuJC*(2306.2181 + (1.09468 + 0.018203*tuJC)*tuJC ) *np.pi/6.48e5
    
    earththetaA = tuJC*(2004.3109 - (0.42665 + 0.041833*tuJC)*tuJC )*np.pi/6.48e5
    
    # --------------------------------------------------------------------------
    # Now adding approx nutation (= short-period,forced motion, by definition).
    # These two dominant terms, with periods 18.6 yrs (big term) and
    # 0.500 yrs (small term),respp., give nutation to around 1 arc sec see
    # p. 120 of Explan. Supp. The forced nutation amplitude
    # is around 17 arcsec.
    #
    # Note the unforced motion or Chandler wobble (called ``polar motion''
    # in Explanatory Supp) is not included here. However its amplitude is
    # order of (and a somewhat less than) 1 arcsec see plot on p. 270 of
    # Explanatory Supplement to Ast. Alm.
    # --------------------------------------------------------------------------
    
    # define variables for storing sin/np.cos of deltaE, eps0 etc+
    
    earthdelpsi = [(-0.0048*np.pi/180.0)
    * np.sin( (125.0 - 0.05295*daysSinceJ2000)*np.pi/180.0 ) 
    - (4.e-4*np.pi/180.)* np.sin( (200.9 + 1.97129*daysSinceJ2000)*np.pi/180.0 )]
    
    
    earthdeleps = [(0.0026*np.pi/180.0)* 
        np.cos( (125.0 - 0.05295*daysSinceJ2000)*np.pi/180.0 ) 
        + (2.e-4*np.pi/180.)* 
        np.cos( (200.9 + 1.97129*daysSinceJ2000)*np.pi/180.0 )]
    
    # '`equation of the equinoxes''
    earthgastRad = gmstRad + earthdelpsi[0]*np.cos(eps0)
    
    # ------------------------------------------------------------------------
    # Now calculating 3 terms, describing  lunisolar precession
    # see Eqs. 3.212 on pp. 104-5 in Explanatory Supp.
    # -------------------------------------------------------------------------
    
    earthtzeA = [tuJC*(2306.2181 + (0.30188 + 0.017998*tuJC)*tuJC )
        *np.pi/6.48e5]
    
    earthzA = [tuJC*(2306.2181 + (1.09468 + 0.018203*tuJC)*tuJC )
        *np.pi/6.48e5]
    
    earththetaA = [tuJC*(2004.3109 - (0.42665 + 0.041833*tuJC)*tuJC )
        *np.pi/6.48e5]
    
    # --------------------------------------------------------------------------
    # Now adding approx nutation (= short-period,forced motion, by definition).
    # These two dominant terms, with periods 18.6 yrs (big term) and
    # 0.500 yrs (small term),respp., give nutation to around 1 arc sec see
    # p. 120 of Explan. Supp. The forced nutation amplitude
    # is around 17 arcsec.
    #
    # Note the unforced motion or Chandler wobble (called ``polar motion''
    # in Explanatory Supp) is not included here. However its amplitude is
    # order of (and a somewhat less than) 1 arcsec see plot on p. 270 of
    # Explanatory Supplement to Ast. Alm.
    # --------------------------------------------------------------------------
    
    # define variables for storing sin/np.cos of deltaE, eps0 etc
    
    earthdelpsi = [(-0.0048*np.pi/180.0)* 
        np.sin( (125.0 - 0.05295*daysSinceJ2000)*np.pi/180.0 ) 
        - (4.e-4*np.pi/180.)* 
        np.sin( (200.9 + 1.97129*daysSinceJ2000)*np.pi/180.0 )]
    
    
    earthdeleps = [(0.0026*np.pi/180.0)* 
        np.cos( (125.0 - 0.05295*daysSinceJ2000)*np.pi/180.0 ) 
        + (2.e-4*np.pi/180.)* 
        np.cos( (200.9 + 1.97129*daysSinceJ2000)*np.pi/180.0 )]
    
    # '`equation of the equinoxes''
    earthgastRad = gmstRad + earthdelpsi[0]*np.cos(eps0)
    
    
    # **********************************************************************
    # Now calculating Einstein delay. This is just difference between
    # TDB and TDT.
    # We steal from TEMPO the approx 20 biggest terms in an expansion.
    # -----------------------------------------------------------------------
    # jedtdt is Julian date (TDT) MINUS 2451545.0 (TDT)
    #  -7300.5 = 2444244.5 (Julian date when gps clock started)
    #        - 2451545.0 TDT (approx. J2000.0)
    # Note the 51.184 s difference between TDT and GPS
    # -----------------------------------------------------------------------
    
    jedtdt = -7300.5 + (tgps[0] + 51.184 + tgps[1]*1.e-9)/8.64e4
    # converting to TEMPO expansion param = Julian millenium, NOT Julian
    # century
    jt = jedtdt/3.6525e5
    
    eartheinstein = 1.e-6*( 
        1656.674564e0 * np.sin(6283.075849991*jt + 6.240054195 ) + 
        22.417471e0 * np.sin(5753.384884897*jt + 4.296977442 )  + 
        13.839792e0 * np.sin(12566.151699983*jt + 6.196904410 )  + 
        4.770086e0 * np.sin(529.690965095*jt + 0.444401603 )   + 
        4.676740e0 * np.sin(6069.776754553 *jt + 4.021195093 )   + 
        2.256707e0 * np.sin(213.299095438 *jt + 5.543113262 )   + 
    1.694205e0 * np.sin(-3.523118349 *jt + 5.025132748 )   + 
        1.554905e0 * np.sin(77713.771467920 *jt + 5.198467090 )   + 
        1.276839e0 * np.sin(7860.419392439 *jt + 5.988822341 )   + 
        1.193379e0 * np.sin(5223.693919802 *jt + 3.649823730 )   + 
        1.115322e0 * np.sin(3930.209696220 *jt + 1.422745069 )   + 
        0.794185e0 * np.sin(11506.769769794 *jt + 2.322313077 )   + 
        0.447061e0 * np.sin(26.298319800 *jt + 3.615796498 )   + 
        0.435206e0 * np.sin(-398.149003408 *jt + 4.349338347 )   + 
        0.600309e0 * np.sin(1577.343542448 *jt + 2.678271909 )   + 
        0.496817e0 * np.sin(6208.294251424 *jt + 5.696701824 )   + 
        0.486306e0 * np.sin(5884.926846583 *jt + 0.520007179 )   + 
        0.432392e0 * np.sin(74.781598567 *jt + 2.435898309 )   + 
        0.468597e0 * np.sin(6244.942814354 *jt + 5.866398759 )   + 
        0.375510e0 * np.sin(5507.553238667 *jt + 4.103476804 ))
    
    # now adding NEXT biggest (2nd-tier) terms from Tempo
    eartheinstein = eartheinstein + 1.e-6*( 
    0.243085 * np.sin(-775.522611324 *jt + 3.651837925 )   + 
    0.173435 * np.sin(18849.227549974 *jt + 6.153743485 )   + 
    0.230685 * np.sin(5856.477659115 *jt + 4.773852582 )   + 
        0.203747 * np.sin(12036.460734888 *jt + 4.333987818 )   + 
        0.143935 * np.sin(-796.298006816 *jt + 5.957517795 )   + 
        0.159080 * np.sin(10977.078804699 *jt + 1.890075226 )   + 
        0.119979 * np.sin(38.133035638 *jt + 4.551585768 )   + 
        0.118971 * np.sin(5486.777843175 *jt + 1.914547226 )   + 
        0.116120 * np.sin(1059.381930189 *jt + 0.873504123 )   + 
        0.137927 * np.sin(11790.629088659 *jt + 1.135934669 )   + 
        0.098358 * np.sin(2544.314419883 *jt + 0.092793886 )   + 
        0.101868 * np.sin(-5573.142801634 *jt + 5.984503847 )   + 
        0.080164 * np.sin(206.185548437 *jt + 2.095377709 )   + 
        0.079645 * np.sin(4694.002954708 *jt + 2.949233637 )   + 
        0.062617 * np.sin(20.775395492 *jt + 2.654394814 )   + 
        0.075019 * np.sin(2942.463423292 *jt + 4.980931759 )   + 
        0.064397 * np.sin(5746.271337896 *jt + 1.280308748 )   + 
        0.063814 * np.sin(5760.498431898 *jt + 4.167901731 )   + 
        0.048042 * np.sin(2146.165416475 *jt + 1.495846011 )   + 
        0.048373 * np.sin(155.420399434 *jt + 2.251573730 ))
    
    
    # below, I've just taken derivative of above expression for einstein,
    # then commented out terms that contribute less than around 10^{-12}
    # to tDotBary
    
    earthdeinstein = 1.e-6*( 
    1656.674564*6283.075849991* 
    np.cos(6283.075849991*jt + 6.240054195 )+ 
        22.417471*5753.384884897* 
        np.cos(5753.384884897*jt + 4.296977442 )  + 
        13.839792*12566.151699983* 
        np.cos(12566.151699983*jt + 6.196904410 ) + 
        4.676740*6069.776754553* 
        np.cos(6069.776754553*jt + 4.021195093 )   + 
        1.554905*77713.771467920* 
        np.cos(77713.771467920*jt + 5.198467090 ) 
        )/(8.64e4*3.6525e5)
    
    # neglect the following terms next terms (all also divided by
    # 8.64e4*3.6525e5)
    #
    #       4.770086e0*529.690965095e0*
    #               np.cos(529.690965095e0*jt + 0.444401603e0 )   +
    #       2.256707e0*213.299095438e0*
    #              np.cos(213.299095438e0*jt + 5.543113262e0 )    +
    #
    #       1.694205e0*-3.523118349e0*
    #              np.cos(-3.523118349e0*jt + 5.025132748e0 )     +
    #       +   1.276839e0*7860.419392439e0*
    #              np.cos(7860.419392439e0*jt + 5.988822341e0 )   +
    #
    #       1.193379e0*5223.693919802e0*
    #              np.cos(5223.693919802e0*jt + 3.649823730e0 )   +
    #
    #       1.115322e0*3930.209696220e0*
    #              np.cos(3930.209696220e0*jt + 1.422745069e0 )   +
    #
    #       0.794185e0*11506.769769794e0*
    #              np.cos(11506.769769794e0 *jt + 2.322313077e0 )  +
    #
    #       0.447061e0*26.298319800e0*
    #              np.cos(26.298319800e0*jt + 3.615796498e0 )     +
    #
    #       0.435206e0*-398.149003408e0*
    #              np.cos(-398.149003408e0*jt + 4.349338347e0 )   +
    #
    #       0.600309e0*1577.343542448e0*
    #              np.cos(1577.343542448e0*jt + 2.678271909e0 )   +
    #
    #       0.496817e0*6208.294251424e0*
    #              np.cos(6208.294251424e0*jt + 5.696701824e0 )   +
    #
    #       0.486306e0*5884.926846583e0*
    #              np.cos(5884.926846583e0*jt + 0.520007179e0 )   +
    #
    #       0.432392e0*74.781598567e0*
    #              np.cos(74.781598567e0*jt + 2.435898309e0 )     +
    #
    #       0.468597e0*6244.942814354e0*
    #              np.cos(6244.942814354e0*jt + 5.866398759e0 )   +
    #
    #       0.375510e0*5507.553238667e0*
    #              np.cos(5507.553238667e0*jt + 4.103476804e0 )
    #
    
    # Note for above: I don't bother adding 2nd-tier terms to deinstein_tempo
    # either
    
    # ********************************************************************
    # Now calculating Earth-Sun separation vector, as needed
    # for Shanp.piro delay calculation.
    # --------------------------------------------------------------------
    
    sunPos = np.array([ephemSpos[0][ientryS], ephemSpos[1][ientryS], ephemSpos[2][ientryS]]) 
    sunVel = np.array([ephemSvel[0][ientryS], ephemSvel[1][ientryS], ephemSvel[2][ientryS]])
    sunAcc = np.array([ephemSacc[0][ientryS], ephemSacc[1][ientryS], ephemSacc[2][ientryS]])
    
    sunPosNow = np.zeros([3,1])
    sunVelNow = np.zeros([3,1])
    
    earthse = np.zeros([3,1])
    earthdse = np.zeros([3,1])
    
    earthdrse = 0
    rse2 = earthdrse
    
    for j in range(3):
        sunPosNow[j] = sunPos[j] + sunVel[j]*tdiffS + 0.5*sunAcc[j]*tdiff2S
        sunVelNow[j] = sunVel[j] + sunAcc[j]*tdiffS
    
        earthse[j] = earthposNow[j] - sunPosNow[j]
        earthdse[j] = earthvelNow[j] - sunVelNow[j]
        rse2 = rse2 + earthse[j]*earthse[j]
        earthdrse = earthdrse + earthse[j] * earthdse[j]
    
    
    earthrse=np.sqrt(rse2)
    earthstruct = np.array([[earthposNow,earthvelNow,earthgmstRad], 
            [earthtzeA, earthzA, earththetaA],
            [earthdelpsi, earthdeleps, earthgastRad],
            [eartheinstein, earthdeinstein], 
            [earthse, earthdse, earthdrse, earthrse]])
        
    earthrse=np.sqrt(rse2)
    earthdrse = earthdrse/earthrse
    earthstruct = np.array([[earthposNow,earthvelNow,earthgmstRad], 
            [earthtzeA, earthzA, earththetaA],
            [earthdelpsi, earthdeleps, earthgastRad],
            [eartheinstein, earthdeinstein], 
            [earthse, earthdse, earthdrse, earthrse]])
    return earthstruct
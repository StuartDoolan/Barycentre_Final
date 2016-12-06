#script to run Barycentring and Greedy_Bary functions to create and test RB model
#and interpolation of this model

import numpy as np
import random as r
import Greedy_Bary_Functions as gbf
from init_barycentref import init_barycentre
from get_baryf import get_bary_in_loop

#import Eephem & Sephem data outside loop 
[Eephem, Sephem] = init_barycentre( 'earth00-19-DE405.dat', 'sun00-19-DE405.dat')
  
##set detector
detector = 'h1'

#sets array of time intervals of length wl from random acceptable starting point
#for one days worth of data
wl = 1024
day = 86400
start = r.randint(630720013, 1261872018-365*day)
tGPS = np.linspace(start, start+365*day, wl)
dt = tGPS[1]-tGPS[0]

#sets training set size and initialises source arrays
tssize = 1000
sourcealpha = np.zeros(tssize)
sourcedelta = np.zeros(tssize)

for i in range(tssize):
    
    #randomise RA and Dec between 0,2pi and -pi/2, pi/2 respectively
    sourcealpha[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
source = sourcealpha, sourcedelta

#Creates Reduced Basis by calculating emitdt values for source using greedy_bary
RB=gbf.greedy_bary(detector, tssize, wl, source, tGPS, Eephem, Sephem, dt)

##Creating fit of emitdt as linear combination of RB and testing against new Edt 

#number of 'check' training sets
csize = 100

#initialise source, new emitdt vectors
sourcealpha2 = np.zeros(csize)
sourcedelta2 = np.zeros(csize)
Edt = np.zeros((csize,wl))

#number of basis vectors in RB, used to determine number of points in each Edt
#vector used to create solutions to matrix equation
no_v = np.shape(RB)[0]

for i in range(csize):

    #define new sources
    sourcealpha2[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta2[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
    
    #apply get_bary_in_loop to find emitdt data for these sources
    emit = get_bary_in_loop(tGPS, detector,[sourcealpha2[i], sourcedelta2[i]], Eephem, Sephem)
    [emitdt2, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
    Edt[i]= np.reshape(emitdt2, wl)

   
#Creates model emitdt values from RB 
#returns fitted emitdt values, coefficients of matrix equation,index of tGPS times
#used, Edt values used and ndarray of specific points in RB used
newEdt, A, p, x, pRB = gbf.lin_com(csize, no_v, RB, Edt)

diff=Edt-newEdt
print('the mean difference is ' + str(np.mean(diff)))+'s'
print('the largest difference is ' + str(np.amax(abs(diff))))+'s'

##Interpolation 

# determines fineness of time vector used in evaluating interpolation
step = dt//10

#performs spline interpolation on newEdt, tGPS. For each step between two points 
#in tGPS returns fitted values. Returns ndarray of emitdt estimates and time vector
inEdt, newtime = gbf.interpnewEdt(newEdt, tGPS, step, wl, csize)


#initialise new emitdt values for newtime points to test interpolation accuracy
Exdt = np.zeros((csize,np.size(newtime)))

for i in range(csize):

    #apply get_bary to find emitdt data for these sources and newtime array
    emit = get_bary_in_loop(newtime, detector,[sourcealpha2[i], sourcedelta2[i]], Eephem, Sephem)
    [emitdt2, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
    Exdt[i]= np.reshape(emitdt2, np.size(newtime))

#test spline estimated points along tGPS against calculated values
diff2 = Exdt-inEdt
print('the mean difference is ' + str(np.mean(diff2)))+'s'
print('the largest difference is ' + str(np.amax(abs(diff2))))+'s'
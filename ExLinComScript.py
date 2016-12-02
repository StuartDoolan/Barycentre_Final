#script to run 

import numpy as np
import random as r
import Greedy_Bary_Functions as gbf
from init_barycentref import init_barycentre
from get_baryf import get_bary_in_loop

#import Eephem & Sephem data outside loop 
[Eephem, Sephem] = init_barycentre( 'earth00-19-DE405.dat', 'sun00-19-DE405.dat')
  
##set detector
detector = 'h1'


#sets vector of time intervals of length wl from random acceptable starting point for one days worth of data
wl = 1024
day = 86400
start = r.randint(630720013, 1261872018-365*day)
tGPS = np.linspace(start, start+365*day, wl)
dt = tGPS[1]-tGPS[0]

#sets training set size and initialises source arrays
tssize = 1000
sourcealpha = np.zeros(tssize)
sourcedelta = np.zeros(tssize)
source = sourcealpha, sourcedelta

for i in range(tssize):
    
    #randomise RA and Dec between 0,2pi and -pi/2, pi/2 respectively
    sourcealpha[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
source = sourcealpha, sourcedelta

#Creates Reduced Basis by calculating emitdt values for source using greedy_bary
RB=gbf.greedy_bary(detector, tssize, wl, source, tGPS, Eephem, Sephem, dt)

##Creating fit of emitdt as linear combination and testing against new Edt values

#number of 'check' training sets
csize = 100

#initialise source, new emitdt vectors
sourcealpha2 = np.zeros(csize)
sourcedelta2 = np.zeros(csize)
Edt = np.zeros((csize,wl))

#number of basis vectors in RB, used to determine number of points in each Edt
#vector used to create 
no_v = np.shape(RB)[0]

for i in range(csize):

    #define new sources
    sourcealpha2[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta2[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
    source = np.array([sourcealpha2, sourcedelta2 ])
    
    #apply get_bary_in_loop to find emitdt data for these sources
    emit = get_bary_in_loop(tGPS, detector,[sourcealpha2[i], sourcedelta2[i]], Eephem, Sephem)
    [emitdt2, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
    Edt[i]= np.reshape(emitdt2, wl)
    
#Creates model emitdt values from RB 
newEdt, AB, p, x, tRB = gbf.lin_com(csize, no_v, RB, Edt)

diff=Edt-newEdt
print('the mean difference is ' + str(np.mean(diff)))
print('the maximum difference is ' + str(np.amax(abs(diff))))

##Interpolation 

# determines fineness of time vector used in linear fit
step = dt//10

#performs interpolation on newEdt, tGPS. For each step between two points in tGPS
#returns linearly fitted values. Errors are pretty dependent on wl & tGPS
#Returns ndarray of linear emitdt estimates
inEdt = gbf.interpnewEdt(newEdt, tGPS, step, wl)

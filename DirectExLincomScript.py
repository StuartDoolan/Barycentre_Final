#script to run Barycentring functions to then directly create and test RB model
#and interpolation of this model

import numpy as np
import random as r
import greedy
from init_barycentref import init_barycentre
from get_baryf import get_bary_in_loop
from scipy import interpolate

#import Eephem & Sephem data outside loop 
[Eephem, Sephem] = init_barycentre( 'earth00-19-DE405.dat', 'sun00-19-DE405.dat')
  
#set detector
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
source = sourcealpha, sourcedelta

#initialises training sets
TS = np.zeros((tssize, wl))
for i in range(tssize):
    
    #randomise RA and Dec between 0,2pi and -pi/2, pi/2 respectively
    sourcealpha[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
    
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




#no of vectors in RB 
no_v= np.shape(RB)[0]

##Creating fit of emitdt as linear combination of RB and testing against new Edt 

#number of 'check' training sets
csize = 100

#initialise source, new emitdt vectors
sourcealpha2 = np.zeros(csize)
sourcedelta2 = np.zeros(csize)
Edt = np.zeros((csize,wl))

#initialises arrays for use in creating fit of Edt
#array of Edt points corresponding to p locations
x= np.zeros((csize,no_v))
#array of RB points corresponding to p locations
pRB = np.zeros((no_v,no_v))
#Coefficient of matrix equation
A = np.zeros((csize,no_v))
#array of linearly spaced points along tGPS (not incl endpoints)
p = np.zeros(no_v)


#creates array of specific integer locations along Edt, missing end points
l = np.linspace(csize/(2*no_v),csize,no_v,False)
p = l.astype(int)
for i in range(no_v):
    for j in range(no_v):
        #creates vectors of reduced bases corresponding to time location of points used 
        pRB[i][j] = RB[i][p[j]]
    
#inverts pRB for use solving x = (A)pRB
C = np.linalg.inv((pRB))
##creates Edt to be tested with RB
for i in range(csize):

    #define location to test (arbitrary)
    sourcealpha2[i] = 2*np.pi*(r.uniform(0,1))
    sourcedelta2[i] = np.arccos(2*(r.uniform(0,1))-1)-np.pi/2
    source = np.array([sourcealpha2, sourcedelta2 ])
    
    #apply get barycentre and find emitdt data for these sources
    emit = get_bary_in_loop(tGPS, detector,[sourcealpha2[i], sourcedelta2[i]], Eephem, Sephem)
    [emitdt2, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
    Edt[i]= np.reshape(emitdt2, wl)
    
   
    
    #takes points from basis vectors at corresponding tGPS entries
    ## Want no_v points for each x[i] extracted from Edt[j]
    for j in range(no_v):
        x[i][j] = Edt[i][p[j]]    
        
#solves x=(A)(pRB) by A = (x)inv(pRB)
A = np.dot(x,C)

#Create fit of Edt data
newEdt=np.dot(A,RB)

#check residuals 
resids=Edt-newEdt
print('the mean difference is ' + str(np.mean(resids)))
print('the largest difference is ' + str(np.max(resids)))


###Interpolation

#creates time vector finer than tGPS but with same period
newtime = np.arange(tGPS[0], tGPS[wl-1], dt//10)

#initialise set of emitdt vectors for midwy between existing tGPS points
Exdt = np.zeros((csize,np.size(newtime)))

#calculate specific emitdt values for new time vectors at particular source
for i in range(csize):


    
    #apply get barycentre and find emitdt data for these sources
    emit = get_bary_in_loop(newtime, detector,[sourcealpha2[i], sourcedelta2[i]], Eephem, Sephem)
    [emitdt2, emitte, emitdd, emitR, emitER, emitE, emitS] = emit
    Exdt[i]= np.reshape(emitdt2, np.size(newtime))


#initialise nd array for spline estimated emitdt values
interEdt = np.zeros((csize, np.size(newtime)))
for i in range(csize):
    #finds spline represention of the tGPS, newEdt curve
    tck = interpolate.splrep(tGPS, newEdt[i], s=0)
    #evaluates spline for newEdt vector and assigns to interEdt
    interEdt[i] = interpolate.splev(newtime, tck, der=0)

#calculates residuals by comparing midpoints between tGPS points
diff = Exdt[i]-interEdt[i]    
#display mean and largest residuals 
print('the mean difference is ' + str(np.mean(diff)))
print('the largest difference is ' + str(np.amax(abs(diff))))
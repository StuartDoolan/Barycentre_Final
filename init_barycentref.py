import numpy as np

def init_barycentre(efile, sfile):
    """
    function [ephemE, ephemS] = init_barycenter(efile, sfile)

    This function takes in the filenames of a file containing the Earth
    ephemeris (efile) and Sun ephemeris (sfile) in the format of those within
    LAL e.g. earth00-19-DE405.dat, sun00-19-DE405.dat. It outputs that data in a format
    usuable by the barycentring codes - positions, velocities and
    acceleration:
    ephemEpos - vector of x, y and z positions (light seconds)
    ephemEvel - vector of x, y and z velocities (light seconds/second)
    ephemEacc - vector of x, y and z accelerations (light seconds/second^2)
    
    ephemEgps - vector of GPS times of the entries
    ephemnEentries - number of entries in file
    ephemdEttable - times difference between entries (seconds)
    and equivalently for Sun data
    This function is copied from the LAL function LALInitBarycenter."""
   
    # read in file
    f = open(efile, 'r')
    
    #test succesfully opened file
    if False: 
        print('Error, could not open Earth ephemeris file')
        raise SystemExit()
   
   #read in data from file
    filecontents = f.readlines()
    f.close()
    
    
    # skip through header lines starting with '#'
    filestart = 0
    while 1:
        if filecontents[filestart][0] == '#':
            filestart += 1
        else:
            break
    
    # assign details of header to variables: initgps, no. of reading, no. of entries
    [Einitgps, Edttables, Eentries] = np.array([
    float((filecontents[filestart].split()[0]).strip()),
    float((filecontents[filestart].split()[1]).strip()), 
    int((filecontents[filestart].split()[2]).strip())])
    
    # create array for data (nentries rows by 10 columns)
    ephemdata = np.zeros((Eentries, 10)) 
    
    # read in the actual data
    for i in range(int(Eentries)):
        thisline = []
        for j in range(4):
            lines = filecontents[filestart+1+(i*4)+j].split()
            for val in lines:
                thisline.append(float(val.strip()))
        ephemdata[i,:] = np.array(thisline)
    
    #assign each variable
    [ephemEgps, xpos, ypos, zpos, velx, vely, velz, accx, accy, accz] = np.array([
    ephemdata[:,0], ephemdata[:,1], ephemdata[:,2], ephemdata[:,3], ephemdata[:,4],
    ephemdata[:,5], ephemdata[:,6], ephemdata[:,7], ephemdata[:,8], ephemdata[:,9]])
    ephemEpos = np.array([xpos, ypos, zpos])
    ephemEvel = np.array([velx, vely, velz])
    ephemEacc = np.array([accx, accy, accz])
    
    Eephem = np.array([ephemEgps, Edttables, Eentries, ephemEpos, ephemEvel, ephemEacc])
    
    ####### Now performing same actions on Sun file
    
    # read in file
    f = open(sfile, 'r')
    if False:
        print('Error, could not open Sun ephemeris file')
        raise SystemExit()
    filecontents = f.readlines()
    f.close()
    
    
    # skip through header lines starting with '#'
    filestart = 0
    while 1:
        if filecontents[filestart][0] == '#':
            filestart += 1
        else:
            break
    
    # find number of entries in the file
    [Sinitgps, Sdttables, Sentries] = np.array([
    float((filecontents[filestart].split()[0]).strip()),
    float((filecontents[filestart].split()[1]).strip()), 
    int((filecontents[filestart].split()[2]).strip())])
    # create array for data (nentries rows by 10 columns)
    ephemdata = np.zeros((Sentries, 10)) 
    
    # read in the actual data
    for i in range(int(Sentries)):
        thisline = []
        for j in range(4):
            lines = filecontents[filestart+1+(i*4)+j].split()
            for val in lines:
                thisline.append(float(val.strip()))
        ephemdata[i,:] = np.array(thisline)
    
    #assign each variable
    [ephemSgps, Sxpos, Sypos, Szpos, Svelx, Svely, Svelz, Saccx, Saccy, Saccz] = np.array([
    ephemdata[:,0], ephemdata[:,1], ephemdata[:,2], ephemdata[:,3], ephemdata[:,4],
    ephemdata[:,5], ephemdata[:,6], ephemdata[:,7], ephemdata[:,8], ephemdata[:,9]])
    
    ephemSpos = np.array([Sxpos, Sypos, Szpos])
    ephemSvel = np.array([Svelx, Svely, Svelz])
    ephemSacc = np.array([Saccx, Saccy, Saccz])
    
    ### Returns ndarrays Eephem & Sephem, Eephem[3][0] = xpos, for example
    Sephem = np.array([ephemSgps, Sdttables, Sentries, ephemSpos, ephemSvel, ephemSacc]) 
    return Eephem, Sephem

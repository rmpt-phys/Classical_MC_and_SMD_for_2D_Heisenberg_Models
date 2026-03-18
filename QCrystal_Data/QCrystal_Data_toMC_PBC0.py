import statistics as stat

import numpy as np

import sys

#....................
# Define system size:
    
NsRef = 1393 # 239, 1393, 8119, 47321, 275807
            
#======================================
# Function for computing the neighbors:

def get_bonds(Ns, zn, nbDist, xvec, yvec):

    #.............
    # Preparation:
        
    fc = 0.01 #( distance match tolerance )        

    r1 = nbDist * (1.0 - fc)
    r2 = nbDist * (1.0 + fc)
        
    bondList = np.zeros((Ns, zn), dtype = np.int32)
       
    #...........................
    # Execute loop calculations:
     
    for n in range(Ns):
        ###           
        x0 = xvec[n]
        y0 = yvec[n]
        
        i = 0 #( bondList start index )
        
        bondVec = np.zeros(zn, dtype = np.int32)
        
        for m in range(Ns):
            #--------------------------------#
            xp = xvec[m]
            yp = yvec[m]
            
            dist = np.sqrt( pow(xp - x0, 2) +
                            pow(yp - y0, 2) )
                    
            if ((dist > r1) and (dist < r2)):
                ###                
                bondVec[i] = m + 1; i = i + 1
            #--------------------------------#
            
        for j in range(zn):
            ###
            bondList[n][j] = bondVec[j]
    ###
    ###( n-loop END )
        
    return bondList       

#////////////////////////
# Begin main code now ...

print(" \n >> System size : " + str(NsRef) + "\n")
       
#========================
# Open file in read mode:

szTag = str(NsRef)

sdir = "no_PBC/" + szTag

fileTag = "/appxt_" + szTag

znumData = fileTag + "_znumber.dat"
bondData = fileTag + "_connect.dat"

siteData = fileTag + "_vertices.dat"

#==========================
# Read z-numbers from file:

inFile = sdir + znumData

with open(inFile, 'r') as zfile:
    ###
    lines = zfile.readlines()
    
    Ns = len(lines) #( number of sites )
    
    iNs = 1.0 / Ns  #( inverse Ns )

    if (Ns != NsRef):
        ###
        sys.exit("\n > Invalid data:\n\n" + inFile + "\n")
    else:
        #-------------------------------------#
        znum1 = np.zeros(Ns, dtype = np.int32)

        for n in range(Ns):
            ###
            znum1[n] = lines[n].split()[0]
        #-------------------------------------#
            
zmax1 = max(znum1)

#==========================
# Read bond-data from file:

inFile = sdir + bondData

with open(inFile, 'r') as bfile:
    ###
    lines = bfile.readlines()

    nrows = len(lines)

    ncols = len(lines[0].split())

    if ((nrows != Ns) or (ncols != zmax1)):
        ###
        sys.exit("\n > Invalid data:\n\n" + inFile + "\n")
    else:
        #----------------------------------------------#
        bonds = np.zeros((Ns, zmax1), dtype = np.int32)

        for n in range(Ns):
            ###
            for m in range(zmax1):
                ###
                bonds[n][m] = lines[n].split()[m]
        #----------------------------------------------#
                        
#==========================
# Read site-data from file:

inFile = sdir + siteData

with open(inFile, 'r') as sfile:
    ###
    lines = sfile.readlines()

    nrows = len(lines)

    ncols = len(lines[0].split())

    if ((nrows != Ns) or (ncols != 2)):
        ###
        sys.exit("\n > Invalid data:\n\n" + inFile + "\n")
    else:
        #-------------------------------------------#
        sites = np.zeros((Ns, 2), dtype = np.double)

        for n in range(Ns):
            ###
            sites[n][0] = lines[n].split()[0] # [ x ]
            sites[n][1] = lines[n].split()[1] # [ y ]
        #-------------------------------------------#
            
xvec = [pt[0] for pt in sites] #| List of x & y
yvec = [pt[1] for pt in sites] #| site positions;

#=======================
# Check lattice spacing:

lsList = []

for n in range(Ns):
    ###
    x0 = xvec[n]
    y0 = yvec[n]

    zn = znum1[n]   
        
    for m in range(zn):
        #----------------------------------------#
        nb = bonds[n][m] - 1
        
        stDist = np.sqrt( pow(xvec[nb] - x0, 2) +
                          pow(yvec[nb] - y0, 2) )
            
        if (stDist < 1.0): lsList.append(stDist)    
        #----------------------------------------#

lspc = stat.mean(lsList)

std_dev = stat.stdev(lsList)

#===========================================
# Make clock-ordered nearest neighbors list:
    
n1bonds = np.zeros((Ns, zmax1), dtype = np.int32)

Pi4 = np.pi / 4

for n in range(Ns):
    ###
    nlist = []
    
    x0 = xvec[n]
    y0 = yvec[n]
    
    zn = znum1[n]
    
    for m in range(zn):
        #-------------------------#
        k = bonds[n][m]
        
        nb = k - 1
        
        xp = xvec[nb] - x0
        yp = yvec[nb] - y0
        
        theta = np.arctan2(yp, xp)
        
        t0 = round(theta / Pi4) + 9
        
        if (t0 > 8): 
            ##
            tnum = t0 - 9 
        else:
            tnum = t0 - 1
                    
        nlist.append([k, tnum])
        #-------------------------#
            
    for m in range(zn):
        ##
        m0 = nlist[m][1]   
        
        n1bonds[n][m0] = nlist[m][0]        
    ###
    ###( n-loop END)
        
#===================================
# Find nearest neighbors (2nd & 5th):

nbDist2 = np.sqrt(2.0)

nbDist5 = np.sqrt(2.0 / (1.0 - 0.5 * nbDist2))

sys.stdout.write(" >> Finding neighbors ...")

n2bonds = np.zeros((Ns, zmax1), dtype = np.int32) 
n5bonds = np.zeros((Ns, zmax1), dtype = np.int32)

n2bonds = get_bonds(Ns, zmax1, nbDist2, xvec, yvec)
n5bonds = get_bonds(Ns, zmax1, nbDist5, xvec, yvec)

sys.stdout.write(" Done!\n\n")

zmax2 = 0 # maximum number
zmax5 = 0 # of neighbors;

for n in range(Ns):
    ###
    isum2 = 0
    isum5 = 0
    
    for i in range(zmax1):
        ##-------------------------------#               
        nb2 = n2bonds[n][i]
        nb5 = n5bonds[n][i]
        
        if (nb2 > 0): isum2 = isum2 + 1
        if (nb5 > 0): isum5 = isum5 + 1
        
        if (isum2 > zmax2): zmax2 = isum2
        if (isum5 > zmax5): zmax5 = isum5
        ##-------------------------------#         
        
#========================================
# Get z-value for neighbors of two kinds:

znum2 = np.full(Ns, zmax2, dtype = np.int32)
znum5 = np.full(Ns, zmax5, dtype = np.int32)
   
for n in range(Ns):
    ###
    ikey = 1
    jkey = 1
        
    for i in range(zmax2):
        #-----------------------------#
        nb = n2bonds[n][i]                                       
        
        if ((nb == 0) and (ikey == 1)):
            ###
            znum2[n] = i; ikey = 0
        #-----------------------------#

    for j in range(zmax5):
        #-----------------------------#
        nb = n5bonds[n][j]                                       
        
        if ((nb == 0) and (jkey == 1)):
            ###
            znum5[n] = j; jkey = 0
        #-----------------------------#

z1avg = stat.mean(znum1)
z2avg = stat.mean(znum2)
z5avg = stat.mean(znum5)

print(zmax1)
print(zmax2)
print(zmax5)

sys.exit("\n > Test over!\n")
        
#========================
# Set Neel configuration:
    
NeelConfig = np.zeros(Ns, dtype = np.int32)

marker = np.zeros(Ns, dtype = np.int32)    

stList = [Ns - 1] #( start with last site )

mlist = [] # Auxiliary list;

spin = 1 # Spin state (up / down);

for k in range(Ns):
    ###
    k = k - 1 #( fix site index )
    
    if (marker[k] == 0):
        ###
        stList = [ k ]
        
        work = True
        
        while (work):
            ###                 
            for n in stList:
                ###          
                if (NeelConfig[n] == 0.0):
                    #--------------------#
                    NeelConfig[n] = spin;             
                    
                    marker[n] = 1                
                    #--------------------#
        
            spin = (- 1) * spin
            
            mlist.clear()
                
            for n in stList:
                ###              
                for m in range(znum1[n]):
                    #-----------------------------------------#
                    nb = bonds[n][m] - 1 #( n-neighbor index )            
                    
                    sp = NeelConfig[nb] # Get neighbor spin;
                    
                    mk = marker[nb] # Get marker value;
                                
                    if ((sp == 0.0) and (mk == 0)): 
                        ###                
                        mlist.append(nb); marker[nb] = 1                
                    #-----------------------------------#    
                                                        
            if (len(mlist) == 0): work = False
            
            stList = mlist.copy()
            ###
            ###( while-END )
        ###
        ###( spin-check )
    ###
    ###( k-loop END )
        
#====================================
# Record binary-files for Monte-Carlo:        

qtag = "_AppxtSz(" + szTag + ")"

fname1 = "sitesList" + qtag + ".bin"
fname2 = "nborsList" + qtag + ".bin"
fname3 = "stateNeel" + qtag + ".bin"

with open(fname1, "wb") as sitesFile:
    ###
    rvec = np.zeros(2, dtype = np.double)
    
    for n in range(Ns):
        #-------------------------------#
        rvec[0] = xvec[n] # Lattice site
        rvec[1] = yvec[n] # coordinates;
        
        sitesFile.write(rvec.tobytes())
        #-------------------------------#
        
with open(fname2, "wb") as nborsFile:
    ###
    for n in range(Ns):
        #------------------------------------#
        for m in range(zmax1):
            ###
            nb1 = np.int32(n1bonds[n][m] - 1)
            
            nborsFile.write(nb1.tobytes())

        for m in range(zmax2):
            ###
            nb2 = np.int32(n2bonds[n][m] - 1)
            
            nborsFile.write(nb2.tobytes())

        
        for m in range(zmax5):
            ###
            nb5 = np.int32(n5bonds[n][m] - 1)
            
            nborsFile.write(nb5.tobytes())
        #------------------------------------#
        
with open(fname3, "wb") as nstate:
    ###   
    for n in range(Ns):
        #------------------------------------#
        nstate.write(NeelConfig[n].tobytes())
        #------------------------------------#

#==================================
# Record reference file data check:
        
nd = len(str(Ns))

fname = "zvalsList" + qtag + ".dat"

with open(fname, "w") as zvalsFile:
    ###
    for n in range(Ns):
        #-------------------------------#
        zn1 = np.int32(znum1[n])
        zn2 = np.int32(znum2[n])
        zn5 = np.int32(znum5[n])
            
        zvalsFile.write(f"{zn1:{nd}}  ")
        zvalsFile.write(f"{zn2:{nd}}  ")
        zvalsFile.write(f"{zn5:{nd}}\n")
        #-------------------------------#
        
#//////////////////////////////////////////////////////

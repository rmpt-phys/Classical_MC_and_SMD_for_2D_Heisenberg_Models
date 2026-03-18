import statistics as stat

import numpy as np

import sys

#....................
# Define system size:
    
NsRef = 1393 # 239, 1393, 8119, 47321, 275807
            
#======================================
# Function for computing the neighbors:

def get_bonds(nbDist, fc,
              Ns, NsPlus, zn,
              xvec, xvecPlus,
              yvec, yvecPlus):

    #.............
    # Preparation:              

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
        
        for m in range(NsPlus):
            #---------------------------------------#
            xp = xvecPlus[m]
            yp = yvecPlus[m]
            
            dist = np.sqrt( pow(xp - x0, 2) +
                            pow(yp - y0, 2) )
                    
            if ((dist > r1) and (dist < r2)):
                ###
                nb = (m % Ns) + 1

                if (i < 8): bondVec[i] = nb;
                
                i = i + 1 #( max. value = 8)
                            
            if (i > 8):
                ##
                sys.exit("\n\n Error: get_bonds;\n")            
            #---------------------------------------#
            
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

sdir = "with_PBC/" + szTag

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

    if ((nrows != Ns) or (ncols < 2)):
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

#==========================
# Find lattice center site:

rdistVec = []    

for n in range(Ns):
    ###               
    xp = xvec[n]
    yp = yvec[n]

    rdist = np.sqrt(pow(xp, 2) + pow(yp, 2))
    
    rdistVec.append(rdist)
    
rmin = min(rdistVec)

ncenter = rdistVec.index(rmin)  

#=============================
# Find lattice effective size:

Lsz = 0

rsmall = 1.0E-4

x0 = xvec[Ns - 1]
y0 = yvec[Ns - 1]

for n in range(Ns):
    ###
    x1 = xvec[n]
    y1 = yvec[n]

    if (abs(y1 - y0) <= rsmall):
        #----------------------------#
        xdist = x1 - x0       
        
        if (xdist > Lsz): Lsz = xdist
        #----------------------------#

pshift = Lsz + lspc;

#======================
# Extend lattice (PBC):
    
NsPlus = 9 * Ns    

svec = np.zeros((9, 2), dtype = np.float32)
    
rsites = np.zeros((NsPlus, 2), dtype = np.float32)

svec[1][0] = + pshift; svec[2][0] = - pshift
svec[1][1] = + 0.0000; svec[2][1] = - 0.0000

svec[3][0] = + 0.0000; svec[4][0] = - 0.0000
svec[3][1] = + pshift; svec[4][1] = - pshift

svec[5][0] = + pshift; svec[6][0] = - pshift
svec[5][1] = + pshift; svec[6][1] = - pshift

svec[7][0] = + pshift; svec[8][0] = - pshift
svec[7][1] = - pshift; svec[8][1] = + pshift

for m in range(9):
    ###
    for n in range(Ns):
        ###
        p = n + m * Ns
        
        rsites[p][0] = sites[n][0] + svec[m][0]
        rsites[p][1] = sites[n][1] + svec[m][1]

xvecPlus = [pt[0] for pt in rsites] #| x
yvecPlus = [pt[1] for pt in rsites] #| y

#===========================================
# Make clock-ordered nearest neighbors list:
    
n1bonds = np.zeros((Ns, zmax1), dtype = np.int32)

nbDist1 = lspc + 0.1 #( distance + error margin )

Pi4 = np.pi / 4

for n in range(Ns):
    ##
    nlist = []
    
    x0 = xvec[n]
    y0 = yvec[n]
    
    zn = znum1[n]
    
    for m in range(zn):
        ##
        k = bonds[n][m]
        
        nb = k - 1
                
        for i in range(9):
            ##
            p = i * Ns
                    
            xdel = xvecPlus[nb + p] - x0
            ydel = yvecPlus[nb + p] - y0

            dist = np.sqrt( pow(xdel, 2) +
                            pow(ydel, 2) )
            
            if (dist <= nbDist1):
                #-----------------------------#                
                theta = np.arctan2(ydel, xdel)
        
                t0 = round(theta / Pi4) + 9
                
                if (t0 > 8): 
                    ##
                    tnum = t0 - 9 
                else:
                    tnum = t0 - 1
                            
                nlist.append([k, tnum])
                #-----------------------------# 
            
    for m in range(zn):
        ##
        m0 = nlist[m][1]   
        
        n1bonds[n][m0] = nlist[m][0]        
    ###
    ###( n-loop END)

#================
# Check 'nbonds':
    
for n in range(Ns):
    ###
    cnt = 0
    
    for m in range(zmax1):
        ###
        nb = n1bonds[n][m] 
        
        if (nb > 0): cnt = cnt + 1

    if (cnt != znum1[n]): 
        ###    
        sys.exit(" Error: nbonds is ill-defined; \n")    
        
#====================================
# Find nearest neighbors (2nd & 3rd):

fc2 = 0.05 #( distance match tolerance )
fc3 = 0.03

nbDist2 = np.sqrt(2.0) * lspc

nbDist3 = np.sqrt(2.0 + np.sqrt(2.0)) * lspc

sys.stdout.write(" >> Finding neighbors ...")

n2bonds = np.zeros((Ns, zmax1), dtype = np.int32) 
n3bonds = np.zeros((Ns, zmax1), dtype = np.int32)

n2bonds = get_bonds(nbDist2, fc2, Ns, NsPlus, zmax1,
                    xvec, xvecPlus, yvec, yvecPlus)

n3bonds = get_bonds(nbDist3, fc3, Ns, NsPlus, zmax1,
                    xvec, xvecPlus, yvec, yvecPlus)

sys.stdout.write(" Done!\n\n")

zmax2 = 0 # maximum number
zmax3 = 0 # of neighbors;

for n in range(Ns):
    ###
    isum2 = 0
    isum3 = 0
    
    for i in range(zmax1):
        ##-------------------------------#               
        nb2 = n2bonds[n][i]
        nb3 = n3bonds[n][i]
        
        if (nb2 > 0): isum2 = isum2 + 1
        if (nb3 > 0): isum3 = isum3 + 1
        
        if (isum2 > zmax2): zmax2 = isum2
        if (isum3 > zmax3): zmax3 = isum3
        ##-------------------------------#         
        
#========================================
# Get z-value for neighbors of two kinds:

znum2 = np.full(Ns, zmax2, dtype = np.int32)
znum3 = np.full(Ns, zmax3, dtype = np.int32)
   
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

    for j in range(zmax3):
        #-----------------------------#
        nb = n3bonds[n][j]
        
        if ((nb == 0) and (jkey == 1)):
            ###
            znum3[n] = j; jkey = 0
        #-----------------------------#

z1avg = stat.mean(znum1)
z2avg = stat.mean(znum2)
z3avg = stat.mean(znum3)

print(z1avg)
print(z2avg)
print(z3avg)

sys.exit("\n\n Done!\n") #XXXXXX     
        
#========================
# Set Neel configuration:
    
NeelConfig = np.zeros(Ns, dtype = np.int32)

marker = np.zeros(Ns, dtype = np.int32)    

stList = [Ns - 1] #( start with last site )

mlist = [] # Auxiliary list;

spin = 1 # Spin state (up / down);

dmax = lspc + 0.1 # Max. NN distance;

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
                    #-----------------------------------------------#
                    nb = bonds[n][m] - 1 #( n-neighbor index )            
                    
                    sp = NeelConfig[nb] # Get neighbor spin;
                    
                    mk = marker[nb] # Get marker value;
                    
                    dist = np.sqrt( pow(xvec[n] - xvec[nb], 2) +
                                    pow(yvec[n] - yvec[nb], 2) )
                                
                    if ((sp == 0.0) and (mk == 0) and (dist < dmax)): 
                        ###                
                        mlist.append(nb); marker[nb] = 1                
                    #-----------------------------------------------#    
                                                        
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

rszFac = 1.0 / lspc

with open(fname1, "wb") as sitesFile:
    ###
    rvec = np.zeros(2, dtype = np.double)
    
    for n in range(Ns):
        #----------------------------------------#
        rvec[0] = rszFac * (xvec[n] + 0.5 * Lsz) # Lattice site
        rvec[1] = rszFac * (yvec[n] + 0.5 * Lsz) # coordinates;
        
        sitesFile.write(rvec.tobytes())
        #------------------------------#
        
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

        
        for m in range(zmax3):
            ###
            nb3 = np.int32(n3bonds[n][m] - 1)
            
            nborsFile.write(nb3.tobytes())
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
        zn3 = np.int32(znum3[n])
            
        zvalsFile.write(f"{zn1:{nd}}  ")
        zvalsFile.write(f"{zn2:{nd}}  ")
        zvalsFile.write(f"{zn3:{nd}}\n")
        #-------------------------------#
        
#//////////////////////////////////////////////////////

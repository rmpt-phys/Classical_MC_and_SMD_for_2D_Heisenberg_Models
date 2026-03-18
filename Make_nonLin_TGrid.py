"""
 This code produces a nonlinear T-grid by solving
 a system of coupled harmonic oscillators, dT cor-
 responds to the spacing between each oscillator,
 which is controlled by the dT-parameters below;

 The output is a binary file 'Temp_List_TAG(X).bin',
 where TAG is a string (J1, J2 or extH), and X is a
 number; the user must move the file to the 'outBin'
 folder within the simulation directory and set the
 temperature grid parameter (MC_SIM) to 'custom';

"""

import numpy as np

import struct

#|======================
#| Grid main parameters:

npts = 48 # Number of T-grid points;

dT_crit_index = 32 # T-grid index where dT is critical;

dT_limit_factor = 5.00 # Controls max. T-grid spacing;

dT_start_factor = 5.00 # Sets initial grid spacing;

dT_final_factor = 0.01 # Sets final grid spacing;

#======================================#
def kfunction0(i, i0, sz, k1, k2, kmax):

    # -------------------------------
    # Return k-value based on inputs:

    kdiff1 = kmax - k1

    kdiff2 = kmax - k2

    alpha1 = kdiff1 / i0
    
    alpha2 = kdiff2 / (i0 - sz - 1)
    
    if i < i0 :
        return alpha1 * i + k1 #( Linear increase )
    else:
        return alpha2 * (i - i0) + kmax #( Linear decrease )
    
#=========================================#
def kfunction(i, i0, sz, k1, k2, kmax, Cd):

    # -------------------------------
    # Return k-value based on inputs:

    kdiff1 = kmax - k1

    kdiff2 = kmax - k2

    alpha1 = kdiff1 / i0
        
    if i < i0 :
        return alpha1 * i + k1 #( Linear increase )
    else:
        return kdiff2 * np.exp(Cd * (i - i0)) + k2 #( Exp. decrease ) 

#===================================================#
def solve_system(n, kstart, kfinal, kvalues, x1, x2):

    #----------------------
    # Tridiagonal matrix A:
    
    A = np.zeros((n, n))

    #-------------------------
    # Right-hand side vector b
    # for boundary conditions:
    
    b = np.zeros(n)

    #------------------------------
    # Fill the matrix A & vector b:
    
    for i in range(n):
        
        if i == 0:
            
            # First mass (fixed to x1):
            
            A[i, i] = kstart + kvalues[0]
            
            A[i, i + 1] = - kvalues[0]

            # Boundary condition for the first mass:
            
            b[i] = kstart * x1
            
        elif i == n - 1:
            
            # Last mass (fixed to x2):
            
            A[i, i] = kfinal + kvalues[n - 2]
            
            A[i, i - 1] = - kvalues[n - 2]

            # Boundary condition for the last mass:
            
            b[i] = kfinal * x2
            
        else:
            
            # Interior masses:

            k0 = kvalues[i - 1]
            
            k1 = kvalues[i]
            
            A[i, i - 1] = - k0
            
            A[i, i + 0] = + k0 + k1
            
            A[i, i + 1] = - k1

    #--------------------------
    # Solve the system A x = b:
    
    xvec = np.linalg.solve(A, b)
    
    return xvec

#|========================
#| Open files for reading:

InputFile1 = "Settings_MODEL.txt"

InputFile2 = "Settings_MC_SIM.txt"

with open(InputFile1, 'r') as file:
    ##
    lines = file.readlines()

hval = float(lines[7].strip())
tmp1 = float(lines[8].strip())
tmp2 = float(lines[9].strip())

with open(InputFile2, 'r') as file:
    ##
    lines = file.readlines()  

file_tag = float(lines[5].strip())

#|===========================
#| Build kvalues dynamically:

kmax = dT_limit_factor # Maximum k-value;

imax = dT_crit_index # T-grid point where k = kmax;

kstart = dT_start_factor # Initial k-value;

kfinal = dT_final_factor # Wall/last k-value;

kvalues = [] # k-values vector/list;

coeff = (- 0.25) # Decay coefficient;

n = npts - 2 # Inner points/masses;

for i in range(n):

    kval = kfunction(i + 1, imax, n,
                     kstart, kfinal, kmax, coeff);
    
    kvalues.append(kval)
                
#|================================
#| Solve for the static positions:

if kfinal != kvalues[n-1]: kfinal = kvalues[n-1]

tvalues = solve_system(n, kstart, kfinal, kvalues, tmp1, tmp2)

#|==================================
#| Open a file & write final result:

tvalues = [tmp1] + tvalues.tolist() + [tmp2]

kvalues = [kstart] + kvalues + [kfinal]

TGridData = "GridInfo.dat"

with open(TGridData, 'w') as file:
    
    for i in range(npts):

        tval = tvalues[i]

        kval = kvalues[i]
        
        file.write(f"{tval:8.4f}  {kval:8.4f}\n")

#|===============================
#| Write x-grid in a binary file:

htag = hval #( tag number )

smag = f"{htag:.3f}"

if file_tag == "Htag":
    ##
    str0 = f"_extH({smag})"

elif file_tag == "J2tag":
    ##
    str0 = f"_J2({smag})"

else:
    ##
    str0 = f"_J1({smag})"

TGridBin = "Temp_List" + str0 + ".bin"

with open(TGridBin, "wb") as outGrid:
    
    for tval in tvalues:

        outGrid.write(struct.pack('d', tval))

#|======================
#| Print final messages:

tref = tvalues[imax]

print("\n Grid range: [", f"{tmp1:4.2f} , {tmp2:4.2f}", "]")

print("\n Number of grid-points:", npts)

print("\n Turning temperature:", f"{tref:4.2f}")

print("\n Grid-points & generators recorded:", TGridData, "\n")

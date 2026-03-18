"""
 This code produces a custom T-grid by assigning user
 custom T-values (set by the list of values 'tvalues'
 below) between the two extreme temperatures defined
 in the MODEL input file;

 The output is a binary file 'Temp_List_TAG(X).bin',
 where TAG is a string (J1, J2 or extH), and X is a
 number; the user must move the file to the 'outBin'
 folder within the simulation directory and set the
 temperature grid parameter (MC_SIM) to 'custom';

"""

import numpy as np

import struct

#|=========================
#| Define tvalues manually:

tvalues = [ 0.1, 0.5, 1.0,
            1.1, 1.2, 1.5 ]

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

#|==================================
#| Open a file & write final result:

tvalues = [tmp1] + tvalues + [tmp2]

TGridData = "GridInfo.dat"

npts = len(tvalues)

with open(TGridData, 'w') as file:
    
    for i in range(npts):

        tval = tvalues[i]
        
        file.write(f"{tval:8.4f}\n")

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

print("\n Grid range: [", f"{tmp1:4.2f} , {tmp2:4.2f}", "]")

print("\n Number of grid-points:", npts)

print("\n Grid-points & generators recorded:", TGridData, "\n")

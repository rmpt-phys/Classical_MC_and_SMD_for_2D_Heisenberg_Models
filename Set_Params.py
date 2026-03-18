#------------------------------------
# Change parameters & settings below: (see next section for a guide)

parms = []

parms.append({

    "MODEL_PARAMS" : {
        "0)  Lattice_Linear_Size" : 12,
        "1)  J1_Coupling_Value"   : 1.000,
        "2)  J2_Coupling_Value"   : 0.600,
        "3)  J3|J5_Coupling_Val"  : 0.000,
        "4)  J2_ALM_dWave_Delta"  : 0.200,
        "5)  J1_SzSz_XXZ_Factor"  : 0.900,
        "6)  J2_SzSz_XXZ_Factor"  : 0.900,
        "7)  Ext_Magnetic_Field"  : 0.000,
        "8)  Small_Temperature"   : 0.010,
        "9)  Large_Temperature"   : 0.100,
        "10) Lattice_Geometry"    : "lieb"
    },

    "MC_PARAMS" : {
        "0) Thermal_Sweeps"    : 4000,
        "1) Measure_Sweeps"    : 8000,
        "2) Initial_State"     : "high_T",
        "3) Temperature_Grid"  : "linear",
        "4) Output_Files_Tag"  : "J2tag",
        "5) Field_Record_Step" : 1000,
        "6) Video_Record_Step" : 0,
        "7) Lattice_Analysis"  : 0
    },

    "SMD_PARAMS" : {
        "0) Time_Evolution_Steps"    : 2000,
        "1) Temperature_List_Size"   : 12,
        "2) Input_Temperature_Index" : 0,
        "3) Temporal_Discretization" : 0.05,
        "4) Temporal_Window_Form"    : "lancz",
        "5) Field_Evolution_Video"   : 0,
        "6) Structure_Factor_Video"  : 0
    },

    "DISORDER_CONFIG" : {
        "0) Impurity_Disorder_Ratio"  : 0.0,
        "1) New_J1_Impurity_Crystal"  : 1.0,
        "2) New_J2_Impurity_Crystal"  : 1.0,
        "3) New_JX_Impurity_Crystal"  : 0.0,
        "4) New_J1_Impurity_Impurity" : 1.0,
        "5) New_J2_Impurity_Impurity" : 0.0,
        "6) New_JX_Impurity_Impurity" : 0.0
    },

    "CODE_EXEC" : {
        "0)  Parallel_Tempering"    : 0,
        "1)  PT_Grid_Adapt_Mode"    : 0,
        "2)  PT_Replica_Tracking"   : 0,
        "3)  Temp_Annealing_Mode"   : 0,
        "4)  Enable_FFTW_Library"   : 0,
        "5)  Run_SMD_For_1Sample"   : 0,
        "6)  Disable_Lattice_PBC"   : 0,
        "7)  Simulate_Ising_Model"  : 0,
        "8)  MC_For_Quasi_Crystals" : 0,
        "9)  MC_Record_SMD_Samples" : 1,
        "10) MC_Compute_Static_SFs" : 1,
        "11) QC_Force_Stripe_Confs" : 0,
        "12) QC_Set_Null_Ref_State" : 0,
        "13) QC_Assume_Neels_Phase" : 0,
        "14) QC_Remove_Lattice_PBC" : 0,
        "15) QC_Plot_Ord_Parameter" : 0
    }

})

###-----------------------------------------
### Instructions for using this Python-code:
"""
 > Step 1: change parameters & settings for both Monte-Carlo (MC)
           and Semiclassical-molecular-dynamics (SMD) simulations;

 > Step 2: run this Python-code on a console with Intel OneAPI +
           MKL avaiable, image/video generation requires OpenCV;

 > Step 3: this code executes the shell-scripts MC/SMD_Prep.sh,
           the user input is required to set the simulation dir;

 > Step 4: in '../Word_Dir/HM_SIM_(SIM_NAME)', the input files
           can be edited again before executing the simulations
           with the custom scripts: 0Run_MC/SMD_SIM.sh nthreads;

 > About 'nthreads': number of MPI threads running in parallel,
   it must be equal to the number of temperature grid points;

 > About the model: Heisenberg (or Ising) Hamiltonian with mul-
   tiple exchange couplings and external magnetic field for a
   crystal or quasi-crystal (QC) 2-dimensional lattice system:

 H = J1 · sum_{<i,j>}     S[i] · S[j] | Nearest neighbors (NN);
   + J2 · sum_{<<i,j>>}   S[i] · S[j] | Next-NN (NNN);
   + JX · sum_{<<<i,j>>>} S[i] · S[j] | Further-NN;
   - hz · sum_i           S[i]_z      | Zeeman term;

 > Notations:

 S[i] = ( S[i]_x , S[i]_y , S[i]_z ) : spin vector at site i;

 <i,j>      :  NN site pairs (i,j) sum;
 <<i,j>>    : NNN site pairs (i,j) sum;
 <<<i,j>>>  : Same for pairs of 3rd or 5th (QC case) NN;

 J1, J2, JX : exchange coupling constants (*);
 hz (extH)  : external mag field applied along z-direction;

 *) JX = J3 (crystal case) or J5 (QC case); """

###----------------------------------------
### Quick guide for each set of parameters: (more details in ReadMe.txt)
"""
 > MODEL_PARAMS --> 11 parameters (Settings_HM_SYS.txt)

 Set Heisenberg/Ising model parameters like J1, J2
 and JX (this is J3, or J5 for QCs), delta[J2] for
 NNN d-wave type ALM coupling, NN & NNN Sz-Sz ani-
 sotropy coupling factors (XXZ model), external mag-
 netic field (Zeeman term | z-axis), the smallest &
 largest simulation temperatures, lattice size and
 geometry (square, triang, hexagn, kagome, lieb);

 > MC_PARAMS --> 8 parameters (Settings_MC_SIM.txt)

 Set Monte-Carlo (MC) simulation parameters such as
 the number of thermalization/measurements sweeps,
 initial spin-state (loww_T, high_T, zpolar, etc),
 temperature grid form (linear, nonlin, custom),
 and spin-configuration recording skip;

 > SMD_PARAMS --> 7 parameters (Settings_SMD_SIM.txt)

 Set semiclassical molecular simulation (SMD) para-
 meters like the number of time steps for evolution,
 temperature list size (temperature grid size), in-
 put temperature index (selects the sample file for
 a given T on the grid), temporal discretization, &
 temporal window form (gauss, lancz, nowin);

 DISORDER_CONFIG --> 7 parameters (Settings_Disorder.txt)

 Set disorder ratio and all six exchange couplings
 associated with the interaction between the crystal
 sites and impurity sites;

 CODE_EXEC --> 16 parameters (Settings_CodeExec.txt)

 Enable/disable simulation code features, so that the
 user does not need to compile the whole code again
 each time a different setting is needed for testing;

 1: enable  code feature;
 0: disable code feature; """

#--------------------------------------------------
# Author's comments to append to each config. file:

Comment1 = """
 Input data details:

 0) Lattice linear size (Lsz);
 1) J1-value (nearest neighbors | crystal-crystal);     
 2) J2-value (next-nearest-nbrs | crystal-crystal); 
 3) JX-value (further neighbors | crystal-crystal);
 4) J2_delta : ALM d-wave couplings (Lieb lattice);
 5) J1 Sz-Sz XXZ model coupling factor (zLamb1); (#
 6) J2 Sz-Sz XXZ model coupling factor (zLamb2); (#
 7) External magnetic field amplitude (extH);
 8) Small temperature value (temp1); 
 9) Large temperature value (temp2);

 10) Lattice geometry: 
     square, triang, hexagn, kagome, lieb, Aamman;

 (#) Control parameter (lambda):
 
 lambda != 1.0 --> XXZ model;
 lambda == 1.0 --> Heisenberg model;
 lambda <= 1.0 --> Favors coplanar spin fields;
 lambda >> 1.0 --> Ising model limit (approx.);
"""

Comment2 = """
 Input data details:

 0) Number of thermalization sweeps;   
 1) Number of measurements sweeps;     
 2) Initial spin-state (loww_T, high_T, zpolar, etc);
 3) Temperature grid form/function (linear, nonlin);
 4) Output files label string (Htag, J1tag, J2tag);
 5) Spin-configuration recording skip (for SMD code);

 6) MC-sampling video recording skip (set to 0 to disable);  (#
 7) Vision-mode feature (enable with options: on 1 | off 0); (#
 
 (#) These features require OpenCV;
"""

Comment3 = """
 Input data details:

 0) Number of time steps (SMD evolution);
 1) Temperature list size (MPI worldSz);
 2) Input temperature list position (0,1,2...);   
 3) Temporal discretization (see commentary below);
 4) Temporal window form (gauss, hannX, lancz, nowin);
 
 5) Make spins-lattice video (1-st temporal evolution);  |#
 6) Make static-SF freq. slices video (see usage below); |#

 Parameter 3 (dtm): dtm = pi / wmax;
 
 The latter must be small enough to ensure a reasonable
 energy conservation during the time evolution via RK4;
 
 tmax = ntm * dtm; (ntm is set by the parameter 0)

 dwf = 2.0 * wmax / ntm = 2.0 * pi / tmax;

 #--> These features require OpenCV, input (5) can
      be set to 2 to skip the SMD-code & read the
      spectral data from a previously rec. file;
"""

Comment4 = """
 Input data details:

 0) Disorder ratio (double <= 1.00);  
 1) ic : impurity-crystal  J1-value;
 2) ic : impurity-crystal  J2-value;  
 3) ic : impurity-crystal  JX-value;  
 4) ii : impurity-impurity J1-value;
 5) ii : impurity-impurity J2-value;  
 6) ii : impurity-impurity JX-value;
"""

code_exec_labels = [
    "ptON",
    "ptAdapt",
    "ptTrack",
    "ANNL_ON",
    "FFTW_ON",
    "RKCheck",
    "PBC_OFF",
    "IsiModel",
    "qcrystal",
    "with_recField",
    "with_DFTcodes",
    "force_Stripes",
    "refState_Zero",
    "qct_NeelPhase",
    "qct_RemovePBC",
    "rec_orderMaps"
]

#-------------------
# Writing functions:

def write_values(filename, values, comment):
    ##
    with open(filename, "w") as f:
        ##
        for v in values:
            #
            f.write(f"{v}\n")

        f.write(comment)

def write_key_and_value(filename, values, labels):
    ##
    with open(filename, "w") as f:
        ##
        for tag, val in zip(labels, values):
            #            
            f.write(f"{tag} {val}\n")

#---------------------------
# Recording input files now:

p = parms[0]

write_values("Settings_MODEL.txt", p["MODEL_PARAMS"].values(), Comment1)

write_values("Settings_MC_SIM.txt", p["MC_PARAMS"].values(), Comment2)

write_values("Settings_SMD_SIM.txt", p["SMD_PARAMS"].values(), Comment3)

write_values("Settings_Disorder.txt", p["DISORDER_CONFIG"].values(), Comment4)

write_key_and_value("Settings_CodeExec.txt", p["CODE_EXEC"].values(), code_exec_labels)

#----------------------------------
# Verify if all files were created:

import os

expected_files = [
    "Settings_MODEL.txt",
    "Settings_MC_SIM.txt",
    "Settings_SMD_SIM.txt",
    "Settings_Disorder.txt",
    "Settings_CodeExec.txt"
]

missing_files = [f for f in expected_files if not os.path.isfile(f)]

if missing_files:
    ##
    print("\n > ERROR: The following files were not created:\n")
    
    for f in missing_files:
        ##
        print(f"   - {f}")
        
    print("\n > Aborting...\n"); exit(1)

print("\n > All input files recorded and ready:\n")

for f in expected_files:
    ##
    print(f"   - {f}")

#----------------------------------------
# Finish job or proceed with compilation:

print("\n > Proceed with compilation?")
print("   [1] Compile MC code")
print("   [2] Compile SMD code")
print("   [3] Compile BOTH codes")
print("   [0] Skip (manual compilation)")

choice = input("\n > Enter your choice: ").strip()

work_path = "../Work_Dir"

if choice in ("1", "2", "3"):
    ##
    if not os.path.exists(work_path):
        ##
        os.makedirs(work_path)
    
        print("\n > Work directory created: '../Work_Dir'")

if choice == "1":
    ##
    print("\n > Compiling MC code...")
    
    os.system("bash MC_Prep.sh")

elif choice == "2":
    ##
    print("\n > Compiling SMD code...")
    
    os.system("bash SMD_Prep.sh")

elif choice == "3":
    ##
    print("\n > Compiling BOTH codes...")
    
    os.system("bash MC_Prep.sh" )
    os.system("bash SMD_Prep.sh")

elif choice == "0":
    ##
    print("\n > Skipping compilation...\n")
    
    print(" > Manual compilation with:\n")
    
    print("   - bash MC_Prep.sh" )
    print("   - bash SMD_Prep.sh")

else:
    print("\n > Invalid option. Skipping compilation...")

print(" ")

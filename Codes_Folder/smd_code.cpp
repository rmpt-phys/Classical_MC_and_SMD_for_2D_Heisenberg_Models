/* ---------------------------------------------------------------------------
   
   Author: Rafael M. P. Teixeira

   OrcID: https://orcid.org/0000-0001-7290-3573
   
   Email: rafael.mpt@gmail.com
   
   ......................
   About this repository: 
  
   This repository contains C++ and Python codes developed during a two-year
   postdoctoral research project focused on the investigation of spin excita-
   tions in disordered magnetic systems and the magnetic properties of quasi-
   crystals (QCs). The implementation combines Monte Carlo (MC) simulations
   with parallel tempering (parallelization with OpenMPI) and semiclassical
   molecular dynamics (SMD) to study J1–J2 Heisenberg and Ising models on
   two-dimensional lattices;

   For Heisenberg models, spin updates are carried out using the heat-bath
   algorithm, while for Ising models, the standard single-spin-flip Metropo-
   lis algorithm is employed. In the SMD framework , spin dynamics are nume-
   rically obtained by integrating the Heisenberg equations of motion in the
   classical limit , where they reduce to the LLG equations describing spin
   precession (without damping) in an effective magnetic field. The fourth-
   order Runge–Kutta (RK4) method is employed and supplemented by an energy-
   correction scheme, ensuring stable numerical evolution (see the published
   articles for more details and references);

   Available periodic geometries include square, triangular, Lieb, hexagonal,
   and Kagome lattices. Other geometries must be configured manually within
   the code or loaded at runtime, as in the case of the Ammann–Beenker qua-
   sicrystal (QC) approximants provided (see the directory QCrystal_Data);

   For periodic systems, a J3 exchange coupling is implemented through the pa-
   rameter JX. For the aforementioned QC approximants, this parameter instead
   corresponds to a J5 exchange coupling. An external (z-axis) magnetic field
   can be included by setting a finite value for the corresponding parameter;

   Interaction anisotropy can be introduced by setting the Sz–Sz coupling fac-
   tors to realize an XXZ model. A system with disorder due to lattice impuri-
   ties can be obtained by setting the disorder ratio (or fraction) parameter
   in one of the configuration files (if set to 0, the system is clean);

   For SMD simulations, MC–generated spin configurations (samples) recorded at
   temperatures below a certain threshold (defined within the code) during the
   measurement stage are required. A specific input configuration file then
   defines the target sample file;

   The main output of the SMD simulations is the averaged dynamical structure
   factor (SF). This quantity is recorded for several frequency slices & wave
   vectors within the first Brillouin zone, as well as along a predefined path
   for varying frequencies;

   Additionally, MC and SMD codes employ OpenCV functions to produce images of
   sampled spin configurations (including final configurations from both ther-
   malization and measurement stages), as well as videos showing system evolu-
   tion in MC time and real time. A lattice inspection feature is also imple-
   mented using OpenCV, allowing the user to interactively verify the neigh-
   bors of each lattice site; 

   Read the file 'README.md' for details about the codes, models and methods;

   -----------------------------------------------------------------------------
   
   Project Title: Excitações de spin em sistemas de spin desordenados

   Affiliation: Instituto de Física da Universidade de São Paulo
   
   Principal Investigator: Rafael Marques Paes Teixeira
   
   Project Supervisor: Eric de Castro e Andrade
   
   Funder: São Paulo Research Foundation (FAPESP)
   
   Funding opportunity number: 2023/06682-8
   
   Grant: https://bv.fapesp.br/pt/pesquisador/726791/rafael-marques-paes-teixeira/
   
   -----------------------------------------------------------------------------*/

#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <complex>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <cmath>

#include <gsl/gsl_statistics.h>
#include <gsl/gsl_rng.h>

#include <experimental/filesystem>
#include <regex>

#include <algorithm>
#include <initializer_list>

#include <cstring>
#include <cctype>
#include <locale>

#include <sys/stat.h>

#include <mpi.h>
#include <mkl.h>

#include <fftw3.h>

#if WITH_OPENCV == 1
///
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/photo.hpp>
#include <opencv2/core/mat.hpp>

using namespace cv;
///
#endif

using namespace std;
using namespace std::chrono;

namespace fs = experimental::filesystem;

#include "subrouts.cpp" // Subroutines for the code below;

///////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////

//=============================================================
// Main code | Temporal evolution via the 4th order Runge-Kutta
// --------- | method of Heisenberg model spin-configurations:

int main(int argc, char** argv)
{ 
  const int num_Settings = 16;
  
  string inputs0[num_Settings];

  string inputs1[11], inputs2[8], inputs3[7], inputs4[7];
   
  double rx, ry, rz, fcw, norm, normSum, rval;

  int n0, m0, ns, nx, ny, nz, ch0, inum;

  int flag, halt, signal, ncuts2, evoCnt;

  unsigned int i, j, k, n, m; //(#)
  
  /* In (#), single letter integers are 
     defined, these are always unsigned
     in this main code, only use them
     for counters in loops... */
  
  //=======================================
  // Initialization of the MPI environment:
    
  int wSize = 1;

  int wRank = 0;

  bool iAmRoot = true;

  const int mpi_ctag = 0;
    
  if (MPI_Init(&argc, &argv) == MPI_SUCCESS)
    {         
      MPI_Comm_size(MPI_COMM_WORLD, &wSize);
      MPI_Comm_rank(MPI_COMM_WORLD, &wRank);

      if (wRank != root)
	{
	  iAmRoot = false;
	}
      
      fcw = 1.0 / wSize; flag = 0;
    }
  else 
    { flag = 1; } // If MPI-Init. failed...
  
  MPI_Barrier(MPI_COMM_WORLD); 
  
  if (flag > 0){MPI_Finalize(); return 0;}

  /*--------------------------------
    Print code resources information */  
  
  if (iAmRoot)
    {
      cerr << "\n MPI env. initialized: ";

      cerr << wSize << " threads;\n";

#if WITH_OPENCV == 1
      ////  
      cerr << "\n OpenCV codes enabled;\n";
      ////
#endif      
      waitAndJump();
    }

  //=================================
  // Get code-execution settings and 
  // configure simulation parameters:

  int *codeSetList = new int[num_Settings];
  
  if (iAmRoot)
    {
      flag = 0;

      ifstream inputFile0(fset0);

      if (!inputFile0.is_open())
	{
	  cerr << "\n Error: "
	       << "Unable to open the input file(s);";
      
	  cerr << "\n" << endl; flag = 1;
	}
      else//( get code-exec settings )
	{
	  string keyName; int keyGate;
			    
	  for (i = 0; i < num_Settings; i++)
	    {
	      if (!(inputFile0 >> keyName >> keyGate) && flag == 0)
		{
		  cerr << "\n Error: invalid setting pair (" << i << ");\n";

		  cerr << "\n Expected format: <string> <int>";
		  
		  cerr << "\n\n"; flag = 1;
		}
	    }
	  
	  if (flag == 0)
	    {
	      int idx = 0;

	      inputFile0.seekg(0); //( rewind reading )

	      cerr << " Code-execution settings ...\n\n";

	      while (inputFile0 >> keyName >> keyGate)
		{
		  cerr << " > "
		       << setw(13) << left 
		       << keyName  << " = " << keyGate;

		  if (keyName != Code_Settings[idx])
		    {
		      cerr << " ( fix name : " << Code_Settings[idx] << " )";
		    }
		  
		  if ((keyGate == 0) || (keyGate == 1) )
		    {
		      codeSetList[idx] = keyGate;
		    }
		  else
		    { cerr << " ( X )"; flag = 1; }

		  idx++; cerr << endl;
		}

	      if (flag > 0){ cerr << "\n Error: invalid code-exec setting!\n\n"; }
	    }
	  
	  inputFile0.close();
	}
    }
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);    
  
  if (flag > 0)
    {
      MPI_Finalize(); return 0;
    }
  else//( configure code-execution )
    {
      MPI_Bcast(codeSetList,
		num_Settings,
		MPI_INT, root,
		MPI_COMM_WORLD);

      /*............................*/
       
      ptON = codeSetList[0];

      ptAdapt = codeSetList[1];

      ptTrack = codeSetList[2];

      /*............................*/

      ANNL_ON = codeSetList[3];

      FFTW_ON = codeSetList[4];

      RKCheck = codeSetList[5];

      PBC_OFF = codeSetList[6];

      /*............................*/

      IsiModel = codeSetList[7];

      qcrystal = codeSetList[8];

      /*............................*/

      with_recField = codeSetList[9];
      
      with_DFTcodes = codeSetList[10];

      force_Stripes = codeSetList[11];
      
      refState_Zero = codeSetList[12];

      qct_NeelPhase = codeSetList[13];

      qct_RemovePBC = codeSetList[14];

      rec_orderMaps = codeSetList[15];
    }

  delete[] codeSetList;
  
  if (wSize > 1)
    {
      pcMode = true;
    }
  else
    { pcMode = false; }

  if ((RKCheck) && (pcMode))
    {
      flag = 1;

      if (iAmRoot) // Root reports error...
	{
	  cerr << "\n RKCheck requires 1-thread execution!\n";

	  waitAndJump();
	}  
    }

  if (iAmRoot)     // FFTW codes and padded-lattice DFTs are,
    {              // not being used, the DFTs are now execu-
      if (FFTW_ON) // ted with Intel's MKL-Dfti procedure;
	{   		    
	  cerr << "\n Error: please, disable FFTW_ON option to"
	       << "\n ...... avoid using deprecated subrouts!!";
      
	  cerr << "\n\n"; flag = 1;
	}

      if (qcrystal)
	{
	  if ((qct_RemovePBC) || (!PBC_OFF))
	    {
	      cerr << "\n Error: as qct_RemovePBC is true,"
		   << "\n ...... PBC_OFF must be true too!";

	      cerr << "\n\n"; flag = 1;
	    }
	}
      
      if ((RKCheck) && (pcMode))
	{
	  cerr << "\n RKCheck requires 1-thread execution!";

	  cerr << "\n\n"; flag = 1;
	}	    
    }///[ Root code options check ]
 
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  //========================================
  // Initialize C "random" number generator:

  int *randVec, irand, seed;

  double drand1, drand2;

  if (pcMode)
    { 
      if (iAmRoot)
	{
	  srand(time(NULL));

	  randVec = new int[wSize];
	  	  
	  for (i = 0; i < wSize; i++)
	    {
	      randVec[i] = rand(); // [0, RAND_MAX];
	    }
	}

      MPI_Scatter(randVec, 1, MPI_INT, &irand, 1,
		  MPI_INT, root, MPI_COMM_WORLD);

      if (iAmRoot){ delete[] randVec; }
    }
  else//( single thread )
    {
      srand(time(NULL));

      irand = rand();
    }
   
  //=========================================
  // Initialize dSFMT generator with "irand":
  
  seed = abs(irand * (wRank + 1));
  
  dSFMT_init(seed);

  /*..................
    Record seed vector */

  int *seedVec;

  if (iAmRoot){ seedVec = new int[wSize]; }
  
  if (pcMode)
    {
      MPI_Gather(&seed, 1, MPI_INT, seedVec, 1,
		 MPI_INT, root, MPI_COMM_WORLD);
    }
  
  if (iAmRoot)
    {
      ofstream seedFile("seedsRK.dat", ios::app | ios::out);
      
      for (i = 0; i < wSize; i++)
	{
	  seedFile << i << X2 << seedVec[i] << endl;
	}

      seedFile << endl;

      seedFile.close();

      delete[] seedVec;
    }
  
  /*........................ | dSFMT_sample.dat
    Test the dSFMT generator | sphereX.dat */

  if (iAmRoot)
    {
      ch0 = 0; // Swicth (1/0);

      if (ch0 == 1){dSMT_test();}
    }
      
  MPI_Barrier(MPI_COMM_WORLD);
  
  //====================================
  // Initialization (check input files): ROOT process
 
  if (iAmRoot)
    {     
      flag = 0;

      ifstream inputFile1(fset1); // Code input 
      ifstream inputFile2(fset2); // parameters...
      ifstream inputFile3(fset3); // (see subrouts)
      ifstream inputFile4(fset4);
      
      if ( (!inputFile1.is_open()) ||
	   (!inputFile2.is_open()) ||
	   (!inputFile3.is_open()) ||
	   (!inputFile4.is_open()) )
	{
	  cerr << "\n Error: "
	       << "Unable to open the input file(s);";
      
	  cerr << "\n" << endl; flag = 1;
	}

      if (flag == 0) // BARRIER_START
	{
	  //--------------------------------
	  // Get input arguments from files:
	    
	  for (i = 0; i < 11; ++i)
	    {
	      if (!(inputFile1 >> inputs1[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (MODEL);";
	    
		  cerr << endl; flag = 1;
		}
	    }

	  for (i = 0; i < 8; ++i)
	    {
	      if (!(inputFile2 >> inputs2[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (MC_SIM);";
	    
		  cerr << endl; flag = 2;
		}
	    }

	  for (i = 0; i < 7; ++i)
	    {
	      if (!(inputFile3 >> inputs3[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (SMD_SIM);";
	    
		  cerr << endl; flag = 3;
		}
	    }

	  for (i = 0; i < 7; ++i)
	    {
	      if (!(inputFile4 >> inputs4[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (Disorder);";
	    
		  cerr << endl; flag = 4;
		}
	    }
  
	  if (flag == 1)
	    {
	      cout << "\n Input data 1 details:\n\n";

	      cout << "  0) Lattice size;                       \n"
		   << "  1) J1-value (nearest neighbors);       \n" 
		   << "  2) J2-value (next-nearest neighbors);  \n"
		   << "  3) JX-value (distant neighbors: J3/J5);\n"
		   << "  4) J2_delta (J2 asymmetry / Lieb geom);\n"
		   << "  5) J1 Sz-Sz XXZ model factor (zLamb1); \n"
		   << "  6) J2 Sz-Sz XXZ model factor (zLamb2); \n"
		   << "  7) External magnetic field amplitude;  \n"
		   << "  8) Small temperature value (Temp1);    \n"
		   << "  9) Large temperature value (Temp2);    \n"
		   << " 10) Geometry: square, triang, (...);    \n";

	      cout << endl;
	    }

	  if (flag == 2)
	    {
	      cout << "\n Input data 2 details:\n\n";

	      cout << " 0) Number of thermalization sweeps;   \n"
		   << " 1) Number of measurements sweeps;     \n"
		   << " 2) Initial spin-state (lowT or highT);\n"
		   << " 3) Temp. grid form (linear, nonlin);  \n"
		   << " 4) Output label string: Htag or JXtag;\n"
		   << " 5) Spin-configuration recording skip; \n"
		   << " 6) MC-sampling video recording skip;  \n"
		   << " 7) Vision-mode feature (on 1 ; off 0);\n";
	      
	      cout << endl;
	    }

	  if (flag == 3)
	    {
	      cout << "\n Input data 3 details:\n\n";

	      cout << " 0) Number of time steps (RK evolution);      \n"
		   << " 1) Temperature list size (MPI worldSz);      \n"
		   << " 2) Input temp. list position (0,1,2,...);    \n"
		   << " 3) Temporal evolution discretization (dtm);  \n"
		   << " 4) Temporal window form (gauss, hann1, ...); \n"
		   << " 5) Make spins-lattice video (1-st evolution);\n"
		   << " 6) Make spectral-form video (static SFactor);\n";
	      
	      cout << endl;
	    }
	  
	  if (flag == 4)
	    {
	      cout << "\n Input data 4 details:\n\n";

	      cout << " 0) Disorder ratio (real/double number < 1);\n"  
		   << " 1) ic : impurity-crystal  J1-value;\n"
		   << " 2) ic : impurity-crystal  J2-value;\n"
		   << " 3) ic : impurity-crystal  J3-value;\n"
		   << " 4) ii : impurity-impurity J1-value;\n"
		   << " 5) ii : impurity-impurity J2-value;\n"
		   << " 6) ii : impurity-impurity J3-value;\n";
	      
	      cout << endl;
	    }

	  inputFile1.close();
	  inputFile2.close();
	  inputFile3.close();
	  inputFile4.close();
  
	}//// BARRIER_END
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  //=====================
  // Get input arguments: non-ROOT processes
  
  if (!iAmRoot)
    {
      ifstream inputFile1(fset1);
      ifstream inputFile2(fset2);
      ifstream inputFile3(fset3);
      ifstream inputFile4(fset4);
	    
      for (i = 0; i < 11; ++i)
	{
	  inputFile1 >> inputs1[i];
	}

      for (i = 0; i < 8; ++i)
	{
	  inputFile2 >> inputs2[i];
	}

      for (i = 0; i < 7; ++i)
	{
	  inputFile3 >> inputs3[i];
	}

      for (i = 0; i < 7; ++i)
	{
	  inputFile4 >> inputs4[i];
	}
	    
      inputFile1.close();
      inputFile2.close();
      inputFile3.close();
      inputFile4.close();
    }
  
  //====================================================
  // Inputs conversion and check: string --> int/double:

  inum = 0; flag = 0;

  convert_inputs2data(inputs1, inputs2,
		      inputs3, inputs4, wRank, inum, flag);
  
  npt = tpLstSz; /* Input temperature list size, must
		    be equal to the number of replicas
		    simulated in the MC sim. with PT; */
  
  if (iAmRoot && (flag > 0))
    {
      string fileTag[4] = {"MODEL", "MC_SIM", "SMD_SIM", "Disorder"};

      n = flag - 1; //( Input error source )
      
      cerr << "\n Invalid input (" << inum;

      cerr << ") on file (" << fileTag[n];

      cerr << ") ... \n\n";
    }

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (flag > 0){MPI_Finalize(); return 0;}

  //==============================
  // Check settings compatibility:
  
#if WITH_OPENCV == 1
  ////  
  if ((iAmRoot) && (recSpinVec) && (Lsz > 24))
    {
      cerr << "\n Warning ... \n\n"
	   << " Lattice size is too big for proper\n"
	   << " execution of all video operations;\n";
    }
  ////( Compatibility-check )
#endif
       
  //==============================
  // Printing simulation settings: (user checks input)

  flag = 0;
  
  if (iAmRoot) // Root-SCOPE (START)
    {
      print_RK_info("onTerminal");

      print_RK_info("recordFile");

      /* Double precision check */
  
      double x0 = 1 / 3.0;
  
      cerr << " Double 17-digits:\n\n 1/3 = "
	   << fmtDbleFix(x0, 17, 20);

      cerr << "\n" << endl;

#if WITH_OPENCV == 1
      ///
      /* Open-CV features */

      if (recSpinVec)
	{
	  if (Lsz > 64)
	    {
	      cerr << " Video of the spin-evolution\n"
		   << " requires a low system size;\n"
		   << " Maximum size: 64 X 64;   \n\n";

	      cerr << " Code terminated ...\n\n";

	      flag = 1; //( Execution will abort )
	    }
	  else
	    { cerr << " Spin-evo movie will be generated;\n\n"; }
	}
      
      if (flag == 0 && recSpecMov)
	{
	  cerr << " Spectral-form movie will be generated;\n\n"; 
	}
#endif/// Video operations checkpoint ...

      if (flag == 0 && getEvoData)
	{
	  cerr << " Temporal evolution will be skipped and \n"
	       << " the code will try to get the spectral  \n"
	       << " time-series data from the input file;\n\n";
	}
            
#if AUTO_START == 0
      ///
      /* Get start-confirmation from user */
      
      if (flag == 0)
	{
	  cerr << " Proceed? (1/0) : ";

	  cin >> ch0; cerr << endl;  

	  if (ch0 != 1)
	    {
	      cerr << " Code terminated ...\n\n";

	      flag = 1;
	    }
	}///| Final checkpoint before
      //////| main code execution ...
#endif
      
    }// Root-SCOPE (END)

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}
  
  //==============================
  // Define useful strings and set
  // input temperature file string:
  
  string outTagBin;
  
  string TGridFile = outDir0;
      
  outTagBin = outLabel + ".bin";
    
  TGridFile += "Temp_List" + outTagBin;

  /*-----------------------
    Verify 'TGridFile' file */
  
  if ((iAmRoot) && (getTGrid))
    {
      const size_t szData0 = npt * dbleSz;
      
      ifstream testFile(TGridFile, ios::binary);

      streampos szFile0;
      
      if (!testFile.is_open())
	{
	  cerr << " Error: could not find input file!\n"
	       << " Input: " << TGridFile;

	  cerr << "\n" << endl; flag = 1;
	}
      else//( File exits, checking data size )
	{ 
	  testFile.seekg(0, ios::end);

	  szFile0 = testFile.tellg(); // Input file size;

	  if (szFile0 != szData0)
	    {
	      cerr << " Error: input file has no valid data!\n"
		   << " Input: " << TGridFile << endl;

	      cerr << " \n Expected size (in bytes): " << szData0
		   << " \n Detected size (in bytes): " << szFile0;	    

	      cerr << "\n" << endl; flag = 1;
	    }

	  testFile.close();
	}
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  //======================
  // Get temperature list:

  double TpVec[npt];
  
  if (iAmRoot)
    {
      if (getTGrid)
	{
	  ifstream TList(TGridFile, ios::binary);
	  
	  size_t szVec = npt * dbleSz;
      
	  TList.read(reinterpret_cast<char*>(TpVec), szVec);

	  TList.close();
	}
      else // Make T-grid from scratch ...
	{
	  double TDiff = Temp2 - Temp1;
	  
	  double delTemp0 = (npt > 1) ? TDiff / (npt - 1) : 0.0;
	   
	  if (linTGrid) //( Linear distribution )
	    {
	      cerr << " T-grid: linear;\n" << endl;
		  
	      for (n = 0; n < npt; n++)
		{
		  TpVec[n] = Temp1 + n * delTemp0;
		}
	    }
	  else //( Nonlinear distribution )
	    {
	      double alpha = 0.2;
		  
	      cerr << " T-grid: nonlinear;\n" << endl;
		  
	      get_gauss_TGrid(delTemp0, alpha, TpVec);
	    }
	}
    }

  MPI_Bcast(TpVec, npt, MPI_DOUBLE, root, MPI_COMM_WORLD); 
  
  //============================
  // Set simulation temperature:
  
  const double Temp = TpVec[tpIndex];

  const double Beta = 1.0 / Temp;

  if (iAmRoot)
    {
      cerr << " Working temperature = " << Temp;

      cerr << "\n" << endl;
      
      if (Temp > TempMax)
	{
	  cerr << " Error: input temperature exceeds\n"
	       << " ------ limit defined by TempMax..."; 

	  cerr << "\n" << endl; flag = 1;
	}
      else
	{ flag = 0; }
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  //==========================
  // Define important strings:

  ostringstream oss0;

  string str0, stemp;

  string tTagDat, tTagBin;

  string samplesFile = outDir0;

  string fwindowFile = outDir0;
  
  oss0 << fixed << setprecision(5) << Temp;
  
  stemp = oss0.str(); // 'Temp' string form;

  str0 = "_T(" + stemp + ")";

  str0 += outLabel;

  tTagDat = str0 + ".dat";
  tTagBin = str0 + ".bin";
  
  samplesFile += "MC_Bin_SpinField" + tTagBin;

  fwindowFile += "winvec.bin";
  
  //============================================
  // Check input files & set reading parameters:

  /*-------------------------------------------
    Verify if the files (see names above) exist */

  flag = 0;
  
  if (iAmRoot)
    {      
      ifstream testFile1(samplesFile, ios::binary);
      
      if (!testFile1.is_open())
	{
	  cerr << " Error: could not find input file!\n"
	       << " Input: " << samplesFile;

	  cerr << "\n" << endl; flag++;
	}
      else
	{ testFile1.close(); }
    }
  
  if ((iAmRoot) && (getWData))
    {
      ifstream testFile2(fwindowFile, ios::binary);
      
      if (!testFile2.is_open())
	{
	  cerr << " Error: could not find input file!\n"
	       << " Input: " << fwindowFile;

	  cerr << "\n" << endl; flag++;
	}
      else
	{ testFile2.close(); }
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}
      
  /*---------------------------------------
    Declare size constants for MPI-parallel 
    reading and get configuration set size */
   
  const int PackSize = NMeas / crSkip;

  const size_t szIMap = Ns * intgSz;
	  
  const size_t szConf = Ns * szSpinVec;
  
  const size_t szCSet = wSize * szConf;

  const size_t szCPack = ( disOrder * szIMap +
			   PackSize * szConf );
  
  int sizeInput, sizeExtra, sizeSData;

  ifstream fieldInput, fwindInput;

  streampos szFile; size_t szRef;

  int PackSz, NPacks; //( important ints defined below )
  
  if (iAmRoot)
    {	    
      fieldInput.open(samplesFile, ios::in | ios::binary);

      fieldInput.seekg(0, ios::end);
	
      szFile = fieldInput.tellg(); // Input file size;

      sizeInput = szFile; // It will be broadcasted soon;

      if (disOrder)
	{
	  szRef = szCPack; }
      else
	{ szRef = szConf ; }

      if (szFile == szRef)
	{
	  cerr << " File has only 1 disorder realization!\n" << endl;

	  NPacks = 1; PackSz = PackSize; //( when testing: set PackSz = 1 )
	}      
      else if (szFile % szRef != 0)
	{		  
	  cerr << " Error: input samples file has no valid data!\n"
	       << " Input: " << samplesFile << endl;

	  cerr << " \n Expected block size (in bytes): " << szRef
	       << " \n Detected total size (in bytes): " << szFile;

	  cerr << " \n" << endl; flag++;
	}
      else//( for multi-samples files with valid data ) 
	{
	  PackSz = PackSize;
	  
	  NPacks = szFile / szCPack;
	 		
	  sizeExtra = disOrder * NPacks * szIMap;

	  n0 = NPacks * PackSz; // Total number of samples (#);

	  m0 = ( sizeInput - sizeExtra) / szConf; //( = # )

	  if (n0 != m0)
	    {
	      cerr << " Error: inconsistency found in the"
		   << " input samples file!\n File : " << samplesFile;

	      cerr << " \n" << endl; flag++;
	    }
	  else//( if data-info variables are consistent )
	    {
	      cerr << " Samples file data information:    "
		   << " \n > number of recorded samples = " << n0
		   << " \n > number of recorded packs   = " << NPacks
		   << " \n > number of samples / pack   = " << PackSz;

	      cerr << " \n\n Samples file size (user-check):\n\n" << X2;

	      sizeSData = n0 * szConf;
	  
	      cerr << sizeInput << " = "
		   << sizeExtra << " (header) + "
		   << sizeSData << " (samples)\n" << endl;
	    }
	}

      fieldInput.close();     
    }

  int ntm0, key = 0; //( may change if getWData = true )
  
  if ((iAmRoot) && (getWData) && (flag == 0))
    {
      fwindInput.open(fwindowFile, ios::in | ios::binary);

      fwindInput.seekg(0, ios::end);
	
      szFile = fwindInput.tellg();

      if (szFile % dbleSz != 0)
	{
	  cerr << " Error: input window file has no valid data!\n"
	       << " Input: " << fwindowFile << endl;

	  cerr << " \n Expected block size (in bytes): " << dbleSz
	       << " \n Detected total size (in bytes): " << szFile;

	  cerr << "\n" << endl; flag++;
	}
      else // The values below will be broadcasted soon;
	{
	  ntm0 = szFile / dbleSz;

	  if (ntm0 != ntm)
	    {
	      cerr << " Number of time steps will be updated:\n"
		   << " ntm = " << ntm << " ---> " << ntm0;

	      cerr << "\n" << endl;

	      key = 1; //( Code will update ntm )
	    }
	}

      fwindInput.close();
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  /*-------------------------------
    Proceed with execution or abort */
  
  if (flag > 0){MPI_Finalize(); return 0;}
  
  MPI_Bcast(&sizeInput, 1, MPI_INT, root, MPI_COMM_WORLD);

  MPI_Bcast(&PackSz, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&NPacks, 1, MPI_INT, root, MPI_COMM_WORLD);

  MPI_Bcast(&key, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (key == 1)
    {  
      MPI_Bcast(&ntm0, 1, MPI_INT, root, MPI_COMM_WORLD);
 
      ntm = ntm0; //<--- Global integer UPDATED!

      npw = ntm / 2; ndim = npw - 1; //( more updates )
    }

  /*--------------------------------
    Define needed integer constants:
    Number of samples & iterations 

    If wSize = 1 : NSets = NSamps; 

    If RKCheck = true : 1-sample test; 

    In Root-CHECK: root process checks
    the compatibility of the parallel
    execution with the input data; */
 
  int NSamps, NSets;

  int NTest = 0; //( Changes 'NSamps' for testing )

  NSamps = NPacks * PackSz;

  if (NTest > 0){ NSamps = NTest; }

  if (iAmRoot) // Root-CHECK (START);
    {
      ostringstream info1_oss, info2_oss;

      info1_oss << PackSz << " | " << wSize << "; ";      
      info2_oss << NSamps << " | " << wSize << "; ";
      
      n0 = NPacks * PackSz; // Total number of samples;

      flag = 0; // Alerts for invalid settings;

      if (disOrder)
	{
	  if (PackSz % wSize != 0)
	    {	    
	      cerr << " Number of working processes must be a    \n"
		   << " multiple of the number of samples/pack:\n\n"
		   << " PackSz | wSize : " << info1_oss.str();
	    
	      cerr << "\n" << endl; flag = 1;
	    }
	}
      else//( If disorder-mode is disabled )
	{
	  if (NSamps % wSize != 0)
	    {	    
	      cerr << " Number of working processes must be a     \n"
		   << " multiple of the total number of samples:\n\n"
		   << " NSamps | wSize : " << info2_oss.str();
	    
	      cerr << "\n" << endl; flag = 1;
	    }
	}
    }//// Root-CHECK (END)
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  /* Configure 'NSets' for the execution */
  
  if (disOrder)
    {
      NSets = NPacks * ( PackSz / wSize ); }
  else
    { NSets = NSamps / wSize; }

  if (RKCheck){ NSets = 1; }
    
  //===============================================
  // Root generates/reads lattice sites vector list
  /*
    ............
    Information: the list is required by the subrou-
    tine "make_nborsTable" and in the definition of
    the pointer "imgSites" needed to plot the latti-
    ce site points on images with OpenCV codes and 
    it is also used in DFT calculations; */
  
  flag = 0;

  rvecList = Alloc_dble_array(Ns, 2);

  if (!qcrystal)
    {
      if (multiSubs)
	{
	  r0List = Alloc_dble_array(Ns0, 2);
	}

      if (with_DFTcodes)
	{
	  dftGrid = Alloc_dble_array(Nsg, 2);

	  gridMap = new int[Nsg];
	}
    }

  if (iAmRoot)
    {  
      if (!qcrystal)
	{
	  cerr << " Making rvec-list ...";
	  
	  make_rvecList(flag);

	  if (flag > 0)
	    {
	      string msg1 = " List size does not match expected size (Ns);";
	      string msg2 = " Error while mapping sites into the DFT-grid;";
		
	      cerr << " Failed!\n\n";

	      cerr << " Problems in make_rvecList!  \n\n"
		   << (flag == 1 ? msg1 : msg2) << "\n\n";
	    }
	  else
	    { cerr << " OK!\n\n"; }
	}
      else//( lattice is defined externally )
	{
	  cerr << " Reading rvec-list ...";
	  
	  const size_t szVec2d = 2 * dbleSz;
	
	  const size_t szTable = Ns * szVec2d;
		    
	  string fname0 = "sitesList" + SzTag + ".bin";

	  string sitesList = outDir0 + fname0;

	  ifstream stFile(sitesList, ios::binary);
	    
	  if (stFile.is_open())
	    {	    
	      streampos szFile;

	      stFile.seekg(0, ios::end);
	
	      szFile = stFile.tellg();

	      if (szFile == szTable)
		{
		  double rvec[2];	      	      

		  stFile.seekg(0, ios::beg);
      
		  for (k = 0; k < Ns; k++)
		    {		      
		      stFile.read(reinterpret_cast<char*>(rvec), szVec2d);

		      rvecList[k][0] = rvec[0];
		      rvecList[k][1] = rvec[1];
		    }

		  cerr << " OK!\n\n"; 
		  
		  cerr << " Sites-table read from file!\n\n";
		}
	      else//( File size differs from the expected value )
		{
		  cerr << " Failed!\n\n";
		  
		  cerr << " Invalid data in: " << sitesList << "\n\n";

		  cerr << " Tip: check data using other program!";

		  cerr << endl << "\n"; flag = 1;
		}

	      stFile.close();
	    }
	  else//( if the needed file is missing )
	    {
	      cerr << " Failed!\n\n";
	      
	      cerr << " File not found: " << sitesList;

	      cerr << endl << "\n"; flag = 1;
	    }
	}     	
    }///| Root procedure;
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);

  if (flag > 0)
    {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  else // Everything okay, broadcasting pointers/lists ...
    {
      if (!qcrystal)
	{
	  /* --------
	     rvecList */
	  
	  int listSz = Ns * 2;

	  double *flatList; //( list to send )

	  flatList = new double[listSz];

	  if (iAmRoot)
	    {
	      dbleFlatten2D(Ns, 2, rvecList, flatList);
	    }
      
	  MPI_Bcast(flatList, Ns * 2,
		    MPI_DOUBLE, root, MPI_COMM_WORLD);

	  if (!iAmRoot)
	    {
	      dbleReshape2D(Ns, 2, flatList, rvecList);
	    }

	  delete[] flatList;

	  /* ------
	     r0List */
	  
	  if (multiSubs)
	    {
	      int listSz = Ns0 * 2;

	      double *flatList; //( list to send )

	      flatList = new double[listSz];

	      if (iAmRoot)
		{
		  dbleFlatten2D(Ns0, 2, r0List, flatList);
		}
      
	      MPI_Bcast(flatList, Ns0 * 2,
			MPI_DOUBLE, root, MPI_COMM_WORLD);

	      if (!iAmRoot)
		{
		  dbleReshape2D(Ns0, 2, flatList, r0List);
		}

	      delete[] flatList;
	    }

	  /* -----------------
	     dftGrid & gridMap */
	  
	  if (with_DFTcodes)
	    {
	      int listSz = Nsg * 2;

	      double *flatList; //( list to send )

	      flatList = new double[listSz];

	      if (iAmRoot)
		{
		  dbleFlatten2D(Nsg, 2, dftGrid, flatList);
		}
      
	      MPI_Bcast(flatList, Nsg * 2,
			MPI_DOUBLE, root, MPI_COMM_WORLD);

	      MPI_Bcast(gridMap, Nsg * 1,
			MPI_INT, root, MPI_COMM_WORLD);

	      if (!iAmRoot)
		{
		  dbleReshape2D(Nsg, 2, flatList, dftGrid);
		}

	      delete[] flatList;
	    }
	}
    }
  
  //===========================================
  // Create list of lattice points 'imgSites' &
  // define global size-type object 'plotSize':
  /*
    imgSites : pointer of type Point;
    |
    |---> n = 0, 1, 2, ..., Ns - 1 ;
    |
    |---> imgSites[n] = Point(i,j) ; 

    plotSize = Size(ncols, nrows);

    Alert: make_latticeGrid & make_vecField
    depend on the global variable 'plotSize'
    defined below, when disorder is enabled
    these procedures also require the global
    pointer (impurity map) 'impField';

    Margins: plotMargin1(2) sets left (right)
    and top (bottom) margins of the lattice-
    grid base image, custom extra margins are
    set by the integers xPlus and yPlus below; */

#if WITH_OPENCV == 1
  ///
  int ncols, nrows;
  
  int gsp = round(gridSpac);
  
  int plotMargin1 = round(1.5 * gsp);  
  int plotMargin2 = round(2.5 * gsp);

  int xPlus1 = 0, yPlus1 = 0;
  int xPlus2 = 0, yPlus2 = 0;
    
  Point nvec; // Aux. point-object;

  imgSites = new Point[Ns];  // Lattice sites on image/plot;

  if (qcrystal)
    {
      xPlus1 = gsp; xPlus2 = round(1.5 * plotMargin2);
      yPlus1 = gsp; yPlus2 = round(1.5 * plotMargin2);
    }
 
  if (geom != "triang")
    {
      if (geom == "kagome")
	{
	  xPlus1 = gsp; xPlus2 = round(2.0 * gridSpac);
	  yPlus2 = gsp;
	}
      else//( square, lieb, hexagonal ) 
	{
	  xPlus2 = round(0.5 * gridSpac);
	  yPlus2 = round(0.5 * gridSpac);
	}
    } 
	  
  for (k = 0; k < Ns; k++)
    {
      imgSites[k].x = plotMargin1 + xPlus1 + round(gridSpac * rvecList[k][0]);
      imgSites[k].y = plotMargin1 + yPlus1 + round(gridSpac * rvecList[k][1]);
    }

  if (multiSubs) // Aux. lattice sites;
    {
      ir0Sites = new Point[Ns0];

      for (k = 0; k < Ns0; k++)
	{
	  ir0Sites[k].x = plotMargin1 + xPlus1 + round(gridSpac * r0List[k][0]);
	  ir0Sites[k].y = plotMargin1 + yPlus1 + round(gridSpac * r0List[k][1]);
	}
    }
    
  if (iAmRoot)
    {      
      if (qcrystal)
	{       
	  double rvec[2];

	  int xmax = 0;
	  int ymax = 0;
	  
	  for (k = 0; k < Ns; k++)
	    {		      
	      rvec[0] = rvecList[k][0];
	      rvec[1] = rvecList[k][1];

	      if (rvec[0] > xmax){ xmax = rvec[0]; }
	      if (rvec[1] > ymax){ ymax = rvec[1]; }
	    }

	  nvec.x = round(xmax * gridSpac) + plotMargin2 + xPlus2;
	  nvec.y = round(ymax * gridSpac) + plotMargin2 + yPlus2;	      
	}
      else//( if the lattice is defined within code )
	{
	  k = Ns - 1; //( last site )
	   
	  nvec.x = round(gridSpac * rvecList[k][0]) + plotMargin2 + xPlus2; 
	  nvec.y = round(gridSpac * rvecList[k][1]) + plotMargin2 + yPlus2;	    
	}	  	 

      if (recSpinVec)
	{	
	  if (nvec.x > 4096 || nvec.y > 4096)
	    {
	      flag = 1; //( Maximum resolution exceeded )
	        
	      cerr << " Video size (plotSize) exceeds \n"
		   << " the maximum resolution (4k)!\n\n";

	      cerr << " Please, disable OpenCV feautures\n"
		   << " and recompile (WITH_OPENCV = 0);\n";
	    
	      cerr << " \n Code terminated ...\n\n";
	    }
	}   
    }///| Root-procedure;
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
    
  if (flag > 0)
    {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  else//( Broadcast data & set global variable )
    {
      MPI_Bcast(&nvec, sizeof(Point), MPI_BYTE, root, MPI_COMM_WORLD);     

      ncols = nvec.x; // Get number of cols & rows
      nrows = nvec.y; // from point obj. 'nvec';
      
      plotSize = Size(ncols, nrows);

      if (iAmRoot)//( print video information )
	{ 
	  cerr << " Output image/video size : "
	       << plotSize.width  << " x "
	       << plotSize.height << " \n" << endl;
	}
    }
  ///( WITH_OPENCV == 1 )
#endif 
  
  //=============================================
  // Generate/read matrices of neighboring sites:
    
  /*.....................
    Pointers description: (all 3 are global)

    nbors1 : matrix of 1st-nearest-neighbors;
    nbors2 : matrix of 2nd-nearest-neighbors;
    nborsX : matrix of Xth-nearest-neighbors;

    For crystal lattices: X = 3, next-next
    ..................... nearest neighbors;

    For qct-lattices: X = 5, 5th-order
    ................. distant neighbors; */

  nbors1 = Alloc_intg_array(Ns, Zn1);
  nbors2 = Alloc_intg_array(Ns, Zn2);
  nborsX = Alloc_intg_array(Ns, ZnX);
  
  if (iAmRoot)
    {      
      const size_t szList1 = Zn1 * intgSz;
      const size_t szList2 = Zn2 * intgSz;
      const size_t szList3 = ZnX * intgSz;

      const size_t szTable = Ns * ( szList1 +
				    szList2 +
				    szList3 );
      
      string fname0 = "nborsList" + SzTag + ".bin";

      string nborsList = outDir0 + fname0;

      ifstream nbFile(nborsList, ios::binary);

      /*..........................
	Possible procedures below:
	
	1) Get neighbors-table data from file;
	2) Generate neighbors-table from scratch; */
      
      if (nbFile.is_open())//(1)
	{	  
	  streampos szFile;

	  nbFile.seekg(0, ios::end);
	
	  szFile = nbFile.tellg();

	  if (szFile == szTable)
	    {
	      nbFile.seekg(0, ios::beg);
	      
	      for (k = 0; k < Ns; k++)
		{
		  nbFile.read(reinterpret_cast<char*>(nbors1[k]), szList1);
		  nbFile.read(reinterpret_cast<char*>(nbors2[k]), szList2);
		  nbFile.read(reinterpret_cast<char*>(nborsX[k]), szList3);
		}

	      cerr << " Neighbors-table read from file!\n\n";
	      
	      if (!qcrystal)
		{
		  for (k = 0; k < Ns; k++)
		    {
		      for (n = 0; n < Zn1; n++)
			{
			  if (nbors1[k][n] < 0){ flag = 1; }
			}
		      for (n = 0; n < Zn2; n++)
			{
			  if (nbors2[k][n] < 0){ flag = 1; }
			}
		      for (n = 0; n < ZnX; n++)
			{
			  if (nborsX[k][n] < 0){ flag = 1; }
			}
		    }

		  if (flag > 0)
		    {
		      cerr << " Invalid data in: " << nborsList << "\n\n";

		      cerr << " Tip: file was generated for a qct-lattice!";

		      cerr << endl << "\n";
		    }
		}

	      if (flag == 0)
		{
		  double lspc; //( lattice spacing )
		  
		  check_latticeSpc(lspc, flag);

		  if (qct_RemovePBC)
		    {
		      rmPBC_nborsTable(lspc);
		    }

		  if (flag > 0)
		    {
		      cerr << " Lattice spacing is not unit!\n\n";
		    }
		}	      
	    }
	  else//( File size differs from the expected value )
	    {
	      cerr << " Invalid data in: " << nborsList << "\n\n";

	      cerr << " Tip: check data or delete file!";

	      cerr << endl << "\n"; flag = 1;
	    }
	  
	  nbFile.close();	 
	}
      else//(2)
	{
	  if (qcrystal)
	    {
	      cerr << " File not found: " << nborsList;

	      cerr << endl << "\n"; flag = 1;
	    }
	  else//( lattice is defined within code )
	    {
	      auto time1 = high_resolution_clock::now();
	  
	      cerr << " Making neighbors-tables ... ";
  
	      make_nborsTable(flag);

	      auto time2 = high_resolution_clock::now();

	      auto dtime = time2 - time1;

	      m = duration_cast<milliseconds>(dtime).count();

	      if (m == 0){m = 1;}

	      if (flag > 0)
		{	  
		  cerr << " Failed!\n\n";

		  cerr << " Problems in make_nborsTable!\n\n";
		}
	      else
		{ cerr << "OK! " << m << " ms\n\n"; }
	    }
	}
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0)
    {
      MPI_Finalize(); return 0;
    }
  else // Everything okay, broadcasting tables ...
    {
      int sz1, sz2, sz3;

      int *nvec1, *nvec2, *nvec3;

      sz1 = Ns * Zn1;
      sz2 = Ns * Zn2;
      sz3 = Ns * ZnX;

      nvec1 = new int[sz1];
      nvec2 = new int[sz2];
      nvec3 = new int[sz3];

      if (iAmRoot)
	{
	  intFlatten2D(Ns, Zn1, nbors1, nvec1);
	  intFlatten2D(Ns, Zn2, nbors2, nvec2);
	  intFlatten2D(Ns, ZnX, nborsX, nvec3);
	}
      
      MPI_Bcast(nvec1, Ns * Zn1, MPI_INT,
		root, MPI_COMM_WORLD);

      MPI_Bcast(nvec2, Ns * Zn2, MPI_INT,
		root, MPI_COMM_WORLD);

      MPI_Bcast(nvec3, Ns * ZnX, MPI_INT,
		root, MPI_COMM_WORLD);

      if (!iAmRoot)
	{
	  intReshape2D(Ns, Zn1, nvec1, nbors1);
	  intReshape2D(Ns, Zn2, nvec2, nbors2);
	  intReshape2D(Ns, ZnX, nvec3, nborsX);
	}

      delete[] nvec1; // These were temporary
      delete[] nvec2; // arrays for MPI_Bcast;
      delete[] nvec3;
    }
      
  //===============================
  // Set discretized momentum space
  // global variables (constants) :
    
  /* Below, some global variables related to the
     1st Brillouin zone (1st BZone) are defined
     (here, pi2 = 2.0 * pi & pi4 = 4.0 * pi): 

     dq  : momentum discretization;

     npk : auxiliary integer for subroutines;

     npPath : KGMYG/YGMXG path total points; 

     In absence of periodic bounday conditions,
     the reciprocal lattice does not exist (no 
     well defined wavevectors, Bloch's theorem
     does not apply), hence dq is arbitrary; */

  dq = (qcrystal ? 4.0 * pi2 / Gsz :  pi2 / Gsz); 

  if ((geom != "square") && (geom != "lieb"))
    { 
      npk = round(Qval / dq) + 1;      
    }
  else // Other geometries ...
    {
      npk = round(pi / dq); 
    }
  
  if (iAmRoot)
    {          
      vector<Vec2d> bwvecPath;
      
      get_qPathWVectors(n0, bwvecPath);

      npPath = n0;
    }

  MPI_Bcast(&npPath, 1, MPI_INT, root, MPI_COMM_WORLD); 
           
  //======================================
  // Create lattice bonds & z-values list:
    
  /*---------------------
    Pointers description: (global object)

    zvalList : z-values list for all sites;
    |
    |---> zvalList[n] = Point( z1 , z2 );

    bondList : list of lattice bonds;
    |
    |---> bondList[n] = Point( i , j );

    z(n) : number of nearest & 
    ...... next-nearest-neighbors;

    Point-type (see 'structures.cpp') object:
    pointer containing all bonds in the latti-
    ce within its components which consist of
    2 integers (each associated with a site);

    The procedure 'make_bondList' called be-
    low computes the pointer 'bondList' and
    the number of bonds, which are then assi-
    gned to the global integers Nb1, Nb2 &
    NbX, with the total number of bonds
    Nb also being global integer;
    
    Nb1 | Number of nearest,
    Nb2 | next-nearest & next-next-
    NbX | Xth-order neighbors bonds;

    WARNING: many subroutines use 'nb' as
    as a varaible, avoid confusion with
    the global integer Nb (capital N); */
   
  if (iAmRoot)
    {
      int nbonds[4];

      vector<Point> ij_List;
            
      auto time1 = high_resolution_clock::now();

      //......................................//
      
      cerr << " Making bond-list ... ";
      
      make_bondList(nbonds, ij_List);

      Nb1 = nbonds[1];
      Nb2 = nbonds[2];
      NbX = nbonds[3];

      Nb = Nb1 + Nb2 + NbX;
      
      //......................................//     

      bondList = new Point[Nb];
      zvalList = new Pts3d[Ns];

      for (k = 0; k < Nb; k++)
	{
	  bondList[k] = ij_List[k];
	}      

      make_zvalList();

      //......................................//

      auto time2 = high_resolution_clock::now();

      auto dtime = time2 - time1;

      m = duration_cast<milliseconds>(dtime).count();

      if (m == 0){m = 1;}

      cerr << "OK! " << m << " ms\n\n";
    }  
  
  /*-------------------------------------
    Broadcast global integers & pointers 
    bondList & zvalList to all processes: */
  
  MPI_Bcast(&Nb1, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&Nb2, 1, MPI_INT, root, MPI_COMM_WORLD);
  MPI_Bcast(&NbX, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  if (!iAmRoot)
    {
      Nb = Nb1 + Nb2 + NbX; //( set for all now )
	    
      bondList = new Point[Nb];
      zvalList = new Pts3d[Ns];
    }
  
  MPI_Bcast(bondList, Nb * sizeof(Point),
	    MPI_BYTE, root, MPI_COMM_WORLD);

  MPI_Bcast(zvalList, Ns * sizeof(Pts3d),
	    MPI_BYTE, root, MPI_COMM_WORLD);

  /*==========================
    Define & allocate magnetic
    order arrays / pointers...

    ...................................
    Phase lists (pointers) description: (global)

    Neel0Config --> AFM-Neel state; | for qct
    StrpXConfig --> 4Stripes state; | lattices;
     
    Q120Phase ----> inplane 120° ordered state;

    XStpPhase ----> stripe state of type X;

    UpDwPhase ----> up-up-down state;

    X : Horizontal, Vertical, Diagonal; */
 
  if (qcrystal)
    {
      Neel0Config = Alloc_dble_array(Ns, 3);
      
      Strp1Config = Alloc_dble_array(Ns, 3);
      Strp2Config = Alloc_dble_array(Ns, 3);
      Strp3Config = Alloc_dble_array(Ns, 3);
      Strp4Config = Alloc_dble_array(Ns, 3);
                  
      if (iAmRoot)
	{
	  if (refState_Zero)
	    {
	      cerr << " Reference Neel & Stripe are set to 0!\n\n";
	      
	      init_dble_array(Neel0Config, Ns, 3, 0.0);
	      init_dble_array(Strp1Config, Ns, 3, 0.0);
	      init_dble_array(Strp2Config, Ns, 3, 0.0);
	      init_dble_array(Strp3Config, Ns, 3, 0.0);
	      init_dble_array(Strp4Config, Ns, 3, 0.0);
	    }
	  else//( get reference states from files )
	    {	  
	      string fname0 = outDir0 + "stateNeel0" + SzTag + ".bin";
	      string fname1 = outDir0 + "stateStrp1" + SzTag + ".bin";
	      string fname2 = outDir0 + "stateStrp2" + SzTag + ".bin";
	      string fname3 = outDir0 + "stateStrp3" + SzTag + ".bin";
	      string fname4 = outDir0 + "stateStrp4" + SzTag + ".bin";

	      ifstream Neel0Map(fname0, ios::binary);
	      ifstream Strp1Map(fname1, ios::binary);
	      ifstream Strp2Map(fname2, ios::binary);
	      ifstream Strp3Map(fname3, ios::binary);
	      ifstream Strp4Map(fname4, ios::binary);

	      vector<int> flagVec = {0, 0, 0, 0, 0};

	      check_binFile(Ns * szSpinVec, fname0, flagVec[0]);
	      check_binFile(Ns * szSpinVec, fname1, flagVec[1]);
	      check_binFile(Ns * szSpinVec, fname2, flagVec[2]);
	      check_binFile(Ns * szSpinVec, fname3, flagVec[3]);
	      check_binFile(Ns * szSpinVec, fname4, flagVec[4]);

	      flag = ( flagVec[0] + flagVec[1] +
		       flagVec[2] + flagVec[3] + flagVec[4] );
	  	    
	      if (flag == 0)
		{
		  cerr << " Reading Neel & Stripe states ...";
	      
		  for (k = 0; k < Ns; k++)
		    {
		      Neel0Map.read(reinterpret_cast<char*>(Neel0Config[k]), szSpinVec);
		      Strp1Map.read(reinterpret_cast<char*>(Strp1Config[k]), szSpinVec);
		      Strp2Map.read(reinterpret_cast<char*>(Strp2Config[k]), szSpinVec);
		      Strp3Map.read(reinterpret_cast<char*>(Strp3Config[k]), szSpinVec);
		      Strp4Map.read(reinterpret_cast<char*>(Strp4Config[k]), szSpinVec);
		    }

		  cerr << " Done!\n\n";
		}
	  
	      Neel0Map.close();
	  
	      Strp1Map.close(); Strp2Map.close();
	      Strp3Map.close(); Strp4Map.close();
	    }
	}
      
      MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
      
      if (flag != 0)
	{
	  MPI_Finalize(); return 0;
	}
      else//( broadcast configurations )
	{
	  double *vecField0 = new double[configSz];
	  double *vecField1 = new double[configSz];
	  double *vecField2 = new double[configSz];
	  double *vecField3 = new double[configSz];
	  double *vecField4 = new double[configSz];

	  if (iAmRoot)
	    {	      
	      dbleFlatten2D(Ns, 3, Neel0Config, vecField0);
	      dbleFlatten2D(Ns, 3, Strp1Config, vecField1);
	      dbleFlatten2D(Ns, 3, Strp2Config, vecField2);
	      dbleFlatten2D(Ns, 3, Strp3Config, vecField3);
	      dbleFlatten2D(Ns, 3, Strp4Config, vecField4);
	    }

	  MPI_Bcast(vecField0, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);
	  MPI_Bcast(vecField1, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);
	  MPI_Bcast(vecField2, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);
	  MPI_Bcast(vecField3, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);
	  MPI_Bcast(vecField4, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);

	  if (!iAmRoot)
	    {
	      dbleReshape2D(Ns, 3, vecField0, Neel0Config);
	      dbleReshape2D(Ns, 3, vecField1, Strp1Config);
	      dbleReshape2D(Ns, 3, vecField2, Strp2Config);
	      dbleReshape2D(Ns, 3, vecField3, Strp3Config);
	      dbleReshape2D(Ns, 3, vecField4, Strp4Config);
	    }
	   
	  delete[] vecField0;
	  delete[] vecField1;
	  delete[] vecField2;
	  delete[] vecField3;
	  delete[] vecField4;
	}
    }
  else//( set phase list for some types of magnetic order )
    {
      Q120Phase = new complex<double>[Ns];
   
      HStpPhase = new double[Ns];
      VStpPhase = new double[Ns];
      DStpPhase = new double[Ns];
      UpDwPhase = new double[Ns];

      make_Q120PhaseList(); // Defines 'Q120Phase';
      make_StrpPhaseList(); // Defines 'XStpPhase';
      make_2U1DPhaseList(); // Defines 'UpDwPhase';
    }
  
  /*====================================
    Preparation for the Runge-Kutta code
    ====================================

    --> Set time & frequency parameters:

    ntm : num. of time & frequency steps;
    |
    | GV acquired before from the input
    | data via 'convert_inputs2data';
    
    Below, dtm, dw, tmax & wmax are GVs
    used in the time evolution proc.;

    wmax : 1D DFT max. frequency;
    tmax : evolution maximum time;

    dwf : frequency discretization;
    dtm : real time discretization;
   
    Frequency range: [ - wmax, + wmax ]
    |
    | Interval length = 2 * wmax = Lw
    |
    | dwf = Lw / ntm

    Real time range: [ 0, tmax ]
    |
    | tmax =  ntm * dtm;

    Above, we can use the fact that the
    frequency discretization is give by
    
    dwf = Lw / ntm = pi2 / tmax;

    2 * wmax / ntm = pi2 / tmax;

    wmax / ntm = pi / (ntm * dtm);

    wmax = pi / dtm;

    Thus, the real time discretization
    can be obtained via the relation

    dtm = pi / wmax; 

    Note: the factor 'wfac' below extends
    the frequency range given by the vari-
    able 'wmax', a wider w-range gives mo-
    re information about the system dyna-
    mics, but 'ntm' must be large enough
    to compensate for that and ensure a
    small 'dwf' (see below); */
    
  // OLD-WAY:
  //
  // wmax = wfac * maxLocField;
  //
  // dtm = pi / wmax;
  //
  // tmax = ntm * dtm;
  //
  // dwf = 2.0 * wmax / ntm;

  wmax = pi / dtm;

  tmax = ntm * dtm;

  dwf = 2.0 * pi / tmax;

  if (iAmRoot)
    {      
      flag = 0;
      
      if (dwf > dwf_min)
	{
	  flag = 1;

	  cerr << " Frequency-discretization is larger\n"
	       << " than the minimum value: " << dwf_min;
	  
	  cerr << " \n\n Try to increase the number of\n"
	       << " time-points to solve this issue.\n\n";
	}

      if (flag == 0)
	{
	  string fileName = outDir1 + "RK_Check/0RK_Evo_Info.txt";

	  ofstream RK_InfoFile(fileName, ios::out);

	  RK_InfoFile << " ntm  = " << ntm << endl << endl;
	  
	  RK_InfoFile << " dtm  = " << fmtDbleFix(dtm , 5, 9) << endl;
	  RK_InfoFile << " dwf  = " << fmtDbleFix(dwf , 5, 9) << endl;	   
	  RK_InfoFile << " tmax = " << fmtDbleFix(tmax, 5, 9) << endl;
	  RK_InfoFile << " wmax = " << fmtDbleFix(wmax, 5, 9) << endl;

	  RK_InfoFile.close();
	}
    }//// Check frequency-discretization...

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}
  
  /*------------------------------------------ | FFTW is not used anymore,
    Initiate FFT objects & optimize procedures | MKL-Dfti replaced it;

    FFTW_ON --> prepare_xyzPlan_fftw2D(); */
  
  if (iAmRoot)
    {
      cerr << " RK-code DFT method: ";

      cerr << (qcrystal ? "general (direct sum)" : "FFT with Intel-MKL");

      cerr << "\n" << endl;
    }

  n0 = 0; //( error var. init. )
      
  if (!qcrystal)
    {
      if (iAmRoot){cerr << " FFTW/Dfti optimization ... ";}

      auto time1 = high_resolution_clock::now();   

      MKL_LONG dims[2] = {Gsz, Gsz};

      mklStat = DftiCreateDescriptor(&handle,
				     DFTI_DOUBLE,
				     DFTI_COMPLEX, 2, dims);

      mklStat = DftiCommitDescriptor(handle);

      if (mklStat == DFTI_NO_ERROR)
	{
	  mklStat = DftiCommitDescriptor(handle);
	}
      
      n0 = (mklStat != DFTI_NO_ERROR ? 1 : 0);
      
      auto time2 = high_resolution_clock::now();

      auto dtime = time2 - time1;

      m = duration_cast<milliseconds>(dtime).count();

      if (m == 0){m = 1;}

      if (iAmRoot)
	{
	  cerr << (n0 == 0 ? "OK! " : "Failed! ") << m << " ms \n\n";
	}
    }
  
  MPI_Reduce(&n0, &flag, 1, MPI_INT,
	     MPI_SUM, root, MPI_COMM_WORLD);

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD); 

  if (flag > 0)
    {
      if (iAmRoot){ cerr << " Error during FFT-plan setup!\n\n"; }
      
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

  /*-----------------------
    Allocate working arrays
    
    .............
    Commentaries:

    The vector 'qSStVec' is the spin-spin
    temporal correlation measurement:

    qSStVec(t) = Sum[i]{ Spin(i,t) * Spin(i,0) }

    Above, Spin(i,t) is a 3D vector object
    defining the physical spin at the site
    i and time instant t, the actual value
    of qSStVec(t) is normalized by 1 / Ns;

    The 'spinField' pointer represents the
    lattice spin configuration that will be
    obtained from the input binary file that
    was opened & checked in the code above;

    The CMT-type pointers below are used by
    the root process to accumulate the data
    from measurements of the 3 spins momen-
    tum forms evolution (freq-series) and
    of the spin-spin temporal correlation;

    For a crystal lattice, the number of
    wavevectors in the 1st Brillouin zo-
    ne is equal to the number of sites:

    Nq = Ns & iNq = iNs ;

    For a quase-crystal, that is not the
    case, Nq can be set to any value de-
    pending on Lsz which is arbitrary in
    this case (in this code, we choose
    Lsz = sqrt(Ns) : rounded & even);

    Alert: make_latticeGrid/vecField depend
    on the GV 'plotSize' & also require the
    global pointer (impurity map) 'impField'
    allocated below (map loaded later); */
   
  double **spinField;
  
  double **SqxWVec, **CMT_SqxWVec;
  double **SqyWVec, **CMT_SqyWVec;
  double **SqzWVec, **CMT_SqzWVec;

  double *qSStVec = new double[ntm];

  double *CMT_qSStVec;
    
  spinField = Alloc_dble_array(Ns, 3);

  SqxWVec = Alloc_dble_array(Nq, ntm);
  SqyWVec = Alloc_dble_array(Nq, ntm);
  SqzWVec = Alloc_dble_array(Nq, ntm);

  if (disOrder)//( alloc. & initialize imp. map ) 
    {
      impField = new int[Ns];

      for (k = 0; k < Ns; k++)
	{
	  impField[k] = 0;
	}
    }

  /*--------------------------------------------
    DFT procedure check + static SF calculation
    & record samples local-spin-field histograms */
  
  if (iAmRoot)
    {
      /*....................................
	Prepare Mat objects for imaging and
	video writer to recrod impurity maps */
      
      bool recImpMap = false;
      bool recOprMap = false;
  
#if WITH_OPENCV == 1
      ///
      Mat gridMat;  // Lattice grid (background for field); 
  
      Mat vecsMat;  // Vector field on lattice;

      Mat gridIMap; // Lattice grid + impurities;

      Vec2d *orderMap; // Order-parameter map (qct-lattice);
      
      VideoWriter omapVideo, imapVideo;

      string cdc = "H264"; //( video codec type )

      string outImg = outDir2 + "lattice_image.png";

      string vidName1 = "orderPar_Map"; // output 
      string vidName2 = "impurity_Map"; // names;

      string outVideo1 = outDir3 + vidName1 + ".avi";
      string outVideo2 = outDir3 + vidName2 + ".avi";

      int fps1 = 5, fps2 = 2; //( frame-rate )

      unsigned int codec; //( video codec )
      
      make_latticeGrid(gridMat, outImg);

      codec = VideoWriter::fourcc(cdc[0], cdc[1],
				  cdc[2], cdc[3]);

      if ((qcrystal) && (rec_orderMaps))// && (Lsz < 64)|( add this if figs. are too large )
	{
	  recOprMap = true;
	  
	  orderMap = new Vec2d[Ns];

	  omapVideo.open(outVideo1, codec, fps1, plotSize);
	} 
      
      if ((disOrder) && (Lsz < 64)) //( set imapVideo )
	{
	  recImpMap = true;
	  
	  imapVideo.open(outVideo2, codec, fps2, plotSize);
	}
      ///
#endif///[ Imaging stuff ]
      
      /*---------------------------
	Define histogram parameters

	1) Number of histogram bars (odd);
	2) Histogram full range length;
	3) Box-size: bar counter length;
	4) Factor required for indexing; */
    
      const int NHist = 201; //(1)

      const double hRange = 2.0 * maxLocField; //(2)
      
      const double boxSz = hRange / (NHist - 1); //(3)
 
      const double boxFc = 1.0 / boxSz; //(4)

      /*............................
	Define variables, image-tag
	strings & output image names */
      
      bool recKey; 	 

      Vec3d locField;

      string stag, ftag, infoStr;

      string tailName, outImg0, outImg1;
      
      string imgDir = outDir2 + subDir4;
 
      string headTag0 = "spinsMap_sample(";
      string headTag1 = "orderMap_sample(";
	  
      string outName0 = imgDir + headTag0;
      string outName1 = imgDir + headTag1;
      
      /*................
	Prepare pointers */
           
      complex<double> *Sqx, *Sqy, *Sqz;

      double **specField, **CMT_spcFd;

      int **locFieldHist;     
      
      Sqx = new complex<double>[Nq];
      Sqy = new complex<double>[Nq];
      Sqz = new complex<double>[Nq];
        
      specField = Alloc_dble_array(Nq, 3);
      CMT_spcFd = Alloc_dble_array(Nq, 3);

      locFieldHist = Alloc_intg_array(NHist, 3);
  
      /*........................................
	Open samples file & prepare CMT-pointers */
      
      fieldInput.open(samplesFile, ios::in | ios::binary);

      init_dble_array(CMT_spcFd, Nq, 3, 0.0);

      init_intg_array(locFieldHist, NHist, 3, 0);
      
      /*..................................
	Check DFT procedure from Intel-MKL */

      flag = 0; //( look for 'test_iMKL_DFT2d' )

      cerr << " Testing the DFT procedures ...";

      auto time1 = high_resolution_clock::now();
      
      if (!FFTW_ON)
	{
	  if (disOrder)
	    {		    
	      for (k = 0; k < Ns; k++)
		{		      
		  fieldInput.read
		    (reinterpret_cast<char*>(&impField[k]), intgSz);
		}
	    }////[ Skip 1st impurity map on file header ]
	  
	  for (k = 0; k < Ns; k++)
	    {
	      fieldInput.read
		(reinterpret_cast<char*>(spinField[k]), szSpinVec);
	    }
 
	  flag = test_iMKL_DFT2d(spinField, Sqx, Sqy, Sqz);
	}

      /*............................
	Read all samples and perform 
	some procedures (see below); */

      norm = 0.0; //( look for 'fieldNorm' )

      if (flag == 0)
	{
	  cerr << " Okay! (test_iMKL_DFT2d)\n\n";

	  cerr << " Procedures in execution (all samples): \n"
	       << " - make spin configuration vector-plot; \n"
	       << " - record all impurity maps on video;   \n"
	       << " - compute local-field histograms;      \n"
	       << " - calculate average static SF;         \n";

	  cerr << endl;
	}
      else
	{ cerr << " Failed! (test_iMKL_DFT2d)\n" << endl; }
      
      if (flag == 0)
	{          
	  fieldInput.seekg(0, ios::beg);
	  
	  for (n = 0; n < NSamps; n++)
	    {
	      recKey = false;
	      
	      if ( (disOrder) && (n % PackSz == 0) )
		{		  
		  for (k = 0; k < Ns; k++)// Read impurity map of
		    {		          // each pack header ...
		      fieldInput.read
			(reinterpret_cast<char*>(&impField[k]), intgSz);
		    }

		  recKey = true;
		}
	      
	      for (k = 0; k < Ns; k++)// Read spin field ...
		{
		  fieldInput.read(reinterpret_cast<char*>(spinField[k]), szSpinVec);		  
		}

	      for (k = 0; k < Ns; k++)
		{
		  get_localField(k, spinField, locField);
	      
		  nx = get_HistIndex(locField[0], maxLocField, boxFc, NHist);
		  ny = get_HistIndex(locField[1], maxLocField, boxFc, NHist);
		  nz = get_HistIndex(locField[2], maxLocField, boxFc, NHist);

		  locFieldHist[nx][0] += 1;
		  locFieldHist[ny][1] += 1;
		  locFieldHist[nz][2] += 1;
		}

#if WITH_OPENCV == 1 //| Generate image plots from samples
	      /////////| and record impurity map to video:

	      if (n % 5 == 0)//( Image generation step : 5 )
		{
		  stag = to_string(n);
	  
		  tailName = stag + ").png";
	      
		  if (recKey)
		    {		  
		      make_latticeGrid(gridIMap, "SKIP");

		      gridMat = gridIMap.clone();
		  
		      if (recImpMap)
			{
			  imapVideo.write(gridIMap);
			}
		    }
		  
		  outImg0 = outName0 + tailName;
		  	      
		  infoStr = magnetInfoString(Temp, spinField);
		
		  ftag = " sample : " + stag;

		  make_vecField(infoStr, ftag,
				spinField, gridMat, vecsMat);
	      
		  imwrite(outImg0, vecsMat);

		  if (recOprMap)
		    {
		      outImg1 = outName1 + tailName;
	      
		      get_qctSMagField(spinField, orderMap);		     	

		      make_vecMap(infoStr, ftag, orderMap, gridMat, vecsMat);

		      imwrite(outImg1, vecsMat);

		      omapVideo.write(vecsMat);
		    }
		}
#endif//////////[ End of imaging procedures ]

	      norm += fieldNorm(spinField);

	      if ((qcrystal) || (PBC_OFF))
		{
		  get_qct_SqData(spinField, Sqx, Sqy, Sqz); }
	      else
		{
		  if (FFTW_ON)
		    {
		      get_SqData_FFTW(spinField, Sqx, Sqy, Sqz); }
		  else
		    { get_SqData_iMKL(spinField, Sqx, Sqy, Sqz); }
		  }
		
	      for (k = 0; k < Nq; k++)// Impose cutoff, look for
		{                     // spec0Max (subrouts.cpp);
		  rx = abs(Sqx[k]);
		  ry = abs(Sqy[k]);
		  rz = abs(Sqz[k]);
      
		  Sqx[k] = min(spec0Max, rx) * Sqx[k] / rx;
		  Sqy[k] = min(spec0Max, ry) * Sqy[k] / ry;
		  Sqz[k] = min(spec0Max, rz) * Sqz[k] / rz;
		}
		
	      get_StaticSFac(Sqx, Sqy, Sqz, specField);
	  
	      for (k = 0; k < Nq; k++)
		{
		  for (m = 0; m < 3; m++)
		    {
		      CMT_spcFd[k][m] += specField[k][m];
		    }
		}///| Accumulate spectral-data ...
	    }
	}

      n0 = round(norm * (1.0 / NSamps) - 1.0);
      
      if (n0 > 0)
	{
	  cerr << " Failed! Invalid data...\n\n";
	}
      
      /*.............................
	Compute average values of the
	static SF and record results */

      flag += n0;
      
      if (flag == 0)
	{
	  auto time2 = high_resolution_clock::now();

	  auto dtime = time2 - time1;
      
	  m = max(1, static_cast<int>
		  (duration_cast<milliseconds>(dtime).count()));
      
	  cerr << " Done! " << m << " ms\n\n";
	        
	  /* .......................................... */

	  double **xySpField, **yzSpField;
	  double **zzSpField, **ttSpField;

	  const double fc = 1.0 / max(NSamps - 1, 1);
  	      
	  xySpField = Alloc_dble_array(Gsz, Gsz);
	  yzSpField = Alloc_dble_array(Gsz, Gsz);
	  zzSpField = Alloc_dble_array(Gsz, Gsz);
	  ttSpField = Alloc_dble_array(Gsz, Gsz);
              
	  for (k = 0; k < Nq; k++)
	    {
	      for (m = 0; m < 3; m++)
		{
		  specField[k][m] = fc * CMT_spcFd[k][m];
		}
	    }////[ Compute average spectral-data ]

	  get_OrderedSpecArray2D(specField,
				 xySpField, yzSpField,
				 zzSpField, ttSpField);

	  /* .......................................... */
  
	  string str1, str2, str3, str4;

	  string baseName = "MCSamples_DFT_";

	  str1 = baseName + "xySpec" + tTagDat;
	  str2 = baseName + "yzSpec" + tTagDat;
	  str3 = baseName + "zzSpec" + tTagDat;
	  str4 = baseName + "ttSpec" + tTagDat;
      
	  record_SpecArray2D(xySpField, str1, 1); 
	  record_SpecArray2D(yzSpField, str2, 1); 
	  record_SpecArray2D(zzSpField, str3, 1); 
	  record_SpecArray2D(ttSpField, str4, 1); 

	  deAlloc_dble_array(xySpField, Gsz, Gsz);
	  deAlloc_dble_array(yzSpField, Gsz, Gsz);
	  deAlloc_dble_array(zzSpField, Gsz, Gsz);
	  deAlloc_dble_array(ttSpField, Gsz, Gsz);
      	}

      /*..................................
	Record local-spin-field histograms */

      string fname = "locFieldHist";

      double histNorm[3] = {0.0 , 0.0 , 0.0};
      
      double xpos, hval; ofstream recHist;

      fname += tTagDat;
      
      recHist.open(outDir1 + subDir1 + fname);

      for (k = 0; k < NHist; k++)
	{
	  for (n = 0; n < 3; n++)
	    {
	      histNorm[n] += locFieldHist[k][n];
	    }
	}
	
      for (k = 0; k < NHist; k++)
	{
	  xpos = k * boxSz - maxLocField;
	  
	  recHist << fmtDbleFix(xpos, 4, 8) << X2;

	  for (n = 0; n < 3; n++)
	    {
	      hval = locFieldHist[k][n] / histNorm[n];
	      
	      recHist << fmtDbleFix(hval, 4, 10) << X2;
	    }
	  
	  recHist << endl;
	}

      recHist.close();
      
      /*........................
	Close file & free memory */

      fieldInput.close();      
      
      delete[] Sqx;
      delete[] Sqy;
      delete[] Sqz;
      
      if (disOrder){ delete[] impField; }

      deAlloc_dble_array(specField, Nq, 3);
      deAlloc_dble_array(CMT_spcFd, Nq, 3);

      deAlloc_intg_array(locFieldHist, NHist, 3);

#if WITH_OPENCV == 1
      ///
      gridMat.release();
      vecsMat.release();

      if (recOprMap)
	{
	  delete[] orderMap;
	  
	  omapVideo.release();
	}
      
      if (recImpMap)
	{      
	  gridIMap.release();

	  imapVideo.release();
	}
      //| Release memory from
#endif//| Mat-type objects...
    }

  /*----------------------------------
    Free pointers used by subroutines:
    make_latticeFigure & make_rvecList */

  if (!qcrystal)
    {
      if (multiSubs)
	{
	  deAlloc_dble_array(r0List, Ns0, 2);
	}

      if (with_DFTcodes)
	{
	  deAlloc_dble_array(dftGrid, Nsg, 2);
	}
    }

  /*-----------------------------------
    Stop simulation if 'flag' is finite
    or if the user chooses to candel it */
  
#if AUTO_START == 0
  ///
  if ((iAmRoot) && (flag == 0))
    {
      cerr << " Proceed? (1/0) : ";

      cin >> ch0; cerr << endl;  

      if (ch0 != 1)
	{
	  cerr << " Code terminated ...\n\n";

	  flag = 1;
	}
    }///| User may stop the simulation
#endif//| here to study the recorded data...

  if (IsiModel)
    {
      if (iAmRoot)
	{
	  cerr << " Code-halt: this code is not valid\n"
	       << " for Ising models, code terminated!!";
	  
	  cerr << "\n" << endl;
	}
      
      flag = 1;
    }
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){MPI_Finalize(); return 0;}

  /* |====================================
     | Proceed with the temporal evolution
     | or skip to the next code stage ... */
  ///|
  if (!getEvoData)
    {  
      /*------------------------------
	Allocate & initiate CMT arrays */

      CMT_qSStVec = new double[ntm];

      for(n = 0; n < ntm; n++){CMT_qSStVec[n] = 0.0;}
      
      CMT_SqxWVec = Alloc_dble_array(Nq, ntm);
      CMT_SqyWVec = Alloc_dble_array(Nq, ntm);
      CMT_SqzWVec = Alloc_dble_array(Nq, ntm);

      init_dble_array(CMT_SqxWVec, Nq, ntm, 0.0);
      init_dble_array(CMT_SqyWVec, Nq, ntm, 0.0);
      init_dble_array(CMT_SqzWVec, Nq, ntm, 0.0);

      /*---------------------------
	Define temporal window filter
	for temporal integration ... */

      double *tmWinVec = new double[ntm];
  
      if (iAmRoot)
	{
	  /*...................................
	    Define output file strings & tWForm */
      
	  string prefix1 = outDir1 + "RK_Check/0WinTVec_";
	  string prefix2 = outDir1 + "RK_Check/0WinSpec_";

	  string strWin1 = prefix1 + tWForm + ".dat";
	  string strWin2 = prefix2 + tWForm + ".dat";
      
	  cerr << " Temporal window form is set to: ";

	  cerr << tWForm << "\n" << endl;	 
	  
	  if (tWForm == "wread")
	    {
	      double fc, twval, twmax = 0.0;
	      
	      size_t szTVec = ntm * dbleSz;

	      ifstream inData(fwindowFile, ios::in | ios::binary);
     	      
	      inData.read(reinterpret_cast<char*>(tmWinVec), szTVec);

	      for (n = 0; n < ntm; n++)
		{
		  twval = tmWinVec[n];

		  if (twval > twmax){ twmax = twval; }
		}

	      fc = 1.0 / twmax;

	      for (n = 0; n < ntm; n++)
		{
		  twval = fc * tmWinVec[n];

		  tmWinVec[n] = twval;
		}

	      inData.close();
	    }
	  else//( Predefined window forms )
	    {
	      if (tWForm == "nowin")
		{
		  for (n = 0; n < ntm; n++)
		    {
		      tmWinVec[n] = 1.0;
		    }
		}
	      else if (tWForm == "gauss")
		{
		  gaussWindow(ntm, tmWinVec);
		}
	      else if (tWForm == "ntall")
		{
		  blackNtWindow(ntm, tmWinVec);
		}
	      else if (tWForm == "black")
		{
		  blackHrWindow(ntm, tmWinVec);
		}
	      else if (tWForm == "flatw")
		{
		  flatTopWindow(ntm, tmWinVec);
		}
	      else if (tWForm == "lancz")
		{
		  lanczosWindow(ntm, 0.5, tmWinVec);
		}
	      else if (tWForm == "sincw")
		{
		  sincWindow(ntm, 1.0, tmWinVec);
		}
	      else//( Hann-window ) 
		{
		  if (tWForm == "hann1"){ m0 = 1; }
		  if (tWForm == "hann2"){ m0 = 2; }
		  if (tWForm == "hann3"){ m0 = 3; }
		  
		  hannWindow(ntm, m0, tmWinVec);
		}		
	    }//// tWForm CHECK;

	  /*......................
	    RK4 evolution settings */

	  cerr << " Evolution & spectral information:\n";
	  cerr << " ( see RK_Check/0RK_Evo_Info.txt )\n" << endl;
	    
	  cerr << " dtm  = " << fmtDbleFix(dtm , 5, 9) << " , ";
	  cerr << " tmax = " << fmtDbleFix(tmax, 5, 9) <<  endl;
	  cerr << " dwf  = " << fmtDbleFix(dwf , 5, 9) << " , ";
	  cerr << " wmax = " << fmtDbleFix(wmax, 5, 9) <<  endl;

	  cerr << endl;

	  /*........................
	    Set complex input vector */

	  complex<double> *twvec;

	  twvec = new complex<double>[ntm];

	  prepare_wPlan_fftw1D();

	  for (n = 0; n < ntm; n++)
	    {
	      twvec[n] = complex<double>(tmWinVec[n], 0.0);
	    }
      
	  set_Input_fftw1D(twvec);
      
	  /*.......................
	    Compute window spectrum */
     
	  double *wvec0, *wvec1, wref;

	  complex<double> wc;
 
	  wvec0 = new double[ntm];
	  wvec1 = new double[ntm];
      
	  fftw_execute(wPlan);
    
	  for (n = 0; n < ntm; n++)
	    {	   
	      wc = complex<double>(wfData[n][0], wfData[n][1]);

	      wvec0[n] = (1.0 / ntm) * real(wc * conj(wc));
	    }

	  get_OrderedSpecArray1D(wvec0, wvec1);

	  wref = wvec1[0]; //( Reference value )

	  for (n = 1; n < ntm; n++)
	    {
	      if (wvec1[n] > wref){ wref = wvec1[n]; }
	    }
      
	  /*.................................
	    Record window form & its spectrum */

	  ofstream recWin1(strWin1, ios::out);
	  ofstream recWin2(strWin2, ios::out);

	  double fs0 = 2.0 / ntm;

	  double time, freq, ampSpec;
      
	  for (n = 0; n < ntm; n++)
	    {
	      time = n * dtm;

	      freq = n * fs0 - 1.0;

	      ampSpec = log10(wvec1[n] / wref);
	  
	      recWin1 << time << X3 << tmWinVec[n] << endl;
	  
	      recWin2 << freq << X3 << ampSpec << endl;
	    }
      
	  /*.........................
	    Free memory & close files */

	  destroy_wPlan_fftw1D();
	  
	  recWin1.close();
	  recWin2.close();

	  delete[] twvec;
      
	  delete[] wvec0;
	  delete[] wvec1;
	}

      MPI_Bcast(tmWinVec, ntm, MPI_DOUBLE, root, MPI_COMM_WORLD);

      /*----------------------------
	Set and initialize impurity  
	map reading/checking control */

      int IMapCounter = 0;
      
      bool getIMap = false;

      int *impTVec; double *impRVec; 
      
      double impTVal, impRVal, impRatioRef; 

      if (disOrder)//( Impurity map field/vector )
	{
	  impField = new int[Ns];

	  make_impurityField(impRatio, flag);

	  impRatioRef = impurityRatio();

	  if (iAmRoot)
	    {	      
	      impTVec = new int[wSize];

	      impRVec = new double[wSize];
	    }
	}////[ Pointers for disorder-mode ]
            
      /*-------------------------
	Allocate transfer pointer */
 
      const int dptr1 = 3 * Ns;
  
      double *vecField = new double[dptr1];
      
      /*-----------------------
	RK evolution begins now */

      MPI_File sField_Input;

      MPI_Status status;

      MPI_Offset offset;
	    
      if (!pcMode)
	{ 
	  if (iAmRoot)
	    {
	      fieldInput.open(samplesFile, ios::in | ios::binary);
	    }
	}
      else //( wSize > 1 )
	{
	  MPI_File_open(MPI_COMM_WORLD, samplesFile.c_str(),
			MPI_MODE_RDONLY, MPI_INFO_NULL, &sField_Input);   
	}
      
      if (iAmRoot)
	{
	  if (RKCheck)
	    {
	      cerr << " Single sample test enabled!" << endl;
	    }
	  else//( Normal execution )
	    {
	      string str0 = X2;

	      if (NTest > 0){ str0 = " / Test "; }
	      
	      cerr << " RK-samples information \n"
		   << " ---------------------- \n"
		   << " | NSamples = " << NSamps << str0 << endl
		   << " | PackSize = " << PackSz << endl
		   << " | NPacks   = " << NPacks << endl
		   << " | NTreads  = " << wSize  << endl
		   << " | NSets    = " << NSets  << endl;
	    }

	  cerr << "\n RK-evolution running ... ";
	}

      auto time1 = high_resolution_clock::now();

      evoCnt = 0; //( evolution counter )
      
      /// RK_LOOP (Iterator: nrk | START)
      ///      
      for (int nrk = 0; nrk < NSets; nrk++)//( nrk < NSets )
	{
	  /*.......................
	    Check current iteration	   

	    If the disorder-mode is enabled, then, after
	    each pack of samples processed, the impurity
	    map must be updated (see the code below ...); */

	  if (disOrder)
	    {
	      ns = nrk * wSize;
	  
	      if (ns % PackSz == 0)
		{
		  getIMap = true;
		}
	    }//// Unlock reading-code [#];

	  /*.......................................
	    Get spin-configuration & impurity field */

	  halt = 0; //( Impurity ratio check )
	    
	  if (!pcMode)
	    {
	      if (getIMap)//[#]
		{	  
		  for (k = 0; k < Ns; k++)
		    {		      
		      fieldInput.read
			(reinterpret_cast<char*>(&impField[k]), intgSz);
		    }

		  getIMap = false; //( Lock reading for next iterarions )
		  
		  if (impurityRatio() != impRatioRef){ halt = 1; }
		}
	      
	      for (k = 0; k < Ns; k++)
		{
		  fieldInput.read
		    (reinterpret_cast<char*>(spinField[k]), szSpinVec);
		}

	      norm = fieldNorm(spinField);
	      
	      if (norm > 1.0){ flag = 1; }
	    }
	  else//[ Parallel-data read ( wSize > 1) ]
	    {
	      offset = nrk * szCSet + IMapCounter * szIMap;
	      
	      if (getIMap)//[#]
		{
		  MPI_File_read_at(sField_Input, offset,
				   impField, Ns, MPI_INT, &status);

		  IMapCounter += 1; // Increase impurity field/map offset counter;
		  
		  offset += szIMap; // Add offset for the spin field reading part;

		  getIMap = false;  // Lock this reading-code for next iterarions;

		  impTVal = impurityTrace();
		  impRVal = impurityRatio();

		  MPI_Gather(&impTVal, 1, MPI_INT, impTVec, 1,
			     MPI_INT, root, MPI_COMM_WORLD);
		  
		  MPI_Gather(&impRVal, 1, MPI_DOUBLE, impRVec, 1,
			     MPI_DOUBLE, root, MPI_COMM_WORLD);

		  if (iAmRoot)// Data-check (START);
		    {
		      impTVal = impTVec[root]; // Impurity field
		      impRVal = impRVec[root]; // trace/ratio value;
		      
		      if (impRVal != impRatioRef)
			{
			  halt = 1;
			}
		      else//( Check values across the MPI-world )
			{
			  for (n = 1; n < wSize; n++)
			    {
			      if ( impTVec[n] != impTVal ||
				   impRVec[n] != impRVal )
				{
				  halt = 1;
				}
			    }
			}//// Root process
		    }//////// data-check (END);

		  MPI_Barrier(MPI_COMM_WORLD);
		}

	      offset += wRank * szConf;

	      MPI_File_read_at(sField_Input, offset,
			       vecField, dptr1, MPI_DOUBLE, &status);

	      dbleReshape2D(Ns, 3, vecField, spinField);

	      norm = fieldNorm(spinField);
	      
	      MPI_Reduce(&norm, &normSum, 1, MPI_DOUBLE,
			 MPI_SUM, root, MPI_COMM_WORLD);

	      if (iAmRoot)
		{
		  norm = fcw * normSum;
	      
		  if (norm > 1.0)
		    {
		      flag = 1; }
		  else
		    { flag = 0; }
		}
	    }////[ Closure of the reading-codes ]

	  MPI_Bcast(&halt, 1, MPI_INT, root, MPI_COMM_WORLD);

	  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  

	  /*...............................
	    Verify the data integrity flags */
	  
	  if (halt > 0)//( Impurity-maps CHECK )
	    {
	      if (iAmRoot)
		{
		  cerr << " \n\n";

		  if (!pcMode)
		    {
		      cerr << " Error: impurity ratio-check failed! \n"; }
		  else
		    { cerr << " Error: impurity fields do not match!\n"; }

		  waitAndJump();
		}

	      MPI_Barrier(MPI_COMM_WORLD);
	      
	      MPI_Finalize(); return 0;
	    }

	  if (flag > 0)//( Spin-field samples CHECK )
	    {
	      if (iAmRoot)
		{
		  cerr << " \n\n Error: invalid spin-field data! \n";

		  waitAndJump();
		}

	      for (n = 0; n < wSize; n++)
		{
		  if (wRank == n)
		    {
		      norm = fieldNorm(spinField);
		      
		      cerr << " wRank : " << iformat2(wRank) << " ,";
		      
		      cerr << " fNorm : " << fmtDbleSci(norm, 5, 8) << endl;
		    }

		  MPI_Barrier(MPI_COMM_WORLD);
		}

	      if (iAmRoot){ waitAndJump(); }
	      
	      MPI_Barrier(MPI_COMM_WORLD);
	      
	      MPI_Finalize(); return 0;
	    }
   
	  /*.......................................
	    Perform measurement of the dynamical SF

	    Notes about the procedure:
    
	    1) The current spin configuration is used as
	    input for the real time evolution of the 2D
	    lattice system, which is described by the
	    dynamics of the associated Heisenberg
	    equations (coupled ODEs);

	    2) The procedure below is based on the 4th
	    order RK method, the resulting data within
	    the 'tSeries' type pointers is then accumu-
	    lated in the root-only CMT-type arrays; */
	  
	  get_dynSpectrum(wRank, tmWinVec, spinField,
			  SqxWVec, SqyWVec, SqzWVec,
			  qSStVec);
	  
	  for (k = 0; k < Nq; k++)
	    {
	      for (n = 0; n < ntm; n++)
		{
		  CMT_SqxWVec[k][n] += SqxWVec[k][n];
		  CMT_SqyWVec[k][n] += SqyWVec[k][n];
		  CMT_SqzWVec[k][n] += SqzWVec[k][n];
		}
	    }

	  for (n = 0; n < ntm; n++)
	    {
	      CMT_qSStVec[n] += qSStVec[n];
	    }

	  if (iAmRoot){ evoCnt += wSize; }

#if WITH_OPENCV == 1
	  ///
	  if (recSpinVec)
	    {
	      if (iAmRoot)
		{
		  cerr << "Evo-movies recorded!";
		}
	      
	      break; //( Only a single iteration )
	    }
	  ///
#endif///( Report information )
 
	  if (iAmRoot){ report(nrk, NSets); }
      
	}/* RK_LOOP  (END)
	    Iterator: nrk */
  
      MPI_Barrier(MPI_COMM_WORLD);

      /*------------------------------
	Allocate xyz-transfer pointers */

      const int dptr2 =  Nq * ntm;
  
      double *xVecData;
      double *yVecData;
      double *zVecData;
      
      xVecData = new double[dptr2];
      yVecData = new double[dptr2];
      zVecData = new double[dptr2];

      /*---------------------------------
	Transfer data to main/root thread */

      if (pcMode) // GET-WDATA (START, wSize > 1)
	{
	  /* Below, each working process send the data
	     obtained before to the root process ... */
	  
	  for (i = 1; i < wSize; i++)
	    {  
	      if (wRank == i)
		{
		  dbleFlatten2D(Nq, ntm, CMT_SqxWVec, xVecData);
		  dbleFlatten2D(Nq, ntm, CMT_SqyWVec, yVecData);
		  dbleFlatten2D(Nq, ntm, CMT_SqzWVec, zVecData);
		      		      			
		  MPI_Send(xVecData, dptr2, MPI_DOUBLE,
			   root, mpi_ctag, MPI_COMM_WORLD);

		  MPI_Send(yVecData, dptr2, MPI_DOUBLE,
			   root, mpi_ctag, MPI_COMM_WORLD);

		  MPI_Send(zVecData, dptr2, MPI_DOUBLE,
			   root, mpi_ctag, MPI_COMM_WORLD);

		  MPI_Send(CMT_qSStVec, ntm, MPI_DOUBLE,
			   root, mpi_ctag, MPI_COMM_WORLD);
		}

	      if (wRank == root) // ROOT-WORK (START)
		{		      
		  MPI_Recv(xVecData, dptr2,
			   MPI_DOUBLE, i, mpi_ctag,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  MPI_Recv(yVecData, dptr2,
			   MPI_DOUBLE, i, mpi_ctag,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  MPI_Recv(zVecData, dptr2,
			   MPI_DOUBLE, i, mpi_ctag,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  MPI_Recv(qSStVec, ntm, MPI_DOUBLE, i, mpi_ctag,
			   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		  dbleReshape2D(Nq, ntm, xVecData, SqxWVec);
		  dbleReshape2D(Nq, ntm, yVecData, SqyWVec);
		  dbleReshape2D(Nq, ntm, zVecData, SqzWVec);
		  
		  for (k = 0; k < Nq; k++)
		    {
		      for (n = 0; n < ntm; n++)
			{
			  CMT_SqxWVec[k][n] += SqxWVec[k][n];
			  CMT_SqyWVec[k][n] += SqyWVec[k][n];
			  CMT_SqzWVec[k][n] += SqzWVec[k][n];
			}
		    }

		  for (n = 0; n < ntm; n++)
		    {
		      CMT_qSStVec[n] += qSStVec[n];
		    }
		}//// ROOT-WORK (END)
	    
	      MPI_Barrier(MPI_COMM_WORLD);
	    }	  	  
	}//// GET-WDATA (END)

      delete[] xVecData;
      delete[] yVecData;
      delete[] zVecData;

      /*-------------------
	Deallocate pointers */

      delete[] tmWinVec;
      delete[] vecField;		       

      if (disOrder)
	{
	  delete[] impField;

	  if (iAmRoot)
	    {
	      delete[] impRVec;
	      delete[] impTVec;
	    }
	}////[ Pointers for disorder-mode ]

      /*------------------------
	Close input/output files */
      
      if (pcMode)
	{ 
	  MPI_File_close(&sField_Input);    
	}
      else //( wSize = 1 )
	{	 	      
	  fieldInput.close();
	}
      
      /*----------------------
	Print some information */
  
      unsigned int RK_time;
    
      auto time2 = high_resolution_clock::now();

      auto dtime = time2 - time1;

      RK_time = duration_cast<milliseconds>(dtime).count();

      if(iAmRoot)
	{
	  cerr << "\n" << endl;
      
	  cerr << " Time elapsed: " << RK_time << " ms\n";

	  waitAndJump();
	}
    }
  else//( If getEvoData is true )
    {
      if (iAmRoot)
	{      
	  cerr << " RK-evolution skipped!\n";

	  waitAndJump();
	}
    }//// END of getEvoData-barrier;
    
  MPI_Barrier(MPI_COMM_WORLD);

  /*----------------------------
    Deallocate/destroy DFT plans

    FFTW_ON --> destroy_xyzPlan_fftw2D();
	
    Note: FFTW is not used anymore,
    ..... MKL-Dfti replaced it; */
		        
  if (!qcrystal) // Free FFT-plan (MKL's FFT);
    {
      mklStat = DftiFreeDescriptor(&handle);
    } 
  
  //=============================================
  // Record average SF-data in zone & path forms:

  /*---------------------
    Pointers description:
    
    pathLPos : length position on the path asso-
    .......... ciated with each point on SpPath;
    
    The double-type pointer 'pathLPos' provides
    the distance along the KGMYG/YGMXG path asso-
    ciated with each point (wavevector) within it,
    this object is useful when recording the struc-
    ture factor amplitude along this path (SpPath); */
 
  if ((!RKCheck) && (iAmRoot))// Root-WORK (START)
    {
      /* Define size constants */
      
      const size_t szEvoDat = ntm * dbleSz;

      const size_t szSpcDat = szEvoDat * (3 * Nq + 1);
      
      /* Define string variables */  

      streampos szFile;

      ostringstream wOss;

      ifstream testFile, specInput;
      
      ofstream outfile1, outfile2, outfile3;
      
      ofstream outfile4, outfile5, Bin0;

      string str1, str2, str3, str4, str5, wStr;

      string fileTag = "MC_Bin_SpecData";

      string fileName = outDir0 + fileTag + tTagBin;

      /* Define double-type variables */    
      
      double fc, wfc, qx, qy, z0, w0, cut1, cut2;

      double qtFac, itFac, wfreq, reTime, lpVal, qSSVal;
      
      double xVal, yVal, zVal, xyVal, yzVal, zzVal, ttVal;

      Vec2d wref, wvec; vector<Vec2d> bwvecPath;

      /*...........................................
	Get spectral time-series data from file or
	compute average data from CMT-type pointers */
      
      if (getEvoData)
	{
	  /* Check if the input file exist */
	  
	  testFile.open(fileName, ios::binary);
      
	  if (!testFile.is_open())
	    {
	      cerr << " Error: could not find input file!\n"
		   << " Input: " << fileName;

	      cerr << "\n" << endl; flag = 1;
	    }
	  else
	    { testFile.close(); flag = 0; }

	  /* Check if the file has a valid amount of data */
	  
	  if (flag == 0)
	    {
	      specInput.open(fileName, ios::in | ios::binary);

	      specInput.seekg(0, ios::end);
	
	      szFile = specInput.tellg();

	      if (szFile != szSpcDat)
		{
		  cerr << " Error: input file has no valid data!\n"
		       << " Input: " << fileName << endl;

		  cerr << " \n Measured size (in bytes): " << szFile
		       << " \n Expected size (in bytes): " << szSpcDat;

		  cerr << "\n" << endl; flag = 1;
		}
	      else { flag = 0; }
	    }

	  /* Read data from file */
	  
	  if (flag == 0)
	    {
	      specInput.seekg(0, ios::beg);
	      	      
	      for (k = 0; k < Nq; k++)
		{
		  specInput.read(reinterpret_cast<char*>(SqxWVec[k]), szEvoDat);
		  specInput.read(reinterpret_cast<char*>(SqyWVec[k]), szEvoDat);
		  specInput.read(reinterpret_cast<char*>(SqzWVec[k]), szEvoDat);
		}

	      for (n = 0; n < ntm; n++)
		{
		  specInput.read(reinterpret_cast<char*>(&qSStVec[n]), dbleSz);
		}
	    }//// Reading stage (END);

	  specInput.close();
	}
      else//( Use the spectral-data computed during code execution )
	{
	  flag = 0;   

	  fc = 1.0 / evoCnt;
  
	  for (k = 0; k < Nq; k++)
	    {
	      for (n = 0; n < ntm; n++)
		{	    	      
		  SqxWVec[k][n] = fc * CMT_SqxWVec[k][n];
		  SqyWVec[k][n] = fc * CMT_SqyWVec[k][n];
		  SqzWVec[k][n] = fc * CMT_SqzWVec[k][n];
		}	
	    }

	  for (n = 0; n < ntm; n++)
	    {
	      qSStVec[n] = fc * CMT_qSStVec[n];
	    }
     
	  /* Record average spectral data (binary form) */

	  remove(fileName.c_str()); //( Delete file if it exists )

	  Bin0.open(fileName, ios::out | ios::binary | ios::app);
	  
	  for (k = 0; k < Nq; k++)
	    {
	      Bin0.write(reinterpret_cast<const char*>(SqxWVec[k]), szEvoDat);
	      Bin0.write(reinterpret_cast<const char*>(SqyWVec[k]), szEvoDat);
	      Bin0.write(reinterpret_cast<const char*>(SqzWVec[k]), szEvoDat);
	    }

	  for (n = 0; n < ntm; n++)
	    {
	      Bin0.write(reinterpret_cast<const char*>(&qSStVec[n]), dbleSz);
	    }
	  
	  Bin0.close();
	}

      if (flag > 0)
	{
	  cerr << " Failed to get spectral time-series data!";

	  cerr << "\n" << endl;
	}
      else//( Normal execution path if getEvoData is false )
	{
	  /*.............................................
	    Apply quantum correction to the spectrum data */

	  itFac = tmax / pi2; //( time-integral factor )
	  
	  for (k = 0; k < Nq; k++)
	    {
	      qtFac = 1.0; //( Zero-frequency value )

	      for (n = 0; n < ntm; n++)
		{
		  wfreq = n * dwf; // Frequency value;

		  fc = Beta * wfreq; // Auxiliary factor;

		  if (n > 0) // Set quantum correction factor;
		    {
		      qtFac = fc / (1.0 - exp(- fc));
		    }
	      
		  xVal = itFac * qtFac * SqxWVec[k][n];
		  zVal = itFac * qtFac * SqyWVec[k][n];
		  zVal = itFac * qtFac * SqzWVec[k][n];
		  
		  SqxWVec[k][n] = xVal;
		  SqyWVec[k][n] = yVal;
		  SqzWVec[k][n] = zVal;
		}		      
	    }
	  
	  /*...........................................
	    Extract spectrum data along the predefined 
	    path (KGMYG/YGMXG) for positive frequencies

	    npw = ntm / 2 : number of positive freqs. 
	    ............... points including the 0Hz; */

	  double *xySpecPath = new double[npPath];
	  double *yzSpecPath = new double[npPath];
	  double *zzSpecPath = new double[npPath];
	  double *ttSpecPath = new double[npPath];
	  
	  double **xySpPathWVec, **yzSpPathWVec;
	  double **zzSpPathWVec, **ttSpPathWVec;

	  double **xySpecArray, **yzSpecArray;
	  double **zzSpecArray, **ttSpecArray;
	  
	  double **specField;

	  specField = Alloc_dble_array(Nq, 3);

	  xySpecArray = Alloc_dble_array(Nq, Nq);
	  yzSpecArray = Alloc_dble_array(Nq, Nq);
	  zzSpecArray = Alloc_dble_array(Nq, Nq);
	  ttSpecArray = Alloc_dble_array(Nq, Nq);

	  xySpPathWVec = Alloc_dble_array(npPath, npw);
	  yzSpPathWVec = Alloc_dble_array(npPath, npw);
	  zzSpPathWVec = Alloc_dble_array(npPath, npw);
	  ttSpPathWVec = Alloc_dble_array(npPath, npw);
	  
	  for (n = 0; n < npw; n++)
	    {	      
	      for (k = 0; k < Nq; k++) // Get spectrum slice;
		{
		  specField[k][0] = SqxWVec[k][n];
		  specField[k][1] = SqyWVec[k][n];
		  specField[k][2] = SqzWVec[k][n];		  
		}
	  
	      get_OrderedSpecArray2D(specField,
				     xySpecArray, yzSpecArray,
				     zzSpecArray, ttSpecArray);

	      get_Spec_qPath(xySpecArray, xySpecPath);
	      get_Spec_qPath(yzSpecArray, yzSpecPath);
	      get_Spec_qPath(zzSpecArray, zzSpecPath);
	      get_Spec_qPath(ttSpecArray, ttSpecPath);

	      for (j = 0; j < npPath; j++)
		{  
		  xySpPathWVec[j][n] = xySpecPath[j];
		  yzSpPathWVec[j][n] = yzSpecPath[j];
		  zzSpPathWVec[j][n] = zzSpecPath[j];
		  ttSpPathWVec[j][n] = ttSpecPath[j];
		}	  
	    }
          
	  delete[] xySpecPath; delete[] yzSpecPath;
	  delete[] zzSpecPath; delete[] ttSpecPath;

	  /*.................................
	    Find maximum relevant frequencies */

	  double xyWfMax, yzWfMax;
	  double zzWfMax, ttWfMax;
	  
	  double wf1 = 0.0, sum1 = 0.0;
	  double wf2 = 0.0, sum2 = 0.0;
  	  double wf3 = 0.0, sum3 = 0.0;
	  double wf4 = 0.0, sum4 = 0.0;
	  
	  for (k = 0; k < npPath; k++)
	    {
	      for (n = 0; n < npw; n++)
		{
		  wfreq = n * dwf;
	  
		  xyVal = xySpPathWVec[k][n];
		  yzVal = yzSpPathWVec[k][n];
		  zzVal = zzSpPathWVec[k][n];
		  ttVal = ttSpPathWVec[k][n];

		  wf1 += xyVal * pow(wfreq, 8);
		  wf2 += yzVal * pow(wfreq, 8);
		  wf3 += zzVal * pow(wfreq, 8);
		  wf4 += ttVal * pow(wfreq, 8);

		  sum1 += xyVal; sum2 += yzVal;
		  sum3 += zzVal; sum4 += ttVal;
		}
	    }
	  
	  xyWfMax = 3.0 * sqrt(sqrt(sqrt(wf1 / sum1)));
	  yzWfMax = 3.0 * sqrt(sqrt(sqrt(wf2 / sum2)));
	  zzWfMax = 3.0 * sqrt(sqrt(sqrt(wf3 / sum3)));
	  ttWfMax = 3.0 * sqrt(sqrt(sqrt(wf4 / sum4)));

	  if (isnan(xyWfMax)){ xyWfMax = npw * dwf; }
	  if (isnan(yzWfMax)){ yzWfMax = npw * dwf; }
	  if (isnan(zzWfMax)){ zzWfMax = npw * dwf; }
	  if (isnan(ttWfMax)){ ttWfMax = npw * dwf; }

	  /*..............................................
	    Display maximum frequencies for relevant the
	    spectral data & compute the associated indices */

	  int xyIndex, yzIndex, zzIndex, ttIndex;

	  int n1, n2, nmax;
	  
	  cerr << " Max. relevant frequency:\n\n";

	  cerr << " xy : " << fmtDbleFix(xyWfMax, 2, 6) << endl;
	  cerr << " yz : " << fmtDbleFix(yzWfMax, 2, 6) << endl;
	  cerr << " zz : " << fmtDbleFix(zzWfMax, 2, 6) << endl;
	  cerr << " tt : " << fmtDbleFix(ttWfMax, 2, 6) << endl;

	  cerr << endl;

	  xyIndex = ceil(xyWfMax / dwf) + 1;
	  yzIndex = ceil(yzWfMax / dwf) + 1;
	  zzIndex = ceil(zzWfMax / dwf) + 1;
	  ttIndex = ceil(ttWfMax / dwf) + 1;

	  n1 = max(xyIndex, yzIndex);
	  n2 = max(zzIndex, ttIndex);
	  
	  nmax = max(n1, n2);

	  cerr << " Max. relevant frequency index:\n\n";

	  cerr << " nmax | npw : " << nmax << " | ";

	  cerr << npw << "\n" << endl;

	  nmax = npw; //( Recording full spectrum )
	    
	  /*...........................
	    Record 2D spectrum BZ1-data 
	    for some frequencies slices */
	  
	  const int nwSkip = ceil(0.1 * nmax);
	  
	  cerr << " Recording spectral slices ... ";
	  
	  for (n = 0; n < nmax; n++)
	    {
	      wfreq = n * dwf;
	      
	      if (n % nwSkip == 0)
		{                  		  	      	      
		  wOss << fixed
		       << setprecision(3) << wfreq;
  
		  wStr = wOss.str(); //( freq. string form )
  
		  str1 = "DFT_xySpec_wfreq(" + wStr + ")" + tTagDat;
		  str2 = "DFT_yzSpec_wfreq(" + wStr + ")" + tTagDat;
		  str3 = "DFT_zzSpec_wfreq(" + wStr + ")" + tTagDat;
		  str4 = "DFT_ttSpec_wfreq(" + wStr + ")" + tTagDat;

		  for (k = 0; k < Nq; k++) // Get spectrum slice;
		    {
		      specField[k][0] = SqxWVec[k][n];
		      specField[k][1] = SqyWVec[k][n];
		      specField[k][2] = SqzWVec[k][n];
		    }

		  get_OrderedSpecArray2D(specField,
					 xySpecArray, yzSpecArray,
					 zzSpecArray, ttSpecArray);		  
		  
		  record_SpecArray2D(xySpecArray, str1, 1); 
		  record_SpecArray2D(yzSpecArray, str2, 1); 
		  record_SpecArray2D(zzSpecArray, str3, 1); 
		  record_SpecArray2D(ttSpecArray, str4, 1); 

		  wOss.str(""); // Clear contents;
		}

	      report(n, nmax);
	    }

	  cerr << "\n" << endl;

	  /*.........................................
	    Record 2D spectrum BZ1-data in video form */
	  
#if WITH_OPENCV == 1
	  ///
	  /* Prepare video writer & files
  
	     1) Output video FPS;
	     2) Video writer codec;
	     3) Output videos base-name;
	     4) Tag showing output details;
	     5) Output evolution video name;
	     6) File-management string variables;
	     7) Mat-objects for video operations; */

	  if (recSpecMov)
	    {
	      int fps0 = 12; //(1) 

	      string cdc = "H264"; //(2)

	      string vdname1 = "xySpecEvo"; //(3)	     
	      string vdname2 = "yzSpecEvo";
	      string vdname3 = "zzSpecEvo";
	      string vdname4 = "ttSpecEvo";

	      string frmTag = "_Set" + to_string(wRank); //(4)

	      string outVid1 = outDir3 + vdname1 + frmTag + ".avi"; //(5)   
	      string outVid2 = outDir3 + vdname2 + frmTag + ".avi";
	      string outVid3 = outDir3 + vdname3 + frmTag + ".avi";
	      string outVid4 = outDir3 + vdname4 + frmTag + ".avi";
	  
	      string ftag, datNames[4]; //(6)

	      Mat xySpecFrm, yzSpecFrm; //(7)
	      Mat zzSpecFrm, ttSpecFrm;

	      /* Set dimensions of spectrum-plot videos */
	      
	      const int dpiFac = 300;
      
	      const int ncols = 4 * dpiFac;
	      const int nrows = 3 * dpiFac;

	      string dim1, dim2, szInfo;

	      dim1 = to_string(ncols) + X2;
	      dim2 = to_string(nrows) + X2;

	      szInfo = X2 + dim1 + dim2;

	      /* Prepare video-output */
      
	      unsigned int codec;
     
	      Size vdSz = Size(ncols, nrows);
      
	      VideoWriter xySpecVid, yzSpecVid, zzSpecVid, ttSpecVid;
      
	      codec = VideoWriter::fourcc(cdc[0], cdc[1], cdc[2], cdc[3]);
      
	      xySpecVid.open(outVid1, codec, fps0, vdSz);
	      yzSpecVid.open(outVid2, codec, fps0, vdSz);
	      zzSpecVid.open(outVid3, codec, fps0, vdSz);
	      ttSpecVid.open(outVid4, codec, fps0, vdSz);

	      cerr << " Recording spectral movie ... ";
	      
	      for (n = 0; n < nmax; n++)
		{	  		
		  ftag = "n=" + to_string(n);
	      
		  datNames[0] = "DFT_xySpec_frame" + frmTag + ".dat";
		  datNames[1] = "DFT_yzSpec_frame" + frmTag + ".dat";
		  datNames[2] = "DFT_zzSpec_frame" + frmTag + ".dat";
		  datNames[3] = "DFT_ttSpec_frame" + frmTag + ".dat";

		  for (k = 0; k < Nq; k++) // Get spectrum slice;
		    {
		      specField[k][0] = SqxWVec[k][n];
		      specField[k][1] = SqyWVec[k][n];
		      specField[k][2] = SqzWVec[k][n];
		    }
		  
		  get_OrderedSpecArray2D(specField,
					 xySpecArray, yzSpecArray,
					 zzSpecArray, ttSpecArray);

		  record_SpecArray2D(xySpecArray, datNames[0], 1);   
		  record_SpecArray2D(yzSpecArray, datNames[1], 1); 
		  record_SpecArray2D(zzSpecArray, datNames[2], 1); 
		  record_SpecArray2D(ttSpecArray, datNames[3], 1); 

		  make_specPlots(frmTag, szInfo, ftag,
				 datNames[0], xySpecFrm);
		  
		  make_specPlots(frmTag, szInfo, ftag,
				 datNames[1], yzSpecFrm);

		  make_specPlots(frmTag, szInfo, ftag,
				 datNames[2], zzSpecFrm);

		  make_specPlots(frmTag, szInfo, ftag,
				 datNames[3], ttSpecFrm);

		  xySpecVid.write(xySpecFrm);
		  yzSpecVid.write(yzSpecFrm);
		  zzSpecVid.write(zzSpecFrm);
		  ttSpecVid.write(ttSpecFrm);

		  report(n, nmax);
		}

	      cerr << "\n" << endl;

	      xySpecFrm.release(); xySpecVid.release();
	      yzSpecFrm.release(); yzSpecVid.release();
	      zzSpecFrm.release(); zzSpecVid.release();
	      ttSpecFrm.release(); ttSpecVid.release();
	    }
#endif///////( Record spectral movie | recSpecMov = true )

	  deAlloc_dble_array(specField, Nq, 3);
	  	  
	  deAlloc_dble_array(xySpecArray, Gsz, Gsz);
	  deAlloc_dble_array(yzSpecArray, Gsz, Gsz);   
	  deAlloc_dble_array(zzSpecArray, Gsz, Gsz);
	  deAlloc_dble_array(ttSpecArray, Gsz, Gsz);   
      
	  /*.........................................
	    Make length position (KGMYG path) pointer */

	  double *pathLPos;

	  pathLPos = new double[npPath];
      
	  get_qPathWVectors(n0, bwvecPath);     

	  wref = bwvecPath[0]; w0 = 0.0;

	  for (k = 0; k < npPath; k++)
	    {
	      wvec = bwvecPath[k];
	  
	      qx = wvec[0] - wref[0];
	      qy = wvec[1] - wref[1];
	  
	      z0 = pow(qx, 2) + pow(qy, 2);

	      w0 = w0 + sqrt(z0);

	      pathLPos[k] = w0;

	      wref = wvec;
	    }

	  /*..............................................
	    Record spin-spin temporal correlation function */

	  str1 = "SpinSpin_Temporal_Corr" + tTagDat;
      
	  outfile1.open(outDir1 + subDir1 + str1);
      
	  for (n = 0; n < ntm; n++) 
	    {
	      reTime = n * dtm; // Real time paramter;

	      qSSVal = qSStVec[n];
	  
	      outfile1 << fmtDbleSci(reTime,  8, 15) << X4
		       << fmtDbleSci(qSSVal, 12, 22) << endl;
	    }
      
	  outfile1.close();
	  
	  /*................................
	    Prepare some output file strings */
        
	  if ((geom != "square") && (geom != "lieb"))
	    {
	      str1 = "KGMYG_wDepn_DFT_xySpec" + tTagDat;
	      str2 = "KGMYG_wDepn_DFT_yzSpec" + tTagDat;
	      str3 = "KGMYG_wDepn_DFT_zzSpec" + tTagDat;
	      str4 = "KGMYG_wDepn_DFT_ttSpec" + tTagDat;

	      str5 = "KGMCut_wDepn_DFT_zzSpec" + tTagDat;

	      cut1 = 0.0; // NOT right path	      
	      cut2 =  pi; // cut (change later);
	    }
	  else // Square geometry ...
	    {
	      str1 = "YGMXG_wDepn_DFT_xySpec" + tTagDat;
	      str2 = "YGMXG_wDepn_DFT_yzSpec" + tTagDat;
	      str3 = "YGMXG_wDepn_DFT_zzSpec" + tTagDat;
	      str4 = "YGMXG_wDepn_DFT_ttSpec" + tTagDat;

	      str5 = "YGMCut_wDepn_DFT_zzSpec" + tTagDat;

	      cut1 = pi * 0.5;
	      
	      cut2 = pi * (1.0 + 0.5 * sq2);
	    }
           
	  /*.....................................
	    Record spectrum KGMYG/YGMXG path data
	    |
	    | k = 0 --> K/Y-point of the BZ;
	    | n = 0 --> static contribution;
	    | n > 0 --> finite frequency terms; */
	  
	  outfile1.open(outDir1 + subDir2 + str1);
	  outfile2.open(outDir1 + subDir2 + str2);
	  outfile3.open(outDir1 + subDir2 + str3);
	  outfile4.open(outDir1 + subDir2 + str4);

	  outfile5.open(outDir1 + subDir2 + str5);

	  cerr << " Recording spectral data ... ";
	  
	  for (k = 0; k < npPath; k++)
	    {
	      lpVal = pathLPos[k];
	
	      for (n = 0; n < nmax; n++)//( skip static part with: n = 1; ... )
		{
		  wfreq = n * dwf;
	  
		  xyVal = xySpPathWVec[k][n];
		  yzVal = yzSpPathWVec[k][n];
		  zzVal = zzSpPathWVec[k][n];
		  ttVal = ttSpPathWVec[k][n];
	      
		  outfile1 << fmtDbleSci(lpVal,  8, 15) << X4
			   << fmtDbleSci(wfreq,  8, 15) << X4
			   << fmtDbleSci(xyVal, 12, 22) << endl;

		  outfile2 << fmtDbleSci(lpVal,  8, 15) << X4
			   << fmtDbleSci(wfreq,  8, 15) << X4
			   << fmtDbleSci(yzVal, 12, 22) << endl;
	
		  outfile3 << fmtDbleSci(lpVal,  8, 15) << X4
			   << fmtDbleSci(wfreq,  8, 15) << X4
			   << fmtDbleSci(zzVal, 12, 22) << endl;

		  outfile4 << fmtDbleSci(lpVal,  8, 15) << X4
			   << fmtDbleSci(wfreq,  8, 15) << X4
			   << fmtDbleSci(ttVal, 12, 22) << endl;

		  if ((lpVal >= cut1) && (lpVal <= cut2))
		    {
		      outfile5 << fmtDbleSci(lpVal,  8, 15) << X4
			       << fmtDbleSci(wfreq,  8, 15) << X4
			       << fmtDbleSci(zzVal, 12, 22) << endl;
		    }
		}
	
	      outfile1 << endl; outfile2 << endl;
	      outfile3 << endl; outfile4 << endl;

	      outfile5 << endl;

	      report(k, npPath);
	    }

	  cerr << "\n" << endl;

	  outfile1.close(); outfile2.close();
	  outfile3.close(); outfile4.close();

	  outfile5.close();
	    
	  /*......................
	    Delete arrays/pointers */

	  deAlloc_dble_array(xySpPathWVec, npPath, npw);
	  deAlloc_dble_array(yzSpPathWVec, npPath, npw);
	  deAlloc_dble_array(zzSpPathWVec, npPath, npw);
	  deAlloc_dble_array(ttSpPathWVec, npPath, npw);
	  
	  delete[] pathLPos;
      	  
	}//( flag = 0 )
      
    }//// Root-WORK (END);
  
  //============================
  // Free memory (deallocation):
 
  deAlloc_dble_array(rvecList, Ns, 2);
  
  deAlloc_intg_array(nbors1, Ns, Zn1);
  deAlloc_intg_array(nbors2, Ns, Zn2);
  deAlloc_intg_array(nborsX, Ns, ZnX);
  
  delete[] bondList;
  delete[] zvalList;
 
  if (qcrystal)
    {            
      delete[] Neel0Config;      
      delete[] Strp1Config;
      delete[] Strp2Config;
      delete[] Strp3Config;
      delete[] Strp4Config;
    }
  else//( periodic lattices )
    {     
      if (with_DFTcodes){ delete[] gridMap; }
    }
  
  deAlloc_dble_array(spinField, Ns, 3);
  
  deAlloc_dble_array(SqxWVec, Nq, ntm);
  deAlloc_dble_array(SqyWVec, Nq, ntm);
  deAlloc_dble_array(SqzWVec, Nq, ntm);

  if (!getEvoData)
    {  
      deAlloc_dble_array(CMT_SqxWVec, Nq, ntm);  
      deAlloc_dble_array(CMT_SqyWVec, Nq, ntm);
      deAlloc_dble_array(CMT_SqzWVec, Nq, ntm);    
  
      delete[] CMT_qSStVec;
    }

#if WITH_OPENCV == 1
  ///
  if (recSpinVec)
    {
      delete[] imgSites;
    }
  ///
#endif
    
  /*>>>> END OF CODE <<<<*/
  
  if (iAmRoot){cerr << " End of code!\n" << endl;}

  MPI_Barrier(MPI_COMM_WORLD);   
  
  MPI_Finalize(); return 0;
}

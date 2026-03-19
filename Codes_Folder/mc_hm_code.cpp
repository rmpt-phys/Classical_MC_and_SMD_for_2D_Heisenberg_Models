/* ---------------------------------------------------------------------------
   
   Author: Rafael M. P. Teixeira

   OrcID: https://orcid.org/0000-0001-7290-3573
   
   Email: rafael.mpt@gmail.com
   
   ................
   About this code: 

   This code was developed as part of a postdoctoral research project focused
   on spin excitations in disordered spin systems. The current implementation
   performs Monte Carlo (MC) simulations with parallel tempering and semiclas-
   sical molecular dynamics (SMD) for J1–J2 Heisenberg and Ising models on 2-
   dimensional lattice systems. For Heisenberg models, spin updates are carri-
   ed out using the heat-bath algorithm, while for Ising models, the standard
   single-spin-flip Metropolis algorithm is employed;

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

//////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////

//==================================================================
// Main code | Monte-Carlo simulation of the Heisenberg/Ising model:

int main(int argc, char** argv)
{
  const int num_Settings = 16;
  
  string inputs0[num_Settings];

  string inputs1[11], inputs2[8], inputs3[7], inputs4[7];

  string outImg0, outImg1, outVid1, outVid2;
  
  string str0, str1, str2, ftag, infoStr, geoStr;
    
  string outTagDat, outTagBin, tTagDat, tTagBin, tTagImg;
   
  double x0, y0, z0, w0, x, y, z, w, qx, qy, fcw;

  double norm, normSum, tht, phi, spin, localFd;

  double flipProb, BVal, EVal, deltaB, deltaE;

  int flag, stnum, inum, iwMax, nSamp, signal;
  
  int n0, n1, n2, m1, m2, iw, ch0, ch1;
  
  unsigned int i, j, k, n, m, index; //(#)

  ofstream outfile1, outfile2, outfile3;
  
  /* In (#), single letter integers are 
     defined, these are always unsigned
     in this main code, only use them
     for counters in loops... 

     Caution: any attempt to calculate
     a negative quantity with unsigned
     integers can lead unexepected
     results as shown below:

     x = - (i + 1) * qx ---> ERROR; 
  
     x = - qx * (i + 1) ---> OKAY;

     Above, the first command tries to
     compute the double 'x' by conver-
     ting i + 1 to a negative integer,
     but, since i is defined as unsig-
     ned int, this results is an erro-
     neous behavior. In practive, the
     negative operator on any unsigned
     integer variable cause its value
     to be set to INT_MAX (where the
     latter is the max. posite int.); */
  
  //=======================================
  // Initialization of the MPI environment:
    
  int wSize = 1;

  int wRank = 0;

  int hwSz; //( wSize / 2 )

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
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

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
      MPI_Abort(MPI_COMM_WORLD, 1);
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
  
  /*--------------------------------------------------
    If the global integer 'npt' (number of temperature
    points) is greater than one, parallel computations 
    is enabled (pcMode --> true) , but if the parallel
    tempering feature is disabled (ptON = false), then
    the features T-grid optmization & replica-tracking
    are turned off (see #-code below); */  
  
  npt = wSize;

  if (ANNL_ON)
    {               // Disable PT when using
      ptON = false; // simulated annealing;
    }
            
  if (wSize > 1)
    {                // Multi-thread mode (MPI):
      pcMode = true; // parallel computations enabled;

      hwSz = wSize / 2;

      if (iAmRoot)
	{
	  flag = 0;
		
	  if (wSize % 2 != 0)
	    {
	      cerr << "\n Error: "
		   << "number of threads must be even;";
	      
	      cerr << "\n" << endl; flag = 1;
	    }
	}

      if (!ptON)//(#)
	{
	  ptAdapt = false;
	  ptTrack = false;
	}
    }
  else//( Single-thread mode )
    {
      pcMode  = false; // Disable all
      ptAdapt = false; // parallelization
      ptTrack = false; // dependent features;
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
    }///[ Root code options check ]
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }
  
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
      ofstream seedFile("seeds.dat", ios::app | ios::out);
      
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
	    
	  for (i = 0; i < 11; i++)
	    {
	      if (!(inputFile1 >> inputs1[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (MODEL);";
	    
		  cerr << endl; flag = 1;
		}
	    }

	  for (i = 0; i < 8; i++)
	    {
	      if (!(inputFile2 >> inputs2[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (MC_SIM);";
	    
		  cerr << endl; flag = 2;
		}
	    }

	  for (i = 0; i < 7; i++)
	    {
	      if (!(inputFile3 >> inputs3[i]))
		{
		  cerr << "\n Error: "
		       << "Invalid data on input file (SMD_SIM);";
	    
		  cerr << endl; flag = 3;
		}
	    }

	  for (i = 0; i < 7; i++)
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
		   << " 3) Skip RK-evolution & get data from file;   \n"
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
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

  //=====================
  // Get input arguments: non-ROOT processes
  
  if (!iAmRoot)
    {
      ifstream inputFile1(fset1);
      ifstream inputFile2(fset2);
      ifstream inputFile3(fset3);
      ifstream inputFile4(fset4);
	    
      for (i = 0; i < 11; i++)
	{
	  inputFile1 >> inputs1[i];
	}

      for (i = 0; i < 8; i++)
	{
	  inputFile2 >> inputs2[i];
	}

      for (i = 0; i < 7; i++)
	{
	  inputFile3 >> inputs3[i];
	}

      for (i = 0; i < 7; i++)
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

  if (iAmRoot && (flag > 0))
    {
      string fileTag[4] = {"MODEL", "MC_SIM", "SMD_SIM", "Disorder"};

      n = flag - 1; //( Input error source )
      
      cerr << "\n Invalid input (" << inum;

      cerr << ") on file (" << fileTag[n];

      cerr << ") ... \n\n";
    }

  MPI_Barrier(MPI_COMM_WORLD);
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

  //========================================
  // Check MC-sampling video recording skip:
  
#if WITH_OPENCV == 1
  ///
  if (vrSkip == 0)
    {                     // Disable video feature
      recSpinVec = false; // if vrSkip is null ...
    }
  else
    { recSpinVec = true; }
  ///
#endif
  
  //==============================
  // Printing simulation settings: (user checks input)

  flag = 0;
      
  if (iAmRoot) // Root-SCOPE (START)
    {
      print_MC_info("onTerminal");

      print_MC_info("recordFile");

      /* Double precision check */
  
      x0 = 1 / 3.0;
  
      cerr << " Double 17-digits:\n\n 1/3 = "
	   << fmtDbleFix(x0, 17, 20);

      cerr << "\n" << endl;
      
#if WITH_OPENCV == 1
      ///
      /* Open-CV features */
      
      if (vision)
	{
	  cerr << " Interactive use enabled;";

	  cerr << "\n" << endl;
	}

      if (recSpinVec)
	{
	  if (Lsz > 64)
	    {
	      flag = 1; //( Execution will abort )
	      
	      cerr << " Video of the spin-sampling\n"
		   << " requires low system size; \n"
		   << " Maximum size: 64 X 64;  \n\n";

	      cerr << " Code terminated ...\n\n";
	    }
	  else //( small lattice detected ) 
	    {
	      cerr << " MC sampling video rec-skip: " << vrSkip;
	      
	      cerr << "\n" << endl;
	    }
	}
      else //( if MC-sampling video recording skip is zero )
	{
	  cerr << " MC sampling video is disabled;\n" << endl;
	}
#endif/// Video operations checkpoint ...
                  
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
	}//// Final checkpoint before
      /////// main code execution ...
#endif
      
    }// Root-SCOPE (END)

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

  //==================================
  // Build crystal/impurity map field: (global pointer) 

  if (disOrder)
    {
      impField = new int[Ns];

      if (iAmRoot)// Root-code scope (START)
	{
	  double min_ratio = 1.0 / Ns;
      
	  if (impRatio < min_ratio)
	    {
	      cerr << " Error: impurity-ratio is too low!\n\n";
	  
	      cerr << " Min. value: " << fmtDbleFix(min_ratio, 2, 6) << " %";

	      cerr << "\n" << endl; flag = 1;
	    }
	  else
	    {      
	      make_impurityField(impRatio, flag);

	      if (flag == 0)
		{
		  vector<int> impList;	     

		  for (i = 0; i < Ns; i++)
		    {
		      if (impField[i] == 1)
			{
			  impList.push_back(i);
			}
		    }

		  size_t listLength = impList.size();

		  n0 = int(listLength);
	      
		  cerr << " Number of impurities / sites & ratio : ";
	  
		  cerr << n0 << " / " << Ns << " & ";

		  cerr << fmtDbleFix(impurityRatio(), 2, 6) << " %\n" << endl;

		  if (n0 > 10){ n0 = 10; str0 = "..."; }
		  
		  cerr << " Impurity sites : ";

		  for (i = 0; i < n0; i++)
		    {
		      cerr << impList[i] << X2;
		    }

		  cerr << str0 << " ;\n " << endl;
		}
	      else
		{
		  cerr << " Problems in 'make_impField'...\n";
	  
		  cerr << endl; flag = 1;
		}
	    }//// Impurity map generation;
	}//////// Root-code scope (END)...

      MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
      if (flag == 0)
	{
	  MPI_Bcast(impField, Ns * intgSz,
		    MPI_BYTE, root, MPI_COMM_WORLD); }
      else
	{ MPI_Abort(MPI_COMM_WORLD, 1); }
    }

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
      
      double lspc; //( lattice spacing )
      
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
		  check_latticeSpc(lspc, flag);

		  if (qct_RemovePBC)
		    {
		      rmPBC_nborsTable(lspc);
		    }

		  cerr << " Lattice spacing is"
		       << (flag > 0 ? " not" : "")
		       << " equal unity: " << lspc << "\n\n";
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
		  cerr << " Failed!\n\n"
		       << " Problems in make_nborsTable: \n"
		       << " Invalid reference to number of "
		       << (flag == 1 ? "sites (Ns)" : "neighbors (Zn)") << ";\n\n";
		}
	      else{ cerr << "OK! " << m << " ms\n\n"; }

	      if (flag == 0)
		{
		  check_latticeSpc(lspc, flag);

		  cerr << " Lattice spacing is "
		       << (flag > 0 ? "not" : "")
		       << " equal unity: " << lspc << "\n\n";
		}
	    }
	}
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0)
    {
      MPI_Abort(MPI_COMM_WORLD, 1);
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
    
  //============================================
  // Make lists of wavevectors & record to file:
    
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
      npk = round(Qval / dq + 1);      
    }
  else // Other geometries ...
    {
      npk = round(pi / dq); 
    }
  
  /*.....................
    Root-only procedures:

    1) Record wavevectors (momentum grid)
    .. as 2d-points associated with the
    .. discrete 2D-Fourier transforms;

    2) Record 1st BZone wavevectors & set
    .. the global integer 'npPath' value; */

  if (!qcrystal)
    {
      if (iAmRoot)
	{
	  rec_qGrid(); //(1)

	  rec_wvectors_and_set_npPath(); //(2)
	}

      MPI_Bcast(&npPath, 1, MPI_INT, root, MPI_COMM_WORLD);
    
      MPI_Barrier(MPI_COMM_WORLD);
    }

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

  //=========================
  // Record neighbors-tables:
  
  if (iAmRoot)
    {
      int znum1, znum2, znumX;
	
      outfile1.open(outDir1 + subDir0 + "nborsTable1.dat");
      outfile2.open(outDir1 + subDir0 + "nborsTable2.dat");
      outfile3.open(outDir1 + subDir0 + "nborsTableX.dat");
      
      for (i = 0; i < Ns; i++)
	{
	  znum1 = zvalList[i].x; // Number of NN,
	  znum2 = zvalList[i].y; // next-NN and 
	  znumX = zvalList[i].z; // next-next-NN;
	  
	  for (k = 0; k < znum1; k++)
	    {
	      outfile1 << i << X3 << nbors1[i][k] << endl;
	    }

	  for (k = 0; k < znum2; k++)
	    {
	      outfile2 << i << X3 << nbors2[i][k] << endl;
	    }

	  for (k = 0; k < znumX; k++)
	    {
	      outfile3 << i << X3 << nborsX[i][k] << endl;
	    }

	  outfile1 << endl;
	  outfile2 << endl;
	  outfile3 << endl;
	}

      outfile1.close();
      outfile2.close();
      outfile3.close();
    }

  MPI_Barrier(MPI_COMM_WORLD);
  
  //=============================================
  // Make lattice figure & interactively find and
  // neighboring site based on user's mouse input:
  /*
    Warning: the procedure 'make_latticeFigure'
    requires many pointers to work properly...

    rvecList, zvalList, nbors1, nbors2, nborsX; */

#if WITH_OPENCV == 1
  ///
  flag = 0;
  
  if (iAmRoot)
    {
      if (qcrystal)
	{
	  cerr << " Plot of the lattice points with \n"
	       << " site numbers is not avaiable for\n"
	       << " the current settings; \n " << endl;
	}
      else//( code-defined periodic lattices )
	{
	  make_latticeFigure(flag);
      
	  if (flag > 0)
	    {
	      cerr << " Code exec. failed...\n" << endl;

	      flag = 1;
	    }
	}
    }

  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
  
  if (flag > 0)
    {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  ///
#endif

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

	      if (IState == "inputf") 
		{
		  cerr << " Input initial qct-state is invalid, \n"
		       << " reference states must be avaiable!\n\n";
		  
		  flag = 1;
		}
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
	  MPI_Abort(MPI_COMM_WORLD, 1);
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

  //======================================
  // Convert Qct-States to numerical data:

  if ((iAmRoot) && (qcrystal) && (!refState_Zero))
    {
      ofstream outState0;
      
      ofstream outState1, outState2;
      ofstream outState3, outState4;

      outState0.open(outDir1 + subDir0 + "Neel_State.dat");
      
      outState1.open(outDir1 + subDir0 + "Stripe1_State.dat");
      outState2.open(outDir1 + subDir0 + "Stripe2_State.dat");
      outState3.open(outDir1 + subDir0 + "Stripe3_State.dat");
      outState4.open(outDir1 + subDir0 + "Stripe4_State.dat");
      
      for (k = 0; k < Ns; k++)
	{
	  outState0 << Neel0Config[k][0] << endl;
	  outState1 << Strp1Config[k][0] << endl;
	  outState2 << Strp2Config[k][0] << endl;
	  outState3 << Strp3Config[k][0] << endl;
	  outState4 << Strp4Config[k][0] << endl;
	}

      outState0.close();
      
      outState1.close(); outState2.close();
      outState3.close(); outState4.close();     
    }

  /*==========================================
    Setting the initial spin-configuration &
    spin-vector (Vec3d) objects for the code:
    
    spinField: spin-configuration,
    .......... Ns sites + 3 components; */
  
  double **spinField;

  Vec3d spinVec, spinVecNew;
  
  spinField = Alloc_dble_array(Ns, 3);

  init_dble_array(spinField, Ns, 3, 0.0);
  
  m = pltSeq[0]; /* 0 --> xy-plane;
		    2 --> yz-plane; */
  if (iAmRoot)
    { 
      cerr << " Setting initial state: " << IState << "\n\n";
    }
    
  if (IState != "inputf")
    {
      set_initialSpinField(wRank, infoStr, spinField);
    }  
  else// Read spin-data from file ...
    {
      if (qcrystal)
	{
	  n0 = 4; //[ selector for qct-stripes ]
	  
	  if (wRank % n0 == 0)
	    {
	      for (k = 0; k < Ns; k++)
		{
		  spinField[k][0] = Strp1Config[k][0];
		}
	    }
	  else if (wRank % n0 == 1)
	    {
	      for (k = 0; k < Ns; k++)
		{
		  spinField[k][0] = Strp2Config[k][0];
		}
	    }
	  else if (wRank % n0 == 2)
	    {
	      for (k = 0; k < Ns; k++)
		{
		  spinField[k][0] = Strp3Config[k][0];
		}
	    }
	  else if (wRank % n0 == 3)
	    {
	      for (k = 0; k < Ns; k++)
		{
		  spinField[k][0] = Strp4Config[k][0];
		}
	    }
	}
      else//( crystal lattices )
	{
	  double *vecField = new double[configSz];
      
	  infoStr = "Input-File";

	  if (iAmRoot)
	    {
	      string fname = outDir0 + "inputSConf.bin";

	      ifstream fieldInput(fname, ios::binary);

	      check_binFile(Ns * szSpinVec, fname, flag);
	    
	      if (flag == 0)// Root reads input binary file;
		{   	      
		  for (k = 0; k < Ns; k++)
		    {
		      fieldInput.read
			(reinterpret_cast<char*>(spinField[k]), szSpinVec);
		    }

		  dbleFlatten2D(Ns, 3, spinField, vecField);
		}	  
	  
	      fieldInput.close();
	    }            

	  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
      
	  if (flag != 0)
	    {
	      MPI_Abort(MPI_COMM_WORLD, 1);
	    }
	  else//( broadcast spin-field )
	    {
	      MPI_Bcast(vecField, configSz, MPI_DOUBLE, root, MPI_COMM_WORLD);

	      if (!iAmRoot)
		{
		  dbleReshape2D(Ns, 3, vecField, spinField);
		}

	      delete[] vecField;
	    }
	}
    }

  /*------------------------------
    Check initial states & report:

    #) The code will abort if the spin state
    has an average norm different from 1 and
    warn if the local field at some site = 0; */
  
  norm = fieldNorm(spinField);
  
  n0 = check_zeroLocField(spinField);
	      
  MPI_Reduce(&norm, &normSum, 1, MPI_DOUBLE,
	     MPI_SUM, root, MPI_COMM_WORLD);

  MPI_Reduce(&n0, &signal, 1, MPI_INT,
	     MPI_SUM, root, MPI_COMM_WORLD);
  
  if (iAmRoot)
    {           
      cerr << " State info.:";
	  
      cerr << " (" << pltSeq[0] << pltSeq[1] << ")"; 

      norm = fcw * normSum;
      
      if (norm > 1.0)
	{
	  str0 = "Failed (invalid field norm)"; flag = 1; //(#)
	}
      else//( input spin field is fine )
	{
	  flag = 0;
	   
	  str0 = "Okay (norm = 1)";

	  if (signal > 0)
	    {
	      str0 += " + free spins";
	    }
	}

      cerr << "\n\n Field-check : " + str0 << "\n\n";
    }
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

  /*--------------------------
    Record the type of initial
    state for user checking... */
  
  if (iAmRoot)
    {
      str0 = outDir1 + "0IStates.txt";
      
      ofstream outFile(str0, ios::out);

      outFile << iformat2(wRank) << X2;

      outFile << infoStr << endl;

      outFile.close();
    }
	    
  MPI_Barrier(MPI_COMM_WORLD);
    
  for (i = 1; i < wSize; i++)
    {  
      if (wRank == i)
	{		
	  MPI_Send(infoStr.c_str(), infoStr.size() + 1,
		   MPI_CHAR, root, mpi_ctag, MPI_COMM_WORLD);
	}

      if (iAmRoot)
	{
	  char strVec[50]; //( infoStr ---> strVec )

	  ofstream outFile(str0, ios::app | ios::out);
			    
	  MPI_Recv(strVec, 50, MPI_CHAR, i, mpi_ctag,
		   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
	  outFile << iformat2(i) << X2;

	  outFile << strVec << endl;

	  outFile.close();
	}
	    
      MPI_Barrier(MPI_COMM_WORLD);
    }

  //================= | Outputs: reflect_test, rotation_test,
  // Test procedures: | ........ heatBath_X_test (dat files);

  ch0 = 0; // Swicth (1/0);

  if (iAmRoot && (ch0 == 1))
    {
      test_reflectionProc(spinField);

      test_rotationProc();

      test_heatBath();
    }

  //====================== | fft_out.dat
  // Test FFTW procedures: | fft_inv.dat
 
  ch0 = 0; // Swicth (1/0);

  if (iAmRoot && (ch0 == 1))
    {     
      int N = 256;

      int M = 256;
      
      double w1 = 1.5;

      double w2 = 2.5;

      double w3 = 1.0;

      string type = "sine";
      
      fftw2D_WavePacketTest(N, w1, w2);

      fftw1D_WavePacketTest(N, w1, type);

      fftw3D_WavePacketTest(N, M, w1, w2, w3);
    }
  
  //=========================================
  // Plot input-state spin field (root-only):
  /*
    Note: the Mat-objects 'gridMat' & 'vecsMat'
    are of type CV_8UC3 , these are initialized
    within the procedures called below, the 1st
    is a 3-channel color-based Mat-object which
    represents the lattice grid, it is used as
    a background for the 2nd object where the
    spin configuration is drawn (2d vectors); 

    #) DO NOT release 'gridMat' until END; */	    

#if WITH_OPENCV == 1
  ///
  Mat gridMat; // Lattice grid (background for field);

  Mat vecsMat; // Vector field on lattice;
  
  if (iAmRoot)
    {
      ftag = "InputState";
      
      outImg0 = outDir2 + "lattice_image.png";
      outImg1 = outDir2 + "spin3d_input0.png";
            
      make_latticeGrid(gridMat, outImg0);
  
      make_vecField(IState, ftag,
		    spinField, gridMat, vecsMat);

      imwrite(outImg1, vecsMat); // Plot field;
    }
  else//( only generate gridMat )
    {
      make_latticeGrid(gridMat, "SKIP");
    }

  if (multiSubs){ delete[] ir0Sites; }
  ///
#endif
 
  /*=====================================
    Preparation for the Monte-Carlo codes
    ===================================== */ 
 
  /*-----------------------
    Define useful strings &
    set input T-grid file */
  {    
    outTagDat = outLabel + ".dat";
    outTagBin = outLabel + ".bin";
  }

  string TGridFile = outDir0;

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
  
  if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }
  
  /*--------------------------
    Set simulation temperature */

  double delTemp0, Temp, Beta;

  double TempDiff = Temp2 - Temp1;

  double *TpVec; //( root only )
  
  if (pcMode)
    {
      delTemp0 = TempDiff / (npt - 1);

      if (isZero(delTemp0))
	{
	  if (iAmRoot)
	    {
	      cerr << " Running in fixed "
		   << " temperature mode!" << "\n\n";
	    }
	}
    }
  else
    { delTemp0 = 0.0; }

  if ((iAmRoot) && (pcMode)) // Initial T-grid (START);
    {
      TpVec = new double[npt];
      
      if (getTGrid)
	{
	  cerr << " T-grid: input file;\n" << endl;
	      
	  ifstream TList(TGridFile, ios::binary);
	  
	  size_t szVec = npt * dbleSz;
	  
	  TList.read(reinterpret_cast<char*>(TpVec), szVec);
	  
	  TList.close();
	}
      else // Make T-grid from scratch ...
	{	      
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
    
      ofstream outT0(outDir1 + "PT_Check/PT_T0.dat", ios::out);

      for (n = 0; n < npt; n++)
	{                             // Record starting
	  outT0 << TpVec[n] << endl;} // simulation T-grid;
	
      outT0.close();
	  
    }//// Initial T-grid (END)

  /*--------------------------------------
    Distribute temperatures to all threads

    Info: root sends 'TpVec[wRank]' to each thread
    whcih assigns the value to the variable 'Temp'; */
 
  if (pcMode)
    {
      MPI_Scatter(TpVec, 1, MPI_DOUBLE, &Temp, 1,
		  MPI_DOUBLE, root, MPI_COMM_WORLD);
    }
  else{ Temp = Temp1; } // Initial temperature;
  
  Beta = 1.0 / Temp;    // Initial inverse temperature;
  
  /*-----------------------
    Set field recording key */
  
  bool recField;

  if (with_recField)
    {
      if (Temp > TempMax)
	{
	  recField = false; }
      else
	{ recField = true;  }
    }
  else
    { recField = false; }

  /*--------------------------------
    Set annealing parameters with T 
    grid as the target distribution */

  int TpSteps = NTerm / ANNL_Step - 1;

  double delT = (Temp0 - Temp) / TpSteps;
  
  if (ANNL_ON)
    {
      Temp = Temp0;
      
      Beta = 1.0 / Temp0;
    }

  /*--------------------------
    Define rescaling constants */
  
  const double szFac1 = 1.0 / Ns;
  
  const double szFac2 = pow(szFac1, 2);
     
  /*----------------------
    Declare common objects */

  int *siteList = new int[Ns];
  
  int measCnt1, measCnt2;
   
  double fc, fc1, fc2;

  double RMat[9]; // Rotation matrix;
  
  Vec3d locField, unitVec;

  /*----------------------------
    Declare PT auxiliary objects */
  
  int ptCnt, ptExc;

  double **swapField;

  if ((pcMode) && (ptON))
    {
      swapField = Alloc_dble_array(Ns, 3);
    }

  /*------------------------------- | This part only repeats some
    Set time & frequency parameters | code within the RK-code ...
   
    Below, dtm, dw, tmax & wmax are global
    variables (GVs) used in the real time
    evolution procedure (not performed du-
    ring the thermalization stage). In the
    latter, we employ the evolution method
    called Runge-Kutta  (RK) of 4th order.
    Here, the mentioned GVs are set based
    on another GV, ntm (input);
  
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
  
  wmax = wfac * maxLocField;

  dtm = pi / wmax;

  tmax = ntm * dtm;

  dwf = 2.0 * wmax / ntm;
       
  /*======================================
    Monte-Carlo code: thermalization stage
    ====================================== */

  //// TMC_SCOPE (START)
  {
    /*---------------------------
      Declare physical quantities
      
      1) Spin system configuration 
      ** energy density (squared value);

      2) Q-order magnetization for any
      ** lattice geometry (squared value); 

      3) Stripe order magnetization for any
      ** lattice geometry (squared value);
      
      4) Energy range variables, during the
      ** thermalization the code attempt to
      ** find the range [EMin , EMax];

      5) Energy vector (all replicas), used
      ** only when PT adaptation is enabled;
    
      CMT_X : cumulative measurement of X; */

    double EnrgTotal;
  
    double EDen2, CMT_EDen2; //(1)
  
    double QMag2, CMT_QMag2; //(2)

    double SMag2, CMT_SMag2; //(3)

    double EMax, EMin; //(4)

    /*---------------------
      Define needed strings */

    string wTagDat, wTagImg, wTagVid;
    
    str0 = "_wRank(" + to_string(wRank) + ")";
      
    wTagDat = str0 + outLabel + ".dat";
    wTagImg = str0 + outLabel + ".png";
    wTagVid = str0 + outLabel + ".avi";
    
    /*-------------------------
      Define formatting strings */

    string Out0 = "%05d  %16.10e \n";

    string OutFormat = "%05d" + X2;
    
    for (i = 0; i < 3; i++)
      {
	OutFormat += "%16.10e" + X3;
      }

    OutFormat += " \n";

    const char* Fmt_i1f3 = OutFormat.c_str();

    const char* Fmt0 = Out0.c_str();   

    /*------------------------------------
      Initialize PT-code auxiliary objects */

    int grCnt, rootPos, *tracker;

    double ptRatio, ptProb;
    
    double *EnrgVec, *BetaVec;

    double *ptRtVec, *CMT_EnrgVec;
    
    ostringstream PT_OSS1, PT_OSS2;

    string PT_STR; //( PT_OSS1 + PT_OSS2 )

    string PTA_File = outDir1 + "PT_Check/PTA_ProgInfo";

    PTA_File += outTagDat; //( add tag )
    
    /*................................
      PT counters & information string */

    ptRatio = 0.0; //[ PT-ratio starts ]

    ptCnt = 0; //[ PT-attempts counter ]

    ptExc = 0; //[ PT-exchange counter ]

    grCnt = 0; //[ Grid-change counter ]
    
    PT_OSS1 << " Temperature = ";
    
    PT_OSS2 << " PT accRatio = ";

    /*.................................
      Energy vectors & progression file (npt = wSize ) */
    
    if ((iAmRoot) && (ptAdapt))
      {	
	EnrgVec = new double[npt];

	BetaVec = new double[npt];

	ptRtVec = new double[npt];

	CMT_EnrgVec = new double[npt];

	for (i = 1; i < wSize; i++)
	  {
	    CMT_EnrgVec[i] = 0.0;
	  }
	    
	ofstream outcheck0(PTA_File, ios::out);

	outcheck0 << "#N" << X2;
	  
	outcheck0 << "Temperature   " << X2
		  << "Average energy" << X2
		  << "PT accep-ratio" << endl;

	outcheck0.close();
      }

    /*..........................
      PT tracking pointer & file 

      rootPos : gives the T-grid location 
      (pointer indice) where the replica
      initially associated with the root
      process is located in both tracker
      and TpVec pointers;

      tracker : it can be used to monitor
      the exchange flow from low to high
      temperatures during the PT-code;

      If the user desires to monitor the
      replica motion after each PT itera-
      tion, the proc. below can be used:

      ofstream outTrack;
	
      str1  = outDir1 + "PT_Check/";

      str1 += "PT_Track" + outTagDat;

      outTrack.open(str1, ios::out);
	
      tracker = new int[npt];

      outTrack << X3;
	
      for (n = 0; n < npt; n++)
      {
      | tracker[n] = (int)n;
      |
      | outTrack << iformat3(n) << X3;
      }

      outTrack << endl;

      outTrack.close(); */
    
    if ((iAmRoot) && (pcMode) && (ptTrack))
      {
	rootPos = 0;
	
	tracker = new int[npt];
	
	for (n = 0; n < npt; n++)
	  {
	    tracker[n] = (int)n;
	  }		
      }
        
    /*----------------------------------
      Set output data files (in C-style) */

    FILE *file1, *file2;

    string fname1, fname2;

    bool tmcRec = false;

    if (tmcRec) // Enable/disable above;
      {
	fname1 = outDir1 + subDir1;
	fname2 = outDir1 + subDir1;
	
	fname1 += "TMC_Meas_Quants" + wTagDat;
	fname2 += "TMC_Moving_Avgs" + wTagDat;
	
	file1 = fopen(fname1.c_str(), "w");
	file2 = fopen(fname2.c_str(), "w");
      }
   
    /*--------------------------------
      Initialize measurable quantities */
    
    CMT_EDen2 = 0.0;  
    CMT_QMag2 = 0.0;
    CMT_SMag2 = 0.0;

    get_energyValue(spinField, EnrgTotal);

    EMax = EnrgTotal; // Starting values for
    EMin = EnrgTotal; // energy range variables;

    measCnt1 = 0; // Used if tmcRec = true;

    /*----------------------------------------------
      Initialize auxiliary variables for qct-stripe
      generation (enabled when force_Stripes = true) */

    double vecProd0, vecProd1;
    
    double qctAngle = wRank * a90; //( vector OP angle )

    Vec2d SVec, SVec0 = {cos(qctAngle), sin(qctAngle)};

#if WITH_OPENCV == 1
    ////////////////
    
    /*-------------------------------------------
      Prepare stuff for imaging & video recording
  
      1) Output video FPS;
      2) Video writer codec;
      3) Output videos base-name;

      Important: videos are only recorded when
      temperature <= TempMax (recField = true); */

    unsigned int codec;
    
    int fps = 12; //(1) 

    string cdc = "H264"; //(2)

    string vdname1 = "TMC_Out1"; //(3)
    string vdname2 = "TMC_Out2";

    VideoWriter vecVideo, magVideo;

    Vec2d *orderMap; //( order-parameter map / used if rec_orderMaps = true )

    if ((qcrystal) && (rec_orderMaps))
      {
	orderMap = new Vec2d[Ns];
      }
    
    if ((recField) && (recSpinVec)) //( Open video recorder & add 1st frame )
      {
	ftag = " --- "; // 1st frame title;
	
	codec = VideoWriter::fourcc(cdc[0], cdc[1], cdc[2], cdc[3]);      
	
	outVid1 = outDir3 + vdname1 + wTagVid;
	
	vecVideo.open(outVid1, codec, fps, plotSize);
  
	make_vecField(X2, ftag, spinField, gridMat, vecsMat);

	vecVideo.write(vecsMat);

	if ((qcrystal) && (rec_orderMaps))
	  {	    
	    outVid2 = outDir3 + vdname2 + wTagVid;
		
	    magVideo.open(outVid2, codec, fps, plotSize);
		
	    make_vecMap(X2, ftag, orderMap, gridMat, vecsMat);

	    magVideo.write(vecsMat);
	  }
      }

#endif///( WITH_OPENCV == 1)
    
    /*-------------------------------
      Thermalization loop begins here */

    int ncan, nmic;

    bool longHBath = false;

    auto time1 = high_resolution_clock::now();
    
    if (iAmRoot)
      {
	cerr << " Thermalization-stage MC running ... ";
      }

    /// TMC_LOOP (Iterator: nmc | START)
    ///
    for (int nmc = 0; nmc < NTerm; nmc++)
      {
	/*---------------------------
	  Adjust procedures if needed */
	
	if (longHBath)
	  {
	    nmic = nmicro2; // If PT exchange
	    ncan = ncanon2; // succeeded ...

	    longHBath = false;
	  }
	else // Otherwise ...
	  {
	    nmic = nmicro; // Default values
	    ncan = ncanon; // for MP and HBP; 
	  }
	
	/*----------------------
	  Microcaninal procedure
	 
	  The sites in the lattice are visited in random
	  order, a new spin vector for each visited site
	  is taken from the reflection of the old one
	  about the local spin field vector (this pro-
	  cedure applies to the Heisenberg model only); */

	if (!IsiModel)
	  {
	    for (k = 0; k < nmic; k++)
	      { 
		get_shuffledList(Ns, siteList);

		for (n = 0; n < Ns; n++)
		  {
		    i = siteList[n];
	      
		    get_localSpin(i, spinField, spinVec);

		    get_localField(i, spinField, locField);

		    reflect_aboutVec(spinVec,
				     locField, spinVecNew);

		    set_localSpin(i, spinVecNew, spinField);
		  }
	      }
	  }///| Model check;
      
	/*-----------------------------------
	  Select model for sampling procedure */
	 
	if (IsiModel)//( Metropolis algorithm )
	  {
	    get_shuffledList(Ns, siteList);

	    if ((qcrystal) && (force_Stripes))//| Force all four kinds
	      {                               //| of stripes in the qct;
		for (n = 0; n < Ns; n++)
		  {
		    /*.................................................*/
		    
		    i = siteList[n];

		    spin = spinField[i][0];

		    get_localFieldX(i, spinField, localFd);

		    deltaE = 2.0 * spin * localFd; //( E0 - E1)

		    /*.................................................*/

		    get_onSite_qctSVec(i, spinField, SVec);

		    vecProd0 = 1.0 - dotProduct2d((+ 1.0) * SVec, SVec0);
		    vecProd1 = 1.0 - dotProduct2d((- 1.0) * SVec, SVec0);

		    /*.................................................*/

		    deltaE += 2.0 * J2 * ( vecProd0 - vecProd1 );

		    flipProb = min(1.0, exp(+ Beta * deltaE));

		    drand1 = dSFMT_getrnum();

		    if (drand1 < flipProb)//( spin-flip )
		      {
			spinField[i][0] = (- spin);
		      }
		    /*.................................................*/
		  }
	      }
	    else//( normal sampling with spin-flip )
	      {
		for (n = 0; n < Ns; n++)
		  {
		    /*...........................................*/
		    
		    i = siteList[n];

		    spin = spinField[i][0];

		    get_localFieldX(i, spinField, localFd);

		    deltaE = 2.0 * spin * localFd; //( E0 - E1)	

		    flipProb = min(1.0, exp(+ Beta * deltaE));

		    drand1 = dSFMT_getrnum();

		    if (drand1 < flipProb)//( spin-flip )
		      {
			spinField[i][0] = (- spin);
		      }		      
		    /*...........................................*/
		  }
	      }
	  }
	else//( Heisenberg model / Heat-bath algorithm ) 
	  {
	    /* First, the rotation matrix that transforms
	       the pole vector (0,0,1) into the unit vector
	       associated with a local field vector (which
	       is not normalized) is obtained (the proce-
	       dure 'get_rotZ2VMat' requires a unit
	       vector as input);

	       Then, for each site (in random order), the
	       new spin vector is taken from the distribu-
	       bution associated with its local spin field
	       as it was aligned with the z-axis, so the
	       components are simple to calculate from
	       usual azimuthal and polar angles;

	       Lastly, the rotation matrix applied to this
	       spin vector maps its components to the ones
	       given by the coordinate system where the 
	       z-axis is aligned with the local field,
	       i.e., the final vector has such field
	       as the reference pole for the angles; */
	    
	    for (k = 0; k < ncan; k++)
	      { 
		get_shuffledList(Ns, siteList);

		for (n = 0; n < Ns; n++)
		  {
		    i = siteList[n];
	      
		    /* Get rotation matrix */
	      
		    get_localField(i, spinField, locField);

		    unitVec = normVec3d(locField);
	      
		    get_rotZ2VMat(unitVec, RMat);
	      
		    /* Generate random vector */
	      
		    drand1 = dSFMT_getrnum();
		    drand2 = dSFMT_getrnum();
	  
		    tht = 2.0 * pi * drand1;

		    get_hBathSample(drand2, Beta, locField, z);
      
		    w = sqrt(1.0 - z * z); // z = cos(phi);
		
		    spinVec[0] = w * cos(tht);
		    spinVec[1] = w * sin(tht); 
		    spinVec[2] = z;

		    /* Rotate the vector using 'RMat' */
	      
		    spinVecNew = MxVecProduct(RMat, spinVec);

		    set_localSpin(i, spinVecNew, spinField);
		  }
	      }
	  }///| Model selection;
	
	/*--------------------------------------
	  Get total energy & update energy range */
	    
	get_energyValue(spinField, EnrgTotal);

	if (nmc > NTd2)
	  {
	    if (EnrgTotal > EMax){EMax = EnrgTotal;}
	    if (EnrgTotal < EMin){EMin = EnrgTotal;}
	  }

	/*------------------------------------------
	  Perform some measurements & record to file */
	    
	if ((tmcRec) && (!qcrystal))
	  {
	    measCnt1 += 1;

	    EDen2 = pow(szFac1 * EnrgTotal, 2);
	    
	    get_QMagSquared(spinField, QMag2);
	    get_SMagSquared(spinField, SMag2);

	    CMT_EDen2 += EDen2;	  
	    CMT_QMag2 += QMag2;
	    CMT_SMag2 += SMag2;	   
		  
	    fc = 1.0 / measCnt1;

	    fprintf(file1, Fmt_i1f3, nmc,
		    EDen2, QMag2, SMag2);

	    fprintf(file2, Fmt_i1f3, nmc,
		    fc * CMT_EDen2,
		    fc * CMT_QMag2,
		    fc * CMT_SMag2);
	  }

	/*--------------------------------
	  Decrease temperatute (annealing) */

	if (ANNL_ON)
	  {
	    if ((nmc > 0) && (nmc % ANNL_Step == 0))
	      {	    
		Temp = Temp - delT;

		Beta = 1.0 / Temp;
	      }
	  }///[ Increase temperature ]	
	
	/*-----------------------
	  Parallel tempering (PT) */

	MPI_Barrier(MPI_COMM_WORLD);
	
	if ((nmc > 100) && (pcMode) && (ptON)) // PT-CODE (START)
	  {
	    int ptN1, ptN2;
	    
	    double Enrg = EnrgTotal;
	    	    
	    double swapInfo[2]; // Information vector;

	    char *buffer = new char[Ns * szSpinVec];
	    		
	    n0 = nmc % 2;

	    if (n0 == 0)
	      {
		iwMax = hwSz - 1;}
	    else
	      {
		iwMax = hwSz - 2;}

	    for (iw = 0; iw <= iwMax; iw++) // PT-LOOP (START)
	      {
		n1 = n0 + 2 * iw;
	    
		n2 = n1 + 1;
   
		if (wRank == n1)
		  {
		    /*..........................................
		      Pair thread 1: first sender (wait signal) */
		
		    signal = 0; ptCnt += 1;

		    swapInfo[0] = Enrg;
		    swapInfo[1] = Beta;
		
		    MPI_Send(swapInfo, 2, MPI_DOUBLE,
			     n2, mpi_ctag, MPI_COMM_WORLD);

		    MPI_Recv(&signal, 1, MPI_DOUBLE, n2, mpi_ctag,
			     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		    if (signal == 1)
		      {			
			set_BufferData(Ns, szSpinVec, spinField, buffer);

			MPI_Send(buffer, Ns * szSpinVec,
				 MPI_CHAR, n2, mpi_ctag, MPI_COMM_WORLD);
			  
			MPI_Recv(buffer, Ns * szSpinVec,
				 MPI_CHAR, n2, mpi_ctag,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			get_BufferData(Ns, szSpinVec, buffer, spinField);
						
			ptExc += 1; longHBath = true;

			ptN1 = n1; ptN2 = n2;
		      }
		    /*..........................................*/
		  }
		else if (wRank == n2)
		  {
		    /*..........................................
		      Pair thread 2: first receiver (signaller) */
		    
		    signal = 0; ptCnt += 1;

		    ptRatio = (ptExc + 1) * (100.0 / ptCnt);
		
		    MPI_Recv(swapInfo, 2, MPI_DOUBLE, n1, mpi_ctag,
			     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		    EVal = swapInfo[0];
		    BVal = swapInfo[1];

		    deltaE = Enrg - EVal;
		    deltaB = Beta - BVal;

		    fc = deltaB * deltaE;
		
		    ptProb = min(1.0, exp(+ fc));

		    drand1 = dSFMT_getrnum();

		    if ((drand1 < ptProb) && (ptRatio <= 50.0))
		      {
			signal = 1; //( accept exchange )
		      }
	    		    
		    MPI_Send(&signal, 1, MPI_INT,
			     n1, mpi_ctag, MPI_COMM_WORLD);
		
		    if (signal == 1)
		      {			
			MPI_Recv(buffer, Ns * szSpinVec,
				 MPI_CHAR, n1, mpi_ctag,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			get_BufferData(Ns, szSpinVec, buffer, swapField);

			set_BufferData(Ns, szSpinVec, spinField, buffer);

			MPI_Send(buffer, Ns * szSpinVec,
				 MPI_CHAR, n1, mpi_ctag, MPI_COMM_WORLD);
			  
			copy_field(swapField, spinField);

			ptExc += 1; longHBath = true;

			ptN1 = n1; ptN2 = n2;
		      }
		    /*..........................................*/
		  }

		MPI_Barrier(MPI_COMM_WORLD);
		
	      }//// PT-LOOP (END)

	    ///-------------------------------------
	    /// Optimize temperature grid | PT-ADAPT (START)
	    	    
	    if ((ptAdapt) && (nmc < NTd2))
	      {
		const int k0 = hwSz - 1;
		
		double ptRt = (1.0 / ptCnt) * ptExc;
		    			
		double E0, Eb, Ef, BetaOld, BetaNew;

		bool avgCheck = true;
		
		/*.....................................
		  Gather energy, beta & PT-ratio values (root) */
		
		MPI_Gather(&EnrgTotal, 1, MPI_DOUBLE, EnrgVec, 1,
			   MPI_DOUBLE, root, MPI_COMM_WORLD);

		MPI_Gather(&Beta, 1, MPI_DOUBLE, BetaVec, 1,
			   MPI_DOUBLE, root, MPI_COMM_WORLD);

		MPI_Gather(&ptRt, 1, MPI_DOUBLE, ptRtVec, 1,
			   MPI_DOUBLE, root, MPI_COMM_WORLD);

		/*......................................
		  Accumulate the energy values & perform
		  the procedure if all conditions are met */
		
		if (iAmRoot)// Root-WORK (START-CODE)
		  {
		    signal = 0;
		    
		    for (i = 0; i < wSize; i++)
		      { 
			CMT_EnrgVec[i] += EnrgVec[i];
		      }

		    if (nmc % PTAd_Step == 0)
		      {
			for (i = 1; i < wSize; i++)
			  {
			    Eb = CMT_EnrgVec[i-1]; // Check for valid energy
			    E0 = CMT_EnrgVec[i+0]; // behavior (monotonic);
			    
			    if (Eb > E0){ avgCheck = false; }
			  }

			if (avgCheck)// Energy-CHECK (BARRIER-IN)
			  {
			    signal = 1; grCnt++;
			
			    /* Get average energy vector: */
			
			    fc = 1.0 / PTAd_Step;
			
			    for (i = 0; i < wSize; i++)
			      { 
				EnrgVec[i] = fc * CMT_EnrgVec[i];

				CMT_EnrgVec[i] = 0.0;
			      }

			    /* Record grid-ratio progression: */

			    if (grCnt > 1)
			      {
				ofstream outcheck0(PTA_File, ios::app | ios::out);

				for (i = 0; i < wSize; i++)
				  {
				    x0 = 1.0 / BetaVec[i];
				
				    y0 = EnrgVec[i];				
				    z0 = ptRtVec[i];
			    
				    outcheck0 << iformat2(grCnt) << X2;
				
				    outcheck0 << fmtDbleSci(x0, 5, 14) << X2
					      << fmtDbleSci(y0, 5, 14) << X2
					      << fmtDbleSci(z0, 5, 14) << endl;   
				  }

				outcheck0 << endl;

				outcheck0.close();
			      }

			    /* Find new temperature grid (mid-values) for
			       equal exchange rates during the PT-code...
			   
			       Mid-replicas: 0 < i < wSize - 1 ( wRank )

			       i = 2, 4, 6, 8, wSize - 2 | even (n = 0)

			       i = 1, 3, 5, 7, wSize - 3 | odd  (n = 1)
			   
			       # : Find temp. values for equal exchange
			       *** rates between neighboring replicas; */
			    			          
			    for (n = 0; n < 2; n++)
			      {
				for (k = 0; k < k0; k++)//(#)
				  {
				    i = 2 * (k + 1) - n;

				    Eb = EnrgVec[i-1];
				    E0 = EnrgVec[i+0];
				    Ef = EnrgVec[i+1];

				    BetaOld = BetaVec[i];

				    w0 = ( BetaVec[i-1] * ( E0 - Eb ) +
					   BetaVec[i+1] * ( Ef - E0 ) );
			
				    BetaNew = w0 / ( Ef - Eb );

				    BetaVec[i] = 0.5 * ( BetaOld + BetaNew );
				  }
			      }
			  }//// Energy-CHECK (BARRIER-OUT)
		      }
		  }//// Root-WORK (END-CODE)

		/*..........................
		  Distribute new beta values */
		
		MPI_Bcast(&signal, 1, MPI_INT, root, MPI_COMM_WORLD);  

		if (signal == 1)
		  {
		    MPI_Scatter(BetaVec, 1, MPI_DOUBLE, &Beta, 1,
				MPI_DOUBLE, root, MPI_COMM_WORLD);

		    Temp = 1.0 / Beta; //<<--[ Update temperature ]

		    ptCnt = 0; //[ Reset both PT ]
		    ptExc = 0; //[ counters to 0 ]
		  }			         
	      }//// PT-ADAPT (END)
    		
	    ///-----------------
	    /// Replica-tracking (START)

	    if ((ptTrack) && (nmc < NTd2))
	      {	  
		int ival, ptInfo[3];

		ptInfo[0] = signal;
	    
		ptInfo[1] = ptN1;
		ptInfo[2] = ptN2;

		for (i = 1; i < wSize; i++)
		  {  
		    if (wRank == i)
		      {		
			MPI_Send(ptInfo, 3, MPI_INT,
				 root, mpi_ctag, MPI_COMM_WORLD);
		      }

		    if (iAmRoot)
		      {
			MPI_Recv(ptInfo, 3, MPI_INT, i, mpi_ctag,
				 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			n0 = ptInfo[0];

			n1 = ptInfo[1];
			n2 = ptInfo[2];
		    
			if (n0 == 1 && i == n2)
			  {			
			    m1 = tracker[n1];
			    m2 = tracker[n2];

			    tracker[n1] = m2; // Exchange tracker
			    tracker[n2] = m1; // values (wRank > 0);
			  }			
		      }
	    
		    MPI_Barrier(MPI_COMM_WORLD);
		  }

		if (iAmRoot) // Find root-replica position;
		  {		    		    
		    for (n = 0; n < npt; n++)
		      {
			ival = tracker[n];
			
			if (ival == 0 && n > rootPos)
			  {
			    rootPos = n;
			  }
		      }
		  }	      
	      }//// TRACKING (END)
	    
	    delete[] buffer;
	    
	  }//// PT-CODE (END)

	if ( (nmc + 1) % 100 == 0 )
	  {
	    norm = fieldNorm(spinField);
	      
	    MPI_Reduce(&norm, &normSum, 1, MPI_DOUBLE,
		       MPI_SUM, root, MPI_COMM_WORLD);
	    
	    if (iAmRoot)
	      {
		report(nmc, NTerm);

		norm = fcw * normSum;

		x0 = abs(norm - 1.0);
	      
		if (x0 > dbleSmall)
		  {
		    flag = 1;
		    
		    cerr << " Error!\n\n";
	    
		    cerr << " Unphysical spin"
			 << " field detected!";

		    cerr << " DiffNorm = " << fmtDbleSci(x0, 4, 12);

		    cerr << "\n\n Simulation aborted! \n\n";
		  }
		else
		  { flag = 0; }
	      }

	    MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
	    if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }
  	  }
	
#if WITH_OPENCV == 1
	////////////
	
	if ((recField) && (recSpinVec))
	  {	      
	    if (nmc % vrSkip == 0)
	      {		
		/*-------------------------------
		  Record spin configuration frame */

		infoStr = magnetInfoString(Temp, spinField);
		
		ftag = " n = " + to_string(nmc);
		
		make_vecField(infoStr, ftag, spinField, gridMat, vecsMat);
		
		vecVideo.write(vecsMat);

		if ((qcrystal) && (rec_orderMaps))
		  {
		    get_qctSMagField(spinField, orderMap);
		      
		    infoStr = X2; ftag = " n = " + to_string(nmc);

		    make_vecMap(infoStr, ftag, orderMap, gridMat, vecsMat);

		    magVideo.write(vecsMat);
		  }
	      }
	  }////( rec-CHECK )
	
#endif//( WITH_OPENCV == 1 )
	
      }/* TMC_LOOP (END)
	  Iterator: nmc */
    
    /*------------------------------
      Update root temperature vector 
      and record last spin field ... */

    if ((pcMode) && (ptAdapt || ANNL_ON))
      {	
	MPI_Gather(&Temp, 1, MPI_DOUBLE, TpVec, 1,
		   MPI_DOUBLE, root, MPI_COMM_WORLD);
      }

    if (ANNL_ON)
      {	
	ofstream finalState;

	string tail = outLabel + ".bin";

	string wTag = "_wRank(" + to_string(wRank) + ")" + tail;

	string fbin = outDir0 + "MC_AnnlState" + wTag;

	finalState.open(fbin, ios::binary | ios::app);
	    
	for (k = 0; k < Ns; k++)
	  {
	    finalState.write
	      (reinterpret_cast
	       <const char*>(spinField[k]), szSpinVec);
	  }

	finalState.close();
      }
	
    /*------------------
      Report information */

    if (iAmRoot){cerr << endl;}
    
    if ((pcMode) && (ptON))
      {
	if (iAmRoot){cerr << endl;}
	
	ptRatio = (100.0 / ptCnt) * ptExc;	
	
	MPI_Barrier(MPI_COMM_WORLD);

	for (i = 0; i < wSize; i++)
	  {
	    if (wRank == i)
	      {
		PT_OSS1 << fmtDbleFix(Temp, 4, 8) << " ;";
		
		PT_OSS2 << fmtDbleFix(ptRatio, 3, 7) << " %";

		PT_STR = PT_OSS1.str() + PT_OSS2.str();
		      
		cerr << PT_STR << endl;		
	      }
	
	    MPI_Barrier(MPI_COMM_WORLD);
	  }
      }

    unsigned int MC_time;
    
    auto time2 = high_resolution_clock::now();

    auto dtime = time2 - time1;

    MC_time = duration_cast<milliseconds>(dtime).count();

    if (iAmRoot){waitAndJump();}
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if(iAmRoot)
      {
	cerr << " TMC time elapsed: " << MC_time;

	cerr << " ms\n" << endl;
      }

    MPI_Barrier(MPI_COMM_WORLD);

    /*-----------------
      Record new T-grid */

    if ((iAmRoot) && (pcMode) && (!getTGrid))
      {
	str1 = outDir0 + "Temp_List" + outTagBin;
	
	str2 = outDir1 + "PT_Check/PT_Grid" + outTagDat;

	ofstream outTList(str1, ios::binary);
	
	ofstream outGrids(str2, ios::out);

	size_t szVec = npt * dbleSz;

	outTList.write(reinterpret_cast<
		       const char*>(TpVec), szVec);

	outTList.close();
	
	for (n = 0; n < npt; n++)
	  {
	    x0 = TpVec[n];
	    
	    outGrids << fmtDbleSci(x0, 12, 22) << endl;
	  }

	outGrids.close();
      }

    /*-------------------------------
      Collect PT information & record */

    if ((pcMode) && (ptON)) // PT-INFO START ...
      {
	double swapInfo[4];

	double ptRt = ptRatio;

	swapInfo[0] = Beta;
	swapInfo[1] = EMin;
	swapInfo[2] = EMax;
	swapInfo[3] = ptRt;
		
	if (iAmRoot)
	  {
	    str1 = outDir1 + "PT_Check/PT_Prob" + outTagDat;
	    str2 = outDir1 + "PT_Check/PT_Info" + outTagDat;
	    
	    ofstream outcheck1(str1, ios::out);
	    ofstream outcheck2(str2, ios::out);

	    outcheck1 << PT_STR << endl;

	    outcheck2 << "# Inverse Temp" << X4
		      << "Minimum Energy" << X4
		      << "Maximum Energy" << X4
		      << "PT accep-ratio" << endl;
	    
	    outcheck2 << fmtDbleSci(Beta, 5, 14) << X4
		      << fmtDbleSci(EMin, 5, 14) << X4
		      << fmtDbleSci(EMax, 5, 14) << X4
		      << fmtDbleSci(ptRt, 5, 14) << endl;

	    outcheck1.close();
	    outcheck2.close();
	  }
	    
	MPI_Barrier(MPI_COMM_WORLD);
          
	for (i = 1; i < wSize; i++)
	  {  
	    if (wRank == i)
	      {		
		MPI_Send(PT_STR.c_str(), PT_STR.size() + 1,
			 MPI_CHAR, root, mpi_ctag, MPI_COMM_WORLD);

		MPI_Send(swapInfo, 4, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);
	      }

	    if (iAmRoot)
	      {
		char strVec[150]; //( PT_STR ---> strVec )

		ofstream outcheck1(str1, ios::app | ios::out);
		ofstream outcheck2(str2, ios::app | ios::out);
			    
		MPI_Recv(strVec, 150, MPI_CHAR, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(swapInfo, 4, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		x0 = swapInfo[0];
		y0 = swapInfo[1];
		z0 = swapInfo[2];
		w0 = swapInfo[3];
		
		outcheck1 << strVec << endl;
		
		outcheck2 << fmtDbleSci(x0, 5, 14) << X4
			  << fmtDbleSci(y0, 5, 14) << X4
			  << fmtDbleSci(z0, 5, 14) << X4
			  << fmtDbleSci(w0, 5, 14) << endl;

		outcheck1.close();
		outcheck2.close();
	      }
	    
	    MPI_Barrier(MPI_COMM_WORLD);
	  }
      }//// PT-INFO END ...

    /*------------------------------------
      Close files & deallocate phase lists */

    if (tmcRec)
      {
	fclose(file1);
	fclose(file2);
      }

    if (!qcrystal)
      {
	delete[] Q120Phase;
  
	delete[] HStpPhase;
	delete[] VStpPhase;
	delete[] DStpPhase;
	delete[] UpDwPhase;
      }
    
    if ((iAmRoot) && (pcMode))
      {
	if (ptAdapt)
	  {
	    delete[] BetaVec;
	
	    delete[] EnrgVec;

	    delete[] ptRtVec;

	    delete[] CMT_EnrgVec;
	  }

	if (ptTrack)
	  {	    	    
	    delete[] tracker;
	  }
      }    
    
#if WITH_OPENCV == 1
    ////////////////
	  
    /*-----------------------------------
      Generate images & plots of the spin
      configuration after thermalization */

    ftag = "TrmState"; //( Location: bottom margin )

    outImg0 = outDir2 + "spin3d_config0";

    outImg0 += "_wRank(" + to_string(wRank) + ")";

    outImg0 += outLabel + ".png";

    infoStr = magnetInfoString(Temp, spinField);
    
    make_vecField(infoStr, ftag, spinField, gridMat, vecsMat);
      
    imwrite(outImg0, vecsMat);

    if ((qcrystal) && (rec_orderMaps))
      {
	outImg0 = outDir2 + "orderMap0";

	outImg0 += "_wRank(" + to_string(wRank) + ").png";

	infoStr = magnetInfoString(Temp, spinField);
	
	get_qctSMagField(spinField, orderMap);		     	

	make_vecMap(infoStr, ftag, orderMap, gridMat, vecsMat);

	imwrite(outImg0, vecsMat);
      }

    /*--------------------------------
      Release memory from video object */

    if ((recField) && (recSpinVec))
      {
	vecVideo.release();

	if ((qcrystal) && (rec_orderMaps))
	  {
	    delete[] orderMap;

	    magVideo.release();
	  }
      }

#endif ///( WITH_OPENCV == 1)///

  }/// TMC_SCOPE (END)
 
  /*============================
    Monte-Carlo code: main stage
    ============================

    Notes: Absolut value: 1 (suffix)
    ...... Squared value: 2 (suffix)
      
    CMT_X : cumulative version of X; 

    Fac-type: rescaling factors for
    calculations of susceptibilities,
    spin stiffness and specific heat;

    Useful size-type constants giving
    the number of bytes of the object:

    szConf --> spin-vector field;
    szIMap --> impurity field;

    ...................................
    NBins : number of measurement bins
    of size 'BinSz' for which the ave-
    rage value and the statistics are
    computed for each one of the Ms1 
    quantities calculated during the
    main stage (after thermalization);

    ...................................
    NPacks : number of recorded packs
    of samples (spin configurations),
    it can be altered / updated when
    checking the output binary files;

    ...................................
    Note: change Ms1 (global parameter)
    if more/less data is read/recorded
    during this stage, other variables
    depend on it (look for STAT_SCOPE 
    in the final sections);    
    ................................... */

  const double fac_suscp = Ns * Beta;
  
  const double fac_RhoSt = (- 2.0) / (sq3 * Ns);
    
  const double fac_sheat = Ns * pow(Beta, 2);
  
  const size_t szConf = Ns * szSpinVec;

  const size_t szIMap = Ns * intgSz;

  int NBins = NMeas / BinSz;

  int NPacks = 0; 

  /*---------------------------
    Declare physical quantities (PART 1)

    1) Contributions for the spin stiffness;

    2) Staggered, stripe and ferromagnetic mag-
    ** netization moments (i.e. 1st, 2nd & 4th-
    ** power -- Mag1, Mag2 and Mag4) variables
    ** used in the calculations of the Binder
    ** cumulants and suscepbilities; 
    
    3) Stripe-order magnetizations for types
    ** horizontal, vertical and diagonal;
    
    4) Vector auxiliary variables for calcula-
    ** ting the spin stiffness contributions;

    >> Below, the CMT variables are the cumu-
    ** lative partners of the variables above,
    ** these provide binned averages as usual; */

  double RhoS1, RhoS2; //(1)
  
  double FMag1, FMag2, FMag4; //(2)  
  double QMag1, QMag2, QMag4;
  double SMag1, SMag2, SMag4;

  double SMag1H, SMag1V, SMag1D; //(3)

  Vec3d RhoS1vec, RhoS2vec; //(4)

  double CMT_FMag1, CMT_FMag2, CMT_FMag4;
  double CMT_QMag1, CMT_QMag2, CMT_QMag4;
  double CMT_SMag1, CMT_SMag2, CMT_SMag4;  
    
  double CMT_SMag1H, CMT_SMag1V, CMT_SMag1D;
  
  Vec3d CMT_RhoS1vec, CMT_RhoS2vec;

  /*---------------------------
    Declare physical quantities (PART 2)
    
    1) Spin system configuration 
    ** energy & related variables;

    2) Transverse-direction magnetization
    ** linear(1) & squared(2) forms only;
    
    3) Magnitude of the complex order
    ** parameter for a C3-symmetric latti-
    ** ce systems with the presence of an 
    ** external transverse magnetic field; 

    4) Ising-like order parameter for the
    ** Z(2)-symmetric stripe order (repla-
    ** ces the previous one in the case
    ** of a square lattice);

    5) Binder cumulants (4th-order) for
    ** continuous O(3)-symmetric order
    ** parameters and discrete (Ising) 
    ** Z(2)-symmetric order parameter;

    6) FBinC4 : ferromag. Binder cumulant;
    **
    ** XMag1 , XBinC4 = PsiMag1 , PBinC4
    ** XMag1 , XBinC4 = IsiMag1 , IBinC4; */
    
  double EDen1, Enrg1, CMT_Enrg1; //(1)
  double EDen2, Enrg2, CMT_Enrg2;

  double TMag1, CMT_TMag1; //(2)  
  double TMag2, CMT_TMag2;

  double PsiMag1, CMT_PsiMag1; //(3)
  double PsiMag2, CMT_PsiMag2;
  double PsiMag4, CMT_PsiMag4;

  double IsiMag1, CMT_IsiMag1; //(4)
  double IsiMag2, CMT_IsiMag2;
  double IsiMag4, CMT_IsiMag4;

  double QBinC4, SBinC4;
  double PBinC4, IBinC4; //(5)

  double FBinC4, XMag1, XBinC4; //(6)
 
  /*---------------------------
    Declare physical quantities (PART 3)
    
    Order-parameter susceptibilities for
    all orders previously defined (stag-
    gered-AFM, stripe, ferromagnetic, etc)
    plus specific heat and spin-stiffness;

    Note: chiXMag = chiPMag or chiIMag
    ..... depending lattice symmetry; */

  double chiQMag, chiSMag;
  double chiPMag, chiIMag;
  double chiXMag, chiTMag;

  double chiFMag, spcHeat, rhoStff;

  /*---------------------------
    Declare physical quantities (PART 4)
   
    Correlation lengths & peak-intensity:
    extracted from the static structure
    factor at 4 points in the BZ1:
    
    K-point, Mx & My points, Gamma point; */

  Vec3d sumVec; //( aux. variable for sum tasks )
     
  Vec5d xyCLen, CMT_xyCLen;
  Vec5d yzCLen, CMT_yzCLen;
  Vec5d zzCLen, CMT_zzCLen;
  Vec5d ttCLen, CMT_ttCLen;

  Vec5d xySVal, CMT_xySVal;
  Vec5d yzSVal, CMT_yzSVal;
  Vec5d zzSVal, CMT_zzSVal;
  Vec5d ttSVal, CMT_ttSVal;

  /*--------------------------------
    Declare quasi-crystal quantities (PART 5)

    PsiStrp: complex order-parameter com-
    posed by the phase-factor weighted 
    sum of all four stripe states for  
    the octogonal quasi-crystal;

    QctMag: order-parameter for the octo-
    gonal quasi-crystal (magnetization);
    
    StPrjVc: vector carrying the projec-
    tion value of the spin-state onto the
    four perfect stripe states (input con-
    figs) for an octogonal quasi-crystal; 
    
    SqPks4: vector carrying the combined
    amplitude of the spectral-peak-pairs
    in momentum space for quasi-crystals;

    StPrjBin & SqPksBin: Binder cumulants
    for each stripe projection & its asso-
    ciated pair of structure factor peaks
    in momentum space; */

  complex<double> PsiStrp;
  
  double QctMag1, CMT_QctMag1;
  double QctMag2, CMT_QctMag2; 
  double QctMag4, CMT_QctMag4;

  double N1fpar1, CMT_N1fpar1;
  double N1fpar2, CMT_N1fpar2;
  double N1fpar4, CMT_N1fpar4;  

  double NeelPrjt; //[ Neel state projection ]
  
  Vec4d  StrpPrjt; //[ Stripe states projection ]

  Vec4d StPrjVc1, CMT_StPrjVc1;
  Vec4d StPrjVc2, CMT_StPrjVc2;
  Vec4d StPrjVc4, CMT_StPrjVc4;  

  Vec4d SqPks4P1, CMT_SqPks4P1;  
  Vec4d SqPks4P2, CMT_SqPks4P2;
  Vec4d SqPks4P4, CMT_SqPks4P4;

  /* Binders & susceptibilities */
  
  double QctBinC4, chiQctMag;
  double N1fBinC4, chiN1fpar;
  
  Vec4d StPrjBin, StPrjChi;
  Vec4d SqPksBin, SqPksChi;

  /*-------------------------------
    Declare needed variable for the
    auto-correlation analisys (P7) */

  vector<double> DataTimeSeries;
    
  /*---------------------
    Define useful strings */
  
  ostringstream oss0; string stemp;

  oss0 << fixed << setprecision(5) << Temp;
  
  stemp = oss0.str(); // 'Temp' string form;

  str0 = "_T(" + stemp + ")";

  str0 += outLabel;
   
  tTagDat = str0 + ".dat";
  tTagBin = str0 + ".bin";
  tTagImg = str0 + ".png";

  /*-----------------------------------
    Check output binary files: data for
    statistics and spin-configurations */

  string fbin0, fbin1, fbin2, fbin3;

  string samplesFile;

  fbin0 = outDir0 + "MC_Bin_SpinField";
  fbin1 = outDir0 + "MC_Bin_AvgValue1";
  fbin2 = outDir0 + "MC_Bin_AvgValue2";
  fbin3 = outDir0 + "MC_Bin_AvgValue3";

  fbin0 += tTagBin;
  fbin1 += tTagBin;
  fbin2 += tTagBin;
  fbin3 += tTagBin;

  samplesFile = fbin0;

  if (iAmRoot)
    {
      ifstream testFile0(fbin0, ios::binary);
      ifstream testFile1(fbin1, ios::binary);
      ifstream testFile2(fbin2, ios::binary);
      ifstream testFile3(fbin3, ios::binary);
      
      streampos szFile0, szFile1, szFile2, szFile3;
  
      const int NSamples0 = NMeas / crSkip;

      const size_t szCPack = ( disOrder  * szIMap +
			       NSamples0 * szConf );
      flag = 0;
	    
      if (testFile0.is_open())
	{
	  testFile0.seekg(0, ios::end); 
	
	  szFile0 = testFile0.tellg();

	  if (disOrder)
	    {
	      if (szFile0 % szCPack != 0)
		{		  
		  cerr << " Disorder-mode is enabled, the number    \n"
		       << " of samples must be fixed, but the data  \n"
		       << " in the existing binary file is invalid! \n";

		  cerr << endl; flag = 1;
		}
	    }
	  else//( Disorder-mode is disabled )
	    {
	      if (szFile0 % szConf != 0)
		{		  
		  cerr << " Disorder-mode is disabled, the number \n"
		       << " of samples can vary , but the data in \n"
		       << " the existing binary file is invalid!  \n";

		  cerr << endl; flag = 1;
		}
	    }	  
	  
	  if (flag == 0)
	    {
	      NPacks = szFile0 / szCPack;
		
	      n0 = NPacks * NSamples0;

	      cerr << " Samples file already exists:      "
		   << " \n > number of recorded samples = " << n0
		   << " \n > number of recorded packs   = " << NPacks
		   << " \n > number of samples / pack   = " << NSamples0;

	      cerr << "\n" << endl;
	    }
	  else
	    {	      
	      cerr << " Error: existing binary file has no"
		   << " compatible data!\n File : " << fbin0;

	      cerr << "\n" << endl;
	    }
	
	  testFile0.close();	
	}

      if (testFile1.is_open())
	{
	  testFile1.seekg(0, ios::end); 
	
	  szFile1 = testFile1.tellg();
	    
	  if (szFile1 % szDataVec1 != 0)
	    {
	      cerr << " Error: existing binary file has no"
		   << " compatible data!\n File : " << fbin1;

	      cerr << "\n" << endl; flag = 1;
	    }
	
	  testFile1.close();	
	}

      if (testFile2.is_open())
	{
	  testFile2.seekg(0, ios::end); 
	
	  szFile2 = testFile2.tellg();
	    
	  if (szFile2 % szDataVec2 != 0)
	    {
	      cerr << " Error: existing binary file has no"
		   << " compatible data!\n File : " << fbin2;

	      cerr << "\n" << endl; flag = 1;
	    }
	
	  testFile2.close();	
	}

      if (testFile3.is_open())
	{
	  testFile3.seekg(0, ios::end); 
	
	  szFile3 = testFile3.tellg();
	    
	  if (szFile3 % szDataVec3 != 0)
	    {
	      cerr << " Error: existing binary file has no"
		   << " compatible data!\n File : " << fbin3;

	      cerr << "\n" << endl; flag = 1;
	    }
	
	  testFile3.close();	
	}
    }//// Binary-files CHECK;
  
  MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
    
  if (flag > 0)
    {
      MPI_Abort(MPI_COMM_WORLD, 1); }
  else
    { MPI_Bcast(&NPacks, 1, MPI_INT, root, MPI_COMM_WORLD); }
    
  /*-----------------------
    Set output binary files */
  
  ofstream binAvgFile1, binAvgFile2;

  ofstream binAvgFile3, configFile;

  if (recField)//( Only if Temp < TempMax )
    {            
      configFile.open(fbin0, ios::binary | ios::app);

      if (disOrder)
	{
	  for (k = 0; k < Ns; k++)
	    {
	      configFile.write(reinterpret_cast<
			       const char*>(&impField[k]), intgSz);
	    }// Impurity field ...
	}
    }

  binAvgFile1.open(fbin1, ios::binary | ios::app);
  binAvgFile2.open(fbin2, ios::binary | ios::app);
  binAvgFile3.open(fbin3, ios::binary | ios::app);

  /*----------------------------
    Set PT information variables */

  double ptRatio, ptProb;
  
  ostringstream PT_OSS1, PT_OSS2;

  string PT_STR; //( PT_OSS1 + PT_OSS2 )

  PT_OSS1 << " Temperature = ";
    
  PT_OSS2 << " PT accRatio = ";

  /*---------------------------------------
    Create discrete Fourier spectrum arrays

    The complex pointers Sqx0, Sqy0 & Sqz0
    represent the momentum space form (wa-
    vevector q) associated with each spin
    component (x,y,z), the zero frequency
    (static) structure factor (SF) is com-
    puted from these pointers, the values
    are stored in the array 'specField';

    After each completed binning iteration,
    the 2D discrete Fourier (DFT) date is
    obtained for the current 'spinField',
    we use FFTW subroutines for that;

    For a crystal lattice, the number of
    wavevectors in the 1st Brillouin zo-
    ne is equal to the number of sites:

    Nq = Ns & iNq = iNs ;

    For a quase-crystal, that is not the
    case, Nq can be set to any value de-
    pending on Lsz which is arbitrary in
    this case (in this code, we choose
    Lsz = sqrt(Ns) : rounded & even); */

  double **specField, **SqField;
  
  complex<double> *Sqx0, *Sqy0, *Sqz0;

  vector<int> nxSqPeak(8), nySqPeak(8);
  
  int spctCnt = 0; // DFT counter;   
  
  if (with_DFTcodes)
    {
      specField = Alloc_dble_array(Nq, 3);
      
      SqField = Alloc_dble_array(Nq, 6);

      Sqx0 = new complex<double>[Nq];
      Sqy0 = new complex<double>[Nq];
      Sqz0 = new complex<double>[Nq];
    }
  
  /*------------------------------------------ | FFTW is not used anymore,
    Initiate FFT objects & optimize procedures | MKL-Dfti replaced it;

    FFTW_ON --> prepare_xyzPlan_fftw2D(); */

  n0 = 0; //( error var. init. )

  if ((with_DFTcodes) && (!qcrystal))
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
  
  /*------------------------------------
    Initialization of the 3 measurement
    pointers for the spin momentum forms

    ...........................
    Notes about the procedures:

    1) The discrete 2D Fourier transform of
    ** the spin configuration involves the
    ** steps descrived below:

    -- Execute the associated fftw-plans with
    -- the current 'spinField' as input with
    -- the results within the fftw_complex
    -- pointers qwxData, qwyData & qwzData;

    -- Transfer the data in the fftw_complex
    -- pointers to the suitable complex-type
    -- pointers Sqx0, Sqy0 & Sqz0;

    -- Compute the spectrum amplitude & then
    -- assign the data to the SF pointer;    

    If only the static SF pointer 'specField'
    is desired , one can call the subrotuine
    'get_FourierSpectrum' to avoid the expli-
    cit call of the steps described above;
    
    2) The pointer 'SqField' accumulates the
    data from the complex pointer Sqx0, Sqy0
    & Sqz0, after all iterations are done, it
    can be used to calculate the thermal ave-
    rage (i.e., the MC estimate for the cano-
    nical ensemble average) of the static SF;
    
    3) In order to obtain the momentum space
    ordered form of the specField-array, the
    procedure 'get_OrderedSpecArray2D' splits
    it into the Lsz X Lsz lattice form arrays
    representing the static SF for 3 cases:

    xy (inplane), yz (mixed) & zz (transverse),

    They contain the zero frequency SF for 
    each wavevector in the corresponding re-
    scaled reciprocal lattice, which doesnt 
    cover the whole 1st BZone (the procedu-
    re 'record_SpecArray2D' uses the pedio-
    dicity of the 2D DFT to generate the SF 
    for all wavevectors in the desired zone); */

  if (with_DFTcodes)
    {
      const size_t sz0 = 3 * Nq * cplxSz;

      const size_t szRef = sz0 + intgSz;
    
      string fileName = outDir0 + "MC_Bin_w0SqField";

      string specFile = fileName + tTagBin;
      
      ifstream testFile(specFile, ios::binary);

      streampos szDat; //( File data size )

      bool abortSIM = false;

      bool specRead = false;

      /*......................................
	Check if an initialization file exists */

      n0 = 0;
      
      if (testFile.is_open())
	{
	  testFile.seekg(0, ios::end);

	  szDat = testFile.tellg();

	  testFile.close();
		
	  if (szDat != szRef)
	    {
	      abortSIM = true; }
	  else
	    { specRead = true; }
	}
      else { n0 = 1; }

      MPI_Reduce(&n0, &signal, 1, MPI_INT,
		 MPI_SUM, root, MPI_COMM_WORLD);

      if (iAmRoot)
	{
	  n0 = wSize - signal;
	
	  if (n0 == 0 || n0 == wSize) // All files found ||
	    {                         // All files missing:
	      flag = 0; }             // Fresh simulations;
	  else
	    { flag = 1; } //( Error: some files are missing )
	
	  if (abortSIM)
	    {
	      cerr << " Error: old spec-data file is not valid!\n"
		   << " Input: " << specFile << endl;

	      cerr << " \n Expected size (in bytes): " << szRef
		   << " \n Detected size (in bytes): " << szDat;	    

	      cerr << "\n" << endl; flag = 1;
	    }
	  else if (flag > 0)
	    {
	      cerr << " Error: spec-data file missing!";

	      cerr << "\n" << endl;
	    }
	
	  if (flag == 0)
	    {
	      cerr << " Init. of the spectral forms ... ";
	    }
	}//// Report status;

      MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
    
      if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }
    
      /*.......................................................
	Read complex pointer values from bin-file or initialize
	them with the 2D DFT spectra of the thermalized states  */

      auto time1 = high_resolution_clock::now();
    
      if (specRead)
	{
	  ifstream specData(specFile, ios::in | ios::binary);

	  specData.read(reinterpret_cast<char*>(&spctCnt), intgSz); //(#)
    
	  for (k = 0; k < Nq; k++)
	    {
	      specData.read(reinterpret_cast<char*>(&Sqx0[k]), cplxSz);
	      specData.read(reinterpret_cast<char*>(&Sqy0[k]), cplxSz);
	      specData.read(reinterpret_cast<char*>(&Sqz0[k]), cplxSz);
	    }
    
	  specData.close();
	}
      else//[ Compute initial pointer values from scratch ]
	{
	  if ((qcrystal) || (PBC_OFF))
	    {
	      get_qct_SqData(spinField, Sqx0, Sqy0, Sqz0);
	    }
	  else//( DFT using Intel-MKL ) 
	    {
	      get_SqData_iMKL(spinField, Sqx0, Sqy0, Sqz0);
	    }
   
	  get_StaticSFac(Sqx0, Sqy0, Sqz0, specField);       

	  spctCnt += 1; //(#)	
	}

      /*.................................
	Fourier transform of tha lattice: */

      if (iAmRoot)
	{
	  double **xraySpec;
    
	  xraySpec = Alloc_dble_array(Gsz, Gsz);

	  get_crystalXRay(xraySpec);

	  ftag = "xraySpec.dat";

	  record_SpecArray2D(xraySpec, ftag, 0);

	  deAlloc_dble_array(xraySpec, Gsz, Gsz);
	}
   
      /*..............
	# Information:
      
	If 'specRead' is true, the measurement counter
	called 'spctCnt' is obtained from the binary
	file, if not it is set to 1 above & must be
	increased by 1 after each data accumulation
	as below for the pointers in question (+=); */
    
      for (k = 0; k < Nq; k++)
	{
	  SqField[k][0] = abs(Sqx0[k]);
	  SqField[k][1] = abs(Sqy0[k]);
	  SqField[k][2] = abs(Sqz0[k]);
	
	  SqField[k][3] = arg(Sqx0[k]);
	  SqField[k][4] = arg(Sqy0[k]);
	  SqField[k][5] = arg(Sqz0[k]);
	}       
    
      /*.....................................
	Find structure-factor peak locations: */
    
      if ((qcrystal) && (iAmRoot))
	{
	  if (qct_NeelPhase)
	    {
	      get_qct_SqPkLoc(Neel0Config, nxSqPeak, nySqPeak); }
	  else
	    { get_qct_SqPkLoc(Strp1Config, nxSqPeak, nySqPeak); }
	}

      MPI_Bcast(nxSqPeak.data(),
		nxSqPeak.size(), MPI_INT, root, MPI_COMM_WORLD);
    
      MPI_Bcast(nySqPeak.data(),
		nySqPeak.size(), MPI_INT, root, MPI_COMM_WORLD);
    
      /*..................
	Report information */
    
      auto time2 = high_resolution_clock::now();

      auto dtime = time2 - time1;

      m = duration_cast<milliseconds>(dtime).count();

      if (m == 0){m = 1;}

      if (iAmRoot)
	{
	  cerr << "OK! " << m << " ms\n\n";

	  cerr << " Using spectral data from\n"
	       << " previous MC-simulations : ";
	
	  if (specRead)
	    {
	      cerr << "YES\n\n"; }
	  else
	    { cerr << "NO \n\n"; }
	}
    }

  MPI_Barrier(MPI_COMM_WORLD);

  /*-------------------------------------
    Define & initialize the 1d-histograms
    pointers for ordered state projection

    1) Integer type list with 1-component;
    2) Integer type list with 4-components;
    3) Automatic list initialization to zero; */

  const double boxSz1 = 2.0 / (Hst1dSz - 1); 
 
  const double boxFc1 = 1.0 / boxSz1;

  Lst1i *NeelHist; //(1)  
  Lst4i *StrpHist; //(2) 
  
  if (qcrystal)
    {
      NeelHist = new Lst1i[Hst1dSz]; //(3)    
      StrpHist = new Lst4i[Hst1dSz];

      /*......................................
	Check if an initialization files exist */

      const size_t szRef1 = Hst1dSz * szLst1i;
      const size_t szRef2 = Hst1dSz * szLst4i;
    
      string fileName1 = outDir0 + "MC_Bin_NeelHist";
      string fileName2 = outDir0 + "MC_Bin_StrpHist";

      string histFile1 = fileName1 + tTagBin;
      string histFile2 = fileName2 + tTagBin;
      
      ifstream testFile1(histFile1, ios::binary);
      ifstream testFile2(histFile2, ios::binary);

      streampos szDat1, szDat2; //( Files data size )

      bool abortSIM = false;

      bool histRead = false;

      n0 = 0; //( Reduce to signal )
      
      if (testFile1.is_open() && testFile2.is_open())
	{
	  testFile1.seekg(0, ios::end);
	  testFile2.seekg(0, ios::end);

	  szDat1 = testFile1.tellg();
	  szDat2 = testFile2.tellg();

	  testFile1.close();
	  testFile2.close();
		
	  if (szDat1 != szRef1 ||
	      szDat2 != szRef2 )
	    {
	      abortSIM = true; }
	  else
	    { histRead = true; }
	}
      else { n0 = 1; }

      MPI_Reduce(&n0, &signal, 1, MPI_INT,
		 MPI_SUM, root, MPI_COMM_WORLD);

      if (iAmRoot)
	{
	  n0 = wSize - signal;
	
	  if (n0 == 0 || n0 == wSize) // All files found ||
	    {                         // All files missing:
	      flag = 0; }             // Fresh simulations;
	  else
	    { flag = 1; } //( Error: some files are missing )
	
	  if (abortSIM)
	    {
	      cerr << " Error :: old hist-data file is not valid!\n"
		   << " Input-1: " << histFile1 << endl
		   << " Input-2: " << histFile2 << endl;

	      cerr << " \n Expected size-1 (in bytes): " << szRef1
		   << " \n Expected size-2 (in bytes): " << szRef2
		   << " \n Detected size-1 (in bytes): " << szDat1
		   << " \n Detected size-2 (in bytes): " << szDat2;

	      cerr << "\n" << endl; flag = 1;
	    }
	  else if (flag > 0)
	    {
	      cerr << " Error: hist-data file(s) missing!\n\n";
	    }
	
	  if (flag == 0)
	    {
	      cerr << " Init. of the histograms ... ";
	    }
	}//// Report status;

      MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
    
      if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

      /*....................................
	Initialize with binary data (or not) */
      
      if (histRead)
	{
	  ifstream histData1(histFile1, ios::in | ios::binary);
	  ifstream histData2(histFile2, ios::in | ios::binary);
    
	  for (k = 0; k < Hst1dSz; k++)
	    {
	      histData1.read(reinterpret_cast<char*>(&NeelHist[k]), szLst1i);
	      histData2.read(reinterpret_cast<char*>(&StrpHist[k]), szLst4i);
	    }

	  histData1.close();
	  histData2.close();
	}

      if (iAmRoot)
	{
	  cerr << "OK!\n\n";

	  cerr << " Using 1d-histogram-data       \n"
	       << " from previous MC-simulations : ";
	
	  if (histRead)
	    {
	      cerr << "YES\n\n"; }
	  else
	    { cerr << "NO \n\n"; }
	} 
    }

  /*-------------------------------------
    Define & initialize the 2d-histogram 
    pointer of the vector order-parameter

    1) Total number of histogram counters; 

    2) Stripe order-parameter histogram for
    |  counters in a 2d square region with 
    |  Hst2dSz X Hst2dSz = NHist2d boxes; */
     
  const double boxSz2 = 2.0 / (Hst2dSz - 1);  
 
  const double boxFc2 = 1.0 / boxSz2;
  
  const int NHist2d = pow(Hst2dSz, 2); //(1)

  int *SMagVecHist; //(2)
  
  if (qcrystal)
    { 
      SMagVecHist = new int[NHist2d];

      for (i = 0; i < NHist2d; i++)
	{
	  SMagVecHist[i] = 0;
	}

      /*......................................
	Check if an initialization file exists */
      
      const size_t szRef = NHist2d * intgSz;
    
      string fileName = outDir0 + "MC_Bin_SMagVecHist";

      string histFile = fileName + tTagBin;
      
      ifstream testFile(histFile, ios::binary);

      streampos szDat; //( File data size )

      bool abortSIM = false;

      bool histRead = false;

      n0 = 0; //( Reduce to signal )
      
      if (testFile.is_open())
	{
	  testFile.seekg(0, ios::end);

	  szDat = testFile.tellg();

	  testFile.close();
		
	  if (szDat != szRef)
	    {
	      abortSIM = true; }
	  else
	    { histRead = true; }
	}
      else { n0 = 1; }

      MPI_Reduce(&n0, &signal, 1, MPI_INT,
		 MPI_SUM, root, MPI_COMM_WORLD);

      if (iAmRoot)
	{
	  n0 = wSize - signal;
	
	  if (n0 == 0 || n0 == wSize) // All files found ||
	    {                         // All files missing:
	      flag = 0; }             // Fresh simulations;
	  else
	    { flag = 1; } //( Error: some files are missing )
	
	  if (abortSIM)
	    {
	      cerr << " Error: old hist-data file is not valid!\n"
		   << " Input: " << histFile << endl;

	      cerr << " \n Expected size (in bytes): " << szRef
		   << " \n Detected size (in bytes): " << szDat;	    

	      cerr << "\n" << endl; flag = 1;
	    }
	  else if (flag > 0)
	    {
	      cerr << " Error: hist-data file missing!\n\n";
	    }
	
	  if (flag == 0)
	    {
	      cerr << " Init. of the histograms ... ";
	    }
	}//// Report status;

      MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);
    
      if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

      /*....................................
	Initialize with binary data (or not) */
      
      if (histRead)
	{
	  ifstream histData(histFile, ios::in | ios::binary);
    
	  for (k = 0; k < NHist2d; k++)
	    {
	      histData.read
		(reinterpret_cast<char*>(&SMagVecHist[k]), intgSz);
	    }

	  histData.close();
	}

      if (iAmRoot)
	{
	  cerr << "OK!\n\n";

	  cerr << " Using 2d-histogram-data       \n"
	       << " from previous MC-simulations : ";
	
	  if (histRead)
	    {
	      cerr << "YES\n\n"; }
	  else
	    { cerr << "NO \n\n"; }
	} 
    }
      
  /*-------------------------------
    Initialize PT counters & ratio: */

  ptRatio = 0.0; //( PT-ratio starts )

  ptCnt = 0; //( PT-general  counter )

  ptExc = 0; //( PT-exchange counter )
  
  /*---------------------
    Main loops begin now: 

    External loop: data binning;
    Internal loop:  MC sampling;

    NSamps: counts the number of
    ....... samples recorded; */

  int ncan, nmic;

  int NSamps = 0;

  bool longHBath = false;
  
  auto time1 = high_resolution_clock::now();
  
  if (iAmRoot)
    {
      cerr << " Main-stage MC running ... ";
    }

  /// MC_OUTER_LOOP (Iterator: nmc | START)
  ///
  for (int nmc = 0; nmc < NBins; nmc++)
    {     
      /* ************************************
	 (Re)Initialize measurable quantities */            

      CMT_Enrg1 = 0.0; CMT_TMag1 = 0.0;
      CMT_Enrg2 = 0.0; CMT_TMag2 = 0.0;

      CMT_QMag1 = 0.0; CMT_SMag1 = 0.0;
      CMT_QMag2 = 0.0; CMT_SMag2 = 0.0;
      CMT_QMag4 = 0.0; CMT_SMag4 = 0.0;

      CMT_SMag1H = 0.0; CMT_FMag1 = 0.0;
      CMT_SMag1V = 0.0; CMT_FMag2 = 0.0;
      CMT_SMag1D = 0.0; CMT_FMag4 = 0.0;

      CMT_QctMag1 = 0.0; CMT_N1fpar1 = 0.0;
      CMT_QctMag2 = 0.0; CMT_N1fpar2 = 0.0;
      CMT_QctMag4 = 0.0; CMT_N1fpar4 = 0.0;
      
      CMT_PsiMag1 = 0.0; CMT_IsiMag1 = 0.0;
      CMT_PsiMag2 = 0.0; CMT_IsiMag2 = 0.0;
      CMT_PsiMag4 = 0.0; CMT_IsiMag4 = 0.0;             
                                                  
      CMT_xyCLen = null5d; CMT_xySVal = null5d;
      CMT_yzCLen = null5d; CMT_yzSVal = null5d;
      CMT_zzCLen = null5d; CMT_zzSVal = null5d;
      CMT_ttCLen = null5d; CMT_ttSVal = null5d;	     

      CMT_StPrjVc1 = null4d; CMT_SqPks4P1 = null4d;
      CMT_StPrjVc2 = null4d; CMT_SqPks4P2 = null4d;
      CMT_StPrjVc4 = null4d; CMT_SqPks4P4 = null4d;

      CMT_RhoS1vec = null3d;
      CMT_RhoS2vec = null3d;
      
      measCnt1 = 0; // Measure-counters
      measCnt2 = 0; // reset ...
  
      /// MC_INNER_LOOP (Iterator: nbin | START)
      ///
      for (int nbin = 0; nbin < BinSz; nbin++)
	{
	  /*---------------------------
	    Adjust procedures if needed */
	
	  if (longHBath)
	    {
	      nmic = nmicro2;
	      ncan = ncanon2;

	      longHBath = false;
	    }
	  else // Restore default values ...
	    {
	      nmic = nmicro;
	      ncan = ncanon;
	    }
	
	  /*----------------------
	    Microcaninal procedure */

	  if (!IsiModel)
	    {
	      for (k = 0; k < nmic; k++)
		{ 
		  get_shuffledList(Ns, siteList);

		  for (n = 0; n < Ns; n++)
		    {
		      i = siteList[n];
	      
		      get_localSpin(i, spinField, spinVec);

		      get_localField(i, spinField, locField);

		      reflect_aboutVec(spinVec,
				       locField, spinVecNew);

		      set_localSpin(i, spinVecNew, spinField);
		    }
		}
	    }///| Model check;
      
	  /*-----------------------------------
	    Select model for sampling procedure */

	  if (IsiModel)//( Metropolis algorithm )
	    {
	      get_shuffledList(Ns, siteList);
	    
	      for (n = 0; n < Ns; n++)
		{
		  i = siteList[n];

		  spin = spinField[i][0];

		  get_localFieldX(i, spinField, localFd);

		  deltaE = 2.0 * spin * localFd; //( E0 - E1)
		
		  flipProb = min(1.0, exp(+ Beta * deltaE));

		  drand1 = dSFMT_getrnum();

		  if (drand1 < flipProb)//( spin-flip )
		    {
		      spinField[i][0] = (- spin);
		    }
		}
	    }
	  else//( Heat-bath algorithm )
	    {
	      for (k = 0; k < ncan; k++)
		{ 
		  get_shuffledList(Ns, siteList);

		  for (n = 0; n < Ns; n++)
		    {
		      i = siteList[n];
	      
		      /* Get rotation matrix */
	      
		      get_localField(i, spinField, locField);

		      unitVec = normVec3d(locField);
	      
		      get_rotZ2VMat(unitVec, RMat);
	      
		      /* Generate random vector */
	      
		      drand1 = dSFMT_getrnum();
		      drand2 = dSFMT_getrnum();
	  
		      tht = 2.0 * pi * drand1;

		      get_hBathSample(drand2, Beta, locField, z);
      
		      w = sqrt(1.0 - z * z); // z = cos(phi);
	      
		      spinVec[0] = w * cos(tht);
		      spinVec[1] = w * sin(tht); 
		      spinVec[2] = z;

		      /* Rotate the vector using 'RMat' */
	      
		      spinVecNew = MxVecProduct(RMat, spinVec);

		      set_localSpin(i, spinVecNew, spinField);
		    }	      	      
		}
	    }///| Model selection;

	  /*--------------------
	    Perform measurements */

	  measCnt1 += 1; //( Measure-counter 1 )

	  sumVec = null3d; //( Used below )

	  /*...........................
	    Ferromagnetic magnetization */
	  
	  for (n = 0; n < Ns; n++)
	    {
	      get_localSpin(n, spinField, spinVec);
	      
	      sumVec += spinVec;
	    }

	  FMag1 = iNs * sqrt(dotProduct(sumVec, sumVec));

	  FMag2 = pow(FMag1, 2);
	  FMag4 = pow(FMag2, 2);
	  
	  CMT_FMag1 += FMag1;
	  CMT_FMag2 += FMag2;
	  CMT_FMag4 += FMag4;
		      
	  if (qcrystal)
	    {
	      /*..............................
		Projection-type magnetizations */	      
	      
	      get_qctNeelPrjt(spinField, NeelPrjt);
	      
	      get_qctStrpPrjt(spinField, StrpPrjt);     
	      
	      StPrjVc1 = absForm4d(StrpPrjt);
	      
	      StPrjVc2 = sqrForm4d(StPrjVc1);
	      StPrjVc4 = sqrForm4d(StPrjVc2);

	      QMag1 = abs(NeelPrjt);

	      SMag1 = sqrt(dotProduct4d(StrpPrjt, StrpPrjt));
	      
	      QMag2 = pow(QMag1, 2); QMag4 = pow(QMag2, 2);
	      SMag2 = pow(SMag1, 2); SMag4 = pow(SMag2, 2);	      

	      CMT_QMag1 += QMag1; CMT_SMag1 += SMag1;
	      CMT_QMag2 += QMag2; CMT_SMag2 += SMag2;
	      CMT_QMag4 += QMag4; CMT_SMag4 += SMag4;

	      CMT_StPrjVc1 += StPrjVc1;
	      CMT_StPrjVc2 += StPrjVc2;
	      CMT_StPrjVc4 += StPrjVc4;

	      /*..................
		Feed 1d-histograms */
	      
	      n1 = get_HistIndex(NeelPrjt, 1.0, boxFc1, Hst1dSz);

	      NeelHist[n1][0] += 1;

	      for (i = 0; i < 4; i++)
		{
		  n1 = get_HistIndex(StrpPrjt[i], 1.0, boxFc1, Hst1dSz) ;		  

		  StrpHist[n1][i] += 1;
		}

	      /*......................................
		Feed 2d-histogram at the (nx,ny) point */
	      
	      PsiStrp = ( abs(StrpPrjt[0]) * xfac1 + abs(StrpPrjt[1]) * yfac1 +
			  abs(StrpPrjt[2]) * xfac2 + abs(StrpPrjt[3]) * yfac2 );

	      n1 = get_HistIndex(PsiStrp.real(), 1.0, boxFc2, Hst2dSz);
	      n2 = get_HistIndex(PsiStrp.imag(), 1.0, boxFc2, Hst2dSz);	     

	      index = n2 * Hst2dSz + n1; // Histogram index; 
	          
	      SMagVecHist[index] += 1;

	      /*........................
		Qct-stripe magnetization */

	      get_qctStripeMag(spinField, QctMag1);

	      QctMag2 = pow(QctMag1, 2);
	      QctMag4 = pow(QctMag2, 2);
	      
	      CMT_QctMag1 += QctMag1;		
	      CMT_QctMag2 += QctMag2;
	      CMT_QctMag4 += QctMag4;

	      /*......................
		Nearest-neighbors Neel
		frustration parameter */
	      
	      get_qctNeel_N1fpar(spinField, w0);

	      N1fpar1 = abs(w0);
	      
	      N1fpar2 = pow(N1fpar1, 2);
	      N1fpar4 = pow(N1fpar2, 2);
	      
	      CMT_N1fpar1 += N1fpar1;
	      CMT_N1fpar2 += N1fpar2;	      
	      CMT_N1fpar4 += N1fpar4;
	    }
	  else//( code-defined periodic lattices )
	    {	      
	      /*......................
		C2 & C3 magnetizations */
	      
	      QMag1 = absStaggMag(spinField);
	      
	      SMag1H = absStripeMag(spinField, "Horz");
	      SMag1V = absStripeMag(spinField, "Vert");
	      SMag1D = absStripeMag(spinField, "Diag");

	      SMag1 = sqrt( pow(SMag1H, 2) +
			    pow(SMag1V, 2) +
			    pow(SMag1D, 2) * C3SYM );

	      QMag2 = pow(QMag1, 2); QMag4 = pow(QMag2, 2);
	      SMag2 = pow(SMag1, 2); SMag4 = pow(SMag2, 2);

	      CMT_QMag1 += QMag1; CMT_SMag1 += SMag1;
	      CMT_QMag2 += QMag2; CMT_SMag2 += SMag2;
	      CMT_QMag4 += QMag4; CMT_SMag4 += SMag4;

	      CMT_SMag1H += SMag1H;
	      CMT_SMag1V += SMag1V;
	      CMT_SMag1D += SMag1D;
	    }
	  
	  /*..............
	    Spin stiffness */
	      
	  get_spinStiff(spinField,
			RhoS1vec, RhoS2vec);
	      
	  CMT_RhoS1vec += RhoS1vec;
	  CMT_RhoS2vec += RhoS2vec;

	  /*...................
	    Energy measurements */

	  get_energyValue(spinField, Enrg1);
 	      
	  Enrg2 = pow(Enrg1, 2);

	  CMT_Enrg1 += Enrg1;
	  CMT_Enrg2 += Enrg2;

	  if (rec_autocTime)
	    {
	      DataTimeSeries.push_back(SMag1);
	    }

	  /*.................................
	    Transverse (Z-axis) magnetization */

	  get_TrMagnet(spinField, TMag1);

	  if (!extH_ON){TMag1 = abs(TMag1);}
	      
	  TMag2 = pow(TMag1, 2);

	  CMT_TMag1 += TMag1;
	  CMT_TMag2 += TMag2;
	      	      
	  /*......................................
	    Psi-type & Ising-type order parameters
	    and SF-peaks amplitude (quasi-crystal) */

	  if (!qcrystal)
	    {
	      if (C3SYM)
		{
		  PsiMag1 = cplxC3Parameter(spinField);

		  PsiMag2 = pow(PsiMag1, 2);
		  PsiMag4 = pow(PsiMag2, 2);
		
		  CMT_PsiMag1 += PsiMag1;
		  CMT_PsiMag2 += PsiMag2;
		  CMT_PsiMag4 += PsiMag4;
		}
	      else//( square lattice crystal )
		{		  
		  IsiMag1 = IsingZ2Parameter(spinField);

		  IsiMag2 = pow(IsiMag1, 2);
		  IsiMag4 = pow(IsiMag2, 2);

		  CMT_IsiMag1 += abs(IsiMag1);
		
		  CMT_IsiMag2 += IsiMag2;
		  CMT_IsiMag4 += IsiMag4;		    
		}
	    }
	  else//( get SF amplitude at all 8 hotspots ) 
	    {
	      get_qct_SqPks4(spinField,
			     nxSqPeak, nySqPeak, SqPks4P1);

	      SqPks4P2 = sqrForm4d(SqPks4P1);
	      SqPks4P4 = sqrForm4d(SqPks4P2);
		  
	      CMT_SqPks4P1 += SqPks4P1;
	      CMT_SqPks4P2 += SqPks4P2;
	      CMT_SqPks4P4 += SqPks4P4;
	    }
	  
	  /*-------------------------------------
	    Record current spin-configuration for
	    later usage in dynamical studies ... */

	  if (recField && (nbin % crSkip == 0))
	    {	      
	      for (k = 0; k < Ns; k++)
		{
		  configFile.write
		    (reinterpret_cast
		     <const char*>(spinField[k]), szSpinVec);
		}

	      NSamps++; //( final value is needed later )	     
	    }
	  
	  /*-----------------------------------------
	    Compute 2-D (momentum space) DFT spectrum
	    (zero frequency: static SF, corr. length) */      

	  if (with_DFTcodes)
	    {
	      if (nbin % dftDelay1 == 0)
		{	      
		  double **xySpField, **yzSpField;
		  double **zzSpField, **ttSpField;
	      
		  xySpField = Alloc_dble_array(Gsz, Gsz);
		  yzSpField = Alloc_dble_array(Gsz, Gsz);  
		  zzSpField = Alloc_dble_array(Gsz, Gsz);
		  ttSpField = Alloc_dble_array(Gsz, Gsz);	      

		  if ((qcrystal) || (PBC_OFF))
		    {
		      get_qct_SqData(spinField, Sqx0, Sqy0, Sqz0);
		    }
		  else//( DFT using Intel-MKL )
		    {
		      get_SqData_iMKL(spinField, Sqx0, Sqy0, Sqz0);
		    }
		  
		  for (k = 0; k < Nq; k++)
		    {
		      SqField[k][0] += abs(Sqx0[k]);
		      SqField[k][1] += abs(Sqy0[k]);
		      SqField[k][2] += abs(Sqz0[k]);
	
		      SqField[k][3] += arg(Sqx0[k]);
		      SqField[k][4] += arg(Sqy0[k]);
		      SqField[k][5] += arg(Sqz0[k]);
		    }

		  get_StaticSFac(Sqx0, Sqy0, Sqz0, specField);
    	      
		  get_OrderedSpecArray2D(specField,
					 xySpField, yzSpField,
					 zzSpField, ttSpField);

		  get_corrLen(xySpField, xyCLen, xySVal);
		  get_corrLen(yzSpField, yzCLen, yzSVal);
		  get_corrLen(zzSpField, zzCLen, zzSVal);
		  get_corrLen(ttSpField, ttCLen, ttSVal);

		  /* Correlation lengths */
	      
		  CMT_xyCLen += xyCLen; CMT_yzCLen += yzCLen;
		  CMT_zzCLen += zzCLen; CMT_ttCLen += ttCLen;

		  /* Peak values at 5 vectors */
	      
		  CMT_xySVal += xySVal; CMT_yzSVal += yzSVal;
		  CMT_zzSVal += zzSVal; CMT_ttSVal += ttSVal;

		  /* Free pointers & increase counters */
	      
		  deAlloc_dble_array(xySpField, Gsz, Gsz);
		  deAlloc_dble_array(yzSpField, Gsz, Gsz); 
		  deAlloc_dble_array(zzSpField, Gsz, Gsz);
		  deAlloc_dble_array(ttSpField, Gsz, Gsz);
		
		  spctCnt += 1; //( DFT-counter )
	      
		  measCnt2 += 1; //( Measure-counter 2 )

		}///[ DFT-scope END ]
	    }
	  
	  /*-----------------------
	    Parallel tempering (PT) */

	  MPI_Barrier(MPI_COMM_WORLD);

	  if ((pcMode) && (ptON)) // PT-CODE (START)	 
	    {	      
	      double Enrg = Enrg1;
	    	    
	      double swapInfo[2]; // Information vector;

	      char *buffer = new char[Ns * szSpinVec];
	  		
	      n0 = nmc % 2;

	      if (n0 == 0)
		{
		  iwMax = hwSz - 1;}
	      else
		{
		  iwMax = hwSz - 2;}

	      for (iw = 0; iw <= iwMax; iw++) // PT-LOOP (START)
		{
		  n1 = n0 + 2 * iw;
	    
		  n2 = n1 + 1;
    
		  if (wRank == n1)
		    {
		      /*..........................................
			Pair thread 1: first sender (wait signal) */

		      signal = 0; ptCnt += 1;

		      swapInfo[0] = Enrg;
		      swapInfo[1] = Beta;
		
		      MPI_Send(swapInfo, 2, MPI_DOUBLE,
			       n2, mpi_ctag, MPI_COMM_WORLD);

		      MPI_Recv(&signal, 1, MPI_DOUBLE, n2, mpi_ctag,
			       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		      if (signal == 1)
			{
			  set_BufferData(Ns, szSpinVec, spinField, buffer);

			  MPI_Send(buffer, Ns * szSpinVec,
				   MPI_CHAR, n2, mpi_ctag, MPI_COMM_WORLD);
			  
			  MPI_Recv(buffer, Ns * szSpinVec,
				   MPI_CHAR, n2, mpi_ctag,
				   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			  get_BufferData(Ns, szSpinVec, buffer, spinField);
			  			
			  ptExc += 1; longHBath = true;
			}
		      /*..........................................*/
		    }
		  else if (wRank == n2)
		    {
		      /*..........................................
			Pair thread 2: first receiver (signaller) */

		      signal = 0; ptCnt += 1;

		      ptRatio = (ptExc + 1) * (100.0 / ptCnt);
	
		      MPI_Recv(swapInfo, 2, MPI_DOUBLE, n1, mpi_ctag,
			       MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		      EVal = swapInfo[0];
		      BVal = swapInfo[1];

		      deltaE = Enrg - EVal;
		      deltaB = Beta - BVal;

		      fc = deltaB * deltaE;
		
		      ptProb = min(1.0, exp(+ fc));
		  
		      drand1 = dSFMT_getrnum();		      
		
		      if ((drand1 < ptProb) && (ptRatio <= 50.0))
			{
			  signal = 1; //( accept exchange )
			}
		    		    
		      MPI_Send(&signal, 1, MPI_INT,
			       n1, mpi_ctag, MPI_COMM_WORLD);
		
		      if (signal == 1)
			{
			  MPI_Recv(buffer, Ns * szSpinVec,
				   MPI_CHAR, n1, mpi_ctag,
				   MPI_COMM_WORLD, MPI_STATUS_IGNORE);

			  get_BufferData(Ns, szSpinVec, buffer, swapField);

			  set_BufferData(Ns, szSpinVec, spinField, buffer);

			  MPI_Send(buffer, Ns * szSpinVec,
				   MPI_CHAR, n1, mpi_ctag, MPI_COMM_WORLD);
			  
			  copy_field(swapField, spinField);
			
			  ptExc += 1; longHBath = true;
			}
		      /*..........................................*/
		    }

		  MPI_Barrier(MPI_COMM_WORLD);
		
		}//// PT-LOOP (END)

	      delete[] buffer;
	      
	    }//// PT-CODE (END)
      
	}/* MC_INNER_LOOP (END)
	    Iterator: nbin ... */

      /* **************************
	 Compute bin average values */
      {
	fc1 = 1.0 / measCnt1;
	fc2 = 1.0 / measCnt2;

	QMag1 = fc1 * CMT_QMag1;
	QMag2 = fc1 * CMT_QMag2;
	QMag4 = fc1 * CMT_QMag4;

	SMag1 = fc1 * CMT_SMag1;
	SMag2 = fc1 * CMT_SMag2;
	SMag4 = fc1 * CMT_SMag4;

	SMag1H = fc1 * CMT_SMag1H;
	SMag1V = fc1 * CMT_SMag1V;
	SMag1D = fc1 * CMT_SMag1D;

	FMag1 = fc1 * CMT_FMag1;
	FMag2 = fc1 * CMT_FMag2;
	FMag4 = fc1 * CMT_FMag4;

	if (qcrystal)
	  {	    
	    StPrjVc1 = fc1 * CMT_StPrjVc1;
	    StPrjVc2 = fc1 * CMT_StPrjVc2;
	    StPrjVc4 = fc1 * CMT_StPrjVc4;

	    SqPks4P1 = fc1 * CMT_SqPks4P1;
	    SqPks4P2 = fc1 * CMT_SqPks4P2;
	    SqPks4P4 = fc1 * CMT_SqPks4P4;

	    N1fpar1 = fc1 * CMT_N1fpar1;
	    N1fpar2 = fc1 * CMT_N1fpar2;
	    N1fpar4 = fc1 * CMT_N1fpar4;
	  }

	RhoS1vec = fc1 * CMT_RhoS1vec;
	RhoS2vec = fc1 * CMT_RhoS2vec;

	RhoS1 = 1.00 * fac_RhoSt * vecAvg3d(RhoS1vec);
	RhoS2 = Beta * fac_RhoSt * vecAvg3d(RhoS2vec);

	EDen1 = szFac1 * fc1 * CMT_Enrg1;
	EDen2 = szFac2 * fc1 * CMT_Enrg2;

	TMag1 = fc1 * CMT_TMag1;
	TMag2 = fc1 * CMT_TMag2;

	if (qcrystal)
	  {
	    QctMag1 = fc1 * CMT_QctMag1;
	    QctMag2 = fc1 * CMT_QctMag2;
	    QctMag4 = fc1 * CMT_QctMag4;
	  }
	else//( crystal lattices )
	  {
	    if (C3SYM)
	      {
		PsiMag1 = fc1 * CMT_PsiMag1;
		PsiMag2 = fc1 * CMT_PsiMag2;
		PsiMag4 = fc1 * CMT_PsiMag4;
	      }
	    else//( square lattice crystal )
	      {
		IsiMag1 = fc1 * CMT_IsiMag1;
		IsiMag2 = fc1 * CMT_IsiMag2;
		IsiMag4 = fc1 * CMT_IsiMag4;
	      }
	  }
	
	for (k = 0; k < 5; k++)
	  {
	    xyCLen[k] = (fc2 / Gsz) * CMT_xyCLen[k];
	    yzCLen[k] = (fc2 / Gsz) * CMT_yzCLen[k];
	    zzCLen[k] = (fc2 / Gsz) * CMT_zzCLen[k];
	    ttCLen[k] = (fc2 / Gsz) * CMT_ttCLen[k];

	    xySVal[k] = (fc2 / Gsz) * CMT_xySVal[k];
	    yzSVal[k] = (fc2 / Gsz) * CMT_yzSVal[k];
	    zzSVal[k] = (fc2 / Gsz) * CMT_zzSVal[k];
	    ttSVal[k] = (fc2 / Gsz) * CMT_ttSVal[k];
	  }

	/*...........................
	  Calculate Binder cumulants:
	  (value may be unnormalized) */

	if (IsiModel)
	  {
	    FBinC4 = (FMag2 == 0.0) ? 0.0 : 1.5 * (1.0 - FMag4 / (3.0 * pow(FMag2, 2)));
	  }
	else//( Heisenberg model )
	  {
	    FBinC4 = (FMag2 == 0.0) ? 0.0 : 0.5 * (5.0 - 3.0 * FMag4 / pow(FMag2, 2));
	  }
	
	if (qcrystal)
	  {
	    QBinC4 = (QMag2 == 0.0) ? 0.0 : 1.0 - QMag4 / (3.0 * pow(QMag2, 2));
	    SBinC4 = (SMag2 == 0.0) ? 0.0 : 1.5 - SMag4 / (1.0 * pow(SMag2, 2));

	    N1fBinC4 = (N1fpar2 == 0.0) ? 0.0 : 1.0 - N1fpar4 / (3.0 * pow(N1fpar2, 2));
	    QctBinC4 = (QctMag2 == 0.0) ? 0.0 : 1.0 - QctMag4 / (1.0 * pow(QctMag2, 2));

	    StPrjBin = null4d;
	    SqPksBin = null4d;

	    for (k = 0; k < 4; k++)
	      {
		if (StPrjVc2[k] > 0.0)
		  {
		    StPrjBin[k] = 1.0 - StPrjVc4[k] / (3.0 * pow(StPrjVc2[k], 2));
		  }
		
		if (SqPks4P2[k] > 0.0)
		  {
		    SqPksBin[k] = 2.0 - SqPks4P4[k] / pow(SqPks4P2[k], 2);
		  }
	      }
	  }
	else//( crystal lattices )
	  {
	    if (IsiModel)
	      {
		QBinC4 = (QMag2 == 0.0) ? 0.0 : 1.5 * (1.0 - QMag4 / (3.0 * pow(QMag2, 2)));
		SBinC4 = (SMag2 == 0.0) ? 0.0 : 2.0 * (1.0 - SMag4 / (2.0 * pow(SMag2, 2)));
	      }
	    else//( Heisenberg model )
	      {
		QBinC4 = (QMag2 == 0.0) ? 0.0 : 0.5 * (5.0 - 3.0 * QMag4 / pow(QMag2, 2));
		SBinC4 = (SMag2 == 0.0) ? 0.0 : 1.0 * (4.0 - 3.0 * SMag4 / pow(SMag2, 2));
	      }
	    
	    if (C3SYM)
	      {
		PBinC4 = (PsiMag2 == 0.0) ? 0.0 : 1.0 - PsiMag4 / (2.0 * pow(PsiMag2, 2));
	      }
	    else//( Ising-type Binder )
	      {
		IBinC4 = (IsiMag2 == 0.0) ? 0.0 : 1.0 - IsiMag4 / (3.0 * pow(IsiMag2, 2));
	      }
	   }
	
	/*...............................
	  Calculate additional quantities:
	  susceptibility & spin-stiffness

	  fac_suscp: susceptibility factor;
      
	  fac_sheat: specific heat factor;

	  In terms of the energy averages,
	  the latter factor is given by

	  fac_sheat = pow(Beta, 2) / Ns;

	  But, since we have the averages
	  for the energy density, we need
	  to compensate for Ns * Ns factor
	  in the denominator as follows...

	  fac_sheat <--- fac_sheat * pow(Ns, 2);

	  So that: fac_sheat = Beta * fac_suscp; */

	chiFMag = fac_suscp * (FMag2 - pow(FMag1, 2));	
	chiQMag = fac_suscp * (QMag2 - pow(QMag1, 2));
	chiSMag = fac_suscp * (SMag2 - pow(SMag1, 2));

	if (qcrystal)
	  {
	    chiQctMag = fac_suscp * (QctMag2 - pow(QctMag1, 2));
	    chiN1fpar = fac_suscp * (N1fpar2 - pow(N1fpar1, 2));

	    for (k = 0; k < 4; k++)
	      {	
		StPrjChi[k] = fac_suscp * (StPrjVc2[k] - pow(StPrjVc1[k], 2));		  			
		SqPksChi[k] = fac_suscp * (SqPks4P2[k] - pow(SqPks4P1[k], 2));
	      }
	  }
	else//( crystal lattices )
	  {
	    if (C3SYM)
	      {
		chiPMag = fac_suscp * (PsiMag2 - pow(PsiMag1, 2));
	      }
	    else//( Ising-type susceptibility )
	      {
		chiIMag = fac_suscp * (IsiMag2 - pow(IsiMag1, 2));
	      }
	  }

	chiTMag = fac_suscp * (TMag2 - pow(TMag1, 2));	
	spcHeat = fac_sheat * (EDen2 - pow(EDen1, 2));
    
	rhoStff = RhoS1 + RhoS2;
			
	/*............................
	  Prepare bin-average results: */

	if (qcrystal)
	  {
	    XMag1 = abs(QctMag1);
	    
	    XBinC4 = 1.5 * QctBinC4;
	    
	    chiXMag = chiQctMag;
	  }
	else//( crystal lattices )
	  {
	    if (C3SYM)
	      {
		XMag1 = abs(PsiMag1);
	    
		XBinC4 = 1.5 * PBinC4;
	    
		chiXMag = chiPMag;
	      }	   
	    else//( square lattice crystal )
	      {
		XMag1 = abs(IsiMag1);
	    
		XBinC4 = 1.5 * IBinC4;
	    
		chiXMag = chiIMag;
	      }
	  }
	
	vector<double> N1fprSet = {N1fpar1, N1fpar2,
				   N1fpar4, N1fBinC4, chiN1fpar};	
	
	double Values1[] =
	  {
	    EDen1, EDen2, spcHeat,
	    QBinC4, chiQMag, SBinC4, chiSMag,
	    XBinC4, chiXMag, FBinC4, chiFMag,
	    chiTMag, QMag1, SMag1, XMag1, FMag1, TMag1,
	    SMag1H, SMag1V, SMag1D, RhoS1, RhoS2, rhoStff
	  };

	double Values2[] =
	  {    
	    xyCLen[0], xyCLen[1], xyCLen[2], xyCLen[3], xyCLen[4],
	    yzCLen[0], yzCLen[1], yzCLen[2], yzCLen[3], yzCLen[4],
	    zzCLen[0], zzCLen[1], zzCLen[2], zzCLen[3], zzCLen[4],
	    ttCLen[0], ttCLen[1], ttCLen[2], ttCLen[3], ttCLen[4],
	    xySVal[0], xySVal[1], xySVal[2], xySVal[3], xySVal[4],
	    yzSVal[0], yzSVal[1], yzSVal[2], yzSVal[3], yzSVal[4],
	    zzSVal[0], zzSVal[1], zzSVal[2], zzSVal[3], zzSVal[4],
	    ttSVal[0], ttSVal[1], ttSVal[2], ttSVal[3], ttSVal[4],
	  };
	
	double Values3[] =
	  {	    
	    StPrjVc1[0], StPrjVc2[0], StPrjVc4[0], StPrjBin[0], StPrjChi[0],
	    StPrjVc1[1], StPrjVc2[1], StPrjVc4[1], StPrjBin[1], StPrjChi[1],
	    StPrjVc1[2], StPrjVc2[2], StPrjVc4[2], StPrjBin[2], StPrjChi[2],
	    StPrjVc1[3], StPrjVc2[3], StPrjVc4[3], StPrjBin[3], StPrjChi[3],
	    SqPks4P1[0], SqPks4P2[0], SqPks4P4[0], SqPksBin[0], SqPksChi[0],
	    SqPks4P1[1], SqPks4P2[1], SqPks4P4[1], SqPksBin[1], SqPksChi[1],
	    SqPks4P1[2], SqPks4P2[2], SqPks4P4[2], SqPksBin[2], SqPksChi[2],
	    SqPks4P1[3], SqPks4P2[3], SqPks4P4[3], SqPksBin[3], SqPksChi[3],
	    N1fprSet[0], N1fprSet[1], N1fprSet[2], N1fprSet[3], N1fprSet[4]
	  };

	/*........................................
	  Check values and record to binary files: */

	for (double &val : Values1)
	  {
	    if (!isNotZero(val)){ val = 0.0; }
	  }

	for (double &val : Values2)
	  {
	    if (!isNotZero(val)){ val = 0.0; }
	  }
			
	for (double &val : Values3)
	  {
	    if (!isNotZero(val)){ val = 0.0; }
	  }
	
	char *charData1, *charData2, *charData3;

	charData1 = reinterpret_cast<char*>(Values1);
	charData2 = reinterpret_cast<char*>(Values2);
	charData3 = reinterpret_cast<char*>(Values3);
	
	binAvgFile1.write(charData1, sizeof(Values1));
	binAvgFile2.write(charData2, sizeof(Values2));
	binAvgFile3.write(charData3, sizeof(Values3));
      }

      if (iAmRoot){report(nmc, NBins);}
      
    }/* MC_OUTER_LOOP (END)
	Iterator: nmc  ... */

  /*------------------------------------
    Close data file & report information */
 
  binAvgFile1.close();
  binAvgFile2.close();
  binAvgFile3.close();

  if (recField){configFile.close();}

  if (iAmRoot){cerr << endl;}
  
  if ((pcMode) && (ptON))
    {
      if (iAmRoot){cerr << endl;}
      
      ptRatio = (100.0 / ptCnt) * ptExc;       
	
      MPI_Barrier(MPI_COMM_WORLD);

      for (i = 0; i < wSize; i++)
	{
	  if (wRank == i)
	    {
	      PT_OSS1 << fmtDbleFix(Temp, 4, 8) << " ;";
		
	      PT_OSS2 << fmtDbleFix(ptRatio, 3, 7) << " %";

	      PT_STR = PT_OSS1.str() + PT_OSS2.str();
		      
	      cerr << PT_STR << endl;		
	    }
	
	  MPI_Barrier(MPI_COMM_WORLD);
	}
    }

  unsigned int MC_time;

  auto time2 = high_resolution_clock::now();

  auto dtime = time2 - time1;

  MC_time = duration_cast<milliseconds>(dtime).count();

  if (iAmRoot){waitAndJump();}
  
  MPI_Barrier(MPI_COMM_WORLD);
    
  if(iAmRoot)
    {
      cerr << " MC time elapsed: " << MC_time;

      cerr << " ms\n" << endl;
    }

  MPI_Barrier(MPI_COMM_WORLD);  	    

  /*========================================
    Record average structure factors data in
    zone & path forms, destroy FFTW objects:   

    SpField : full DFT spectrum (FFTW ordered)
    ......... defined on the reciprocal lattice;

    SpPath : spectrum along the KGMYG/YGMXG path
    ........ in the 1st BZone (length ordered);
    
    pathLPos : length position on the path asso-
    .......... ciated with each point on SpPath;
    
    The double-type pointer 'pathLPos' provides
    the distance along the KGMYG/YGMXG path asso-
    ciated with each point (wavevector) within it,
    this object is useful when recording the struc-
    ture factor amplitude along this path (SpPath); */

  if (with_DFTcodes)
    {       
      /*----------------------------------------
	Compute average spin momentum forms and
	record accumulated values in binary form */

      string fileName = "MC_Bin_w0SqField";

      string outName = outDir0 + fileName + tTagBin;

      ofstream Bin0; //( Binary file needed for RK-code )

      double ax, ay, az; //( Amplitude values )

      complex<double> xc, yc, zc; //( Complex forms )

      remove(outName.c_str()); //( Delete 'Bin0' if it exists )
    
      Bin0.open(outName, ios::out | ios::binary | ios::app);

      Bin0.write(reinterpret_cast<const char*>(&spctCnt), intgSz);

      fc = 1.0 / spctCnt;

      for (k = 0; k < Nq; k++)
	{
	  ax = fc * SqField[k][3];
	  ay = fc * SqField[k][4];
	  az = fc * SqField[k][5];

	  xc.real(SqField[k][0] * cos(ax));
	  xc.imag(SqField[k][0] * sin(ax));

	  yc.real(SqField[k][1] * cos(ay));
	  yc.imag(SqField[k][1] * sin(ay));

	  zc.real(SqField[k][2] * cos(az));
	  zc.imag(SqField[k][2] * sin(az));
	
	  Sqx0[k] = fc * xc;
	  Sqy0[k] = fc * yc;
	  Sqz0[k] = fc * zc;

	  Bin0.write(reinterpret_cast<const char*>(&xc), cplxSz);
	  Bin0.write(reinterpret_cast<const char*>(&yc), cplxSz);
	  Bin0.write(reinterpret_cast<const char*>(&zc), cplxSz);
	}

      Bin0.close();      
    
      /*------------------------------------
	Allocate needed arrays for next part */
    
      double **xySpField, **yzSpField;
      double **zzSpField, **ttSpField;
    
      xySpField = Alloc_dble_array(Gsz, Gsz);
      yzSpField = Alloc_dble_array(Gsz, Gsz);
      zzSpField = Alloc_dble_array(Gsz, Gsz);  
      ttSpField = Alloc_dble_array(Gsz, Gsz);

      /*-------------------------------------
	Get static SF & record 1st BZone data */

      string ftag1, ftag2, ftag3, ftag4;

      get_StaticSFac(Sqx0, Sqy0, Sqz0, specField);
    	      
      get_OrderedSpecArray2D(specField,
			     xySpField, yzSpField,
			     zzSpField, ttSpField);
     
      ftag1 = "DFT_xySpec" + tTagDat;
      ftag2 = "DFT_yzSpec" + tTagDat;
      ftag3 = "DFT_zzSpec" + tTagDat;
      ftag4 = "DFT_ttSpec" + tTagDat;

      record_SpecArray2D(xySpField, ftag1, 0);
      record_SpecArray2D(yzSpField, ftag2, 0);
      record_SpecArray2D(zzSpField, ftag3, 0);
      record_SpecArray2D(ttSpField, ftag4, 0);

      /*-------------------------------------
	Record spectrum along path on the 1BZ */

      if (!qcrystal)
	{
	  /*......................
	    Allocate needed arrays */
    
	  double *xySpPath, *yzSpPath, *pathLPos;
	  double *zzSpPath, *ttSpPath;

	  pathLPos = new double[npPath];
    
	  xySpPath = new double[npPath];
	  yzSpPath = new double[npPath];
	  zzSpPath = new double[npPath];
	  ttSpPath = new double[npPath];
       
	  /*..........................................
	    Make length pos.(KGMYG/YGMXG path) pointer */

	  vector<Vec2d> bwvecPath;

	  Vec2d wref, wvec;
    
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

	  /*..........................................
	    Extract static spectrum path data & record */

	  ofstream outDat1, outDat2, outDat3, outDat4;

	  double lpVal, xyVal, yzVal, zzVal, ttVal;
    
	  get_Spec_qPath(xySpField, xySpPath);
	  get_Spec_qPath(yzSpField, yzSpPath);
	  get_Spec_qPath(zzSpField, zzSpPath);
	  get_Spec_qPath(ttSpField, ttSpPath);

	  if (C3SYM)
	    {
	      ftag1 = "KGMYG_wZero_DFT_xySpec" + tTagDat;
	      ftag2 = "KGMYG_wZero_DFT_yzSpec" + tTagDat;
	      ftag3 = "KGMYG_wZero_DFT_zzSpec" + tTagDat;
	      ftag4 = "KGMYG_wZero_DFT_ttSpec" + tTagDat;
	    }
	  else // Square-like geometry ...
	    {
	      ftag1 = "YGMXG_wZero_DFT_xySpec" + tTagDat;
	      ftag2 = "YGMXG_wZero_DFT_yzSpec" + tTagDat;
	      ftag3 = "YGMXG_wZero_DFT_zzSpec" + tTagDat;
	      ftag4 = "YGMXG_wZero_DFT_ttSpec" + tTagDat;	
	    }
    
	  outDat1.open(outDir1 + subDir1 + ftag1);
	  outDat2.open(outDir1 + subDir1 + ftag2);
	  outDat3.open(outDir1 + subDir1 + ftag3);
	  outDat4.open(outDir1 + subDir1 + ftag4);
	
	  for (k = 0; k < npPath; k++)
	    {
	      lpVal = pathLPos[k];
	  
	      xyVal = xySpPath[k];
	      yzVal = yzSpPath[k];
	      zzVal = zzSpPath[k];
	      ttVal = ttSpPath[k];
	
	      outDat1 << fmtDbleSci(lpVal,  8, 15)
		      << fmtDbleSci(xyVal, 12, 22) << endl;
	
	      outDat2 << fmtDbleSci(lpVal,  8, 15)
		      << fmtDbleSci(yzVal, 12, 22) << endl;

	      outDat3 << fmtDbleSci(lpVal,  8, 15)
		      << fmtDbleSci(zzVal, 12, 22) << endl;

	      outDat4 << fmtDbleSci(lpVal,  8, 15)
		      << fmtDbleSci(ttVal, 12, 22) << endl;
	    }

	  outDat1.close(); outDat2.close();
	  outDat3.close(); outDat4.close();

	  /*.........................
	    Deallocate local pointers */

	  delete[] pathLPos;
    
	  delete[] xySpPath; delete[] yzSpPath;
	  delete[] zzSpPath; delete[] ttSpPath;
	}

      /*----------------------------------------
	Deallocate spec-pointers & destroy plans

	FFTW_ON --> destroy_xyzPlan_fftw2D();
	
	Note: FFTW is not used anymore,
	..... MKL-Dfti replaced it; */

      deAlloc_dble_array(xySpField, Gsz, Gsz);
      deAlloc_dble_array(yzSpField, Gsz, Gsz);   
      deAlloc_dble_array(zzSpField, Gsz, Gsz);
      deAlloc_dble_array(ttSpField, Gsz, Gsz);   

      if ((with_DFTcodes) && (!qcrystal)) // Free FFT-plan (MKL's FFT);
	{
	  mklStat = DftiFreeDescriptor(&handle);
	}   
    }

  //===================================================
  // Record 1d-histograms of ordered state projections:
  /*
    The histogram is made of N * Hst1dSz counters, with
    N = 1 (Neel state) or N = 4 (stripe states for qct); */

  if (qcrystal)
    {
      /*-----------------------
       Record binary data files */ 
      
      string fileName1 = "MC_Bin_NeelHist";
      string fileName2 = "MC_Bin_StrpHist";

      string outName1 = outDir0 + fileName1 + tTagBin;
      string outName2 = outDir0 + fileName2 + tTagBin;

      ofstream Bin1, Bin2; //( Binary files for initialization )

      remove(outName1.c_str()); // Delete 'BinX'
      remove(outName2.c_str()); // if it exists;
      
      Bin1.open(outName1, ios::out | ios::binary | ios::app);
      Bin2.open(outName2, ios::out | ios::binary | ios::app);

      for (k = 0; k < Hst1dSz; k++)
	{
	  Bin1.write(reinterpret_cast<const char*>(&NeelHist[k]), szLst1i);
	  Bin2.write(reinterpret_cast<const char*>(&StrpHist[k]), szLst4i);
	}
      
      Bin1.close();
      Bin2.close();

      /*-----------------------------
	Compute normalization factors */

      vector<double> normVec(4, 0.0);

      vector<int> icntVec(4, 0);

      double normFac; int icnt = 0;
      
      for (k = 0; k < Hst1dSz; k++)
	{	  	  
	  for (n = 0; n < 4; n++)
	    {
	      icntVec[n] += StrpHist[k][n];
	    }

	  icnt += NeelHist[k][0];
	}

      for (n = 0; n < 4; n++)
	{
	  normVec[n] = 1.0 / icntVec[n];
	}

      normFac = 1.0 / icnt;

      /*------------------------------------------------------
	Record 1d-histograms numerical data file (2 + N cols): */

      string fname1 = "NeelHist" + tTagDat;
      string fname2 = "StrpHist" + tTagDat;
      
      ofstream recHist1(outDir1 + subDir1 + fname1);
      ofstream recHist2(outDir1 + subDir1 + fname2);                     
      
      double xpos, HistVal;
      
      for (i = 0; i < Hst1dSz; i++)
	{
	  xpos = i * boxSz1 - 1.0;
	  
	  recHist1 << fmtDbleFix(xpos, 4, 8) << X2;
	  recHist2 << fmtDbleFix(xpos, 4, 8) << X2;

	  HistVal = normFac * NeelHist[i][0];
			      
	  recHist1 << fmtDbleSci(HistVal, 6, 14) << X2;
	  
	  for (n = 0; n < 4; n++)
	    {
	      HistVal = normVec[n] * StrpHist[i][n];
			      
	      recHist2 << fmtDbleSci(HistVal, 6, 14) << X2;
	    }

	  recHist1 << endl;
	  recHist2 << endl;
	}

      recHist1.close();
      recHist2.close();

      delete[] NeelHist;
      delete[] StrpHist;
    }
  
  //===================================================
  // Record 2d-histogram of the vector order-parameter:

  if (qcrystal)
    {
      /*----------------------
       Record binary data file */
      
      string fileName = "MC_Bin_SMagVecHist";

      string outName = outDir0 + fileName + tTagBin;

      ofstream Bin0; //( Binary file for initialization )

      remove(outName.c_str()); //( Delete 'Bin0' if it exists )
    
      Bin0.open(outName, ios::out | ios::binary | ios::app);

      for (k = 0; k < NHist2d; k++)
	{
	  Bin0.write
	    (reinterpret_cast<const char*>(&SMagVecHist[k]), intgSz);
	}
      
      Bin0.close();

      /*----------------------------
	Compute normalization factor */

      double normFac; int HistSum = 0;
      
      for (k = 0; k < NHist2d; k++)
	{
	  HistSum += SMagVecHist[k];	  
	}

      normFac = 1.0 / HistSum;

      /*--------------------------------------------------
	Record histogram numerical data file (2 + 8 cols):
	counter location (xpos, ypos) & normalized values */

      string fname = "SMagVecHist" + tTagDat;
      
      ofstream recHist(outDir1 + subDir1 + fname);
      
      double xpos, ypos, HistVal;
      
      for (i = 0; i < Hst2dSz; i++)
	{
	  xpos = i * boxSz2 - 1.0;
	  
	  for (j = 0; j < Hst2dSz; j++)
	    {	      
	      ypos = j * boxSz2 - 1.0;

	      k = j * Hst2dSz + i; //( Histogram index )
		  
	      HistVal = normFac * SMagVecHist[k];

	      recHist << fmtDbleFix(xpos, 4, 8) << X2;
	      recHist << fmtDbleFix(ypos, 4, 8) << X2;
			      
	      recHist << fmtDbleSci(HistVal, 6, 14) << endl;
	    }
	  
	  recHist << endl;
	}

      recHist.close();
      
      delete[] SMagVecHist;
    }

  //======================================
  // Read spin-field samples from files &
  // make vector-plots (+ other plots) and 
  // also calculate local-field histograms:
  
  if (recField)
    {
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
      
      /*----------------
	Prepare pointers */

      Vec2d *orderMap;  
      
      int **locFieldHist;

      if ((qcrystal) && (rec_orderMaps))
	{
	  orderMap = new Vec2d[Ns];
	}

      locFieldHist = Alloc_intg_array(NHist, 3);             

      init_intg_array(locFieldHist, NHist, 3, 0);

      /*------------------------
	Define image-tag strings
	and output image names */
      
      string wtag = to_string(wRank);
  
      string imgDir = outDir2 + subDir4;
 
      string headTag0 = "spinsMap_wRank(" + wtag;
      string headTag1 = "orderMap_wRank(" + wtag;
      
      /*-------------------------------
	Read samples, make vector-plots
	of samples & feed the histogram */

      ifstream fieldInput;

      int nmax = NPacks + 1;

      int nx, ny, nz;

      fieldInput.open(samplesFile, ios::in | ios::binary);

      if(iAmRoot)
	{
	  cerr << " Making samples plots + calculate\n"
	       << " & record local-field histograms : ";

	  auto time1 = high_resolution_clock::now();	 
	}
      
      for (n = 0; n < nmax; n++)
	{	  
	  string ptag = to_string(n);
	  
	  string packCode = ")_pack(" + ptag;

	  string outName0 = imgDir + headTag0 + packCode;
	  string outName1 = imgDir + headTag1 + packCode;
      
	  if (disOrder)
	    {		  
	      for (k = 0; k < Ns; k++)// Read impurity map
		{		      // on file header...
		  fieldInput.read
		    (reinterpret_cast<char*>(&impField[k]), intgSz);
		}
	    }
      
	  for (i = 0; i < NSamps; i++)
	    {
	      //............................................
	      // Read sample & calculate/feed the histogram:
	  
	      for (k = 0; k < Ns; k++)
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

	      //.......................................
	      // Generate image plots from the samples:
	      //
#if WITH_OPENCV == 1
	      //
	      string stag = to_string(i);
	  
	      string tailName = ")_sample(" + stag + ").png";

	      outImg0 = outName0 + tailName;
		  	      
	      infoStr = magnetInfoString(Temp, spinField);
		
	      ftag = " sample : " + stag;

	      make_vecField(infoStr, ftag,
			    spinField, gridMat, vecsMat);
	      
	      imwrite(outImg0, vecsMat);

	      if ((qcrystal) && (rec_orderMaps))
		{
		  outImg1 = outName1 + tailName;
	      
		  get_qctSMagField(spinField, orderMap);		     	

		  make_vecMap(infoStr, ftag, orderMap, gridMat, vecsMat);

		  imwrite(outImg1, vecsMat);
		}
	      ///
#endif///////////[ Make vector-plots & order-par. plots ]
	    }
	}

      if (iAmRoot)
	{
	  auto time2 = high_resolution_clock::now();

	  auto dtime = time2 - time1;

	  m = duration_cast<milliseconds>(dtime).count();
	      
	  cerr << "Done! " << m << " ms\n\n";
	}

      fieldInput.close();
	    
      /*----------------------------------
	Record local-spin-field histograms */

      string fname = "locFieldHist0";

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
	      
	      recHist << fmtDbleSci(hval, 6, 14) << X2;
	    }
	  
	  recHist << endl;
	}

      recHist.close();
      
      /*-------------------------
	Free memory from pointers */

      if ((qcrystal) && (rec_orderMaps))
	{
	  delete[] orderMap;
	}
      
      deAlloc_intg_array(locFieldHist, NHist, 3);
    }

  MPI_Barrier(MPI_COMM_WORLD); 

  //=================================================
  // Plot final spin field (vectors on lattice grid):

#if WITH_OPENCV == 1
  ///
  {
    infoStr = magnetInfoString(Temp, spinField);

    ftag = "EndState"; //( Location: bottom margin )
    
    outImg0 = outDir2 + "spin3d_config1";

    outImg0 += "_wRank(" + to_string(wRank) + ")";

    outImg0 += outLabel + ".png";
    
    make_vecField(infoStr, ftag, spinField, gridMat, vecsMat);

    imwrite(outImg0, vecsMat);
    
    gridMat.release(); // Release memory from these
    vecsMat.release(); // auxiliary Mat objects ...
  }
  ///
#endif
  
  //============================
  // Free memory (deallocation):
  
  /* Global objects */

  if (disOrder)
    {
      delete[] impField;
    }

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

  /* Local objects */
  
  delete[] siteList;
  
  deAlloc_dble_array(spinField, Ns, 3);  

  if ((pcMode) && (ptON))
    {      
      deAlloc_dble_array(swapField, Ns, 3);
    }
  
  if (with_DFTcodes)
    {            
      delete[] Sqx0;
      delete[] Sqy0;
      delete[] Sqz0;
  
      deAlloc_dble_array(specField, Nq, 3);  
  
      deAlloc_dble_array(SqField, Nq, 6);
    }
  
#if WITH_OPENCV == 1
  ///
  delete[] imgSites;
  ///
#endif
  
  //=======================================
  // Read binary file & perform operations:
  // record statistical results and more...

  /* -------------------------------
     If the numerical data is needed
     -------------------------------

     1) Set bin-averages output file
     ** for backup numerical data:

     FILE *bckFile; //( C style file )

     string bckName = outDir1 + subDir1;
  
     bckName += "MC_Bin_AvgValues";

     bckName += tTagDat;

     ......................................

     2) Open file and add recording 
     ** command to the reading loop:
       
     bckFile = fopen(bckName.c_str(), "w");

     fprintf(bckFile, Fmt_i1fX, k, (...) );

     fclose(bckFile);
     
     ...................................... */
  
  MPI_Barrier(MPI_COMM_WORLD);

  //// STAT_SCOPE (START)
  {
    /*-----------------------
      Settings & preparations

      ........................................
      Note: change Ms1, Ms2 & Ms3 values in
      the 'Parameters' section of the MC-code
      if more data is read from the input fi-
      le and/or recorded on the output files: 
      inFile1, inFile2 & statFile (see below);
      ........................................ */
    
    /* Set input file: bin averages */

    ifstream inFile1, inFile2, inFile3;

    string inputName1 = outDir0;
    string inputName2 = outDir0;
    string inputName3 = outDir0;

    inputName1 += "MC_Bin_AvgValue1";
    inputName2 += "MC_Bin_AvgValue2";
    inputName3 += "MC_Bin_AvgValue3";

    inputName1 += tTagBin;
    inputName2 += tTagBin;
    inputName3 += tTagBin;

    /* Set output file: statistical results */

    ofstream statFile1, statFile2, statFile3;

    string outName1 = outDir1;
    string outName2 = outDir1;
    string outName3 = outDir1;
    
    outName1 += "MC_Stat_Results1";
    outName2 += "MC_Stat_Results2";
    outName3 += "MC_Stat_Results3";

    outName1 += outLabel + ".dat";
    outName2 += outLabel + ".dat";
    outName3 += outLabel + ".dat";
    
    /*------------------------
      Define formatting string */

    string col1, col2;

    string OutFormat = "%05d  ";
    
    for (i = 0; i < Ms1; i++)
      {
	OutFormat += "%16.10e   ";
      }

    OutFormat += " \n";

    const char* Fmt_i1fX = OutFormat.c_str();
    
    /*-----------------------
      Open input binary files */

    inFile1.open(inputName1, ios::binary);
    inFile2.open(inputName2, ios::binary);
    inFile3.open(inputName3, ios::binary);

    if (iAmRoot)
      {
	flag == 0;

	if (!inFile1.is_open())
	  {
	    cerr << " Error: could not open input binary"
		 << " file! \n Input: " << inputName1;

	    cerr << "\n" << endl; flag = 1;
	  }

	if (!inFile2.is_open())
	  {
	    cerr << " Error: could not open input binary"
		 << " file! \n Input: " << inputName2;

	    cerr << "\n" << endl; flag = 1;
	  }

	if (!inFile3.is_open())
	  {
	    cerr << " Error: could not open input binary"
		 << " file! \n Input: " << inputName3;

	    cerr << "\n" << endl; flag = 1;
	  }
      }//// Input-CHECK(1);

    MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
    if (flag > 0){ MPI_Abort(MPI_COMM_WORLD, 1); }

    /*--------------------------------
      Check data & find number of bins */

    streampos szFile1, szFile2, szFile3;

    inFile1.seekg(0, ios::end);
    inFile2.seekg(0, ios::end);
    inFile3.seekg(0, ios::end);
	
    szFile1 = inFile1.tellg();
    szFile2 = inFile2.tellg();
    szFile3 = inFile3.tellg();

    if (iAmRoot)
      {
	flag = 0;
	
	if (szFile1 % szDataVec1 != 0)
	  {
	    cerr << " Error: input binary file has no"
		 << " valid data!\n Input: " << inputName1;

	    cerr << "\n" << endl; flag = 1;
	  }

	if (szFile2 % szDataVec2 != 0)
	  {
	    cerr << " Error: input binary file has no"
		 << " valid data!\n Input: " << inputName2;

	    cerr << "\n" << endl; flag = 1;
	  }

	if (szFile3 % szDataVec3 != 0)
	  {
	    cerr << " Error: input binary file has no"
		 << " valid data!\n Input: " << inputName3;

	    cerr << "\n" << endl; flag = 1;
	  }
      }//// Input-CHECK(2);

    MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
    if (flag > 0)
      {
	inFile1.close();
	inFile2.close();
	inFile3.close();

	MPI_Abort(MPI_COMM_WORLD, 1);
      }
    else // Move cursor to beggining of files...
      {
	inFile1.seekg(0, ios::beg);
	inFile2.seekg(0, ios::beg);
	inFile3.seekg(0, ios::beg);
      }

    /*---------------------------
      Update 'NBins' if necessary */
    
    const int NBins01 = szFile1 / szDataVec1;
    const int NBins02 = szFile2 / szDataVec2;
    const int NBins03 = szFile3 / szDataVec3;

    if (iAmRoot)
      {
	flag = 0;
	
	if ( (NBins01 != NBins02) ||
	     (NBins01 != NBins03) ||
	     (NBins02 != NBins03) )
	  {
	    cerr << " Bin-Error :: input binary"
		 << " files are not compatible!";

	    cerr << "\n" << endl; flag = 1;
	  }
	else//( last check before stat. analysis )
	  {
	    int NBins0 = NBins01;
	    
	    if (NBins0 > NBins)
	      {
		cerr << " NBins was increased due to \n"
		     << " previous data in bin files:\n"
		     << " \n NBins:  " << NBins
		     << " -- new --> " << NBins0;

		cerr << "\n" << endl;

		NBins = NBins0;
	      }
	  }
      }//// Update NBins (root);

    MPI_Bcast(&flag, 1, MPI_INT, root, MPI_COMM_WORLD);  
  
    if (flag > 0)
      {
	inFile1.close();
	inFile2.close();
	inFile3.close();

	MPI_Abort(MPI_COMM_WORLD, 1);
      }
    else//( Update -NBins: all other threads )
      {
	MPI_Bcast(&NBins, 1, MPI_INT, root, MPI_COMM_WORLD); 
      }

    /*----------------------------
      Prepare pointers for reading */
        
    char* charBuffer;
  
    double **dataArray1, dataVec1[Ms1];
    double **dataArray2, dataVec2[Ms2];
    double **dataArray3, dataVec3[Ms3];

    dataArray1 = Alloc_dble_array(NBins, Ms1);
    dataArray2 = Alloc_dble_array(NBins, Ms2);
    dataArray3 = Alloc_dble_array(NBins, Ms3);

    /*----------------------------------------------
      Read binary file & store values in 'dataArray'

      .............................................
      EDen1, EDen2, spcHeat,
      QBinC4, chiQMag, SBinC4, chiSMag,
      XBinC4, chiXMag, FBinC4, chiFMag,
      chiTMag, QMag1, SMag1, XMag1, FMag1, TMag1,
      SMag1H, SMag1V, SMag1D, RhoS1, RhoS2, rhoStff     
      ......................................................
      xyCLen[0], xyCLen[1], xyCLen[2], xyCLen[3], xyCLen[4],
      yzCLen[0], yzCLen[1], yzCLen[2], yzCLen[3], yzCLen[4],
      zzCLen[0], zzCLen[1], zzCLen[2], zzCLen[3], zzCLen[4],
      ttCLen[0], ttCLen[1], ttCLen[2], ttCLen[3], ttCLen[4],
      xySVal[0], xySVal[1], xySVal[2], xySVal[3], xySVal[4],
      yzSVal[0], yzSVal[1], yzSVal[2], yzSVal[3], yzSVal[4],
      zzSVal[0], zzSVal[1], zzSVal[2], zzSVal[3], zzSVal[4],
      ttSVal[0], ttSVal[1], ttSVal[2], ttSVal[3], ttSVal[4];
      ................................................................
      StPrjVc1[0], StPrjVc2[0], StPrjVc4[0], StPrjBin[0], StPrjChi[0],
      StPrjVc1[1], StPrjVc2[1], StPrjVc4[1], StPrjBin[1], StPrjChi[1],
      StPrjVc1[2], StPrjVc2[2], StPrjVc4[2], StPrjBin[2], StPrjChi[2],
      StPrjVc1[3], StPrjVc2[3], StPrjVc4[3], StPrjBin[3], StPrjChi[3],
      SqPks4P1[0], SqPks4P2[0], SqPks4P4[0], SqPksBin[0], SqPksChi[0],
      SqPks4P1[1], SqPks4P2[1], SqPks4P4[1], SqPksBin[1], SqPksChi[1],
      SqPks4P1[2], SqPks4P2[2], SqPks4P4[2], SqPksBin[2], SqPksChi[2],
      SqPks4P1[3], SqPks4P2[3], SqPks4P4[3], SqPksBin[3], SqPksChi[3],
      N1fprSet[0], N1fprSet[1], N1fprSet[2], N1fprSet[3], N1fprSet[4];
      ................................................................*/
    
    if (iAmRoot)
      {
	cerr << " Statistical analysis ... ";
      }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (k = 0; k < NBins; k++)
      {
	charBuffer = reinterpret_cast<char*>(dataVec1);
      
	inFile1.read(charBuffer, sizeof(dataVec1));

	for (n = 0; n < Ms1; n++)
	  {
	    dataArray1[k][n] = dataVec1[n];
	  }		
      }//// Data-READ (1);

    for (k = 0; k < NBins; k++)
      {
	charBuffer = reinterpret_cast<char*>(dataVec2);
      
	inFile2.read(charBuffer, sizeof(dataVec2));

	for (n = 0; n < Ms2; n++)
	  {
	    dataArray2[k][n] = dataVec2[n];
	  }
      }//// Data-READ (2);

    for (k = 0; k < NBins; k++)
      {
	charBuffer = reinterpret_cast<char*>(dataVec3);
      
	inFile3.read(charBuffer, sizeof(dataVec3));

	for (n = 0; n < Ms3; n++)
	  {
	    dataArray3[k][n] = dataVec3[n];
	  }
      }//// Data-READ (3);
	
    inFile1.close();
    inFile2.close();
    inFile3.close();
    
    /*---------------------------------
      Get mean-value & variance vectors */

    bool USE_GSL = true;

    int GSL_seed = seed;
    
    double mean, sdev;
    
    double *meanList1, *sdevList1;
    double *meanList2, *sdevList2;
    double *meanList3, *sdevList3;
    
    meanList1 = new double[Ms1];   
    sdevList1 = new double[Ms1];

    meanList2 = new double[Ms2];
    sdevList2 = new double[Ms2];

    meanList3 = new double[Ms3];
    sdevList3 = new double[Ms3];

    get_statVectors(USE_GSL, GSL_seed, NBins, Ms1,
		    dataArray1, meanList1, sdevList1);

    get_statVectors(USE_GSL, GSL_seed, NBins, Ms2,
		    dataArray2, meanList2, sdevList2);

    get_statVectors(USE_GSL, GSL_seed, NBins, Ms3,
		    dataArray3, meanList3, sdevList3);
	
    /*------------------------------- 
      Prepare output file (root only) */
    
    if (iAmRoot)
      {
	/* Define names-vector (1) */
    
	string *nameVec1;

	nameVec1 = new string[Ms1];

	nameVec1[0] = "EnrgDens (p1)";
	nameVec1[1] = "EnrgDens (p2)";	
	nameVec1[2] = "Sys spec-heat";

	nameVec1[3] = "Q-Binder4thCM";
	nameVec1[4] = "Q-mag suscept";

	nameVec1[5] = "S-Binder4thCM";
	nameVec1[6] = "S-mag suscept";

	nameVec1[7] = "X-Binder4thCM";
	nameVec1[8] = "X-mag suscept";

	nameVec1[ 9] = "F-Binder4thCM";
	nameVec1[10] = "F-mag suscept";
	
	nameVec1[11] = "T-mag suscept";
	
	nameVec1[12] = "Q-magnet (p1)";
	nameVec1[13] = "S-magnet (p1)";
	nameVec1[14] = "X-magnet (p1)";
	nameVec1[15] = "F-magnet (p1)";
	nameVec1[16] = "T-magnet (p1)";

	nameVec1[17] = "S-magnet (HS)";
	nameVec1[18] = "S-magnet (VS)";
	nameVec1[19] = "S-magnet (DS)";

	nameVec1[20] = "Spin stiff C1";
	nameVec1[21] = "Spin stiff C2";	
	nameVec1[22] = "Sys stiffness"; // [Ms1 - 1];

	/* Define names-vector (2) */
    
	string *nameVec2, tagVec[4];

	nameVec2 = new string[Ms2];
	
	tagVec[0] = "(xy)";
	tagVec[1] = "(yz)";
	tagVec[2] = "(zz)";
	tagVec[3] = "(tt)";

	for (i = 0; i < 4; i++)
	  {
	    n = 5 * i;
	    
	    nameVec2[n + 0] = "SCLen KP " + tagVec[i];
	    nameVec2[n + 1] = "SCLen MX " + tagVec[i];
	    nameVec2[n + 2] = "SCLen MY " + tagVec[i];
	    nameVec2[n + 3] = "SCLen GP " + tagVec[i];
	    nameVec2[n + 4] = "SCLen ZP " + tagVec[i];
	  }

	for (i = 0; i < 4; i++)
	  {
	    n = 5 * i + 20;
	    
	    nameVec2[n + 0] = "SpVal KP " + tagVec[i];
	    nameVec2[n + 1] = "SpVal MX " + tagVec[i];
	    nameVec2[n + 2] = "SpVal MY " + tagVec[i];
	    nameVec2[n + 3] = "SpVal GP " + tagVec[i];
	    nameVec2[n + 4] = "SpVal ZP " + tagVec[i];
	  }

	/* Define names-vector (3) */

	string *nameVec3;

	nameVec3 = new string[Ms3];

	tagVec[0] = "(p0)";
	tagVec[1] = "(p1)";
	tagVec[2] = "(p2)";
	tagVec[3] = "(p3)";

	for (i = 0; i < 4; i++)
	  {
	    n = 5 * i;
	    
	    nameVec3[n + 0] = "StPrjVc1 " + tagVec[i];
	    nameVec3[n + 1] = "StPrjVc2 " + tagVec[i];
	    nameVec3[n + 2] = "StPrjVc4 " + tagVec[i];
	    nameVec3[n + 3] = "StPrjBin " + tagVec[i];
	    nameVec3[n + 4] = "StPrjChi " + tagVec[i];
	  }

	for (i = 0; i < 4; i++)
	  {
	    n = 5 * i + 20;
	    
	    nameVec3[n + 0] = "SpPks4P1 " + tagVec[i];
	    nameVec3[n + 1] = "SpPks4P2 " + tagVec[i];
	    nameVec3[n + 2] = "SpPks4P4 " + tagVec[i];
	    nameVec3[n + 3] = "SqPksBin " + tagVec[i];
	    nameVec3[n + 4] = "SqPksChi " + tagVec[i];
	  }

	i = n + 5; //( previous last index )

	nameVec3[i + 0] = "N1fpar1  " + X4;
	nameVec3[i + 1] = "N1fpar2  " + X4;
	nameVec3[i + 2] = "N1fpar4  " + X4;
	nameVec3[i + 3] = "N1fBinC4 " + X4;
	nameVec3[i + 4] = "chiN1fpar" + X4;
			
	/* Record header tags: nameVec1 ---> statFile1 */

	string tempTxt = "# Temperature (kB = 1)";

	statFile1.open(outName1);

	statFile1 << tempTxt << X3;
	statFile1 << outInfo << X3;

	for (n = 0; n < Ms1; n++)
	  {
	    m = 2 * (n + 1);
	    	    
	    col1 = iformat2(m + 1);
	    col2 = iformat2(m + 2);
	    
	    str1 = nameVec1[n] + " (MV) c" + col1;
	    str2 = nameVec1[n] + " (DV) c" + col2;
	
	    statFile1 << str1 << X3
		      << str2 << X3;
	  }

	/* Record header tags: nameVec2 ---> statFile2 */

	statFile2.open(outName2);

	statFile2 << tempTxt << X3;
	statFile2 << outInfo << X3;
	
	for (n = 0; n < Ms2; n++)
	  {
	    m = 2 * (n + 1);
	    	    
	    col1 = iformat2(m + 1);
	    col2 = iformat2(m + 2);
	    
	    str1 = nameVec2[n] + " (MV) c" + col1;
	    str2 = nameVec2[n] + " (DV) c" + col2;
	
	    statFile2 << str1 << X3
		      << str2 << X3;
	  }

	/* Record header tags: nameVec3 ---> statFile3 */

	statFile3.open(outName3);

	statFile3 << tempTxt << X3;
	statFile3 << outInfo << X3;
	
	for (n = 0; n < Ms3; n++)
	  {
	    m = 2 * (n + 1);
	    	    
	    col1 = iformat2(m + 1);
	    col2 = iformat2(m + 2);
	    
	    str1 = nameVec3[n] + " (MV) c" + col1;
	    str2 = nameVec3[n] + " (DV) c" + col2;
	
	    statFile3 << str1 << X3
		      << str2 << X3;
	  }
	
	/* End lines & delete pointers */  

	statFile1 << endl;
	statFile2 << endl;
	statFile3 << endl;

	delete[] nameVec1;
	delete[] nameVec2;
	delete[] nameVec3;
      }
    
    /*---------------------------------------
      Record first set of results (root only) */

    double lval = labelValue;
    
    if (iAmRoot)
      {
	if (pcMode)
	  {    
	    Temp = TpVec[0]; }
	else
	  { Temp = Temp1; }	

	/*...............................................*/
	
	statFile1 << fmtDbleSci(Temp, 12, 22) << X3;      
	statFile1 << fmtDbleSci(lval, 12, 22) << X3;
    
	for (n = 0; n < Ms1; n++)
	  {
	    mean = meanList1[n];	
	    sdev = sdevList1[n];

	    statFile1 << fmtDbleSci(mean, 12, 22) << X3
		      << fmtDbleSci(sdev, 12, 22) << X3;
	  }

	statFile1 << endl;

	/*...............................................*/

	statFile2 << fmtDbleSci(Temp, 12, 22) << X3;	
	statFile2 << fmtDbleSci(lval, 12, 22) << X3;
	
	for (n = 0; n < Ms2; n++)
	  {
	    mean = meanList2[n];	
	    sdev = sdevList2[n];

	    statFile2 << fmtDbleSci(mean, 12, 22) << X3
		      << fmtDbleSci(sdev, 12, 22) << X3;
	  }

	statFile2 << endl;

	/*...............................................*/
	
	statFile3 << fmtDbleSci(Temp, 12, 22) << X3;	
	statFile3 << fmtDbleSci(lval, 12, 22) << X3;
	
	for (n = 0; n < Ms3; n++)
	  {
	    mean = meanList3[n];	
	    sdev = sdevList3[n];

	    statFile3 << fmtDbleSci(mean, 12, 22) << X3
		      << fmtDbleSci(sdev, 12, 22) << X3;
	  }

	statFile3 << endl;
      }

    /*---------------------------------------
      Collect results & record (Temp > Temp1) */

    double tvar; // Auxiliary variable;
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (pcMode)
      {       
	for (i = 1; i < wSize; i++)
	  {
	    // All non-root processes send data to the root:
	    
	    if (wRank == i)
	      {
		/*.............................................*/
		
		MPI_Send(meanList1, Ms1, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);
				  
		MPI_Send(sdevList1, Ms1, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);

		MPI_Send(meanList2, Ms2, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);
				  
		MPI_Send(sdevList2, Ms2, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);

		MPI_Send(meanList3, Ms3, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);
				  
		MPI_Send(sdevList3, Ms3, MPI_DOUBLE,
			 root, mpi_ctag, MPI_COMM_WORLD);

		/*.............................................*/
	      }

	    // Root receives data from all other processes
	    // and record the output file (MC_Stat_Results):
	    
	    if (iAmRoot)
	      {
		/*.............................................*/
		
		MPI_Recv(meanList1, Ms1, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(sdevList1, Ms1, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(meanList2, Ms2, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(sdevList2, Ms2, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(meanList3, Ms3, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		MPI_Recv(sdevList3, Ms3, MPI_DOUBLE, i, mpi_ctag,
			 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		tvar = TpVec[i];

		statFile1 << fmtDbleSci(tvar, 12, 22) << X3;
		statFile1 << fmtDbleSci(lval, 12, 22) << X3;

		statFile2 << fmtDbleSci(tvar, 12, 22) << X3;
		statFile2 << fmtDbleSci(lval, 12, 22) << X3;

		statFile3 << fmtDbleSci(tvar, 12, 22) << X3;
		statFile3 << fmtDbleSci(lval, 12, 22) << X3;
				
		for (n = 0; n < Ms1; n++)
		  {
		    mean = meanList1[n];	
		    sdev = sdevList1[n];

		    statFile1 << fmtDbleSci(mean, 12, 22) << X3
			      << fmtDbleSci(sdev, 12, 22) << X3;
		  }

		for (n = 0; n < Ms2; n++)
		  {
		    mean = meanList2[n];	
		    sdev = sdevList2[n];

		    statFile2 << fmtDbleSci(mean, 12, 22) << X3
			      << fmtDbleSci(sdev, 12, 22) << X3;
		  }

		for (n = 0; n < Ms3; n++)
		  {
		    mean = meanList3[n];	
		    sdev = sdevList3[n];

		    statFile3 << fmtDbleSci(mean, 12, 22) << X3
			      << fmtDbleSci(sdev, 12, 22) << X3;
		  }

		statFile1 << endl;
		statFile2 << endl;
		statFile3 << endl;
		
		/*.............................................*/
	      }

	    MPI_Barrier(MPI_COMM_WORLD);
	  }
      }
    
    /*---------------------------------
      Close files & deallocate pointers */

    if (iAmRoot)
      {
	statFile1.close();
	statFile2.close();
	statFile3.close();
      }

    deAlloc_dble_array(dataArray1, NBins, Ms1);
    deAlloc_dble_array(dataArray2, NBins, Ms2);
    deAlloc_dble_array(dataArray3, NBins, Ms3);
    
    delete[] meanList1;
    delete[] sdevList1;
    
    delete[] meanList2;    
    delete[] sdevList2;

    delete[] meanList3;    
    delete[] sdevList3;
  }
  //// STAT_SCOPE (END)

  if (iAmRoot){cerr << "done!\n" << endl;}

  //---------------------------------------
  // Append results for each temperature in
  // the corresponding (new or old) file:

  if (iAmRoot) // Root-Append-Data (START);
    {
      double tvar;
      
      ostringstream oss0;
    
      string shortTmp, longTmp;
    
      string inName1 = outDir1 + "MC_Stat_Results1";
      string inName2 = outDir1 + "MC_Stat_Results2";
      string inName3 = outDir1 + "MC_Stat_Results3";

      string prefix1 = outDir1 + subDir3 + "MC_Stat_FixTemp1";
      string prefix2 = outDir1 + subDir3 + "MC_Stat_FixTemp2";
      string prefix3 = outDir1 + subDir3 + "MC_Stat_FixTemp3";

      string outName1, outName2, outName3;
   
      inName1 += outLabel + ".dat";
      inName2 += outLabel + ".dat";
      inName3 += outLabel + ".dat";
    
      tvar = Temp1; // First temperature;
	  
      for (i = 0; i < wSize; i++)
	{
	  if (pcMode){ tvar = TpVec[i]; }
    
	  oss0 << fixed << setprecision(5) << tvar;
    
	  shortTmp = oss0.str();

	  longTmp = fmtDbleSci(tvar, 12, 22);

	  outName1 = prefix1 + "(" + shortTmp + ").dat";
	  outName2 = prefix2 + "(" + shortTmp + ").dat";
	  outName3 = prefix3 + "(" + shortTmp + ").dat";
  
	  collect_data(outDir1, outName1, inName1, longTmp);
	  collect_data(outDir1, outName2, inName2, longTmp);
	  collect_data(outDir1, outName3, inName3, longTmp);

	  oss0.str("");
	}  
    }//// Root-Append-Data (END);

  //-------------------------------
  // Measure auto-correlation time:

  if (iAmRoot)
    {
      if (rec_autocTime)
	{
	  int lsz = int(DataTimeSeries.size());

	  outfile1.open(outDir1 + subDir0 + "mc_rawData.dat");

	  for (k = 0; k < lsz; k++)
	    {
	      outfile1 << DataTimeSeries[k] << endl;
	    }

	  outfile1.close();
	}
    }
  
  /*|=====================|
    |>>>> END OF CODE <<<<|*/
  
  if (iAmRoot)
    {
      if (pcMode){ delete[] TpVec; }
      
      cerr << " End of code!\n" << endl;
    }

  MPI_Barrier(MPI_COMM_WORLD);   
  
  MPI_Finalize(); return 0;
}

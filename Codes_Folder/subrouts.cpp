///////////////////////////////////////////////////////////////////////
//-------------------------------------------------------------------//
// SUBROUTINES FOR THE MAIN CODES (mc_hm_code.cpp & rk_evo_code.cpp) //
//-------------------------------------------------------------------//
/////////////////////////////////////////////////////////////////////// 

/*
  The subroutines and functions here depend on libraries given
  by header files that must be accessible on the system:

  >> OpenCV library:

  #include <opencv2/core.hpp>
  #include <opencv2/opencv.hpp>
  #include <opencv2/videoio.hpp>
  #include <opencv2/highgui.hpp>

  #include <opencv2/photo.hpp>
  #include <opencv2/core/mat.hpp>

  >> FFTW library:

  #include <fftw3.h>

  >> C++ headers:

  #include <iostream>
  #include <stdio.h>
  #include <fstream>
  #include <complex>
  #include <string>
  #include <vector>
  #include <chrono>
  #include <cmath>

  #include <algorithm> 
  #include <cctype>
  #include <locale>

  #include <sys/stat.h>

  >> C++ namespace:

  using namespace cv;
  using namespace std;
  using namespace std::chrono;

  As shown above, we use some namespaces
  in order to simplify the code: cv, std; */

/*====================================
  Loading auxiliary codes and headers:

  1) Limits: provide access to constants 
  ** which can be useful when working
  ** with integer: INT_MAX, INT_MIN; 

  2) C++ code containing many subroutines
  ** and functions needed for the code; */

#include "limits.h" //(1)

#include "aux_code.cpp" //(2)

//===============================
// Global variables of all types:

int xmouse, ymouse; // Mouse click feature (OpenCV);

string fset0 = "Settings_CodeExec.txt";

string fset1 = "Settings_MODEL.txt";

string fset2 = "Settings_MC_SIM.txt";

string fset3 = "Settings_SMD_SIM.txt";

string fset4 = "Settings_Disorder.txt";

/*-----------------------------------------------
  Enable/disable features for the code execution:
  |
  | The default settings are 'false' (off), the
  | values in 'Settings_CodeExec.txt' provide
  | the actual values for the execution;

  PT: parallel tempering;
  
  RKC: Runge-Kutta code;

  TMC: Thermalization Monte-Carlo;

  ptON: enable/disable replica exchange;

  ptAdapt: PT adaptation mode (TMC-code);

  ptTrack: PT replica flow are tracked;

  RKCheck: record energy & mag. variations (RKC);  

  ANNL_ON: use simulated annealing with TMC;

  FFTW_ON: enable FFTW-DFTs in the RKC; */

vector<string> Code_Settings =
  {
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
  };

bool ptON = false; 

bool ptAdapt = false;

bool ptTrack = false;

/*........................*/

bool ANNL_ON = false;

bool FFTW_ON = false;

bool RKCheck = false;

bool PBC_OFF = false;

/*........................*/

bool IsiModel = false;

bool qcrystal = false;

/*........................*/

bool with_recField = false;

bool with_DFTcodes = false;

bool force_Stripes = false;

bool refState_Zero = false;

bool qct_NeelPhase = false;

bool qct_RemovePBC = false;

bool rec_orderMaps = false;

/*.........................
  The variable below cannot
  be set after compilation: */

bool rec_autocTime = true;

/*----------------------------
   Automatic feature switches:

   linTGrid: T-grid with equal spacing, it
   ......... can be changed by the u-input;

   getTGrid: T-grid is read from input file
   ......... checked upon initialization;

   getWData: filter window is read from input 
   ......... file checked upon initialization;
  
   The initial values set below MUST BE FALSE,        
   the actual simulation values depend on the
   user input (see 'convert_inputs2data'); */

bool linTGrid = false;

bool getTGrid = false;

bool getWData = false;

/* ---------------------------------
   Integers & doubles related to the  
   lattice and the Heisenberg model:

   0) Lattice size (Lsz) & number of sites (Ns);
   1) Number of sublattices composing the system;
   2) Local and total number of bonds (Nb);
   3) Total number of 1BZ-wavevectors (Nq);
   4) Total number of neighbor bonds (3 types);
   5) Local number of neighbor bonds (3 types);
   6) Exchange interaction strengths (3 types);
   7) Temperature values (parallel tempering);
   8) Applied external mag. field (see 'dMag');
   9) Bond-dependent interaction strengths;
   ** .......................
   ** cc : crystal-crystal  ; ---> { J1 , J2 , JX }
   ** ic : impurity-crystal ;
   ** ii : impurity-impurity;

   Ns = Lsz * Lsz ;

   Ns2 = Ns * Ns ; iNs = 1.0 / Ns ;

   Gsz : DFT-grid size (Nsg = Gsz * Gsz);

   gridFac : DFT-grid amplification factor;

   nSubs : number of underlying sublattices;

   If get_crystal = false (crystal lattice),
   we have Nq = Ns, this equivalence is used
   in the DFT procedures using FFTW/MKL whe-
   re the data periodicity is always assumed; */

int Lsz, Gsz, Ns, Ns0, Ns2, Nsg, nSubs, gridFac; //(0 & 1)

int Nb, Nq, Nb1, Nb2, NbX, Zn1, Zn2, ZnX; //(2 & 5)

double J1, J2, JX, Temp1, Temp2, extH; //(6 - 8)

double J1cc, J1ii, J1ic, J1Mat[2][2]; //(9.1)

double J2cc, J2ii, J2ic, J2Mat[2][2]; //(9.2)

double JXcc, JXii, JXic, JXMat[2][2]; //(9.3);

double impRatio, zLamb1, zLamb2, avgLamb; /* Lambda values & 
					     impurity ratio; */
double iNs, iNq; //( inverse Ns & Nq )

double J2_delta; //( J2 asymmetry for Lieb lattice )

double labelValue; //( output files label value )

Vec3d lambdaVec1, lambdaVec2, lambdaVec3;

/* --------------------------------------------------
   Integers & doubles related to the discretization
   of the 1st Brillouin zone (for short: BZone, 1BZ):

   1) Number of points in the KGMYG/YGMXG path;
   2) Number of horizontal (kx) points in the 1BZ; 
   3) Momentum discretization for the wavevectors; */

int npPath, npk; //(1 & 2)

double dq; //(3)
 
/*------------------------------------
  Parameters for the Monte-Carlo code:

  0) Lattice geometry & initial spin state
  .. and output files label control vars;
  
  1) T-grid form & temporal window form;
  2) Number of thermalization sweeps;
  3) Number of measurements sweeps;
  4) Parallel tempering n-points;
  5) SConf & Video recording skip;
  6) DFT-2D procedure exec. delay;
  7) Microcanonical iterations; 
  8) Can. heat-bath iterations; 
  9) Measurements binning size;

  0) External mag. field direction: 
  **
  ** dMag = 2; dMag = 0 (x), 1 (y); 

  X) Ms1 & Ms2 define the number of
  ** values recorded on the output
  ** files during the MC code:
  **
  ** szDataVec1 = Ms1 * dbleSz;
  ** szDataVec2 = Ms2 * dbleSz;
  ** szDataVec3 = Ms3 * dbleSz;

  Y) PT-grid optimization step, if
  ** too small, the average energy
  ** values may not be good enough;

  Z) Annealing step; */

string geom, IState; //(0)

string outTag, outLabel, outInfo; //(0)

string TpGrid, tWForm, SzTag; //(1)

int NTerm, NMeas, npt, NTd2, NMd2; //(2-4)

int crSkip, vrSkip, configSz; //(5)

int dftDelay1, dftDelay2;// (6)

const int nmicro = 5; //(7)

const int ncanon = 5; //(8)

const int BinSz = 1000; //(9)

const int dMag = 2; //(0)

const int Ms1 = 23; //(X)
const int Ms2 = 40; //(X)
const int Ms3 = 45; //(X)

const int PTAd_Step = 5000; // (Y)

const int ANNL_Step = 1000; // (Z)

const int nmicro2 = 2; // Alternative
const int ncanon2 = 8; // parameters;

const int Hst2dSz = 41; // 2d-histogram dim. (odd);
const int Hst1dSz = 41; // 1d-histogram dim. (odd);

/*--------------------------------------
  Parameters for the Runge-Kutta method:

  1) Number of time & frequency steps, 
  ** number of space-time points, and
  ** number of spectrum frequency cuts;
  
  2) Temperature list size (npt : number
  ** of replicas used in parallel tempe-
  ** ring, same as MPI's world size used
  ** in the MC simulation);

  3) Temperature index within the mentio-
  ** list, it designates the input spin-
  ** configurations for the RK procedure
  ** that performs the time evolution;
  
  4) Number of positive freqs. points w/
  ** the 0Hz mode (npw = ntm / 2 + 1
  ** and ndim = npw - 1);

  5) Real time discretization, maximum
  ** time for RK temporal evolution and
  ** frequency discretization & maximum
  ** frequency for 1D Fourier transform;

  6) Frequency range extender factor &
  ** maximum local-spin-field;

  7) Maximum temperature for RK temporal
  ** evolution (MC stage does not record
  ** spin configs. if Temp > TempMax);

  8) Minimum frequency discretization 
  ** for improved output data quality;

  9) Cutoff (max. value) imposed on the
  ** static structure factor intensity,
  ** it must be small enough to ensure
  ** that the spectral intensity maps
  ** can be plotted in linear scale
  ** instead of logarithmic one;

  0) Annealing very-high starting temp.;

  IMPORTANT | wfac:
  .................
  Frequency range extender, it controls
  the the value of 'wmax' which sets the 
  time discretization: dtm = pi / wmax;
  The latter must be small enough to en-
  sure a reasonable energy conservation
  during the time evolution RK-procdure; */

int ntm, Nst, ncuts; //(1) | Nst = Ns * ntm;

int tpLstSz, tpIndex, npw, ndim; //(2, 3 & 4)

double dtm, tmax, dwf, wmax; //(5)

double wfac, maxLocField; //(6)

const double TempMax = 2.00; //(7)

const double dwf_min = 0.10; //(8)

const double spec0Max = 10.0E0; //(9)

const double Temp0 = 2.0; //(0)

/*------------------
  Boolean variables:
  
  0) True for finite next-nearest 
  ** neighbors exchange interaction ( J2 );

  0) True for finite distant-neighbors 
  ** exchange interaction strength ( JX ); 

  2) True for triangular-type geometries;

  3) True for asymmetric ( XXZ ) models;

  4) True for a finite external mag. field;

  5) True for a finite disorder ratio;
  
  6) True if: MClive = 1; 
  7) True if:    npt > 1;

  8) J2 asymmetry (Lieb lattice);

  9) True if nSubs > 1 (multi-sublattices); */

bool J2_ON = false; //(0)
bool JX_ON = false; //(1)

bool C3SYM = false; //(2)

bool XXZ_ON = false; //(3)

bool extH_ON = false; //(4)

bool disOrder = false; //(5)

bool vision, pcMode; //(6 & 7)

bool J2ab_ON = false; //(8)

bool multiSubs = false; //(9)

bool recSpinVec, recSpecMov, getEvoData;

/*-----------------
  Numerical limits: */

const double minDble = numeric_limits<double>::lowest();

const double maxDble = numeric_limits<double>::max();

/*----------------
  More parameters:

  1) Root process number;
  2) Spin related objetcs dimensions;
  3) Maximum and minimum display sizes;
  4) Conversion factor: CV_8UC1 to CV_64F;
  5) Useful angles for incode calculations;
  #) For the number Pi, one can use: CV_PI;

  Olde-codes: dbleTiny = 1.0e-25; */

const int root = 0; //(1)

const unsigned int sdim = 3;      //(2.1)

const unsigned int sdim2 = 9;     //(2.2)

const unsigned int minsz = 600;   //(3.1)
   
const unsigned int maxsz = 1200;  //(3.2)

const double fc255 = 1.0 / 255.0; //(4)

const double pi = acos(-1.0);     //(5.1)

const double pi2 = 2.0 * pi;      //(5.2)

const double pi4 = 4.0 * pi;      //(5.2)

const double pi8 = 8.0 * pi;      //(5.2)

const double a120 = pi2 / 3.0 ;   //(5.2)

const double a90 = pi / 2.0;      //(5.2)

const double a60 = pi / 3.0;      //(5.2)

const double a45 = pi / 4.0;      //(5.2)

const double a30 = pi / 6.0;      //(5.2)

const double sq2 = sqrt(2.0);

const double sq3 = sqrt(3.0);

const double dbleTiny = 1.0e-16;

const double dbleSmall = 1.0e-12;

const double txtsz = 0.4; //(Text size for images)

const double pluss1 = +1.0;  // Auxiliary
const double minus1 = -1.0;  // unit factors;

const double c60 = cos(a60); // Auxiliary 60°
const double s60 = sin(a60); // cos/sin factors;

const double c30 = cos(a30); // Auxiliary 30°
const double s30 = sin(a30); // cos/sin factors;

const complex<double> zero = {0.0, 0.0};

const complex<double> xfac1 = {+1.0, 0.0};
const complex<double> xfac2 = {-1.0, 0.0};

const complex<double> yfac1 = {0.0, +1.0};
const complex<double> yfac2 = {0.0, -1.0};

/*------------------
  Auxiliary vectors: */

const cplxVec zeroVec = {zero, zero, zero};

const Vec3d oneVec = {1.0, 1.0, 1.0};

const Vec3d xxVec = {1.0, 0.0, 0.0};
const Vec3d yyVec = {0.0, 1.0, 0.0};
const Vec3d zzVec = {0.0, 0.0, 1.0};

const Vec3d xyVec = {1.0, 1.0, 0.0};
const Vec3d xzVec = {1.0, 0.0, 1.0};
const Vec3d yzVec = {0.0, 1.0, 1.0};

/*-----------------------
  Auxiliary null-vectors: */

const Vec2d null2d = {0.0, 0.0};
const Vec3d null3d = {0.0, 0.0, 0.0};
const Vec4d null4d = {0.0, 0.0, 0.0, 0.0};
const Vec5d null5d = {0.0, 0.0, 0.0, 0.0, 0.0};

/*---------------------
  Size-type parameters: */

const size_t intgSz = sizeof(int);

const size_t boolSz = sizeof(bool);

const size_t dbleSz = sizeof(double);

const size_t cplxSz = sizeof(complex<double>);

const size_t szSpinVec = 3 * dbleSz;

const size_t szDataVec1 = Ms1 * dbleSz;
const size_t szDataVec2 = Ms2 * dbleSz;
const size_t szDataVec3 = Ms3 * dbleSz;

const size_t szLst1i = sizeof(Lst1i);
const size_t szLst4i = sizeof(Lst4i);
const size_t szLst5i = sizeof(Lst5i);

/*-------------------------------------------
  Vectors spanning the lattice (unit spacing)
  and the associated reciprocal lattice:

  #) Stores the shift vectors from each sub-lattice,
  .. initialization with zeroes already included; */

const Vec2d avec1 = {1.0, 0.0};

const Vec2d avec2 = {c60, s60};

const Vec2d bvec1 = {pi2, - pi / s60};

const Vec2d bvec2 = {0.0,  pi2 / s60};

const Vec2d bvec3 = 2.0 * bvec1 + bvec2;

const Vec2d bvec4 = bvec1 + bvec2;

const Vec2d cvec1 =  {pi2, 0.0};

const Vec2d cvec2 =  {0.0, pi2};

vector<double> subDx(3), subDy(3); //(#)

/*-------------------------------------------
  Lattice related double & integer variables:

  1) Auxiliary factor for lattice site location; 
  2) Main component of the 120° order wavevector; */

const double avecDet = (+ avec1[0] * avec2[1]
			- avec1[1] * avec2[0]); //(1)

const double Qval = 4.0 * pi / 3.0; //(2)

/*--------------------
  Useful wave-vectors:
  
  Here, Qv0, Qv1, ..., Qv5 are the 6 wavevectors 
  associated with each edge of the hexagonal 1st
  Brillouin zone for the triangular lattice; */

const Vec2d Qvec0 = {Qval, 0.0};
  
const Vec2d Qvec1 = {+ c60 * Qval, s60 * Qval};
const Vec2d Qvec2 = {- c60 * Qval, s60 * Qval};

const Vec2d Qvec3 = minus1 * Qvec0;  
const Vec2d Qvec4 = minus1 * Qvec1;
const Vec2d Qvec5 = minus1 * Qvec2;

const vector<Vec2d> QvecList = //(List of Qvecs)
  {
    Qvec0, Qvec1, Qvec2, Qvec3, Qvec4, Qvec5};

/*--------------------------------
  Pointers related to the lattice:

  1) Impurity map pointer & DFT-grip index map;

  2) NN, next-NN & distant-neighbors pointers;

  3) Lattice sites vector list (x and y positions);

  4) Aux-lattice and DFT-grid vectors lists;
  
  5) Lattice bonds list (all nearest-neighbors);

  6) Lattice site points on images/plots; 

  7) List of z-values for both types of neighbors; */

int *impField, *gridMap; //(1)

int **nbors1, **nbors2, **nborsX; //(2)

double **rvecList, **r0List, **dftGrid; //(3 & 4)

Point *bondList, *imgSites, *ir0Sites; //(5 & 6)

Pts3d *zvalList; //(7)

/*--------------------------------------
  Pointers for the ordered state phases:

  See: make_Q120PhaseList ,
  .... make_StrpPhaseList ,
  .... make_2U1DPhaseList ; */

complex<double> *Q120Phase;

double *HStpPhase, *VStpPhase;

double *DStpPhase, *UpDwPhase;

double **Strp1Config, **Strp2Config;
double **Strp3Config, **Strp4Config;

double **Neel0Config;

/*--------------------------------------
  FFTW plans & input/output pointers: */

DFTI_DESCRIPTOR_HANDLE handle = NULL;
    
MKL_LONG mklStat; //( used when calling Dfti )

/* 3D-objects (complex to complex) */

fftw_plan wxPlan, wyPlan, wzPlan;

fftw_complex *rtxData, *qwxData;
fftw_complex *rtyData, *qwyData;
fftw_complex *rtzData, *qwzData;

/* 2D-objects (complex to complex) */

fftw_plan xPlan, yPlan, zPlan;

fftw_complex *rxData, *qxData;
fftw_complex *ryData, *qyData;
fftw_complex *rzData, *qzData;

/* 1D-objects (complex to complex) */

fftw_plan wPlan;

fftw_complex *tmData, *wfData;

/*----------------------------------------
  Strings (based on avaiable directories): */

string outDir0 = "outBin/";

string outDir1 = "outData/";

string outDir2 = "outFigs/";

string outDir3 = "outVids/";

string subDir0 = "Code_Check/";

string subDir1 = "MC_Results/";

string subDir2 = "RK_Results/";

string subDir3 = "MC_Collect/";

string subDir4 = "Samples_Plot/";

string subDirVec[2] = {subDir1, subDir2};

string X2 =  "  "; // Skip 2 spaces;

string X3 = "   "; // Skip 3 spaces;

string X4 = X2 + X2;

string X5 = X2 + X3;

string X8 = X3 + X5;

/*-------------------------------------------
  Plot size & global parameters for plotting:

  Look for: make_latticeGrid & make_vecField;
        
  1) Vector line width;
  2) Grid lines width (lattice)
  3) Spacing between lattice sites;
  4) Radius of the site filled circles;
  5) Vector length (less than 'spc / 2');
  6) Size of the vector tip (arrow); 

  X) Plot sequence (spin components);

  Note: gridSpac minimum value is 80.0; */

#if WITH_OPENCV == 1
//
Size plotSize; //( Cols X Rows )

const Scalar darkkGray(010, 010, 010);
const Scalar fc100Gray(100, 100, 100);
const Scalar lightGray(100, 100, 100);
const Scalar whiteFull(255, 255, 255);
const Scalar blackFull(000, 000, 000);

const Scalar bluColour(255, 000, 000);
const Scalar grnColour(000, 255, 000);
const Scalar redColour(000, 000, 255);

const Scalar ylwColour(000, 200, 255);
const Scalar iceColour(255, 100, 100);
const Scalar magColour(180, 035, 220);
//
#endif

const int vecLwd1 = 5; //(1.1)

const int vecLwd2 = 4; //(1.2)
  
const int gridLwd = 2; //(2)

const double gridSpac = 90.0; //(3)

const double ptRadius = 10.0; //(4)

const double vecSz1 = 0.3 * gridSpac; //(5.1)

const double vecSz2 = 0.2 * gridSpac; //(5.2)

const double tipSz = 0.35; //(6)

int pltSeq[3] = { 0 , 1 , 2 }; //( x y z )

/*==================
  Zero double check: */

bool isZero(double value, double epsilon = 1.0e-12)
{
    return abs(value) < epsilon;
}

bool isNotZero(double value)
{
  constexpr double epsilon = 1.0e-12;
    
  return abs(value) > epsilon;
}

/*===================
  Get histogram index: */

int get_HistIndex(double r1,
		  double r0,
		  double fc,
		  int maxIndex)
{
  int index = round((r1 + r0) * fc);
  
  if (index < 0)
    {
      index = 0;
    }  
  else if (index >= maxIndex)
    {
      index = maxIndex - 1;
    }

  return index;
}

/*========================================
  Loading more external codes and headers:
  
  We get pseudo-random numbers from the 
  double precision SIMD-oriented Fast 
  Mersenne Twister (dSFMT) generator; */

#include "dSFMT_Files/dSFMT.h"

#include "dSFMT_Files/dSFMT.c"

//---------------------------
// dSFMT PRNG initialization:

int dSFMT_init(int seed) 
{
  dsfmt_t dsfmt;

  dsfmt_gv_init_gen_rand(seed);

  return 0;  
}

//-------------------------------
// dSFMT random number generator:

double dSFMT_getrnum()
{
  dsfmt_t dsfmt;
    
  double rnum;

  rnum = dsfmt_gv_genrand_close1_open2();

  rnum = rnum - 1.0; // Shifts distributon 
  //.................// to the interval [0,1);
  
  return(rnum);
}

//----------------------
// Test dSFMT generator:

void dSMT_test()
{
  double x, y, z, phi, tht, drand1, drand2;

  ofstream outfile1, outfile2;
  
  /* Firtly, some random numbers in the plane */

  outfile1.open(outDir1 + "dSFMT_sample.dat");
  
  for (int i = 0; i < 100000; i++)
    {
      drand1 = dSFMT_getrnum();

      outfile1 << i << X4 << drand1 << endl;
    }
  
  outfile1.close();
  
  /* Now, the spherical distributions */

  outfile1.open(outDir1 + "sphere1.dat"); // (1)
  outfile2.open(outDir1 + "sphere2.dat"); // (2)
  
  for (int i = 0; i < 10000; i++)
    {
      //...............
      // WRONG way: (1)
      
      drand1 = dSFMT_getrnum();
      drand2 = dSFMT_getrnum();
       
      tht = 2.0 * pi * drand1;
      phi = 1.0 * pi * drand2;
      
      x = sin(phi) * cos(tht);
      y = sin(phi) * sin(tht);
      z = cos(phi);
      
      outfile1 <<  x  << X4
	       <<  y  << X4
	       <<  z  << X4
	       << tht << X4
	       << phi << endl;

      //.................
      // CORRECT way: (2)
      
      tht = 2.0 * pi * drand1;
      
      phi = acos(1.0 - 2.0 * drand2);
      
      x = sin(phi) * cos(tht);
      y = sin(phi) * sin(tht);
      z = cos(phi);
      
      outfile2 <<  x  << X4
	       <<  y  << X4
	       <<  z  << X4
	       << tht << X4
	       << phi << endl;
    }
  
  outfile1.close();
  outfile2.close();
}

/* -----------
   dSFMT Notes
   -----------

   Generator is faster when working at the
   interval [1,2), the procedure above uses
   this fact and shifts the result to the
   appropriate range;

   The dSFMT generator can be used by calling
   the functions defined above or via the pre-
   defined procedures below:

   Usage via 'dSFMT global variables':

   1) dsfmt_gv_init_gen_rand(seed);

   2) rnum = dsfmt_gv_genrand_close1_open2();

   Usage via the &dsfmt variable:

   1) dsfmt_init_gen_rand(&dsfmt, seed);

   2) rnum = dsfmt_genrand_close1_open2(&dsfmt); */

//===============================================
// Convert user data parameters from input files:

/* NOTE: this procedure sets essential global
   ----- variables for the code execution! */

void convert_inputs2data(string inputs1[11],
			 string inputs2[8],
			 string inputs3[7],
			 string inputs4[7],
			 int wrank,
			 int &inum,
			 int &flag)
{
  bool quit, warn;
  
  /* ============
     Input list 1 
     ============ */

  string latSize, J1ccVal, J2ccVal, JXccVal, J2delta;

  string SzFac1, SzFac2, extMagF, tmp1val, tmp2val, latGeom;
    
  latSize = inputs1[0]; // Lattice size ( Lsz );

  J1ccVal = inputs1[1]; // Nearest, next-nearest and
  J2ccVal = inputs1[2]; // distant-neighbors interaction
  JXccVal = inputs1[3]; // strength (crystal-crystal);

  J2delta = inputs1[4]; // J2 delta (Lieb lattice);

  SzFac1 = inputs1[5]; // Sz-Sz control parameter 1;
  SzFac2 = inputs1[6]; // Sz-Sz control parameter 2;

  extMagF = inputs1[7]; // Applied external magnetic field;

  tmp1val = inputs1[8]; // Small/large temperature values 
  tmp2val = inputs1[9]; // defining the range for PT;

  latGeom = inputs1[10]; // Lattice geometry;
  
  /*------------------------------
    Define number of sublattices &
    number of local NN and dist-N

    NN: nearest neighbors;

    Lattice geometry options:
    square, triang, hexagn, kagome;
    
    nSubs : number of sublattices;

    Zn1(2): number of NN & next-NN; */
    
  quit = false;
  
  if (qcrystal)
    {
      if (latGeom == "Aamman")
	{      
	  nSubs = 1; gridFac = 1;

	  Zn1 = 8; Zn2 = 3; ZnX = 8;
	}
      else // Input error ...
	{
	  if (wrank == root)
	    {
	      cerr << "\n Unknown quasi-geometry : ";

	      cerr << latGeom << endl;
	    }

	  quit = true;
	}
    }
  else//( lattice defined within code )
    {	  
      if (latGeom == "square") 
	{      
	  nSubs = 1;

	  gridFac = 1;

	  Zn1 = 4; Zn2 = 4; ZnX = 4;
	}
      else if (latGeom == "triang")
	{
	  nSubs = 1;

	  gridFac = 1;

	  Zn1 = 6; Zn2 = 6; ZnX = 6;
	}
      else if (latGeom == "hexagn")
	{
	  nSubs = 2;

	  gridFac = 3;

	  Zn1 = 3; Zn2 = 6; ZnX = 3;	  		       
	}
      else if (latGeom == "kagome")
	{
	  nSubs = 3;

	  gridFac = 2;

	  Zn1 = 4; Zn2 = 4; ZnX = 6;
	}
      else if (latGeom == "lieb")
	{	  
	  nSubs = 2;

	  gridFac = 2;

	  Zn1 = 4; Zn2 = 4; ZnX = 4;
	}
      else // Input error ...
	{
	  if (wrank == root)
	    {
	      cerr << "\n Unknown geometry : ";

	      cerr << latGeom << endl;
	    }

	  quit = true;
	}
    }///[ Geometry settings (END)]

  if (quit){ inum = 10; flag = 1; return; }

  /*-----------------------------------
    Set global geometry-type variables:
    string tag (geom) & boolean (C3SYM) */
    
  if (latGeom == "Aamman")
    {
      geom = "square"; }
  else
    { geom = latGeom; }
  
  if ((!qcrystal) && (latGeom != "square") && (latGeom != "lieb"))
    {
      C3SYM = true;
    }

  /*---------------------------------------
    Check & assign data to global variables */

  quit = false;
  
  if (isNumber(latSize))
    {      
      if (qcrystal)//( qct : quasi-crystal )
	{
	  Ns = stoi(latSize);

	  SzTag = "_AppxtSz(" + to_string(Ns) + ")";

	  Lsz = (static_cast<int>(sqrt(Ns)) / 2) * 2;

	  Ns0 = Lsz * Lsz; //( Lsz can be any positive integer )
	}
      else//( if the lattice is defined within code )
	{
	  Lsz = stoi(latSize);
	  
	  if (Lsz % 2 != 0)
	    {
	      if (wrank == root)
		{
		  cerr << "\n Lattice length"
		       << " must multiple of 2!\n";
		}

	      quit = true;
	    }  
      
	  if ( ( Lsz % 3 != 0 ) && C3SYM )
	    {
	      if (wrank == root)
		{
		  cerr << "\n Lattice length"
		       << " must multiple of 3!\n";
		}

	      quit = true;
	    }

	  Ns0 = Lsz * Lsz;
      
	  Ns = nSubs * Ns0;

	  SzTag = ""; // Empty tag;
	}

      Ns2 = pow(Ns, 2); // Auxiliary constant;

      Gsz = gridFac * Lsz; // DFT-grid size;

      Nsg = Gsz * Gsz; // Num. points in DFT-grid;

      Nq = Nsg; // Dim. of the reciprocal-lattice grid;
      
      iNs = 1.0 / Ns; // Useful inverse 
      iNq = 1.0 / Nq; // constants;

      if (nSubs > 1){ multiSubs = true; }
    }
  else
    {
      quit = true;
    }

  if (quit){ inum = 0; flag = 1; return; }

  /*-------------------------------------*/

  quit = false;
  
  if (isFloat(J1ccVal))
    {
      J1cc = stod(J1ccVal); J1 = J1cc;

      if (J1cc < 0.0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n Warning: J1cc"
		   << " is negative; \n";
	    }

	  //quit = true; //( uncomment to abort sim. if J1cc < 0 )
	}    
    }
  else{ quit = true; }

  if (quit){ inum = 1; flag = 1; return; }

  /*-------------------------------------*/
  
  quit = false;

  if (isFloat(J2ccVal))
    {       
      J2cc = stod(J2ccVal); J2 = J2cc;
            
      if (isNotZero(J2cc)){ J2_ON = true; }
    }
  else{ quit = true; }

  if (quit){ inum = 2; flag = 1; return; }

  /*-------------------------------------*/
  
  quit = false;

  if (isFloat(JXccVal))
    {       
      JXcc = stod(JXccVal); JX = JXcc;
            
      if (isNotZero(JXcc)){ JX_ON = true; }
    }
  else{ quit = true; }

  if (quit){ inum = 3; flag = 1; return; }

  /*-------------------------------------*/
  
  quit = false;

  if (isFloat(J2delta))
    {       
      J2_delta = stod(J2delta);

      if (isNotZero(J2_delta))
	{
	  J2_ON = true;
	  
	  J2ab_ON = true;

	  if (latGeom != "lieb")
	    {
	      if (wrank == root)
		{		
		  cerr << " \n AM d-wave J2-coupling requires\n"
		       << " the Lieb lattice (change inputs).\n";
		}

	      quit = true;	    
	    }
	}      
    }
  else{ quit = true; }

  if (quit){ inum = 4; flag = 1; return; }

  /*-------------------------------------*/
  
  quit = false;

  if (isFloat(SzFac1))
    {       
      zLamb1 = stod(SzFac1);
      
      if (zLamb1 != 1.0){ XXZ_ON = true; } 
    }
  else{ quit = true; }

  if (quit){ inum = 5; flag = 1; return; }

  /*-------------------------------------*/

  quit = false;
  
  if (isFloat(SzFac2))
    {       
      zLamb2 = stod(SzFac2);
      
      if (zLamb2 != 1.0){ XXZ_ON = true; } 
    }
  else{ quit = true; }

  if (quit){ inum = 6; flag = 1; return; }
  
  /*-------------------------------------*/
  
  if (isFloat(extMagF))
    {
      extH = stod(extMagF);
      
      if (isNotZero(extH))
	{
	  extH_ON = true;

	  if ((IsiModel) && (dMag != 0))
	    {
	      if (wrank == root)
		{		
		  cerr << " \n Ext. magnetic field must point\n"
		       << " in the x-direction (Ising model)!\n";
		}

	      quit = true;	    
	    }
	}///[ Finite external field ]
    }
  else{ inum = 7; flag = 1; return; }

  /*-------------------------------------*/

  quit = false;
  
  if (isFloat(tmp1val))
    {
      Temp1 = stod(tmp1val);

      if (Temp1 <= 0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n Invalid"
		   << " temperature (1);\n";
	    }

	  quit = true;
	}

      if ((ANNL_ON) && ((Temp0 < Temp1) || (Temp0 < Temp2)))
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Annealing is enable, Temp1(2) must be \n"
		   << " smaller then Temp0 defined on code!   \n";
	    }

	  quit = true;
	}      
    }
  else{ quit = true; }

  if (quit){ inum = 8; flag = 1; return; }

  /*-------------------------------------*/

  quit = false;
  
  if (isFloat(tmp2val))
    {
      Temp2 = stod(tmp2val);

      if (Temp2 < Temp1)
	{
	  if (wrank == root)
	    {
	      cerr << "\n Invalid"
		   << " temperature (2);\n";
	    }

	  quit = true;
	} 
    }
  else{ quit = true; }

  if (quit){ inum = 9; flag = 1; return; }

  /* ============
     Input list 2
     ============ */

  string TMnum, MSnum, SCRnum, VSKnum, MClive;
    
  TMnum = inputs2[0];  // Thermal. sweeps;
  MSnum = inputs2[1];  // Measure. sweeps;
  
  IState = inputs2[2]; // Initial spin-state;
  TpGrid = inputs2[3]; // Temp. grid function;

  outTag = inputs2[4]; // Output files tag;
  
  SCRnum = inputs2[5]; // SConf record iskip;  
  VSKnum = inputs2[6]; // Video record iskip;
  MClive = inputs2[7]; // Live sampling mode;

  /*---------------------------------------
    Check & assign data to global variables */
  
  if (isNumber(TMnum))
    {
      NTerm = stoi(TMnum);

      NTd2 = round(0.5 * NTerm);

      if ((ANNL_ON) && (NTerm % ANNL_Step != 0))
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Annealing is enable, but the number of\n"
		   << " thermalization steps is not a multiple\n"
		   << " of the annealing step: ANNL_Step = "
		   << ANNL_Step << endl;
	    }

	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 0; flag = 2; return; }

  /*---------------------------------------*/

  quit = false;
  
  if (isNumber(MSnum))
    {
      NMeas = stoi(MSnum);

      NMd2 = round(0.5 * NMeas);
      
      if (NMeas % BinSz != 0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Number of measurements"
		   << " must be a multiple  \n"
		   << " of the binning size : "
		   << " BinSz = "
		   <<   BinSz << endl;
	    }

	  quit = true;
	}

      if (NMeas / BinSz == 1)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Number of bins must be"
		   << " greater than 1!  \n" 
		   << " Tip: Increase NMeas"
		   << " or decrease BinSz; "
		   << endl;
	    }
	  
	  quit = true;
	} 
    }
  else{ quit = true; }

  if (quit){ inum = 1; flag = 2; return; }

  /*-------------------------------------- 
    For now, there are only 3 possible ini-
    tial state for the Monte-Carlo process
    of thermalization; */

  quit = false;
  
  if ( IState != "loww_T" &&
       IState != "high_T" &&
       IState != "zpolar" &&
       IState != "2up1dw" &&
       IState != "hstrip" &&
       IState != "vstrip" &&
       IState != "inputf" )
    {
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Initial state string is\n"
	       << " invalid (check inputs);\n";
	}
	
      quit = true;
    }

  if ( geom == "square" && IState == "2up1dw" )
    {      
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Initial state string is\n"
	       << " not compatible with the\n"
	       << " chosen system geometry;\n";
	}

      quit = true; 
    }
  
  if ( geom   != "square" &&
       geom   != "triang" &&
       IState != "high_T" &&
       IState != "zpolar" &&
       IState != "inputf" )
    {      
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Initial state string is\n"
	       << " not compatible with the\n"
	       << " chosen system geometry;\n";
	}

      quit = true; 
    }

  if (IsiModel)
    {
      if ( IState != "high_T" &&
	   IState != "zpolar" &&
	   IState != "inputf" )
	{      
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Initial state string is\n"
		   << " not compatible with the\n"
		   << " chosen spin-model type;\n";
	    }

	  quit = true; 
	}
    }

  if (quit){ inum = 2; flag = 2; return; }

  /*-----------------------------------
    See next procedure for more details */

  quit = false;

  if ( TpGrid == "linear" )
    {
      linTGrid = true;
    }
  else if ( TpGrid == "custom" )
    {
      getTGrid = true;
      
      ptAdapt = false;
    }
  
  if ( TpGrid != "linear" &&
       TpGrid != "nonlin" &&
       TpGrid != "custom" )
    {
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Invalid T-grid option,\n"
	       << " (please check inputs);\n";
	}
	
      quit = true;
    }

  if (quit){ inum = 3; flag = 2; return; }

  /*---------------------------------------*/

  quit = false;
  
  if ( outTag != "Htag"  &&
       outTag != "J1tag" &&
       outTag != "J2tag" )
    {
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Invalid output label option,  \n"
	       << " (please check and fix inputs);\n";
	}
	
      quit = true;
    }

  if (quit){ inum = 4; flag = 2; return; }

  /*---------------------------------------*/

  quit = false;
  
  if (isNumber(SCRnum))
    {
      crSkip = stoi(SCRnum);

      if (crSkip <= 0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Spin configuration"
		   << " recording skip\n"
		   << " number must be a"
		   << " positive int ..."
		   << endl;
	    }

	  quit = true;
	}
      else //( crSkip > 0 )
	{
	  if (BinSz % crSkip)
	    {
	      if (wrank == root)
		{
		  cerr << "\n"
		       << " Spin configuration"
		       << " recording skip  \n"
		       << " number has to be a"
		       << " divisor of BinSz; "
		       << endl;
		}
	  
	      quit = true;
	    }
	}
      
      if (crSkip > BinSz)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Spin configuration"
		   << " recording skip\n"
		   << " number has to be"
		   << " smaller than"
		   << " BinSz" << endl;
	    }
	  
	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 5; flag = 2; return; }
  
  /*---------------------------------------*/

  quit = false;
  
  if (isNumber(VSKnum))
    {
      vrSkip = stoi(VSKnum);

      if (vrSkip < 0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " MC-sampling video"
		   << " recording skip \n"
		   << " has to be zero or"
		   << " positive int ... "
		   << endl;
	    }
	   
	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 6; flag = 2; return; }

  /*---------------------------------------
    The boolean global variable 'vision' is
    set below, it is the key for the user
    interactive part of the code; */
  
  if (isNumber(MClive))
    {
      int m0 = stoi(MClive);

      vision = false;

      if (m0 == 1) vision = true;
    }
  else{ inum = 7; flag = 2; return; }

  /* ============
     Input list 3
     ============ */

  string RKstep, TLstSz, TLstId, dtSet, tWStr, RecVd1, RecVd2;
  
  RKstep = inputs3[0]; // RK number of temporal mesh points;
  TLstSz = inputs3[1]; // Temperature list size (see wSize);
  TLstId = inputs3[2]; // Input temperature list position;
  
  dtSet = inputs3[3];  // Temporal range discretization;
  tWStr = inputs3[4];  // Temporal window function/form;
  
  RecVd1 = inputs3[5]; // Video recording feature 1 (on/off);
  RecVd2 = inputs3[6]; // Video recording feature 2 (on/off);

  /*---------------------------------------
    Check & assign data to global variables */

  quit = false;
  
  if (isNumber(RKstep))
    {
      ntm = stoi(RKstep);

      if (ntm < 10)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Number of time steps for the \n"
		   << " evolution RK-procedure is too\n"
		   << " small for a proper execution!\n";
	    }
	  
	  quit = true;
	}

      Nst = Ns * ntm; //( used by FFTW-DFT routines )

      npw = ntm / 2; ndim = npw - 1;
    }
  else{ quit = true; }

  if (quit){ inum = 0; flag = 3; return; }

  /*---------------------------------------*/

  quit = false;
  
  if (isNumber(TLstSz))
    {      
      tpLstSz = stoi(TLstSz);

      if (tpLstSz < 1)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Input temperature list size\n"
		   << " must be positive integer...\n";
	    }
	   
	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 1; flag = 3; return; }
  
  /*---------------------------------------*/

  quit = false;
  
  if (isNumber(TLstId))
    {
      int i1 = 0;
      
      int i2 = tpLstSz - 1;
      
      tpIndex = stoi(TLstId);

      if ( (tpIndex < i1) ||
	   (tpIndex > i2) )
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Input temp. list position"
		   << " has to be an  \n unsigned"
		   << " integer smaller"
		   << " than tpLstSz;\n";
	    }
	   
	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 2; flag = 3; return; }

  /*---------------------------------------*/

  quit = false;

  if (isFloat(dtSet))
    {
      dtm = stod(dtSet);
      
      if (dtm > 0.05)
	{
	  if (wrank == root)
	    {
	      cerr << "\n"
		   << " Temporal discretization is too\n"
		   << " large to keep energy conserved\n"
		   << " during evolution, use RKCheck \n"
		   << " feature to investigate...     \n";
	    }
	   
	  quit = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 3; flag = 3; return; }

  /*---------------------------------------
    For now, there are 4 possible temporal
    window forms / functions as shown below */

  quit = false;
 
  if ( tWStr != "wread" &&
       tWStr != "gauss" && tWStr != "ntall" &&
       tWStr != "black" && tWStr != "flatw" &&
       tWStr != "hann1" && tWStr != "sincw" &&
       tWStr != "hann2" && tWStr != "lancz" &&
       tWStr != "hann3" && tWStr != "nowin" )
    {
      if (wrank == root)
	{
	  cerr << "\n"
	       << " Invalid temporal window\n"
	       << " (please check inputs); \n";
	}
	
      quit = true;
    }
  else//( set associated global variables )
    {
      tWForm = tWStr;

      if (tWForm == "wread"){ getWData = true; }
    }

  if (quit){ inum = 4; flag = 2; return; }
  
  /*-----------------------------------------
    Below, the boolean variable recSpinVec is
    redefined by VSKnum on the input list 2
    during the execution of the MC-code ... */
  
  quit = false;
  
  if (isNumber(RecVd1))
    {
      int m0 = stoi(RecVd1);

      recSpinVec = false;

      getEvoData = false;
	    
      if (m0 == 1)
	{                    
	  recSpinVec = true;
	}
      else if (m0 == 2)
	{
	  getEvoData = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 5; flag = 3; return; }

  /*---------------------------------------*/
  
  quit = false;
  
  if (isNumber(RecVd2))
    {
      int m0 = stoi(RecVd2);

      recSpecMov = false;
	    
      if (m0 == 1){ recSpecMov = true; }
    }
  else{ quit = true; }

  if (quit){ inum = 6; flag = 3; return; }

  /* ============
     Input list 4
     ============ */

  string ipRatio;

  string J1icVal, J2icVal, JXicVal;

  string J1iiVal, J2iiVal, JXiiVal;
 
  ipRatio = inputs4[0]; // Disorder/impurity ratio;
    
  J1icVal = inputs4[1]; // Impurity-crystal J1-value;
  J2icVal = inputs4[2]; // Impurity-crystal J2-value;
  JXicVal = inputs4[3]; // Impurity-crystal JX-value;

  J1iiVal = inputs4[4]; // Impurity-impurity J1-value;
  J2iiVal = inputs4[5]; // Impurity-impurity J2-value;
  JXiiVal = inputs4[6]; // Impurity-impurity JX-value; 

  /*---------------------------------------*/

  quit = false;
  
  if (isFloat(ipRatio))
    {
      impRatio = stod(ipRatio);

      if (impRatio < 0.0 || impRatio >= 1.0)
	{
	  if (wrank == root)
	    {
	      cerr << "\n Impurity ratio must be a "
		   << "number in the range [0,1);\n";
	    }

	  quit = true;
	}
      else if (impRatio > 0.0)
	{
	  disOrder = true;
	}
    }
  else{ quit = true; }

  if (quit){ inum = 0; flag = 4; return; }
  
  /*---------------------------------------
    Set J-values for impurity-crystal (ic)
    and impurity-impurity (ii) interactions */
  
  if (!disOrder)
    {
      J1ii = 0.0; J1ic = 0.0; 
      J2ii = 0.0; J2ic = 0.0;
      JXii = 0.0; JXic = 0.0;
    }
  else//( impurity ratio is positive )
    {
      quit = false; warn = false;

      /*.....................................*/
      
      if (isFloat(J1icVal))
	{
	  J1ic = stod(J1icVal);

	  if (J1ic < 0.0){ warn = true; }    
	}
      else{ quit = true; }

      if (quit){ inum = 1; flag = 4; return; }

      /*.....................................*/
    
      if (isFloat(J2icVal))
	{       
	  J2ic = stod(J2icVal);

	  if (J2ic < 0.0){ warn = true; }  
      
	  if (isNotZero(J2ic)){ J2_ON = true; } 
	}
      else{ quit = true; }

      if (quit){ inum = 2; flag = 4; return; }

      /*.....................................*/
    
      if (isFloat(JXicVal))
	{       
	  JXic = stod(JXicVal);

	  if (JXic < 0.0){ warn = true; }  
      
	  if (isNotZero(JXic)){ JX_ON = true; } 
	}
      else{ quit = true; }

      if (quit){ inum = 3; flag = 4; return; }
      
      /*.....................................*/

      if (isFloat(J1iiVal))
	{       
	  J1ii = stod(J1iiVal);

	  if (J1ii < 0.0){ warn = true; }  
	}
      else{ quit = true; }

      if (quit){ inum = 4; flag = 4; return; }

      /*.....................................*/

      if (isFloat(J2iiVal))
	{       
	  J2ii = stod(J2iiVal);

	  if (J2ii < 0.0){ warn = true; }  
      
	  if (isNotZero(J2ii)){ J2_ON = true; } 
	}
      else{ quit = true; }

      if (quit){ inum = 5; flag = 4; return; }

      /*.....................................*/

      if (isFloat(JXiiVal))
	{       
	  JXii = stod(JXiiVal);

	  if (JXii < 0.0){ warn = true; }  
      
	  if (isNotZero(JXii)){ JX_ON = true; } 
	}
      else{ quit = true; }

      if (quit){ inum = 6; flag = 4; return; }
      
      /*.....................................*/

      if ((warn) && (wrank == root))
	{
	  cerr << "\n Warning: some J-values"
	       << " are negative!\n ";
	}
    }

  /*---------------------------------
    Set shift vectors for sublattices
    forming periodic geometries ...

    Note: size-factor is used in another subroutine
    to define an auxiliary lattice (square, triang.)
    with increased lattice spacing in order to ensu-
    re that the resultant lattice composed by the
    sublattices have unity lattice spacing; but,
    here we want the opposite, so we divide by
    szFac the components of the shift vectors
    associted with each sublattice; */

  double angle, szFac; int n;

  if ((!qcrystal) && (multiSubs))
    {      
      if (latGeom == "hexagn")
	{
	  szFac = sq3;

	  subDx[0] = 0.0; subDx[1] = 0.5 * sq3 / szFac;
	  subDy[0] = 0.0; subDy[1] = 0.5 * 1.0 / szFac;		  		       
	}
      else if (latGeom == "kagome")
	{
	  szFac = 2.0;

	  for (n = 0; n < nSubs; n++)
	    {
	      angle = pi - n * a60;
		
	      subDx[n] = cos(angle) / szFac;
	      subDy[n] = sin(angle) / szFac;	      
	    }
	}
      else if (latGeom == "lieb")
	{
	  szFac = sq2;

	  for (n = 0; n < nSubs; n++)
	    {
	      angle = pi - n * a90;
		
	      subDx[n] = 0.5 * sq2 * cos(angle) / szFac;
	      subDy[n] = 0.5 * sq2 * sin(angle) / szFac;
	    }	  
	}
    }
      
  /*---------------------------------
    Set J-values on the components of
    of J1, J2 & JX auxiliary matrices */

  double J1max, J2max, JXmax;

  J1max = max({abs(J1cc), abs(J1ii), abs(J1ic)});
  J2max = max({abs(J2cc), abs(J2ii), abs(J2ic)});
  JXmax = max({abs(JXcc), abs(JXii), abs(JXic)});

  maxLocField = J1max * Zn1;
  
  if (J2_ON){ maxLocField += J2max * Zn2; }  
  if (JX_ON){ maxLocField += JXmax * ZnX; }

  J1Mat[0][0] = J1cc; J1Mat[1][1] = J1ii; // Same ion types;
  J2Mat[0][0] = J2cc; J2Mat[1][1] = J2ii;
  JXMat[0][0] = JXcc; JXMat[1][1] = JXii;
     
  J1Mat[0][1] = J1ic; J1Mat[1][0] = J1ic; // Different ion types; 
  J2Mat[0][1] = J2ic; J2Mat[1][0] = J2ic;
  JXMat[0][1] = JXic; JXMat[1][0] = JXic;

  /*--------------------------
    Set plot-sequency vector & 
    vector of lambda factors */

  if ((extH_ON) && (dMag == 2))
    {
      pltSeq[0] = 2; // [ z ]
      pltSeq[1] = 1; // [ y ]
      pltSeq[2] = 0; // [ x ]
    }
  
  if (XXZ_ON)
    {
      avgLamb = 0.5 * ( zLamb1 + zLamb2 );

      if (avgLamb > 1.0) // Change plot sequence
	{                // on 'make_vecField'...
	  pltSeq[0] = 2; // [ z ]
	  pltSeq[1] = 1; // [ y ]
	  pltSeq[2] = 0; // [ x ]
	}
            
      lambdaVec1 = xyVec + zLamb1 * zzVec;
      lambdaVec2 = xyVec + zLamb2 * zzVec;

      lambdaVec3 = lambdaVec2; //( zLamb3 = zLamb2 )
    }
  else//[ O(3) symmetryc model ]
    {
      lambdaVec1 = oneVec;
      lambdaVec2 = oneVec;
      lambdaVec3 = oneVec;
    }

  /*--------------------------
    Configuration pointer size */

  configSz = 3 * Ns;

  /*--------------------------------
    DFT-2D procedure execution delay */

  if (qcrystal)
    {
      dftDelay1 = BinSz;

      dftDelay2 = 100;
    }
  else//( short delay | FFTW )
    {
      dftDelay1 = 10;
      dftDelay2 = 10;
    }

  /*---------------------
    Set tag/label strings */ 
  
  if (outTag == "Htag")
    {
      outLabel = "_extH("; labelValue = extH;

      outInfo = "  Extn. Magnetic Field";
    }
  else if (outTag == "J2tag")
    {
      outLabel = "_J2("; labelValue = J2;
      
      outInfo = "  Coupling strength J2";
    }
  else if (outTag == "J1tag")
    {
      outLabel = "_J1("; labelValue = J1;

      outInfo = "  Coupling strength J1";
    }

  ostringstream oss0;

  string labelStr;

  oss0 << fixed << setprecision(3) << labelValue;
 
  labelStr = oss0.str();

  outLabel = outLabel + labelStr + ")";
}

//===================================================
// Print MC simulation information on screen or file:

void print_MC_info(string fileName)
{
  int NBins = NMeas / BinSz;

  int NSamp = NBins * BinSz / crSkip;
  
  double key = 1;
  
  if (fileName == "onTerminal"){key = 0;}

  if (key == 1) // Print on file...
    {
      string str1 = outDir1 + "0MCSIM_INFO.txt";

      ofstream rec(str1, ios::app | ios::out);
          
      rec << endl;
      
      rec << "  ===================="
	  << "======================" << endl;
      
      rec << "  Monte-Carlo code"
	  << " for the Heisenberg model: " << endl;

      rec << "  (" << geom;
     	  
      rec << " geometry with J(1,2,X) couplings)" << "\n";

      if (IsiModel)
	{
	  rec << "\n Model selection: "
	      << "Heisenberg --> Ising" << endl;

	  if (XXZ_ON)
	    {
	      rec << "\n XXZ parameters will be ignored!\n";
	    }
	}

      if (qcrystal)
	{
	  rec << "\n Quasi-crystal size: " << Ns
	      << " sites (in-plane); \n\n";
	}
      else//( normal/Bravais lattices )
	{
	  rec << "\n Lattice size: " << Lsz << " x " << Lsz;

	  rec << " | " << Ns << " sites (in-plane);\n\n";
	}

      if ((PBC_OFF) || (qct_RemovePBC))
	{
	  rec << " Lattice-PBC are turned off!\n\n";
	}
    
      rec << " J1 (crystal-crystal) = " << J1cc << "\n"  ;
      rec << " J2 (crystal-crystal) = " << J2cc << "\n"  ;
      rec << " JX (crystal-crystal) = " << JXcc << "\n\n";

      if (J2ab_ON)
	{
	  rec << " J2_delta (J2 asymmetry) = " << J2_delta << "\n\n";
	}

      if (disOrder)
	{
	  rec << " J1 (impurity-crystal)  = " << J1ic << "\n"  ;
	  rec << " J2 (impurity-crystal)  = " << J2ic << "\n"  ;
	  rec << " JX (impurity-crystal)  = " << JXic << "\n\n";

	  rec << " J1 (impurity-impurity) = " << J1ii << "\n"  ;
	  rec << " J2 (impurity-impurity) = " << J2ii << "\n"  ;
	  rec << " JX (impurity-impurity) = " << JXii << "\n\n";

	  rec << " Imp. ratio = " << fmtDbleFix(impRatio, 2, 6) << " %\n\n";
	}

      if (XXZ_ON)
	{	  
	  rec << " zLambda1 (Sz-Sz factor 1) = " << zLamb1 << "\n"  ;
	  rec << " zLambda2 (Sz-Sz factor 2) = " << zLamb2 << "\n\n";
	}

      rec << " Temperature 1 = " << Temp1 << "\n"  ;
      rec << " Temperature 2 = " << Temp2 << "\n\n";
    
      rec << " External field value = " << extH << "\n\n";

      rec << " MC initial state: " << IState << "\n\n";
      
      rec << " Thermalization steps = " << NTerm << "\n"  ;
      rec << " MC-measurement steps = " << NMeas << "\n\n";
  
      rec << " Number of bins = " << NBins << "\n"  ;
      rec << " Data-bins size = " << BinSz << "\n\n";

      rec << " Spins config. rec-skip = " << crSkip << "\n"  ;
      rec << " Samples to be recorded = " << NSamp  << "\n\n";
  
      if (pcMode)
	{
	  rec << " MPI enabled: npt = " << npt << "\n\n";
	}

      if (ANNL_ON)
	{
	  rec << " Annealing is enabled!\n\n";
	}

      if (ptON)
	{
	  rec << " PT enabled with";

	  if (ptAdapt)
	    {
	      rec << " optimized T-grid!"; }
	  else
	    {
	      rec << " fixed T-grid!"; }

	  rec << "\n\n";
	}
      else
	{ rec << " PT disabled!\n\n"; }
  	  
      rec << " Outputs label: " << outTag << "\n\n";

      rec.close();
    }
  else // Print on terminal...
    {
      cerr << endl;
      
      cerr << "  ===================="
	   << "======================" << endl;
      
      cerr << "  Monte-Carlo code"
	   << " for the Heisenberg model: " << endl;
      
      cerr << "  (" << geom;      
	  
      cerr << " geometry with J(1,2,X) couplings)" << "\n";

      if (IsiModel)
	{
	  cerr << "\n Model selection: "
	       << "Heisenberg --> Ising" << endl;

	  if (XXZ_ON)
	    {
	      cerr << "\n XXZ parameters will be ignored!\n";
	    }
	}
      
      if (qcrystal)
	{
	  cerr << "\n Quasi-crystal size: " << Ns
	       << " sites (in-plane); \n\n";
	}
      else//( normal/Bravais lattices )
	{
	  cerr << "\n Lattice size: " << Lsz << " x " << Lsz;

	  cerr << " | " << Ns << " sites (in-plane);\n\n";
	}

      if ((PBC_OFF) || (qct_RemovePBC))
	{
	  cerr << " Lattice-PBC are turned off!\n\n";
	}
    
      cerr << " J1 (crystal-crystal) = " << J1cc << "\n"  ;
      cerr << " J2 (crystal-crystal) = " << J2cc << "\n"  ;
      cerr << " JX (crystal-crystal) = " << JXcc << "\n\n";

      if (J2ab_ON)
	{
	  cerr << " J2_delta (J2 asymmetry) = " << J2_delta << "\n\n";
	}

      if (disOrder)
	{
	  cerr << " J1 (impurity-crystal)  = " << J1ic << "\n"  ;
	  cerr << " J2 (impurity-crystal)  = " << J2ic << "\n"  ;
	  cerr << " JX (impurity-crystal)  = " << JXic << "\n\n";

	  cerr << " J1 (impurity-impurity) = " << J1ii << "\n"  ;
	  cerr << " J2 (impurity-impurity) = " << J2ii << "\n"  ;
	  cerr << " JX (impurity-impurity) = " << JXii << "\n\n";

	  cerr << " Imp. ratio = " << fmtDbleFix(impRatio, 2, 6) << " %\n\n";
	}

      if (XXZ_ON)
	{	  
	  cerr << " zLambda1 (Sz-Sz factor 1) = " << zLamb1 << "\n"  ;
	  cerr << " zLambda2 (Sz-Sz factor 2) = " << zLamb2 << "\n\n";
	}

      cerr << " Temperature 1 = " << Temp1 << "\n"  ;
      cerr << " Temperature 2 = " << Temp2 << "\n\n";
    
      cerr << " External field value = " << extH << "\n\n";
  
      cerr << " MC initial state: " << IState << "\n\n";
      
      cerr << " Thermalization steps = " << NTerm << "\n"  ;
      cerr << " MC-measurement steps = " << NMeas << "\n\n";
  
      cerr << " Number of bins = " << NBins << "\n"  ;
      cerr << " Data-bins size = " << BinSz << "\n\n";

      cerr << " Spins config. rec-skip = " << crSkip << "\n"  ;
      cerr << " Samples to be recorded = " << NSamp  << "\n\n";  
  
      if (pcMode)
	{
	  cerr << " MPI enabled: npt = " << npt << "\n\n";
	}

      if (ANNL_ON)
	{
	  cerr << " Annealing is enabled!\n\n";
	}

      if (ptON)
	{
	  cerr << " PT enabled with";

	  if (ptAdapt)
	    {
	      cerr << " optimized T-grid!"; }
	  else
	    {
	      cerr << " fixed T-grid!"; }

	  cerr << "\n\n";
	}
      else
	{ cerr << " PT disabled!\n\n"; }

      cerr << " Outputs label: " << outTag << "\n\n";
    }
}

//===================================================
// Print RK simulation information on screen or file:

void print_RK_info(string fileName)
{  
  double key = 1;
  
  if (fileName == "onTerminal"){key = 0;}

  if (key == 1) // Print on file...
    {
      string str1 = outDir1 + "0RKSIM_INFO.txt";

      ofstream rec(str1, ios::app | ios::out);
          
      rec << endl;
      
      rec << "  ===================="
	  << "======================" << endl;
      
      rec << "  Runge-Kutta code"
	  << " for the Heisenberg model: " << endl;
      
      rec << "  (" << geom;   
	  
      rec << " geometry with J(1,2,X) couplings)" << "\n";

      if (IsiModel)
	{
	  rec << "\n Model selection: "
	      << "Heisenberg --> Ising" << endl;
	  
	  if (XXZ_ON)
	    {
	      rec << "\n XXZ parameters will be ignored!\n";
	    }
	}
	    
      if (qcrystal)
	{
	  rec << "\n Quasi-crystal size: " << Ns
	      << " sites (in-plane); \n\n";
	}
      else//( normal/Bravais lattices )
	{
	  rec << "\n Lattice size: " << Lsz << " x " << Lsz;

	  rec << " | " << Ns << " sites (in-plane);\n\n";
	}

      if ((PBC_OFF) || (qct_RemovePBC))
	{
	  rec << " Lattice-PBC are turned off!\n\n";
	}
    
      rec << " J1 (crystal-crystal) = " << J1cc << "\n"  ;
      rec << " J2 (crystal-crystal) = " << J2cc << "\n"  ;
      rec << " JX (crystal-crystal) = " << JXcc << "\n\n";

      if (J2ab_ON)
	{
	  rec << " J2_delta (J2 asymmetry) = " << J2_delta << "\n\n";
	}

      if (disOrder)
	{
	  rec << " J1 (impurity-crystal)  = " << J1ic << "\n"  ;
	  rec << " J2 (impurity-crystal)  = " << J2ic << "\n"  ;
	  rec << " JX (impurity-crystal)  = " << JXic << "\n\n";

	  rec << " J1 (impurity-impurity) = " << J1ii << "\n"  ;
	  rec << " J2 (impurity-impurity) = " << J2ii << "\n"  ;
	  rec << " JX (impurity-impurity) = " << JXii << "\n\n";

	  rec << " Imp. ratio = " << fmtDbleFix(impRatio, 2, 6) << " %\n\n";
	}

      if (XXZ_ON)
	{	  
	  rec << " zLambda1 (Sz-Sz factor 1) = " << zLamb1 << "\n"  ;
	  rec << " zLambda2 (Sz-Sz factor 2) = " << zLamb2 << "\n\n";
	}
    
      rec << " External field value = " << extH << "\n\n";

      rec << " Number of time steps (*): " << ntm << "\n\n";

      rec << " Temperature list size: " << tpLstSz << "\n"  ;
      rec << " Input temp. list pos.: " << tpIndex << "\n";

      rec << " \n (*) May change for custom temporal filters!\n\n";
    
      rec.close();
    }
  else // Print on terminal...
    {
      cerr << endl;
      
      cerr << "  ===================="
	   << "======================" << endl;
      
      cerr << "  Runge-Kutta code"
	   << " for the Heisenberg model: " << endl;

      cerr << "  (" << geom;    
	  
      cerr << " geometry with J(1,2,X) couplings)" << "\n";

      if (IsiModel)
	{
	  cerr << "\n Model selection: "
	       << "Heisenberg --> Ising" << endl;

	  if (XXZ_ON)
	    {
	      cerr << "\n XXZ parameters will be ignored!\n";
	    }
	}
     
      if (qcrystal)
	{
	  cerr << "\n Quasi-crystal size: " << Ns
	       << " sites (in-plane); \n\n";
	}
      else//( normal/Bravais lattices )
	{
	  cerr << "\n Lattice size: " << Lsz << " x " << Lsz;

	  cerr << " | " << Ns << " sites (in-plane);\n\n";
	}

      if ((PBC_OFF) || (qct_RemovePBC))
	{
	  cerr << " Lattice-PBC are turned off!\n\n";
	}
    
      cerr << " J1 (crystal-crystal) = " << J1cc << "\n"  ;
      cerr << " J2 (crystal-crystal) = " << J2cc << "\n"  ;
      cerr << " JX (crystal-crystal) = " << JXcc << "\n\n";

      if (J2ab_ON)
	{
	  cerr << " J2_delta (J2 asymmetry) = " << J2_delta << "\n\n";
	}

      if (disOrder)
	{
	  cerr << " J1 (impurity-crystal)  = " << J1ic << "\n"  ;
	  cerr << " J2 (impurity-crystal)  = " << J2ic << "\n"  ;
	  cerr << " JX (impurity-crystal)  = " << JXic << "\n\n";

	  cerr << " J1 (impurity-impurity) = " << J1ii << "\n"  ;
	  cerr << " J2 (impurity-impurity) = " << J2ii << "\n"  ;
	  cerr << " JX (impurity-impurity) = " << JXii << "\n\n";

	  cerr << " Imp. ratio = " << fmtDbleFix(impRatio, 2, 6) << " %\n\n";
	}

      if (XXZ_ON)
	{	    
	  cerr << " zLambda1 (Sz-Sz factor 1) = " << zLamb1 << "\n"  ;
	  cerr << " zLambda2 (Sz-Sz factor 2) = " << zLamb2 << "\n\n";
	}
    
      cerr << " External field value = " << extH << "\n\n";

      cerr << " Number of time steps (*): " << ntm << "\n\n";

      cerr << " Temperature list size: " << tpLstSz << "\n"  ;      
      cerr << " Input temp. list pos.: " << tpIndex << "\n";

      cerr << " \n (*) May change for custom temporal filters!\n\n";
    }
}

//=================================
// Return the impurity trace value:

int impurityTrace()
{
  int isum = 0;
  
  for (int i = 0; i < Ns; i++)
    {
      if (impField[i] == 1)
	{
	  isum += i;
	}            
    }

  return isum;
}

//=================================
// Return the impurity ratio value:

double impurityRatio()
{
  double sum = 0.0;
  
  for (int i = 0; i < Ns; i++)
    {
      sum += static_cast<double>(impField[i]);
    }

  return sum * iNs;
}

//=================================
// Ordered list ---> Shuffled list:

/* Notes: we use the Fisher-Yates,
   algorithm, the random index is 
   taken from the operation::

   j = rand() % (i + 1);

   Above, 'rand()' is an integer number
   in the interval [0, RAND_MAX], the
   indice resulting from the modulus  
   operator '%' gives an integer in
   the interval [0,i], for instance:
   
   4 % 4 = 0     9 % 4 = 1
   5 % 4 = 1    10 % 4 = 2
   6 % 4 = 2    11 % 4 = 3
   7 % 4 = 3    12 % 4 = 0
   8 % 4 = 0    ...       */

void get_shuffledList(int sz, int *outList)
{
  double rnum1, rnum2;

  int i0, j0, i, j;

  /*----------------------------------
    Initialize with ordered values: */

  for (i = 0; i < sz; i++)
    {
      outList[i] = i;
    }

  /*-------------------------------
    Procedure to shuffle values: */

  i0 = sz - 1;
  
  for (i = i0; i > 0; --i)
    {	
      j = rand() % (i + 1);
			   
      j0 = outList[i];
	  
      outList[i] = outList[j];
      
      outList[j] = j0;
    }
}

//===========================================
// Return the site number in row-major order:
// (column-by-column numbering on the lattice)

unsigned int siteNumber(unsigned int i,
			unsigned int j)
{
  unsigned int stnum;
  
  stnum = i * Lsz + j;
  
  return stnum;
}

//=========================================
// Return list of lattice site coordinates:

void make_rvecList(int &flag)
{
  double angle, szFac, fc;

  double xval, yval, rvec[2];
  
  int i, j, k, n, i0, cnt;

  /*----------------------------------------------
    Set size-factor for main or auxiliary lattice

    Hexagonal, Kagome and Lieb lattices are created
    from an auxiliary lattice with bigger lattice
    spacing (size-factor below), so that the 
    final lattice has spacing equal to 1; */
  
  if (geom == "square" || geom == "triang")
    {
      szFac = 1.0;
    }
  else if (geom == "hexagn")
    {
      szFac = sq3;
    }
  else if ( geom == "kagome")
    {
      szFac = 2.0;
    }
  else if ( geom == "lieb" )
    {
      szFac = sq2;
    }

  /*------------------------
    Generate lattice points: (loops over i & j)

    .............
    Commentaries: (triangular geometry)
      
    Site number/label : i * Lsz + j; 

    Notice that the site number increases with the 
    indice i, so that it multiplies the horizontal
    vector 'avec1' of the lattice, while j multi-
    plies the tilted vector 'avec2';

    In the xy-frame of the lattice (y-axis tilted):

    i  --->  x coordinate;
    j  --->  y coordinate;

    In the XY-frame of the laboratory:

    i  --->  X   coordinate ;
    j  --->  X,Y coordinates;

    The vector 'avec2' has an horizontal component
    and as j increases an offset in the X-coordina-
    te is produced, it is given by: j * cos(60°);

    ..........................
    Lattice generation method:

    >> Square and triangular geometry...
    |
    | Direct spanning of the lattice points by means
    | of the Bravais lattice vectors avec1 & avec2;

    >> Hexagonal geometry...
    |
    | Two triangular lattices intertwined, the 2nd
    | one is merely the 1st translated by the vector
    | given by ( 0.5 * sq3, 0.5 ) -- see code below;
    
    >> Kagome geometry...
    |
    | Three-site base forming two sides of an hexagon,
    | so that each hexagon contains 3 sites where the
    | the set of all center points form a triangular
    | lattice;
    .................................................*/
 
  k = 0; /* Site counter intialization!
	    Loop order: i (1) --> j (2) */
  flag = 0;  
  
  for (i = 0; i < Lsz; i++)
    {
      for (j = 0; j < Lsz; j++)
	{
	  if (geom == "square" || geom == "lieb")
	    {
	      rvec[0] = i * szFac;
	      rvec[1] = j * szFac;
	    }
	  else // Triangular-like geometries ...
	    {
	      rvec[0] = szFac * (i * avec1[0] + j * avec2[0]);
	      rvec[1] = szFac * (i * avec1[1] + j * avec2[1]);
	    }

	  if (multiSubs)
	    {
	      n = j + i * Lsz; // Auxiliary lattice index;
	  
	      r0List[n][0] = rvec[0];
	      r0List[n][1] = rvec[1];
	    }

	  if (geom == "square" || geom == "triang")
	    {
	      /* Square or triangle edge point */
	      
	      rvecList[k][0] = rvec[0];
	      rvecList[k][1] = rvec[1];
		
	      k = k + 1;
	    }
	  else if (geom == "hexagn")
	    {
	      /* Single-shift points (lattice 1) */
	      
	      xval = rvec[0] + 0.5 * sq3;
	      yval = rvec[1] + 0.5 * 1.0;

	      rvecList[k][0] = xval;
	      rvecList[k][1] = yval;

	      k = k + 1;

	      /* Double-shift points (lattice 2) */
	      
	      xval = rvec[0] + 1.0 * sq3;
	      yval = rvec[1] + 1.0 * 1.0;

	      rvecList[k][0] = xval;
	      rvecList[k][1] = yval;

	      k = k + 1;
	    }
	  else if (geom == "kagome")
	    {
	      /* 3-site base loop */
	      
	      for (n = 0; n < nSubs; n++)
		{
		  angle = pi - n * a60;
		
		  xval = rvec[0] + cos(angle);
		  yval = rvec[1] + sin(angle);

		  rvecList[k][0] = xval;
		  rvecList[k][1] = yval;

		  k = k + 1;		      
		}	      
	    }
	  else if (geom == "lieb")
	    {
	      /* 2-site base loop */

	      for (n = 0; n < nSubs; n++)
		{
		  angle = pi - n * a90;
		
		  xval = rvec[0] + 0.5 * szFac * cos(angle);
		  yval = rvec[1] + 0.5 * szFac * sin(angle);

		  rvecList[k][0] = xval;
		  rvecList[k][1] = yval;

		  k = k + 1;		      
		}	      
	    }//// Geometry check (END)
	  
	}//// j-loop (END)
      
    }//// i-loop (END) 
 
  if (k != Ns){ flag = 1; return; }

  /*----------------------------------------------------
    Generate DFT-grid & index map: lattice <--> DFT-grid */

  if (with_DFTcodes)
    {  
      fc = szFac / gridFac; //( reduced lattice spacing )

      i0 = 0; //( Lieb and Kagome start with a left shift )

      if (geom == "square" || geom == "lieb")
	{
	  if (geom == "lieb"){ i0 = 1; }
      
	  for (  i = 0; i < Gsz; i++){
	    for (j = 0; j < Gsz; j++)
	      {	  
		rvec[0] = (i - i0) * fc; rvec[1] = j * fc;

		n = j + i * Gsz;
	  
		dftGrid[n][0] = rvec[0];
		dftGrid[n][1] = rvec[1];
	      }
	  }
	}
      else // Triangular-like geometries ...
	{
	  if (geom == "kagome"){ i0 = 1; }
      
	  for (  i = 0; i < Gsz; i++){
	    for (j = 0; j < Gsz; j++)
	      {
		rvec[0] = fc * ((i - i0) * avec1[0] + j * avec2[0]);
		rvec[1] = fc * ((i - i0) * avec1[1] + j * avec2[1]);

		n = j + i * Gsz; 
	  
		dftGrid[n][0] = rvec[0];
		dftGrid[n][1] = rvec[1];
	      }
	  }
	}//[ geom-check end ]

      /* ..............
	 Make index map */

      cnt = 0; //( grid check )

      for (n = 0; n < Nsg; n++)
	{  
	  xval = dftGrid[n][0];
	  yval = dftGrid[n][1];

	  gridMap[n] = (- 1);
	  
	  for (k = 0; k < Ns; k++)
	    { 
	      if ( isZero(xval - rvecList[k][0], 1.0e-8) &&
		   isZero(yval - rvecList[k][1], 1.0e-8) )
		{
		  gridMap[n] = k; cnt++;
		}
	    }
	}
      
      if (cnt != Ns){ flag = 2; return; }
    }
}

//======================================
// Comparison function to sort by angle:

bool compareAngle(const nbInfo &nb1, const nbInfo &nb2)
{
  return nb1.angle > nb2.angle;
}

//================================================
// Subroutine to build pointers listing the neigh-
// bors from many orders associated to each site:

/*...................................
  Description of the variables below:
    
  1) Lattice spacing (> 100 in this code);
  2) Bravais vectors (rescaled by 'spc' ); 
  ...........................................*/
  
void make_nborsTable(int &flag)
{
  const double spc = 500.0; //(1)

  const Vec2d a1 = spc * avec1; //(2)
  const Vec2d a2 = spc * avec2;
 
  int i, j, k, m, n, stnum;

  int n1, n2, nx, ny;

  double angle, szFac;

  /*----------------------------------------------
    Set size-factor for main or auxiliary lattice

    Hexagonal, Kagome and Lieb lattices are created
    from an auxiliary lattice with bigger lattice
    spacing (size-factor below), so that the 
    final lattice has spacing equal to 1; */

  if (geom == "square" || geom == "triang")
    {
      szFac = 1.0;
    }
  else if (geom == "hexagn")
    {
      szFac = sq3;
    }
  else if ( geom == "kagome")
    {
      szFac = 2.0;
    }
  else if ( geom == "lieb" )
    {
      szFac = sq2;
    }
  
  /*------------------------
    Generate lattice points: (loops over i & j) */
 
  Pts3d sitePoint;

  vector<Pts3d> siteList;
  
  for (k = 0; k < Ns; k++)
    {
      nx = round(spc * rvecList[k][0]);
      ny = round(spc * rvecList[k][1]);
      
      sitePoint = Pts3d(nx, ny, k);

      siteList.push_back(sitePoint);
    }
  
  /*..........................
    Get total number of sites: ( nSites = Ns ) */

  size_t listSz = siteList.size();

  int nSites = int(listSz);

  flag = 0;
  
  if (nSites != Ns)
   {
     flag = 1; return;
   }  
  
  //----------------------------------------------------
  // Extend lattice (PBC: periodic boundary conditions):

  Point pt0, rvec[8];

  int nx1, nx2, nSitesPlus;
 
  if (geom == "square" || geom == "lieb")
    {
      nx1 = round(Lsz * szFac * spc);
      nx2 = round(Lsz * szFac * spc);    
      ny  = round(Lsz * szFac * spc);
    }
  else // Triangular-like geometries ...
    {
      nx1 = round(Lsz * szFac * a1[0]);
      nx2 = round(Lsz * szFac * a2[0]);
	  
      ny  = round(Lsz * szFac * (a1[1] + a2[1])) - 1;
    }

  /*...................
    Make shift vectors: */
  
  rvec[0] = Point(+ nx1, 0);
  rvec[1] = Point(- nx1, 0);
  
  rvec[2] = Point(nx2, + ny);
  rvec[3] = Point(nx2, - ny);

  rvec[4] = Point(- nx2, + ny);
  rvec[5] = Point(- nx2, - ny);

  if (geom == "square" || geom == "lieb")
    {
      rvec[6] = Point(0, + ny);
      rvec[7] = Point(0, - ny);
    }
  else // Triangular-like geometries ...
    {
      rvec[6] = Point(+ nx2 + nx1, + ny);
      rvec[7] = Point(- nx2 - nx1, - ny);
    }

  /*........................
    Replicate lattice (PBC): */
   
  for (n = 0; n < nSites; n++)
    {
      sitePoint = siteList[n];

      stnum = sitePoint.z;

      for (i = 0; i < 8; i++)
	{
	  pt0 = rvec[i];
	  
	  n1 = pt0.x + sitePoint.x;
	  n2 = pt0.y + sitePoint.y;

	  siteList.push_back(Pts3d(n1, n2, stnum));
	}
    }

  /*.....................................
    Update numSites (add extended zones): */
  
  listSz = siteList.size();

  nSitesPlus = int(listSz); // Includes extended points;
      
  /*-------------------------------------
    Find site number from coordinates &
    search for its 1st and 2nd neighbors: */

  const double fc1 = 1.10;

  const double fc2 = 1.05;
  
  double dval, angleSum, avgAngle;

  int dist, ic1, ic2, ic3, i0, m1, m2, m3;
  
  int rArea1, rArea2, rArea3, num1, num2, num3;

  vector<nbInfo> nbors1List, nbors2List, nborsXList;
  
  rArea1 = floor(spc * fc1);

  rArea3 = floor(2.0 * spc * fc2);
  
  if (geom == "square" || geom == "lieb")
    {
      rArea2 = floor(sq2 * spc * fc2);
    }
  else // Triangular-like geometries ...
    {
      rArea2 = floor(sq3 * spc * fc2);      
    }

  flag = 0; // Remains zero if no problems appear;
  
  for (n = 0; n < nSites; n++)
    {
      /*.......................................
	Select a site on the lattice and search
	for its nearest 1st and 2nd neighbors: */

      sitePoint = siteList[n];
    	  
      nx = sitePoint.x;
      ny = sitePoint.y;

      stnum = sitePoint.z;

      angleSum = 0.0;

      for (m = 0; m < nSitesPlus; m++) // Lattice search
	{                              // loop (START)
	  sitePoint = siteList[m];

	  i0 = sitePoint.z;
		  
	  if (i0 != stnum)
	    {
	      n1 = sitePoint.x;
	      n2 = sitePoint.y;
	      
	      dist = floor(sqrt( pow(n1 - nx, 2) +
				 pow(n2 - ny, 2) ));

	      angle = atan2(n2 - ny, n1 - nx);

	      if ((m >= nSites) && (PBC_OFF))
		{
		  i0 = -1; //( ghost neighbor )
		}
	  
	      if (dist < rArea1)
		{
		  angleSum += angle;
		  
		  nbors1List.push_back(nbInfo(i0, angle));
		}
	      else if (dist > rArea1 && dist < rArea2)
		{		  
		  nbors2List.push_back(nbInfo(i0, angle));
		}
	      else if (dist > rArea2 && dist < rArea3)
		{		  
		  nborsXList.push_back(nbInfo(i0, angle));
		}
	    }	      
	}//// Lattice search loop (END);

      size_t listSz1, listSz2, listSz3;

      listSz1 = nbors1List.size();
      listSz2 = nbors2List.size();
      listSz3 = nborsXList.size();

      num1 = int(listSz1); // Number of 1stm
      num2 = int(listSz2); // 2nd and Xth
      num3 = int(listSz3); // neighbors;

      if ( (num1 != Zn1) ||
	   (num2 != Zn2) ||
	   (num3 != ZnX) )
	{
	  flag = 2; return; 
	}
      
      /*............................... | NOTE: ordering of next-next-
	Reorder list elements so that   | NN is not implemented for 
	it follows the plaquette order: | geom = "hexagn" or "kagome"; */
            
      sort(nbors1List.begin(), nbors1List.end(), compareAngle);
      sort(nbors2List.begin(), nbors2List.end(), compareAngle);
      sort(nborsXList.begin(), nborsXList.end(), compareAngle);

      if (geom == "hexagn")
	{
	  if (stnum % 2 == 0)
	    {
	      rotate(nbors2List.begin(),
		     nbors2List.begin() + 1, nbors2List.end());
	    }
	  else
	    {
	      rotate(nbors2List.begin(),
		     nbors2List.begin() + 2, nbors2List.end());
	    }
	}
      else if (geom == "kagome")
	{
	  ic1 = 1; // Zero average
	  ic2 = 1; // angle ic-values;

	  avgAngle = angleSum * (1.0 / num1);
	 	  
	  if (avgAngle > 0.0)
	    {
	      if (avgAngle > 1.0)
		{
		  ic2 = ic2 - 1;
		}
	      else
		{
		  ic1 = ic1 - 1;
		  ic2 = ic2 - 1;
		}
	    }//// Fix rotations;
	   
	  rotate(nbors1List.begin(),
		 nbors1List.begin() + ic1, nbors1List.end());
	  
	  rotate(nbors2List.begin(),
		 nbors2List.begin() + ic2, nbors2List.end());
	}

      /*.............................
	Transfer updated list values: */
	        
      for (k = 0; k < num1; k++)
	{	  
	  nbors1[stnum][k] = nbors1List[k].tag;
	}

      for (k = 0; k < num2; k++)
	{
	  nbors2[stnum][k] = nbors2List[k].tag;
	}

      for (k = 0; k < num3; k++)
	{
	  nborsX[stnum][k] = nborsXList[k].tag;
	}

      nbors1List.clear();
      nbors2List.clear();
      nborsXList.clear();
    }

  /*----------------------------
    Record lists to binary file: */
  
  ofstream nbFile;

  size_t szList1 = Zn1 * intgSz;
  size_t szList2 = Zn2 * intgSz;
  size_t szList3 = ZnX * intgSz;

  string nborsList = outDir0 + "nborsList.bin";
	  
  nbFile.open(nborsList, ios::out | ios::binary);

  for (k = 0; k < Ns; k++)
    {
      nbFile.write(reinterpret_cast<const char*>(nbors1[k]), szList1);
      nbFile.write(reinterpret_cast<const char*>(nbors2[k]), szList2);
      nbFile.write(reinterpret_cast<const char*>(nborsX[k]), szList3);
    }

  nbFile.close();
}

//==================================
// Make impurity map field (vector):

void make_impurityField(double impRatio, int &flag)
{
  /*----------------------------------
    Make shuffled list of site numbers */

  int *siteList = new int[Ns];
  
  get_shuffledList(Ns, siteList);

  /*--------------------------------------
    Assign values to the 'impField' vector 

    Crystal tag  : 0 |
    Impurity tag : 1 | */

  double ratio;

  int i, is, i0 = 1;

  for (i = 0; i < Ns; i++)
    {
      is = siteList[i];
      
      impField[is] = i0;

      ratio = (i + 2) * iNs;
      
      if (ratio > impRatio){ i0 = 0; }
    }

  flag = i0;
  
  delete[] siteList;  
}

//=====================================
/* Get list/vector of z-values (z1,z2):

   In the code (#)-marks, the neighbor
   site indice can be negative in the
   case of a general lattice obtained
   from input files (qcrystal), as
   for quasi-crystal approximants; */

void make_zvalList()
{
  int i, k, nb;

  int znum1, znum2, znumX;
  
  for (i = 0; i < Ns; i++)
    {
      znum1 = 0;
      znum2 = 0;
      znumX = 0;
      
      for (k = 0; k < Zn1; k++)
	{
	  nb = nbors1[i][k];

	  if (nb >= 0)//(#)
	    {
	      znum1 = znum1 + 1;
	    }
	}

      if (J2_ON)
	{
	  for (k = 0; k < Zn2; k++)
	    {	      
	      nb = nbors2[i][k];

	      if (nb >= 0)//(#)
		{
		  znum2 = znum2 + 1;
		}
	    }
	}

      if (JX_ON)
	{
	  for (k = 0; k < ZnX; k++)
	    {	      
	      nb = nborsX[i][k];

	      if (nb >= 0)//(#)
		{
		  znumX = znumX + 1;
		}
	    }
	}
	    
      zvalList[i] = Pts3d(znum1, znum2, znumX);
    }
}

//====================================
/* Get list / vector of lattice bonds:

   In the code (#)-marks, the neighbor
   site indice can be negative in the
   case of a general lattice obtained
   from input files (qcrystal), as
   for quasi-crystal approximants; */

void make_bondList(int nbonds[4],
		   vector<Point> &ij_List)
{
  unsigned int i, j, k;

  /*--------------------
    Define marker array:

    If the marker for the bond (i,j) is 1,
    this bond is considered in the making
    of 'bondList', after that the markers
    for both (i,j) and (j,i) are set to 0; */
  
  int **marker;
   
  marker = Alloc_intg_array(Ns, Ns);
  
  for   ( i = 0; i < Ns; i++ ){
    for ( j = 0; j < Ns; j++ )
      {
	marker[i][j] = 1;}}

  //--------------------
  // Make list of bonds: 

  Point indices;  

  int nb, mk;
  
  /*..........................
    1) Nearest-neighbors bonds */
  
  for (i = 0; i < Ns; i++)
    {      
      for (k = 0; k < Zn1; k++)
	{
	  mk = 0;
	  
	  nb = nbors1[i][k];

	  if (nb >= 0)//(#)
	    {
	      j = nb; //( nb is unsigned )
	      
	      mk = marker[i][j];
	    }
			
	  if (mk == 1)
	    {	      
	      indices = Point(i,j);
	      
	      ij_List.push_back(indices);
	      
	      marker[i][j] = 0;
	      marker[j][i] = 0;
	    }
	}
    }
  
  nbonds[1] = int(ij_List.size());

  /*...............................
    2) Next-nearest-neighbors bonds */
  
  if (!J2_ON)
    {
      nbonds[2] = 0;
    }
  else//[ Include bonds ]
    {
      for (i = 0; i < Ns; i++)
	{	  
	  for (k = 0; k < Zn2; k++)
	    {
	      mk = 0;
	      
	      nb = nbors2[i][k];

	      if (nb >= 0)//(#)
		{
		  j = nb; //( nb is unsigned )
		  
		  mk = marker[i][j];
		}
	  
	      if (mk == 1)
		{	      
		  indices = Point(i,j);
	      
		  ij_List.push_back(indices);
	      
		  marker[i][j] = 0;
		  marker[j][i] = 0;
		}
	    }
	}
  
      nbonds[2] = int(ij_List.size()) - nbonds[1];
    }

  /*.................................................
    3) Distant neighbors bonds (3rd or 5th neighbors) */

  int nbonds12 = nbonds[1] + nbonds[2];
  
  if (!JX_ON)
    {
      nbonds[3] = 0;
    }
  else//[ Include bonds ]
    {      
      for (i = 0; i < Ns; i++)
	{	    
	  for (k = 0; k < ZnX; k++)
	    {
	      mk = 0;
	      
	      nb = nborsX[i][k];

	      if (nb >= 0)//(#)
		{
		  j = nb; //( nb is unsigned )
		  
		  mk = marker[i][j];
		}
	  
	      if (mk == 1)
		{	      
		  indices = Point(i,j);
	      
		  ij_List.push_back(indices);
	      
		  marker[i][j] = 0;
		  marker[j][i] = 0;
		}
	    }
	}
  
      nbonds[3] = int(ij_List.size()) - nbonds12;
    }

  nbonds[0] = ( nbonds[1] + nbonds[2] + nbonds[3] );

  deAlloc_intg_array(marker, Ns, Ns);
}

//===============================================
// Generate list of Bloch-wavevectors in the 1BZ:

/* ------
   Notes:

   The short notation '1BZ' stands for 1st
   Brillouin zone which here considers the
   triangular lattice. The wavevectors are
   the physical Bloch vectors computed by
   means of the rescaled reciprocal latti-
   ce vectors 'bvec1' & 'bvec2';

   The list of wavevectors is given by the
   output vector 'WVecList' consisting of
   two-dimensional vectors 'Vec2d' which
   are appended to 'WVecList' via the
   push_back command. Here, 'nvecs'
   is the number of vectors; */

void get_BlochWaveVectors(int &nvecs,
			  vector<Vec2d> &bwvecList)
{
  const double fc = 1.0 / Gsz;
  
  const Vec2d b1 = fc * bvec1;
  const Vec2d b2 = fc * bvec2;      

  vector<Vec2d> vecList;
  
  Vec2d vec1, vec2;

  int i, j, n, m, Sz;

  /*----------------------------
    Generate wide set of points:
    ( many are outside the 1BZ ) 

    #) Gives length after loops; */

  Sz = 0;

  if ((geom != "square") && (geom != "lieb"))
    {
      for (n = -npk; n <= npk; n++)
	{
	  for (m = -npk; m <= npk; m++)
	    {
	      vecList.push_back(n * b1 - m * b2);
	      
	      Sz += 1; //( # )
	    }
	}
    }
  else // Square geometry: wide set not needed;
    {
      for (n = -npk; n <= npk; n++)
	{
	  for (m = -npk; m <= npk; m++)
	    {
	      vec1[0] = dq * n;
	      vec1[1] = dq * m;
	      
	      bwvecList.push_back(vec1);
	      
	      Sz += 1; //( # )
	    }
	}
    }
      
  /*----------------------------------
    Filter out points outside the 1BZ: (if needed) */

  if ((geom != "square") && (geom != "lieb"))
    {
      const double Qs = Qval + dq / 3.0;

      double x, y, w0, w1, w2, w3, w4;
   
      for (n = 0; n < Sz; n++)
	{
	  vec1 = vecList[n];

	  x = vec1[0];
	  y = vec1[1];

	  w0 = abs(y) - Qs * s60;
      
	  w1 = sq3 * (x - Qs) - y;
	  w2 = sq3 * (x - Qs) + y;
	  w3 = sq3 * (x + Qs) - y;
	  w4 = sq3 * (x + Qs) + y;  

	  if ( (w0 < 0.0) && // Vertical filter
	       (w1 < 0.0) && // and hexagonal
	       (w2 < 0.0) && // sides filter
	       (w3 > 0.0) && // (1 to 4);
	       (w4 > 0.0) )
	    {
	      bwvecList.push_back(vec1);
	    }	
	}
    }

  /*------------------
    Set output integer: */
  
  size_t listLength = bwvecList.size();

  nvecs = int(listLength);  
}

//==========================================
// Generate list of Bloch-wavevectors across
// the KGMYG/YGMXG path within the 1BZ:

void get_qPathWVectors(int &nvecs,
		       vector<Vec2d> &bwvecPath)
{
  const int l2 = Gsz / 2;
  
  const double fc = 1.0 / Gsz;
    
  const Vec2d b1 = fc * bvec1;
  const Vec2d b2 = fc * bvec2;
    
  int i, j, i0, j0, m1, m2;

  Vec2d wvec, wref; 

  /*-------------------------
    Compute path wavevectors: */

  if (C3SYM)
    {
      /*..........
	KGMYG path */
	    
      // K --> G
  
      j0 = l2 / 3;

      for (j = j0; j <= l2; j++)
	{	      
	  i = Gsz - j;
      
	  m1 = i - l2;
	  m2 = j - l2;

	  wvec = m1 * b1 + m2 * b2;

	  bwvecPath.push_back(wvec);
	}

      // G --> M //
  
      j0 = l2 - 1;
  
      for (j = j0; j >= 0; j--)
	{	      
	  i = l2;
      
	  m1 = i - l2;
	  m2 = j - l2;

	  wvec = m1 * b1 + m2 * b2;

	  bwvecPath.push_back(wvec);
	}

      // M --> Y //
  
      for (j = 1; j <= l2 / 2; j++)
	{	      
	  i = l2 - j;
      
	  m1 = i - l2;
	  m2 = j - l2;

	  wvec = m1 * b1 + m2 * b2;

	  bwvecPath.push_back(wvec);
	}

      // Y --> G //

      i0 = l2 / 2 + 1;
 
      for (i = i0; i <= l2; i++)
	{	      
	  j = i;
      
	  m1 = i - l2;
	  m2 = j - l2;

	  wvec = m1 * b1 + m2 * b2;

	  bwvecPath.push_back(wvec);
	}
    }
  else // Square-like geometry ...
    {
      /*..........
	YGMXG path 

      Y : i = l2; j = 0
      G : i = l2; j = l2
      X : i =  0; j = l2
      K : i =  0; j = 0 

      dq = pi2 * fc; 

      vecList : partial path list
      that can be reversed in or-
      der to get YGXMG or YGMXG; */

      vector<Vec2d> vecList;

      // Y --> G //

      wvec[0] = 0.0;
      
      for (i = 0; i < l2; i++)
	{	            	  
	  wvec[1] = i * dq - pi;

	  bwvecPath.push_back(wvec);
	}

      // G --> X //

      j0 = l2;

      wvec[1] = 0.0;
      
      for (j = j0; j >= 0; j--)
	{	            
	  wvec[0] = j * dq - pi;	 

	  vecList.push_back(wvec);
	}

      // X --> M //

      i0 = l2 - 1;

      wvec[0] = (- pi);
      
      for (i = i0; i >= 0; i--)
	{	       	  	  
	  wvec[1] = i * dq - pi;

	  vecList.push_back(wvec);
	}

      // M --> G //

      for (i = 1; i <= l2; i++)
	{	      
	  wvec[0] = i * dq - pi;
	  wvec[1] = i * dq - pi;

	  vecList.push_back(wvec);
	}

      // YGMXG : reverse path order & join lists;
      // YGXMG : comment rev. below & join lists;

      reverse(vecList.begin(), vecList.end());

      bwvecPath.insert(bwvecPath.end(),
		       vecList.begin(), vecList.end());
    }
      
  /*------------------
    Set output integer: */
  
  size_t listLength = bwvecPath.size();

  nvecs = int(listLength);  
}

//=======================================
// Record momentum grid points resulting
// from the 2D discrete Fourier transform:

void rec_qGrid()
{
  const int l2 = Gsz / 2;
  
  const double fc = 1.0 / Gsz;
    
  const Vec2d b1 = fc * bvec1;
  const Vec2d b2 = fc * bvec2;
  
  int i, j, m1, m2;

  double qx, qy;

  Vec2d qvec;

  //---------------------
  // Prepare output file:

  ofstream outfile;

  string outPath = outDir1 + subDir0;
  
  outfile.open(outPath + "zonePts.dat");

  //-----------------------------
  // Record momentum grid points:

  if (geom != "square")
    {
      for (i = 0; i < Gsz; i++)
	{	      
	  for (j = 0; j < Gsz; j++)
	    {      
	      m1 = i - l2;
	      m2 = j - l2;

	      qvec = m1 * b1 + m2 * b2;

	      qx = qvec[0];
	      qy = qvec[1];
		  
	      outfile << qx << X3
		      << qy << endl;
	    }
	}
    }
  else //( Square-type geometry )
    {
      for (i = 0; i < Gsz; i++)
	{	      
	  for (j = 0; j < Gsz; j++)
	    {      
	      m1 = i - l2;
	      m2 = j - l2;

	      qx = dq * m1;
	      qy = dq * m2;

	      outfile << qx << X3
		      << qy << endl;
	    }
	}
    }

  outfile.close();	      
}

//============================= (ROOT ONLY)
/* Record 1st BZone wavevectors
   and set global int npPath...

   .......................................
   Below, some global variables related to
   the 1st Brillouin zone (1st BZone) are
   defined (recall that pi2 = 2.0 * pi): 

   get_BlochWaveVectors ---> np1BZN;
   get_qPathWaveVectors ---> npPath;

   npPath : KGMYG/YGMXG path total points; */

void rec_wvectors_and_set_npPath()
{
  int n0, n, k;

  double w, qx, qy;

  double x0, y0, z0, w0;

  /*-----------------------
    Set local vectors/lists

   1) List of all wavevectors within
   .. the hexagonal/square 1st BZ region;
	 	 
   2) List of vectors on the BZ path:
   .. K --> G --> M --> Y --> G (triang)
   .. Y --> G --> X --> M --> G (square)

   3) Length position on the BZ path:
   .. distance along the path associated
   .. with each point/wavevector in (2); */

  vector<Vec2d> bwvecList; //(1)
	  
  vector<Vec2d> bwvecPath; //(2)	  	 

  double *pathLPos; //(3)

  Vec2d wref, wvec; 
	  	  
  /*-------------------------
    Compute list-1 and record
    Bloch-wavevectors to file */

  ofstream outfile;

  get_BlochWaveVectors(n0, bwvecList);

  cerr << " Total k-points (np1BZN): " << n0;

  cerr << "\n" << endl;
	  
  outfile.open(outDir1 +
	       subDir0 + "wavevec_1BZ.dat");
	  
  for (k = 0; k < n0; k++)
    {
      wvec = bwvecList[k];
      
      outfile << wvec[0] << X3
	      << wvec[1] << endl;
    }

  outfile.close();

  /*-------------------------
    Record 1BZ border to file */

  string fname;
    
  if (geom != "square")
    {
      fname = outDir1 + subDir0 + "Hex1BZ_Border.dat"; }
  else
    { fname = outDir1 + subDir0 + "Sqr1BZ_Border.dat"; }
    
  outfile.open(fname);  

  if (C3SYM)
    {
      for (n = 0; n < 6; n++)
	{
	  wvec = QvecList[n];

	  outfile << wvec[0] << X3
		  << wvec[1] << endl;
	}

      wvec = QvecList[0];
	
      outfile << wvec[0] << X3    // Join with
	      << wvec[1] << endl; // 1st point;
    }
  else // Square-like geometry ...
    {
      w = sq2 * pi;
	    
      for (n = 0; n < 5; n++)
	{
	  x0 = w * cos(a45 + n * a90);
	  y0 = w * sin(a45 + n * a90);

	  outfile << x0 << X3
		  << y0 << endl;
	}
    }
        
  outfile.close();

  /*..................................
    Compute list-2, get npPath value &
    make length position pointer ... */

  get_qPathWVectors(n0, bwvecPath);

  npPath = n0; //( Set global int-variable ) 

  pathLPos = new double[npPath];
	  
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
	  
  /*.....................................
    Record KGMYG / YGMXG path wavevectors
    & their length position to the file */

  if (C3SYM)
    {
      cerr << " KGMYG-points (npPath): ";
	    
      fname = outDir1 + subDir0 + "KGMYG_qPath.dat";
    }
  else // Square-like geometry ...
    {
      cerr << " YGMXG-points (npPath): ";
	      
      fname = outDir1 + subDir0 + "YGMXG_qPath.dat";
    }

  cerr << npPath << "\n" << endl;
	
  outfile.open(fname);   
   
  for (k = 0; k < npPath; k++)
    {	  
      wvec = bwvecPath[k];

      outfile << wvec[0] << "  "
	      << wvec[1] << "  "
	      << pathLPos[k] << endl;
    }
    
  outfile.close();

  delete[] pathLPos;
}

//======================
// Gaussian-type T-grid:

void get_gauss_TGrid(double dtRef,
		     double alpha,
		     double *TVec)
{   
  const double dx = 10.0 / (npt - 1);

  const double Tdiff = Temp2 - Temp1;
  
  double x, x0, fc, TVal; int n;

  double rVec[npt - 1];

  /*---------------------
    Make temperature list */
   
  for (n = 0; n < npt - 1; n++) 
    {
      x = (n + 1) * dx;

      rVec[n] = dtRef * (1.0 - exp(- alpha * pow(x, 2)));
    }

  TVec[0] = 0.0;
	
  for (n = 0; n < npt - 1; n++) 
    {     
      TVec[n + 1] = TVec[n] + rVec[n];
    }

  fc = Tdiff / TVec[npt - 1];

  for (n = 0; n < npt; n++) 
    {
      TVal = fc * TVec[n];
      
      TVec[n] = TVal + Temp1;
    }
}

//==================
// Tanh-type T-grid:

void get_xtanh_TGrid(double shift,
		     double *TVec)
{
  const double A = 2.00;
    
  const double B = 0.35;
    
  const double dx = 10.0 / (npt - 1);

  const double Tdiff = Temp2 - Temp1;
  
  double x, x0, t0, t1, fc, TVal; int n;

  if (shift > 0.5)
    {
      shift = shift - 0.05;
    }
  else if (shift < 0.5)
    {
      shift = shift + 0.05;
    }

  x0 = 10.0 * shift - 5.0;
    
  /*---------------------
    Make temperature list */
   
  for (n = 0; n < npt; n++) 
    {
      x = n * dx - 5.0;

      TVec[n] = 2.0 * (x - A * tanh(B * (x - x0))); 
    }

  t0 = TVec[0];

  t1 = TVec[npt - 1];
  
  fc = 1.0 / (t1 - t0);
	
  for (n = 0; n < npt; n++) 
    {
      TVal = TVec[n];
      
      TVec[n] = fc * (TVal - t0);
    }

  for (n = 0; n < npt; n++) 
    {
      TVal = Tdiff * TVec[n];
      
      TVec[n] = TVal + Temp1;
    }
}

//====================================
// Tanh-type T-grid with 2 parameters:

void get_xtanh_TGrid2(double x1,
		      double x2, double *TVec)
{
  const double A = 0.05;

  const double B = 7.00;
  
  const double xmax = 1.0;
  
  const double dx = xmax / (npt - 1);

  const double Tdiff = Temp2 - Temp1;
  
  double x, w1, w2, r1, r2, t0, t1;

  double fc, TVal; int n;
    
  /*---------------------
    Make temperature list */
   
  for (n = 0; n < npt; n++) 
    {
      x = n * dx;

      w1 = 1.0 - pow(x - x1 - 1.0, 2);
      w2 = 1.0 - pow(x - x2 - 1.0, 2);

      r1 = 0.5 * x - A * tanh(B * w1);
      r2 = 0.5 * x - A * tanh(B * w2);
               
      TVec[n] = r1 + r2;
    }

  t0 = TVec[0];

  t1 = TVec[npt - 1];
  
  fc = 1.0 / (t1 - t0);
	
  for (n = 0; n < npt; n++) 
    {
      TVal = TVec[n];
      
      TVec[n] = fc * (TVal - t0);
    }

  for (n = 0; n < npt; n++) 
    {
      TVal = Tdiff * TVec[n];
      
      TVec[n] = TVal + Temp1;
    }
}

//=====================================
// Generate 120° (C3) order phase list:

/* There are 6 types of C3 symmetric states given
   by the wavevectors associated with each edge of
   the hexagonal 1st Brillouin zone for the trian-
   gular lattice. However these are 3-fold degene-
   rate, hence there are effectively 2 states gi-
   ven by the wavevectors:

   Q+ = (+ Qval, 0) : n = 0 , right wavevector;

   Q- = (- Qval, 0) : n = 1 , left wavevector;

   The mentioned degeneracy comes from the C3 sym-
   metry so that states with ordering wavevectors
   related by a 120° rotation are equivalent;

   See the declaration for 'QvecList' in the hea-
   der with global variables; */

void make_Q120PhaseList()
{
  unsigned int i, j, k;

  double x, tht, pX, pY;

  /*--------------------------
    Make phase factor pointer:
  
    The angle theta (tht) at each site is given
    by the dot product between the site vector
    (x,y) and the ordering wavevector for the
    corresponding ordered state (Qx,Qy):

    tht = Qx * x + Qy * y;
    
    Here we are using the horizontal ordering
    wavevectors, i.e., those with Qy = 0;
  
    The complex phase factor associated with
    the 120° order is given by:

    phi = cos(tht) + i * sin(tht);

    For each site in the lattice, the global
    pointer 'Q120Phase' stores the real and
    imaginary parts of this factor for the
    case of an ordering wavevector Q = Q+;

    For Q = Q- one simply changes the sign 
    of the imaginary part of 'Q120Phase';
   
    In (#), we use the value of the 'avec1'
    and 'avec2' components directly; 

    Note that the inplane state can be set
    in any other plane, one just needs to
    use the phases in the correct manner
    to achieve that... */

  for   (i = 0; i < Lsz; i++){
    for (j = 0; j < Lsz; j++)
      {	  
	tht = Qval * (i + j * c60); //(#)	  
	  
	k = i * Lsz + j;

	pX = cos(tht); pY = sin(tht);

	Q120Phase[k] = complex<double>(pX, pY);
      }
  }
}

//======================================
// Generate 180° (C2) order phase lists:

/* There are 3 types of C2 symmetric states which
   correspond to horizontal (x), vertical (y) and
   diagonal (xy) stripes on triangular lattice; */

void make_StrpPhaseList()
{
  unsigned int i, j, k;

  double tht;

  /*---------------------------
    Make phase factor pointers:

    Site Label: i * Lsz + j ; (i,j) --> (x,y) ;
  
    Horizontal stripes are given by a phase angle
    that depends on j only, i.e., as the indice i
    varies (x-coordinate increases) and the indi-
    ce j remains fixed, the phase must not change; 
    
    Vertical stripes are analagous, we just swap
    the indices j and i so that the phase angle
    in this case must depend on i only. Below,
    we see that j varies (y-coord.increases) 
    and 'VStpPhase' remains fixed;

    Note: XStpPhase is a global pointer; */
  
  for   (i = 0; i < Lsz; i++){
    for (j = 0; j < Lsz; j++)
      {
	k = i * Lsz + j;

	tht = pi * (i + j);

	HStpPhase[k] = cos(j * pi); 
	
	VStpPhase[k] = cos(i * pi);

	DStpPhase[k] = cos(tht);
      }
  }
}

//===========================================
// Generate xy-state: 2 spin up + 1 spin down

void make_2U1DPhaseList()
{
  unsigned int i, j, k, n, stnum;

  double tht;

  /*--------------------------
    Make phase factor pointer: */

  k = 0;
  
  for   (i = 0; i < Lsz; i++){
    for (j = 0; j < Lsz; j++)
      {
	stnum = i * Lsz + j;

	n = (i + j + k + 2) % 3;
	
	tht = n * pi;

	UpDwPhase[stnum] = cos(tht);
      }

    k += 1;
  }
}

//=========================================
// Copy spin-field values to another field:

void copy_field(double **sField, double **clone)
{
  unsigned int i, n;

  for (i = 0; i < Ns; i++)
    {
      for (n = 0; n < 3; n++)
	{
	  clone[i][n] = sField[i][n];
	}
    }
}

//================================================
// Set spin-vector at some site on the spin-field:

void set_localSpin(unsigned int i,
		   const Vec3d &spinVec, double **spinField)
{
  unsigned int n;

  for (n = 0; n < 3; n++)
    {
      spinField[i][n] = spinVec[n];
    }
}

//======================================================
// Extract spin-vector at some site from the spin-field:

void get_localSpin(unsigned int i,
		   double **spinField, Vec3d &spinVec)
{
  unsigned int n;

  for (n = 0; n < 3; n++)
    {
      spinVec[n] = spinField[i][n];
    }
}

//=====================================
/* Compute the effective local field at
   the input i-site resulting from the
   spin-field in its neighborhood:    

   In the code (#)-marks, the neighbor
   site indice can be negative in the
   case of a general lattice obtained
   from input files (qcrystal), as
   for quasi-crystal approximants; */

void get_localField(unsigned int i,
		    double **spinField, Vec3d &locField)
{
  const unsigned int i0 = i;
  
  unsigned int n, k;

  int nb, a1, a2;

  double J1Val, J2Val, JXVal;

  Vec3d J1Vec, J2Vec, JXVec;

  locField[0] = 0.0;
  locField[1] = 0.0;
  locField[2] = 0.0;
  
  if (extH_ON){locField[dMag] = (- extH);}

  for (k = 0; k < Zn1; k++)
    {  
      nb = nbors1[i0][k];

      if (nb >= 0)//(#)
	{
	  if (disOrder)
	    {
	      a1 = impField[i0];
	      a2 = impField[nb];

	      J1Val = J1Mat[a1][a2];
	    }
	  else
	    { J1Val = J1; }

	  J1Vec = J1Val * lambdaVec1;
      
	  for (n = 0; n < 3; n++)
	    {     
	      locField[n] += J1Vec[n] * spinField[nb][n];
	    }
	}
    }//// NN contribution;

  if (J2_ON)
    {
      for (k = 0; k < Zn2; k++)
	{  
	  nb = nbors2[i0][k];

	  if (nb >= 0)//(#)
	    {
	      if (disOrder)
		{
		  a1 = impField[i0];
		  a2 = impField[nb];

		  J2Val = J2Mat[a1][a2];
		}
	      else
		{ J2Val = J2; }

	      if (J2ab_ON)//( Lieb lattice )
		{
		  J2Val += ( ( (i0 & 1) ? -1 : 1 ) *
			     ( ( k & 1) ? -1 : 1 ) * J2_delta );
		}
	  
	      J2Vec = J2Val * lambdaVec2;
      
	      for (n = 0; n < 3; n++)
		{     
		  locField[n] += J2Vec[n] * spinField[nb][n];
		}
	    }
	}
    }//// Next-NN contribution;

  if (JX_ON)
    {
      for (k = 0; k < ZnX; k++)
	{  
	  nb = nborsX[i0][k];

	  if (nb >= 0)//(#)
	    {
	      if (disOrder)
		{
		  a1 = impField[i0];
		  a2 = impField[nb];

		  JXVal = JXMat[a1][a2];
		}
	      else
		{ JXVal = JX; }
	  
	      JXVec = JXVal * lambdaVec3;
      
	      for (n = 0; n < 3; n++)
		{     
		  locField[n] += JXVec[n] * spinField[nb][n];
		}
	    }
	}
    }//// Distant neighbors contribution (3rd or 5th);
}

//===========================================
/* Compute the effective local field at the
   input i-site resulting from the spin-field
   in its neighborhood (Ising model version): */

void get_localFieldX(unsigned int i,
		     double **spinField, double &locField)
{
  const unsigned int i0 = i;
  
  unsigned int n, k;

  int nb, a1, a2;

  double J1Val, J2Val, JXVal;
  
  if (extH_ON)
    {
      locField = (- extH); }
  else
    { locField = 0.0; }

  for (k = 0; k < Zn1; k++)
    {  
      nb = nbors1[i0][k];

      if (nb >= 0)//(#)
	{
	  if (disOrder)
	    {
	      a1 = impField[i0];
	      a2 = impField[nb];

	      J1Val = J1Mat[a1][a2];
	    }
	  else
	    { J1Val = J1; }
          
	  locField += J1Val * spinField[nb][0];
	}
    }//// NN contribution;

  if (J2_ON)
    {
      for (k = 0; k < Zn2; k++)
	{  
	  nb = nbors2[i0][k];

	  if (nb >= 0)//(#)
	    {
	      if (disOrder)
		{
		  a1 = impField[i0];
		  a2 = impField[nb];

		  J2Val = J2Mat[a1][a2];
		}
	      else
		{ J2Val = J2; }

	      if (J2ab_ON)//( Lieb lattice )
		{
		  J2Val += ( ( (i0 & 1) ? -1 : 1 ) *
			     ( ( k & 1) ? -1 : 1 ) * J2_delta );
		}
         
	      locField += J2Val * spinField[nb][0];
	    }
	}
    }//// Next-NN contribution;

  if (JX_ON)
    {
      for (k = 0; k < ZnX; k++)
	{  
	  nb = nborsX[i0][k];

	  if (nb >= 0)//(#)
	    {
	      if (disOrder)
		{
		  a1 = impField[i0];
		  a2 = impField[nb];

		  JXVal = JXMat[a1][a2];
		}
	      else
		{ JXVal = JX; }
         
	      locField += JXVal * spinField[nb][0];
	    }
	}
    }//// Distant-neighbors contribution;
}

//====================================
// Set initial configuration of spins:

void set_initialSpinField(int wRank,
			  string &infoStr,
			  double **spinField)
{
  /*--------------------
    Auxiliary variables: */
    
  double drand1, drand2, tht, phi, rnum, spin;

  double vecSz = 0.05; //( 5% of norm )

  Vec3d spinVec, spinVecNew;

  Vec3d spinVec0, randVec;

  int k, n, m, n0, m1, m2, nb;

  m = pltSeq[0]; /* 0 --> xy-plane;
		    2 --> yz-plane; */
  
  /*----------------------------------
    Low-temperature type inplane state
    for triangular/squared geometries: */

  if (IsiModel)
    {
      if (IState == "zpolar")
	{
	  for (k = 0; k < Ns; k++)
	    {
	      spinField[k][0] = 1.0;
	    }
	}
      else//( high_T ) 
	{
	  for (k = 0; k < Ns; k++)
	    {
	      drand1 = dSFMT_getrnum();
	  
	      rnum = 2.0 * drand1 - 1.0;

	      if (rnum >= 0.0)
		{
		  spinField[k][0] = (+ 1.0);
		}
	      else//( up or down spin state )
		{
		  spinField[k][0] = (- 1.0);
		}
	    }
	}
    }
  else//( Heisenberg model )
    {
      if (IState == "loww_T")
	{      
	  if (J2 < J1 / 2)//( staggered order )
	    {
	      if (C3SYM)
		{
		  /*.................
		    120-degrees state */

		  m1 = pltSeq[0]; //[ x|z ]
		  m2 = pltSeq[1]; //[  y  ]  

		  infoStr = "Néel-C3";
	      
		  for (k = 0; k < Ns; k++)
		    {
		      spinField[k][m1] = Q120Phase[k].real();  
		      spinField[k][m2] = Q120Phase[k].imag();
		    }
		}
	      else//( geom == "square" )
		{
		  /*..................
		    Néel up-down state */

		  infoStr = "Néel-C2";
	      
		  for (k = 0; k < Ns; k++)
		    {  
		      spinField[k][m] = DStpPhase[k]; //[ y|z ] 
		    }
		}
	    }
	  else//( J2 > J1 / 2 ---> stripe order )
	    {
	      /*....................................
		Collinear horizontal (even replicas)
		or vertical (odd ones) stripes state */

	      n0 = 2; //[ selector for C2 symmetry ]
	  
	      if (C3SYM){ n0++; }
	  
	      if (wRank % n0 == 0)
		{
		  infoStr = "H-Stripes";
	      
		  for (k = 0; k < Ns; k++)
		    {	  
		      spinField[k][m] = HStpPhase[k];
		    }	      
		}
	      else if (wRank % n0 == 1)
		{
		  infoStr = "V-Stripes";
	      
		  for (k = 0; k < Ns; k++)
		    {	  
		      spinField[k][m] = VStpPhase[k];
		    }
		}
	      else//( C3SYM has to be true )
		{
		  infoStr = "D-Stripes";
		      
		  for (k = 0; k < Ns; k++)
		    {	  
		      spinField[k][m] = DStpPhase[k];
		    }
		}
	    }
	}//// Select low-temp. state;

      /*--------------------------------------
	2 UP + 1 DOWN collinear inplane state: */

      if (geom == "triang" && IState == "2up1dw")
	{
	  infoStr = "2-up + 1-down";
      
	  for (k = 0; k < Ns; k++)
	    {  
	      spinField[k][m] = UpDwPhase[k];
	    }
	}

      /*-------------------------------------
	Horizontal or vertical stripes state: */
  
      if (IState == "hstrip")
	{
	  infoStr = "horz_stripes";
	      
	  for (k = 0; k < Ns; k++)
	    {	  
	      spinField[k][m] = HStpPhase[k];
	    }
	}
      else if (IState == "vstrip")
	{
	  infoStr = "vert_stripes";
	      
	  for (k = 0; k < Ns; k++)
	    {	  
	      spinField[k][m] = VStpPhase[k];
	    }
	}

      /*------------------------------------
	High-temperature (disordered) state: */
 
      if (IState == "high_T") 
	{	
	  /*............
	    Random state */

	  infoStr = "Disordered";
      
	  for (k = 0; k < Ns; k++)
	    {	      
	      drand1 = dSFMT_getrnum();
	      drand2 = dSFMT_getrnum();
	  
	      tht = 2.0 * pi * drand1;
      
	      phi = acos(1.0 - 2.0 * drand2);

	      spinVec[0] = sin(phi) * cos(tht); //[ x ]
	      spinVec[1] = sin(phi) * sin(tht); //[ y ]
	      spinVec[2] = cos(phi);            //[ z ]
	      
	      for (n = 0; n < 3; n++)
		{
		  spinField[k][n] = spinVec[n];
		}
	    }
	}
      else if (IState == "zpolar")
	{
	  /*.............................
	    High energy z-polarized state */

	  infoStr = "Polarized";
      
	  drand1 = dSFMT_getrnum();
	  
	  tht = 0.0; phi = a90;
	      
	  for (k = 0; k < Ns; k++)
	    {        
	      spinField[k][2] = 1.0; //[ z ]  
	    }
	}//// Select high-temp. state;

      /*---------------------------------
	Generate small deviations on the
	perfect init. spin configuration: */

      if (IState != "high_T") 
	{      
	  for (k = 0; k < Ns; k++)
	    {	      
	      drand1 = dSFMT_getrnum();
	      drand2 = dSFMT_getrnum();
	  
	      tht = 2.0 * pi * drand1;
      
	      phi = acos(1.0 - 2.0 * drand2);

	      randVec[0] = sin(phi) * cos(tht); //[ x ]
	      randVec[1] = sin(phi) * sin(tht); //[ y ]
	      randVec[2] = cos(phi);            //[ z ]

	      get_localSpin(k, spinField, spinVec0);

	      spinVec = spinVec0 + vecSz * randVec;

	      spinVecNew = normVec3d(spinVec);
	  
	      set_localSpin(k, spinVecNew, spinField);
	    }
	}
    }///[ Select model-type ]
}

//=============================
// Return one if the input spin
// field has NO erroneous data:

double fieldNorm(double **spinField)
{  
  double normSum = 0.0;

  Vec3d spinVec;
  
  for (int i = 0; i < Ns; i++)
    {
      get_localSpin(i, spinField, spinVec);

      normSum += dotProduct(spinVec, spinVec);
    }

  return (iNs * abs(normSum));
}

//=========================================
// Return one if the input field has a zero
// local field at some site in the lattice:

int check_zeroLocField(double **spinField)
{  
  Vec3d locField;

  double vecNorm;

  int signal = 0;
  
  for (int k = 0; k < Ns; k++)
    {	      
      get_localField(k, spinField, locField);

      vecNorm = dotProduct(locField, locField);

      if (vecNorm == 0.0){ signal = 1; break; }
    }

  return signal;
}

//===================================================
// Perform vector-reflection about input local field:

void reflect_aboutVec(const Vec3d &spinVec,
		      const Vec3d &axisVec,
		      Vec3d &spinVecNew)
{
  unsigned int n;

  double sqLength, fac, rnorm;

  sqLength = dotProduct(axisVec, axisVec); //(1)

  rnorm = 2.0 / sqLength; //(2)

  // 1) Squared length of 'axixVec';
  // 2) Renormalization constant;

  if (sqLength > dbleSmall)
    {
      fac = rnorm * dotProduct(spinVec, axisVec);

      for (n = 0; n < 3; n++)
	{
	  spinVecNew[n] = fac * axisVec[n] - spinVec[n];
	}
    }
  else // Squared length of 'axixVec' (sqLength) is too small;
    {  
      for (n = 0; n < 3; n++){spinVecNew[n] = spinVec[n];}
    }
}

//==================================================
// Generate special matrix (Rodrigues' rot. matrix):

/* The code return the 3 x 3 matrix 'rotMat' that
   transforms the vector 'Avec' onto the fixed
   vector 'Bvec', i.e., rotMat * Avec --> Bvec,
   with * being the matrix product;

   If a third vector 'Cvec' is such that:

   Avec . Cvec = X (dot product)

   Then, the angles that relate these two vectors
   is preserved by the rotation given by 'rotMat':

   Avec . (rotMat * Cvec) = X;

   Below, both input vectors are assumed to be
   normalized (this simplifies calculations)! */

void get_rotA2BMat(const Vec3d &Avec,
		   const Vec3d &Bvec,
		   double Rmx[sdim2])
{
  const int n = sdim;

  const int nn = sdim2;
  
  double sfac, fc1, fc2;

  //--------------------------------------
  // Define & compute the vector products:
  
  Vec3d AxB = crossProduct(Avec, Bvec);
  
  double W[n] = {AxB[0], AxB[1], AxB[2]};

  double dotAB = dotProduct(Avec, Bvec);

  //-------------------------------------
  // Define auxiliary matrices K1 and K2:
  
  double KMat1[nn] =
    {
      + 0.00, - W[2], + W[1],
      + W[2], + 0.00, - W[0],
      - W[1], + W[0], + 0.00};

  double KMat2[nn] =
    {
      0.0, 0.0, 0.0,  // Zero matrix now,
      0.0, 0.0, 0.0,  // KMat1 squared is
      0.0, 0.0, 0.0}; // added to it later;

  //-----------------------------------
  // Define and init. the main matrix:
  // rotmMAT -> rotation matrix A to B
  
  double rotMat[nn] =
    {
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      0.0, 0.0, 1.0}; // Identity matrix;  

  //----------------------------------
  // Compute the K1 and K2 prefactors:
  
  sfac = sqrt(cblas_ddot(n, W, 1, W, 1));

  if (sfac == 0)
    {
      fc1 = 0.0; fc2 = 0.0;
    }
  else
    {
      fc1 = 1.0;
      
      fc2 = (1.0 - dotAB) / (sfac * sfac);
    }

  /*----------------------
    Compute the matrix K2:
    
    K2 = K2 + alpha * (KMat1 * KMat1),

    where, below, alpha is set to 1.0,
    recall that K2 was initialized to zero; */

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
	      n, n, n, 1.0, KMat1, n, KMat1, n, 0.0, KMat2, n);
  
  /*----------------------------
    Compute the rotation matrix:
    
    rotMat = rotMat + fc1 * KMat1 + fc2 * KMat2,

    where the addition (+) operations are performed in 
    2 stages by means of the BLAS subroutine used below; */
  
  cblas_daxpy(nn, fc1, KMat1, 1, rotMat, 1);
  cblas_daxpy(nn, fc2, KMat2, 1, rotMat, 1);

  copy(rotMat, rotMat + nn, Rmx); // Copy: rotMat --> Rmx;
}

//===============================================
// This gives the result of 'get_rotA2BMat' for a
// fixed vector 'Avec' that is taken as (0,0,1):

void get_rotZ2VMat(const Vec3d &V, double Rmx[sdim2])
{
  const int n = sdim;

  const int nn = sdim2;
  
  double perp, sfac;

  //.....................................
  // First, the xy-amplitude is computed:
  
  perp = V[0] * V[0] + V[1] * V[1];

  //.....................................
  // Now the rotation matrix is computed:

  double rotMat[nn] =
    {
      1.0, 0.0, 0.0, // Identity matrix
      0.0, 1.0, 0.0, //    ( 3 x 3 )
      0.0, 0.0, 1.0};

  if (perp > 0) // V not aligned with Z;
    {      
      sfac = (1.0 - V[2]) / perp;

      double x11, x22, x33;

      double x12, x21;

      x11 = 1.0 - sfac * V[0] * V[0];
      x22 = 1.0 - sfac * V[1] * V[1];

      x33 = x11 + x22 - 1.0;

      x12 = V[0] * V[1] * (- sfac);

      x21 = x12; // Minus sign above is important!
      
      double work[n * n] = //(*)
	{
	  + x11 , + x12 ,  V[0],
	  + x21 , + x22 ,  V[1],
	  -V[0] , -V[1] ,  x33 };

      copy(work, work + nn, rotMat);
    }
  /// *) Because the fixed vector is the z-unit vector,
  ///    the rotation matrix assumes a simple form that
  ///    can be expressed directly as shown above;

  copy(rotMat, rotMat + nn, Rmx);
}

//=====================================
// Generate canonical heat-bath sample:

/* The input uniform random number 'rand'
   is mapped to a another number within
   the range [-1,+1] according to the
   heat-bath canonical distribution
   associated with the provided
   local spin field; */

void get_hBathSample(double rand, double beta,
		     const Vec3d &locField, double &chi)
{  
  double H, x0, fc;

  H = sqrt(dotProduct(locField, locField));

  if (isNotZero(H))
    { 
      x0 = 1.0 + rand * (exp(- 2.0 * beta * H) - 1.0);
      
      fc = (- 1.0) / (beta * H);
  
      chi = fc * log(x0) - 1.0;
    }
  else
    { chi = 2.0 * rand - 1.0; }
}

//========================================
// Compute the energy using list of bonds:

void get_energyValue(double **spinField, double &HamE)
{     
  unsigned int i, j, n, n1, n2, a1, a2;

  double ec1, ec2, ec3, J1Val, J2Val, JXVal;

  Vec3d J1Vec, J2Vec, JXVec, iSpin, jSpin;

  HamE = 0.0; //[ Initialize energy value ]
  
  //--------------------
  // Compute the energy:  

  n1 = 0;

  n2 = n1 + Nb1;
    
  for (n = n1; n < n2; n++)
    {
      i = bondList[n].x;
      j = bondList[n].y;

      if (disOrder)
	{
	  a1 = impField[i];
	  a2 = impField[j];
	  
	  J1Val = J1Mat[a1][a2];
	}
      else
	{ J1Val = J1; }

      J1Vec = J1Val * lambdaVec1;

      get_localSpin(i, spinField, iSpin);
      get_localSpin(j, spinField, jSpin);
	
      ec1 = wdotProduct(J1Vec, iSpin, jSpin);
			
      HamE += ec1;
    }

  if (J2_ON)
    {
      n1 = Nb1;

      n2 = n1 + Nb2;
      
      for (n = n1; n < n2; n++)
	{
	  i = bondList[n].x;
	  j = bondList[n].y;

	  if (disOrder)
	    {
	      a1 = impField[i];
	      a2 = impField[j];
	  
	      J2Val = J2Mat[a1][a2];
	    }
	  else
	    { J2Val = J2; }

	  if (J2ab_ON)//( Lieb lattice )
	    {
	      J2Val += ( ( (i & 1) ? -1 : 1 ) *
			 ( (j & 1) ? -1 : 1 ) * J2_delta );
	    }

	  J2Vec = J2Val * lambdaVec2;

	  get_localSpin(i, spinField, iSpin);
	  get_localSpin(j, spinField, jSpin);
	
	  ec2 = wdotProduct(J2Vec, iSpin, jSpin);
			
	  HamE += ec2;
	}
    }

   if (JX_ON)
    {
      n1 = Nb1 + Nb2;

      n2 = n1 + NbX;
      
      for (n = n1; n < n2; n++)
	{
	  i = bondList[n].x;
	  j = bondList[n].y;

	  if (disOrder)
	    {
	      a1 = impField[i];
	      a2 = impField[j];
	  
	      JXVal = JXMat[a1][a2];
	    }
	  else
	    { JXVal = JX; }

	  JXVec = JXVal * lambdaVec3;

	  get_localSpin(i, spinField, iSpin);
	  get_localSpin(j, spinField, jSpin);
	
	  ec3 = wdotProduct(JXVec, iSpin, jSpin);
			
	  HamE += ec3;
	}
    }
   
  if (extH_ON) 
    {  
      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, spinField, iSpin);
	  
	  HamE += (- extH) * iSpin[dMag];
	}
    }
}

//=====================================================
// Compute energy (Hamiltonian) for a given spin-field:

/* The total energy obtained from the Hamiltonian
   evaluated for the input spin-field configura-
   tion is assigned to the variable 'HamE';
     
   In the mark (#), the '0.5' factor compensates
   for the double counting of site bonds; */

void get_energyVal(double **spinField, double &HamE)
{
  const int dble_Ns = (double)Ns;
    
  unsigned int k, n;

  double energy0, energy1;

  Vec3d locField;
  
  HamE = 0.0; //(Hamiltonian energy)
    
  for (k = 0; k < Ns; k++)
    {     
      // Interaction of each spin in the
      // lattice with its local field:
	  
      get_localField(k, spinField, locField);
	
      energy0 = 0.0;
	
      for (n = 0; n < 3; n++)
	{
	  energy0 = energy0
	    + spinField[k][n] * locField[n];
	}

      energy0 = energy0 * 0.5; //(#)

      // Interaction with the 
      // external magnetic field:

      energy1 = 0.0;
	
      if (extH_ON) 
	{        
	  energy1 = (- extH) * spinField[k][dMag];
	}

      // Add total energy to 'HamE':
	
      HamE += energy0 + energy1;
    }
}

//====================================
// Total transvese-axis magnetization:

void get_TrMagnet(double **spinField, double &ZMag) 
{
  const int index = (extH_ON) ? dMag : 2;
  
  ZMag = 0.0;

  for (int k = 0; k < Ns; k++)
    {
      ZMag += spinField[k][index];
    }
	  
  ZMag = ZMag * iNs;
}

//=======================================
// Compute Q-order magnetization squared:

/* Return the absolute staggered magnetization
   for lattices with C2 or C3 symmetry ... */

double absStaggMag(double **spinField) 
{  
  unsigned int i, j, k, n, n0, ifc, stnum;

  vector<Vec3d> subMags = {null3d, null3d, null3d};

  Vec3d spinVec, dotVec, magVec;
  
  if (C3SYM)//( n0 : number of sublattices )
    {
      n0 = 3; ifc = 1; }
  else
    { n0 = 2; ifc = 0; }
  
//--------------------------------------------
// Loop over lattice / compute magnetizations:

  if (geom != "lieb")
    {
      k = 0;
  
      for (i = 0; i < Lsz; i++)
	{
	  for (j = 0; j < Lsz; j++)
	    {
	      stnum = i * Lsz + j;

	      n = (i + j + ifc * k) % n0;
	
	      spinVec = Vec3d(spinField[stnum][0],
			      spinField[stnum][1],
			      spinField[stnum][2]);

	      subMags[n] += spinVec;
	    }

	  k += 1;
	}
    }
  else//( Lieb lattice has a different indexing ) 
    {
      for (k = 0; k < Ns; k++)
	{ 
	  n = k % n0;
	
	  spinVec = Vec3d(spinField[k][0],
			  spinField[k][1],
			  spinField[k][2]);

	  subMags[n] += spinVec;
	}
    }
  
  //----------------------------------------------
  // Calculate & return abs. stagg. magnetization:

  const double norm = double(n0) * iNs;

  const double fc = 1.0 / sqrt(n0);
  
  Vec3d sumVec = null3d;
  
  double sumSqr = 0.0;

  for (n = 0; n < n0; n++)
    {
        const Vec3d magVec = norm * subMags[n];
	
        sumVec += magVec;
	
        sumSqr += dotProduct(magVec, magVec);
    }

  return fc * sqrt(sumSqr - dotProduct(sumVec, sumVec) / double(n0));
}

//=======================================
// Sublattice stripe-order magnetization:

/* Return the absolute stripe magnetization
   for lattices with C2 or C3 symmetry, the
   type of stripe config. must be specified; */

double absStripeMag(double **spinField, string stype) 
{  
  unsigned int i, j, k, n, i1, i2, n1, n2, stnum; 

  vector<Vec3d> subMags = {null3d, null3d};

  Vec3d spinVec, spinVec1, spinVec2;

  Vec2d magVec;

  //------------------------------------------------- 
  // Compute magnetizations for the given input type:

  if (geom != "lieb")
    {
      if (stype == "Horz")//............................
	{ 
	  for   (i = 0; i < Lsz; i++){// (i,j) --> (x,y)    
	    for (j = 0; j < Lsz; j++)
	      {
		stnum = i * Lsz + j;

		spinVec = Vec3d(spinField[stnum][0],
				spinField[stnum][1],
				spinField[stnum][2]);

		n = j % 2; // Varies with y-coordinate;
		    
		subMags[n] += spinVec;
	      }
	  }
	}
      else if (stype == "Vert")//.......................
	{ 
	  for   (i = 0; i < Lsz; i++){// (i,j) --> (x,y)    
	    for (j = 0; j < Lsz; j++)
	      {
		stnum = i * Lsz + j;

		spinVec = Vec3d(spinField[stnum][0],
				spinField[stnum][1],
				spinField[stnum][2]);

		n = i % 2; // Varies with x-coordinate;
		    
		subMags[n] += spinVec;
	      }
	  }
	}
      else if (stype == "Diag")//.......................
	{ 
	  for   (i = 0; i < Lsz; i++){// (i,j) --> (x,y)     
	    for (j = 0; j < Lsz; j++)
	      {	    		    
		stnum = i * Lsz + j;

		spinVec = Vec3d(spinField[stnum][0],
				spinField[stnum][1],
				spinField[stnum][2]);

		n = (i + j) % 2; // Varies at each step;
	    
		subMags[n] += spinVec;
	      }
	  }
	}
    }
  else//( Lieb lattice has a different indexing ) 
    {
      if (stype == "Horz")//............................
	{
	  n = 0;
  
	  for (i = 0; i < Lsz; i++)
	    {
	      for (j = 0; j < Lsz; j++)
		{	    		
		  i1 = n; i2 = n + 1;

		  spinVec1 = Vec3d(spinField[i1][0],
				   spinField[i1][1],
				   spinField[i1][2]);

		  spinVec2 = Vec3d(spinField[i2][0],
				   spinField[i2][1],
				   spinField[i2][2]);

		  subMags[(i + j) & 1] += spinVec1 + spinVec2;

		   n = n + 2;
		}      
	    }
	}
      else if (stype == "Vert")//.......................
	{
	  n = 0;
	  
	  for (i = 0; i < Lsz; i++)
	    {
	      for (j = 0; j < Lsz; j++)
		{
		  i1 = n; i2 = n + 1;
		  
		  spinVec1 = Vec3d(spinField[i1][0],
				   spinField[i1][1],
				   spinField[i1][2]);

		  spinVec2 = Vec3d(spinField[i2][0],
				   spinField[i2][1],
				   spinField[i2][2]);

		  n1 = (i + j) % 2; n2 = 1 - n1;

		  subMags[n1] += spinVec1;
		  subMags[n2] += spinVec2;

		  n = n + 2;
		}
	    }
	 
	}
      else if (stype == "Diag")//....................... 
	{ 
	  for (k = 0; k < Ns; k++)
	    { 
	      spinVec = Vec3d(spinField[k][0],
			      spinField[k][1],
			      spinField[k][2]);

	      subMags[k % 2] += spinVec;
	    }
	}
    }

  //----------------------------------------------
  // Calculate & return abs. stagg. magnetization:

  const double norm = 2.0 * iNs;

  const double fc = 1.0 / sqrt(2.0);
  
  Vec3d sumVec = null3d;
  
  double sumSqr = 0.0;

  for (k = 0; k < 2; k++)
    {
        const Vec3d magVec = norm * subMags[k];
	
        sumVec += magVec;
	
        sumSqr += dotProduct(magVec, magVec);
    }

  return fc * sqrt(sumSqr - 0.5 * dotProduct(sumVec, sumVec));     
}

//======================================
// Ising-like order parameter for square
// plaquette spin stripe configurations: (C2 geometry only)

/* Order parameter of Ising character for the O(3) x Z2
   symmetric stripe phase (J2 > 0.5 * J1) capturing the
   Z2 symmetry breaking induced by order-by-thermal dis-
   order effect which selects either a horizontal stripe 
   configuration or a vertical one. Un-normalized value:

   [ S(i) - S(i + x + y) ] . [ S(i + x) - S(i + y) ]

   For an horizontal or vertical stripe configuration of
   spin, the order parameter value is equal to +1 or -1,
   respectively. When computing the magnetic susceptibi-
   lity, one has to use the absolute value!
   
   DOI: 10.1103/PhysRevLett.91.177202 */

double IsingZ2Parameter(double **spinField) 
{  
  Vec3d Spin0, Spin0xx, Spin0yy, Spin0xy;

  double fc, dprod, magSum, IsiMag;

  int i, nxx, nyy, nxy;

  /*---------------------------------
    Loop over the lattice and measure
    the un-normalized order parameter: */
  
  magSum = 0.0;

  for (i = 0; i < Ns; i++)
    {
      get_localSpin(i, spinField, Spin0);
      
      nxx = nbors1[i][0]; // Left
      nyy = nbors1[i][1]; // Down
      nxy = nbors2[i][0]; // Left + Down

      if (PBC_OFF)
	{
	  if ((nxx < 0) && (nxy < 0) && (nyy >= 0))
	    {
	      nxx = nbors1[i][2]; // Right
	      nxy = nbors2[i][1]; // Right + Down
	    }

	  if ((nyy < 0) && (nxy < 0) && (nxx >= 0))
	    {
	      nyy = nbors1[i][3]; // Up
	      nxy = nbors2[i][3]; // Up + Left
	    }

	  if ((nxx < 0) && (nyy < 0) && (nxy < 0))
	    {
	      nxx = nbors1[i][2]; // Right
	      nyy = nbors1[i][3]; // Up
	      nxy = nbors2[i][2]; // Up + Right
	    }
	}

      get_localSpin(nxx, spinField, Spin0xx);
      get_localSpin(nyy, spinField, Spin0yy);            
      get_localSpin(nxy, spinField, Spin0xy);

      dprod = dotProduct(Spin0 - Spin0xy, Spin0xx - Spin0yy);

      if (isNotZero(dprod))
	{
	  fc =  1.0 / abs(dprod); }
      else
	{ fc = 0.25; }

      magSum += fc * dprod;
    }

  /*---------------------------------
    Calculate final value and return: */
  
  IsiMag = iNs * magSum;

  return IsiMag;
}

//===============================
// Order parameter for triangular
// plaquette spin configurations: (C3 geometry only)

/* Coplanar Y state: 1 spin pinned in the ext. magnetic 
   field direction (given by 'dMag') + 2 spins canting up;

   Collinear state (extH = 3): 2 spins up + 1 spin down;

   Coplanar 2:1 canted state: intermediary configuration
   that smoothly interpolates toward a collinear saturated
   paramagnet at high field (extH = 9);

   -------------------------------------------------------

   The function output gives the magnitude of the complex
   order parameter defined in the article: "Phase diagram
   of the classical Heisenberg antiferromagnet on a trian-
   gular lattice in an applied magnetic field";

   DOI: 10.1103/PhysRevB.84.214418; */

double cplxC3Parameter(double **spinField)
{
  const unsigned int n0 = dMag;
  
  const double fc1 = sqrt(6.0) * iNs;

  const double fc2 = (- 3.0) * sqrt(2.0) * iNs;
  
  unsigned int i, j, k, n, stnum;
  
  double MagPsi, RePsi, ImPsi, spinProjc; 

  double subMags[3] = {0.0, 0.0, 0.0};

  /*------------------------------------------
    Compute magnetizations in the n-direction: */
  
  k = 0;
  
  for   (i = 0; i < Lsz; i++){
    for (j = 0; j < Lsz; j++)
      {
	stnum = i * Lsz + j;

	n = (i + j + k) % 3;
	
	spinProjc = spinField[stnum][n0];

	subMags[n] += spinProjc;
      }

    k += 1;
  }

  /*-------------------------------- 
    Compute complex order parameter: 

    For a perfect spin configuration of
    plaquette arrangement 2-up + 1-down
    as given by 'UpDwPhase', one finds 
    that 'ImPsi' is zero because the z
    magnetization for the sublattices 
    of index n = 0 & n = 1 are equal; */
  
  RePsi = subMags[0] + subMags[1] - 2.0 * subMags[2];

  ImPsi = subMags[1] - subMags[0];

  RePsi = fc1 * RePsi;
  ImPsi = fc2 * ImPsi;

  complex<double> Psi = {RePsi, ImPsi};

  /*--------------------------
    Return magnitude of 'Psi': */
  
  MagPsi = sqrt(real(Psi * conj(Psi)));

  /*--------------------------------------
    Expected results for perfect states...
    |
    | > Collinear state 2-up + 1-down: 8 / sqrt(6);
    |
    | > Coplanar 120° state: sqrt(6);
    |
    | > Coplanar stripe state: 0; */
				     
  return MagPsi;
}

//=======================
// Check lattice spacing:

void check_latticeSpc(double &lspc, int &flag)
{ 
  double x0, y0, dx, dy, rdist;

  lspc = 100.0;
		  
  x0 = rvecList[0][0]; // First site
  y0 = rvecList[0][1]; // position;
		  
  for (int n = 0; n < Zn1; n++)
    {
      int nb = nbors1[0][n];		      
			
      if (nb > 0)
	{
	  dx = rvecList[nb][0] - x0;
	  dy = rvecList[nb][1] - y0;

	  rdist = sqrt(dx * dx + dy * dy);
			  
	  if (rdist < lspc){ lspc = rdist; }
	}
    }
		  
  if ((lspc < 0.95) || (lspc > 1.05))
    {
      flag = 1; }
  else
    { flag = 0; }
}

//======================================
// Remove PBC from all neighbors tables:

void rmPBC_nborsTable(double lspc)
{
  double x0, y0, dx, dy;

  double rdist, maxDist;
  		  
  int k, n, nb;

  //--------------
  // Table: nbors1

  maxDist = 1.5 * lspc;

  for (k = 0; k < Ns; k++)
    {
      x0 = rvecList[k][0];
      y0 = rvecList[k][1];
		  
      for (n = 0; n < Zn1; n++)
	{
	  nb = nbors1[k][n];		      
			
	  if (nb >= 0)
	    {
	      dx = rvecList[nb][0] - x0;
	      dy = rvecList[nb][1] - y0;

	      rdist = sqrt(dx * dx + dy * dy);
			  
	      if (rdist > maxDist)
		{
		  nbors1[k][n] = (- 1);
		}
	    }
	}
    }

  //--------------
  // Table: nbors2

  maxDist = 1.5 * (sqrt(2.0) * lspc);

  for (k = 0; k < Ns; k++)
    {
      x0 = rvecList[k][0];
      y0 = rvecList[k][1];
		  
      for (n = 0; n < Zn2; n++)
	{
	  nb = nbors2[k][n];		      
			
	  if (nb >= 0)
	    {
	      dx = rvecList[nb][0] - x0;
	      dy = rvecList[nb][1] - y0;

	      rdist = sqrt(dx * dx + dy * dy);
			  
	      if (rdist > maxDist)
		{
		  nbors2[k][n] = (- 1);
		}
	    }
	}
    }

  //--------------
  // Table: nborsX

  maxDist = 1.5 * (sqrt(2.0 + sqrt(2.0)) * lspc);

  for (k = 0; k < Ns; k++)
    {
      x0 = rvecList[k][0];
      y0 = rvecList[k][1];
		  
      for (n = 0; n < ZnX; n++)
	{
	  nb = nborsX[k][n];		      
			
	  if (nb >= 0)
	    {
	      dx = rvecList[nb][0] - x0;
	      dy = rvecList[nb][1] - y0;

	      rdist = sqrt(dx * dx + dy * dy);
			  
	      if (rdist > maxDist)
		{
		  nborsX[k][n] = (- 1);
		}
	    }
	}
    }
}

//=====================================================
/* Calculate energy map comparing Néel and Stripe local
   configuration energy across the quasicrystal-lattice: */

void get_qctEnergyDiff(double *EnergyMap)
{
  double StrpEnergy1, StrpEnergy2;
  double StrpEnergy3, StrpEnergy4;

  double StrpEnergy, NeelEnergy;
  
  Vec3d locField1, locField2;
  Vec3d locField3, locField4;

  Vec3d locField0;
 
  //--------------------------
  // Compute energies locally:
    
  for (int i = 0; i < Ns; i++)
    {
      NeelEnergy = 0.0;
      
      StrpEnergy1 = 0.0;
      StrpEnergy2 = 0.0;
      StrpEnergy3 = 0.0;
      StrpEnergy4 = 0.0;
  
      get_localField(i, Neel0Config, locField0);
      
      get_localField(i, Strp1Config, locField1);
      get_localField(i, Strp2Config, locField2);
      get_localField(i, Strp3Config, locField3);
      get_localField(i, Strp4Config, locField4);
		
      for (int n = 0; n < 3; n++)
	{
	  NeelEnergy += Neel0Config[i][n] * locField0[n];
	  
	  StrpEnergy1 += Strp1Config[i][n] * locField1[n];
	  StrpEnergy2 += Strp2Config[i][n] * locField2[n];
	  StrpEnergy3 += Strp3Config[i][n] * locField3[n];
	  StrpEnergy4 += Strp4Config[i][n] * locField4[n];
	}

      StrpEnergy = 0.25 * (StrpEnergy1 + StrpEnergy2 +
			   StrpEnergy3 + StrpEnergy4 );

      NeelEnergy = NeelEnergy * 0.5;     
      StrpEnergy = StrpEnergy * 0.5;

      EnergyMap[i] = 0.5 * (abs(NeelEnergy) - abs(StrpEnergy));
    }  
}

////////////////////////////////////////

void get_qctLocalFMap(double *LocalFMap)
{
  Vec3d locField1, locField2;
  Vec3d locField3, locField4;
    
  for (int i = 0; i < Ns; i++)
    {      
      get_localField(i, Strp1Config, locField1);
      get_localField(i, Strp2Config, locField2);
      get_localField(i, Strp3Config, locField3);
      get_localField(i, Strp4Config, locField4);

      LocalFMap[i] = 0.25 * ( vecAvg3d(locField1) + vecAvg3d(locField1) +
			      vecAvg3d(locField1) + vecAvg3d(locField1) );
    }  
}

//=========================================
// Calculate the Néel frustration parameter | qct: quasi-crystal 
// for nearest neighbors (0 = unfrustrated)

void get_qctNeel_N1fpar(double **sfield, double &Neel_N1fpar)
{  
  Vec3d stSpin, nbSpin;

  Vec3d avgSpin, sumSpin;

  int k, n, nb; double z1fac;
  
  double fpar = 0.0;  
  
  for (k = 0; k < Ns; k++)
    {      
      sumSpin = null3d;
		    
      for (n = 0; n < Zn1; n++)
	{
	  nb = nbors1[k][n];

	  if (nb >= 0)
	    {
	      get_localSpin(nb, sfield, nbSpin);

	      sumSpin += nbSpin;
	    }
	}     
      
      z1fac = 1.0 / max(1, zvalList[k].x);
      
      avgSpin = z1fac * sumSpin;

      get_localSpin(k, sfield, stSpin);

      fpar += 0.5 * (1.0 - dotProduct(stSpin, avgSpin));
    } 

  Neel_N1fpar = iNs * fpar;
}

//============================================
// Calculate the Néel magnetization via projec- | qct: quasi-crystal 
// tion method (reference state: Neel0Config):  | ..................

void get_qctNeelPrjt(double **sfield, double &NeelPrjt)
{
  /*---------------------------------
    Calculate projection of the input
    spin-state onto the Neel state... */

  int k, n, m;
  
  Vec3d kspin; 

  double NMag = 0.0;
      
  for (k = 0; k < Ns; k++)
    {
      Vec3d refSpin =
	{
	  Neel0Config[k][0],
	  Neel0Config[k][1],
	  Neel0Config[k][2]
	};
      
      get_localSpin(k, sfield, kspin);
		    
      NMag += dotProduct(kspin, refSpin);
    }

  NeelPrjt = iNs * NMag;
}

//=============================================
// Calculate the four stripe magnetizations via  | qct: quasi-crystal 
// projection method (ref. states: StrpXConfig): | ..................

void get_qctStrpPrjt(double **sfield, Vec4d &StrpPrjt)
{
  /*---------------------------------
    Calculate projection of the input
    spin-state onto each stripe state */

  int k;
  
  Vec3d kspin;

  double SMag1 = 0.0;
  double SMag2 = 0.0;
  double SMag3 = 0.0;
  double SMag4 = 0.0;
	      
  for (int k = 0; k < Ns; k++)
    {
      Vec3d refSpin1 =
	{
	  Strp1Config[k][0],
	  Strp1Config[k][1],
	  Strp1Config[k][2]
	};

      Vec3d refSpin2 =
	{
	  Strp2Config[k][0],
	  Strp2Config[k][1],
	  Strp2Config[k][2]
	};

      Vec3d refSpin3 =
	{
	  Strp3Config[k][0],
	  Strp3Config[k][1],
	  Strp3Config[k][2]
	};

      Vec3d refSpin4 =
	{
	  Strp4Config[k][0],
	  Strp4Config[k][1],
	  Strp4Config[k][2]
	};
      
      get_localSpin(k, sfield, kspin);
		    
      SMag1 += dotProduct(kspin, refSpin1);
      SMag2 += dotProduct(kspin, refSpin2);
      SMag3 += dotProduct(kspin, refSpin3);
      SMag4 += dotProduct(kspin, refSpin4);
    }

  StrpPrjt[0] = iNs * SMag1;
  StrpPrjt[1] = iNs * SMag2;
  StrpPrjt[2] = iNs * SMag3;
  StrpPrjt[3] = iNs * SMag4;

  /*--------------------------------
    Check for maximum projection and
    round the small others to zero... */

  bool cut = false;
  
  int kmax = 0;
  
  for (k = 0; k < 4; k++)
    {
      if (isZero(abs(StrpPrjt[k]) - 1.0))
	{
	  kmax = k; cut = true;
	}
    }

  if (cut)
    {      
      for (k = 0; k < 4; k++)
	{
	  if (k != kmax)
	    {
	      StrpPrjt[k] = 0.0;
	    }
	}
    }
}
  
//==========================================
// Calculate the vector order parameter (2D) | qct: quasi-crystal
// for the Z4-stripe magnetization (requires | ..................
// a clock-ordered nearest site lists):
/*
  ...................
  Subroutine outputs:

  StrpMag = |SMagVec| : [ -1 , +1 ] (order parameter);
  .................................................... */

void get_qctStripeMag(double **sfield, double &StrpMag)
{  
  unsigned int i, j, k;

  double rx, ry, rnorm, rfac;
  
  Vec3d spinVec, ispin, vec1, vec2;

  Vec3d xvec1, yvec1, xvec2, yvec2;
  
  /*-----------------------------------------
    Iterate over all lattice sites to compute
    the order parameter & feed the histogram */    

  double SMag1 = 0.0;
  double SMag2 = 0.0;
  
  for (i = 0; i < Ns; i++)
    {      
      int znum1 = zvalList[i].x;

      double fcz = (1.0 / znum1);

      /*...........................
	Build spin-set of 8 vectors */

      vector<Vec3d> spinSet;

      for (j = 0; j < Zn1; j++)
	{
	  int nb = nbors1[i][j];
        
	  if (nb >= 0)
	    {	      
	      k = nb; //( nb is unsigned )

	      get_localSpin(k, sfield, spinVec);

	      spinSet.push_back(spinVec);
	    }
	  else
	    { spinSet.push_back(null3d); }
	}

      /*.....................................
	Calculate 2-component order parameter */

      xvec1 = ( spinSet[0] + spinSet[1] +
		spinSet[4] + spinSet[5] );
      
      yvec1 = ( spinSet[2] + spinSet[3] +
		spinSet[6] + spinSet[7] );

      xvec2 = ( spinSet[1] + spinSet[2] +
		spinSet[5] + spinSet[6] );

      yvec2 = ( spinSet[3] + spinSet[4] +
		spinSet[7] + spinSet[0] );

      vec1 = (xvec1 - yvec1);
      vec2 = (xvec2 - yvec2);      
          
      get_localSpin(i, sfield, ispin);

      rx = fcz * dotProduct(ispin, vec1);
      ry = fcz * dotProduct(ispin, vec2);

      rnorm = sqrt(pow(rx, 2) + pow(ry, 2));

      rfac = (rnorm > 1.0) ? 1.0 / rnorm : 1.0;
      	  
      SMag1 += rfac * rx;
      SMag2 += rfac * ry;
    }

  /*-------------------------------------
    Compute final value (lattice average) */

  Vec2d SMagVec = { iNs * SMag1 , iNs * SMag2 }; 

  StrpMag = sqrt(dotProduct2d(SMagVec, SMagVec));
}

//===========================
// Calculate the vector order | qct: quasi-crystal
// parameter (2D) on-site:    | ..................

void get_onSite_qctSVec(unsigned int i,
			double **sfield, Vec2d &SVec)
{
  int znum1 = zvalList[i].x;

  double fcz = (1.0 / znum1);
      
  /*--------------------------------------
    Compute the order parameter components */

  unsigned int j, k;
  
  double rx, ry, rnorm, rfac;
  
  Vec3d spinVec, ispin, vec1, vec2;

  Vec3d xvec1, yvec1, xvec2, yvec2;

  vector<Vec3d> spinSet;
             
  for (j = 0; j < Zn1; j++)
    {
      int nb = nbors1[i][j];
        
      if (nb >= 0)
	{	      
	  k = nb; //( nb is unsigned )

	  get_localSpin(k, sfield, spinVec);

	  spinSet.push_back(spinVec);
	}
      else
	{ spinSet.push_back(null3d); }
    }
   
  xvec1 = ( spinSet[0] + spinSet[1] +
	    spinSet[4] + spinSet[5] );
      
  yvec1 = ( spinSet[2] + spinSet[3] +
	    spinSet[6] + spinSet[7] );

  xvec2 = ( spinSet[1] + spinSet[2] +
	    spinSet[5] + spinSet[6] );

  yvec2 = ( spinSet[3] + spinSet[4] +
	    spinSet[7] + spinSet[0] );

  vec1 = (xvec1 - yvec1);
  vec2 = (xvec2 - yvec2);
      
  get_localSpin(i, sfield, ispin);

  rx = fcz * dotProduct(ispin, vec1);
  ry = fcz * dotProduct(ispin, vec2);

  rnorm = sqrt(pow(rx, 2) + pow(ry, 2));

  rfac = (rnorm > 1.0) ? 1.0 / rnorm : 1.0;

  SVec[0] = rfac * rx;
  SVec[1] = rfac * ry;
}

//=================================================
/* Calculate the 2D-vector order parameter map for
   the Z4-stripe magnetization at each lattice site:

   qct: quasi-crystal (short-form notation)
   
   Output: double-pointer (parameter-map array);
   
   Requires: clock-ordered nearest site lists);
   ............................................. */

void get_qctSMagField(double **spinField, Vec2d *OrderMap)
{
  unsigned int i, j, k;

  double rx, ry, rnorm, rfac;
      
  Vec3d spinVec, ispin, vec1, vec2;

  Vec3d xvec1, yvec1, xvec2, yvec2;  

  /*--------------------------------------
    Iterate over lattice sites and nearest 
    neighbors & compute the order parameter */
  
  for (i = 0; i < Ns; i++)
    {            
      int znum1 = zvalList[i].x;

      double fcz = (1.0 / znum1);

      vector<Vec3d> spinSet;
           
      for (j = 0; j < Zn1; j++)
	{
	  int nb = nbors1[i][j];
        
	  if (nb >= 0)
	    {	      
	      k = nb; //( nb is unsigned )

	      get_localSpin(k, spinField, spinVec);

	      spinSet.push_back(spinVec);
	    }
	  else
	    { spinSet.push_back(null3d); }
	}
   
      xvec1 = ( spinSet[0] + spinSet[1] +
		spinSet[4] + spinSet[5] );
      
      yvec1 = ( spinSet[2] + spinSet[3] +
		spinSet[6] + spinSet[7] );

      xvec2 = ( spinSet[1] + spinSet[2] +
		spinSet[5] + spinSet[6] );

      yvec2 = ( spinSet[3] + spinSet[4] +
		spinSet[7] + spinSet[0] );

      vec1 = (xvec1 - yvec1);
      vec2 = (xvec2 - yvec2);
      
      get_localSpin(i, spinField, ispin);

      rx = fcz * dotProduct(ispin, vec1);
      ry = fcz * dotProduct(ispin, vec2);

      rnorm = sqrt(pow(rx, 2) + pow(ry, 2));

      rfac = (rnorm > 1.0) ? 1.0 / rnorm : 1.0;
      
      OrderMap[i][0] = rfac * rx;
      OrderMap[i][1] = rfac * ry; 
    }
}

//=============================================
// Compute the linear & the quadratic spin-spin
// correlations for each bond in the lattice:

/* Spin correlations for spin i and its
   Zn1 or Zn1 + Zn2 bonds with the spins j:

   Spin-Spin-CF1 =  spin(i) . spin(j);
     
   Spin-Spin-CF2 = (spin(i) . spin(j))^2; */

void get_spinCorr(double **spinField,
		  double  *spinCorr1,
		  double  *spinCorr2)
{
  unsigned int i, j, k, n, n1, n2;

  Vec3d iSpin, jSpin;

  double dprod;

  //--------------------------------
  // Initialize 'spinCorrX' to zero:
  
  for (n = 0; n < Nb; n++)
    {
      spinCorr1[n] = 0.0;
      spinCorr2[n] = 0.0;
    }

  //-----------------------------------------
  // Stage 1: nearest-neighbors correlations:

  n1 = 0;

  n2 = n1 + Nb1;
    
  for (n = n1; n < n2; n++)
    {
      i = bondList[n].x;
      j = bondList[n].y;
      
      get_localSpin(i, spinField, iSpin);
      get_localSpin(j, spinField, jSpin);      

      dprod = dotProduct(iSpin, jSpin);
	  
      spinCorr1[n] += dprod;
      spinCorr2[n] += dprod * dprod;
    }

  //----------------------------------------------
  // Stage 2: next-nearest-neighbors correlations:
  
  if (J2_ON)
    {
      n1 = Nb1;

      n2 = n1 + Nb2;
      
      for (n = n1; n < n2; n++)
	{
	  i = bondList[n].x;
	  j = bondList[n].y;
      
	  get_localSpin(i, spinField, iSpin);
	  get_localSpin(j, spinField, jSpin);      

	  dprod = dotProduct(iSpin, jSpin);
	  
	  spinCorr1[n] += dprod;
	  spinCorr2[n] += dprod * dprod;
	}
    }

  //-----------------------------------------
  // Stage 3: distant-neighbors correlations:
  
  if (JX_ON)
    {
      n1 = Nb1 + Nb2;

      n2 = n1 + NbX;
      
      for (n = n1; n < n2; n++)
	{
	  i = bondList[n].x;
	  j = bondList[n].y;
      
	  get_localSpin(i, spinField, iSpin);
	  get_localSpin(j, spinField, jSpin);      

	  dprod = dotProduct(iSpin, jSpin);
	  
	  spinCorr1[n] += dprod;
	  spinCorr2[n] += dprod * dprod;
	}
    }
}

//============================================= | Only working for square
/* Calculate the spin stiffness (rhoS quantity) | and triangular lattices;
   for the triangular lattice Heisenberg model:

   Info: the lattice C-factors are defined for
   ..... the case of a triang. lattice only... */

void get_spinStiff(double **spinField,
		   Vec3d &rhoS1vec,
		   Vec3d &rhoS2vec)
{  
  const int nvec[4] = { 0, Nb1, Nb1 + Nb2, Nb1 + Nb2 + NbX };

  const double JcList[3] = { J1, J2, JX };

  const double idMat[3][3] = { {1.0, 0.0, 0.0} ,
			       {0.0, 1.0, 0.0} ,
			       {0.0, 0.0, 1.0} };
    
  unsigned int i, j, k, n, n1, n2, a1, a2, s1, s2;
    
  double rhoS1_C0, rhoS1_C1, rhoS1_C2;
  
  double rhoS2_C0, rhoS2_C1, rhoS2_C2;

  double C0, C1, C2, dprod, vprod;

  double x1, y1, x2, y2, dx, dy, Jc;
   
  Vec3d iSpin, jSpin;
      
  //-----------------------
  // Initialize quantities:

  rhoS1_C0 = 0.0; rhoS2_C0 = 0.0;
  rhoS1_C1 = 0.0; rhoS2_C1 = 0.0;
  rhoS1_C2 = 0.0; rhoS2_C2 = 0.0;
  
  //------------------------------------
  // Calculations using the bond-method:
    
  for (k = 0; k < 3; k++)
    {
      n1 = nvec[k + 0];
      n2 = nvec[k + 1];
	      
      for (n = n1; n < n2; n++)
	{
	  /* Get bond sites & lattice vectors */  
      
	  s1 = bondList[n].x;
	  s2 = bondList[n].y;
 
	  x1 = rvecList[s1][0];
	  y1 = rvecList[s1][1];

	  x2 = rvecList[s2][0];
	  y2 = rvecList[s2][1];

	  dx = x1 - x2;
	  dy = y1 - y2;

	  /* Get lattice C-factors */

	  C0 = ( avec1[1] * dy + avec1[0] * dx );
	  
	  C1 = ( avec2[1] * dy + avec2[0] * dx );
	  C2 = ( avec2[1] * dy - avec2[0] * dx );

	  /* Get spin vectors & compute products */
      
	  get_localSpin(s1, spinField, iSpin);
	  get_localSpin(s2, spinField, jSpin);	 

	  if (disOrder)
	    {
	      a1 = impField[s1];
	      a2 = impField[s2];
	  
	      Jc = ( idMat[k][0] * J1Mat[a1][a2] +
		     idMat[k][1] * J2Mat[a1][a2] +
		     idMat[k][2] * JXMat[a1][a2] );
	    }
	  else
	    { Jc = JcList[k]; } 
  
	  dprod = Jc * ( iSpin[0] * jSpin[0] + iSpin[1] * jSpin[1] );
	  vprod = Jc * ( iSpin[0] * jSpin[1] - iSpin[1] * jSpin[0] );

	  /* Compute 1st-type terms (rhoS1) */      
	  
	  rhoS1_C0 += pow(C0, 2) * dprod;
	  rhoS1_C1 += pow(C1, 2) * dprod;
	  rhoS1_C2 += pow(C2, 2) * dprod;

	  /* Compute 2nd-type terms (rhoS2) */
		    
	  rhoS2_C0 += C0 * vprod;
	  rhoS2_C1 += C1 * vprod;
	  rhoS2_C2 += C2 * vprod;
	}
    }

  //-------------------------------------
  // Assign results to the Vec3d outputs:
  
  rhoS1vec[0] = rhoS1_C0;
  rhoS1vec[1] = rhoS1_C1;
  rhoS1vec[2] = rhoS1_C2;

  rhoS2vec[0] = pow(rhoS2_C0, 2);
  rhoS2vec[1] = pow(rhoS2_C1, 2);
  rhoS2vec[2] = pow(rhoS2_C2, 2);
}

///////////////////////////////////////////

void get_spinStiff_TEST(double **spinField,
			Vec3d &rhoS1vec,
			Vec3d &rhoS2vec)
{  
  int i, j, k, n, nb;

  int n1, n2, a1, a2, s1, s2;
    
  double rhoS1_C0, rhoS1_C1, rhoS1_C2;
  double rhoS2_C0, rhoS2_C1, rhoS2_C2;

  double C0, C1, C2, dprod, vprod;

  double x1, y1, x2, y2, dx, dy;
   
  Vec3d kSpin, nbSpin;
      
  //-----------------------
  // Initialize quantities:

  rhoS1_C0 = 0.0; rhoS2_C0 = 0.0;
  rhoS1_C1 = 0.0; rhoS2_C1 = 0.0;
  rhoS1_C2 = 0.0; rhoS2_C2 = 0.0;
  
  //------------------------------------
  // Calculations using the site-method:

  for (k = 0; k < Ns; k++)
    {
      x1 = rvecList[k][0];
      y1 = rvecList[k][1];

      get_localSpin(k, spinField, kSpin);
		  
      for (n = 0; n < Zn1; n++)
	{
	  nb = nbors1[k][n];

	  x2 = rvecList[nb][0];
	  y2 = rvecList[nb][1];

	  get_localSpin(nb, spinField, nbSpin);
  
	  dprod = J1 * ( kSpin[0] * nbSpin[0] + kSpin[1] * nbSpin[1] );
	  vprod = J1 * ( kSpin[0] * nbSpin[1] - kSpin[1] * nbSpin[0] );

	  dx = x1 - x2;
	  dy = y1 - y2;
			
	  C0 = ( avec1[1] * dy + avec1[0] * dx );
	  
	  C1 = ( avec2[1] * dy + avec2[0] * dx );
	  C2 = ( avec2[1] * dy - avec2[0] * dx );
	  
	  rhoS1_C0 += pow(C0, 2) * dprod;
	  rhoS1_C1 += pow(C1, 2) * dprod;
	  rhoS1_C2 += pow(C2, 2) * dprod;
		    
	  rhoS2_C0 += C0 * vprod;
	  rhoS2_C1 += C1 * vprod;
	  rhoS2_C2 += C2 * vprod;
	}
    }

  //-------------------------------------
  // Assign results to the Vec3d outputs:
  
  rhoS1vec[0] = 0.5 * rhoS1_C0;
  rhoS1vec[1] = 0.5 * rhoS1_C1;
  rhoS1vec[2] = 0.5 * rhoS1_C2;

  rhoS2vec[0] = pow(0.5 * rhoS2_C0, 2);
  rhoS2vec[1] = pow(0.5 * rhoS2_C1, 2);
  rhoS2vec[2] = pow(0.5 * rhoS2_C2, 2);
}

//==============================
// Transverse spin fluctuations:
// (Z-axis magnetization squared)

void get_TrMagSquared(double **spinField, double &ZMag2) 
{
  const int index = (extH_ON) ? dMag : 2;
  
  double ZMag1 = 0.0;

  for (int k = 0; k < Ns; k++)
    {
      ZMag1 += spinField[k][index];
    }
	  
  ZMag2 = pow(ZMag1, 2) / Ns2;
}
  
//===============================
/* Q-order squared magnetization:
   
  ............
  Information:
  
  Square geometry: Q --> 90° (staggered/C2 state);
  
  Triangular-type geometry: Q --> 120° (C3 state);

  #) Complex 3d vectors objects (C3SYM = true);
  ................................................*/

void get_QMagSquared(double **spinField, double &QMag2) 
{   
  unsigned int k; Vec3d spinVec;

  if (C3SYM)
    {
      //--------------------------
      // Declare needed variables:
      
      double cs, sn; 
      
      cplxVec MagVec1, MagVec2; //(#)

      complex<double> phaseFactor1, phaseFactor2;
  
      //--------------------------------------------
      // Initialize Q-ordered magnetization vectors:
  
      MagVec1 = zeroVec; // Q+
      MagVec2 = zeroVec; // Q-
  
      //--------------
      // Lattice loop: (see 'get_C3spinState')

      /* ..................
	 Procedure details:
     
	 1) Get spin for site with label 'k';

	 2) Get real & imaginary parts of the complex
	 ** phase factor associated with 120° order,
	 ** then, define the phase factor both orde-
	 ** ring wavevectors: Q+ and Q-;

	 3) Multiply the spin by the phase factors
	 ** and accumulate the results iteratively
	 ** in the complex vectors 'MagVec1(2)'; */
    
      for (k = 0; k < Ns; k++)
	{
	  /* Get spin & phase factor for site k */
      
	  spinVec = Vec3d(spinField[k][0],
			  spinField[k][1],
			  spinField[k][2]);

	  cs = Q120Phase[k].real();
	  sn = Q120Phase[k].imag();

	  phaseFactor1 = complex<double>(cs, + sn);
	  phaseFactor2 = complex<double>(cs, - sn); 

	  /* Q+ contribution: Q = (+ Qval, 0) */
      
	  MagVec1.x += phaseFactor1 * spinVec[0];
	  MagVec1.y += phaseFactor1 * spinVec[1];
	  MagVec1.z += phaseFactor1 * spinVec[2];

	  /* Q- contribution: Q = (- Qval, 0) */
      
	  MagVec2.x += phaseFactor2 * spinVec[0];
	  MagVec2.y += phaseFactor2 * spinVec[1];
	  MagVec2.z += phaseFactor2 * spinVec[2];
	}

      //-----------------------------------------
      // Compute the total squared magnetization:
  
      QMag2 = 0.0;
  
      QMag2 += real(MagVec1.x * conj(MagVec1.x) +
		    MagVec1.y * conj(MagVec1.y) +
		    MagVec1.z * conj(MagVec1.z));

      QMag2 += real(MagVec2.x * conj(MagVec2.x) +
		    MagVec2.y * conj(MagVec2.y) +
		    MagVec2.z * conj(MagVec2.z));
    }
  else/* ****************************************
	 Square lattice geometry (staggered mag.) */
    {
      Vec3d Stagg_MagVec = null3d;
      
      for (k = 0; k < Ns; k++)
	{
	  spinVec = Vec3d(spinField[k][0],
			  spinField[k][1],
			  spinField[k][2]);

	  Stagg_MagVec += DStpPhase[k] * spinVec;
	}

      QMag2 = dotProduct(Stagg_MagVec, Stagg_MagVec);
    }
  
  QMag2 = QMag2 / Ns2;
}

//=======================================================
// Stripe-order (180° : C2 state) squared magnetizations:

void get_SMagSquared(double **spinField, double &SMag2) 
{
  unsigned int k;

  Vec3d spinVec, HStp_MVec, VStp_MVec, DStp_MVec;

  //----------------------------------------------
  // Initialize stripe-type magnetization vectors:
  
  HStp_MVec = null3d; // Horizontal;
  VStp_MVec = null3d; // Vertical;
  DStp_MVec = null3d; // Diagonal;
  
  //--------------
  // Lattice loop:
    
  for (k = 0; k < Ns; k++)
    {
      /* Get spin for site k */
      
      spinVec = Vec3d(spinField[k][0],
		      spinField[k][1],
		      spinField[k][2]);
	    
      /* Horizontal stripes contribution */
      
      HStp_MVec += HStpPhase[k] * spinVec;
      
      /* Vertical stripes contribution */
                        
      VStp_MVec += VStpPhase[k] * spinVec;
      
      /* Diagonal stripes contribution */
            
      DStp_MVec += DStpPhase[k] * spinVec;
    }

  //---------------------------------
  // Final calculations (set output):

  if (!C3SYM){ DStp_MVec = null3d; }

  SMag2 = (dotProduct(HStp_MVec, HStp_MVec) +
	   dotProduct(VStp_MVec, VStp_MVec) +
	   dotProduct(DStp_MVec, DStp_MVec)) / Ns2;
}

//===============================================
// Returns a string with the basic magnetic order
// parameters values for the input spin-field ...

string magnetInfoString(double TVal, double **spinField)
{ 
  string infoStr = ""; //( void )

  if (qcrystal)
    {
      double NeelMag, StrpMag; // Neel & Stripe magnetization;

      Vec4d StrpPrjt; // Four-stripes state projection values;
           
      get_qctNeelPrjt(spinField, NeelMag);

      get_qctStrpPrjt(spinField, StrpPrjt);

      StrpMag = sqrt(dotProduct4d(StrpPrjt, StrpPrjt));
      
      infoStr += "NM=" + fmtDbleFix(NeelMag, 2, 4) + X2;
      infoStr += "SM=" + fmtDbleFix(StrpMag, 2, 4) + X2;

      infoStr += "SVec=(";

      infoStr += fmtDbleFix(StrpPrjt[0], 2, 4) + ", ";
      infoStr += fmtDbleFix(StrpPrjt[1], 2, 4) + ", ";
      infoStr += fmtDbleFix(StrpPrjt[2], 2, 4) + ", ";
      infoStr += fmtDbleFix(StrpPrjt[3], 2, 4) + ") ";

      infoStr += " ";
    }
  else//( normal / Bravais lattices )
    {		
      double FMag, QMag, SMgH, SMgV, SMgD, IMag;

      Vec3d spinVec, sumVec = null3d;
	  
      for (int n = 0; n < Ns; n++)
	{
	  get_localSpin(n, spinField, spinVec);
	      
	  sumVec += spinVec;
	}

      FMag = iNs * sqrt(dotProduct(sumVec, sumVec));

      infoStr += "FM=" + fmtDbleFix(FMag, 2, 4) + X2;
  
      QMag = absStaggMag(spinField);

      infoStr += "QM=" + fmtDbleFix(QMag, 2, 4) + X2;
  
      SMgH = absStripeMag(spinField, "Horz");
      SMgV = absStripeMag(spinField, "Vert");

      infoStr += "HS=" + fmtDbleFix(SMgH, 2, 4) + X2;
      infoStr += "VS=" + fmtDbleFix(SMgV, 2, 4) + X2;

      if (C3SYM)
	{      
	  SMgD = absStripeMag(spinField, "Diag");
      
	  infoStr += "DS=" + fmtDbleFix(SMgD, 2, 4);
	}
      else//( square geometry )
	{
	  IMag = IsingZ2Parameter(spinField);
	
	  infoStr += "IM=" + fmtDbleFix(IMag, 2, 4) + X2;
	}
    }

  if (ANNL_ON)
    {
      infoStr += "T=" + fmtDbleFix(TVal, 3, 6);
    }

  return infoStr;
}

//========================================
// Test procedure for Intel-MKL DFT usage:

int test_iMKL_DFT2d(double **spinField,
		    complex<double> *Sqx,
		    complex<double> *Sqy,
		    complex<double> *Sqz)
{  
  const int N = Lsz; // Inputs numbers of 
  const int M = Lsz; // rows and columns;

  /*---------------------------------
    Transfer field real-values to the
    two-dimensional complex pointers */
 
  complex<double> SxMat[N][M]; 
  complex<double> SyMat[N][M];
  complex<double> SzMat[N][M];

  double fx0, fy0, fz0;

  int i, j, k;
  
  for (i = 0; i < N; i++)
    {
      for (j = 0; j < M; j++)
	{    
	  k = i * N + j;
	  
	  fx0 = spinField[k][0];
	  fy0 = spinField[k][1];
	  fz0 = spinField[k][2];
      
	  SxMat[i][j] = complex<double>(fx0, 0.0);
      	  SyMat[i][j] = complex<double>(fy0, 0.0);
	  SzMat[i][j] = complex<double>(fz0, 0.0);
	}
    }

  /*------------------------------------
    Create a descriptor for the FFT plan */
    
  DFTI_DESCRIPTOR_HANDLE handle = NULL;
    
  MKL_LONG stat, dims[2] = {N, M};

  /*-----------------------
    Initialize the FFT plan */
    
  stat = DftiCreateDescriptor(&handle,
			      DFTI_DOUBLE,
			      DFTI_COMPLEX, 2, dims);
  if (stat != 0)
    {
      cerr << "Error! (0)\n\n";

      return stat;
    }

  /*-------------------
    Commit the FFT plan */
    
  stat = DftiCommitDescriptor(handle);
    
  if (stat != 0)
    {
      cerr << "Error! (1)\n\n";
	
      DftiFreeDescriptor(&handle);

      return stat;
    }
  
  /*--------------------------
    Perform the 2D forward DFT
    and free the descriptor... */

  MKL_LONG xStatus, yStatus, zStatus;
  
  xStatus = DftiComputeForward(handle, SxMat);
  yStatus = DftiComputeForward(handle, SyMat);
  zStatus = DftiComputeForward(handle, SzMat);

  stat = xStatus + yStatus + zStatus;
    
  if (stat != 0)
    {
      cerr << "Error! (2)\n\n";
	
      DftiFreeDescriptor(&handle);

      return stat;
    }

  for (i = 0; i < N; i++)
    {
      for (j = 0; j < M; j++)
	{    
	  k = i * N + j;
	  
	  Sqx[k] = SxMat[i][j];
	  Sqy[k] = SyMat[i][j];
	  Sqz[k] = SzMat[i][j];
	}
    }
    
  DftiFreeDescriptor(&handle);

  return stat;
}

//=================================
// FFTW3D procedures: study & tests

void fftw3D_WavePacketTest(int N, int M,
			   double Kx0,
			   double Ky0,
			   double Kz0)
{   
  int Sz = N * N * M; // Input size;

  int n0 = N / 2;

  int m0 = M / 2;
  
  double ds = 1.0 / N;

  double dt = 1.0 / M;
    
  double qx0 = pi2 / N;

  double wf0 = pi2 / M;

  int i, j, k, n, m1, m2, m3;

  double x, y, z, w, w0;

  double qx, qy, wf, fc, r0;

  double Kx, Ky, Kz, spcPower;    
    
  complex<double> zc;

  ofstream out1, out2, out3;
       
  //------------------------------
  // Create q-Wavevector pointers:

  int *ivec, *jvec;

  double *qxVec, *qyVec;
  
  ivec = new int[N];
  jvec = new int[N];
      
  qxVec = new double[N];
  qyVec = new double[N];
  
  for (i = 0; i < N; i++)
    {
      n = 2 * (i / n0);

      k = i - n0 * (n - 1);
	
      ivec[i] = k;
	
      qxVec[i] = qx0 * (k - n0);
    }

  copy(ivec, ivec + N, jvec);
    
  copy(qxVec, qxVec + N, qyVec);

  //---------------------------
  // Create frequency pointers:

  int *kvec;

  double *wfVec;
  
  kvec = new int[M];
      
  wfVec = new double[M];
  
  for (i = 0; i < M; i++)
    {
      n = 2 * (i / m0);

      k = i - m0 * (n - 1);
	
      kvec[i] = k;
	
      wfVec[i] = wf0 * (k - m0);
    }

  //---------------------
  // Create FFTW objetcs:
    
  fftw_complex *inpData, *outData;

  fftw_plan fwPlan, bwPlan;
    
  inpData = new fftw_complex[Sz];
  outData = new fftw_complex[Sz];

  fwPlan = fftw_plan_dft_3d(N, N, M, inpData, outData,
			    FFTW_FORWARD, FFTW_ESTIMATE);

  bwPlan = fftw_plan_dft_3d(N, N, M, outData, inpData,
			    FFTW_BACKWARD, FFTW_ESTIMATE);

  //--------------------------------------
  // Define the input data & output files:
  
  Kx = Kx0;
  Ky = Ky0;
  Kz = Kz0;

  fc = 1.0 / Sz;
      
  for     (k = 0; k < M; k++){
    for   (j = 0; j < N; j++){
      for (i = 0; i < N; i++)
	{
	  x = i - N / 2; // Indices mapped
	  y = j - N / 2; // to coordinates
	  z = k - M / 2; // ...

	  w0 = sin(Kx * x + Ky * y + Kz * z);

	  r0 = pow(x, 2) + pow(y, 2) + pow(z, 2);
	  
	  w = w0 * exp(- fc * r0);

	  n = i + (j + k * N) * N;
	  
	  inpData[n][0] = w;
	  inpData[n][1] = 0.0;
	}
    }
  }
         
  out1.open(outDir1 + "fft3d_qOut.dat");
  out2.open(outDir1 + "fft3d_wOut.dat");
  
  out3.open(outDir1 + "fft3d_qInv.dat");
  
  //--------------------------------
  // Perform forward FFT and record:
    
  fftw_execute(fwPlan);

  k = m0;

  m3 = kvec[k];
  
  wf = wfVec[m3];

  for (i = 0; i < N; i++)
    {
      m1 = ivec[i]; qx = qxVec[m1];
	
      for (j = 0; j < N; j++)
	{
	  m2 = jvec[j]; qy = qyVec[m2];
	    
	  n = m1 + (m2 + m3 * N) * N;

	  zc = complex<double>(outData[n][0],
			       outData[n][1]);
	  
	  spcPower = sqrt(real(zc * conj(zc)));

	  if (spcPower < dbleSmall)
	    {
	      spcPower = dbleSmall;}

	  out1 << qx << X4
	       << qy << X4
	       << log(spcPower) << endl;
	}
      
      out1 << endl;
    }

  j = n0;

  m2 = jvec[j];
  
  qy = wfVec[m2];

  for (i = 0; i < N; i++)
    {
      m1 = ivec[i]; qx = qxVec[m1];
	
      for (k = 0; k < M; k++)
	{
	  m3 = kvec[k]; wf = wfVec[m3];
	    
	  n = m1 + (m2 + m3 * N) * N;

	  zc = complex<double>(outData[n][0],
			       outData[n][1]);
	  
	  spcPower = sqrt(real(zc * conj(zc)));

	  if (spcPower < dbleSmall)
	    {
	      spcPower = dbleSmall;}

	  out2 << qx << X4
	       << wf << X4
	       << log(spcPower) << endl;
	}
      
      out2 << endl;
    }

  //---------------------------------
  // Perform backward FFT and record:
           
  fftw_execute(bwPlan);

  fc = 1.0 / Sz; // Normalization factor;
	
  for     (i = 0; i < N; i++){
    for   (j = 0; j < N; j++){
      for (k = 0; k < M; k++)
	{
	  x = i * ds;
	  y = j * ds;
	  z = k * dt;
	  
	  k = i + (j + k * N) * N;

	  w = fc * inpData[k][0];
	    
	  out3 << x << X4
	       << y << X4
	       << z << X4
	       << w << endl;
	}
      
      out3 << endl;
    }
    
    out3 << endl;
  }
    
  //---------------------------
  // Close files & free memory:
    
  out1.close();
  out2.close();
  out3.close();

  fftw_destroy_plan(fwPlan);
  fftw_destroy_plan(bwPlan);

  delete[] ivec;
  delete[] jvec;
  delete[] kvec;

  delete[] qxVec;
  delete[] qyVec;
  delete[] wfVec;
    
  delete[] inpData;
  delete[] outData;
}

//=================================
// FFTW2D procedures: study & tests

/*..............................
  Notes about the FFT procedure:

  FFT index: I = [0 : N/2-1 , -N/2 : -1]
	    
  FFT frequency vector: W = w0 * I

  Fundamental angular freq.: w0 = 2 * pi / N;
       	
  In this subroutiune, we consider a squared form
  N x N complex function as input, so that the 2D
  Fourier transform is performed on the 2D spacial
  domain given by: [-pi, pi] x [-pi, pi]. We shall
  interpret the frequencies as a components of wa-
  vector. The later is not ordered as indicated 
  above, the integer vectors are used to order
  the values from -pi to pi; 

  For an input signal represented by the data set
  {X(j)} containing N values (j = 0, 1, 2, 3, ...
  N - 2, N - 1), the forward FFT computes the set
  {Y(k)} of Fourier coefficients associated with
  the wavevector:

  q(k) = 2 * Pi * k / N = k * q0;

  q0 = 2 * pi / N, (fundamental wave-vector);

  k = -N/2, -N/2 + 1, ... , N/2 - 2, N/2 - 1;

  The core procedure of the forward FFT in 1D is 
  given by the sum:

  Y(k) = Sum(j = 0)(j = N - 1){X(j) * E(j,k)};

  E(j,k) = Exp[- I * r(j) * q(k)];

  r(j) = j : position vector/index;

  I : complex imaginary unit;

  The input data can be thought as successive N 
  measurements in time or space, here we need to
  connect these two interpretations in order to
  understand the output of the FFTW package;

  For a simple sine-wave form signal measured N
  times during a total sampling length/time 'L', 
  we can define the input as:

  X(j) = Sin(K * j * ds);

  ds = L / N : sample length (or time);

  K : characteristic wave-vector;

  In the context of signal & image processing,
  we take: L = 1.0. The inverse of 'ds' gives
  the sampling frequency, which is then equal
  to N. In the mentioned context, 'ds' is ca-
  lled shutter speed or exposure time;

  We can write the vector 'K' in terms of its
  wavelenght 'lamb' with the latter being a
  certain number 'M' of sample lengths:

  lamb = M * ds = M / N;

  K = 2 * Pi / lamb = 2 * Pi * N / M;

  In the context of lattice systems, 'ds' is
  the lattice spacing and each component of
  the input signal is then associated with 
  a site in the lattice. Hence, we need to
  work in units where: ds = 1. Thus:

  L = N ---> ds = 1.0, (unit spacing);

  lamb0 = M, (wavelenght for ds = 1);

  K0 = 2 * Pi / lamb0 = 2 * Pi / M;

  The wavevector 'K0' has the value that we
  want to see as a peak in the Fourier spec-
  trum for the input sine-wave, i.e., these
  peaks should emerge due to the components:

  Y(+ K0) & Y(- K0);

  For that to happen, we need to set the wa-
  vector 'K' that goes in the input as:

  K = K0 * N; | ( Important relation );
  ''''''''''''''''''''''''''''''''''''' */

void fftw2D_WavePacketTest(int N,
			   double Kx0,
			   double Ky0)
{   
  int Sz = N * N; // Input size;

  int n0 = N / 2; // Half-length;
  
  double ds = 1.0 / N;
    
  double q0 = pi2 / N;

  int i, j, k, n, m1, m2;

  double x, y, z, z0, qx, qy, fc;

  double Kx, Ky, spcPower;    
    
  complex<double> zc;

  ofstream out1, out2;
       
  //--------------------------------
  // Create index & moment pointers:

  int *ivec, *jvec;

  ivec = new int[N];
  jvec = new int[N];

  double *qxVec, *qyVec;
      
  qxVec = new double[N];
  qyVec = new double[N];
    
  /* Positive components:
     |
     | i = 0, 1, 2, ... , n0 - 2, n0 - 1
     |
     | ivec = n0, n0 + 1, ... , N - 2, N - 1
     |
     | qxVec / q0 = 0, 1, ... , n0 - 2, n0 - 1

     Negative components:
     |
     | i = n0, n0 + 1, ... , N - 2, N - 1
     |
     | ivec = 0, 1, 2, ... , n0 - 2, n0 - 1
     |
     | qxVec / q0 = -n0, -n0 + 1, ... , -2, -1 */
    
  for (i = 0; i < N; i++)
    {
      n = 2 * (i / n0);

      k = i - n0 * (n - 1);
	
      ivec[i] = k;
	
      qxVec[i] = q0 * (k - n0);
    }

  /* Copy values to the other pointers: */

  copy(ivec, ivec + N, jvec);
    
  copy(qxVec, qxVec + N, qyVec);

  //---------------------
  // Create FFTW objetcs:
    
  fftw_complex *inpData, *outData;

  fftw_plan fwPlan, bwPlan;
    
  inpData = new fftw_complex[Sz];
  outData = new fftw_complex[Sz];

  fwPlan = fftw_plan_dft_2d(N, N, inpData, outData,
			    FFTW_FORWARD, FFTW_ESTIMATE);

  bwPlan = fftw_plan_dft_2d(N, N, outData, inpData,
			    FFTW_BACKWARD, FFTW_ESTIMATE);

  //--------------------------------------
  /* Define the input data & output files:

     >> Options (see the notes)...

     1) Input data defined in a lattice/array
     ** with unit spacing: ds = 1
     
     (Kx , Ky) = (Kx0 , Ky0);

     (x,y) & (i,j) are equivalent;

     x = i | Indices defining the complex
     y = j | valued input array/function;

     fc = 1.0 / Sz (attenuation factor);

     2) Input data defined in a spatial domain
     ** with mesh discretization: ds = 1.0 / N
     
     (Kx , Ky) = (N * Kx0 , N * Ky0);

     (x,y) & (i,j) are NOT equivalent;

     x = i * ds | Spatial coordinates defining
     y = j * ds | the input complex function;

     fc = 1.0 (no attenuation factor needed);

     Alternatively, one could set q0 = pi2, in
     this manner Kx = Kx0 & Ky = Ky0. See the
     documentation folder for more details;

     In (#), we set the index k in the same
     way we do in the function 'siteNumber';
     ......................................... */
  
  Kx = Kx0; // We work with  
  Ky = Ky0; // the option (1);

  fc = 1.0 / Sz;
      
  for   (i = 0; i < N; i++){
    for (j = 0; j < N; j++)
      {
	x = i; // Indices mapped to
	y = j; // coordinates...

	z0 = sin(Kx * x + Ky * y);
	  
	z = z0 * exp(- fc * (x * x + y * y));

	k = i + j * N; //(#)

	inpData[k][0] = z;

	inpData[k][1] = 0.0;
      }
  }
         
  out1.open(outDir1 + "fft2d_out.dat");
  out2.open(outDir1 + "fft2d_inv.dat");
    
  //--------------------------------
  // Perform forward FFT and record:
    
  fftw_execute(fwPlan);

  for (i = 0; i < N; i++)
    {
      m1 = ivec[i]; qx = qxVec[m1];
	
      for (j = 0; j < N; j++)
	{
	  m2 = jvec[j]; qy = qyVec[m2];
	    
	  k = m1 + m2 * N;

	  zc = complex<double>(outData[k][0],
			       outData[k][1]);
	  
	  spcPower = sqrt(real(zc * conj(zc)));

	  if (spcPower < dbleSmall)
	    {
	      spcPower = dbleSmall;}

	  out1 << qx << X4
	       << qy << X4
	       << log(spcPower) << endl;
	}
      
      out1 << endl;
    }

  //---------------------------------
  /* Perform backward FFT and record:
       
     The FFTW procedures are not normalized,
     each transform multiplies the data by
     the length N, thus we need to define
     the factor 'fc' below as the inverse
     of the size 'Sz = N * N' in order to
     normalize the resulting pointer; */
    
  fftw_execute(bwPlan);

  fc = 1.0 / Sz; // Normalization factor;
	
  for   (i = 0; i < N; i++){
    for (j = 0; j < N; j++)
      {
	x = i * ds;
	y = j * ds;
	  
	k = j + i * N; //(#)

	z = fc * inpData[k][0];
	    
	out2 << x << X4
	     << y << X4
	     << z << endl;
      }
      
    out2 << endl;
  }

  //---------------------------
  // Close files & free memory:
    
  out1.close();
  out2.close();

  fftw_destroy_plan(fwPlan);
  fftw_destroy_plan(bwPlan);

  delete[] ivec;
  delete[] jvec;

  delete[] qxVec;
  delete[] qyVec;
    
  delete[] inpData;
  delete[] outData;

  // #) Footnote:
  /*
    If the input data represemys a matrix
    defined on a rectangular domain N by
    M, then the pointer index 'k' within
    the loops must vary as follows:

    for (i = 0; i < N; i++){       		
    for (j = 0; j < M; j++){
      
    k = j + i * M; (...); }}

    Above, each i-iteration means that the
    index j covers M elements in the i-th
    row, so the index k scans the target
    pointer in the intended row-major
    order (data is stored row by row); */
}

//=================================
// FFTW1D procedures: study & tests

void fftw1D_WavePacketTest(int N,
			   double Kx0,
			   string knd)
{   
  int n0 = N / 2; // Half-length;
  
  double ds = 1.0 / N;
    
  double q0 = pi2 / N;

  int i, k, n;

  double x, z, z0, qx, fc;

  double Kx, spcPower;    
    
  complex<double> zc;

  ofstream out1, out2;
       
  //--------------------------------
  // Create index & moment pointers:

  int *ivec; double *qxVec;

  ivec = new int[N];
      
  qxVec = new double[N];
    
  for (i = 0; i < N; i++)
    {
      n = 2 * (i / n0);

      k = i - n0 * (n - 1);
	
      ivec[i] = k;
	
      qxVec[i] = q0 * (k - n0);
    }

  //---------------------
  // Create FFTW objetcs:
    
  fftw_complex *inpData, *outData;

  fftw_plan fwPlan, bwPlan;
    
  inpData = new fftw_complex[N];
  outData = new fftw_complex[N];

  fwPlan = fftw_plan_dft_1d(N, inpData, outData,
			    FFTW_FORWARD, FFTW_ESTIMATE);

  bwPlan = fftw_plan_dft_1d(N, outData, inpData,
			    FFTW_BACKWARD, FFTW_ESTIMATE);

  //--------------------------------------
  // Define the input data & output files:
  
  Kx = Kx0;

  fc = 0.05 / N;

  if (knd == "rect") /* Step-function signal */
    {
      for (i = 0; i < N; i++)
	{
	  x = i;

	  if (x == n0)
	    {
	      z = 0.0;
	    }
	  else
	    { z = (x - n0) / abs(x - n0); }

	  inpData[i][0] = z;

	  inpData[i][1] = 0.0;
	}
    }
  else /* Sine wave within Gaussian envelop
	  ( main frequencies are enhanced ) */
    {           
      for (i = 0; i < N; i++)
	{
	  x = (i - n0);

	  z0 = sin(Kx * i);
	  
	  z = z0 * exp(- fc * (x * x));

	  inpData[i][0] = z;

	  inpData[i][1] = 0.0;
	}
    }
         
  out1.open(outDir1 + "fft1d_out.dat");
  out2.open(outDir1 + "fft1d_inv.dat");
    
  //--------------------------------
  // Perform forward FFT and record:

  double cutoff = 0.5 * pi;
  
  fftw_execute(fwPlan);

  for (i = 0; i < N; i++)
    {
      k = ivec[i];

      qx = qxVec[k];

      if (knd == "rect")
	{
	  if (abs(qx) > cutoff)
	    {
	      outData[k][0] = 0.0;
	      outData[k][1] = 0.0;
	    }
	}
      
      zc = complex<double>(outData[k][0],
			   outData[k][1]);
	  
      spcPower = sqrt(real(zc * conj(zc)));

      if (spcPower < dbleSmall)
	{
	  spcPower = dbleSmall;}

      out1 << qx << X4 << log(spcPower) << endl;
    }

  //---------------------------------
  // Perform backward FFT and record:
    
  fftw_execute(bwPlan);

  fc = 1.0 / N; // Normalization factor;
	
  for (i = 0; i < N; i++)
    {
      x = i * ds;

      z = fc * inpData[i][0];
	    
      out2 << x << X4 << z << endl;
    }

  //---------------------------
  // Close files & free memory:
    
  out1.close();
  out2.close();

  fftw_destroy_plan(fwPlan);
  fftw_destroy_plan(bwPlan);

  delete[] ivec;

  delete[] qxVec;
    
  delete[] inpData;
  delete[] outData;
}

//==================================================
// Prepare global FFTW objects (3D space-time data):
// .................................................
// ALERT: although these plans are going to be used
// ------ for input pointers associated with 'Lsz X 
/*        Lsz X ntm' arrays (2D space + time forms),
	  the FFTW3D plans below need to be defined
	  with the frequency dimension (3rd index :
	  n = 0, 1, 2, ... , ntm - 1) being the 1st
	  input, i.e., we specify the input for the 
	  input integers for 'fftw_plan_dft_3d' as
	  a pointer associated with an 'ntm X Lsz X
	  Lsz' array , this ensures that the output
	  pointer can be correctly mapped to another
	  array with the usual form where the momen-
	  tum indices come first & frequency slices
	  can be obtained by fixing the third index;
*/
// INFO : the output of all DFT subroutines used in
// ------ this code are FFTW objects (pointers) of
/*        the type 'fftw_complex', for this class,
	  the real and imaginary parts of each
	  component can be accessed as follows:

	  dataVec[n][0] <--- real part;
	  dataVec[n][1] <--- imag part; */

void prepare_xyzPlan_fftw3D()
{
  rtxData = new fftw_complex[Nst];
  qwxData = new fftw_complex[Nst];

  rtyData = new fftw_complex[Nst];
  qwyData = new fftw_complex[Nst];

  rtzData = new fftw_complex[Nst];
  qwzData = new fftw_complex[Nst];

  wxPlan = fftw_plan_dft_3d(ntm, Lsz, Lsz, rtxData, qwxData,
			    FFTW_FORWARD, FFTW_PATIENT);

  wyPlan = fftw_plan_dft_3d(ntm, Lsz, Lsz, rtyData, qwyData,
			    FFTW_FORWARD, FFTW_PATIENT);

  wzPlan = fftw_plan_dft_3d(ntm, Lsz, Lsz, rtzData, qwzData,
			    FFTW_FORWARD, FFTW_PATIENT);
}

//=============================================
// Prepare global FFTW objects (2D field data):

void prepare_xyzPlan_fftw2D()
{
  rxData = new fftw_complex[Ns];
  qxData = new fftw_complex[Ns];

  ryData = new fftw_complex[Ns];
  qyData = new fftw_complex[Ns];

  rzData = new fftw_complex[Ns];
  qzData = new fftw_complex[Ns];

  xPlan = fftw_plan_dft_2d(Lsz, Lsz, rxData, qxData,
			   FFTW_FORWARD, FFTW_PATIENT);

  yPlan = fftw_plan_dft_2d(Lsz, Lsz, ryData, qyData,
			   FFTW_FORWARD, FFTW_PATIENT);

  zPlan = fftw_plan_dft_2d(Lsz, Lsz, rzData, qzData,
			   FFTW_FORWARD, FFTW_PATIENT);
}

//================================================
// Prepare global FFTW objects (1D temporal data):

void prepare_wPlan_fftw1D()
{
  tmData = new fftw_complex[ntm];
  wfData = new fftw_complex[ntm];

  wPlan = fftw_plan_dft_1d(ntm, tmData, wfData,
			   FFTW_FORWARD, FFTW_PATIENT);
}

//======================
// Destroy FFTW objects:

/* 3D objects */

void destroy_xyzPlan_fftw3D()
{
  delete[] rtxData; delete[] qwxData;
  delete[] rtyData; delete[] qwyData;
  delete[] rtzData; delete[] qwzData;
  
  fftw_destroy_plan(wxPlan);
  fftw_destroy_plan(wyPlan);
  fftw_destroy_plan(wzPlan);
}

/* 2D objects */

void destroy_xyzPlan_fftw2D()
{
  delete[] rxData; delete[] qxData;
  delete[] ryData; delete[] qyData; 
  delete[] rzData; delete[] qzData;    
  
  fftw_destroy_plan(xPlan);
  fftw_destroy_plan(yPlan);
  fftw_destroy_plan(zPlan);
}

/* 1D objects */

void destroy_wPlan_fftw1D()
{
  delete[] tmData;
  delete[] wfData;   
  
  fftw_destroy_plan(wPlan);
}

//===============================================
// Copy input real field to global FFTW pointers:

/* ----------------------------------
   CALL AFTER: prepare_xyzPlan_fftw3D

   INPUT SIZE: Ns X ntm X 3 (fixed)
   ---------------------------------- */

void set_xyzInput_fftw3D(double **tfield)
{ 
  double fx0, fy0, fz0;

  for (int k = 0; k < Nst; k++)
    {
      fx0 = tfield[k][0];
      fy0 = tfield[k][1];
      fz0 = tfield[k][2];
      
      rtxData[k][0] = fx0;
      rtxData[k][1] = 0.0;

      rtyData[k][0] = fy0;
      rtyData[k][1] = 0.0;

      rtzData[k][0] = fz0;
      rtzData[k][1] = 0.0;
    }
}

/* ----------------------------------
   CALL AFTER: prepare_xyzPlan_fftw2D

   INPUT SIZE: Ns X 3 (fixed)
   ---------------------------------- */

void set_xyzInput_fftw2D(double **field)
{
  double fx0, fy0, fz0;

  for (int k = 0; k < Ns; k++)
    {
      fx0 = field[k][0];
      fy0 = field[k][1];
      fz0 = field[k][2];
      
      rxData[k][0] = fx0;
      rxData[k][1] = 0.0;
      
      ryData[k][0] = fy0;
      ryData[k][1] = 0.0;
      
      rzData[k][0] = fz0;
      rzData[k][1] = 0.0;
    }
}

//==================================================
// Copy input complex vector to global FFTW pointer:

/* --------------------------------
   CALL AFTER: prepare_wPlan_fftw1D

   INPUT SIZE: ntm (fixed)
   -------------------------------- */

void set_Input_fftw1D(complex<double> *tvec)
{
  double rp0, ip0;
  
  for (int n = 0; n < ntm; n++)
    {
      rp0 = tvec[n].real();
      ip0 = tvec[n].imag();
      
      tmData[n][0] = rp0;
      tmData[n][1] = ip0;
    }
}

//============================================ //( limited code )
// Perform 3D discrete Fourier transform of an 
// input 3D real array (double type pointer) &
// return spectrum amplitude directly:

/* Note: in the physical context of the main
   code, the input is a time-dependent spin-
   field describing the system spin configu-
   ration evolution according to the Heisen-
   berg equations, the output is a frequency
   dependent spectrum amplitude (here multi-
   plied by the factor 1 / Nst) with the two
   first indices of the 3D pointer associa-
   ted with a Bloch wavevector within a re-
   gion of the 1st BZ, so that the ensemble
   average of the sum of the 3 components
   (each associated with a spin component)
   yield the dynamical structure factor
   with the frequency given by the 3rd
   (last) index of the 3D pointer; */

void get_FourierSpectrum3D(double **tfield,
			   double **wspecs)
{  
  const double fc = sqrt(1.0 / Nst);
  
  double pwxx, pwyy, pwzz;

  complex<double> xc, yc, zc;

  //---------------------------------
  // Execute 3D forward discrete FFT:

  set_xyzInput_fftw3D(tfield);
  
  fftw_execute(wxPlan);
  fftw_execute(wyPlan);
  fftw_execute(wzPlan);

  //-----------------------------------
  // Copy spectrum amplitude to output:
  
  for (int k = 0; k < Nst; k++)
    {      
      xc = fc * complex<double>(qwxData[k][0],
				qwxData[k][1]);

      yc = fc * complex<double>(qwyData[k][0],
				qwyData[k][1]);

      zc = fc * complex<double>(qwzData[k][0],
				qwzData[k][1]);

      pwxx = max(dbleTiny, real(xc * conj(xc)));
      pwyy = max(dbleTiny, real(yc * conj(yc)));
      pwzz = max(dbleTiny, real(zc * conj(zc)));
	  
      wspecs[k][0] = (pwxx + pwyy) * 0.5; // Inplane;
      
      wspecs[k][1] = pwzz; // Transverse;
    }
}

//============================================
// Perform 2D discrete Fourier transform of an
// input 2D real array (double type pointer) &
// return spectrum amplitude directly:

/* Note: in the physical context of the main
   code, the input is the spin-field descri-
   bing the spin configuration on the latti-
   system, the output is the spectrum ampli-
   tude multiplied by the factor 1 / Ns for
   each Bloch wavevector within a region of
   the 1st BZ, so that the ensemble average
   of the sum of the three components (each
   associated with a spin component) yield
   the static structure factor. The latter
   has its range extended by the procedure
   'record_SpecArray'; */

void get_FourierSpectrum2D(double **field,
			   double **specs)
{  
  const double fc = sqrt(iNs);

  double pwxx, pwyy, pwzz;

  complex<double> xc, yc, zc;

  //---------------------------------
  // Execute 2D forward discrete FFT:

  set_xyzInput_fftw2D(field);
  
  fftw_execute(xPlan);
  fftw_execute(yPlan);
  fftw_execute(zPlan);

  //-----------------------------------
  // Copy spectrum amplitude to output:
  
  for (int k = 0; k < Ns; k++)
    {      
      xc = fc * complex<double>(qxData[k][0],
				qxData[k][1]);

      yc = fc * complex<double>(qyData[k][0],
				qyData[k][1]);

      zc = fc * complex<double>(qzData[k][0],
				qzData[k][1]);

      pwxx = max(dbleTiny, real(xc * conj(xc)));
      pwyy = max(dbleTiny, real(yc * conj(yc)));
      pwzz = max(dbleTiny, real(zc * conj(zc)));
	  
      specs[k][0] = (pwxx + pwyy) * 0.5; // Inplane;
      
      specs[k][1] = pwzz; // Transverse;
    }
}

/* ............................................
   The procedure 'get_FourierSpectrum2D' above
   performs the tasks done by the two next ones
   below 'get_SqData' & 'get_StaticSF', but the
   complex forms represented by the three FFTW
   qwData-type pointers cannot be retrieved;
   ............................................ */

//=============================================
// Execute fftw-plans (2D) with the input field
// and transfer the data from the FFTW pointers
// to the complex-type pointers Sqx, Sqy & Sqz:
/*
  ................
  Procedure steps:
  
  -- Assign the current 'spinField' 
  -- data to the FFTW2D input pointer;

  -- Execute the associated fftw-plans to
  -- compute qwxData, qwyData & qwzData; 

  -- Assign results scaled by the factor
  -- sqrt(1.0 / Ns) to preserve L2 norm,
  -- but other scalings can be used...

  Note: this code (and next one too), work
  for square & triangular lattices only, use
  get_SqData_iMKL for general case of lattices
  composed of sublattices (Kagome, Lieb, etc.); */

void get_SqData_FFTW(double **field,
		     complex<double> *Sqx,
		     complex<double> *Sqy,
		     complex<double> *Sqz)
{
  const double fc = sqrt(iNs);
  
  set_xyzInput_fftw2D(field);
  
  fftw_execute(xPlan);
  fftw_execute(yPlan);
  fftw_execute(zPlan);
  
  for (int k = 0; k < Ns; k++) //( Ns = Ns0 = Lsz * Lsz )
    {      
      Sqx[k] = fc * complex<double>(qxData[k][0],
				    qxData[k][1]);

      Sqy[k] = fc * complex<double>(qyData[k][0],
				    qyData[k][1]);

      Sqz[k] = fc * complex<double>(qzData[k][0],
				    qzData[k][1]);
    }
}

/* ***********************
   Intel-MKL version below */

void get_SqData_iMKL_v0(double **field,
			complex<double> *Sqx,
			complex<double> *Sqy,
			complex<double> *Sqz)
{    
  /*---------------------------------
    Transfer field real-values to the
    two-dimensional complex pointers */

  int i, j, k;
  
  double fx0, fy0, fz0;
  
  complex<double> SxMat[Lsz][Lsz]; 
  complex<double> SyMat[Lsz][Lsz];
  complex<double> SzMat[Lsz][Lsz];
  
  for (i = 0; i < Lsz; i++)
    {
      for (j = 0; j < Lsz; j++)
	{    
	  k = i * Lsz + j;
	  
	  fx0 = field[k][0];
	  fy0 = field[k][1];
	  fz0 = field[k][2];
      
	  SxMat[i][j] = complex<double>(fx0, 0.0);
      	  SyMat[i][j] = complex<double>(fy0, 0.0);
	  SzMat[i][j] = complex<double>(fz0, 0.0);
	}
    }

  /*------------------------------------
    Create a descriptor for the FFT plan */
    
  DFTI_DESCRIPTOR_HANDLE handle = NULL;
    
  MKL_LONG stat, dims[2] = {Lsz, Lsz};

  /*--------------------------------
    Initialize & Commit the FFT plan */
    
  stat = DftiCreateDescriptor(&handle,
			      DFTI_DOUBLE,
			      DFTI_COMPLEX, 2, dims);
  
  stat = DftiCommitDescriptor(handle);
  
  /*--------------------------
    Perform the 2D forward DFT
    and free the descriptor... */

  MKL_LONG xStatus, yStatus, zStatus;
   
  xStatus = DftiComputeForward(handle, SxMat);
  yStatus = DftiComputeForward(handle, SyMat);
  zStatus = DftiComputeForward(handle, SzMat);

  DftiFreeDescriptor(&handle);
  
  /*----------------------------------
    Transfer values to output pointers
    with scaling factor (fc) included */

  const double fc = sqrt(iNs);

  complex<double> cx, cy, cz;
  
  for (i = 0; i < Lsz; i++)
    {
      for (j = 0; j < Lsz; j++)
	{	  
	  cx = SxMat[i][j];
	  cy = SyMat[i][j];
	  cz = SzMat[i][j];
	  
	  k = i * Lsz + j;
	  
	  Sqx[k] = fc * cx; 
	  Sqy[k] = fc * cy;
	  Sqz[k] = fc * cz;
	}
    }
}

//=============================================
// Compute the spin structure-factor components
// S(q) using Intel MKL 2D complex FFTs for each
// sublattice of a vector (input) spin field:

void get_SqData_iMKL(double **field,
		     complex<double> *Sqx,
		     complex<double> *Sqy,
		     complex<double> *Sqz)
{  
  const int l2 = Gsz / 2;
  
  const double isz = 1.0 / Gsz;
  
  const double fc = sqrt(iNq);
  
  /*-------------------------------
    Reciprocal lattice grid vectors */
  
  const Vec2d rclV1 = (C3SYM ? isz * bvec1 : isz * cvec1);
  const Vec2d rclV2 = (C3SYM ? isz * bvec2 : isz * cvec2);

  Vec2d kvec; //( Bloch k wave-vector )
  
  /*-----------------------
    Define needed variables */  
  
  complex<double> SxMat[Gsz][Gsz]; 
  complex<double> SyMat[Gsz][Gsz];
  complex<double> SzMat[Gsz][Gsz];

  double phi, ii, jj, qx, qy;

  complex<double> pfac;
  
  int i, j, k, n; 
  
  /*----------------------------------------
    Compute spin q-forms within the DFT-grid

    .............
    Commentaries:
    
    k --> grid site index (0, 1, ... , Nsg - 1);

    idx --> physical site indice (0, 1, ... , Ns - 1);

    Below, idx < 0 means that the grid site does not
    map into a physical site in the lattice, so the
    input spin is set as zero, otherwise we use the
    site index given by 'gridMap' to transfer the
    spin to the input 'S(x,y,z)Mat' matrices defi-
    ned within the grid of size Gsz x Gsz; such
    condition only occurs when the lattice is
    composed by multiple sub-lattices and DFT
    requires a uniform well-defined grid; */

  /* Transfer spin-field to input matrices: */ 
  
  for (i = 0; i < Gsz; i++)
    {
      for (j = 0; j < Gsz; j++)
	{    
	  k = i * Gsz + j; //(#)

	  int idx = gridMap[k];

	  if (idx < 0)
	    {
	      SxMat[i][j] = zero;
	      SyMat[i][j] = zero;
	      SzMat[i][j] = zero;
	    }
	  else//( transfer physical spin components )
	    {
	      Vec3d spinVec = { field[idx][0],
				field[idx][1],
				field[idx][2] };
	      
	      SxMat[i][j] = complex<double>(spinVec[0], 0.0);
	      SyMat[i][j] = complex<double>(spinVec[1], 0.0);
	      SzMat[i][j] = complex<double>(spinVec[2], 0.0);
	    }
	}
    }
  
  /*  Perform the 2D forward DFTs (inplace): */           

  MKL_LONG xStatus, yStatus, zStatus;
  
  xStatus = DftiComputeForward(handle, SxMat);
  yStatus = DftiComputeForward(handle, SyMat);
  zStatus = DftiComputeForward(handle, SzMat);

  /* Transfer values to output pointers: */
      
  for (i = 0; i < Gsz; i++)
    {	    
      for (j = 0; j < Gsz; j++)
	{	      	      
	  k = i * Gsz + j;
 
	  Sqx[k] = fc * SxMat[i][j]; 
	  Sqy[k] = fc * SyMat[i][j];
	  Sqz[k] = fc * SzMat[i][j];
	}
    }
}

//==================================================  
// 2D-DFT procedure for a general input qct-lattice:
/*
  ................
  About this code: ( qct: quasi-crystal )
  
  Input lattice vector positions are obtained
  from the global pointer 'rvecList' which is
  acquired from an input file (quasi-crystal);
  
  The transform is explicitly calculated via
  direct calculation of complex sums, no pe-
  ridiocity assumed;

  Output pointers are indexed accordingly to
  the FFTW convention, i.e., we map the indi-
  ces so that positive frequencies come first
  and negative frequencies in second (#):
  
  I = [0 : N/2-1 , -N/2 : -1];

  We use the ternary operation:
  
  condition ? value_if_true : value_if_false;
  ........................................... */

void get_qct_SqData(double **field,
		    complex<double> *Sqx,
		    complex<double> *Sqy,
		    complex<double> *Sqz)
{
  /* -----------------
     Declare constants */
  
  const int l2 = Lsz / 2;

  const double fc = sqrt(iNs);

  const complex<double> inum = { 0.0, - dq };

  /* -----------------
     Declare variables */

  int i, j, nx, ny, k, index;

  double qx, qy, qrad;

  complex<double> cplxFac;

  /* --------------------
     Perform triple loop:

     (i,j) : output index;

     k: site index [0, Ns - 1];
     
     Wavevectors: q = dq * (nx , ny); */
  
  for (j = 0; j < Lsz; j++)
    {
      ny = (j < l2) ? j : j - Lsz; //(#)

      qy = ny * dq;

      for (i = 0; i < Lsz; i++)
	{
	  nx = (i < l2) ? i : i - Lsz; //(#)

	  qx = nx * dq;

	  qrad = sqrt(pow(qx, 2) + pow(qy, 2));
      
	  complex<double> Sq[3] = {zero, zero, zero};
	  
	  for (k = 0; k < Ns; k++)
	    {	
	      cplxFac = exp( inum * ( nx * rvecList[k][0] +
				      ny * rvecList[k][1] ) );

	      Sq[0] += field[k][0] * cplxFac;
	      Sq[1] += field[k][1] * cplxFac;
	      Sq[2] += field[k][2] * cplxFac;
	    }

	  index = j * Lsz + i;
	    
	  Sqx[index] = fc * Sq[0];
	  Sqy[index] = fc * Sq[1];
	  Sqz[index] = fc * Sq[2];
	}
    }
}

/* ***************************************
   get_qct_SqData ---> find hotspot points 
   *************************************** */

void get_qct_SqPkLoc(double **field,
		     vector<int> &nxSqPeak,
		     vector<int> &nySqPeak)
{
  /* -----------------
     Declare constants */
  
  const int l2 = Lsz / 2;

  const double fc = 1.0 / Nq;

  const double f1d3 = 1.0 / 3.0;

  const complex<double> inum = { 0.0, - dq };

  /* -----------------
     Declare variables */

  int i, j, nx, ny, k, m, index;
  
  double qrad, qtht, specVal, specRef;

  double qx, qy, qxMax, qyMax;
 
  complex<double> cplxFac;
  
  /* --------------------
     Perform triple loop: */

  specRef = 0.0;
  
  for (j = 0; j < Lsz; j++)
    {
      ny = (j < l2) ? j : j - Lsz;

      qy = ny * dq;

      for (i = 0; i < Lsz; i++)
	{
	  nx = (i < l2) ? i : i - Lsz;

	  qx = nx * dq;

	  qtht = atan2(qy, qx);

	  qrad = sqrt(pow(qx, 2) + pow(qy, 2));
	  
	  if ( ((qrad > 2.0) && (qrad < 5.0)) &&
	       ((qtht > 0.0) && (qtht < a45)) )
	    {  
	      complex<double> Sq[3] = {zero, zero, zero};
			  
	      for (k = 0; k < Ns; k++)
		{	
		  cplxFac = exp( inum * ( nx * rvecList[k][0] +
					  ny * rvecList[k][1] ) );

		  Sq[0] += field[k][0] * cplxFac;
		  Sq[1] += field[k][1] * cplxFac;
		  Sq[2] += field[k][2] * cplxFac;
		}

	      specVal = ( real(Sq[0] * conj(Sq[0])) +
			  real(Sq[1] * conj(Sq[1])) +
			  real(Sq[2] * conj(Sq[2])) );

	      if (specVal > specRef)
		{
		  specRef = specVal;

		  qxMax = qx;
		  qyMax = qy;
		}				
	    }///[ Arc-region filter ]
	}
    }
  
  /* -----------------------------
     Find indices for the 8 peaks: */
  
  vector<double> qxVec(8), qyVec(8);
 
  qxVec[0] = (+ qxMax); qxVec[1] = (- qxMax);
  qyVec[0] = (+ qyMax); qyVec[1] = (- qyMax);

  qxVec[2] = (+ qyMax); qxVec[3] = (- qyMax);
  qyVec[2] = (+ qxMax); qyVec[3] = (- qxMax);

  qxVec[4] = (- qxMax); qxVec[5] = (+ qyMax);
  qyVec[4] = (+ qyMax); qyVec[5] = (- qxMax);

  qxVec[6] = (+ qxMax); qxVec[7] = (- qyMax);
  qyVec[6] = (- qyMax); qyVec[7] = (+ qxMax);

  for (i = 0; i < 8; i++)
    {
      nxSqPeak[i] = round(qxVec[i] / dq);
      nySqPeak[i] = round(qyVec[i] / dq);
    }
}


/* *******************************************************
   get_qct_SqData ---> get amplitude at the hotspot points 
   ******************************************************* */

void get_qct_SqPks4(double **field,		    
		    vector<int> &nxSqPeak,
		    vector<int> &nySqPeak, Vec4d &SqPeaks)
{
  /* -----------------
     Declare constants */
  
  const int l2 = Lsz / 2;

  const double fc = 0.5 / Nq;

  const double f1d3 = 1.0 / 3.0;

  const complex<double> inum = { 0.0, - dq };

  /* -----------------
     Declare variables */

  int i, k, nx, ny;

  double qx, qy, specVal;

  complex<double> cplxFac;

  vector<double> specVec(8);

  /* --------------------
     Perform triple loop:

     (i,j) : output index;

     k: site index [0, Ns - 1];
     
     Wavevectors: q = dq * (nx , ny); */
  
  for (i = 0; i < 8; i++)
    {
      complex<double> Sq[3] = {zero, zero, zero};
      
      nx = nxSqPeak[i]; qx = nx * dq;
      ny = nySqPeak[i]; qy = ny * dq;
			  
      for (k = 0; k < Ns; k++)
	{	
	  cplxFac = exp( inum * ( nx * rvecList[k][0] +
				  ny * rvecList[k][1] ) );

	  Sq[0] += field[k][0] * cplxFac;
	  Sq[1] += field[k][1] * cplxFac;
	  Sq[2] += field[k][2] * cplxFac;
	}

      specVec[i] = f1d3 * ( real(Sq[0] * conj(Sq[0])) +
			    real(Sq[1] * conj(Sq[1])) +
			    real(Sq[2] * conj(Sq[2])) );
    }

  SqPeaks[0] = sqrt(fc * (specVec[0] + specVec[1]));
  SqPeaks[1] = sqrt(fc * (specVec[2] + specVec[3]));
  SqPeaks[2] = sqrt(fc * (specVec[4] + specVec[5]));
  SqPeaks[3] = sqrt(fc * (specVec[6] + specVec[7]));
}

//==================================== 
// Calculate the spectrum amplitude of
// the DFT of a general input lattice:

void get_crystalXRay(double **xraySpec)
{
  /* -----------------
     Declare constants */
  
  const int l2 = Gsz / 2;

  const double fc = sqrt(iNq);

  const complex<double> inum = { 0.0, - dq };

  /* -----------------
     Declare variables */

  int i, j, n, k, m1, m2, nx, ny, index;

  complex<double> cplxFac, znum;

  complex<double> *crystalSpec;

  /*---------------------
    Create index pointer: */

  int *mvec = new int[Gsz];
       
  for (i = 0; i < Gsz; i++)
    {
      n = 2 * (i / l2);

      k = i - l2 * (n - 1);
	
      mvec[i] = k;
    }

  /* ----------------------------
     Perform DFT of the lattice &
     include scaling factor (fc) */

  crystalSpec = new complex<double>[Nq];

  for (j = 0; j < Gsz; j++)
    {
      ny = (j < l2) ? j : j - Gsz;

      for (i = 0; i < Gsz; i++)
	{
	  nx = (i < l2) ? i : i - Gsz;
      
	  znum = zero;
	  
	  for (k = 0; k < Ns; k++)
	    {	
	      cplxFac = exp( inum * ( nx * rvecList[k][0] +
				      ny * rvecList[k][1] ) );

	      znum += cplxFac;
	    }

	  index = j * Gsz + i;
	    
	  crystalSpec[index] = fc * znum;
	}
    }   
  
  /*---------------------------
    Compute spectrum amplitude: */
  
  for (i = 0; i < Gsz; i++)
    {
      m1 = mvec[i];
	
      for (j = 0; j < Gsz; j++)
	{
	  m2 = mvec[j];

	  k = m1 * Gsz + m2;

	  znum = crystalSpec[k];

	  xraySpec[i][j] = real(znum * conj(znum));
	}
    }
 
  /*--------------------
    Deallocate pointers: */
      
  delete[] mvec;

  delete[] crystalSpec;
}

//==========================================
// Compute static (zero frequency) structure 
// factor from input DFT-2D complex pointers:

void get_StaticSFac(complex<double> *Sqx,
		    complex<double> *Sqy,
		    complex<double> *Sqz,
		    double **specData)
{  
  double pwxx, pwyy, pwzz;

  complex<double> xc, yc, zc;  

  //----------------------------------------
  // Compute spectrum amplitude (static SF):
 
  for (int k = 0; k < Nq; k++)
    {      
      xc = Sqx[k];
      yc = Sqy[k];
      zc = Sqz[k];

      pwxx = max(dbleTiny, real(xc * conj(xc)));
      pwyy = max(dbleTiny, real(yc * conj(yc)));
      pwzz = max(dbleTiny, real(zc * conj(zc)));
	  
      specData[k][0] = pwxx;
      specData[k][1] = pwyy;  
      specData[k][2] = pwzz;
    }
}

//========================================================
// 1D discrete Fourier transform of complex temporal data:

/* Note: the output contains the spectrum amplitude (real
   number) with the rescaling factor (1.0 / ntm) included; */

void get_FourierSpectrum1D(complex<double> *tvec, double *wvec)
{   
  const double fc = sqrt(1.0 / ntm);
  
  double pw; complex<double> wc;

  /*----------------------------------
    Execute 1D forward discrete FFT &
    copy spectrum amplitude to output: */

  set_Input_fftw1D(tvec);
  
  fftw_execute(wPlan);
    
  for (int n = 0; n < ntm; n++)
    {	   
      wc = fc * complex<double>(wfData[n][0],
				wfData[n][1]);

      pw = real(wc * conj(wc));

      wvec[n] = max(dbleTiny, pw);	 
    }
}

//==================================================
// Generate a lattice form Gsz x Gsz arrays with the
// ordered components of the input spectrum field:

/* Input : FFTW ordered 'SpecField' amplitudes of
   ------- the DFT spectrum: x [0], y [1], z [3];

   Output: Normal ordered array form outputs in 
   ------- three forms: xy, yz, zz, tt (total); */

void get_OrderedSpecArray2D(double **SpecField,
			    double **xySpField, 
			    double **yzSpField,
			    double **zzSpField,
			    double **ttSpField)
{
  const double f1d2 = 1.0 / 2.0;
  const double f1d3 = 1.0 / 3.0;
  
  const int l2 = Gsz / 2;
   
  int i, j, k, n, m1, m2, *mvec;
       
  /*---------------------
    Create index pointer: */

  mvec = new int[Gsz];
       
  for (i = 0; i < Gsz; i++)
    {
      n = 2 * (i / l2);

      k = i - l2 * (n - 1);
	
      mvec[i] = k;
    }
    
  /*----------------------------------
    Reshape and assign ordered values:
    
    SpecField ----> New Array
    |               | 
    |> N-sites x 2  |> Length x Length */
  
  for (i = 0; i < Gsz; i++)
    {
      m1 = mvec[i];
	
      for (j = 0; j < Gsz; j++)
	{
	  m2 = mvec[j];

	  k = m1 * Gsz + m2;
	      
	  xySpField[i][j] = f1d2 * ( SpecField[k][0] +
				     SpecField[k][1] );
	  
	  yzSpField[i][j] = f1d2 * ( SpecField[k][1] +
				     SpecField[k][2] );
	  
	  ttSpField[i][j] = f1d3 * ( SpecField[k][0] +
				     SpecField[k][1] +
				     SpecField[k][2] );

	  zzSpField[i][j] = SpecField[k][2];
	}
    }
    
  delete[] mvec;
}

//=====================================
// Return 1D spectrum amplitude pointer
// with ordered frequency components:

void get_OrderedSpecArray1D(double *wvec0,
			    double *wvec1)
{
  const int n0 = ntm / 2;
   
  int i, j, k, n, m, *mvec;
       
  /*---------------------
    Create index pointer: */

  mvec = new int[ntm];
       
  for (n = 0; n < ntm; n++)
    {
      j = 2 * (n / n0);

      k = n - n0 * (j - 1);
	
      mvec[n] = k;
    }
    
  /*--------------------------------
    Reshape & assign ordered values: */
  
  for (n = 0; n < ntm; n++)
    {
      m = mvec[n];
		  
      wvec1[n] = wvec0[m];
    }
    
  delete[] mvec;
}

//============================================
// Record extend range spectrum field to file:

/* Notes: we use the pediodicity of the discrete Fourier
   transform to extend the range of frequencies/wavevec-
   tors over a wide area of the rescaled (1 / Gsz) reci-
   procal lattice, the result is a distribution for the
   spectrum amplitude that covers the hexagonal region
   of the 1st BZ and its surroundings. Here, we filter
   out the points outside the 1st BZone when recording
   the data for the static structure factor; */

void record_SpecArray2D(double **specField,
			string fname, int codeTag)
{  
  const double sz = Gsz; // Linear-size (DFT-grid);
 
  const double fc = 1.0 / sz; // Rescaling factor;

  const int l2 = sz / 2; // Half lattice-size;
    
  const Vec2d b1 = fc * bvec1; // Rescaled reciprocal
  const Vec2d b2 = fc * bvec2; // lattice vectors;

  const Vec2d qvec1 = l2 * b1;
  const Vec2d qvec2 = l2 * b2;

  const string nn = "NaN";

  int i, j, n1, n2, m1, m2;
  
  //----------------------------------
  // Create 1st list of shift vectors:

  Vec2d qvec0, vecList[3][3];

  if ((geom != "square") && (geom != "lieb"))
    {
      for (n1 = 0; n1 < 3; n1++)
	{  
	  m1 = 2 * (n1 - 1);
      
	  for (n2 = 0; n2 < 3; n2++)
	    {  
	      m2 = 2 * (n2 - 1);

	      qvec0 = (m1 * qvec1 +
		       m2 * qvec2);
	  
	      vecList[n1][n2] = qvec0;
	    }
	}
    }

  //--------------------------------
  // Record data & replicas to file:

  int i0 = codeTag;

  ofstream outfile;
  
  double qx0, qx, qy0, qy, pw;

  Vec2d qvec; // Bloch-wave vectors;
  
  outfile.open(outDir1 + subDirVec[i0] + fname);

  if (C3SYM)//--> Triangular, Hexagonal, Kagome;
    {
      for (n1 = 0; n1 < 3; n1++)
	{
	  for (i = 0; i < sz; i++)
	    {
	      for (n2 = 0; n2 < 3; n2++)
		{
		  qvec0 = vecList[n1][n2];

		  qx0 = qvec0[0];
		  qy0 = qvec0[1];
	      
		  for (j = 0; j < sz; j++)
		    {      
		      pw = specField[i][j];

		      m1 = i - l2;
		      m2 = j - l2;

		      qvec = m1 * b1 + m2 * b2;

		      qx = qx0 + qvec[0];
		      qy = qy0 + qvec[1];
		  
		      /*----------------------------
			Hexagonal (1st BZ) filter...
			---------------------------- */
		      
		      if (hexFilter(qx, qy, Qval, dq))
			{
			  outfile << qx << X3
				  << qy << X3
				  << pw << endl;
			}
		      /*-------------------------------
			Below, we set 'pw' as 'NaN'
			to make GnuPlot skip the point;
			------------------------------- */
		      
		      else {outfile << qx << X3
				    << qy << X3
				    << nn << endl;}
		    }
		}//// n2-loop END;
      
	      outfile << endl;
	    }
	}//// n1-loop END;
    }
  else//--> Square, Lieb;
    {  
      for (i = 0; i <= sz; i++)
	{ 
	  for (j = 0; j <= sz; j++)
	    {
	      if (i < sz && j < sz)
		{
		  pw = specField[i][j];
		}
	      else if (i == sz && j == sz)
		{
		  pw = specField[0][0];
		}
	      else if (i == sz)
		{
		  pw = specField[0][j];
		}
	      else if (j == sz)
		{
		  pw = specField[i][0];
		}
	      	    
	      m1 = i - l2;
	      m2 = j - l2;
	    	    
	      qx = dq * m1;
	      qy = dq * m2;

	      outfile << qx << X3
		      << qy << X3
		      << pw << endl;
	    }

	  outfile << endl;
	}      
    }//// Geometry-check END;
  
  outfile.close();
}

//==============================================
// Record 1D spectrum amplitude pointer to file:

void record_SpecArray1D(double *wvec, string fname)
{
  int n; double pw, wfreq;
  
  ofstream outfile;
  
  outfile.open(outDir1 + subDir1 + fname);

  for (n = 0; n < ntm; n++)
    {
      wfreq = n * dwf - wmax;
      
      pw = wvec[n];

      outfile << wfreq << X4 << pw << endl;
    }

  outfile.close();
}

//============================================
// Extract amplitudes across the KGMYG / YGMXG   
// path from the input ordered spectrum field:

/* Input  : ordered spectrum amplitudes;
   
   Output : amplitudes across the path;
   
   The output SpFPath pointer must be
   allocated previously as follows:

   SpFPath = new double[npPath];

   Above, npPath is a global int. defined
   within the main code after the call of
   the suboroutine 'get_qPathWaveVectors'; */

void get_Spec_qPath(double **SpField,
		    double  *SpFPath)
{
  const int l2 = Gsz / 2;
     
  int i0, j0, i, j, n;

  /*---------------------
    Transfer q-path data:
    ( n : path counter ) */
  
  n = 0;

  if (C3SYM)
    {
      /* K ---> G */
  
      j0 = l2 / 3;

      for (j = j0; j <= l2; j++)
	{	      
	  i = Gsz - j;
      
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      /* G ---> M */
  
      i = l2; j0 = l2 - 1;
  
      for (j = j0; j >= 0; j--)
	{	            
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      /* M ---> Y */
 
      for (j = 1; j <= l2 / 2; j++)
	{	      
	  i = l2 - j;
      
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      // Y --> G //

      i0 = l2 / 2 + 1;
 
      for (i = i0; i <= l2; i++)
	{	      
	  j = i;
      
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}
    }
  else // Square-like geometry ...
    {
      // Y : i = l2; j = 0
      // G : i = l2; j = l2
      // X : i =  0; j = l2
      // M : i =  0; j = 0
            
      /* Y ---> G */
  
      i = l2; j = 0;

      for (j = 0; j <= l2; j++)
	{	            
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      /* G ---> M */

      i = l2; j = l2;
    
      for (i = l2 - 1; i >= 0; i--)
	{	      
	  j = i;
      
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      /* M ---> X */

      i = 0; j = 0;
      
      for (j = 1; j <= l2; j++)
	{	            
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}

      /* X ---> G */

      i = 0; j = l2;

      for (i = 1; i <= l2; i++)
	{	            
	  SpFPath[n] = SpField[i][j];

	  n = n + 1;
	}
    }

  //////////////
  /* YGXMG path
  {
    // Y ---> G //
  
    i = l2; 

    for (j = 0; j <= l2; j++)
      {	            
	SpFPath[n] = SpField[i][j];

	n = n + 1;
      }

    // G ---> X //
  
    j = l2; i0 = l2 - 1;
  
    for (i = i0; i >= 0; i--)
      {	            
	SpFPath[n] = SpField[i][j];

	n = n + 1;
      }

    // X ---> M //

    j0 = l2 - 1; i = 0;
      
    for (j = j0; j >= 0; j--)
      {	            
	SpFPath[n] = SpField[i][j];

	n = n + 1;
      }

    // M --> G //

    for (i = 1; i <= l2; i++)
      {	      
	j = i;
      
	SpFPath[n] = SpField[i][j];

	n = n + 1;
      }
  }///*/
}

//=================================================
// Calculate correlation lengths & extract spectrum
// amplitudes at key points of the structure factor:

void get_corrLen(double **specField,
		 Vec5d &SpecVec,
		 Vec5d &CLenVec)
{
  const int l2 = Gsz / 2;
  const int l3 = Gsz / 3;
  const int l4 = Gsz / 4;

  const int Qm1 = + l3; // Qvec within the y < 0
  const int Qm2 = - l3; // right quadrant;
  
  const double fc = 1.0 / Gsz; // Rescaling factor;

  Lst5i i1Vec, i2Vec, j1Vec, j2Vec; 
  
  /* ----------------------------------
     Get components at the wavevectors:

     1) Qvec within the y < 0 right quadrant;
   
     2) Qvec + smallest shift in the y-axis; */

  if (C3SYM)
    {
      /*........ | Hex-edge
	K-Point: | corner */
      
      i1Vec[0] = Qm1 + l2;
      j1Vec[0] = Qm2 + l2;

      i2Vec[0] = Qm1 + l2 + 0;
      j2Vec[0] = Qm2 + l2 + 1;

      /*........... | Hex-side
	M-Point(1): | mid-point */
      
      i1Vec[1] = l2; j1Vec[1] = 0;
      i2Vec[1] = l2; j2Vec[1] = 1;

      /*........... | Hex-side
	M-Point(2): | mid-point */
    
      i1Vec[2] = 0; j1Vec[2] = l2;
      i2Vec[2] = 1; j2Vec[2] = l2;

      /*........ | Center
	G-Point: | point */
      
      i1Vec[3] = l2;
      j1Vec[3] = l2;

      i2Vec[3] = l2 + 0;
      j2Vec[3] = l2 + 1;

      /*........ | Quadrant
	Z-Point: | center point */
      
      i1Vec[4] = l4;
      j1Vec[4] = l4;

      i2Vec[4] = l4 + 0;
      j2Vec[4] = l4 + 1;
    }
  else // Square-like geometry ...
    {
      /*........ | Square
	M-Point: | edge corner */
      
      i1Vec[0] = 0; j1Vec[0] = 0;
      i2Vec[0] = 0; j2Vec[0] = 1;

      /*........ | X-side
	X-Point: | mid-point */

      i1Vec[1] = 0; j1Vec[1] = l2;
      i2Vec[1] = 1; j2Vec[1] = l2; 
      
      /*........ | Y-side
	Y-Point: | mid-point */

      i1Vec[2] = l2; j1Vec[2] = 0;
      i2Vec[2] = l2; j2Vec[2] = 1;

      /*........ | Center
	G-Point: | point */
      
      i1Vec[3] = l2;
      j1Vec[3] = l2;

      i2Vec[3] = l2 + 0;
      j2Vec[3] = l2 + 1;

      /*........ | Quadrant
	Z-Point: | center point */
      
      i1Vec[4] = l4;
      j1Vec[4] = l4;

      i2Vec[4] = l4 + 0;
      j2Vec[4] = l4 + 1;
    }

  /* -----------------------------
     Calculate intensity ratio &
     transfer SQ1 value to vector: */

  int k, i1, i2, j1, j2;
  
  double SQ1, SQ2;

  Vec5d wvec;

  for (k = 0; k < 5; k++)
    {
      i1 = i1Vec[k];
      j1 = j1Vec[k];

      i2 = i2Vec[k];
      j2 = j2Vec[k];
      
      SQ1 = specField[i1][j1];
      SQ2 = specField[i2][j2];

      wvec[k] = ( SQ1 / SQ2 );

      SpecVec[k] = SQ1;
    }

  /* -----------------------------
     Calculate correlation length: */

  double w, ifc, CLen;
  
  ifc = 1.0 / dq;

  for (k = 0; k < 5; k++)
    {
      w = wvec[k];
      
      if (w > 1.0)
	{
	  CLen = ifc * sqrt(w - 1.0);
	}
      else
	{ CLen = 0.0; }

      CLenVec[k] = CLen;
    }
}

//===================================
// Extract spectrum amplitudes at key
// points of the structure factor ...

void get_specAmp(double **specField, Vec4d &SpecVec)
{
  const int l2 = Gsz / 2;
  const int l3 = Gsz / 3;

  const int Qm1 = + l3;
  const int Qm2 = - l3;

  Vec4d iVec, jVec;
  
  /* ----------------------------------
     Get components at the wavevectors:

     1) Qvec within the y < 0 right quadrant;
   
     2) Qvec + smallest shift in the y-axis; */

  if (C3SYM)
    {
      /*........ | Hex-edge
	K-Point: | corner */
      
      iVec[0] = Qm1 + l2;
      jVec[0] = Qm2 + l2;

      /*........... | Hex-side
	M-Point(1): | mid-point */
    
      iVec[1] = l2;
      jVec[1] =  0;

      /*........... | Hex-side
	M-Point(2): | mid-point */
      
      iVec[2] =  0;
      jVec[2] = l2;

      /*........ | Center
	G-Point: | point */
      
      iVec[3] = l2;
      jVec[3] = l2;
    }
  else // Square geometry ...
    {
      /*........ | Square
	M-Point: | edge corner */
      
      iVec[0] = 0;
      jVec[0] = 0;

      /*........ | X-side
	X-Point: | mid-point */

      iVec[1] =  0;
      jVec[1] = l2;
      
      /*........ | Y-side
	Y-Point: | mid-point */      
      
      iVec[2] = l2;
      jVec[2] =  0;

      /*........ | Center
	G-Point: | point */
      
      iVec[3] = l2;
      jVec[3] = l2;
    }

  /* -----------------------------
     Calculate intensity ratio &
     transfer SQ1 value to vector: */

  int k, i, j;
  
  for (k = 0; k < 4; k++)
    {
      i = iVec[k];
      j = jVec[k];
      
      SpecVec[k] = specField[i][j];
    }
}

//======================================
// Test code for the rotation procedure: 

void test_reflectionProc(double **spinField)
{
  Vec3d locField, spinVec, spinVecNew;

  double a1, a2, w;

  ofstream outfile;
    
  outfile.open(outDir1
	       + "reflect_test.dat");

  for (int i = 0; i < Ns; i++)
    {  
      get_localSpin(i, spinField, spinVec);

      get_localField(i, spinField, locField);

      reflect_aboutVec(spinVec, locField, spinVecNew);

      set_localSpin(i, spinVecNew, spinField);

      w = dotProduct(spinVec, locField)
	- dotProduct(spinVecNew, locField);

      a1 = atan2(spinVec[1], spinVec[0]);

      a2 = atan2(spinVecNew[1], spinVecNew[0]);
      
      outfile << setw(3) << i << "  "	
	      << fmtDbleFix(w , 8, 15) << "  "
	      << fmtDbleFix(a1, 8, 15) << "  "
	      << fmtDbleFix(a2, 8, 15) << endl;
    }
  
  outfile.close();
}

//======================================
// Test code for the rotation procedure: 

void test_rotationProc()
{
  double drand1, drand2, tht, phi, RMat[9];
	    
  Vec3d oldVec, newVec, refVec;   

  ofstream outfile;

  outfile.open(outDir1
	       + "rotation_test.dat");
    
  /*.......................
    Define initial vectors: */

  drand1 = dSFMT_getrnum();
  drand2 = dSFMT_getrnum();
	  
  tht = 2.0 * pi * drand1;
      
  phi = acos(1.0 - 2.0 * drand2);
	      	      
  refVec[0] = sin(phi) * cos(tht);
  refVec[1] = sin(phi) * sin(tht); 
  refVec[2] = cos(phi);

  tht = tht + pi;
	      
  oldVec[0] = sin(phi) * cos(tht);
  oldVec[1] = sin(phi) * sin(tht); 
  oldVec[2] = cos(phi);  
    
  /*.........................
    Get rotation matrix: RMat

    |refVec| = 1 (normalized)
       
    RMat * refVec = (0,0,1);

    The matrix align the reference unit
    vector with the z-axis as shown above; */

  get_rotZ2VMat(refVec, RMat);
    
  newVec = MxVecProduct(RMat, oldVec);

  outfile << " refVec = { "
	  << refVec[0] << " , "
	  << refVec[1] << " , "
	  << refVec[2] << " } ";

  outfile << "\n" << endl;
    
  outfile << " oldVec = { "
	  << oldVec[0] << " , "
	  << oldVec[1] << " , "
	  << oldVec[2] << " } ";

  outfile << "\n" << endl;
	
  outfile << " newVec = { "
	  << newVec[0] << " , "
	  << newVec[1] << " , "
	  << newVec[2] << " } " << endl;

  outfile.close();
}

//=======================================
// Test code for the heat-bath procedure: 

void test_heatBath()
{    
  Vec3d locField = {0.0, 0.0, 1.0};

  double chi, bt, drand1;

  int np = 50000;

  ofstream outfile;

  /*............
    High T test: */
    
  bt = 1.0;
    
  outfile.open(outDir1
	       + "heatBath_highT_test.dat");    

  for (int i = 0; i < np; i++)
    {
      drand1 = dSFMT_getrnum();
    
      get_hBathSample(drand1, bt, locField, chi);

      outfile << chi << endl;
    }
    
  outfile.close();

  /*...........
    Low T test: */
    
  bt = 8.0;
     
  outfile.open(outDir1
	       + "heatBath_lowT_test.dat");    

  for (int i = 0; i < np; i++)
    {
      drand1 = dSFMT_getrnum();
    
      get_hBathSample(drand1, bt, locField, chi);

      outfile << chi << endl;
    }
    
  outfile.close();
}

#if WITH_OPENCV == 1
////////////////////

//===================================================
// Subroutine for callback feature from mouse events:

void onMouse(int evt, int x, int y, int flags, void* param)
{
  Point click;
  
  if (evt == EVENT_LBUTTONDOWN)
    {
      click = Point(x, y);

      xmouse = x;
      ymouse = y;

      cout << " -- Pixel coordinates (x,y) : "
	   << xmouse << " , " << ymouse << endl;
    }
}

//========================================
// Interpolate between two values: Ac & Bc

double interpolFunc(double Ac, double Bc,
		    double x1, double x2,
		    double xval)
{
  double x = 10.0 * ( (xval - x1) / (x2 - x1) ) - 5.0;
    
  return ( Ac - (Ac - Bc) * ( 0.5 * (tanh(0.5 * x) + 1.0) ) );
}

//========================================
// Linear interpolation between Ac and Bc

double linearInterp(double Ac, double Bc,
                    double x1, double x2,
                    double xval)
{
  double t = (xval - x1) / (x2 - x1);
  
  return Ac + t * (Bc - Ac);
}

//===========================================
// Return the site number in row-major order:
// (numbering goes row-by-row on the lattice)

Vec3b polarColourBGR(double norm, double theta, int idx)
{
  const unsigned int B1 =  20;
  const unsigned int G1 = 220;
  const unsigned int R1 =  20;
  
  const unsigned int B2 =  10;
  const unsigned int G2 =  10;
  const unsigned int R2 =  10;

  const unsigned int B3 = 220;
  const unsigned int G3 =  20;
  const unsigned int R3 =  20;

  const unsigned int B4 =  20;
  const unsigned int G4 =  20;
  const unsigned int R4 = 220;
  
  /*--------------------------
    Define auxiliary variables */ 

  double tht = (theta >= 0.0 ? theta : pi2 + theta); 

  double fac = clamp(norm, 0.0, 1.0);

  /*------------------------------------------
    Define color based on inputs: norm & theta */
  
  Vec3b colourVec;

  if (idx == 4)
    {
      const double t0 = 0.0 * a90;
      const double t1 = 1.0 * a90;
      const double t2 = 2.0 * a90;
      const double t3 = 3.0 * a90;
      const double t4 = 4.0 * a90;
  
      if ((tht >= t0) && (tht < t1))
	{
	  colourVec[0] = fac * interpolFunc(B1, B2, t0, t1, tht);
	  colourVec[1] = fac * interpolFunc(G1, G2, t0, t1, tht);
	  colourVec[2] = fac * interpolFunc(R1, R2, t0, t1, tht);
	}
      else if ((tht >= t1) && (tht < t2))
	{
	  colourVec[0] = fac * interpolFunc(B2, B3, t1, t2, tht);
	  colourVec[1] = fac * interpolFunc(G2, G3, t1, t2, tht);
	  colourVec[2] = fac * interpolFunc(R2, R3, t1, t2, tht);
	}
      else if ((tht >= t2) && (tht < t3))
	{
	  colourVec[0] = fac * interpolFunc(B3, B4, t2, t3, tht);
	  colourVec[1] = fac * interpolFunc(G3, G4, t2, t3, tht);
	  colourVec[2] = fac * interpolFunc(R3, R4, t2, t3, tht);
	}
      else if ((tht >= t3) && (tht < t4))
	{
	  colourVec[0] = fac * interpolFunc(B4, B1, t3, t4, tht);
	  colourVec[1] = fac * interpolFunc(G4, G1, t3, t4, tht);
	  colourVec[2] = fac * interpolFunc(R4, R1, t3, t4, tht);
	}
    }
  else//( C3SYM : idx == 3 )
    {      
      const double t0 = 0.0 * a120;
      const double t1 = 1.0 * a120;
      const double t2 = 2.0 * a120;
      const double t3 = 3.0 * a120;

      tht = tht + a30;

      if ((tht >= t0) && (tht < t1))
	{
	  colourVec[0] = fac * interpolFunc(B1, B3, t0, t1, tht);
	  colourVec[1] = fac * interpolFunc(G1, G3, t0, t1, tht);
	  colourVec[2] = fac * interpolFunc(R1, R3, t0, t1, tht);
	}
      else if ((tht >= t1) && (tht < t2))
	{
	  colourVec[0] = fac * interpolFunc(B3, B4, t1, t2, tht);
	  colourVec[1] = fac * interpolFunc(G3, G4, t1, t2, tht);
	  colourVec[2] = fac * interpolFunc(R3, R4, t1, t2, tht);
	}
      else if ((tht >= t2) && (tht < t3))
	{
	  colourVec[0] = fac * interpolFunc(B4, B1, t2, t3, tht);
	  colourVec[1] = fac * interpolFunc(G4, G1, t2, t3, tht);
	  colourVec[2] = fac * interpolFunc(R4, R1, t2, t3, tht);
	}
      else//( fill colour gap )
	{
	  colourVec[0] = fac * interpolFunc(B4, B1, t2, t3, tht);
	  colourVec[1] = fac * interpolFunc(G4, G1, t2, t3, tht);
	  colourVec[2] = fac * interpolFunc(R4, R1, t2, t3, tht);
	}
    }
  
  return colourVec;
}

//====================================================
// Compute energy (Hamiltonian) for a given spin-field
// & return the distribution of energy in the system:

/* Similar to 'get_energyValue' with the inclusion of
   the output 'energyMat' that stores the energy
   value calculated for each site:

   Site-vector : Point(i,j); */

void get_energyMat(double **spinField,
		   double &HamE, Mat &EMat)
{
  const int dble_Ns = (double)Ns; 
  
  unsigned int i, j, k, n;

  double energy0, energy1, eDens;

  Vec3d locField; Point rsite; 
  
  HamE = 0.0; //(Hamiltonian energy)
    
  for (i = 0; i < Lsz; i++)
    {     
      for (j = 0; j < Lsz; j++)
	{	  
	  k = i * Lsz + j; //[ siteNumber(i,j) ]

	  // Interaction of each spin in the
	  // lattice with its local field:
	  
	  get_localField(k, spinField, locField);
	
	  energy0 = 0.0;
	
	  for (n = 0; n < 3; n++)
	    {
	      energy0 = energy0
		+ spinField[k][n] * locField[n];
	    }

	  energy0 = energy0 * 0.5; //(#)

	  // Interaction with the 
	  // external magnetic field:
	    
	  energy1 = 0.0;
	
	  if (extH_ON) 
	    {        
	      energy1 = (- extH) * spinField[k][dMag];
	    }

	  // The values are now assigned to the 
	  // corresponding output variables:

	  eDens = (energy0 + energy1) / dble_Ns; //(#)
   
	  HamE += eDens;
	  
	  rsite = Point(i,j);

	  EMat.at<double>(rsite) = eDens;
	}
      
      /* # : we could have used 'Ns' directly in the
	 division since C++ automaticallt converts
	 the lower rank number to the higher one
	 involved oin the operation; */
    }
}

//=====================================================
// Generate a lattice Mat-form of the input spin field:

void spinField2Mat(double **sField, Mat &sMat)
{
  int i, j, k;

  Point rsite;
  
  Vec3d spinVec;

  //-----------------
  // Make Mat-object:
  
  for (i = 0; i < Lsz; i++)
    {	     
      for (j = 0; j < Lsz; j++)
	{
	  k = i * Lsz + j;
	      
	  rsite = Point(i,j);
	    
	  spinVec[0] = sField[k][0];
	  spinVec[1] = sField[k][1];
	  spinVec[2] = sField[k][2];   	      

	  sMat.at<Vec3d>(rsite) = spinVec;
	}
    }
}

//========================================================
// Make projection map of the spin system in the xy-plane:

/* Notes: color-coding varies from Blue to Green 
   and to Red (BGR) at each 120° rotation of the
   inplane spin, a darker pixel/circle indicates
   a greater amplitude of the z-component, i.e.,
   a black circle corresponds to a spin with no
   xy-plane projection.

   When converting to an image pixel with OpenCV
   type 'Vec3b', one can simply take the 'Vec3d'
   produced by the code below and multiply by
   the factor 255, that is:

   Vec3d smapVec; (3-components of type double)
   |
   |---> smapVec[n]: min = 0 , max = 1;

   Vec3b pixel; (3-channels with uint. values);
   |
   |---> pixel[n]: min = 0 , max = 255;
   
   pixel = Vec3b(255 * smapVec[0],  ---> Blue
   .             255 * smapVec[1],  ---> Green
   .             255 * smapVec[2]); ---> Red

   The object 'pixel' literally constitutes one
   pixel of a 3-channel type image (Mat: CV_8UC3); */

void spinMat_map2d(const Mat &spinMat, Mat &sMap)
{
  unsigned int i, j;

  double x0, y0, z0;

  double theta, vecNorm;
  
  Vec3d spinVec, smapVec;
    
  Point rsite;

  //---------------------
  // Mapping to xy-plane:
  
  for (i = 0; i < Lsz; i++)
    {	     
      for (j = 0; j < Lsz; j++)
	{
	  rsite = Point(i,j);

	  spinVec = spinMat.at<Vec3d>(rsite);
	      
	  theta = atan2(spinVec[1],  // Angle (radians):
			spinVec[0]); // [-pi , pi];

	  vecNorm = sqrt(pow(spinVec[0], 2) +
			 pow(spinVec[1], 2));

	  x0 = 0.5 * (cos(theta - Qval * 1.0) + 1.0);
	  y0 = 0.5 * (cos(theta - Qval * 2.0) + 1.0);
	  z0 = 0.5 * (cos(theta - Qval * 3.0) + 1.0);
	  
	  smapVec[0] = vecNorm * pow(x0, 2); // B
	  smapVec[1] = vecNorm * pow(y0, 2); // G
	  smapVec[2] = vecNorm * pow(z0, 2); // R

	  sMap.at<Vec3d>(rsite) = smapVec;
	}
    }
}

//=======================================
// Plot lattice grid to image file (PNG):

/* Warning: the global variable 'plotSize' &
   -------- the global pointer  'imgSites' 
   -------- must be defined in code previouly; */

void make_latticeGrid(Mat &gridMat, string plotName)
{
  const int idx = (C3SYM) ? 3 : 4;
  
  const Vec3b v1color = polarColourBGR(1.0, 0.0, idx);  
  const Vec3b v2color = polarColourBGR(1.0, a90, idx);
  
  const double ptRad = floor(0.8 * ptRadius);

  const double dkMid = 0.75 * ptRad;

  const double ipRad = 4.50 * ptRad;

  const double minSz = 1.50 * ptRad;

  const double vSz = 4.0 * minSz;

  const double diskRad = 0.8 * vSz;

  const double astep = pi2 / 1000.0;

  const string xyzTag[3] = {"x","y","z"};
 
  int i, j, n, i1, i2, j1, j2, n1, n2;
      
  int v0, v1, v2, ncols, nrows;

  Point pt, vecBase, vecHead;
  
  Point P1, P2; Vec3b icolor;

  double angle;

  //------------------
  // Create frame/mat:

  gridMat = Mat(plotSize, CV_8UC3, whiteFull);
  
  //------------------------
  // Draw reference vectors:  

  v0 = pltSeq[0]; // Inplane spin
  v1 = pltSeq[1]; // components &
  v2 = pltSeq[2]; // transverse one;

  if (geom == "square" || geom == "lieb")
    {
      n1 = gridMat.cols - round(0.5 * gridSpac);
      n2 = gridMat.rows - round(0.5 * gridSpac);
    }
  else // Triangular-like geometries ...
    {
      n1 = gridMat.cols - round(0.5 * gridSpac);
      
      n2 = round(1.5 * gridSpac);
    }

  vecBase = Point(n1, n2);

  /*............................................*/
  
  i1 = n1 - round(1.0 * vSz);
  i2 = i1 - round(0.6 * vSz);
    
  vecHead = Point(i1, n2); pt = Point(i2, n2);

  arrowedLine(gridMat, vecBase, vecHead,
	      v1color, vecLwd1, LINE_AA, 0, tipSz);
  
  putText(gridMat, xyzTag[v0],
	  pt, FONT_HERSHEY_SIMPLEX,
	  0.025 * vSz, v1color, 1, LINE_AA);

  /*............................................*/
  
  j1 = n2 - round(1.0 * vSz);
  j2 = j1 - round(0.3 * vSz);
   
  vecHead = Point(n1,j1); pt = Point(n1,j2);
 
  arrowedLine(gridMat, vecBase, vecHead,
	      v2color, vecLwd1, LINE_AA, 0, tipSz);
  
  putText(gridMat, xyzTag[v1],
	  pt, FONT_HERSHEY_SIMPLEX,
	  0.025 * vSz, v2color, 1, LINE_AA);

  //-------------------------
  // Draw reference z-vector:

  if (geom == "square" || geom == "lieb")
    {
      n1 = gridMat.cols - round(0.5 * gridSpac);
      n2 = gridMat.rows - round(0.5 * gridSpac);

      n2 += (- 2 * gridSpac);
    }
  else // Triangular-like geometries ...
    {
      n1 = gridMat.cols - round(0.5 * gridSpac);
      
      n2 = round(3.5 * gridSpac);
    }    

  vecBase = Point(n1,n2);
  
  j1 = n2 - round(1.0 * vSz);
  j2 = j1 - round(0.3 * vSz);
   
  vecHead = Point(n1,j1); pt = Point(n1,j2);
 
  arrowedLine(gridMat, vecBase, vecHead,
	      magColour, vecLwd1, LINE_AA, 0, tipSz);
  
  putText(gridMat, xyzTag[v2],
	  pt, FONT_HERSHEY_SIMPLEX,
	  0.025 * vSz, magColour, 1, LINE_AA);

  //---------------------------
  // Draw reference color disk:

  if (geom == "square" || geom == "lieb")
    {
      n1 = gridMat.cols - round(0.5 * gridSpac);
      n2 = gridMat.rows - round(0.5 * gridSpac);

      n1 += round(- 2.0 * gridSpac);
    }
  else // Triangular-like geometries ...
    {
      n1 = gridMat.cols - round(2.0 * gridSpac);
      
      n2 = round(1.5 * gridSpac);
    }

  P1 = Point(n1,n2);
  
  for (i = 0; i < 1000; i++)
    {
      angle = i * astep - pi; 
      
      icolor = polarColourBGR(1.0, angle, idx);            
      
      n1 = round(diskRad * cos(angle));
      n2 = round(diskRad * sin(angle));
      
      P2 = P1 - Point(n1, n2);

      line(gridMat, P1, P2,
	   icolor, gridLwd, LINE_AA);
    }

  circle(gridMat, P1, dkMid,      // Disk
	 lightGray, -1, LINE_AA); // center;
  
  //---------------------------
  // Draw site points on frame:
  
  for (i = 0; i < Ns; i++)
    {
      P1 = imgSites[i];

      circle(gridMat, P1, ptRad, lightGray, -1, LINE_AA);
	      
      if (disOrder)
	{		  
	  if (impField[i] == 1)
	    {	      
	      circle(gridMat, P1, ipRad, lightGray, +4);
	    }
	}
    }

  if (multiSubs)
    {      
      for (i = 0; i < Ns0; i++)
	{
	  P1 = ir0Sites[i];

	  circle(gridMat, P1, 0.5 * ptRad, lightGray, -1, LINE_AA);
	}
    }
  
  //-------------------------
  // Draw lattice line bonds:
  /*
    ...................
    Check PBC cut using
    
    if (norm(P2 - P1) > 2 * gridSpac):
    |
    | line(gridMat, P1, P2,
    | redColour, gridLwd, LINE_AA); */

  for (i = 0; i < Ns; i++)
    {
      P1 = imgSites[i];
              
      for (n = 0; n < Zn1; n++)
	{        
	  j = nbors1[i][n];

	  if (j >= 0)
	    {	   
	      P2 = imgSites[j];

	      if (norm(P2 - P1) < 2 * gridSpac)
		{
		  line(gridMat, P1, P2,
		       lightGray, gridLwd, LINE_AA);
		}
	    }///[ Real-neighbor check ]
	}
    }//// i-loop END;

  //----------------------------------------
  // Draw next-nearest neighbors line bonds:

  bool plotNNNB = false;

  if (plotNNNB)
    {
      for (i = 0; i < Ns; i++)
	{
	  P1 = imgSites[i];
              
	  for (n = 0; n < Zn2; n++)
	    {        
	      j = nbors2[i][n];

	      if (j >= 0)
		{	   
		  P2 = imgSites[j];

		  if (norm(P2 - P1) < 2 * gridSpac)
		    {
		      line(gridMat, P1, P2,
			   redColour, gridLwd, LINE_AA);
		    }
		}
	    }
	}
    }

  //-----------------------------------------
  // Resize 'gridMat' if too large for image
  // and then record result to PNG file:

  if (plotName != "SKIP")
    {
      unsigned int cnum = gridMat.cols;
      unsigned int rnum = gridMat.rows;
  
      if (cnum > 4096 || rnum > 4096)  // Too large, resizing ...
	{                              // (this is avoided by the code)
	  Mat work = gridMat.clone();

	  double fc = 4096.0 / max(cnum, rnum); // Scale factor;
        
	  resize(gridMat, work, Size(), fc, fc); 

	  imwrite(plotName, work);
	}
      else // Suitable size, resize not needed...
	{
	  imwrite(plotName, gridMat);
	}
    }
}

//=============================================
// Make vector-field representing system state:
// plot spins plane-projection + transverse
// component as 2d-vectors using an input
// grid Mat (lattice) as background ...

void make_vecField(string ftag1,
		   string ftag2, double **spinField, 
		   const Mat &gridMat, Mat &vecImage)
{  
  const double minSz = 1.5 * ptRadius;

  const int idx = (C3SYM) ? 3 : 4;

  int ncols, nrows, ishift, jshift;
  
  int i1, i2, j1, j2, n1, n2, v0, v1, v2;

  double vSz, vecSz, vecNorm, vecLwd;
  
  double dx, dy, dz, theta, tsz, ptSz;
  
  Point pt1, pt2, vecBase, vecHead;

  Point rsite, wvecBase, wvecHead;
  
  Vec3d spinVec;   // VEC-3D (double type)

  Vec3b vecColour; // VEC-3B (integer type)

  //--------------
  // Create frame: (clone input grid)

  Mat Frame = gridMat.clone();

  vecSz = vecSz1;

  vecLwd = vecLwd1;

  //------------------------------------
  // Extract xy|yz plane vectors & plot:

  v0 = pltSeq[0];
  v1 = pltSeq[1];
  v2 = pltSeq[2];
  
  for (int k = 0; k < Ns; k++)
    {      
      /* Get spin vector at the site k */

      get_localSpin(k, spinField, spinVec);
      
      vecNorm = sqrt( pow(spinVec[v0], 2) +  // xy|yz
		      pow(spinVec[v1], 2) ); // plane norm;

      /* Define spin vector colour */

      theta = atan2(spinVec[v1], spinVec[v0]); // [-pi, pi]

      if (IsiModel)
	{
	  if (theta > 0.0)
	    {
	      vecColour = Vec3b(fc100Gray[0],
				fc100Gray[1],
				fc100Gray[2]); }
	  else
	    { vecColour = Vec3b(magColour[0],
				magColour[1],
				magColour[2]); }
	}
      else//( Heisenberg model | 3-component spins )
	{
	  // theta += a45; //( tilt vectors by 45º with this )
	  
	  vecColour = polarColourBGR(vecNorm, theta, idx);
	}
	  
      /* Define inplane arrow representation */

      // theta += pi; // Change interval to [0, pi2] and
      // theta += pi; // rotate for better visualization; 
      //
      // if (IsiModel){ theta += a45; } // Add 45° tilt;

      vSz = vecSz * vecNorm;
	
      dx = vSz * cos(theta);
      dy = vSz * sin(theta);

      ishift = round(dx);
      jshift = round(dy);

      rsite = imgSites[k]; 

      n1 = rsite.x;
      n2 = rsite.y;
	
      i1 = n1 + ishift; j1 = n2 + jshift;
      i2 = n1 - ishift; j2 = n2 - jshift;

      vecBase = Point(i1,j1);
      vecHead = Point(i2,j2);

      /* Define z|x arrow representation */

      dz = vecSz * spinVec[v2];

      jshift = round(dz);
	
      i1 = n1; j1 = n2 + jshift;
      i2 = n1; j2 = n2 - jshift;

      wvecBase = Point(i1,j1);
      wvecHead = Point(i2,j2);

      /* Draw the spin-field local value repre-
	 sentation as a coloured arrow/circle: */
      
      if (IsiModel)
	{
	  ptSz = 1.25 * vSz;

	  vector<Point> diamond
	    {
	      Point(n1, n2 - ptSz),  // top
	      Point(n1 + ptSz, n2),  // right
	      Point(n1, n2 + ptSz),  // bottom
	      Point(n1 - ptSz, n2)   // left
	    };

	  fillConvexPoly(Frame, diamond, vecColour, LINE_AA);
	  
	  // circle(Frame, Point(n1,n2),
	  // ptSz, vecColour, -1, LINE_AA);
	}
      else//( Heisenbeg model )
	{      
	  if (vSz > minSz){
	    arrowedLine(Frame, vecBase, vecHead,
			vecColour, vecLwd, LINE_AA, 0, tipSz);}

	  if (abs(dz) > minSz){
	    arrowedLine(Frame, wvecBase, wvecHead,
		    magColour, vecLwd, LINE_AA, 0, tipSz);}
	}
      
    }//// k-loop (END)

  n1 = round(0.5 * gridSpac);
      
  n2 = Frame.rows - n1;
    
  pt1 = Point(n1, n1);
  pt2 = Point(n1, n2);

  tsz = 0.00075 * min(Frame.rows, 2048);
  
  putText(Frame, ftag1, pt1,
	  FONT_HERSHEY_DUPLEX,
	  tsz, Scalar(0,0,0), 1, LINE_AA);

  putText(Frame, ftag2, pt2,
	  FONT_HERSHEY_DUPLEX,
	  tsz, Scalar(0,0,0), 1, LINE_AA);

  //----------------------------------------
  // Resize 'Frame' if too large for image &
  // clone result to output Mat (vecImage):
 
  unsigned int cnum = Frame.cols;
  unsigned int rnum = Frame.rows;
  
  if (cnum > 4096 || rnum > 4096) // Too large, resizing...
    {
      Mat work = Frame.clone();

      double fc = 4096.0 / max(cnum, rnum); // Scale factor;
        
      resize(Frame, work, Size(), fc, fc);

      vecImage = work.clone();
    }
  else
    { vecImage = Frame.clone(); }
}

//==========================================
// Make order-parameter map: plot 2d-vectors
// using an input grid Mat as background ...

void make_vecMap(string ftag1,
		 string ftag2, Vec2d *OrderMap, 
		 const Mat &gridMat, Mat &vecImage)
{
  const int idx = (C3SYM) ? 3 : 4;
    
  int ncols, nrows, ishift, jshift;
  
  int i1, i2, j1, j2, n1, n2, v0, v1, v2;

  double vSz, vecSz, vecNorm, vecLwd;
  
  double dx, dy, dz, theta, tsz, ptSz;
  
  Point vecBase, vecHead, vecMidl;

  Point rsite, pt1, pt2;
  
  Vec3b vecColour; // VEC-3B (integer type)

  //--------------
  // Create frame: (clone input grid)

  Mat Frame = gridMat.clone();

  vecSz = vecSz1;

  vecLwd = vecLwd1;

  //----------------------------------
  // Plot vectors from the input list:
  
  for (int k = 0; k < Ns; k++)
    {      
      /* Get spin vector at the site k */
    
      Vec2d pvec = OrderMap[k];

      /* Define vector colour */

      vecNorm = sqrt( pow(pvec[0], 2) + 
		      pow(pvec[1], 2) );
      
      theta = atan2(pvec[1], pvec[0]); // [-pi, pi]
      
      vecColour = polarColourBGR(1.0, theta, idx);
	  
      /* Define inplane arrow representation */

      theta += pi; // Change interval to [0, pi2] and
      theta += pi; // rotate for better visualization;

      vSz = vecSz * vecNorm;
	
      dx = vSz * cos(theta);
      dy = vSz * sin(theta);

      ishift = round(dx);
      jshift = round(dy);

      rsite = imgSites[k]; 

      n1 = rsite.x;
      n2 = rsite.y;
	
      i1 = n1 + ishift; j1 = n2 + jshift;
      i2 = n1 - ishift; j2 = n2 - jshift;

      vecBase = Point(i1,j1);
      vecHead = Point(i2,j2);
      vecMidl = Point(n1,n2);

      /* Draw the order-parameter local value
	 representation as a coloured circle: */

      ptSz = 1.25 * vSz;
	  
      circle(Frame, vecMidl,
	     ptSz, vecColour, -1, LINE_AA);
   
    }//// k-loop (END)

  n1 = round(0.5 * gridSpac);
      
  n2 = Frame.rows - n1;
    
  pt1 = Point(n1, n1);
  pt2 = Point(n1, n2);

  tsz = 0.00075 * min(Frame.rows, 2048);
  
  putText(Frame, ftag1, pt1,
	  FONT_HERSHEY_DUPLEX,
	  tsz, Scalar(0,0,0), 1, LINE_AA);

  putText(Frame, ftag2, pt2,
	  FONT_HERSHEY_DUPLEX,
	  tsz, Scalar(0,0,0), 1, LINE_AA);

  //----------------------------------------
  // Resize 'Frame' if too large for image &
  // clone result to output Mat (vecImage):
 
  unsigned int cnum = Frame.cols;
  unsigned int rnum = Frame.rows;
  
  if (cnum > 4096 || rnum > 4096) // Too large, resizing...
    {
      Mat work = Frame.clone();

      double fc = 4096.0 / max(cnum, rnum); // Scale factor;
        
      resize(Frame, work, Size(), fc, fc);

      vecImage = work.clone();
    }
  else
    { vecImage = Frame.clone(); }
}

//===============
// Make sign-map:

void make_signMap(string ftag1,
		  string ftag2, double *dbleMap, 
		  const Mat &gridMat, Mat &vecImage)
{  
  int ncols, nrows, k, n1, n2;
 
  double ptSz, txSz, EVal;

  Vec3b ptColour, colourVec;

  Point rsite, pt1, pt2;

  //--------------
  // Create frame: (clone input grid)

  Mat Frame = gridMat.clone();

  ptSz = 1.25 * vecSz1;

  //----------------------------------
  // Plot circles from the input list:
 
  for (k = 0; k < Ns; k++)
    {
      rsite = imgSites[k];
      
      EVal = dbleMap[k];
      
      if (EVal > 0.0)
	{
	  colourVec = scalarToVec3b(bluColour);
	}
      else if (EVal < 0.0)
	{
	  colourVec = scalarToVec3b(redColour);
	}
      else//(  Eval = 0.0 )
	{
	  colourVec = scalarToVec3b(blackFull);
	}
      
      circle(Frame, Point(rsite.x,rsite.y),
	     ptSz, colourVec, -1, LINE_AA);
   
    }//// k-loop (END)

  n1 = round(0.5 * gridSpac);
      
  n2 = Frame.rows - n1;
    
  pt1 = Point(n1, n1);
  pt2 = Point(n1, n2);

  txSz = 0.00075 * min(Frame.rows, 2048);
  
  putText(Frame, ftag1, pt1,
	  FONT_HERSHEY_DUPLEX,
	  txSz, Scalar(0,0,0), 1, LINE_AA);

  putText(Frame, ftag2, pt2,
	  FONT_HERSHEY_DUPLEX,
	  txSz, Scalar(0,0,0), 1, LINE_AA);

  //----------------------------------------
  // Resize 'Frame' if too large for image &
  // clone result to output Mat (vecImage):
 
  unsigned int cnum = Frame.cols;
  unsigned int rnum = Frame.rows;
  
  if (cnum > 4096 || rnum > 4096) // Too large, resizing...
    {
      Mat work = Frame.clone();

      double fc = 4096.0 / max(cnum, rnum); // Scale factor;
        
      resize(Frame, work, Size(), fc, fc);

      vecImage = work.clone();
    }
  else
    { vecImage = Frame.clone(); }
}

//====================================================
// Make lattice figure & interactively find nearest &
// next-nearest neighbors based on user's mouse input:

/* ......
   Notes: the 1st part of the code also makes a figure
   (PNG image) of the lattice sites with their number
   or labels, then, if the interactive feature is ena-
   bled by the user, there is a part of the code which
   finds and shows the neighbors based on the user's
   mouse click upon the lattice image; */
  
void make_latticeFigure(int &flag)
{
  string outImg0, outImg1;

  string winName, isite;

  int i, j, k, n, m0, n0, n1, n2;

  int nx, ny, nPlus, stnum, ch0;

  int radius, ncols, nrows;

  /*----------------------------------------------
    Set size-factor for main or auxiliary lattice

    Hexagonal, Kagome and Lieb lattices are created
    from an auxiliary lattice with bigger lattice
    spacing (size-factor below), so that the 
    final lattice has spacing equal to 1;

    spc : lattice spacing (for the image); */
  
  double szFac, spc;

  if (geom == "square" || geom == "triang")
    {
      szFac = 1.0; spc = 60.0;
    }
  else if (geom == "hexagn")
    {
      szFac = sq3; spc = 60.0;
    }
  else if ( geom == "kagome")
    {
      szFac = 2.0; spc = 60.0;
    }
  else if ( geom == "lieb" )
    {
      szFac = sq2; spc = 60.0;
    }
  
  /*................................... 
    Description of the variables below:
    
    1) Margin values to guarantee for the image;
    2) Point radius (size of the site circles);
    3) Integers ncols & nrows define the image size; */
    
  nPlus = 3 * round(spc); //(1.1)
  
  n0 = round(1.5 * spc);  //(1.2)
  
  if (geom == "square" ||     
      geom == "triang" ||  geom == "lieb") //(1.3) & (2)
    {       
      m0 = round(0.2 * spc); radius = 4; }
  else
    { m0 = round(0.1 * spc); radius = 6; }
  
  if (J2_ON || JX_ON)
    {
      n0 = 2 * n0;

      nPlus = 2 * nPlus;
    }

  ncols = round(spc * rvecList[Ns - 1][0]) + nPlus; //(3)
  nrows = round(spc * rvecList[Ns - 1][1]) + nPlus;
   
  Size szFrame(ncols, nrows);

  Mat Lattice(szFrame, CV_8UC3); // Image initialization;
 
  //-------------------------
  // Generate lattice points: (loop over i,j)
  
  double ptRad = floor(0.9 * radius);

  double dx, dy, angle;

  Point pt1(0,0), pt0(0,0);

  vector<Pts3d> siteList;
  
  Pts3d sitePoint;

  Scalar stColor;
  
  stColor = whiteFull; //( Default lattice site color )
    
  for (k = 0; k < Ns; k++)
    {	  
      nx = n0 + round(spc * rvecList[k][0]);
      ny = n0 + round(spc * rvecList[k][1]);

      isite = to_string(k);

      if (disOrder)
	{		  
	  if (impField[k] == 1)
	    {
	      stColor = ylwColour; }
	  else
	    { stColor = whiteFull; }
	}

      pt0 = Point(nx, ny);

      pt1 = Point(nx + m0, ny - m0);

      circle(Lattice, pt0, radius, stColor, -1, LINE_AA);
	      
      putText(Lattice, isite, pt1,
	      FONT_HERSHEY_SIMPLEX,
	      txtsz, stColor, 1, LINE_AA);

      sitePoint = Pts3d(nx, ny, k);
		
      siteList.push_back(sitePoint);
    }

  //--------------------------
  // Plot aux. lattice points:
  
  if (multiSubs)
    {
      for (k = 0; k < Ns0; k++)
	{      
	  nx = n0 + round(spc * r0List[k][0]);
	  ny = n0 + round(spc * r0List[k][1]);

	  isite = to_string(k);

	  pt0 = Point(nx, ny);

	  pt1 = Point(nx + m0, ny - m0);

	  circle(Lattice, pt0, radius, magColour, -1, LINE_AA);
	      
	  putText(Lattice, isite, pt1,
		  FONT_HERSHEY_SIMPLEX,
		  txtsz, magColour, 1, LINE_AA);
	}
    }
  
  //------------------------------
  // Plot DFT-grid lattice points:

  if (with_DFTcodes)
    {
      for (k = 0; k < Nsg; k++)
	{        
	  nx = n0 + round(spc * dftGrid[k][0]);
	  ny = n0 + round(spc * dftGrid[k][1]);

	  pt0 = Point(nx, ny);

	  pt1 = Point(nx + m0, ny - m0);

	  circle(Lattice, pt0, 2, ylwColour, -1, LINE_AA);
	}
    }
  
  /*..........................
    Get total number of sites:
    numSites = Ns            */

  size_t listSz = siteList.size();

  int numSites = int(listSz);
  
  //----------------------------------------------------
  // Extend lattice (PBC: periodic boundary conditions):
  
  if (!PBC_OFF)
    {
      Vec2d a1 = spc * avec1; // Bravais vectors re-
      Vec2d a2 = spc * avec2; // scaled for the image;
      
      Point rvec[8];

      int nx1, nx2;

      if (geom == "square" || geom == "lieb")
	{
	  nx1 = round(Lsz * szFac * spc);
	  nx2 = round(Lsz * szFac * spc);    
	  ny  = round(Lsz * szFac * spc);
	}
      else // Triangular-like geometries ...
	{
	  nx1 = round(Lsz * szFac * a1[0]);
	  nx2 = round(Lsz * szFac * a2[0]);
	  
	  ny  = round(Lsz * szFac * (a1[1] + a2[1])) - 1;
	}

      /*...................
	Make shift vectors: */
  
      rvec[0] = Point(+ nx1, 0);
      rvec[1] = Point(- nx1, 0);
  
      rvec[2] = Point(nx2, + ny);
      rvec[3] = Point(nx2, - ny);

      rvec[4] = Point(- nx2, + ny);
      rvec[5] = Point(- nx2, - ny);

      if (geom == "square" || geom == "lieb")
	{
	  rvec[6] = Point(0, + ny);
	  rvec[7] = Point(0, - ny);
	}
      else // Triangular-like geometries ...
	{
	  rvec[6] = Point(+ nx2 + nx1, + ny);
	  rvec[7] = Point(- nx2 - nx1, - ny);
	}

      /*........................
	Replicate lattice (PBC): */
  
      for (n = 0; n < numSites; n++)
	{
	  sitePoint = siteList[n];

	  stnum = sitePoint.z;

	  isite = to_string(stnum);

	  for (i = 0; i < 8; i++)
	    {
	      pt0 = rvec[i];
	  
	      n1 = pt0.x + sitePoint.x;
	      n2 = pt0.y + sitePoint.y;

	      siteList.push_back(Pts3d(n1, n2, stnum));

	      if ( (n1 > 0) && (n1 < ncols) &&
		   (n2 > 0) && (n2 < nrows) )
		{
		  pt0 = Point(n1, n2);
	      
		  circle(Lattice, pt0, radius, iceColour, -1, LINE_AA);

		  pt1 = Point(n1 + m0, n2 - m0);

		  putText(Lattice, isite, pt1,
			  FONT_HERSHEY_SIMPLEX,
			  txtsz, iceColour, 1, LINE_AA);
		}
	    }
	}

      /*.....................................
	Update numSites (add extended zones): */

      listSz = siteList.size();

      numSites = int(listSz);
    }
  
  //-------------------------
  // Preparing image to work:

  int ncl, nrw;
      
  ncl = ncols; // Current Mat's
  nrw = nrows; // cols and rows;

  if (ncols < minsz) ncl = minsz;
  if (nrows < minsz) nrw = minsz;
    
  if (ncols > maxsz) ncl = maxsz;
  if (nrows > maxsz) nrw = maxsz;

  Size winSize(ncl, nrw);

  outImg0 = outDir2 + "lattice.png";

  imwrite(outImg0, Lattice);
    
  /*------------------------------------
    Find site number from coordinates &
    draw/show its 1st and 2nd neighbors: */

  const double fc1 = 1.10;

  const double fc2 = 1.05;
  
  int i0, rArea1, rArea2, rArea3;

  int nbs, znum1, znum2, znum3;

  double w, w0, dval;

  rArea1 = floor(spc * fc1);

  rArea3 = floor(2.0 * spc * fc2);
  
  if (geom == "square" || geom == "lieb")
    {
      rArea2 = floor(sq2 * spc * fc2);
    }
  else // Triangular-like geometries ...
    {
      rArea2 = floor(sq3 * spc * fc2);      
    }

  if (vision) // BARRIER
    {	
      /*...................................
        Create persistent window only once: */

      winName = "Interactive lattice";

      namedWindow(winName, WINDOW_NORMAL);
      
      resizeWindow(winName, winSize);

      /*.......................
	Set working Mat-object: */
      
      Mat baseImg;
      
      baseImg = Lattice.clone();
      
      for (k = 0; k < Ns; k++)
	{
	  xmouse = 0; // Reset mouse
	  ymouse = 0; // cooords;

	  setMouseCallback(winName, onMouse);

	  cout << " Pick a site on the lattice"
	       << " image and press enter:\n" << endl;

	  imshow(winName, Lattice); waitKey(0);

	  if ((xmouse == 0) || (ymouse == 0))
	    {
	      cout << " Cancelled...\n" << endl; break;
	    }
	
	  nx = xmouse; // Mouse coordinates
	  ny = ymouse; // for site search;

	  setMouseCallback(winName, //( disable callback now )
			   nullptr, nullptr); 

	  w0 = 2.0 * spc; n0 = 0;
	  
	  for (n = 0; n < numSites; n++)
	    {
	      sitePoint = siteList[n];

	      n1 = sitePoint.x;
	      n2 = sitePoint.y;

	      w = abs(n1 - nx) + abs(n2 - ny);

	      if (w < w0){ w0 = w; n0 = n; }
	    }

	  sitePoint = siteList[n0];

	  pt0 = Point(sitePoint.x,
		      sitePoint.y);
	  	        	
	  i0 = sitePoint.z;

	  znum1 = zvalList[i0].x;
	  znum2 = zvalList[i0].y;
	  znum3 = zvalList[i0].z;	  
    
	  cout << "\n Site located: " << i0 << endl;
	  
	  cout << "\n Number of 1st | 2nd | 3rd active neighbors: ";
	  
	  cout << znum1 << " | " << znum2 << " | " << znum3;

	  cout << "\n\n 1st neighbors: ";
	  
	  for (n = 0; n < Zn1; n++)
	    {
	      nbs = nbors1[i0][n];
	      
	      if (nbs < 0){ cout << "X" << X2; }
	      else        { cout << nbs << X2; }
	    }

	  if (J2_ON)
	    {
	      cout << "\n\n 2nd neighbors: ";
	      
	      for (n = 0; n < Zn2; n++)
		{
		  nbs = nbors2[i0][n];
		  
		  if (nbs < 0){ cout << "X" << X2; }
		  else        { cout << nbs << X2; }
		}
	    }

	  if (JX_ON)
	    {
	      cout << "\n\n 3rd neighbors: ";
	      
	      for (n = 0; n < ZnX; n++)
		{
		  nbs = nborsX[i0][n];
		  
		  if (nbs < 0){ cout << "X" << X2; }
		  else        { cout << nbs << X2; }
		}
	    }
	  
	  cout << endl;
	  
	  /*.............................
	    Draw search areas on Lattice: */
	  
	  circle(Lattice, pt0, rArea1, redColour, 1);

	  if (J2_ON)
	    { circle(Lattice, pt0, rArea2, grnColour, 1); }
	  
	  if (JX_ON)
	    { circle(Lattice, pt0, rArea3, bluColour, 1); }

	  cout << " \n Neighbors circles drawn, now"
	       << " press enter to continue...   \n" << endl;
	  
	  imshow(winName, Lattice); waitKey(0);
	
	  /*...............	    
	    Repeat or exit: */

	  cout << " Repeat process? (1/0) : ";

	  cin >> ch0; cout << endl; 
	  
	  if (ch0 == 0)
	    {
	      outImg1 = outDir2 + "neighbors_view.png";
	      
	      imwrite(outImg1, Lattice); break;
	    }
	  else//( reload clean image for next iteration )
	    {
	      baseImg = Lattice.clone();
	    }
	}

      destroyWindow(winName);
      
    }//// END BARRIER
}

//===============================================
// Subroutine to generate video frames from plots
// of the spectral forms using an external script:

void make_specPlots(string frmTag, string fInfo,
		    string pltTag, string fName, Mat &frame)
{
  /*........................
    Define auxiliary strings */
  
  string imgName = "plotFrame" + frmTag;

  string move0 = "mv " + outDir1 + subDir2;

  string cInit = "(cd " + outDir3 + " && ";
  
  string cBase = cInit + "./SpecPlot_PNG.sh ";

  string cTail = ") > /dev/null 2>&1";

  /*.........................................
    Define system move & plot command strings */

  string cMove = move0 + fName + X2;
	  
  string cPlot = cBase + fName + X2;

  cMove += outDir3 + fName;
  
  cPlot += geom + X2 + pltTag + X2;
  
  cPlot += imgName + fInfo + cTail;
	  
  /*.........................................
    Execute move & plot commands + read frame */

  string pngFile = outDir3 + imgName + ".png";
  
  int sys1 = system(cMove.c_str());
  int sys2 = system(cPlot.c_str());

  frame = imread(pngFile);
  
  /*..................................
    Remove copied DAT file & PNG image */

  string cRemvDat = "rm " + outDir3 + fName;
  
  string cRemvImg = "rm " + pngFile;

  int sys3 = system(cRemvDat.c_str());
  int sys4 = system(cRemvImg.c_str());
}

//==============================================
// Double 3-channel Mat ---> Mat in Color-scale:
// ( specific implementation: lattice system )

void dble3ch2Frame(const Mat &dbleFrame, Mat &Frame,
		   string ftag)
{
  const double spc = 12.0;
  const double rad =  6.0;

  const double tsz = 0.04 * Lsz;

  const Vec2d a1 = spc * avec1;
  const Vec2d a2 = spc * avec2;

  /*----------------------------------------*/
  
  const unsigned int cnum = dbleFrame.cols;

  const unsigned int rnum = dbleFrame.rows;

  const unsigned int n0 = round(1.5 * spc);

  const unsigned int nb = round(0.4 * spc);
  
  /*----------------------------------------*/
  
  unsigned int i, j, nx, ny, ncols, nrows;

  double pmax, pmin, pdiff, nfac, x0, y0, z0; 

  Point pt(-1, -1); Vec3b pixel;

  bool BLUR_ON = false; // Blurr effect;
  
  //-------------------
  // Define image size:
  
  ncols = (Lsz - 1) * (a1[0] + a2[0]) + 2 * n0;    
  nrows = (Lsz - 1) * (a1[1] + a2[1]) + 2 * n0;

  Size szFrame(ncols, nrows);

  //------------------------------------
  // Find min/max value & define 'nfac':

  pmax = 0.0; pmin = 0.0;
    
  minMaxLoc(dbleFrame, &pmin, &pmax, nullptr, nullptr);

  //-----------------------------
  // Define normalization factor:

  nfac = 255.0;

  pdiff = pmax - pmin;
	
  if (pdiff > dbleSmall){nfac = nfac / pdiff;}
  
  //-----------------
  // Conversion loop:

  Mat workFrm(szFrame, CV_8UC3, Scalar(255,255,255));

  Frame = workFrm.clone();

  for   (i = 0; i < cnum; i++){     
    for (j = 0; j < rnum; j++)
      {
	pixel = Vec3b(255,255,255); // White pixel;
		  
	pt = Point(i, j); // Source pixel coordinates;

	x0 = dbleFrame.at<Vec3d>(pt)[0] - pmin;
	y0 = dbleFrame.at<Vec3d>(pt)[1] - pmin;
	z0 = dbleFrame.at<Vec3d>(pt)[2] - pmin;

	pixel.val[0] = round(x0 * nfac);
	pixel.val[1] = round(y0 * nfac); 
	pixel.val[2] = round(z0 * nfac);

	nx = n0 + round(i * a1[0] + j * a2[0]);
	ny = n0 + round(i * a1[1] + j * a2[1]);

	pt = Point(nx, ny); // Target pixel coordinates;

	circle(workFrm, pt, rad, pixel, -1, LINE_AA);
      }
  }

  //------------------
  // Final procedures:
  
  if (BLUR_ON)
    {  
      GaussianBlur(workFrm, Frame, Size(nb,nb), 1.5);

      normalize(Frame, workFrm, 0, 255, NORM_MINMAX);

      Frame = workFrm.clone();
    }
  else
    {
      normalize(workFrm, Frame, 0, 255, NORM_MINMAX);      
    }

  nx = round(0.70 * ncols);
  ny = round(0.10 * nrows);
    
  pt = Point(nx, ny);
  
  putText(Frame, ftag, pt,
	  FONT_HERSHEY_COMPLEX_SMALL,
	  tsz, Scalar(0,0,0), 1, LINE_AA);
}

//==============================================
// Double 1-channel Mat ---> Mat in Color-scale:
// ( specific implementation: lattice system )

void dble1ch2Frame(const Mat &dbleFrame, Mat &Frame,
		   string ftag)
{
  const double spc = 12.0;
  const double rad =  6.0;

  const double tsz = 0.04 * Lsz;

  const Vec2d a1 = spc * avec1;
  const Vec2d a2 = spc * avec2;

  /*----------------------------------------*/
  
  const unsigned int cnum = dbleFrame.cols;

  const unsigned int rnum = dbleFrame.rows;

  const unsigned int n0 = round(1.5 * spc);

  const unsigned int nb = round(0.4 * spc);
  
  /*----------------------------------------*/
  
  unsigned int i, j, nx, ny, pv, ncols, nrows;

  double pmax, pmin, pdiff, nfac, x0;
  
  Point pt(-1, -1); Vec3b pixel;

  bool BLUR_ON = false; // Blurr effect;

  //-------------------
  // Define image size:
  
  ncols = (Lsz - 1) * (a1[0] + a2[0]) + 2 * n0;    
  nrows = (Lsz - 1) * (a1[1] + a2[1]) + 2 * n0;

  Size szFrame(ncols, nrows);

  //------------------------------------
  // Find min/max value & define 'nfac':

  pmax = 0.0; pmin = 0.0;
    
  minMaxLoc(dbleFrame, &pmin, &pmax, nullptr, nullptr);

  //-----------------------------
  // Define normalization factor:

  nfac = 255.0;

  pdiff = pmax - pmin;
	
  if (pdiff > dbleSmall){nfac = nfac / pdiff;}

  //-----------------
  // Conversion loop:

  Mat workFrm(szFrame, CV_8UC3, Scalar(255,255,255));

  Frame = workFrm.clone();

  for   (i = 0; i < cnum; i++){
    for (j = 0; j < rnum; j++)
      {
	pixel = Vec3b(255,255,255); // White pixel;
		  
	pt = Point(i, j); // Source pixel coordinates;

	x0 = dbleFrame.at<double>(pt) - pmin;

	pv = round(x0 * nfac);

	if (pv <= 85) // Blue domain;
	  {
	    pixel.val[0] = 3 * pv / 1;
	    pixel.val[1] = 3 * pv / 2;
	    pixel.val[2] = 3 * pv / 3;
	  }

	if (pv > 85 && pv <= 170) // Green domain;
	  {
	    pixel.val[0] = 255 - 3 * (pv - 85);
		
	    pixel.val[1] = 3 * pv / 2;
	    pixel.val[2] = 3 * pv / 3;
	  }	      

	if (pv > 170) // Red domain;
	  {	
	    pixel.val[1] = 255 - 3 * (pv - 170);
		  
	    pixel.val[2] = pv;
	  }
	      
	nx = n0 + round(i * a1[0] + j * a2[0]);
	ny = n0 + round(i * a1[1] + j * a2[1]);

	pt = Point(nx, ny); // Target pixel coordinates;

	circle(workFrm, pt, rad, pixel, -1, LINE_AA);
      }
  }

  //------------------
  // Final procedures:

  if (BLUR_ON)
    {  
      GaussianBlur(workFrm, Frame, Size(nb,nb), 1.5);

      normalize(Frame, workFrm, 0, 255, NORM_MINMAX);

      Frame = workFrm.clone();
    }
  else
    {
      normalize(workFrm, Frame, 0, 255, NORM_MINMAX);      
    }

  nx = round(0.70 * ncols);
  ny = round(0.10 * nrows);
    
  pt = Point(nx, ny);
  
  putText(Frame, ftag, pt,
	  FONT_HERSHEY_COMPLEX_SMALL,
	  tsz, Scalar(0,0,0), 1, LINE_AA);
}

#endif ///( WITH_OPENCV == 1)///

//======================================================
// Calculate temporal series of the momentum form needed
// for the dynamical structure factor calculations via
// the time-evolution of the input spin configuration:

/* Note 1: here, we use the Runge-Kutta 4th order
   method to perform the real-time-evolution of
   the coupled Heisenberg equations (1st-order
   ODEs) describing the dynamics of the system;

   Note 2: the 2D discrete Fourier transform of
   the evolving input array (i.e., the pointer
   describing the spin-field) is done directly
   in this code, the outputs are 3 real arrays
   that hold the amplitute of each spin in the
   momentum representation for a range of fre-
   quencies, the 4th output gives the SpinSpin 
   temporal correlation for the input sample;

   Note 3: for a crystal lattice, the number
   of wavevectors in the 1st Brillouin zone
   is equal to the number of sites, i.e.:

   Nq = Ns & iNq = iNs ;

   For a quase-crystal, that is not the case,
   Nq can be set to any value depending on
   Lsz which is arbitrary (in this code, we
   set Lsz = sqrt(Ns) : rounded and even); */

void get_dynSpectrum(int procNum,
		     double  *tWinVec, double **spinField,
		     double **SqxWVec, double **SqyWVec,
		     double **SqzWVec, double  *qSStVec)
{  
  const double f1d6 = 1.0 / 6.0;
  const double f2d6 = 2.0 / 6.0;
 
  const double dpc = 0.0;
   
  unsigned int i, j, k, n;
 
  Vec3d KVec1, KVec2, KVec3, KVec4, PVec, DVec;

  Vec3d spinVec, spinVecNew, locField, dtmVec;

  double projSpin0, scaleFac, ERef, EDenDiff, QMagDiff;
  
  double rx, ry, rz, E0, E1, QMag0, QMag1;
   
  //-----------------------------------
  // Allocate pointers for the temporal
  // evolution via the RK 4th method:
    
  double **prevField, **nextField;

  Vec3d *KC1, *KC2, *KC3, *KC4;

  prevField = Alloc_dble_array(Ns, 3);
  nextField = Alloc_dble_array(Ns, 3);

  KC1 = new Vec3d[Ns];
  KC2 = new Vec3d[Ns];
  KC3 = new Vec3d[Ns];
  KC4 = new Vec3d[Ns];

  //-------------------------------
  // Prepare initial spin pointers:

  double *xSpin0, *ySpin0, *zSpin0;

  xSpin0 = new double[Ns];
  ySpin0 = new double[Ns];
  zSpin0 = new double[Ns];

  for (i = 0; i < Ns; i++)//( t = 0 | n = 0 )
    {
      get_localSpin(i, spinField, spinVec);
      
      xSpin0[i] = spinVec[0];
      ySpin0[i] = spinVec[1];
      zSpin0[i] = spinVec[2];
    }

  //---------------------------
  // Allocate & initialize spin
  // correlations time-series:

  complex<double> **xSpec_tSeries;
  complex<double> **ySpec_tSeries;
  complex<double> **zSpec_tSeries;

  xSpec_tSeries = Alloc_cplx_array(Nq, ntm);
  ySpec_tSeries = Alloc_cplx_array(Nq, ntm);
  zSpec_tSeries = Alloc_cplx_array(Nq, ntm);

  init_cplx_array(xSpec_tSeries, Nq, ntm, zero);
  init_cplx_array(ySpec_tSeries, Nq, ntm, zero);
  init_cplx_array(zSpec_tSeries, Nq, ntm, zero);  
   
  //-----------------------------------------------
  // Allocate complex xyz-pointers for the momentum
  // representation of the evolving configuration:
 
  complex<double> *Sqx0, *Sqy0, *Sqz0;
  
  complex<double> *Sqx, *Sqy, *Sqz;

  Sqx0 = new complex<double>[Nq];
  Sqy0 = new complex<double>[Nq];
  Sqz0 = new complex<double>[Nq];
  
  Sqx = new complex<double>[Nq];
  Sqy = new complex<double>[Nq];
  Sqz = new complex<double>[Nq];

  //------------------------------
  // Calculate static contribution
  // & set zero-time (n = 0) data:

  if (qcrystal)
    {
      get_qct_SqData(spinField, Sqx0, Sqy0, Sqz0);
    }
  else//( if periodicity is assumed )
    {
      if (FFTW_ON)
	{
	  get_SqData_FFTW(spinField, Sqx0, Sqy0, Sqz0); }
      else
	{ get_SqData_iMKL(spinField, Sqx0, Sqy0, Sqz0); }
    }
  
  // for (k = 0; k < Nq; k++) //( uncomment to suppress static responce )
  //   {
  //     rx = abs(Sqx0[k]);
  //     ry = abs(Sqy0[k]);
  //     rz = abs(Sqz0[k]);
  //
  //     Sqx0[k] = min(spec0Max, rx) * Sqx0[k] / rx;
  //     Sqy0[k] = min(spec0Max, ry) * Sqy0[k] / ry;
  //     Sqz0[k] = min(spec0Max, rz) * Sqz0[k] / rz;
  //   }
      
  for (k = 0; k < Nq; k++)
    {	  	  	  
      xSpec_tSeries[k][0] = Sqx0[k];
      ySpec_tSeries[k][0] = Sqy0[k];
      zSpec_tSeries[k][0] = Sqz0[k];
    }

  qSStVec[0] = 1.0; /* Zero-time value is equal to one,
		      since it is associated with the
		      initial field of spins; */
  
  //-----------------------------------
  // Preparations for video-operations:

#if WITH_OPENCV == 1
  ///   
  /* 1) Output video FPS;
     2) Video writer codec;
     3) Output videos base-name;
     4) Tag showing output details;
     5) Background grid file-name;
     6) Output evolution video name;
     7) Lattice grid & vector field; */
    
  int fps = 24; //(1) 

  string cdc = "H264"; //(2)

  string vdname0 = "SpinEvo"; //(3)

  string ftag, tag = "_Set" + to_string(procNum); //(4)

  string inputImg = outDir2 + "lattice_image.png"; //(5)

  string outVid0 = outDir3 + vdname0 + tag + ".avi"; //(6)
      
  Mat gridMat, vecsMat; //(7)
            
  VideoWriter spinVideo;

  unsigned int codec;
      
  codec = VideoWriter::fourcc(cdc[0], cdc[1], cdc[2], cdc[3]);

  if (recSpinVec)
    { 
      gridMat = imread(inputImg, IMREAD_COLOR);       
	   
      spinVideo.open(outVid0, codec, fps, plotSize);
    }      
#endif///( Video-operations ) 

  //----------------------------------
  // Calculate initial energy density:

  ofstream outCheck;
  
  string outName = outDir1;

  outName += "RK_Check/0RK_Stuff.dat";

  if (RKCheck)
    {
      outCheck.open(outName, ios::out);
      
      get_energyValue(spinField, E0);

      QMag0 = absStaggMag(spinField);
    }
      
  //----------------------------
  // Perform temporal evolution: (t > 0)
  
  for (n = 1; n < ntm; n++) //// START: TIME-LOOP
    {
      get_energyValue(spinField, ERef);
      
      //.........................
      // Compute K1-coefficients:
      
      copy_field(spinField, prevField);
  
      for (i = 0; i < Ns; i++)
	{      
	  get_localSpin(i, prevField, spinVec);
	    
	  get_localField(i, prevField, locField);

	  PVec = crossProduct(locField, spinVec);

	  DVec = crossProduct(spinVec, PVec);
	  
	  KVec1 = dtm * (PVec + dpc * DVec);

	  /* Prepare new field for KC2 step */

	  spinVec = spinVec + 0.5 * KVec1;

	  set_localSpin(i, spinVec, nextField);
      
	  KC1[i] = KVec1;
	}

      //.........................
      // Compute K2-coefficients:

      copy_field(nextField, prevField);
  
      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, prevField, spinVec);
	    
	  get_localField(i, prevField, locField);

	  PVec = crossProduct(locField, spinVec);

	  DVec = crossProduct(spinVec, PVec);
	  
	  KVec2 = dtm * (PVec + dpc * DVec);
	  
	  /* Prepare new field for KC3 step */
      
	  get_localSpin(i, spinField, spinVec);

	  spinVec = spinVec + 0.5 * KVec2;

	  set_localSpin(i, spinVec, nextField);
      
	  KC2[i] = KVec2;
	}

      //.........................
      // Compute K3-coefficients:

      copy_field(nextField, prevField);
  
      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, prevField, spinVec);
	    
	  get_localField(i, prevField, locField);

	  PVec = crossProduct(locField, spinVec);

	  DVec = crossProduct(spinVec, PVec);
	  
	  KVec3 = dtm * (PVec + dpc * DVec);
	  
	  /* Prepare new field for KC4 step */
      
	  get_localSpin(i, spinField, spinVec);

	  spinVec = spinVec + KVec3;

	  set_localSpin(i, spinVec, nextField);
      
	  KC3[i] = KVec3;
	}

      //.........................
      // Compute K4-coefficients:

      copy_field(nextField, prevField);
  
      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, prevField, spinVec);
	    
	  get_localField(i, prevField, locField);

	  PVec = crossProduct(locField, spinVec);

	  DVec = crossProduct(spinVec, PVec);
	  
	  KVec4 = dtm * (PVec + dpc * DVec);
	        
	  KC4[i] = KVec4; /* Last coefficient */
	}

      //...........................
      // Apply evolution algorithm:

      copy_field(spinField, prevField);
      
      for (i = 0; i < Ns; i++)
	{
	  dtmVec = ( f1d6 * ( KC1[i] + KC4[i] ) +
		     f2d6 * ( KC2[i] + KC3[i] ) );

	  get_localSpin(i, prevField, spinVec);

	  spinVec = spinVec + dtmVec;

	  spinVecNew = normVec3d(spinVec);     

	  set_localSpin(i, spinVecNew, spinField);
	}

      //...................................
      // Apply energy correction procedure:

      scaleFac = 0.0;

      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, spinField, spinVec);
	    
	  get_localField(i, spinField, locField);

	  PVec = crossProduct(locField, spinVec);
	  
	  scaleFac += dotProduct(PVec, PVec);
	}

      get_energyValue(spinField, E1);

      scaleFac = (ERef - E1) / scaleFac;

      projSpin0 = 0.0; // Look for 'qSStVec' below;

      for (i = 0; i < Ns; i++)
	{
	  get_localSpin(i, spinField, spinVec);
	    
	  get_localField(i, spinField, locField);

	  PVec = crossProduct(locField, spinVec);

	  DVec = crossProduct(spinVec, PVec);

	  spinVec = spinVec + scaleFac * DVec;

	  spinVecNew = normVec3d(spinVec);

	  projSpin0 += ( xSpin0[i] * spinVecNew[0] +
			 ySpin0[i] * spinVecNew[1] +
			 zSpin0[i] * spinVecNew[2] );

	  set_localSpin(i, spinVecNew, spinField);
	}

      qSStVec[n] = iNs * projSpin0;
      
      //................................
      // Get momentum representation and 
      // save result to output pointers:     

      if (qcrystal)
	{
	  get_qct_SqData(spinField, Sqx, Sqy, Sqz);
	}
      else//( if periodicity is assumed )
	{
	  if (FFTW_ON)
	    {
	      get_SqData_FFTW(spinField, Sqx, Sqy, Sqz); }
	  else
	    { get_SqData_iMKL(spinField, Sqx, Sqy, Sqz); }
	}
      
      for (k = 0; k < Nq; k++)
	{	  	  	  
	  xSpec_tSeries[k][n] = Sqx[k];
	  ySpec_tSeries[k][n] = Sqy[k];
	  zSpec_tSeries[k][n] = Sqz[k];
	}

      //..............................................
      // Compute the energy density & mag. variations:

      if (RKCheck)
	{	  	  
	  get_energyValue(spinField, E1);
	  
	  QMag1 = absStaggMag(spinField);

	  EDenDiff = ( E1 - E0 ) / E0;

	  QMagDiff = ( QMag1 - QMag0 );

	  outCheck << n * dtm   << X3
		   << EDenDiff  << X3
		   << QMagDiff  << endl;
	}
      
      //.............................................
      // Record video-frame of spin-system evolution:

#if WITH_OPENCV == 1
      /// 
      if (recSpinVec)
	{  
	  ftag = "n=" + to_string(n);
      
	  make_vecField(X2, ftag, spinField, gridMat, vecsMat);

	  spinVideo.write(vecsMat);

	  if (procNum == root){ report(n, ntm); }
	}
#endif///( Recording spin-evo movie )
      
    }//// END: TIME-LOOP

  if (RKCheck)
    {
      outCheck.close();
    }

  /*-----------------------------------------------
    Compute DFT-1D (time --> frequency) of the time
    series multiplied by the windowing factor (wfc) 

    Note: in 'get_FourierSpectrum1D', the output is
    effectively rescaled by the factor (1.0 / ntm); 

    DFT data preparation: subtract the mean & apply
    the window filter and proceed with the DFT-1D; */

  const double nfc = 1.0 / ntm;

  complex<double> xMean, yMean, zMean;
  
  complex<double> *xMeanVec, *yMeanVec, *zMeanVec;
  
  xMeanVec = new complex<double>[Nq];
  yMeanVec = new complex<double>[Nq];
  zMeanVec = new complex<double>[Nq];  
  
  for (k = 0; k < Nq; k++) // Calculate the mean value 
    {                      // for all temporal data-sets;
      xMean = zero;
      yMean = zero;
      zMean = zero;
      
      for (n = 0; n < ntm; n++)
	{      
	  xMean += xSpec_tSeries[k][n];
	  yMean += ySpec_tSeries[k][n];
	  zMean += zSpec_tSeries[k][n];
	}

      xMeanVec[k] = nfc * xMean;
      yMeanVec[k] = nfc * yMean;
      zMeanVec[k] = nfc * zMean;
    }

  /*...................
    If FFTW_ON = FALSE:

    Data-vectors are used as input/output
    for the inplace 2D-DFT procedures, &
    scaling must be applied to outputs; 

    ALERT: FFTW_ON = true crashes code! */

  complex<double> xDataVec[ntm]; 
  complex<double> yDataVec[ntm]; 
  complex<double> zDataVec[ntm];

  complex<double> wfc;

  if (FFTW_ON)//( 2D-DFT from FFTW library )
    {
      double *xSpecVec, *ySpecVec, *zSpecVec;
      
      xSpecVec = new double[ntm];
      ySpecVec = new double[ntm];
      zSpecVec = new double[ntm];  
  
      for (k = 0; k < Nq; k++)
	{
	  xMean = 0.0; // Mean subtraction with
	  yMean = 0.0; // MeanVec[k] is disabled
	  zMean = 0.0; // for these calculations;
	      
	  for (n = 0; n < ntm; n++)
	    {
	      wfc = complex<double>(tWinVec[n], 0.0);
	      
	      xDataVec[n] = wfc * (xSpec_tSeries[k][n] - xMean);
	      yDataVec[n] = wfc * (ySpec_tSeries[k][n] - yMean);
	      zDataVec[n] = wfc * (zSpec_tSeries[k][n] - zMean);
	    }
	  	  
	  get_FourierSpectrum1D(xDataVec, xSpecVec);
	  get_FourierSpectrum1D(yDataVec, ySpecVec);
	  get_FourierSpectrum1D(zDataVec, zSpecVec);

	  for (n = 0; n < ntm; n++)
	    {
	      SqxWVec[k][n] = xSpecVec[n];
	      SqyWVec[k][n] = ySpecVec[n];
	      SqzWVec[k][n] = zSpecVec[n];
	    }	      
	}//// Wavevector-LOOP (END)
  
      delete[] xSpecVec;
      delete[] ySpecVec;
      delete[] zSpecVec;
    }
  else//( 2D-DFT from Intel-MKL library )
    {
      double sfc = sqrt(1.0 / ntm);
      
      complex<double> cx, cy, cz;
      
      DFTI_DESCRIPTOR_HANDLE handle = NULL;

      MKL_LONG stat, xstat, ystat, zstat;

      stat = DftiCreateDescriptor(&handle, DFTI_DOUBLE,
				  DFTI_COMPLEX, 1, ntm);
  
      stat = DftiCommitDescriptor(handle);
        
      for (k = 0; k < Nq; k++)
	{
	  xMean = 0.0; // Mean subtraction with
	  yMean = 0.0; // MeanVec[k] is disabled
	  zMean = 0.0; // for these calculations;
	      
	  for (n = 0; n < ntm; n++)
	    {	  
	      wfc = complex<double>(tWinVec[n], 0.0);
	      
	      xDataVec[n] = wfc * (xSpec_tSeries[k][n] - xMean);
	      yDataVec[n] = wfc * (ySpec_tSeries[k][n] - yMean);
	      zDataVec[n] = wfc * (zSpec_tSeries[k][n] - zMean);
	    }

	  xstat = DftiComputeForward(handle, xDataVec);
	  ystat = DftiComputeForward(handle, yDataVec);
	  zstat = DftiComputeForward(handle, zDataVec);
	  	  
	  for (n = 0; n < ntm; n++)
	    {
	      cx = sfc * xDataVec[n];
	      cy = sfc * yDataVec[n];
	      cz = sfc * zDataVec[n];
	  
	      SqxWVec[k][n] = real(cx * conj(cx));	      
	      SqyWVec[k][n] = real(cy * conj(cy));	     
	      SqzWVec[k][n] = real(cz * conj(cz));
	    }	      
	}//// Wavevector-LOOP (END)

      DftiFreeDescriptor(&handle);
    }
  
  delete[] xMeanVec;
  delete[] yMeanVec;
  delete[] zMeanVec;
 
  //----------------------------------------
  // Deallocate time-series and RK pointers:

  deAlloc_cplx_array(xSpec_tSeries, Nq, ntm);
  deAlloc_cplx_array(ySpec_tSeries, Nq, ntm);
  deAlloc_cplx_array(zSpec_tSeries, Nq, ntm);
  
  deAlloc_dble_array(nextField, Ns, 3);
  deAlloc_dble_array(prevField, Ns, 3);
  
  delete[] KC1;
  delete[] KC2;
  delete[] KC3;
  delete[] KC4;

  //---------------------------
  // Deallocate other pointers:

  delete[] xSpin0;
  delete[] ySpin0;
  delete[] zSpin0;

  delete[] Sqx0; delete[] Sqx;
  delete[] Sqy0; delete[] Sqy;
  delete[] Sqz0; delete[] Sqz;

#if WITH_OPENCV == 1
  /// 
  gridMat.release();
  vecsMat.release();

  spinVideo.release();
  ///
#endif
}

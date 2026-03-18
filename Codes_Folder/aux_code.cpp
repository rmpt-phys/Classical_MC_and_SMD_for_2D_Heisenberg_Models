/*  We use some namespaces in order
    to simplify the code: cv, std; */

#if WITH_OPENCV == 0
//
#include "structures.cpp"
//
#endif

//======================================================
// 5-components double vector (similar to OpenCV Vec4d):

struct Vec5d
{
  double data[5];

  // Define basic structure:

  Vec5d(double v0, double v1, double v2, double v3, double v4)
  {
    data[0] = v0;
    data[1] = v1;
    data[2] = v2;
    data[3] = v3;
    data[4] = v4;
  }

  Vec5d()
  {
    data[0] = 0.0;
    data[1] = 0.0;
    data[2] = 0.0;
    data[3] = 0.0;
    data[4] = 0.0;
  }

  // Define accessing feature:

  double operator[](int index) const
  {
    return data[index];
  }

  double &operator[](int index)
  {
    return data[index];
  }

  // Overload "+" operator for vector addition:

  Vec5d operator+(const Vec5d &other) const
  {
    return Vec5d(
      data[0] + other[0],
      data[1] + other[1],
      data[2] + other[2],
      data[3] + other[3],
      data[4] + other[4]
    );
  }

  // Overload "-" operator for vector subtraction:

  Vec5d operator-(const Vec5d &other) const
  {
    return Vec5d(
      data[0] - other[0],
      data[1] - other[1],
      data[2] - other[2],
      data[3] - other[3],
      data[4] - other[4]
    );
  }

  // In-place addition operator:

  Vec5d &operator+=(const Vec5d &other)
  {
    data[0] += other[0];
    data[1] += other[1];
    data[2] += other[2];
    data[3] += other[3];
    data[4] += other[4];

    return *this;
  }
};

//===========================================
// Neighbor site label and directional angle:

struct nbInfo
{
  int tag;
  
  double angle;

  // Default constructor:
  
  nbInfo() : tag(0), angle(0.0) {}

  // Parameterized constructor:
  
  nbInfo(int num, double theta) : tag(num), angle(theta) {}
};

//=======================================
// 3-components point (similar to Point):

struct Pts3d
{
  int x, y, z;

  // Default constructor:
  
  Pts3d() : x(0), y(0), z(0) {}

  // Parameterized constructor:
  
  Pts3d(int xVal, 
	int yVal,
	int zVal) : x(xVal), y(yVal), z(zVal) {}
};

//===================
// List of 1 integer:

struct Lst1i
{
  int data[1]; // Array to store the 1 integer;

  // Default constructor:
  
  Lst1i()
  {
    for (int &x : data) x = 0; // Initialize to 0;
  }
  
  // Parameterized constructor:
  
  Lst1i(int z1Val)
  {
    data[0] = z1Val;
  }

  // Operator to access elements like an array:
  
  int& operator[](size_t index)
  {
    return data[index];
  }

  // Const-version of the operator (read-only access):
  
  const int& operator[](size_t index) const
  {
    return data[index];
  }
};

//====================
// List of 4 integers:

struct Lst4i
{
  int data[4]; // Array to store the 4 integers;

  // Default constructor:
  
  Lst4i()
  {
    for (int &x : data) x = 0; // Initialize to 0;
  }
  
  // Parameterized constructor:
  
  Lst4i(int z1Val, int z2Val, int z3Val, int z4Val)
  {
    data[0] = z1Val; data[1] = z2Val;
    data[2] = z3Val; data[3] = z4Val;
  }

  // Operator to access elements like an array:
  
  int& operator[](size_t index)
  {
    return data[index];
  }

  // Const-version of the operator (read-only access):
  
  const int& operator[](size_t index) const
  {
    return data[index];
  }
};

//====================
// List of 5 integers:

struct Lst5i
{
  int data[5]; // Array to store the 8 integers;

  // Default constructor:
  
  Lst5i()
  {
    for (int &x : data) x = 0; // Initialize to 0;
  }
  
  // Parameterized constructor:
  
  Lst5i(int z1Val, int z2Val, int z3Val, int z4Val, int z5Val)
  {
    data[0] = z1Val; data[1] = z2Val;
    data[2] = z3Val; data[3] = z4Val;
    data[4] = z5Val;
  }

  // Operator to access elements like an array:
  
  int& operator[](size_t index)
  {
    return data[index];
  }

  // Const-version of the operator (read-only access):
  
  const int& operator[](size_t index) const
  {
    return data[index];
  }
};

//===============================
// Complex vector (custom class):

struct cplxVec
{
    complex<double> x;
    complex<double> y;
    complex<double> z;
};

//==============
// Sinc-fuction:

double sinc(double x)
{
  double rval;
  
  if (x == 0.0)
    {
      rval = 1.0; }
  else
    { rval = sin(x) / x; }
  
  return rval;
}

//==========================
// Sinc-full window fuction:

void sincWindow(int N, double fc, double *rvec)
{    
  const double twoPi = 2.0 * acos(-1.0);

  const double dx = 2.0 / (N - 1.0);

  int i; double x, rval;
  
  for (i = 0; i < N; i++)
    {
      x = (i * dx - 1.0) * fc;

      rval = sinc(twoPi * x);

      rvec[i] = rval;
    }
}

//========================
// Lanczos window fuction:

void lanczosWindow(int N, double fc, double *rvec)
{    
  const double twoPi = 2.0 * acos(-1.0);

  const double dx = 2.0 / (N - 1.0);

  int i; double x, rval;
  
  for (i = 0; i < N; i++)
    {
      x = (i * dx - 1.0) * fc;

      if (abs(x) > 0.5)
	{
	  rval = 0.0; }
      else
	{ rval = sinc(twoPi * x); }

      rvec[i] = rval;
    }
}

//======================
// Hann window function:

void hannWindow(int N, int pw, double *rvec)
{
  const double twoPi = 2.0 * acos(-1.0);
  
  const double dx = 1.0 / (N - 1.0);

  int i; double x, rval;

  for (i = 0; i < N; i++)
    {
      x = i * dx;

      rval = 0.5 * (1.0 - cos(twoPi * x));

      rvec[i] = pow(rval, pw);
    }
}

//==========================
// Gaussian window function:

void gaussWindow(int N, double *rvec)
{
  const double alpha = 80.0;

  const double dx = 2.0 / (N - 1.0);

  int i; double x;

  for (i = 0; i < N; i++)
    {
      x = i * dx - 1.0;

      rvec[i] = exp(- alpha * pow(x, 2));
    }
}

//==================================
// Blackman-Nuttall window function:

void blackNtWindow(int N, double *rvec)
{
  const double twoPi = 2.0 * acos(-1.0);
  
  const double dx = 1.0 / (N - 1.0);

  const double a0 = 0.3635819;
  const double a1 = 0.4891775;
  const double a2 = 0.1365995;
  const double a3 = 0.0106411;

  int i; double x, rval;

  for (i = 0; i < N; i++)
    {
      x = i * dx;

      rval = (- a1 * cos(1.0 * twoPi * x)
	      + a2 * cos(2.0 * twoPi * x)
	      - a3 * cos(3.0 * twoPi * x));
      
      rvec[i] = a0 + rval;
    }
}

//=================================
// Blackman-Harris window function:

void blackHrWindow(int N, double *rvec)
{
  const double twoPi = 2.0 * acos(-1.0);
  
  const double dx = 1.0 / (N - 1.0);

  const double a0 = 0.35875;
  const double a1 = 0.48829;
  const double a2 = 0.14128;
  const double a3 = 0.01168;

  int i; double x, rval;

  for (i = 0; i < N; i++)
    {
      x = i * dx;

      rval = (- a1 * cos(1.0 * twoPi * x)
	      + a2 * cos(2.0 * twoPi * x)
	      - a3 * cos(3.0 * twoPi * x));
      
      rvec[i] = a0 + rval;
    }
}

//==========================
// Flat-top window function:

void flatTopWindow(int N, double *rvec)
{
  const double twoPi = 2.0 * acos(-1.0);
  
  const double dx = 1.0 / (N - 1.0);

  const double a0 = 0.215578950;
  const double a1 = 0.416631580;
  const double a2 = 0.277263158;
  const double a3 = 0.083578947;
  const double a4 = 0.006947368;

  int i; double x, rval;

  for (i = 0; i < N; i++)
    {
      x = i * dx;

      rval = (- a1 * cos(1.0 * twoPi * x)
	      + a2 * cos(2.0 * twoPi * x)
	      - a3 * cos(3.0 * twoPi * x)
	      + a4 * cos(4.0 * twoPi * x));

      rvec[i] = a0 + rval;
    }
}

//=================================================
// Codes to check if a number if of type: pow(2, n)

bool isPowerOfTwo(int n)
{
  if (n <= 0) return false;
  
  return (n & (n - 1)) == 0;
}

//================================================
// Codes to check if a string represents a number:

bool isNumber(const string &s)
{
  for (char const &ch : s)
    {
      if (isdigit(ch) == 0) return false;
  }
  
  return true;
}

bool isFloat(const string &str)
{
  try
    {
      size_t pos;
	
      stof(str, &pos); // Try to convert: string --> float
	
      return pos == str.length(); // If the conversion consumed the	
    }                             // entire string, it's a valid float;
  catch (const invalid_argument&)
    {
      return false; // Failed to convert (not a valid float);
    }
}

//========================
// Get sign of an integer:

int sign(int i)
{
  if (i > 0){return 1;}
  //
  else if (i < 0){return -1;}
  //
  else{return 0;}
}

//==============================
// Code to check if file exists:

bool fileExists(const string &filePath)
{
  struct stat buffer;
  
  return (stat(filePath.c_str(), &buffer) == 0);
}

//===================
// Get string length:

void strLength(string str)
{
  cout << "\n" << " String : " << str << " , length : "
       << str.length() << "\n";
}

//=======================                     
// String trimming codes:

static inline void ltrim(string &s)
{
  s.erase(s.begin(), find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !isspace(ch);
  }));
}

static inline void rtrim(string &s)
{
  s.erase(find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
    return !isspace(ch);
  }).base(), s.end());
}

static inline void trim(string &s)
{
  ltrim(s);
  rtrim(s);
}

//=============================
// Lambda function for doubles: ( fixed format )

auto fmtDbleFix = [](double number,
		     unsigned int dg,
		     unsigned int wd)
 {
   ostringstream oss;
    
   oss << fixed << showpos
       << setprecision(dg) << setw(wd) << number;
    
   return oss.str();
 };

//=============================
// Lambda function for doubles: ( scientific format )

auto fmtDbleSci = [](double number,
		     unsigned int dg,
		     unsigned int wd)
 {
   ostringstream oss;
    
   oss << scientific << showpos
       << setprecision(dg) << setw(wd) << number;
    
   return oss.str();
};

//=======================================
// Lambda function for 2-digits integers:

auto iformat2 = [](int number)
 {
   ostringstream oss;
    
   oss << setw(2) << setfill('0') << number;
    
   return oss.str();
};

//=======================================
// Lambda function for 3-digits integers:

auto iformat3 = [](int number)
 {
   ostringstream oss;
    
   oss << setw(3) << setfill('0') << number;
    
   return oss.str();
};

//====================
// Wait for some time:

void waitAndJump()
{
  double const dx = 2.0 * acos(-1.0) / INT_MAX;
  
  unsigned int wait_time;
  
  auto time1 = high_resolution_clock::now();

  for (int i = 0; i < INT_MAX; i++)
    {
      auto time2 = high_resolution_clock::now();

      auto dtime = time2 - time1;

      double value = ( pow(sin(0.5 * i * dx), 2) +  // Whatever
		       pow(cos(2.0 * i * dx), 3) ); // calculation;
      
      wait_time = duration_cast<milliseconds>(dtime).count();

      if (wait_time > 200){break;}
    }

  cerr << endl; // Jump 1 line;
}
  
//===========================================
// Subroutines for dynamical allocation of a
// 2-dimensional array/pointer of type (...):

complex<double> **Alloc_cplx_array(int nrow, int ncol)
{  
  complex<double> **ptr2d = NULL;

  ptr2d = new complex<double> *[nrow];
  
  for(int i = 0; i < nrow; i++)
    {
      ptr2d[i] = new complex<double>[ncol];
    }
  
  return ptr2d;
}

double **Alloc_dble_array(int nrow, int ncol)
{ 
  double **ptr2d = NULL;

  ptr2d = new double *[nrow];
  
  for(int i = 0; i < nrow; i++)
    {
      ptr2d[i] = new double[ncol];
    }
  
  return ptr2d;
}

int **Alloc_intg_array(int nrow, int ncol)
{ 
  int **ptr2d = NULL;

  ptr2d = new int *[nrow];
  
  for(int i = 0; i < nrow; i++)
    {
      ptr2d[i] = new int [ncol];
    }
  
  return ptr2d;
}

Vec3d **Alloc_vec3d_array(int nrow, int ncol)
{ 
  Vec3d **vec3dArray = NULL;

  vec3dArray = new Vec3d *[nrow];
  
  for(int i = 0; i < nrow; i++)
    {
      vec3dArray[i] = new Vec3d[ncol];
    }
  
  return vec3dArray;
}

//============================================
// Subroutines for dynamical deallocation of a
// 2-dimensional array/pointer of type (...):

void deAlloc_cplx_array(complex<double> **ptr2d,
			int nrow, int ncol)
{  
  for(int i = 0; i < nrow; i++)
    {
      delete[] ptr2d[i];
    }
  
  delete[] ptr2d;
}

void deAlloc_dble_array(double **ptr2d,
			int nrow, int ncol)
{  
  for(int i = 0; i < nrow; i++)
    {
      delete[] ptr2d[i];
    }
  
  delete[] ptr2d;
}

void deAlloc_intg_array(int **ptr2d,
			int nrow, int ncol)
{
  for(int i = 0; i < nrow; i++)
    {
      delete[] ptr2d[i];
    }
  
  delete[] ptr2d;
}

void deAlloc_vec3d_array(Vec3d **vec3dArray,
			 int nrow, int ncol)
{  
  for(int i = 0; i < nrow; i++)
    {
      delete[] vec3dArray[i];
    }
  
  delete[] vec3dArray;
}

//===============================================
// Subroutines for initialization of 2-d pointer:

void init_cplx_array(complex<double> **ptr2d,
		     int nrow, int ncol,
		     complex<double> ival)
{  
  for   (int i = 0; i < nrow; i++){
    for (int j = 0; j < ncol; j++)
      {/*-------------------------*/
	ptr2d[i][j] = ival;}}
}

void init_dble_array(double **ptr2d,
		     int nrow, int ncol,
		     double ival)
{  
  for   (int i = 0; i < nrow; i++){
    for (int j = 0; j < ncol; j++)
      {/*-------------------------*/
	ptr2d[i][j] = ival;}}
}

void init_intg_array(int **ptr2d,
		     int nrow, int ncol, int ival)
{
  for   (int i = 0; i < nrow; i++){
    for (int j = 0; j < ncol; j++)
      {/*-------------------------*/
	ptr2d[i][j] = ival;}}
}

//====================================
// Loop progress report (single line):

void report(int k, int kmax)
{
  float progress;
  
  progress = 100.0 * (k + 1.0) / kmax;

  cerr << iformat3((int)progress) << " %";

  cerr << "\b\b\b\b\b";
}

//===================
// Triangular number:

int triang_number(int n)
{
    double tnum;

    tnum = n * (n + 1) / 2;

    return tnum;
}

//=============================================
// Find maximum value of a double type pointer:

double maxValue(int sz, double *vec)
{
  int i; double v0, vmax;

  vmax = vec[0];
  
  for (i = 1; i < sz; ++i)
    {
      v0 = vec[i];

      if (v0 > vmax){vmax = v0;}
    }

  return vmax;
}

//===============================
// Normalize double type pointer:

void normalize(int sz, double *vec)
{
  int i; double vmax, fc;

  vmax = maxValue(sz, vec);

  fc = 1.0 / vmax;
  
  for (i = 0; i < sz; ++i)
    {
      vec[i] = vec[i] * fc;
    }
}

//=====================================
// Sum components of a 3d double array:

double sum_dbleVec(int dm, const double *rvec)
{
    double vsum = 0.0;
    
    for (int i = 0; i < dm; ++i){vsum += rvec[i];}

    return vsum;
}

//======================================================
// Return dot-product of two plain vectors (const-type):

double dotProd(const double v1[3], const double v2[3])
{
  return (
    v1[0] * v2[0] +
    v1[1] * v2[1] +
    v1[2] * v2[2]);
}

double dotProd2d(const double v1[2], const double v2[2])
{
  return ( v1[0] * v2[0] + v1[1] * v2[1] );
}

//======================================================
// Return dot-product of two plain vectors (const-type):

void crossProd(const double v1[3],
	       const double v2[3], double v3[3])
{
  v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
  v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
  v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

//==================================================
// Return the array resulting from a matrix product:
/*
  U[3] = MX[3 x 3] * V[3] , (fixed size)

  Above '*' stands for the product operator; */

void MxVecProd(const double MX[9],
	       const double  V[3], double U[3])
{
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
	      3, 3, 1.0, MX, 3,
	      V, 1, 0.0, U, 1);
}

//=======================================
// Subroutine to turn 2d-array to vector:

void dbleFlatten2D(int nrows, int ncols,
		   double **ptr2d,
		   double  *ptr1d)
{
  int i, j;
  
  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  ptr1d[i * ncols + j] = ptr2d[i][j];
	}
    }
}

/* Integer vector version */

void intFlatten2D(int   nrows, int ncols,
		  int **ptr2d, int *ptr1d)
{
  int i, j;
  
  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  ptr1d[i * ncols + j] = ptr2d[i][j];
	}
    }
}

/* Complex vector version (2 output pointers) */

void cplxFlatten2D(int nrows, int ncols,
		   complex<double> **ptr2d,
		   double *rptr1d,
		   double *iptr1d)
{
  int i, j;
  
  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  rptr1d[i * ncols + j] = ptr2d[i][j].real();
	  iptr1d[i * ncols + j] = ptr2d[i][j].imag();
	}
    }
}

//=======================================
// Subroutine to turn 2d-array to vector:

void dbleReshape2D(int nrows, int ncols,
		   double  *ptr1d,
		   double **ptr2d)
{
  int i, j;
  
  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  ptr2d[i][j] = ptr1d[i * ncols + j];
	}
    }
}

/* Integer vector version */

void intReshape2D(int  nrows, int ncols,
		  int *ptr1d, int **ptr2d)
{
  int i, j;
  
  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  ptr2d[i][j] = ptr1d[i * ncols + j];
	}
    }
}

/* Complex vector version (2 input pointers) */

void cplxReshape2D(int nrows, int ncols,
		   double *rptr1d, double *iptr1d,
		   complex<double> **ptr2d)
{
  int i, j;

  for (i = 0; i < nrows; ++i)
    {
      for (j = 0; j < ncols; ++j)
	{
	  ptr2d[i][j] = complex<double>(rptr1d[i * ncols + j], 
					iptr1d[i * ncols + j]);
	}
    }
}

//====================================================
// Fill int-values around a point with a given radius:

/*
  vector<vector<int>> marker(ncols, vector<int>(nrows, 0));
  
  Site marker: all points are set to zero (empty), if a
  a site is defined at some point in the frame region of 
  the ncols X nrows Mat object 'Lattice' (see 'szFrame')
  then a white circle is drawn on it (user sees a white
  ball in the plot region with the site label printed)
  and the marker array at the site coordinates is set
  to 1 (site visited);
  
  If pt0 = Point(n1, n2) is a site point in the lattice,
  pixel = Lattice.at<Vec3b>(pt0) is finite (if zero, the
  pixel is black/empty, i.e., no site at the point pt0),
  and marker[n1][n2] = 1 (again, if zero, pt0 is empty); */

void setMark(vector<vector<int>> &marker,
	     const Point& pt0, int rad)
{
  int i, j, nx, ny, mx, my;

  const int i0 = pt0.x;
  const int j0 = pt0.y;
  
  const int sz = floor(0.5 * rad);
  
  for (i = 0; i < sz; i++)
    {
      nx = i0 + i;
      mx = i0 - i;
	  
      for (j = 0; j < sz; j++)
	{
	  ny = j0 + j;
	  my = j0 - j;
	  
	  marker[nx][ny] = 1;
	  marker[mx][my] = 1;
	}
    }
}

//=============================================
// Calculate mean-value of data vector/pointer:

double get_dbleMeanVal(int Sz, double *dataVec)
{ 
  double fc = 1.0 / Sz;

  double r0 = 0.0;

  for (int k = 0; k < Sz; k++)
    {
      r0 += dataVec[k];
    }
    
  return fc * r0;
}

complex<double> get_cplxMeanVal(int Sz, complex<double> *dataVec)
{ 
  double fc = 1.0 / Sz;

  complex<double> c0 = {0.0, 0.0};

  for (int k = 0; k < Sz; k++)
    {
      c0 += dataVec[k];
    }
    
  return fc * c0;
}

//==============================
// Function to compute bootstrap
// standard deviation using GSL:

double gsl_bootstrap_stddev(int GSL_seed,
			    const vector<double> &dataVec)
{  
  const int num_resamples = 1000;

  const size_t dataSz = dataVec.size();

  /*.....................................
    Configure GLS random-number generator */
  
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

  gsl_rng_set(rng, GSL_seed);

  /*...........................................
    Build vector of means calculated for each
    resampled set via sampling with replacement */

  int i, j;
  
  double resample_mean;
  double bootstrp_sdev;

  vector<double> resample(dataSz);
  
  vector<double> meanVec(num_resamples, 0.0);
    
  for (i = 0; i < num_resamples; i++)
    {      	
      for (j = 0; j < dataSz; ++j)
	{
	  resample[j] = dataVec[gsl_rng_uniform_int(rng, dataSz)];
	}
      
      resample_mean = gsl_stats_mean(resample.data(), 1, dataSz);
	
      meanVec[i] = resample_mean;
    }

  /*.......................................
    Calculate boot-strap standard deviation */
  
  bootstrp_sdev = gsl_stats_sd(meanVec.data(), 1,
			       num_resamples);
    
  gsl_rng_free(rng);
    
  return bootstrp_sdev;
}

//=================================================
// Statistical calculations with measurement array:

void get_statVectors(bool USE_GSL,
		     int GSL_seed,
		     int Sz, int Ms,		     
		     double **datArray,
		     double  *meanList,
		     double  *stdvList)
{
  int i, j, k;
  
  if (USE_GSL)
    {
      /*.........................
	Compute mean and variance
	using GSL stat. functions */

      for (i = 0; i < Ms; i++)
	{
	  vector<double> dataVec(Sz, 0.0);

	  for (k = 0; k < Sz; k++)
	    {
	      dataVec[k] = datArray[k][i];
	    }

	  meanList[i] = gsl_stats_mean(dataVec.data(), 1, Sz);

	  stdvList[i] = gsl_bootstrap_stddev(GSL_seed, dataVec);
	}
    }
  else//( simple method without bootstrap resampling )
    {
      double fc, diff;

      /*.....................
	Initialize mean-value 
	and variance pointers */
  
      for (i = 0; i < Ms; i++)
	{
	  meanList[i] = 0.0;
	  stdvList[i] = 0.0;
	}

      /*...................
	Compute mean-values */

      for (k = 0; k < Sz; k++)
	{
	  for (i = 0; i < Ms; i++)
	    {
	      meanList[i] += datArray[k][i];
	    }
	}
  
      fc = 1.0 / Sz;
  
      for (i = 0; i < Ms; i++)
	{
	  meanList[i] = fc * meanList[i];
	}
    
      /*.......................
	Compute variance-values */
  
      for (k = 0; k < Sz; k++)
	{
	  for (i = 0; i < Ms; i++)
	    {
	      diff = datArray[k][i] - meanList[i];
	  
	      stdvList[i] += pow(diff, 2);
	    }
	}
  
      fc = 1.0 / (Sz - 1.0);
  
      for (i = 0; i < Ms; i++)
	{      
	  stdvList[i] = sqrt(fc * stdvList[i]);
	}
    }
}

//==============================
// Hexagonal region filter code:

bool hexFilter(double qx, double qy,
	       double Q0, double dQ)
{  
  const double sq3 = sqrt(3.0);

  const double s60 = 0.5 * sq3;

  double Qs, w0, w1, w2, w3, w4;

  Qs = Q0 + dQ / 3.0;
   
  w0 = abs(qy) - Qs * s60;
      
  if (w0 < 0.0) // Vertical filter;
    {
      w1 = sq3 * (qx - Qs) - qy;
      w2 = sq3 * (qx - Qs) + qy;
      w3 = sq3 * (qx + Qs) - qy;
      w4 = sq3 * (qx + Qs) + qy;

      if ((w1 < 0.0 && w2 < 0.0) && // Hexagonal
	  (w3 > 0.0 && w4 > 0.0))   // sides filter;
	{
	  return true;
	}
    }
      
  return false;
}

//==========================================
// Verify existence and data of binary file:

void check_binFile(size_t szTable,
		   string inputName, int &flag)
{
  ifstream binFile;
  
  binFile.open(inputName, ios::binary);

  binFile.seekg(0, ios::end);

  if (!binFile.is_open())
    {
      cerr << " Error >>"; flag = 1;
		      
      cerr << " File not found: " << inputName << "\n\n";	      
    }
  else//( if the file was found )
    {
      if (szTable == binFile.tellg())
	{	  
	 flag = 0;	  
	}
      else//( invalid data )
	{
	  cerr << " Error >>"; flag = 1;
	      
	  cerr << " Invalid data in: " << inputName << "\n\n";
	}
    }

  binFile.close();
}

//=====================================================
// Function to find & read files with the given prefix:

void processFiles(const string &prefix,
		  const string &outDir)
{
  // Regular expression to extract
  // the tag from the file name...
  
  regex tagRegex(R"(extH\((\d+(\.\d+)?)\)\.dat)");

  // Iterate over files in the specified directory:
  
  for (const auto &entry : fs::directory_iterator(outDir))
    {
      if (fs::is_regular_file(entry))
	{
	  // Check if the file name starts
	  // with the specified prefix...
	  
	  if (entry.path().filename().string().find(prefix) == 0)
	    {
	      // Extract tag from the file name
	      // using regular expression...
	      
	      string fileName = entry.path().filename().string();

	      smatch match;

	      if (regex_search(fileName, match, tagRegex))
		{
		  // The 1st captured group (index 1) contains
		  // the double value within parentheses...
		  
		  double tagValue = stod(match[1].str());

		  // Now you can use 'tagValue' as needed:
		  
		  cout << " File: " << fileName
		       << ", Tag: " << tagValue << endl;
		}
	      else
		{ cerr << " Error: Unable to find tag in"
		       << " file name: " << fileName << endl;
		}
	    }
	}
    }
}

//==========================================
// Function to copy the data line associated
// with a given start value (sValue) from an
// input (fName2) to an output file (fName1):

void collect_data(const string &outDir,
		  const string &fName1,
		  const string &fName2,
		  const string &sValue)
{ 
  // Check and set output file:

  int key = 0;
  
  ifstream fileExists(fName1); 

  if (!fileExists){key = 1;}

  ofstream outFile(fName1, ios::app | ios::out);

  // Open the file for copying
  // and iterate over the lines:
	      
  ifstream inpFile(fName2);
		  
  string line;
		  
  while (getline(inpFile, line))
    {
      // Copy the 1st line if key = 1:
      
      if (key == 1)
	{
	  outFile << line << endl;

	  key = 0;
	}
      
      // Check if the line starts with the
      // given value, if so, copy the line:
		      
      if (line.find(sValue) == 0)
	{		  
	  outFile << line << endl;
	}
    }

  // Close the output files:

  inpFile.close();
  
  outFile.close();
}

//=============================================
// Transfer data between double type 2D-pointer
// ptr2d to character type 1D-pointer buffer...

void set_BufferData(int N, size_t Sz,
		    double **ptr2d, char *buffer)
{
  size_t offset = 0;
			  
  for (int k = 0; k < N; k++)
    {			      
      memcpy(&buffer[offset],
	     reinterpret_cast
	     <const char*>(ptr2d[k]), Sz);

      offset += Sz;
    }
}

void get_BufferData(int N, size_t Sz,
		    char *buffer, double **ptr2d)
{
  size_t offset = 0;
			  			  
  for (int k = 0; k < N; k++)
    {
      memcpy(reinterpret_cast
	     <char*>(ptr2d[k]),
	     &buffer[offset], Sz);
			      
      offset += Sz;
    }
}
  
/* *******************************************************
   Below, the subroutines and functions are based on compo-
   nents and objects within the OpenCV library, it is good
   to know that there are two vector-like objects that are
   similar to each other in structure and in name:

   Vec3d:
   ------
   represents a 3-channel pixel using 64-bit double-preci-
   sion floating-point numbers. It is used when high preci-
   sion is needed, and you need to work with floating-point
   values for image processing. Each channel of Vec3d can
   store real numbers with a wide range and high precision.
   This type is useful for certain computer vision and sci-
   entific applications that require precise calculations.

   Vec3b:
   ------
   represents a 3-channel pixel using 8-bit unsigned
   integers (uchar). It is typically used for 3-channel
   color images in the BGR color space. Each channel of
   Vec3b can store integer values in the range [0, 255].
   This type is suitable for most standard color images
   and is memory-efficient, as it uses 8 bits per channel.
   
   Note: if OpenCV is absent, some subroutines become un-
   avaiable, but the most important ones are still functi-
   onal through the backup code "structures.cpp" with the
   essential C++ structures needed; */

//========================================
// Return average value of the components:

double vecAvg2d(const Vec2d &vec)
{
  const double fc = 0.5;
  
  return (fc * (vec[0] + vec[1]));
}

double vecAvg3d(const Vec3d &vec)
{
  const double fc = 1.0 / 3.0;
  
  return (fc * (vec[0] + vec[1] + vec[2]));
}

double vecAvg4d(const Vec4d &vec)
{
  const double fc = 0.25;
  
  return (fc * (vec[0] + vec[1] + vec[2] + vec[3]));
}

//==============================================
// Return squared form (each component squared):

Vec2d sqrForm2d(const Vec2d &v1)
{
  Vec2d v2;
     
  v2[0] = pow(v1[0], 2);
  v2[1] = pow(v1[1], 2);

  return v2;
}

Vec3d sqrForm3d(const Vec3d &v1)
{
  Vec3d v2;
     
  v2[0] = pow(v1[0], 2);
  v2[1] = pow(v1[1], 2);
  v2[2] = pow(v1[2], 2);

  return v2;
}

Vec4d sqrForm4d(const Vec4d &v1)
{
  Vec4d v2;
     
  v2[0] = pow(v1[0], 2);
  v2[1] = pow(v1[1], 2);
  v2[2] = pow(v1[2], 2);
  v2[3] = pow(v1[3], 2);

  return v2;
}

//=====================================================
// Return absolute form (abs. value of each component):

Vec2d absForm2d(const Vec2d &v1)
{
  Vec2d v2;
     
  v2[0] = abs(v1[0]);
  v2[1] = abs(v1[1]);

  return v2;
}

Vec3d absForm3d(const Vec3d &v1)
{
  Vec3d v2;
     
  v2[0] = abs(v1[0]);
  v2[1] = abs(v1[1]);
  v2[2] = abs(v1[2]);

  return v2;
}

Vec4d absForm4d(const Vec4d &v1)
{
  Vec4d v2;
     
  v2[0] = abs(v1[0]);
  v2[1] = abs(v1[1]);
  v2[2] = abs(v1[2]);
  v2[3] = abs(v1[3]);

  return v2;
}

//======================================================
// Return dot-product of two Vec2d (const-type) objects:

double dotProduct2d(const Vec2d &v1, const Vec2d &v2)
{
  return ( v1[0] * v2[0] + v1[1] * v2[1] );
}

double dotProduct4d(const Vec4d &v1, const Vec4d &v2)
{
  return ( v1[0] * v2[0] + v1[1] * v2[1] +
	   v1[2] * v2[2] + v1[3] * v2[3] );
}

//=====================================================
// Weighted dot product (weight factors given by 'wt'):

double wdotProduct2d(const Vec2d &wt,
		     const Vec2d &v1,
		     const Vec2d &v2)
{
  return ( wt[0] * v1[0] * v2[0] + wt[1] * v1[1] * v2[1] );
}

double wdotProduct4d(const Vec4d &wt,
		     const Vec4d &v1,
		     const Vec4d &v2)
{
  return ( wt[0] * v1[0] * v2[0] + wt[1] * v1[1] * v2[1] +
	   wt[2] * v1[2] * v2[2] + wt[3] * v1[3] * v2[3] );
}

//======================================================
// Return dot-product of two Vec3d (const-type) objects:

double dotProduct(const Vec3d &v1, const Vec3d &v2)
{
  return (
    v1[0] * v2[0] +
    v1[1] * v2[1] +
    v1[2] * v2[2]); // 'v1' & 'v2' are not changed;
}

double wdotProduct(const Vec3d &wt, // Weighted dot product,
		   const Vec3d &v1, // the weight factors
		   const Vec3d &v2) // are given by 'wt';
{
  return (
    wt[0] * v1[0] * v2[0] +
    wt[1] * v1[1] * v2[1] +
    wt[2] * v1[2] * v2[2]);
}

//========================================================
// Return cross-product of two Vec3d (const-type) objects:

Vec3d crossProduct(const Vec3d &v1, const Vec3d &v2)
{
  Vec3d v3;
     
  v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
  v3[1] = v1[2] * v2[0] - v1[0] * v2[2];
  v3[2] = v1[0] * v2[1] - v1[1] * v2[0];

  return v3;
}

//==================================================
// Return the Vec3d resulting from a matrix product:
/*
  U[3] = MX[3 x 3] * V[3] , (fixed size)

  Above '*' stands for the product operator; */

Vec3d MxVecProduct(const double MX[9], const Vec3d &V)
{
  Vec3d U;
  
  U[0] = MX[0] * V[0] + MX[1] * V[1] + MX[2] * V[2];
  U[1] = MX[3] * V[0] + MX[4] * V[1] + MX[5] * V[2];
  U[2] = MX[6] * V[0] + MX[7] * V[1] + MX[8] * V[2];

  return U;
}

//========================
// Normalize Vec3d object:

Vec3d normVec3d(Vec3d &Vec)
{
  double vecNorm;
  
  Vec3d unitVec;

  unsigned int n;
     
  vecNorm = sqrt(dotProduct(Vec, Vec));

  for (n = 0; n < 3; n++)
    {
      unitVec[n] = Vec[n] / vecNorm;
    }

  return unitVec;
}

////////////////////////////// The subroutines below cannot
///// OpenCV CODES START ///// be used without the OpenCV lib;

#if WITH_OPENCV == 1

Vec3b scalarToVec3b(const Scalar& s)
{
  return Vec3b(
	       static_cast<uchar>(s[0]),
	       static_cast<uchar>(s[1]),
	       static_cast<uchar>(s[2])
	       );
}

//======================
// Get image (Mat) type:

string type2str(int type)
{
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
  case CV_8U:  r = "8U";   break;
  case CV_8S:  r = "8S";   break;
  case CV_16U: r = "16U";  break;
  case CV_16S: r = "16S";  break;
  case CV_32S: r = "32S";  break;
  case CV_32F: r = "32F";  break;
  case CV_64F: r = "64F";  break;
  default:     r = "User"; break;
  }

  r += "C";         // The number after the letter 'C' re-
  r += (chans+'0'); // fers to the number of color channels;

  return r;
}

//=========================================================
// Colored (3-channel) Mat ---> Gray-scale (1-channel) Mat:

void bgr2gray(const Mat &c3Frame, Mat &c1Frame)
{
  const unsigned int cnum = c3Frame.cols;

  const unsigned int rnum = c3Frame.rows;

  unsigned int i, j, pv;

  Vec3b pixel; double x0, fc = 1.0 / 3.0;

  Point pt(-1, -1);

  //-----------------
  // Conversion loop:
  
  for   (i = 0; i < cnum; i++){     
    for (j = 0; j < rnum; j++)
      {
	pt.x = i; // Target pixel
	pt.y = j; // coordinates;
		
	pixel = c3Frame.at<Vec3b>(pt);
	    
	pv = (pixel.val[0] +
	      pixel.val[1] +
	      pixel.val[2]);

	x0 = fc * (double)pv;
		      
	c1Frame.at<uchar>(pt) = round(x0);
      }
  }
}

///// OpenCV CODES END /////
////////////////////////////

#endif

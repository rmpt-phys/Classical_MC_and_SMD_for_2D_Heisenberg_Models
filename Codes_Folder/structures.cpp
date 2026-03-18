/* using namespace std; */

//=======================
// Replacement for Vec2d:

struct Vec2d
{ 
  double data[2];

  // Define basic structure:
  
  Vec2d(double x_val, double y_val)
  {
    data[0] = x_val;
    data[1] = y_val;
  }

  Vec2d()
  {
    data[0] = 0.0;
    data[1] = 0.0;
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
  
  Vec2d operator+(const Vec2d &other) const
  {
    return Vec2d(data[0] + other[0],
		 data[1] + other[1]);
  }

  // Overload "-" operator for vector subtraction:
  
  Vec2d operator-(const Vec2d &other) const
  {
    return Vec2d(data[0] - other[0],
		 data[1] - other[1]);
  }

  // In-place addition operator:
  
  Vec2d &operator+=(const Vec2d &other)
  {
    data[0] += other[0];
    data[1] += other[1];
    
    return *this;
  }
};

//=======================
// Replacement for Vec3d:

struct Vec3d
{ 
  double data[3];

  // Define basic structure:
  
  Vec3d(double x_val, double y_val, double z_val)
  {
    data[0] = x_val;
    data[1] = y_val;
    data[2] = z_val;
  }

  Vec3d()
  {
    data[0] = 0.0;
    data[1] = 0.0;
    data[2] = 0.0;
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
  
  Vec3d operator+(const Vec3d &other) const
  {
    return Vec3d(data[0] + other[0],
		 data[1] + other[1],
		 data[2] + other[2]);
  }

  // Overload "-" operator for vector subtraction:
  
  Vec3d operator-(const Vec3d &other) const
  {
    return Vec3d(data[0] - other[0],
		 data[1] - other[1],
		 data[2] - other[2]);
  }

  // In-place addition operator:
  
  Vec3d &operator+=(const Vec3d &other)
  {
    data[0] += other[0];
    data[1] += other[1];
    data[2] += other[2];
    
    return *this;
  }
};

//=======================
// Replacement for Vec4d:

struct Vec4d
{ 
  double data[4];

  // Define basic structure:
  
  Vec4d(double x_val, double y_val,
	double z_val, double w_val)
  {
    data[0] = x_val;
    data[1] = y_val;
    data[2] = z_val;
    data[3] = w_val;
  }

  Vec4d()
  {
    data[0] = 0.0;
    data[1] = 0.0;
    data[2] = 0.0;
    data[3] = 0.0;
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
  
  Vec4d operator+(const Vec4d &other) const
  {
    return Vec4d(data[0] + other[0],
		 data[1] + other[1],
		 data[2] + other[2],
		 data[3] + other[3]);
  }

  // Overload "-" operator for vector subtraction:
  
  Vec4d operator-(const Vec4d &other) const
  {
    return Vec4d(data[0] - other[0],
		 data[1] - other[1],
		 data[2] - other[2],
		 data[3] - other[3]);
  }

  // In-place addition operator:
  
  Vec4d &operator+=(const Vec4d &other)
  {
    data[0] += other[0];
    data[1] += other[1];
    data[2] += other[2];
    data[3] += other[3];
    
    return *this;
  }
};

//===============================
// Non-member function for scalar
// multiplication with scalar
// on the left...

Vec2d operator*(double scalar, const Vec2d &vec)
{
  return Vec2d(scalar * vec[0],
	       scalar * vec[1]);
}

Vec3d operator*(double scalar, const Vec3d &vec)
{
  return Vec3d(scalar * vec[0],
	       scalar * vec[1],
	       scalar * vec[2]);
}

Vec4d operator*(double scalar, const Vec4d &vec)
{
  return Vec4d(scalar * vec[0],
	       scalar * vec[1],
	       scalar * vec[2],
	       scalar * vec[3]);
}

//===============================
// Non-member function for scalar
// multiplication with scalar
// on the right...

Vec2d operator*(const Vec2d &vec, double scalar)
{
  return Vec2d(scalar * vec[0],
	       scalar * vec[1]);
}

Vec3d operator*(const Vec3d &vec, double scalar)
{
  return Vec3d(scalar * vec[0],
	       scalar * vec[1],
	       scalar * vec[2]);
}

Vec4d operator*(const Vec4d &vec, double scalar)
{
  return Vec4d(scalar * vec[0],
	       scalar * vec[1],
	       scalar * vec[2],
	       scalar * vec[3]);
}

//=======================
// Replacement for Point:

struct Point
{
  int x, y;

  // Default constructor:
  
  Point() : x(0), y(0) {}

  // Parameterized constructor:
  
  Point(int xVal, int yVal) : x(xVal), y(yVal) {}
};

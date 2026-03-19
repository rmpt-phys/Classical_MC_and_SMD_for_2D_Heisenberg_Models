# OpenCV Installation Guide (Ubuntu)

A step-by-step guide for building and installing OpenCV from source on Ubuntu, including FFmpeg and optional VTK support.

---

## Table of Contents

1. [Part 1 — FFmpeg](#part-1--ffmpeg)
2. [Part 2 — VTK (Optional)](#part-2--vtk-optional)
3. [Part 3 — OpenCV](#part-3--opencv)
4. [Troubleshooting](#troubleshooting)

---

## Part 1 — FFmpeg

> FFmpeg must be built from source for best compatibility with OpenCV. **Version 3.x is preferred** — version 4+ can cause issues with OpenCV 4 or later.

### 1.1 Install Dependencies

```bash
sudo apt install build-essential nasm yasm
sudo apt-get install libx264-dev             # H.264 encoder
sudo apt-get install libx265-dev libnuma-dev # H.265/HEVC encoder
sudo apt-get install libvpx-dev              # VP8/VP9 encoder/decoder
```

### 1.2 Configure & Build

Run the following from the FFmpeg source directory:

```bash
./configure \
  --enable-shared \
  --enable-gpl \
  --enable-libx264 \
  --enable-libx265 \
  --enable-libvpx \
  --disable-asm \
  --disable-doc

make -j4 && sudo make install
```

> **Important:** Build with `--enable-shared` so OpenCV can link against FFmpeg at runtime.

### 1.3 Fix Shared Library Errors

If you get an error like:

```
ffmpeg: error while loading shared libraries: libavdevice.so.57: cannot open shared object file
```

Fix it by registering the library path:

```bash
# 1. Find where the library was installed

sudo find / -name "libavdevice.so.57"

# Example result: /usr/local/lib

# 2. Add the path to the dynamic linker config

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/ffmpeg.conf

# 3. Refresh the linker cache

sudo ldconfig

# 4. Verify FFmpeg works

ffmpeg --version
```

---

## Part 2 — VTK (Optional)

> VTK (Visualization Toolkit) is only needed if you require 3D visualization support in OpenCV. Skip this section if not needed.

### 2.1 Install Dependencies

```bash
sudo apt-get install libgl1-mesa-dev libglu1-mesa-dev mesa-utils

# Verify OpenGL is available
glxinfo | grep OpenGL
```

### 2.2 Build & Install VTK

```bash
cd vtk-master
mkdir -p build && cd build

cmake \
  -D BUILD_SHARED_LIBS=ON \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_C_COMPILER='/opt/intel/oneapi/2025.0/bin/icx' \
  -D CMAKE_CXX_COMPILER='/opt/intel/oneapi/2025.0/bin/icpx' \
  -D CMAKE_INSTALL_PREFIX='/usr/local' \
  ../

make && sudo make install
```

> Replace the Intel compiler paths with your own if using GCC or Clang.

---

## Part 3 — OpenCV

### 3.1 Install GUI & Build Dependencies

Choose **one** GUI backend (GTK+ is recommended for headless/server setups):

```bash
sudo apt-get install libgtk-3-dev # GTK+ (recommended)
# OR
sudo apt-get install qtbase5-dev  # Qt5
```

### 3.2 Prepare the Build Directory

```bash
# opencv-master = your OpenCV source folder (e.g., opencv-4.x.y)
cd opencv-master
mkdir -p build && cd build
```

### 3.3 Configure BLAS/LAPACK Paths

Before running CMake, add the system library path so OpenCV can find BLAS/LAPACK:

1. Open `cmake/OpenCVFindOpenBLAS.cmake` in the OpenCV source root.
2. Add `/usr/lib/x86_64-linux-gnu` to both `Open_BLAS_INCLUDE_SEARCH_PATHS` and `Open_BLAS_LIB_SEARCH_PATHS`.

### 3.4 Configure Intel MKL (Optional)

If using Intel MKL, edit `cmake/OpenCVFindMKL.cmake`:

```cmake
# Change this line:

list(APPEND mkl_root_paths "/opt/intel/mkl/")

# To:

list(APPEND mkl_root_paths "/opt/intel/oneapi/mkl/")
```

### 3.5 Set Library Paths

Make sure required libraries are visible at build time:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Add any other needed paths, e.g. Intel OneAPI:

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/oneapi/...
```

### 3.6 Run CMake

**Recommended configuration (GTK+, FFmpeg, no Qt):**

```bash
cmake \
  -D WITH_FFMPEG=ON \
  -D BUILD_SHARED_LIBS=OFF \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D OPENCV_GENERATE_PKGCONFIG=YES \
  -D WITH_QT=OFF \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  ../
```

**Full-featured configuration (Clang, Qt, TBB, OpenCL, OpenGL):**

```bash
cmake \
  -D HAVE_MKL=ON \
  -D WITH_IPP=ON \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_C_COMPILER='/usr/bin/clang-17' \
  -D CMAKE_CXX_COMPILER='/usr/bin/clang++-17' \
  -D OPENCV_GENERATE_PKGCONFIG=YES \
  -D WITH_QT=ON \
  -D WITH_TBB=ON \
  -D WITH_V4L=ON \
  -D WITH_FFMPEG=ON \
  -D WITH_OPENGL=ON \
  -D WITH_OPENCL=ON \
  -D WITH_GSTREAMER=OFF \
  -D CMAKE_INSTALL_PREFIX='/usr/local' \
  ../
```

**GStreamer configuration (alternative backend):**

```bash
cmake \
  -D CMAKE_C_COMPILER='/usr/bin/clang-10' \
  -D CMAKE_CXX_COMPILER='/usr/bin/clang++-10' \
  -D OPENCV_GENERATE_PKGCONFIG=YES \
  -D WITH_TBB=ON \
  -D WITH_V4L=ON \
  -D WITH_QT=ON \
  -D WITH_OPENGL=ON \
  -D WITH_GSTREAMER=ON \
  -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX='/usr/local' \
  ../
```

> **Key flag:** `-D OPENCV_GENERATE_PKGCONFIG=YES` is essential — it generates the `opencv4.pc` file used for compilation later.

### 3.7 Check CMake Output

After CMake finishes, review the configuration summary. If FFmpeg is listed as **not found**, expose its paths manually:

```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
export PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR:/usr/local/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
```

Then re-run the CMake command.

### 3.8 Build & Install

```bash
# Build (logs saved to build.log for easy review)

make -j$(nproc) 2>&1 | tee build.log

# Install

sudo make install
```

### 3.9 Compile Your Code

Use `pkg-config` to automatically include all necessary paths and flags:

```bash
g++ -o my_program my_program.cpp $(pkg-config --cflags --static --libs opencv4)
```

---

## Troubleshooting

### Shared Library Not Found at Runtime

If your compiled binary fails with a missing `libopencv_*.so` error:

```bash
# 1. Find the library

sudo find / -name "libopencv_core.so*"

# Example result: /usr/local/lib/libopencv_core.so.4.x

# 2. Register the path

echo "/usr/local/lib" | sudo tee /etc/ld.so.conf.d/opencv.conf

# 3. Refresh the linker cache

sudo ldconfig -v

# 4. Re-run your binary
```

### FFmpeg Not Detected by OpenCV CMake

- Ensure FFmpeg was built with `--enable-shared`.
- Set `PKG_CONFIG_PATH` and `LD_LIBRARY_PATH` to point to FFmpeg's install location before running CMake.
- Prefer FFmpeg 3.x over 4.x for OpenCV 4 compatibility.

### BLAS/LAPACK Not Found

- Confirm you edited `OpenCVFindOpenBLAS.cmake` to include `/usr/lib/x86_64-linux-gnu`.
- Check that `liblapacke` is installed: `sudo apt-get install liblapack-dev liblapacke-dev`.

---

## FFmpeg compilation reference

- [FFmpeg Compilation Guide (Ubuntu)](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu)

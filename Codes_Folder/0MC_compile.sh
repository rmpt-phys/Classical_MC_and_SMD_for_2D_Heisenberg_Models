#!/bin/bash

#-----------------------
# Check input arguments:

if [[ -z $1 ]];
then
    vmode="OpenCV_OFF"
else
    vmode=$1
fi

if [[ $vmode == "OpenCV_OFF" ]];
then
    export WITH_OPENCV=0
else
    export WITH_OPENCV=1
fi

if [[ -z $2 ]];
then
    amode="AutoStart_OFF"
else
    amode=$2
fi

if [[ $amode == "AutoStart_OFF" ]];
then
    export AUTO_START=0
else
    export AUTO_START=1
fi

warns="OFF"

cmode="${3:-O1}"

case "$cmode" in
    O0|O1|O2) ;;
    WR)
        cmode="O0"
        warns="ON"
        ;;
    *)
        cmode="O0"
        ;;
esac

CM=${cmode: -1}

#----------------------
# Set compilation mode:

if [ $WITH_OPENCV == 0 ];
then
    echo; echo " > OpenCV is disabled;"; key="0";
else
    echo; echo " > OpenCV is enabled;";  key="1";
fi

if [ $AUTO_START == 0 ];
then
    echo; echo " > Simulation will start after confirmation;";
else
    echo; echo " > Simulation will start immediately;";
fi

#----------------
# Set code files:

cpp_code="mc_hm_code";

cpp_exe="xMC_run";

cpp_file=($cpp_code".cpp");

#---------------------
# OpenCV library link:

if [[ $key == "1" ]];
then
    cvlib="`pkg-config --cflags --libs opencv4`"
fi

#--------------------------
# MKL + FFTW library links:

mklib="-I"$MKLROOT"/include/fftw "

mklib+="-L"$MKLROOT"/lib/intel64 "

mklib+="-lmkl_intel_lp64 -lmkl_sequential "

mklib+="-lmkl_core -lpthread -lm -ldl"

#-------------------
# Diagnostics flags:

DIAG_FLAGS=""

SANITIZE_FLAGS=""

if [[ "$warns" == "ON" ]]
then
    DIAG_FLAGS="-g3"
    
    SANITIZE_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
fi

#--------------------------------
# Remove existing compiled files:

if [[ -e $cpp_exe ]]; then rm $cpp_exe; fi

#-----------------------
# Compile C++ main code: (oneAPI compiler: icpx / mpicxx)
#
# Note 1: if FFTW is needed, add -lfftw3 below;
# -------
#
# Note 2: we disabled the "fast_math" from Intel's
# ------- oneAPI "icpx" C-compiler via "-fno-fast-math";

printf "\n > Compiling $cpp_code with O${CM} flag ...\n"    

if [[ $key == "1" ]];
then
    mpicxx $cpp_file $mklib -lstdc++fs -std=c++17 \
	   -DAUTO_START=$AUTO_START \
	   -DWITH_OPENCV=$WITH_OPENCV $cvlib \
	   -I/usr/local/include -O${CM} -march=native \
	   -o $cpp_exe -msse2 -DHAVE_SSE2 -DDSFMT_MEXP=216091 \
	   -Wno-unused-variable -Wno-unused-value -fno-fast-math \
	   -lstdc++fs -std=c++17 -lgsl -lgslcblas -lm \
	   $DIAG_FLAGS $SANITIZE_FLAGS
else
    mpicxx $cpp_file $mklib \
	   -DAUTO_START=$AUTO_START \
	   -DWITH_OPENCV=$WITH_OPENCV \
	   -O${CM} -march=native -o $cpp_exe \
	   -msse2 -DHAVE_SSE2 -DDSFMT_MEXP=216091 \
	   -Wno-unused-variable -Wno-unused-value -fno-fast-math \
	   -lstdc++fs -std=c++17 -lgsl -lgslcblas -lm \
	   $DIAG_FLAGS $SANITIZE_FLAGS
fi

if [[ -e $cpp_exe ]];
then
    printf "\n > Done!\n\n"
else
    printf "\n > Error!\n\n"
fi


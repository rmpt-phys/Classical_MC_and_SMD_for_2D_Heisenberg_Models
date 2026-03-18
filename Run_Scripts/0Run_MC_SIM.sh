#!/bin/bash

export XDG_SESSION_TYPE=x11

export QT_QPA_PLATFORM=xcb

#---------------------------------
# Check input (number of threads):

if [[ -z $1 || $1 -lt 1 ]];
then
    NP="1"
else
    NP=$1
fi

EXC="xMC_run"

if [ ! -f "$EXC" ]; 
then
    printf "\n No executable found!\n\n"; exit 0;
fi

#--------------------
# Run MPI executable: --bind-to core --map-by core

mpirun -genv I_MPI_PIN_DOMAIN=auto:compact --np $NP ./$EXC

# I_MPI_PIN_DOMAIN=auto:compact
# I_MPI_PIN_DOMAIN=core

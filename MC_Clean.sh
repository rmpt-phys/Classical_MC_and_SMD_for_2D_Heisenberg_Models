#!/bin/bash

#------------------
# Check user input:

printf "\n > Directories to be cleaned: \n\n"

printf " outBin, outData, outFigs & outVids \n"

printf "\n > Proceed? (1/0) >> "

read -r OPT1 </dev/tty; echo

if [[ -z $OPT1 ]]; then OPT1=1; fi

if [[ $OPT1 == 0 ]];
then
    printf " > Operation cancelled..."

    printf "\n\n"; exit 0
fi

#-------------------
# Clean directories:

dir1="outBin"

dir2="outData"

dir3="outFigs"

dir4="outVids"

if [[ -d $dir1 ]]
then    
    echo " > Delete all bin-files inside '$dir1'?"
    echo
    read -r -p " > Type 'yes' to confirm (enter to 'no'): " answer
    echo

    if [[ "$answer" == "yes" ]]
    then
        find "$dir1" -type f -name "*.bin" -delete
    else
        find "$dir1" -type f -name "MC_Bin_*" -delete
        find "$dir1" -type f -name "MC_Annl*" -delete
    fi    
else
    printf " > Direc. $dir2 not found...\n\n"
fi

if [[ -d $dir2 ]]
then    
    find "$dir2" -type f -delete
else
    printf " > Direc. $dir2 not found...\n\n"
fi

if [[ -d $dir3 ]]
then    
    find "$dir3" -type f -delete
else
    printf " > Direc. $dir3 not found...\n\n"
fi

if [[ -d $dir4 ]]
then    
    find "$dir4" -type f -delete
else
    printf " > Direc. $dir4 not found...\n\n"
fi

printf " > Done!\n\n"

exit 0

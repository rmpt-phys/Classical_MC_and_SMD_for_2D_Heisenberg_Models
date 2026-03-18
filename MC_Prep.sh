#!/bin/bash

#----------------------------------------------------
# Get full directory path & define code souce folder:

# Absolute path of this script

script_path="$(readlink -f "$0")"

# Directory where the script is located

script_dir="$(dirname "$script_path")"

# The source folder name is the directory name

src="$(basename "$script_dir")"

# The parent directory of the src folder

cd "$script_dir/.."

dir0="$(pwd)"

echo; echo " > Running MC-preparation script..."; echo

echo " - Source folder: $src"
echo " - Parent folder: $dir0"

# Continue with your code

cd "$script_dir"

#-----------------------------
# Get simulation 'tag' string:

printf "\n > Simulation folder tag: (string) "

read -r tag </dev/tty; echo

if [[ -z $tag ]]
then
    tag="0Test"
fi

#--------------------------
# Define directory strings:

smdir0="HM_SIM_($tag)"

smdir1="HM_SIM_\($tag\)"

ipdir=$dir0; ipdir+="/$src"
  
dwork=$dir0; dwork+="/Work_Dir"

scdir0="$dwork/$smdir0"

scdir1="$dwork/$smdir1"

vidDir="$scdir1""/outVids"

#------------------
# Check user input:

printf " > Directory to be created:"

printf "\n\n"

printf "   $scdir0;\n\n > Proceed? (1/0) >> "

read -r OPT1 </dev/tty; echo

if [[ -z $OPT1 ]]; then OPT1=1; fi

if [[ $OPT1 == 0 ]];
then
    printf " > Operation cancelled..."

    printf "\n\n"; exit 0
fi

#-------------------------
# Check working directory:

if [[ $OPT1 == 1 ]] && [[ ! -d "$dwork" ]]
then    
    printf " > Error: Work_Dir directory does not exist!\n"
    printf "   ------ The following direc. is needed: \n\n"
    
    echo "   "$dwork; echo; exit 0
fi

#----------------------------------------------
# Check simulation directory & get user choice:

if [[ -d "$scdir0" ]]
then
    printf " > Directory exists, options: (default 1)\n\n"
    printf "   1) Use current (recover);\n"
    printf "   2) Erase contents;       \n"
    printf "   0) Cancel operation;     \n "
    
    printf " \n >> "

    read -r OPT1 </dev/tty; echo

    if [[ -z $OPT1 ]]; then OPT1=1; fi

    if [[ $OPT1 == 0 ]];
    then
	printf " > Operation cancelled...\n\n"; exit 0
    fi
    
    if [[ $OPT1 == 2 ]];
    then
	if [ "$(ls -A $scdir0)" ];
	then
	    cd $dwork && echo "rm -r $scdir1/*" | bash
	fi
    fi
else
    cd $dwork && echo "mkdir $scdir1" | bash && OPT1=2
fi

#------------------------------
# Prepare simulation directory:

cd $scdir0

if [[ $OPT1 == 2 ]]
then
    makedir="mkdir outBin outData outFigs outVids Plot_Area"

    echo "$makedir" | bash && cd outData/

    makedir1="mkdir MC_Results MC_Collect Code_Check"
    
    makedir2="mkdir PT_Check RK_Results RK_Check"

    echo "$makedir1" | bash
    echo "$makedir2" | bash && cd ../outFigs/
    
    makedir="mkdir Samples_Plot"
    
    echo "$makedir" | bash && cd $dwork
else
    if [[ -d "outBin" ]]
    then
	if [ "$(ls -A outBin)" ]
	then
	    printf " > outBin seems fine, please check it!\n\n"
	else
	    printf " > outBin is empty (check directory);\n\n"
	fi
    else
	printf " > outBin is missing, cannot proceed;"

	printf "\n\n"; exit 0
    fi
fi

cd $ipdir

#-------------------------------------------
# Give executable permissions to the scripts:

cd Codes_Folder/

echo "chmod 755 0MC_compile.sh" | bash

cd ../Run_Scripts/

echo "chmod 755 0Run_MC_SIM.sh" | bash

cd ../

#------------------
# Compilation mode:

printf " > Enable OpenCV? NO (0|DF), YES (1) >> "

read -r OPT3 </dev/tty; echo

if [[ -z $OPT3 ]]; then OPT3=0; fi

if [[ $OPT3 == 0 ]]
then
    VM="OpenCV_OFF"
else    
    VM="OpenCV_ON"
fi   

printf " > Enable AutoStart? NO (0), YES (1|DF) >> "

read -r OPT4 </dev/tty; echo

if [[ -z $OPT4 ]]; then OPT4=1; fi

if [[ $OPT4 == 0 ]]
then
    AS="AutoStart_OFF"
else    
    AS="AutoStart_ON"
fi  

printf " > Choose compilation mode: normal (0|DF), opt (1), debug (2) >> "

read -r OPT5 </dev/tty

if [[ -z $OPT5 ]]; then OPT5=0; fi

cd Codes_Folder/

case "$OPT5" in
    1) MODE="O2" ;;
    2) MODE="WR" ;;
    *) MODE="O0" ;;
esac

echo "./0MC_compile.sh $VM $AS $MODE" | bash

cd ../

#------------------------------------
# Check executable & copy move files:

PROG="Codes_Folder/xMC_run"

FILE_CLEAN="MC_Clean.sh"

GRID_MAKER="MakeTGrid.py"

CONF_MAKER="MakeSConf.py"

RUN_SCRIPT="Run_Scripts/0Run_MC_SIM.sh"

CP_SILENCE="> /dev/null 2>&1" #( set to void to debug problems)

if [[ ! -e $PROG ]]
then
    printf "\n > Compilation failed...\n\n"; exit 0
else
    printf "\n > MC-preparation complete!\n\n"
    
    move="mv $PROG $scdir1"
    
    copy0="cp $FILE_CLEAN $scdir1"
    
    copy1="cp --no-clobber $GRID_MAKER $scdir1 $CP_SILENCE"

    copy2="cp --no-clobber $CONF_MAKER $scdir1 $CP_SILENCE"

    copy3="cp --no-clobber $RUN_SCRIPT $scdir1 $CP_SILENCE"
  
    echo "$move && $copy0 && $copy1 && $copy2 && $copy3" | bash
fi

echo "cp --no-clobber *.txt $scdir1 $CP_SILENCE" | bash

exit 0

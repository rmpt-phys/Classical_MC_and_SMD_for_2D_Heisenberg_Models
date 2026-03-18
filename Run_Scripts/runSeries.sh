#!/bin/bash

#####################
# Number of processes

if [[ -z $1 || $1 -lt 1 ]]
then
    NP=4
else
    NP=$1
fi

################################
# User menu: parameter selection

echo
echo ">> Script for multiple MC simulations:"
echo
echo "> Select the parameter to iterate:"
echo "  1) J1   (Line 2)"
echo "  2) J2   (Line 3)"
echo "  3) extH (Line 6)"
echo
read -p "> Enter option [1-3]: " opt
echo

case $opt in
    1)
        param_name="J1"
        param_tag="J1tag"
        line_num=2
        ;;
    2)
        param_name="J2"
        param_tag="J2tag"
        line_num=3
        ;;
    3)
        param_name="extH"
        param_tag="Htag"
        line_num=6
        ;;
    *)
        echo "> Invalid option. Exiting."
        echo
        exit 1
        ;;
esac

echo "> Loop on parameter: $param_name"
echo "> Output tag set to: $param_tag"
echo

#################
# Parameter range

read -p "> Enter initial value (v1): " v1
read -p "> Enter final value   (v2): " v2
read -p "> Enter step size  (vstep): " vstep

cmp=$(echo "$v2 < $v1" | bc)

if [[ $cmp -eq 1 && $(echo "$vstep >= 0" | bc) -eq 1 ]]; then
    echo
    echo "> ERROR:"
    echo "  v2 < v1 but vstep is not negative."
    echo "  Use a negative step when iterating downward."
    echo
    exit 1
fi

if [[ $cmp -eq 0 && $(echo "$vstep <= 0" | bc) -eq 1 ]]; then
    echo
    echo "> ERROR:"
    echo "  v2 > v1 but vstep is not positive."
    echo "  Use a positive step when iterating upward."
    echo
    exit 1
fi

############################
# Total number of iterations

npt=$(echo "($v2 - $v1) / $vstep + 1" | bc)

ic=0

###########
# Main loop

for val in $(seq $v1 $vstep $v2); do
    ###
    value=$(echo "$val" | tr ',' '.')

    echo
    echo " > Running simulation: $param_name = $value"

    # Replace selected line with new value
    
    sed -i "${line_num}s/.*/$value/" Settings_HM_SYS.txt
    
    # Ensure output files tag is the
    # associated with the varying parameter
    
    sed -i "5s/.*/$param_tag/" Settings_MC_SIM.txt

    # Run simulation
    
    ./0Run_MC_SIM.sh "$NP" >/dev/null 2>&1

    ((ic++))

    progress=$(echo "($ic * 100) / $npt" | bc)

    echo " > Completed: $param_name = $value"
    
    echo " > Progress: $progress %"
done

echo
echo " > All simulations completed!"
echo

#!/bin/bash
# Distribute all json jobs to the cluster nodes

# General settings

TIMEOUT="24h"                                # A node times out after this amount of time
FOMO_DIRECTORY="/storage/users/$USER/fomo"   # Where to find the FOMO directory 

# The following settings are only used when finding nodes automatically:

NODE_ROWS="13456789"  # The node rows to pick from
MAX_NODES=2           # Maximum number of nodes
LOAD_LIMIT=2          # Don't use a node if it has a higher percent load than this

###############################################################################
# Nodes specified on the command line

if [ $# -gt 0 ] ; then

    while getopts ":r:n:" arg; do
        case ${arg} in
            r)
                ROW=${OPTARG}
                ;;
            n)
                NODENUMS=`echo ${OPTARG} | awk -v RS='[,\n]' -F: 'NF == 1; NF == 2 { for (i = $1; i <= $2; ++i) print i }'`
                ;;
            :)
                echo "$0: Must supply an argument to -$OPTARG." >&2
                exit 1
                ;;
            ?)
            echo "Invalid option: -${OPTARG}."
            exit 2
            ;;
        esac
    done

    NODES=""

    for i in $NODENUMS; do
        NODES="${NODES} compute-${ROW}-$i.local"
    done

###############################################################################
# Nothing from command line, find nodes automatically

else
    NODES=(`gstat -i 10.1.1.1 -a1l | grep compute-[${NODE_ROWS}] | awk -v limit=${LOAD_LIMIT} '{ if(strtonum($7) < limit) printf("%s\n", $1); }' | sort -r | head -n ${MAX_NODES}`)
fi

###############################################################################

RUNS=(`ls experimental_setups`)

num_nodes=${#NODES[@]}
num_runs=${#RUNS[@]}

runs_per_node=`python3 -c "import math; print(math.ceil($num_runs / $num_nodes))"`

node_counter=0
run_counter=0

rm -rf *.out *.err output.csv

while [ $run_counter -lt $num_runs ]; do
    node=${NODES[$node_counter]}
    node_counter=$((node_counter + 1))

    echo "Sending to $node:"

    args=""
    for (( i=0 ; i < $runs_per_node ; i++ )); do
        run=${RUNS[$run_counter]}
        echo "  " $run
        run_counter=$((run_counter + 1))
        args="$args experimental_setups/$run"
    done

    ssh $node "cd ${FOMO_DIRECTORY}; shopt -s huponexit; timeout ${TIMEOUT} python3 run.py $args" > ${node}.out 2> ${node}.err &

    echo
done

trap "kill 0" SIGINT

echo "Waiting for completion"
wait

#!/bin/bash
# Distribute all json jobs to the cluster nodes given on the command line

# A node times out after this amount of time:
TIMEOUT="24h"

# Location of the fomo directory:
FOMO_DIR="/home/$USER/research/fomo"

###############################################################################

trap 'kill 0' INT

NODES=( $@ )

RUNS=(`ls experimental_setups`)

num_nodes=${#NODES[@]}
num_runs=${#RUNS[@]}

runs_per_node=`python3 -c "import math; print(math.ceil($num_runs / $num_nodes))"`

node_counter=0
run_counter=0

rm -rf *.out *.err output.csv

while [ $run_counter -lt $num_runs ]; do
    args=""

    for (( i=0 ; i < $runs_per_node ; i++ )); do
        run=${RUNS[$run_counter]}
        run_counter=$((run_counter + 1))
        args="$args experimental_setups/$run"
    done

    node=${NODES[$node_counter]}
    node_counter=$((node_counter + 1))

    echo "Sending to $node: $args"
    ssh $node "cd ${FOMO_DIR}; timeout ${TIMEOUT} python3 run.py $args" > ${node}.out 2> ${node}.err &
done

echo "Waiting for completion"
wait

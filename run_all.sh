#!/bin/bash
# Distribute all json jobs to the cluster nodes

# A node times out after this amount of time:
TIMEOUT="24h"

# These node groups are allowed:
NODE_GROUPS="13456789"

# Maximum number of nodes this script will use:
MAX_NODES=1000

# Don't use a node if it has a higher percent load than the following:
LOAD_LIMIT=2

###############################################################################

NODES=(`gstat -i 10.1.1.1 -a1l | grep compute-[${NODE_GROUPS}] | awk -v limit=${LOAD_LIMIT} '{ if(strtonum($7) < limit) printf("%s\n", $1); }' | sort -r | head -n ${MAX_NODES}`)

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
    ssh $node "cd /storage/users/$USER/fomo; timeout ${TIMEOUT} python3 run.py $args" > ${node}.out 2> ${node}.err &
done

echo "Waiting for completion"
wait

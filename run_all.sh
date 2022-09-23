#!/bin/bash
NODES=(`gstat -a1l | grep compute-1 | awk '{ if(strtonum($8) < 2) printf("%04.1f %s\n", $8, $1); }' | sort | awk '{ print $2 }'`)

RUNS=(`ls experimental_setups`)

num_nodes=${#NODES[@]}
num_runs=${#RUNS[@]}

runs_per_node=`python3 -c "import math; print(math.ceil($num_runs / $num_nodes))"`

node_counter=0
run_counter=0

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
    ssh $node "cd /storage/users/djupdal/fomo; python3 run.py $args" > ${node}.log &
done

echo "Waiting for completion"
wait

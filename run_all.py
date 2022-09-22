#!/bin/python3

import os

nodes = os.popen("gstat -a1l").read()

print(nodes)

# RUNS=(`ls experimental_setups`)

# NUM_NODES=${#NODES[@]}
# NUM_RUNS=${#RUNS[@]}

# echo $RUNS
# echo $NUM_RUNS

# for (( i=0; i<$NUM_RUNS; i++ ))
# do
#   echo "Node "$i" is computing"
#   echo screen -d -m -S node$i ssh "compute-"${NODES[$i]} run.py
# done

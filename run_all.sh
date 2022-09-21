#!/bin/bash

#NODES=`gstat -a1l | grep compute | awk '{ printf("%04.1f %s\n", $8, $1) }' | sort | awk '{ print $2 }'`
NODES=1

RUNS=(`ls experimental_setups`)

NUM_NODES=${#NODES[@]}
NUM_RUNS=${#RUNS[@]}

echo $RUNS
echo $NUM_RUNS

for (( i=0; i<$NUM_RUNS; i++ ))
do
  echo "Node "$i" is computing"
  echo screen -d -m -S node$i ssh "compute-"${NODES[$i]} run.py
done

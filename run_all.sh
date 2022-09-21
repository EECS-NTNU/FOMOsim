#!/bin/bash

#NODES=`gstat -a1l | grep compute | awk '{ if(strtonum($8) < 2) printf("%04.1f %s\n", $8, $1); }' | sort | awk '{ print $2 }'`
NODES=(1 2 3)
RUNS=(`ls experimental_setups`)

NUM_NODES=${#NODES[@]}
NUM_RUNS=${#RUNS[@]}

j=0

for (( i=0; i<$NUM_RUNS; i++ ))
do
  echo "Node "$j" is computing " ${RUNS[$i]}
  echo screen -d -m -S node$i ssh "compute-"${NODES[$i]} run.py ${RUNS[$i]}
done

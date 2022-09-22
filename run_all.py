#!/bin/python3

import os
import math

lines = os.popen("gstat -a1l").read().splitlines()

nodes = []

for line in lines:
    words = line.split()
    node = words[0]
    load = float(words[7][:-1])
    if load < 2: 
        nodes.append(node)

runs = os.listdir("experimental_setups")

runs_per_node = math.ceil(len(runs) / float(len(nodes)))

node_counter = 0

while len(runs) > 0:
    node = nodes[node_counter]
    node_counter += 1

    runs_for_node, runs = runs[:runs_per_node], runs[runs_per_node:]

    args = " experimental_setups/".join(runs_for_node)

    command = "screen -d -m -S " + node + " ssh " + node + " python3 run.py " + args

    os.system(command)

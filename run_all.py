#!/bin/python3

import os

nodes = os.popen("gstat -a1l | grep compute | awk '{ if(strtonum($8) < 2) printf("%04.1f %s\n", $8, $1); }' | sort | awk '{ print $2 }'").read()



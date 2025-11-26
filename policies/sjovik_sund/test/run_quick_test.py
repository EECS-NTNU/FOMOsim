import os
import sys

# Ensure repository root is on sys.path so imports work when running this test directly
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

from policies.sjovik_sund.test import test_subproblem_1 as tst
from init_state.wrapper import read_initial_state
import target_state

# Quick smoke test: 1 run, 1 vehicle
filename = "instances/TD_W34"
START_DAY = 0
START_HOUR = 8
DURATION = 60  # 1 hour
policy = tst.SjovikSundPolicy()

tstate = target_state.USTargetState()

# call test_policy with small params
tst.test_policy(filename, 1, START_DAY, START_HOUR, tstate, policy, DURATION, 1)

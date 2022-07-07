import jsonpickle
import hashlib
import os

import sim

savedStatesDirectory = "saved_states/"

def get_initial_state(source, target_state, **kwargs):
    checksum = hashlib.sha256(jsonpickle.encode(kwargs).encode('utf-8')).hexdigest()
    stateFilename = f"{savedStatesDirectory}/{source.__name__}_{target_state.__name__}_{checksum}.pickle.gz"

    if os.path.isdir(savedStatesDirectory):
        # directory with saved states exists
        if os.path.isfile(stateFilename):
            state = sim.State.load(stateFilename)
            return state

    state = source.get_initial_state(**kwargs)
    tstate = target_state(state)
    state.set_target_state(tstate)

    if not os.path.isdir(savedStatesDirectory):
        os.makedirs(savedStatesDirectory, exist_ok=True) # first time

    state.save(stateFilename)

    return state



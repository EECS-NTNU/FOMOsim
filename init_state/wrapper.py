import jsonpickle
import hashlib
import os

import sim

savedStatesDirectory = "saved_states/"

def get_initial_state(source, target_state, load_from_cache=True, **kwargs):
    all_args = {"source" : source, "target_state" : target_state}
    all_args.update(kwargs)
    checksum = hashlib.sha256(jsonpickle.encode(all_args).encode('utf-8')).hexdigest()
    stateFilename = f"{savedStatesDirectory}/{checksum}.pickle.gz"

    if load_from_cache:
        if os.path.isdir(savedStatesDirectory):
            # directory with saved states exists
            if os.path.isfile(stateFilename):
                print("Loading state from file")
                state = sim.State.load(stateFilename)
                return state

    state = source.get_initial_state(**kwargs)
    tstate = target_state(state)
    state.set_target_state(tstate)

    if not os.path.isdir(savedStatesDirectory):
        os.makedirs(savedStatesDirectory, exist_ok=True) # first time

    print("Saving state to file")
    state.save(stateFilename)

    return state



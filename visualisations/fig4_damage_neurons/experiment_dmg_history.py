import json
from functools import partial
import numpy as np

import sys
sys.path.insert(0, "../..")
import optimize

import tune_optimize_utils as utils


def disable_f(t, disable_count, disable_phase, state_neurons, time, phase):
    
    neuron_signal = [0]*state_neurons

    if disable_phase == "all" or disable_phase in phase:
        dmg_percentage = t/time
        for i in range(int(dmg_percentage*disable_count)):
            neuron_signal[i] = -30

    return neuron_signal


if __name__ == "__main__":

    for dmg_type in ["swing", "stance"]:
        dmg_disable_f = partial(disable_f, disable_count=7,
                                disable_phase=dmg_type)

        history, error, error_phase, _, _, _ = optimize.simulation_error(params=utils.best_params[0], 
                                          progress_bar=True, dmg_f=dmg_disable_f)

        print("error ", error)
        print("error_phase ", error_phase)

        for k in history.keys():
            history[k] = history[k][:, 0].tolist()

        json.dump(history, open(f"experiment_dmg_{dmg_type}_history.json", 'w'))

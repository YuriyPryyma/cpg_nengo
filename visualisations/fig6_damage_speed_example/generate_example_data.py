import os
import sys
sys.path.insert(0, "../..")
from optimize import simulation_error
import json
from tqdm import tqdm
import tune_optimize_utils as utils
from functools import partial
import multiprocessing as mp


def disable_f(t, state_neurons, time, phase):
    neuron_signal = [0]*state_neurons
 
    if "stance" in phase:
        if 5 <= t:
            for i in range(5):
                neuron_signal[i] = -30

    return neuron_signal

def speed_f(t):
    if t > 10:
        return 0.7
    else:
        return 0.1


if __name__ == "__main__":
    res = simulation_error(
        utils.best_params[0], 
        progress_bar=True,
        time=15,
        speed_f=speed_f,
        dmg_f=disable_f,
    )

    history, error, error_phase, _, _, _ = res

    print("error ", error)
    print("error_phase ", error_phase)

    for k in history.keys():
        history[k] = history[k][:, 0].tolist()

    json.dump(history, open(f"speed_recover_damage.json", 'w'))



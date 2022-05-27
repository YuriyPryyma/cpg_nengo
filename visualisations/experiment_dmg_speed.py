import os
import sys
sys.path.insert(0, "../src")
from optimize import simulation_error
import json
from tqdm import tqdm
import tune_optimize_utils as utils
from functools import partial
import multiprocessing as mp


def disable_f(t, disable_count, disable_phase, state_neurons, time, phase):
    neuron_signal = [0]*state_neurons

    if disable_phase == "all" or disable_phase in phase:
        if 5 <= t and t <= 25:
            for i in range(disable_count):
                neuron_signal[i] = -30

    return neuron_signal



if __name__ == "__main__":
    disable_count = 10
    speed = 0
    disable_phase = "stance"

    dmg_disable_f = partial(disable_f, disable_count=disable_count,
                            disable_phase=disable_phase)

    res = simulation_error(utils.best_params[0], 
                            progress_bar=True,
                            time=30,
                            speed_f= lambda _: speed,
                            dmg_f=dmg_disable_f,
                          )

    history, _, error_phase, _, _, _ = res

    print("error_phase ", error_phase)

    # for k in history.keys():
    #     history[k] = history[k][:, 0].tolist()

    # json.dump(history, open(f"test/experiment_dmg_{speed}_{disable_phase}_{disable_count}_history.json", 'w'))


    

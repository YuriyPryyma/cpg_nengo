import time
import multiprocessing as mp
from functools import partial
import json

from tqdm import tqdm
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

def simulation_dmg_error(arg):
    disable_phase, disable_count = arg
    dmg_disable_f = partial(disable_f, disable_count=disable_count,
                            disable_phase=disable_phase)
    _, error, error_phase, _, _, _ = optimize.simulation_error(params=utils.best_params[0], 
                              progress_bar=False, dmg_f=dmg_disable_f)

    return {
        "error":error,
        "error_phase":error_phase,
        "disable_count":disable_count,
        "disable_phase":disable_phase,
    }



if __name__ == "__main__":
    args = []
    args.extend([("all", i) for i in range(1, 11)])
    args.extend([("swing", i) for i in range(1, 11)])
    args.extend([("stance", i) for i in range(1, 11)])    
    args = args * 15

    pool = mp.Pool(mp.cpu_count())
    mapped_values = list(tqdm(pool.imap_unordered(simulation_dmg_error, args), 
                            total=len(args)))

    pool.close()
    pool.join()

    with open('dmg_swing_stance_15.json', 'w') as f:
        json.dump(mapped_values, f, indent=4)
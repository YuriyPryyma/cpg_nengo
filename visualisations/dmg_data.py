import time
import multiprocessing as mp
from functools import partial
import json

from tqdm import tqdm
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


def test(x):
    time.sleep(1)
    return x*x

if __name__ == "__main__":
    args = []
    neurons = list(range(1, 25))
    args.extend([("all", i) for i in range(1, 15)])
    args.extend([("swing", i) for i in range(1, 15)])
    args.extend([("stance", i) for i in range(1, 15)])    
    args = args * 3

    pool = mp.Pool(mp.cpu_count())
    mapped_values = list(tqdm(pool.imap_unordered(simulation_dmg_error, args), 
                            total=len(args)))

    pool.close()
    pool.join()


    with open('dmg_swing_stance_n.json', 'w') as f:
        json.dump(mapped_values, f, indent=4)
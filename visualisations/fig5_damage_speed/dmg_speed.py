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


def get_statistics(args):
    disable_count, speed, disable_phase = args

    dmg_disable_f = partial(disable_f, disable_count=disable_count,
                            disable_phase=disable_phase)

    res = simulation_error(utils.best_params[0], 
                            progress_bar=False,
                            time=30,
                            speed_f= lambda _: speed,
                            dmg_f=dmg_disable_f,
                          )

    _, _, error_phase, _, _, _ = res

    return {
        "disable_count":disable_count,
        "speed":speed,
        "disable_phase":disable_phase,
        "error_phase":error_phase,
    }



if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = '0'

    args = []
    args.extend([(disable_count, speed/50, "swing")
        for speed in range(0, 51) 
        for disable_count in range(0, 26)])
    args.extend([(disable_count, speed/50, "stance")
        for speed in range(0, 51) 
        for disable_count in range(0, 26)])

    pool = mp.Pool(mp.cpu_count())
    mapped_values = list(tqdm(pool.imap_unordered(get_statistics, args), 
                            total=len(args)))

    pool.close()
    pool.join()

    with open('dmg_speed_count_25_may.json', 'w') as f:
        json.dump(mapped_values, f, indent=4)
    

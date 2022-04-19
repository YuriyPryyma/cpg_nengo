import json

import sys
sys.path.insert(0, "../src")
import optimize
import numpy as np
import tune_optimize_utils as utils


if __name__ == "__main__":
    history, error, error_phase, _, _, _ = optimize.simulation_error(params=utils.best_params[0], 
                                      progress_bar=True)

    print("error ", error)
    print("error_phase ", error_phase)


    s1_swing_cycles, s1_stance_cycles = optimize.calc_swing_stance(history["s1_state"])
    swing_cycles_duration = [(right - left) / 1000
                         for left, right in s1_swing_cycles][1:]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in s1_stance_cycles][1:]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)
    true_s1_swing_duration = optimize.cycle_to_swing(combined_cycles)
    true_s1_stance_duration = optimize.cycle_to_stance(combined_cycles)

    for k in history.keys():
        history[k] = history[k][:, 0].tolist()

    history["s1_swing_cycles"] = s1_swing_cycles
    history["s1_stance_cycles"] = s1_stance_cycles
    history["true_s1_swing_duration"] = true_s1_swing_duration.tolist()
    history["true_s1_stance_duration"] = true_s1_stance_duration.tolist()

    json.dump(history, open("experiment_history.json", 'w'))

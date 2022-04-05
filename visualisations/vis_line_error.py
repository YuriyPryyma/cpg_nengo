import sys
sys.path.insert(0, "../src")


import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import optimize
import tune_optimize_utils as utils
from sklearn.metrics import r2_score

tau = 0.01

if __name__ == "__main__":
    history, error, error_phase, _, _, _ = optimize.simulation_error(params=utils.best_params[0], 
                                  progress_bar=True)

    print("error ", error)
    print("error_phase ", error_phase)

    for k in history.keys():
        history[k] = history[k][:, 0].tolist()

    json.dump(history, open("history_new.json", 'w'))

    # history = json.load(open("history.json"))

    plt.style.use('seaborn-pastel')

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42


    for leg in [1, 2]:

        fig = plt.figure(figsize=(10, 10))

        ax = plt.gca()

        state = np.array(history[f"s{leg}_state"])
        swing_cycles, stance_cycles = optimize.calc_swing_stance(state)

        swing_cycles_duration = [(right - left) / 1000
                             for left, right in swing_cycles][1:]

        stance_cycles_duration = [(right - left) / 1000
                                  for left, right in stance_cycles][1:]

        combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

        i = leg - 1

        true_swing_duration = optimize.cycle_to_swing(combined_cycles)

        true_stance_duration = optimize.cycle_to_stance(combined_cycles)

        swing_r2 = r2_score(true_swing_duration, swing_cycles_duration)
        stance_r2 = r2_score(true_stance_duration, stance_cycles_duration)


        fig.suptitle(f"A", fontsize=24)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        r2_swing_name = "$R^{2}_{swing}$"
        r2_stance_name = "$R^{2}_{stance}$"

        ax.text(0.05, 0.95, f"{r2_swing_name} = {swing_r2:.3f}\n{r2_stance_name} = {stance_r2:.3f}", 
                transform=ax.transAxes, fontsize=20,
                verticalalignment='top', bbox=props)

        ax.plot(combined_cycles, swing_cycles_duration, color='#F8550D', linestyle='dashed',
         marker='o', linewidth=1, label='Swing')
        ax.plot(combined_cycles, true_swing_duration, 
            color='black', linewidth=2, label="Halbertsma best-fit")

        ax.plot(combined_cycles, stance_cycles_duration, color='#0081D9', 
            linestyle='dashed', marker='o', linewidth=1, label='Stance')
        ax.plot(combined_cycles, true_stance_duration, 
            color='black', linewidth=2)

        ax.set_xlabel('Cycle duration, s', labelpad=10, fontsize=20)
        ax.set_xticks([0, 0.5, 1.0, 1.5, 2])
        ax.set_xlim([0, 2])

        ax.set_ylabel('Phase duration, s', labelpad=10, fontsize=20)
        ax.set_yticks([0, 0.5, 1.0, 1.5, 2])
        ax.set_ylim([0, 2])

        ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)


        fig.tight_layout()

        f_name = f"phase_durations_error"

        plt.show()
        # plt.savefig(f_name+".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)
        # plt.savefig(f_name+".png", dpi=200, bbox_inches="tight")

        break

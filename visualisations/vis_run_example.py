import sys
sys.path.insert(0, "../src")

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import optimize
import tune_optimize_utils as utils
from pylab import cm
import matplotlib.font_manager as fm

tau = 0.01

if __name__ == "__main__":
    plt.style.use('seaborn-pastel')

    # history = optimize.simulation(params=utils.best_params[0], progress_bar=True, state_neurons=2000)
    # for k in history.keys():
        # history[k] = history[k][:, 0].tolist()

    # json.dump(history, open("history.json", 'w'))

    history = json.load(open("history.json"))

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2    

    for i, j in [(0, 5), (40, 45), (90, 95)]:

        start = i * 1000
        end = j * 1000
        times = np.array(list(range(start, end))) / 1000

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        fig.suptitle(f"Model simulation from {i} to {j} seconds", fontsize=24)

        axes[0].plot(times, history["swing1_state"][start:end], linewidth=2, color="#F8550D", label='Swing')
        axes[0].plot(times, history["stance1_state"][start:end], linewidth=2, color="#0081D9", label='Stance')
        axes[0].set_yticks([0, 1])
        axes[0].set_ylabel('Activation', labelpad=10, fontsize=20)
        axes[0].set_xticks([])
        axes[0].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)


        axes[1].plot(times, history["swing2_state"][start:end], linewidth=2, color="#F8550D", label='Swing')
        axes[1].plot(times, history["stance2_state"][start:end], linewidth=2, color="#0081D9", label='Stance')
        axes[1].set_yticks([0, 1])
        axes[1].set_xlabel('Time, s', labelpad=10)
        axes[1].set_ylabel('Activation', labelpad=10,  fontsize=20)

        axes[1].legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

        fig.tight_layout()

        #fig.savefig(, dpi=300, transparent=False, bbox_inches='tight')

        f_name = f"Model simulation from {i} to {j} seconds"
        plt.savefig(f_name+".pdf", format="pdf", dpi=200, bbox_inches="tight")
        plt.savefig(f_name+".png", dpi=200, bbox_inches="tight")

        plt.close()
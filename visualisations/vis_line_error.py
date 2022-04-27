import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score

tau = 0.01

if __name__ == "__main__":
    history = json.load(open("experiment_history.json"))

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    s1_swing_cycles, s1_stance_cycles = history["s1_swing_cycles"], history["s1_stance_cycles"]

    swing_cycles_duration = [(right - left) / 1000
                              for left, right in s1_swing_cycles][1:]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in s1_stance_cycles][1:]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

    true_swing_duration = history["true_s1_swing_duration"]
    true_stance_duration = history["true_s1_stance_duration"]

    swing_r2 = r2_score(true_swing_duration, swing_cycles_duration)
    stance_r2 = r2_score(true_stance_duration, stance_cycles_duration)

    fig.suptitle(f"A", fontsize=24)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    r2_swing_name = "$R^{2}_{swing}$"
    r2_stance_name = "$R^{2}_{stance}$"

    ax.text(0.05, 0.95, f"{r2_swing_name} = {swing_r2:.3f}\n{r2_stance_name} = {stance_r2:.3f}", 
            transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)

    ax.plot(combined_cycles, swing_cycles_duration, color='#0081D9', linestyle='dashed',
     marker='o', linewidth=1, label='Swing')
    ax.plot(combined_cycles, true_swing_duration, 
        color='black', linewidth=2, label="Halbertsma best-fit")

    ax.plot(combined_cycles, stance_cycles_duration, color='#F8550D', 
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

    os.makedirs("./images", exist_ok=True)

    f_name = f"./images/phase_durations_error"
    plt.savefig(f_name + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(f_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)

    # plt.show()

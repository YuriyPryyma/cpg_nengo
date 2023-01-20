import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

from scipy.optimize import curve_fit

if __name__ == "__main__":
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    history = json.load(open("../data_gen/experiment_history.json"))

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    fig.suptitle("B", fontsize=24)

    s1_swing_cycles, s1_stance_cycles = history["s1_swing_cycles"], history["s1_stance_cycles"],

    swing_cycles_duration = [(right - left) / 1000
                              for left, right in s1_swing_cycles][1:]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in s1_stance_cycles][1:]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

    speed_data = history["speed_state"]

    speed_points = []
    for i in range(len(swing_cycles_duration)):
        cyrcle_start = s1_swing_cycles[i+1][0]
        cyrcle_end = s1_stance_cycles[i+1][1]

        speed_points.append((speed_data[cyrcle_start] + speed_data[cyrcle_end])/2)

    speed_points = np.array(speed_points)


    ax.plot(speed_points, swing_cycles_duration, color="#0081D9", 
          linewidth=2, label='Swing')

    ax.plot(speed_points, stance_cycles_duration, color="#F8550D", 
          linewidth=2, label='Stance')

    ax.plot(speed_points, combined_cycles, color="black", 
          linewidth=2, label='Cyrcle')

    ax.set_xlabel('CPG input', labelpad=10, fontsize=20)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlim([0, 1])

    ax.set_ylabel('time, s', labelpad=10, fontsize=20)
    ax.set_yticks([0, 1, 2])
    ax.set_ylim([0, 2])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)

    fig.tight_layout()

    os.makedirs("./images", exist_ok=True)
    f_name = f"./images/power_relationship"
    plt.savefig(f_name + ".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)
    plt.savefig(f_name + ".png", dpi=200, bbox_inches="tight")

    # plt.show()


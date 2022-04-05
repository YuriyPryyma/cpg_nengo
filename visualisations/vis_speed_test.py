import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

import sys
sys.path.append('../src')
from optimize import calc_swing_stance

if __name__ == "__main__":
    history = json.load(open("history_new.json"))

    swing_cycles, stance_cycles = calc_swing_stance(np.array(history["s2_state"]))


    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    fig.suptitle("C", fontsize=24)

  
    swing_cycles_duration = [(right - left) / 1000
                         for left, right in swing_cycles][1:]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in stance_cycles][1:]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

    speed_data = history["speed_state"]


    speed_points = []
    for i in range(len(swing_cycles_duration)):
      cyrcle_start = swing_cycles[i+1][0]
      cyrcle_end = stance_cycles[i+1][1]

      speed_points.append((speed_data[cyrcle_start] + speed_data[cyrcle_end])/2)

    speed_points = np.array(speed_points)

    # Tc = 0.5445*V^(−0.592)
    V_predicted = (combined_cycles/0.5445)**(-1/0.592)

    ax.plot(speed_points, V_predicted, "black", linewidth=3, label='Speed-Velocity relationships')

    def f(x, A, B):
      return A*x + B

    popt, pcov = curve_fit(f, speed_points, V_predicted)

    
    fit_line = [ speed*popt[0] + popt[1] for speed in speed_points]

    stance_r2 = r2_score(fit_line, V_predicted)

    r2 = "$R^{2}$"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, f"{r2} = {stance_r2:.4f}", 
            transform=ax.transAxes, fontsize=20,
            verticalalignment='top', bbox=props)


    print("stance_r2 ", stance_r2)

    ax.plot(speed_points, fit_line, color="#F8550D", label='best-fit line')

    ax.set_xlabel('CPG input', labelpad=10, fontsize=20)
    ax.set_xticks([0, 0.5, 1.0, ])
    ax.set_xlim([0, 1])

    ax.set_ylabel('velocity, (m/s)', labelpad=10, fontsize=20)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim([0, 1])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)


    fig.tight_layout()

    plt.show()

    # f_name = f"speed relationships"
    # plt.savefig(f_name+".pdf", format="pdf", dpi=200, bbox_inches="tight", transparent=True)
    # plt.savefig(f_name+".png", dpi=200, bbox_inches="tight")



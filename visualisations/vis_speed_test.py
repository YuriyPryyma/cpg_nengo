import sys
sys.path.insert(0, "../src")


import numpy as np
import json
import matplotlib.pyplot as plt
import optimize
import tune_optimize_utils as utils
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
# import tkinter
# import matplotlib
from scipy.optimize import curve_fit

# matplotlib.use('TkAgg')

tau = 0.01

if __name__ == "__main__":
    # plt.style.use('seaborn-pastel')

    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2

    # history = optimize.simulation(params=utils.best_params[0], 
    #                               progress_bar=True, state_neurons=2000)

    # for k in history.keys():
    #     history[k] = history[k][:, 0].tolist()

    # json.dump(history, open("history.json", 'w'))

    history = json.load(open("history.json"))


    fig = plt.figure(figsize=(10, 10))

    ax = plt.gca()

    fig.suptitle("Relationship of input CPG speed and locomotion velocity", fontsize=24)

    state = np.array(history[f"s1_state"])
    swing_cycles, stance_cycles = optimize.calc_swing_stance(state)

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

    ax.plot(speed_points, V_predicted, "black", linestyle='dashed', linewidth=3, label='Speed-Velocity relationships')

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

    ax.plot(speed_points, fit_line, "black", label='best-fit line')

    ax.set_xlabel('CPG input', labelpad=10, fontsize=20)
    ax.set_xticks([0, 0.5, 1.0, ])
    ax.set_xlim([0, 1])

    ax.set_ylabel('velocity, (m/s)', labelpad=10, fontsize=20)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim([0, 1])

    ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=20)


    fig.tight_layout()

    # plt.show()

    f_name = f"speed relationships"
    plt.savefig(f_name+".pdf", format="pdf", dpi=200, bbox_inches="tight")
    plt.savefig(f_name+".png", dpi=200, bbox_inches="tight")



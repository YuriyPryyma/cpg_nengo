import numpy as np
import json
import matplotlib.pyplot as plt
import optimize

tau = 0.01

if __name__ == "__main__":
    plt.style.use('ggplot')

    # params = {
    #     "init_swing": 3.2604,
    #     "init_stance": 0.70337,
    #     "speed_swing": 5.4470,
    #     "speed_stance": 2.6385,
    #     # "inner_inhibit": -0.009,
    #     # "sw_sw_con": 0.0921,
    #     # "st_sw_con": -0.0636,
    #     # "sw_st_con": -0.0934,
    #     # "st_st_con": 0.01246,
    # }

    # history = optimize.simulation(params=params, progress_bar=True)

    # for k in history.keys():
    #     history[k] = history[k][:, 0].tolist()

    # json.dump(history, open("history.json", 'w'))

    history = json.load(open("history.json"))

    # for start, end in [(0, 5), (40, 45), (90, 95)]:

    #     start = start * 1000
    #     end = end * 1000

    #     times = np.array(list(range(start, end))) / 1000

    #     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    #     axes[0].plot(times, history["swing1_state"][start:end], color="r")
    #     axes[0].plot(times, history["stance1_state"][start:end], color="b")
    #     axes[1].plot(times, history["swing2_state"][start:end], color="r")
    #     axes[1].plot(times, history["stance2_state"][start:end], color="b")
    #     plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 15))

    for leg in [1, 2]:
        state = np.array(history[f"s{leg}_state"])
        sw_cycles, st_cycles = optimize.calc_swing_stance(state,
                                                          history["speed_state"])

        error_phase, error_speed = optimize.calc_error(state,
                                                       history["speed_state"])

        combined_cycles = sw_cycles + st_cycles

        i = leg - 1

        axes[i].set_title(f"Leg {leg}", fontsize=20)
        axes[i].plot(combined_cycles, sw_cycles, "r")
        axes[i].plot(combined_cycles, optimize.cycle_to_swing(combined_cycles), 'r--')

        axes[i].plot(combined_cycles, st_cycles, "b")
        axes[i].plot(combined_cycles, optimize.cycle_to_stance(combined_cycles), 'b--')

        axes[i].set_xlim([0.5, 2])

        axes[i].plot([], [], ' ', label=f"error_phase {error_phase:.3f}")
        axes[i].plot([], [], ' ', label=f"error_speed {error_speed:.3f}")

        axes[i].legend(fontsize=16)

        axes[i].set_xlabel("Cycle duration", fontsize=16)
        axes[i].set_ylabel("Swing/Stance duration", fontsize=16)

    plt.show()

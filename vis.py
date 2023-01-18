import numpy as np
import json
import matplotlib.pyplot as plt
import optimize
import tune_optimize_utils as utils

tau = 0.01


if __name__ == "__main__":
    plt.style.use('ggplot')

    # history = optimize.simulation(params=utils.best_params[0], progress_bar=True)

    # for k in history.keys():
    #     history[k] = history[k][:, 0].tolist()

    # json.dump(history, open("history.json", 'w'))

    history = json.load(open("history.json"))

    start = 0 * 1000
    end = 5 * 1000

    times = list(range(start, end))

    s1_state = np.array(history["s1_state"])
    s2_state = np.array(history["s2_state"])

    left_sw_cycles, left_st_cycles = optimize.calc_swing_stance(s1_state)
    right_sw_cycles, right_st_cycles = optimize.calc_swing_stance(s2_state)

    swing1 = s1_state < 0
    stance1 = s1_state > 0

    swing2 = s2_state < 0
    stance2 = s2_state > 0

    print(len(left_sw_cycles), len(right_st_cycles))

    print(optimize.symmetry_error(left_sw_cycles[1:], right_st_cycles))


    print(len(right_sw_cycles), len(left_st_cycles))

    print(optimize.symmetry_error(right_sw_cycles, left_st_cycles[:-1]))


    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    axes[0].plot(times, history["stance1_state"][start:end], color="b")
    axes[0].plot(times, history["swing1_state"][start:end], color="r")
    axes[0].axvline(x=left_st_cycles[0][0], color="black")
    axes[0].axvline(x=left_st_cycles[0][1], color="black")
    axes[0].set_ylim([-0.5, 1.5])
    axes[1].plot(times, history["swing2_state"][start:end], color="r")
    axes[1].plot(times, history["stance2_state"][start:end], color="b")
    axes[1].axvline(x=right_sw_cycles[0][0], color="black")
    axes[1].axvline(x=right_sw_cycles[0][1], color="black")
    axes[1].set_ylim([-0.5, 1.5])

    plt.show()

    exit()


    for i, j in [(0, 5), (40, 45), (90, 95)]:

        start = i * 1000
        end = j * 1000

        times = np.array(list(range(start, end))) / 1000

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        axes[0].plot(times, history["swing1_state"][start:end], color="r")
        axes[0].plot(times, history["stance1_state"][start:end], color="b")
        axes[1].plot(times, history["swing2_state"][start:end], color="r")
        axes[1].plot(times, history["stance2_state"][start:end], color="b")

        fig.savefig(f"{i}_{j}_sec")

        # plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 15))

    for leg in [1, 2]:
        state = np.array(history[f"s{leg}_state"])
        sw_cycles, st_cycles = optimize.calc_swing_stance(state,
                                                          history["speed_state"])

        error_phase, error_speed = optimize.calc_error(state,
                                                       history["speed_state"])

        combined_cycles = sw_cycles + st_cycles

        i = leg - 1

        axes[i].set_title(f"Limb {leg}", fontsize=20)
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

    fig.savefig(f"phase_durations")

    #plt.show()

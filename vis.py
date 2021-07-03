import numpy as np
import json
import matplotlib.pyplot as plt
import optimize
import tune_optimize_utils as utils
import nengo
from sklearn.metrics import r2_score

tau = 0.01


if __name__ == "__main__":
    plt.style.use('ggplot')

    # params = {
    #     "init_stance": 0.60555,
    #     "init_stance_position": 0.71452,
    #     "init_swing": 5.0638,
    #     "inner_inhibit": -0.44935,
    #     "speed_stance": 3.6189,
    #     "speed_swing": 3.4614,
    #     "st_st_con": 0.64869,
    #     "st_sw_con": -0.68238,
    #     "sw_st_con": -0.64869,
    #     "sw_sw_con": 0.24511,
    # }
    # history = optimize.simulation(params=params, progress_bar=True)

    # for k in history.keys():
    #     history[k] = history[k][:, 0].tolist()

    # json.dump(history, open("history.json", 'w'))

    history = json.load(open("history.json"))

    # start = 0 * 1000
    # end = 5 * 1000

    # times = list(range(start, end))

    # s1_state = np.array(history["s1_state"])
    # s2_state = np.array(history["s2_state"])

    # left_sw_cycles, left_st_cycles = optimize.calc_swing_stance(s1_state)
    # right_sw_cycles, right_st_cycles = optimize.calc_swing_stance(s2_state)

    # error_left_phase, error_left_speed = optimize.single_limb_error(left_sw_cycles, left_st_cycles)
    # error_right_phase, error_right_speed = optimize.single_limb_error(right_sw_cycles, right_st_cycles)

    # error_phase = error_left_phase + error_right_phase
    # error_speed = error_left_speed + error_right_speed

    # error_symmetricity_l_r = optimize.symmetry_error(left_sw_cycles, right_st_cycles)
    # error_symmetricity_r_l = optimize.symmetry_error(right_sw_cycles, left_st_cycles)

    # error_symmetricity1 = error_symmetricity_l_r + error_symmetricity_r_l

    # swing1 = s1_state < 0
    # stance1 = s1_state > 0

    # swing2 = s2_state < 0
    # stance2 = s2_state > 0

    # sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
    # sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)

    # error_symmetricity2 = (1 - sw1_in_st2) + (1 - sw2_in_st1)
    # error = 1.2 * error_phase + 0.5 * error_speed + \
    #     0.2 * error_symmetricity1 + error_symmetricity2

    # print(error)

    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    # axes[0].plot(times, history["swing1_state"][start:end], color="r")
    # axes[0].axvline(x=left_sw_cycles[1][0], color="b")
    # axes[0].axvline(x=left_sw_cycles[1][1], color="b")
    # axes[0].set_ylim([-0.5, 1.5])
    # axes[1].plot(times, history["stance2_state"][start:end], color="r")
    # axes[1].axvline(x=right_st_cycles[1][0], color="b")
    # axes[1].axvline(x=right_st_cycles[1][1], color="b")
    # axes[1].set_ylim([-0.5, 1.5])


    # plt.show()

    # exit()


    for i, j in [(0, 5), (40, 43), (90, 95)]:

        start = i * 1000
        end = j * 1000

        times = np.array(list(range(start, end))) / 1000

        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
        axes[0].plot(times, history["swing1_state"][start:end], color="r", label="Swing phase")
        axes[0].plot(times, history["stance1_state"][start:end], color="b", label="Stance phase")
        axes[0].set_title("Limb 1")
        axes[1].plot(times, history["swing2_state"][start:end], color="r", label="Swing phase")
        axes[1].plot(times, history["stance2_state"][start:end], color="b", label="Stance phase")
        axes[1].set_title("Limb")
        axes[1].set_xlabel("Time (s)", fontsize=18)

        axes[0].legend(fontsize=14)
        axes[1].legend(fontsize=14)

        fig.savefig(f"plots/{i}_{j}_sec")

    #     # plt.show()

    # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(30, 15))

    s1_state = np.array(history["s1_state"])
    s2_state = np.array(history["s2_state"])

    left_sw_cycles, left_st_cycles = optimize.calc_swing_stance(s1_state)
    right_sw_cycles, right_st_cycles = optimize.calc_swing_stance(s2_state)

    stance_cycles_duration = [(right - left) / 1000 for left, right in left_st_cycles]+ \
                             [(right - left) / 1000 for left, right in right_st_cycles] 

    swing_cycles_duration = [(right - left) / 1000 for left, right in left_sw_cycles] + \
                             [(right - left) / 1000 for left, right in right_sw_cycles]


    # swing_cycles_duration = swing_cycles_duration[len(swing_cycles_duration)//2+1:]
    # stance_cycles_duration = stance_cycles_duration[len(stance_cycles_duration)//2+1:]
    swing_cycles_duration = swing_cycles_duration[:len(swing_cycles_duration)//2]
    stance_cycles_duration = stance_cycles_duration[:len(stance_cycles_duration)//2]



    # stance_cycles_duration = [x for x, _ in 
    #     sorted(zip(stance_cycles_duration, swing_cycles_duration), key=lambda pair: pair[0] + pair[1], reverse=True)]

    # swing_cycles_duration = [x for _, x in 
    #     sorted(zip(stance_cycles_duration, swing_cycles_duration), key=lambda pair: pair[0] + pair[1], reverse=True)]



    # combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)


    # plt.plot(combined_cycles, swing_cycles_duration, "r", label="Swing phase")
    # plt.plot(combined_cycles, optimize.cycle_to_swing(combined_cycles), 'r--')

    # plt.plot(combined_cycles, stance_cycles_duration, "b", label="Stance phase")
    # plt.plot(combined_cycles, optimize.cycle_to_stance(combined_cycles), 'b--')

    # plt.xlim([0.5, 2])
    # plt.xlabel("cyrcle duration (s)")

    # plt.ylabel("phase duration (s)")
    # plt.legend(fontsize=14)

    # print("swing", r2_score(optimize.cycle_to_swing(combined_cycles), swing_cycles_duration))
    # print("stance", r2_score(optimize.cycle_to_stance(combined_cycles), stance_cycles_duration))

    # # plt.plot([], [], ' ', label=f"error_phase {error_phase:.3f}")
    # # plt.plot([], [], ' ', label=f"error_speed {error_speed:.3f}")

    # # axes[i].legend(fontsize=16)

    # # axes[i].set_xlabel("Cycle duration", fontsize=16)
    # # axes[i].set_ylabel("Swing/Stance duration", fontsize=16)

    # plt.savefig(f"plots/phase_durations")

    # # plt.show()

    speed_state = np.array(history["speed_state"])

    print(min(speed_state))
    print(max(speed_state))

    # V = 1

    # Tc = 0.5445*V^(−0.592)

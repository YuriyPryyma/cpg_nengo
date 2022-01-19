import numpy as np
from sklearn.metrics import mean_squared_error
import nengo
import nengo_ocl
from cpg import create_CPG

tau = 0.01

# Halbert sma minmax(Tc)
MIN_PHASE, MAX_PHASE = (.57, 1.91)


def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    r = []
    while i != -1:
        r.append(i)
        i = s.find(p, i + 1)
    return r


def calc_swing_stance(state_probe):
    s1_state_changes = state_probe < 0

    state_str = "".join([str(int(s)) for s in s1_state_changes])

    stance_swing = findall("01", state_str)
    swing_stance = findall("10", state_str)

    swing_cycles = []
    stance_cycles = []

    for start_swing in stance_swing:   
        start_stance_i = np.searchsorted(swing_stance, start_swing)
        if start_stance_i >= len(swing_stance):
            break
        start_stance = swing_stance[start_stance_i]

        end_stance_i = np.searchsorted(stance_swing, start_stance)
        if end_stance_i >= len(stance_swing):
            break
        end_stance = stance_swing[end_stance_i]

        swing_cycles.append((start_swing, start_stance))
        stance_cycles.append((start_stance, end_stance))

    return swing_cycles, stance_cycles


def cycle_to_swing(cycle):
    return 0.168 + 0.0938 * cycle


def cycle_to_stance(cycle):
    return -0.168 + 0.9062 * cycle


def single_limb_error(swing_cycles, stance_cycles):

    swing_cycles_duration = [(right - left) / 1000
                             for left, right in swing_cycles]

    stance_cycles_duration = [(right - left) / 1000
                              for left, right in stance_cycles]

    combined_cycles = np.array(swing_cycles_duration) + np.array(stance_cycles_duration)

    swing_expected = cycle_to_swing(combined_cycles)
    stance_expected = cycle_to_stance(combined_cycles)
    err_swing = mean_squared_error(swing_expected,
                                   swing_cycles_duration, squared=False)
    err_stance = mean_squared_error(stance_expected,
                                    stance_cycles_duration, squared=False)
    error_phase = err_swing + err_stance

    error_speed = abs(MIN_PHASE - min(combined_cycles)) + \
        abs(MAX_PHASE - max(combined_cycles))

    return error_phase, error_speed

def symmetry_error(swing_cycles, stance_cycles):

    pre_swing_part = [abs(swing[0] - stance[0]) / 1000
                      for swing, stance in zip(swing_cycles, stance_cycles)]

    post_swing_part = [abs(stance[1] - swing[1]) / 1000
                       for swing, stance in zip(swing_cycles, stance_cycles)]

    error = mean_squared_error(pre_swing_part, post_swing_part, squared=False)

    return error


def simulation(params, time=95, progress_bar=False, state_neurons=5000, **args):
    model = create_CPG(params=params, state_neurons=state_neurons, **args)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)
        speed_probe = nengo.Probe(model.speed, synapse=tau)

        swing1_probe = nengo.Probe(model.swing1, synapse=tau)
        stance1_probe = nengo.Probe(model.stance1, synapse=tau)
        swing2_probe = nengo.Probe(model.swing2, synapse=tau)
        stance2_probe = nengo.Probe(model.stance2, synapse=tau)

    with nengo_ocl.Simulator(model, progress_bar=progress_bar) as sim:
        sim.run(time)

    return {
        "s1_state": sim.data[s1_probe],
        "s2_state": sim.data[s2_probe],
        "speed_state": sim.data[speed_probe],
        "swing1_state": sim.data[swing1_probe],
        "stance1_state": sim.data[stance1_probe],
        "swing2_state": sim.data[swing2_probe],
        "stance2_state": sim.data[stance2_probe],
    }


def simulation_error(params, time=95, progress_bar=False, state_neurons=2000, **args):
    history = simulation(params, time, progress_bar, state_neurons=state_neurons, **args)

    s1_state = history["s1_state"]
    s2_state = history["s2_state"]

    try:
        left_sw_cycles, left_st_cycles = calc_swing_stance(s1_state)
        right_sw_cycles, right_st_cycles = calc_swing_stance(s2_state)

        error_left_phase, error_left_speed = single_limb_error(left_sw_cycles, left_st_cycles)
        error_right_phase, error_right_speed = single_limb_error(right_sw_cycles, right_st_cycles)

        error_phase = error_left_phase + error_right_phase
        error_speed = error_left_speed + error_right_speed

        error_symmetricity_l_r = symmetry_error(left_sw_cycles[1:], right_st_cycles)
        error_symmetricity_r_l = symmetry_error(right_sw_cycles, left_st_cycles[:-1])

        error_symmetricity1 = error_symmetricity_l_r + error_symmetricity_r_l

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)

        error_symmetricity2 = (1 - sw1_in_st2) + (1 - sw2_in_st1)

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity1 = 10
        error_symmetricity2 = 10

    error = 2 * error_phase + error_speed + \
        error_symmetricity1 + error_symmetricity2

    return history, error, error_phase, error_speed, error_symmetricity1, error_symmetricity2



def only_error(history, time=95):
    eval_slice = int((time/95)*len(history["s1_state"]))
    s1_state = history["s1_state"][:eval_slice]
    s2_state = history["s2_state"][:eval_slice]

    try:
        left_sw_cycles, left_st_cycles = calc_swing_stance(s1_state)
        right_sw_cycles, right_st_cycles = calc_swing_stance(s2_state)

        error_left_phase, error_left_speed = single_limb_error(left_sw_cycles, left_st_cycles)
        error_right_phase, error_right_speed = single_limb_error(right_sw_cycles, right_st_cycles)

        error_phase = error_left_phase + error_right_phase
        error_speed = 0

        error_symmetricity_l_r = symmetry_error(left_sw_cycles[1:], right_st_cycles)
        error_symmetricity_r_l = symmetry_error(right_sw_cycles, left_st_cycles[:-1])

        error_symmetricity1 = error_symmetricity_l_r + error_symmetricity_r_l

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)

        error_symmetricity2 = (1 - sw1_in_st2) + (1 - sw2_in_st1)

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity1 = 10
        error_symmetricity2 = 10

    error = 2 * error_phase + error_speed + \
        error_symmetricity1 + error_symmetricity2

    return error, error_phase, error_speed, error_symmetricity1, error_symmetricity2

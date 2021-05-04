import numpy as np
from sklearn.metrics import mean_squared_error
import nengo
from cpg import create_CPG

tau = 0.01

# Halbert sma minmax(Tc)
expected_min_range = .57
expected_max_range = 1.91


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

    swing_duration = [(right - left) / 1000 for left, right
                      in zip(stance_swing, swing_stance)]
    stance_duration = [(right - left) / 1000 for left, right
                       in zip(swing_stance, stance_swing[1:])]

    return np.array(swing_duration), np.array(stance_duration)


def cycle_to_swing(cycle):
    return 0.168 + 0.0938 * cycle


def cycle_to_stance(cycle):
    return -0.168 + 0.9062 * cycle


def calc_error(state_probe):
    swing_duration, stance_duration = calc_swing_stance(state_probe)
    full_cycles = min(len(swing_duration), len(stance_duration))
    swing_duration = swing_duration[:full_cycles]
    stance_duration = stance_duration[:full_cycles]
    combined_cycle = swing_duration + stance_duration
    swing_true = cycle_to_swing(combined_cycle)
    stance_true = cycle_to_stance(combined_cycle)

    err_swing = mean_squared_error(swing_true,
                                   swing_duration, squared=False)
    err_stance = mean_squared_error(stance_true,
                                    stance_duration, squared=False)
    rms_error = err_swing + err_stance

    min_range = min(combined_cycle)
    max_range = max(combined_cycle)
    err_range_min = abs(expected_min_range - min_range)
    err_range_max = abs(expected_max_range - max_range)
    error_range = (err_range_min + err_range_max) / \
                  (expected_max_range - expected_min_range)

    return rms_error, error_range


def simulation_error(params):
    model = create_CPG(params=params, state_neurons=2000)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(100)

    s1_state = sim.data[s1_probe]
    s2_state = sim.data[s2_probe]

    try:
        error_left_rms, error_left_range = calc_error(s1_state)
        error_right_rms, error_right_range = calc_error(s2_state)
        error_rms = error_left_rms + error_right_rms
        error_range = error_left_range + error_right_range

        # err_sym_st = np.sum((s1_state > 0) * (s2_state > 0)) / len(s1_state)
        # err_sym_sw = np.sum((s1_state < 0) * (s2_state < 0)) / len(s1_state)

        # error_symmetricity = err_sym_st + err_sym_sw
        
    except Exception as e:
        print("error calc", e)
        error_rms = 10
        error_range = 10

    error = error_rms + error_range

    return error, error_rms, error_range

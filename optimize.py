import numpy as np
from sklearn.metrics import mean_squared_error
import nengo
from cpg import create_CPG

tau = 0.01


def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    r = []
    while i != -1:
        r.append(i)
        i = s.find(p, i + 1)
    return r


def calc_swing_stance(state_probe, speed_state):
    s1_state_changes = state_probe < 0

    state_str = "".join([str(int(s)) for s in s1_state_changes])

    stance_swing = findall("01", state_str)
    swing_stance = findall("10", state_str)

    swing_cycles = [(right - left) / 1000 for left, right
                    in zip(stance_swing, swing_stance)]
    stance_cycles = [(right - left) / 1000 for left, right
                     in zip(swing_stance, stance_swing[1:])]

    swing_speed = [np.mean([speed_state[i] for i in range(left, right)])
                   for left, right in zip(stance_swing, swing_stance)]

    stance_speed = [np.mean([speed_state[i] for i in range(left, right)])
                    for left, right in zip(swing_stance, stance_swing[1:])]

    full_cycles = min(len(swing_cycles), len(stance_cycles))

    swing_cycles = swing_cycles[:full_cycles]
    stance_cycles = stance_cycles[:full_cycles]

    swing_speed = swing_speed[:full_cycles]
    stance_speed = stance_speed[:full_cycles]
    speed_cycles = np.mean([swing_speed, stance_speed], axis=0)

    return np.array(swing_cycles), np.array(stance_cycles), speed_cycles


def cycle_to_swing(cycle):
    return 0.168 + 0.0938 * cycle


def cycle_to_stance(cycle):
    return -0.168 + 0.9062 * cycle


def speed_to_cycle(speed):
    """
    Halbert sma minmax(Tc)
    [.57, 1.91]
    """
    return 1.91 - 1.34 * speed


def calc_error(state_probe, speed_state):
    swing_cycles, stance_cycles, speed_cycles = \
        calc_swing_stance(state_probe, speed_state)

    combined_cycles = swing_cycles + stance_cycles

    swing_expected = cycle_to_swing(combined_cycles)
    stance_expected = cycle_to_stance(combined_cycles)
    err_swing = mean_squared_error(swing_expected,
                                   swing_cycles, squared=False)
    err_stance = mean_squared_error(stance_expected,
                                    stance_cycles, squared=False)
    error_phase = err_swing + err_stance

    cycles_duration_expected = speed_to_cycle(speed_cycles)
    error_speed = mean_squared_error(cycles_duration_expected,
                                     combined_cycles, squared=False)

    return error_phase, error_speed


def simulation_error(params):
    model = create_CPG(params=params, state_neurons=2000)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)
        speed_probe = nengo.Probe(model.speed, synapse=tau)

    with nengo.Simulator(model, progress_bar=False) as sim:
        sim.run(100)

    s1_state = sim.data[s1_probe]
    s2_state = sim.data[s2_probe]
    speed_state = sim.data[speed_probe]

    try:
        error_left_phase, error_left_speed = \
            calc_error(s1_state, speed_state)
        error_right_phase, error_right_speed = \
            calc_error(s2_state, speed_state)
        error_phase = error_left_phase + error_right_phase
        error_speed = error_left_speed + error_right_speed

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        sw1_in_st2 = np.sum(swing1 * stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 * stance1) / np.sum(swing2)
        sw_interect = np.sum(swing1 * swing2) / np.sum(swing1 | swing2)

        error_symmetricity = (1 - sw1_in_st2) + (1 - sw2_in_st1) + sw_interect

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity = 10

    error = error_phase + 0.1 * error_speed + 0.1 * error_symmetricity

    return error, error_phase, error_speed, error_symmetricity

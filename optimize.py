import numpy as np
from sklearn.metrics import mean_squared_error
import nengo
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


def calc_swing_stance(state_probe, speed_state):
    s1_state_changes = state_probe < 0

    state_str = "".join([str(int(s)) for s in s1_state_changes])

    stance_swing = findall("01", state_str)
    swing_stance = findall("10", state_str)

    swing_cycles = [(right - left) / 1000 for left, right
                    in zip(stance_swing, swing_stance)]
    stance_cycles = [(right - left) / 1000 for left, right
                     in zip(swing_stance, stance_swing[1:])]

    full_cycles = min(len(swing_cycles), len(stance_cycles))

    swing_cycles = swing_cycles[:full_cycles]
    stance_cycles = stance_cycles[:full_cycles]

    return np.array(swing_cycles), np.array(stance_cycles)


def cycle_to_swing(cycle):
    return 0.168 + 0.0938 * cycle


def cycle_to_stance(cycle):
    return -0.168 + 0.9062 * cycle


def calc_error(state_probe, speed_state):
    swing_cycles, stance_cycles = calc_swing_stance(state_probe, speed_state)

    combined_cycles = swing_cycles + stance_cycles

    swing_expected = cycle_to_swing(combined_cycles)
    stance_expected = cycle_to_stance(combined_cycles)
    err_swing = mean_squared_error(swing_expected,
                                   swing_cycles, squared=False)
    err_stance = mean_squared_error(stance_expected,
                                    stance_cycles, squared=False)
    error_phase = err_swing + err_stance

    error_speed = abs(MIN_PHASE - min(combined_cycles)) + \
        abs(MAX_PHASE - max(combined_cycles))

    return error_phase, error_speed


def simulation(params, time=95, progress_bar=False):
    model = create_CPG(params=params, state_neurons=2000)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)
        speed_probe = nengo.Probe(model.speed, synapse=tau)

        swing1_probe = nengo.Probe(model.swing1, synapse=tau)
        stance1_probe = nengo.Probe(model.stance1, synapse=tau)
        swing2_probe = nengo.Probe(model.swing2, synapse=tau)
        stance2_probe = nengo.Probe(model.stance2, synapse=tau)

    with nengo.Simulator(model, progress_bar=progress_bar) as sim:
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


def simulation_error(params, time=95):
    history = simulation(params, time)

    s1_state = history["s1_state"]
    s2_state = history["s2_state"]
    speed_state = history["speed_state"]

    try:
        error_left_phase, error_left_speed = calc_error(s1_state, speed_state)
        error_right_phase, error_right_speed = calc_error(s2_state, speed_state)
        error_phase = error_left_phase + error_right_phase
        error_speed = error_left_speed + error_right_speed

        swing1 = s1_state < 0
        stance1 = s1_state > 0

        swing2 = s2_state < 0
        stance2 = s2_state > 0

        sw1_in_st2 = np.sum(swing1 & stance2) / np.sum(swing1)
        sw2_in_st1 = np.sum(swing2 & stance1) / np.sum(swing2)
        sw_interect = np.sum(swing1 & swing2) / np.sum(swing1 | swing2)

        error_symmetricity = (1 - sw1_in_st2) + (1 - sw2_in_st1) + sw_interect

    except Exception as e:
        print("error calc", e)
        error_phase = 10
        error_speed = 10
        error_symmetricity = 10

    error = error_phase + 0.5 * error_speed + 0.5 * error_symmetricity

    return error, error_phase, error_speed, error_symmetricity

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=3)
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from optimparallel import minimize_parallel

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
        i = s.find(p, i+1)
    return r


def calc_swing_stance(state_probe):
    s1_state_changes = state_probe < 0

    state_str = "".join([str(int(s)) for s in s1_state_changes])

    stance_swing = findall("01", state_str)
    swing_stance = findall("10", state_str)

    swing_duration = [(right-left)/1000 for left,right in zip(stance_swing, swing_stance) ]
    stance_duration = [(right-left)/1000 for left,right in zip(swing_stance, stance_swing[1:]) ]

    return np.array(swing_duration), np.array(stance_duration)

def cycle_to_swing(cycle):
    return 0.168 + 0.0938*cycle

def cycle_to_stance(cycle):
    return -0.168 + 0.9062*cycle;

def calc_error(state_probe):
    swing_duration, stance_duration = calc_swing_stance(state_probe)
    full_cycles = min(len(swing_duration), len(stance_duration))
    swing_duration = swing_duration[:full_cycles]
    stance_duration = stance_duration[:full_cycles]
    combined_cycle = swing_duration+stance_duration
    swing_true = cycle_to_swing(combined_cycle)
    stance_true = cycle_to_stance(combined_cycle)

    err_swing = mean_squared_error(swing_true, swing_duration, squared=False)
    err_stance = mean_squared_error(stance_true, stance_duration, squared=False)

    return err_swing + err_stance

def simulation_error(params, info):
    model = create_CPG(params=params, state_neurons=1000)

    with model:
        s1_probe = nengo.Probe(model.s1, synapse=tau)
        s2_probe = nengo.Probe(model.s2, synapse=tau)

    with nengo.Simulator(model, progress_bar=False, optimize=False) as sim:
        sim.run(5)

    err_left = calc_error(sim.data[s1_probe])
    err_right = calc_error(sim.data[s2_probe])

    print(f"step {info['Nfeval']} error {err_left+err_right}")
    if info['Nfeval'] % 20 == 0:
        print(f" params {params}")

    info['Nfeval'] += 1

    return err_left+err_right


if __name__ == "__main__":
    # x0 = [2.7136, 0, 1.1668, 1.6596, -0.009, 0.0921, -0.0636, -0.0934, 0.01246]
    # x0 = [ 3.33476620e+00,  4.62892674e-04,  1.34863186e+00,  1.32987859e+00,
    #    -9.43183373e-03,  7.93225063e-02, -6.15617168e-02, -9.14669267e-02,
    #     1.19109073e-02]

    x0 = [2, 0, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1]

    # print(simulation_error(x0))
    # print(simulation_error(x0_better))

    res = minimize(simulation_error, x0, method='nelder-mead', args=({'Nfeval':0},),
               options={'disp': True, "xatol":1*1e-2, "fatol":1*1e-2, "maxiter":1000})

    print(res)
    






import nengo
from nengo.dists import Uniform, Choice, Exponential
import numpy as np
import matplotlib.pyplot as plt

def recurrent_func(x, tau=0.1):
    x0, x1 = x
    r = np.sqrt(x0 ** 2 + x1 ** 2)
    a = np.arctan2(x1, x0)
    dr = -(r - 2)
    da = 3.0
    r = r + tau * dr
    a = a + tau * da
    return [r * np.cos(a), r * np.sin(a)]


with nengo.Network() as model:
    osc = nengo.Ensemble(1000, 2)
    nengo.Connection(
        osc,
        osc,
        function=recurrent_func,
        synapse=0.1,
    )

    ramp = nengo.Ensemble(100, 1)
    nengo.Connection(
        osc, ramp, function=lambda x: 0.5 + np.arctan2(x[1], x[0]) / (2 * np.pi)
    )

    osc_probe = nengo.Probe(osc, "decoded_output", synapse=0.01)
    ramp_probe = nengo.Probe(ramp, "decoded_output", synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(7)

plt.style.use('ggplot')

plt.figure( figsize=(8, 8))
plt.plot(sim.trange(), sim.data[osc_probe][:,0], "r", label="x1")
plt.plot(sim.trange(), sim.data[osc_probe][:,1], "b", label="x2")
plt.legend(fontsize=16)
plt.xlabel("time (s)", fontsize=16)
# plt.show()
plt.savefig("oscelator1")

plt.style.use('ggplot')
plt.figure( figsize=(8, 8))
plt.plot(sim.trange(), sim.data[ramp_probe], "r", label="arctan2 output")
plt.legend(fontsize=16)
plt.xlabel("time (s)", fontsize=16)
# plt.show()
plt.savefig("oscelator_output")

# plt.plot(combined_cycles, swing_cycles_duration, "r", label="Swing phase")
# plt.plot(combined_cycles, optimize.cycle_to_swing(combined_cycles), 'r--')

# plt.plot(combined_cycles, stance_cycles_duration, "b", label="Stance phase")
# plt.plot(combined_cycles, optimize.cycle_to_stance(combined_cycles), 'b--')

# plt.xlim([0.5, 2])
# plt.xlabel("cyrcle duration (s)")

# plt.ylabel("phase duration (s)")
# plt.legend(fontsize=14)
import nengo
from nengo.dists import Uniform, Choice, Exponential
import numpy as np
import matplotlib.pyplot as plt


tau = 0.01

def grouth_equesions(x):
    return x + 1 * tau

model = nengo.Network(seed=42)
with model:

    integrator = nengo.Ensemble(300, 1)

    nengo.Connection(integrator, integrator, function = grouth_equesions, synapse=tau)

    thresh = nengo.Ensemble(1, 1, intercepts=[0.47], max_rates=[400],
                                        encoders=[[1]])
                          
    nengo.Connection(integrator, thresh,
                             function=lambda x: x - 0.5, synapse=tau)


    nengo.Connection(thresh, integrator.neurons, transform=[[-100]] * 300, synapse=tau)

    integrator_probe = nengo.Probe(integrator, "decoded_output", synapse=0.01)
    thresh_probe = nengo.Probe(thresh, "decoded_output", synapse=0.01)


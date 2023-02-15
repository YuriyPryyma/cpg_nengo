"""
This example builds on top of the Integrator example
We plan to add one additional element which detects
Integrator saturation(when it reaches the radius edge)

We call this element threshold. We do not learn 
threshold behavior using a defined function as we did for Integrator.
Because it could be much easier implemented using only one neuron.
We create a similar Ensemble element 
but now specify its intercepts, max_rates, encoders. 
These are properties of LIF neuron that is used by default in Ensemble.
The most interesting for us is the intercept parameter, 
which controls starting value when the neuron begins firing.
We connect the output from Integrator to 
our threshold element with deduced value by 0.5
We do this because intercept works poorly with values close to the radius.

Know with intercept value 0.45 and deduced output 0.5. 
Our threshold element will start firing when the integrator reaches 0.95.
"""

import nengo

model = nengo.Network()


state_neurons = 500
radius = 1
tau = 0.01

def integrate(x):
    dX = 1 - 0.009 * x
    return dX * tau + x
    

with model:
    state = nengo.Ensemble(state_neurons, 
            1, radius=radius)
            
    nengo.Connection(state, state, 
            function=integrate, synapse=tau)
            
    thresh_x1 = nengo.Ensemble(1, 1, intercepts=[0.45], max_rates=[400],
                                        encoders=[[1]])
                                        
    nengo.Connection(state, thresh_x1,
                      function=lambda x: x - 0.5, synapse=tau)
    
    
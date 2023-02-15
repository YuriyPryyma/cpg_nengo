"""
Example 4 aims to show how we could split
complex model dynamics into smaller and simpler parts.
We would use this principle for building connections for CPG dynamics

Imagine we have a similar integrator as we have in example1
with dX = 5*0.1. Imagine this derivative consists of 5 parts and 
is hard to learn as nengo connection.
We could consider the resulting integration 
as an effect of these 5 different parts separately.

So State1 example implements the original integrator

State2 has 6 simpler connections to replace original
One recurring connection works as memory to "save" the current state.
And 5 incoming connections which each adds a small part of the integration.
In nengo, all incoming connections sum up 
together automatically to produce the final result.

You could see that output of state1 and state2 are the same

"""

import nengo

model = nengo.Network()


def integrate(x):
    dX = 0.1+0.1+0.1+0.1+0.1
    return dX * tau + x
    

def integrate_part(x):
    dX = 0.1
    return dX * tau
    

radius = 1
tau = 0.1
state_neurons = 1000

with model:
    state1 = nengo.Ensemble(state_neurons, 
            1, 
            radius=radius)
    
    nengo.Connection(state1, state1, 
            function=integrate, synapse=tau)
            
    
    state2 = nengo.Ensemble(state_neurons, 
            1, 
            radius=radius)
    
    nengo.Connection(state2, state2, synapse=tau)
            
    for i in range(5):
        damage_count = nengo.Node([1], label="signal")
        nengo.Connection(damage_count, state2, 
            function=integrate_part, synapse=tau)

    
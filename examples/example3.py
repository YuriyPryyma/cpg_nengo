"""
The third example builds on top of the integrator and threshold element.
Now we want to have two antagonistic integrators, 
where only one of them is active(integrating) at the moment 
and the other is set to zero.
We also want to switch "active" integrator 
when one of them starts to saturate.

For this, we would need two additional functions
1) State element that stores current "active" integrator.
2) Functionality to reset saturated Integator.

Let's start with the state element and state change
In our case State element is Ensemble with two neurons.
As we do not need more to store binary variable (active state)

Now the question is how we change the state 
when one of the integrators saturates.
We would use our threshold elements for this task. 
We create two for each of the integrators. When the integrator saturates
it triggers its corresponding threshold element. 
The job of the threshold element is then to change the current state. 
We do this with the simple connections that send in one case huge positive 
and in another huge negative signal to 
set the value of a state to 1 or -1 respectively.

The next step is to reset the value of the integator.
We do this using the inhibitory connection. It's possible in Nengo
to access the decoding weights of an Ensemble. 
Setting decoding weights to zero stops neurons from firing and
resets the ensemble value to zero. We do this by sending strong negative signal to zero out the weights. 
Our only job is to create two functions that inhibit integrators 
when the state is 1 or the state is -1 and send nothing(zero) when opposite.

The resulting system will oscillate and is a crucial building block for CPG

"""
import nengo

model = nengo.Network()

state_neurons = 500
radius = 1
tau = 0.01

def integrate(x):
    dX = 1  
    return dX * tau + x
    

def positive_signal(x):
    if x > 0:
        return [-100] * state_neurons
    else:
        return [0] * state_neurons

def negative_signal(x):
    if x < 0:
        return [-100] * state_neurons
    else:
        return [0] * state_neurons

with model:
    x1 = nengo.Ensemble(state_neurons, 1, radius=radius)
    nengo.Connection(x1, x1, function=integrate, synapse=tau)
    x2 = nengo.Ensemble(state_neurons, 1, radius=radius)
    a = nengo.Connection(x2, x2, function=integrate, synapse=tau)
    
    
    state = nengo.Ensemble(2, 1, radius=radius, intercepts=[0, 0],
                               max_rates=[400, 400],
                               encoders=[[-1], [1]])
    
    nengo.Connection(state, state, synapse=tau)
    
    start_signal = nengo.Node(
                nengo.processes.Piecewise({
                    0: -1 ,
                    0.01: 0,
                }))
    nengo.Connection(start_signal, state, synapse=tau)
    
    
    thresh_x1 = nengo.Ensemble(1, 1, intercepts=[0.45], max_rates=[400],
                                        encoders=[[1]])
                                        
    thresh_x2 = nengo.Ensemble(1, 1, intercepts=[0.45], max_rates=[400],
                                        encoders=[[1]])

    nengo.Connection(x1, thresh_x1,
                     function=lambda x: x - 0.5, synapse=tau)
    
    nengo.Connection(x2, thresh_x2,
                     function=lambda x: x - 0.5, synapse=tau)
                     
    nengo.Connection(thresh_x1, state,
                     transform=[100], synapse=tau)
    
    nengo.Connection(thresh_x2, state,
                     transform=[-100], synapse=tau)
                     
    nengo.Connection(state, x1.neurons,
                             function=positive_signal, synapse=tau)
                             
    nengo.Connection(state, x2.neurons ,
                             function=negative_signal, synapse=tau)
                             
    
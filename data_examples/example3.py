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
                             
    
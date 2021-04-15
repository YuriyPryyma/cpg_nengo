import nengo
import numpy as np
from nengo.processes import Piecewise

radius = np.sqrt(2)
tau = 0.01

def create_CPG(*, params, state_neurons=400):
    init_swing = params[0]
    init_stance = params[1]
    speed_swing = params[2]
    speed_stance = params[3] 
    inner_inhibit = params[4]
    
    swing_swing_connection = params[5]
    stance_swing_connection = params[6]
    
    swing_stance_connection = params[7]
    stance_stance_connection = params[8]
    
    def swing_feedback(state):
        x, speed = state
        dX = init_swing + speed_swing*(1+speed*2) + inner_inhibit*x
        return dX*tau + x
    
    def stance_feedback(state):
        x, speed = state
        dX = init_stance + speed_stance*(1+speed*2) + inner_inhibit*x
        return dX*tau + x
    
    def positive_signal(x):
        if x > 0:
            return [-100]*state_neurons
        else:
            return [0]*state_neurons
    
    def negative_signal(x): 
        if x < 0:
            return [-100]*state_neurons
        else:
            return [0]*state_neurons
        
    model = nengo.Network(seed=42)
    with model:
        swing1 = nengo.Ensemble(state_neurons, 2, radius=radius, label="swing1")
        
        stance1 = nengo.Ensemble(state_neurons, 2, radius=radius, label="stance1")
        
        swing2 = nengo.Ensemble(state_neurons, 2, radius=radius, label="swing2")
        
        stance2 = nengo.Ensemble(state_neurons, 2, radius=radius, label="stance2")
        
        nengo.Connection(swing1, swing1[0], function=swing_feedback, synapse=tau)
        nengo.Connection(stance1, stance1[0], function=stance_feedback, synapse=tau)
        nengo.Connection(swing2, swing2[0], function=swing_feedback, synapse=tau)
        nengo.Connection(stance2, stance2[0], function=stance_feedback, synapse=tau)
        
        for group in [(swing1, stance1, swing2, stance2), (swing2, stance2, swing1, stance1)]:
            swing_left, stance_left, swing_right, stance_right = group
            nengo.Connection(swing_left[0], swing_right[0], 
                function=lambda x: tau*(1-x)*swing_swing_connection, 
                synapse=tau)
                
            nengo.Connection(swing_left[0], stance_right[0], 
                function=lambda x: tau*(1-x)*swing_stance_connection, 
                synapse=tau)
                
            nengo.Connection(stance_left[0], swing_right[0], 
                function=lambda x: tau*(1-x)*stance_swing_connection, 
                synapse=tau)
                
            nengo.Connection(stance_left[0], stance_right[0], 
                function=lambda x: tau*(1-x)*stance_stance_connection, 
                synapse=tau)
        
        
        start_signal = nengo.Node(
            Piecewise({
                0: -1,
                0.01: 0,
            }), label="start_signal")
            
        model.s1 = nengo.Ensemble(2, 1, radius=1, intercepts=[0,0], 
                            encoders=[[-1],[1]], label="s1")
        nengo.Connection(model.s1, model.s1, synapse=tau)
        
        nengo.Connection(start_signal, model.s1, synapse=tau)
        
        nengo.Connection(model.s1, swing1.neurons, function=positive_signal, synapse=tau)
        nengo.Connection(model.s1, stance1.neurons, function=negative_signal, synapse=tau)
        
        thresh1 = nengo.Ensemble(1, 1, intercepts=[0.47], 
                                encoders=[[1]], label="thresh1")
        nengo.Connection(swing1[0], thresh1, function= lambda x: x-0.5, synapse=tau)
        nengo.Connection(thresh1, model.s1,
                        transform=[100], synapse=tau)
                        
        thresh2 = nengo.Ensemble(1, 1, intercepts=[0.47], 
                                encoders=[[1]], label="thresh2")
        nengo.Connection(stance1[0], thresh2, function= lambda x: x-0.5, synapse=tau)
        nengo.Connection(thresh2, model.s1,
                        transform=[-100], synapse=tau)
                        
        
        model.s2 = nengo.Ensemble(2, 1, radius=1, intercepts=[0,0], 
                            encoders=[[-1],[1]], label="s2")
        nengo.Connection(model.s2, model.s2, synapse=tau)
        
        nengo.Connection(start_signal, model.s2, synapse=tau)
        
        nengo.Connection(model.s2, swing2.neurons, function=positive_signal, synapse=tau)
        nengo.Connection(model.s2, stance2.neurons, function=negative_signal, synapse=tau)
        
        thresh3 = nengo.Ensemble(1, 1, intercepts=[0.47], 
                                encoders=[[1]], label="thresh3")
        nengo.Connection(swing2[0], thresh3, function= lambda x: x-0.5, 
                        synapse=tau)
        nengo.Connection(thresh3, model.s2,         
                        transform=[100], synapse=tau)
                        
        thresh4 = nengo.Ensemble(1, 1, intercepts=[0.47], 
                                encoders=[[1]], label="thresh4")
        nengo.Connection(stance2[0], thresh4, function= lambda x: x-0.5, synapse=tau)
        nengo.Connection(thresh4, model.s2,         
                        transform=[-100], synapse=tau)
                        
        speed_signal = nengo.Node([0], label="speed_signal")
        nengo.Connection(speed_signal, swing1[1], synapse=None)
        nengo.Connection(speed_signal, stance1[1], synapse=None)
        nengo.Connection(speed_signal, swing2[1], synapse=None)
        nengo.Connection(speed_signal, stance2[1], synapse=None)
        
        start_state = nengo.Node(
            Piecewise({
                0: [0.01, 0, 0.5, 0],
                0.01: [0, 0, 0, 0],
            }), label="start_state")
        
        nengo.Connection(start_state[0], swing1[0], synapse=None)
        nengo.Connection(start_state[1], stance1[0], synapse=None)
        nengo.Connection(start_state[2], swing2[0], synapse=None)
        nengo.Connection(start_state[3], stance2[0], synapse=None)
    
    return model


params = [2.7136, 0, 1.1668, 1.6596, -0.009, 0.0921, -0.0636, -0.0934, 0.01246]

model = create_CPG(params=params)
    

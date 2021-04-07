import nengo
import numpy as np
from nengo.dists import Uniform, Choice, Exponential
from nengo.processes import Piecewise



def make_thresh_ens_net(threshold=0.5, thresh_func=lambda x: 1,
                        exp_scale=None, num_ens=1, net=None, **args):
    if net is None:
        label_str = args.get('label', 'Threshold_Ens_Net')
        net = nengo.Network(label=label_str)
    if exp_scale is None:
        exp_scale = (1 - threshold) / 10.0

    with net:
        ens_args = dict(args)
        ens_args['n_neurons'] = 5
        ens_args['dimensions'] = 1
        ens_args['intercepts'] = \
            Exponential(scale=exp_scale, shift=threshold,
                        high=1)
        ens_args['encoders'] = Choice([[1]])
        ens_args['eval_points'] = Uniform(min(threshold + 0.1, 1.0), 1.1)
        ens_args['n_eval_points'] = 5000

        net.input = nengo.Node(size_in=num_ens)
        net.output = nengo.Node(size_in=num_ens)

        for i in range(num_ens):
            thresh_ens = nengo.Ensemble(**ens_args)
            nengo.Connection(net.input[i], thresh_ens, synapse=None)
            nengo.Connection(thresh_ens, net.output[i],
                             function=thresh_func, synapse=None)
    return net
    
#u_hat = [1.4148, 0.0003, 1.1477, 0.5842, -0.0114, 0.0777, -0.0953, -0.1080, 0.1455]
u_hat = [ 2.7136, 0.0000, 1.1668, 1.6596, -0.0090, 0.0921, -0.0636, -0.0934, 0.1246]

X0    = np.array([u_hat[0], u_hat[1], u_hat[0], u_hat[1]])
Gu    = np.array([u_hat[2], u_hat[3], u_hat[2], u_hat[3]])
r     = u_hat[4]
r13   = u_hat[5]
r14   = u_hat[6]
r23   = u_hat[7]
r24   = u_hat[8]
Gx    = np.array([[r,   0,   r13, r14],
                  [0,   r,   r23, r24],
                  [r13, r23, r,   0],
                  [r14, r24, 0,   r]])
Gxic = Gx + 0

bZero = np.array([[1, 1, 0, 0],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 0, 1, 1],])

Gxic[bZero==1] = 0
Gx[bZero==0]   = 0;
U = np.array([1,1,1,1])


def odeCPG(x, U, Gu, Gx, Gxic):
    x = np.array(x)
    dX = X0 + np.multiply(Gu, U) + \
        np.dot(Gx, x) + \
        np.multiply(x > 0, np.dot(Gxic, np.array([1, 1, 1, 1])-x))
    
    return dX    

np.testing.assert_almost_equal(odeCPG([0.01, 0, 0.5, 0], U, Gu, Gx, Gxic), 
                             np.array([3.8628, 1.6596, 3.8737, 1.6596]), decimal=4)

np.testing.assert_almost_equal(odeCPG([0, 0, 0, 0], U, Gu, Gx, Gxic), 
                             np.array([3.8804, 1.6596, 3.8804, 1.6596]), decimal=4)
     
radius = 2.02 
state_neurons = 1000
tau = 0.005

init_swing = 2.7136
init_stance = 0
speed_swing = 1.1668
speed_stance = 1.6596
inner_inhibit = -0.009

swing_swing_connection = 0.0921
stance_swing_connection = -0.0636

swing_stance_connection = -0.0934
stance_stance_connection = 0.01246


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

model = nengo.Network(seed=41)
with model:
    swing1 = nengo.Ensemble(state_neurons, 2, radius=radius)
    
    stance1 = nengo.Ensemble(state_neurons, 2, radius=radius)
    
    swing2 = nengo.Ensemble(state_neurons, 2, radius=radius)
    
    stance2 = nengo.Ensemble(state_neurons, 2, radius=radius)
    
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
        }))
        
    s1 = nengo.Ensemble(2, 1, radius=radius, intercepts=[0,0], encoders=[[-1],[1]])
    nengo.Connection(s1, s1, synapse=tau)
    
    nengo.Connection(start_signal, s1, synapse=tau)
    
    nengo.Connection(s1, swing1.neurons, function=positive_signal, synapse=tau)
    nengo.Connection(s1, stance1.neurons, function=negative_signal, synapse=tau)
    
    thresh1 = make_thresh_ens_net(0.47, radius=1)
    nengo.Connection(swing1[0], thresh1.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh1.output, s1,
                    transform=[100], synapse=tau)
                    
    thresh2 = make_thresh_ens_net(0.47, radius=1)  
    nengo.Connection(stance1[0], thresh2.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh2.output, s1,         
                    transform=[-100], synapse=tau)
                    
    
    s2 = nengo.Ensemble(2, 1, radius=radius, intercepts=[0,0], encoders=[[-1],[1]])
    nengo.Connection(s2, s2, synapse=tau)
    
    nengo.Connection(start_signal, s2, synapse=tau)
    
    nengo.Connection(s2, swing2.neurons, function=positive_signal, synapse=tau)
    nengo.Connection(s2, stance2.neurons, function=negative_signal, synapse=tau)
    
    thresh3 = make_thresh_ens_net(0.47, radius=1)
    nengo.Connection(swing2[0], thresh3.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh3.output, s2,         
                    transform=[100], synapse=tau)
                    
    thresh4 = make_thresh_ens_net(0.47, radius=1)  
    nengo.Connection(stance2[0], thresh4.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh4.output, s2,         
                    transform=[-100], synapse=tau)
                    
    speed_signal = nengo.Node([0])
    nengo.Connection(speed_signal, swing1[1], synapse=None)
    nengo.Connection(speed_signal, stance1[1], synapse=None)
    nengo.Connection(speed_signal, swing2[1], synapse=None)
    nengo.Connection(speed_signal, stance2[1], synapse=None)
    
    start_state = nengo.Node(
        Piecewise({
            0: [0.01, 0, 0.5, 0],
            0.01: [0, 0, 0, 0],
        }))
    
    nengo.Connection(start_state[0], swing1[0], synapse=None)
    nengo.Connection(start_state[1], stance1[0], synapse=None)
    nengo.Connection(start_state[2], swing2[0], synapse=None)
    nengo.Connection(start_state[3], stance2[0], synapse=None)
        


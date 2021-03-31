import nengo
import numpy as np
from nengo.dists import Uniform, Choice, Exponential
from nengo.processes import Piecewise

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



X_init = np.array([0, .01, .5, 0])
def start_f(t):
    if t < 0.1:
        return X_init
    else:
        return [0, 0, 0, 0]

def odeCPG(x):
    x = np.array(x)
    dX = X0 + np.multiply(Gu, U) + \
        np.dot(Gx, x) + \
        np.multiply(x > 0, np.dot(Gxic, np.array([1, 1, 1, 1])-x))
    
    return dX    

np.testing.assert_almost_equal(odeCPG([0.01, 0, 0.5, 0]), 
                             np.array([3.8628, 1.6596, 3.8737, 1.6596]), decimal=4)

np.testing.assert_almost_equal(odeCPG([0, 0, 0, 0]), 
                             np.array([3.8804, 1.6596, 3.8804, 1.6596]), decimal=4)



def feedback1(x):
    dX = odeCPG([x, 0, 0, 0])
    return dX[0]*tau + x
    
def feedback2(x):
    dX = odeCPG([0, x, 0, 0])
    return dX[1]*tau + x
    
def feedback3(x):
    dX = odeCPG([0, 0, x, 0])
    return dX[2]*tau + x
    
def feedback4(x):
    dX = odeCPG([0, 0, 0, x])
    return dX[3]*tau + x



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

radius = 1.02
state_neurons = 1000
tau = 0.01

model = nengo.Network(seed=40)
with model:
    x1 = nengo.Ensemble(state_neurons, 1, radius=radius)
    
    x2 = nengo.Ensemble(state_neurons, 1, radius=radius)
    
    x3 = nengo.Ensemble(state_neurons, 1, radius=radius)
    
    x4 = nengo.Ensemble(state_neurons, 1, radius=radius)
    
    nengo.Connection(x1, x1, function=feedback1, synapse=tau)
    nengo.Connection(x2, x2, function=feedback2, synapse=tau)
    nengo.Connection(x3, x3, function=feedback3, synapse=tau)
    nengo.Connection(x4, x4, function=feedback4, synapse=tau)
    
    s1 = nengo.Ensemble(2, 1, radius=radius, intercepts=[0,0], encoders=[[-1],[1]])
    nengo.Connection(s1, s1, synapse=tau)
    
    
    start_s1 = nengo.Node(
        Piecewise({
            0: 1,
            0.01: 0,
        }))
     
    nengo.Connection(start_s1, s1, synapse=tau)
    
    def f1(x):
        if x > 0:
            return [-100]*1000
        else:
            return [0]*1000
    
    def f2(x): 
        if x < 0:
            return [-100]*1000
        else:
            return [0]*1000
          
    nengo.Connection(s1, x1.neurons, function=f1, synapse=tau)
    nengo.Connection(s1, x2.neurons, function=f2, synapse=tau)
    
    thresh1 = make_thresh_ens_net(0.47, radius=1)
    nengo.Connection(x1, thresh1.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh1.output, s1,         
                    transform=[100], synapse=tau)
                    
    thresh2 = make_thresh_ens_net(0.47, radius=1)  
    nengo.Connection(x2, thresh2.input, function= lambda x: x-0.5, synapse=tau)
    nengo.Connection(thresh2.output, s1,         
                    transform=[-100], synapse=tau)

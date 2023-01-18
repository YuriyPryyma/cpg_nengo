import nengo
from nengo.dists import Uniform, Choice, Exponential
import numpy as np


def make_thresh_ens_net(threshold=0.5, thresh_func=lambda x: 1,
                        exp_scale=None, num_ens=1, net=None, **args):
    if net is None:
        label_str = args.get('label', 'Threshold_Ens_Net')
        net = nengo.Network(label=label_str)
    if exp_scale is None:
        exp_scale = (1 - threshold) / 10.0

    with net:
        ens_args = dict(args)
        ens_args['n_neurons'] = 300
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

tau = 0.01

def grouth_equesions(x):
    return x + 4 * tau

model = nengo.Network(seed=42)
with model: 
    # state = nengo.Ensemble(1000, 1, radius=0.52)
    # nengo.Connection(state, state, function=grouth_equesions,  synapse=tau)
    # thresh = make_thresh_ens_net(0.37, radius=1)
    # nengo.Connection(state, thresh.input, function= lambda x: x-0.1, synapse=tau)
    # nengo.Connection(thresh.output, state,
    #                      transform=[-40],  synapse=tau)
    
    state = nengo.Ensemble(1000, 1, radius=1.02)
    nengo.Connection(state, state, function=grouth_equesions,  synapse=tau)
    thresh = make_thresh_ens_net(0.37, radius=1)
    nengo.Connection(state, thresh.input, function= lambda x: x-0.6, synapse=tau)
    nengo.Connection(thresh.output, state.neurons,
                     transform=[[-999]] * state.n_neurons,  synapse=tau)
    
                         
                         
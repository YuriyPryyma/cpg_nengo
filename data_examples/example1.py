"""
How to implement a simple state dynamic in Nengo?
We could start with a simple Integrator
For example, one described by the differential equation
dX = 1 - 0.005 * x
This is the first example of a recurrent network in the demos. 
It shows how neurons can be used to implement stable dynamics. 
Such dynamics are important for memory, noise cleanup, 
statistical inference, 
and many other dynamic transformations.

First, we need an object to "store" our state.
For this, we could use nengo.Ensemble
n_neurons: set number of neurons used to "store" state.
More neurons mean more precision
dimensions: the dimension of the state vector.
In our case, it's 1 as we have scalar
radius - define circle range which limits the value that could be stored
For radius=1 Ensemble could only store values in the [-1, 1] range

State updates we could implement as a python function
It should return a new value of a state after an update
For this, we multiply the calculated dx by tau. tau is the size of the postsynaptic filter.

With our function, we could do recurrent connections for our Ensemble. 
Nengo will automatically learn connections weights to learn provided function. 

Run the following example. Notice the state value going from 0(Ensemble) initial state to
+1 which is our radius. Value can not go farther.

"""
import nengo

model = nengo.Network()


def integrate(x):
    dX = 1 - 0.005 * x
    return dX * tau + x
    

radius = 1
tau = 0.1
state_neurons = 100

with model:
    state = nengo.Ensemble(n_neurons=state_neurons, 
            dimensions=1, 
            radius=radius)
    
    nengo.Connection(state, state, 
            function=integrate, synapse=tau)

    
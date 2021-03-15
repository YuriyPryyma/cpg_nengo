import nengo
import numpy as np
import scipy.io

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

tau = 0.1

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


# print(odeCPG([0.01, 0, 0.5, 0]))
# print(odeCPG([0, 0, 0, 0]))

edge = 1

def feedback(x):
    # print("input ", x)
    x = np.array(x)
    
    s1_old = (x[0]<=edge and x[0]>x[1]) or x[1]>edge
    s2_old = (x[2]<=edge and x[2]>x[3]) or x[3]>edge
    
    dX = odeCPG(x)
    x = dX*tau + x
    
    s1_new = (x[0]<=edge and x[0]>x[1]) or x[1]>edge
    s2_new = (x[2]<=edge and x[2]>x[3]) or x[3]>edge
    
    x[0] = x[0]*s1_new       + .01*((int(s1_old)-int(s1_new))<0);
    x[1] = x[1]*(not s1_new) + .01*((int(s1_old)-int(s1_new))>0);
    x[2] = x[2]*s2_new       + .01*((int(s2_old)-int(s2_new))<0);
    x[3] = x[3]*(not s2_new) + .01*((int(s2_old)-int(s2_new))>0);
    
    # print("output ", x)
    
    return x


mat = scipy.io.loadmat('x_0.mat')
x_0 = mat["X"].T
mat = scipy.io.loadmat('x_norm.mat')
x_norm = mat["X"].T

# x_pairs = [(x_0[i], x_0[i+1]) for i in range(x_0.shape[0]-1) ]

# values = list(map(np.linalg.norm, x_pairs))

# plt.hist(values)
# plt.show()
# sorted_keys = sorted(x_pairs, key=np.linalg.norm)
# print(x_pairs[0])


# eval_points = np.vstack((x_0, x_norm))

# eval_points = np.random.rand(100000, 4)*radius

radius = np.sqrt(2)+0.1

model = nengo.Network(seed=42)
with model:
    # start = nengo.Node(start_f)
    state = nengo.Ensemble(1000, 4, radius=radius)
    
    nengo.Connection(state, state, function=feedback, synapse=tau)#, eval_points=eval_points)
    # nengo.Connection(start, state)
    
    state_probe = nengo.Probe(state, synapse=tau)

with nengo.Simulator(model) as sim:
    sim.run(20)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4)
    axs[0].plot(sim.trange(), sim.data[state_probe][:, 0], label="x1")
    axs[1].plot(sim.trange(), sim.data[state_probe][:, 1], label="x2")
    axs[2].plot(sim.trange(), sim.data[state_probe][:, 2], label="x3")
    axs[3].plot(sim.trange(), sim.data[state_probe][:, 3], label="x4")
    fig.show()
    plt.show()

# np.set_printoptions(precision=3)

# for i in range(1000):
#   c_state = sim.data[state_probe][i]
#   print(i, c_state)


# print(len(sim.data[state_probe]))




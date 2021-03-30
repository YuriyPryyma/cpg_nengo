import nengo
import numpy as np
# import scipy.io

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

tau = 0.05

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


def rk4(x, t_step=0.001):
    x = np.array(x)

    k1 = odeCPG(x)
    k2 = odeCPG(x+t_step*k1/2)
    k3 = odeCPG(x+t_step*k2/2)
    k4 = odeCPG(x+t_step*k3)

    return t_step*(k1+2*k2+2*k3+k4)/6;


np.testing.assert_almost_equal(rk4([0, 0, 0, 0]), 
                             np.array([0.0039, 0.0017, 0.0039, 0.0017]), decimal=4)


def feedback(x):
    # print("input ", x)
    x = np.array(x)
    
    # s1_old = x[0]>=x[1]
    # s2_old = x[2]>=x[3]
    
    dX = odeCPG(x)
    x = dX*tau + x

    # x = x + rk4(x)
    # print(rk4(x))
    
    s1_new = (x[0]<=1 and x[0]>x[1]) or x[1]>1
    s2_new = (x[2]<=1 and x[2]>x[3]) or x[3]>1
    
    # x[0] = x[0]*s1_new       + .01*((int(s1_old)-int(s1_new))<0);
    # x[1] = x[1]*(not s1_new) + .01*((int(s1_old)-int(s1_new))>0);
    # x[2] = x[2]*s2_new       + .01*((int(s2_old)-int(s2_new))<0);
    # x[3] = x[3]*(not s2_new) + .01*((int(s2_old)-int(s2_new))>0);
    
    # print("\n\n")
    # print("int(s1_old)", int(s1_old))
    # print("int(s1_new)", int(s1_new))
    # print("(int(s1_old)-int(s1_new))", (int(s1_old)-int(s1_new)))
    # print("((int(s1_old)-int(s1_new))<0)", ((int(s1_old)-int(s1_new))<0))
    # print(".01*((int(s1_old)-int(s1_new))<0)", .01*((int(s1_old)-int(s1_new))<0))
    
    return x


# mat = scipy.io.loadmat('x_0.mat')
# x_0 = mat["X"].T
# mat = scipy.io.loadmat('x_norm.mat')
# x_norm = mat["X"].T
# eval_points = np.vstack((x_0, x_norm))


#[0.00391085 0.00166522 0.00389087 0.00168925]

radius = np.sqrt(2)+0.1
# eval_points = []

# for i in range(10000):
#     x = np.random.rand(4)*radius
#     for _ in range(4):
#         x = feedback(x)
#         eval_points.append(x)

# print(len(eval_points))

model = nengo.Network(seed=40)
with model:
    state = nengo.Ensemble(1000, 4, radius=radius)
    nengo.Connection(state, state, function=feedback, synapse=tau,)
    start = nengo.Node(start_f)
    nengo.Connection(start, state)
    
    state_probe = nengo.Probe(state, synapse=tau)

# with nengo.Simulator(model) as sim:
#     sim.run(20)

#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(4)
#     axs[0].plot(sim.trange(), sim.data[state_probe][:, 0], label="x1")
#     axs[1].plot(sim.trange(), sim.data[state_probe][:, 1], label="x2")
#     axs[2].plot(sim.trange(), sim.data[state_probe][:, 2], label="x3")
#     axs[3].plot(sim.trange(), sim.data[state_probe][:, 3], label="x4")
#     fig.show()
#     plt.show()

# np.set_printoptions(precision=3)

# for i in range(1000):
#   c_state = sim.data[state_probe][i]
#   print(i, c_state)


# print(len(sim.data[state_probe]))




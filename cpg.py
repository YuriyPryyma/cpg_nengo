import nengo
import numpy as np
from nengo.processes import Piecewise
from nengo.dists import Uniform

radius = np.sqrt(2)
tau = 0.01


def create_CPG(*, params, state_neurons=400):

    def swing_feedback(state):
        x, speed = state
        dX = params["init_swing"] + params["speed_swing"] * speed
        # + params["inner_inhibit"] * x
        return dX * tau + x

    def stance_feedback(state):
        x, speed = state
        dX = params["init_stance"] + params["speed_stance"] * speed
        # + params["inner_inhibit"] * x
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

    model = nengo.Network(seed=42)
    with model:
        eval_points_dist = Uniform(0, 1)

        swing1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                label="swing1",
                                eval_points=eval_points_dist)

        stance1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                 label="stance1",
                                 eval_points=eval_points_dist)

        swing2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                label="swing2",
                                eval_points=eval_points_dist)

        stance2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                 label="stance2",
                                 eval_points=eval_points_dist)

        eval_points_sample = np.random.rand(10000, 2)

        nengo.Connection(swing1, swing1[0], function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(stance1, stance1[0], function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(swing2, swing2[0], function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(stance2, stance2[0], function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)

        # for group in [(swing1, stance1, swing2, stance2),
        #               (swing2, stance2, swing1, stance1)]:
        #     swing_left, stance_left, swing_right, stance_right = group
        #     nengo.Connection(swing_left[0], swing_right[0],
        #                      function=lambda x:
        #                      tau * (1 - x) * params["sw_sw_con"],
        #                      synapse=tau)

        #     nengo.Connection(swing_left[0], stance_right[0],
        #                      function=lambda x:
        #                      tau * (1 - x) * params["sw_st_con"],
        #                      synapse=tau)

        #     nengo.Connection(stance_left[0], swing_right[0],
        #                      function=lambda x:
        #                      tau * (1 - x) * params["st_sw_con"],
        #                      synapse=tau)

        #     nengo.Connection(stance_left[0], stance_right[0],
        #                      function=lambda x:
        #                      tau * (1 - x) * params["st_st_con"],
        #                      synapse=tau)

        start_signal1 = nengo.Node(
            Piecewise({
                0: -1,
                0.01: 0,
            }), label="start_signal1")

        start_signal2 = nengo.Node(
            Piecewise({
                0: 1,
                0.01: 0,
            }), label="start_signal2")

        model.s1 = nengo.Ensemble(2, 1, radius=1, intercepts=[0, 0],
                                  encoders=[[-1], [1]], label="s1")
        nengo.Connection(model.s1, model.s1, synapse=tau)

        nengo.Connection(start_signal1, model.s1, synapse=tau)

        nengo.Connection(model.s1, swing1.neurons,
                         function=positive_signal, synapse=tau)
        nengo.Connection(model.s1, stance1.neurons,
                         function=negative_signal, synapse=tau)

        thresh1 = nengo.Ensemble(1, 1, intercepts=[0.47],
                                 encoders=[[1]], label="thresh1")
        nengo.Connection(swing1[0], thresh1,
                         function=lambda x: x - 0.5, synapse=tau)
        nengo.Connection(thresh1, model.s1,
                         transform=[100], synapse=tau)
        thresh2 = nengo.Ensemble(1, 1, intercepts=[0.47],
                                 encoders=[[1]], label="thresh2")
        nengo.Connection(stance1[0], thresh2,
                         function=lambda x: x - 0.5, synapse=tau)
        nengo.Connection(thresh2, model.s1,
                         transform=[-100], synapse=tau)

        model.s2 = nengo.Ensemble(2, 1, radius=1, intercepts=[0, 0],
                                  encoders=[[-1], [1]], label="s2")
        nengo.Connection(model.s2, model.s2, synapse=tau)

        nengo.Connection(start_signal2, model.s2, synapse=tau)

        nengo.Connection(model.s2, swing2.neurons,
                         function=positive_signal, synapse=tau)
        nengo.Connection(model.s2, stance2.neurons,
                         function=negative_signal, synapse=tau)

        thresh3 = nengo.Ensemble(1, 1, intercepts=[0.47],
                                 encoders=[[1]], label="thresh3")
        nengo.Connection(swing2[0], thresh3,
                         function=lambda x: x - 0.5, synapse=tau)
        nengo.Connection(thresh3, model.s2,
                         transform=[100], synapse=tau)

        thresh4 = nengo.Ensemble(1, 1, intercepts=[0.47],
                                 encoders=[[1]], label="thresh4")
        nengo.Connection(stance2[0], thresh4,
                         function=lambda x: x - 0.5, synapse=tau)
        nengo.Connection(thresh4, model.s2,
                         transform=[-100], synapse=tau)

        model.speed = nengo.Ensemble(state_neurons, 1, label="speed")
        nengo.Connection(model.speed, model.speed, synapse=0.1)
        nengo.Connection(thresh2, model.speed,
                         transform=[0.47], synapse=0.1,
                         eval_points=np.random.rand(10000, 1))

        nengo.Connection(model.speed, swing1[1], synapse=tau)
        nengo.Connection(model.speed, stance1[1], synapse=tau)
        nengo.Connection(model.speed, swing2[1], synapse=tau)
        nengo.Connection(model.speed, stance2[1], synapse=tau)

        # use initi only for stance
        # start_state = nengo.Node(
        #     Piecew    ise({
        #         0: [0.01, 0, 0, 0],
        #         0.01: [0, 0, 0, 0],
        #     }), label="start_state")

        # nengo.Connection(start_state[0], swing1[0], synapse=None)
        # nengo.Connection(start_state[1], stance1[0], synapse=None)
        # nengo.Connection(start_state[2], swing2[0], synapse=None)
        # nengo.Connection(start_state[3], stance2[0], synapse=None)

    return model


params = {
    "init_swing": 2.7136,
    "init_stance": 1,
    "speed_swing": 1.1668,
    "speed_stance": 1.6596,
    "inner_inhibit": -0.009,
    "sw_sw_con": 0.0921,
    "st_sw_con": -0.0636,
    "sw_st_con": -0.0934,
    "st_st_con": 0.01246,
}

model = create_CPG(params=params, state_neurons=2000)

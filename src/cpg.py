import nengo
import numpy as np
from nengo.processes import Piecewise
from nengo.dists import Uniform

radius = np.sqrt(2)
tau = 0.01


def create_CPG(*, params, state_neurons=400):

    def swing_feedback(state):
        x, speed = state
        dX = params["init_swing"] + params["speed_swing"] * speed + \
            params["inner_inhibit"] * x
        return dX * tau + x

    def stance_feedback(state):
        x, speed = state
        dX = params["init_stance"] + params["speed_stance"] * speed + \
            params["inner_inhibit"] * x
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

        model.swing1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                      label="swing1",
                                      eval_points=eval_points_dist)

        model.stance1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                       label="stance1",
                                       eval_points=eval_points_dist)

        model.swing2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                      label="swing2",
                                      eval_points=eval_points_dist)

        model.stance2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                       label="stance2",
                                       eval_points=eval_points_dist)

        eval_points_sample = np.random.rand(10000, 2)

        nengo.Connection(model.swing1, model.swing1[0],
                         function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.stance1, model.stance1[0],
                         function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.swing2, model.swing2[0],
                         function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.stance2, model.stance2[0],
                         function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)

        for group in [(model.swing1, model.stance1, model.swing2, model.stance2),
                      (model.swing2, model.stance2, model.swing1, model.stance1)]:
            swing_left, stance_left, swing_right, stance_right = group
            nengo.Connection(swing_left[0], swing_right[0],
                             function=lambda x:
                             tau * (1 - x) * params["sw_sw_con"],
                             synapse=tau)

            nengo.Connection(swing_left[0], stance_right[0],
                             function=lambda x:
                             tau * (1 - x) * params["sw_st_con"],
                             synapse=tau)

            nengo.Connection(stance_left[0], swing_right[0],
                             function=lambda x:
                             tau * (1 - x) * params["st_sw_con"],
                             synapse=tau)

            nengo.Connection(stance_left[0], stance_right[0],
                             function=lambda x:
                             tau * (1 - x) * params["st_st_con"],
                             synapse=tau)

        def create_switcher(leg, swing, stance, init="swing"):
            start_signal = nengo.Node(
                Piecewise({
                    0: -1 if init == "swing" else 1,
                    0.01: 0,
                }), label=f"init_phase{leg}")

            s = nengo.Ensemble(2, 1, radius=1, intercepts=[0, 0],
                               max_rates=[400, 400],
                               encoders=[[-1], [1]], label=f"s{leg}")
            nengo.Connection(s, s, synapse=tau)

            nengo.Connection(start_signal, s, synapse=tau)

            nengo.Connection(s, swing.neurons,
                             function=positive_signal, synapse=tau)
            nengo.Connection(s, stance.neurons,
                             function=negative_signal, synapse=tau)

            thresh_pos = nengo.Ensemble(1, 1, intercepts=[0.47], max_rates=[400],
                                        encoders=[[1]], label=f"thresh_pos{leg}")
            nengo.Connection(swing[0], thresh_pos,
                             function=lambda x: x - 0.5, synapse=tau)
            nengo.Connection(thresh_pos, s,
                             transform=[100], synapse=tau)
            thresh_neg = nengo.Ensemble(1, 1, intercepts=[0.47], max_rates=[400],
                                        encoders=[[1]], label=f"thresh_neg{leg}")
            nengo.Connection(stance[0], thresh_neg,
                             function=lambda x: x - 0.5, synapse=tau)
            nengo.Connection(thresh_neg, s,
                             transform=[-100], synapse=tau)

            return s, thresh_pos, thresh_neg

        model.s1, thresh1, thresh2 = create_switcher("1", model.swing1,
                                                     model.stance1, init="swing")
        model.s2, thresh3, thresh4 = create_switcher("2", model.swing2,
                                                     model.stance2, init="stance")

        init_stance = nengo.Node(
            Piecewise({
                0: params["init_stance_position"],
                0.01: 0,
            }), label="init_stance")

        nengo.Connection(init_stance, model.stance2[0], synapse=tau)

        model.speed = nengo.Ensemble(state_neurons, 1, label="speed")
        nengo.Connection(model.speed, model.speed, synapse=0.1)
        nengo.Connection(thresh2, model.speed,
                         transform=[0.47], synapse=0.1,
                         eval_points=np.random.rand(10000, 1))

        nengo.Connection(model.speed, model.swing1[1], synapse=tau)
        nengo.Connection(model.speed, model.stance1[1], synapse=tau)
        nengo.Connection(model.speed, model.swing2[1], synapse=tau)
        nengo.Connection(model.speed, model.stance2[1], synapse=tau)

    return model


params = {
    "init_swing": 2.7136,
    "init_stance": 1,
    "speed_swing": 1.1668,
    "speed_stance": 1.6596,
    "inner_inhibit": -0.009,
    "init_stance_position": 0,
    "sw_sw_con": 0.0921,
    "st_sw_con": -0.0636,
    "sw_st_con": -0.0934,
    "st_st_con": 0.01246,
}

model = create_CPG(params=params, state_neurons=2000)

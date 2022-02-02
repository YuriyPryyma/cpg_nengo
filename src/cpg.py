import nengo
import numpy as np
from nengo.processes import Piecewise
from nengo.dists import Uniform

radius = 1
tau = 0.01

def create_CPG(*, params, time, state_neurons=400, **args):

    def swing_feedback(x):
        dX = params["init_swing"] + params["inner_inhibit"] * x
        return dX * tau + x
        
    def stance_feedback(x):
        dX = params["init_stance"] + params["inner_inhibit"] * x
        return dX * tau + x
        
    def speed_swing(speed):
        dx = params["speed_swing"] * speed
        return dX * tau
        
    def speed_stance(speed):
        dx = params["speed_stance"] * speed
        return dX * tau

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

        model.swing1 = nengo.Ensemble(state_neurons, 1, radius=radius,
                                      label="swing1",
                                      eval_points=eval_points_dist)

        model.stance1 = nengo.Ensemble(state_neurons, 1, radius=radius,
                                       label="stance1",
                                       eval_points=eval_points_dist)

        model.swing2 = nengo.Ensemble(state_neurons, 1, radius=radius,
                                      label="swing2",
                                      eval_points=eval_points_dist)

        model.stance2 = nengo.Ensemble(state_neurons, 1, radius=radius,
                                       label="stance2",
                                       eval_points=eval_points_dist)

        eval_points_sample = np.random.rand(10000, 1)

        nengo.Connection(model.swing1, model.swing1,
                         function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.stance1, model.stance1,
                         function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.swing2, model.swing2,
                         function=swing_feedback,
                         synapse=tau, eval_points=eval_points_sample)
        nengo.Connection(model.stance2, model.stance2,
                         function=stance_feedback,
                         synapse=tau, eval_points=eval_points_sample)

        for group in [(model.swing1, model.stance1, model.swing2, model.stance2),
                      (model.swing2, model.stance2, model.swing1, model.stance1)]:
            swing_left, stance_left, swing_right, stance_right = group
            nengo.Connection(swing_left, swing_right,
                             function=lambda x:
                             tau * (1 - x) * params["sw_sw_con"],
                             synapse=tau)

            nengo.Connection(swing_left, stance_right,
                             function=lambda x:
                             tau * (1 - x) * params["sw_st_con"],
                             synapse=tau)

            nengo.Connection(stance_left, swing_right,
                             function=lambda x:
                             tau * (1 - x) * params["st_sw_con"],
                             synapse=tau)

            nengo.Connection(stance_left, stance_right,
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

            thresh_pos = nengo.Ensemble(1, 1, intercepts=[0.4], max_rates=[400],
                                        encoders=[[1]], label=f"thresh_pos{leg}")
            nengo.Connection(swing[0], thresh_pos,
                             function=lambda x: x - 0.5, synapse=tau)
            nengo.Connection(thresh_pos, s,
                             transform=[100], synapse=tau)
            thresh_neg = nengo.Ensemble(1, 1, intercepts=[0.4], max_rates=[400],
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

        model.speed = nengo.Node(lambda t: t/time, label="speed")
        
        nengo.Connection(model.speed, model.swing1,
                             function=lambda speed:
                             tau * speed * params["speed_swing"],
                             synapse=tau)
        
        nengo.Connection(model.speed, model.swing2,
                             function=lambda speed:
                             tau * speed * params["speed_swing"],
                             synapse=tau)
                             
        nengo.Connection(model.speed, model.stance1,
                             function=lambda speed:
                             tau * speed * params["speed_stance"],
                             synapse=tau)
                             
        nengo.Connection(model.speed, model.stance2,
                             function=lambda speed:
                             tau * speed * params["speed_stance"],
                             synapse=tau)

    return model


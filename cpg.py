import nengo
import numpy as np
from nengo.processes import Piecewise
from nengo.dists import Uniform
import tune_optimize_utils as utils

"""
State vector represent two variables in ranges [0, 1]
We need to extend radius to contain point (1,1)
"""
radius = np.sqrt(1**2+1**2)
"""
Synapse controls the size of a filter
In case of a big filter the system is highly stable, but
slowly responding to changes
"""
tau = 0.01


def create_CPG(*, params, state_neurons=400, noise=0, **args):
    """
    Fuctions creates spiking CPG model using input parameters
    
    Parameters
    ----------
    params : dict
        Includes dicionary with CPG model parameters 
        describing model dynamics
    state_neurons : int
        Number of parameners to use for swing and stance 
        state representation including integrator and speed control 
    noise : float
        Controls noise for integrator output for all 4 states
        
    Returns
    -------
    Nengo model
    """
    def swing_feedback(state):
        """
            Function transforms differential equations for
            swing state updates to a format acceptable for nengo
            recurrent connection
        """
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
        """
        Function implements inhibition for state neurons in case
        state variable is bigger then 0
        We use this function to switch active group 
        """
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
        # Becase we integrate from 0 to 1
        # There is no point to train our connections on negative numbers
        eval_points_dist = Uniform(0, 1)
        
        # Additional check in case we want to add noise
        if noise > 0:
            noise_proces = nengo.processes.WhiteNoise(
                dist=nengo.dists.Gaussian(0, noise), seed=1)
        else:
            noise_proces = None
        
        ## creating main state variables
        model.swing1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                      label="swing1",
                                      noise=noise_proces,
                                      eval_points=eval_points_dist)

        model.stance1 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                       label="stance1",
                                       noise=noise_proces,
                                       eval_points=eval_points_dist)

        model.swing2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                      label="swing2",
                                      noise=noise_proces,
                                      eval_points=eval_points_dist)

        model.stance2 = nengo.Ensemble(state_neurons, 2, radius=radius,
                                       label="stance2",
                                       noise=noise_proces,
                                       eval_points=eval_points_dist)
        
        # Nengo automatically sample points
        # In out case we want to control this process
        eval_points_sample = np.random.rand(10000, 2)
        
        # Setting recurrent connections
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

        if "time" in args:
            time = args["time"]
        else:
            time = 95

        model.speed = nengo.Ensemble(state_neurons, 1, label="speed")
        speed_set = nengo.Node(lambda t: t / time)
        nengo.Connection(speed_set, model.speed)

        nengo.Connection(model.speed, model.swing1[1], synapse=tau)
        nengo.Connection(model.speed, model.stance1[1], synapse=tau)
        nengo.Connection(model.speed, model.swing2[1], synapse=tau)
        nengo.Connection(model.speed, model.stance2[1], synapse=tau)
        
        if "disable" in args:
            def funk(t):

                np.random.seed(0)

                disable_count = int(count * (t/95))

                disable_i = np.random.choice(state_neurons, 
                    disable_count, replace=False)
                    
                neuron_signal = np.zeros(state_neurons)
                            
                neuron_signal[disable_i] = -30
                    
                return neuron_signal
                
            if_damage = nengo.Node(funk, label="dmg")

            nengo.Connection(if_damage, 
                    model.swing1.neurons,
                    synapse=tau)

            nengo.Connection(if_damage, 
                    model.stance1.neurons,
                    synapse=tau)

            nengo.Connection(if_damage, 
                    model.swing2.neurons,
                    synapse=tau)

            nengo.Connection(if_damage, 
                    model.stance2.neurons,
                    synapse=tau)
        

    return model


model = create_CPG(params=utils.best_params[0], state_neurons=2000)

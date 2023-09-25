from functools import partial

import nengo
import numpy as np
from nengo.processes import Piecewise
from nengo.dists import Uniform

"""
For one dimensional variable in range (-1, 1)
"""
radius = 1

"""
Synapse controls the size of a filter
In case of a big filter the system is highly stable, but
slowly responding to changes
"""
tau = 0.01

def create_CPG(*, params, time, state_neurons=400, **args):
    """
    Fuctions creates spiking CPG model using input parameters
    
    Parameters
    ----------
    params : dict
        Includes dicionary with CPG model parameters 
        describing model dynamics
    time : int
        Duration of simulation, for liniar speed change from 0 to 1, for that duration
    state_neurons : int
        Number of parameners to use for swing and stance 
        state representation including integrator and speed control      
    Returns
    -------
    Nengo model
    """

    def swing_feedback(x):
        """
        Function transforms differential equations for
        swing state updates to a format acceptable for nengo
        recurrent connection
        """
        dX = params["init_swing"] + params["inner_inhibit"] * x
        return dX * tau + x
        
    def stance_feedback(x):
        dX = params["init_stance"] + params["inner_inhibit"] * x
        return dX * tau + x
        
    def speed_swing(speed):
        """
        Function transforms differential equations for
        speed influence on swing to a nengo format
        """
        dx = params["speed_swing"] * speed
        return dX * tau
        
    def speed_stance(speed):
        dx = params["speed_stance"] * speed
        return dX * tau

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

        ## creating main state variables
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

        # Nengo automatically sample points for training
        # In out case we want to control this process
        eval_points_sample = np.random.rand(10000, 1)

        # Setting reccurent connections that is part of cpg dynamics
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

        # Setting connections between states for two limbs iteratively
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
            """
            Function implements group of nengo elements that are responsible
            for this functions:
                1) Inhibiting swing or stance ensemble depending on the state of switch element
                2) Initializing model dynamics, by setting the switch element to the corresponding state. 
                This initial state will be different for opposite limbs. pPssible values is -1 and 1
                3) Fliping the state of switch element if swing or stance ensemble riches threshold 

            """
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

        # Model start with all states equal zero, 
        # except one stance for one limb which is taken from parameters. 
        # As locomition for two limbs is shifted by phase they all cannot start from zeor 
        init_stance = nengo.Node(
            Piecewise({
                0: params["init_stance_position"],
                0.01: 0,
            }), label="init_stance")
        nengo.Connection(init_stance, model.stance2[0], synapse=tau)

        # User could provide it's own function for speed
        # or it will for from 0 to 1 in "time" seconds
        if "speed_f" in args:
            model.speed = nengo.Node(args["speed_f"], label="speed")
        elif "vis" in args:
            model.speed = nengo.Node([0], label="speed")
        else:
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

        # A user could provide a damage function
        # which will inhibit some state neurons depending on the conditions

        is_damage = False

        if args.get("vis_dmg", False):
            data_source = nengo.Node([0], label="dmg")
            def dmg_f(disable_count, phase):
                neuron_signal = np.zeros(state_neurons)

                for i in range(int(disable_count)):
                    neuron_signal[i] = -30

                return neuron_signal

            is_damage=True

        elif "dmg_f" in args:
            data_source = nengo.Node(lambda t: t, label="sim_time")

            dmg_f = partial(args["dmg_f"],
                state_neurons=state_neurons,
                time=time
            )

            is_damage=True

        if is_damage:
            nengo.Connection(data_source,
                    model.swing1.neurons,
                    function=partial(dmg_f, phase="swing1"),
                    synapse=None)

            nengo.Connection(data_source,
                    model.stance1.neurons,
                    function=partial(dmg_f, phase="stance1"),
                    synapse=None)

            nengo.Connection(data_source,
                    model.swing2.neurons,
                    function=partial(dmg_f, phase="swing2"),
                    synapse=None)

            nengo.Connection(data_source,
                    model.stance2.neurons,
                    function=partial(dmg_f, phase="stance2"),
                    synapse=None)

    return model


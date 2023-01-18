import nengo

from cpg import create_CPG

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

# creating model for visualizations
model = create_CPG(params=params, time=-1, state_neurons=300, vis=True)

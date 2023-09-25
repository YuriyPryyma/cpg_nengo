import nengo

from cpg import create_CPG
import tune_optimize_utils as utils

# creating model for visualizations
model = create_CPG(
    params=utils.best_params[0], 
    time=-1, 
    state_neurons=300, 
    vis=True,
    vis_dmg=False,
)

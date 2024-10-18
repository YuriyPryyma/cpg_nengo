from datetime import datetime
from ray import tune

from optimize import simulation_error

def tune_error(params):
    history, error, error_phase, error_speed, error_sym1, error_sym2 = simulation_error(params)
    return {
        "error":error, 
        "error_phase":error_phase,
        "error_speed":error_speed,
        "error_symmetricity1":error_sym1,
        "error_symmetricity2":error_sym2
    }


def exp_name(Alg):
    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    return f"{Alg}{now}"


best_params = [
    {
        "init_stance": 0.50191,
        "init_stance_position": 0.49604,
        "init_swing": 4.6773,
        "speed_stance": 3.8701,
        "speed_swing": 3.4240,
        "inner_inhibit": -0.44423,
        "st_st_con": 0.77360,
        "st_sw_con": -0.95284,
        "sw_st_con": -0.95586,
        "sw_sw_con": 0.082690,
    }
]

# parameters search space for tune
search_space = {
    "init_stance": tune.uniform(0.5, 1.5),
    "init_stance_position": tune.uniform(0, 1),
    "init_swing": tune.uniform(4.5, 5.5),
    "speed_stance": tune.uniform(3, 4),
    "speed_swing": tune.uniform(2, 5),
    "inner_inhibit": tune.uniform(-1, 0),
    "sw_sw_con": tune.uniform(-1, 1),
    "st_sw_con": tune.uniform(-1, 0),
    "sw_st_con": tune.uniform(-1, 0),
    "st_st_con": tune.uniform(0, 1),
}

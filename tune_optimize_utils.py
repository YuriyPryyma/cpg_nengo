from datetime import datetime
from ray import tune


def ray_wrapper(func):
    def inner(params):
        error, error_phase, error_speed, error_symmetricity = func(params)
        tune.report(error=error, error_phase=error_phase,
                    error_speed=error_speed,
                    error_symmetricity=error_symmetricity)

    return inner


def exp_name(Alg):
    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    return f"{Alg}{now}"


best_params = [
    {
        "init_swing": 3.2604,
        "init_stance": 0.70337,
        "speed_swing": 5.4470,
        "speed_stance": 2.6385,
        "init_stance_position": 0,
    }
]

search_space = {
    "init_swing": tune.uniform(3, 7),
    "init_stance": tune.uniform(0, 5),
    "speed_swing": tune.uniform(0, 7),
    "speed_stance": tune.uniform(0, 4),
    "init_stance_position": tune.uniform(0, 1),
    # "inner_inhibit": tune.uniform(-1, 4),
    # "sw_sw_con": tune.uniform(-1, 4),
    # "st_sw_con": tune.uniform(-1, 4),
    # "sw_st_con": tune.uniform(-1, 4),
    # "st_st_con": tune.uniform(-1, 4),
}

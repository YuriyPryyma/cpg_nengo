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
        "init_swing": 4.0234,
        "init_stance": 0.67401,
        "speed_swing": 5.6165,
        "speed_stance": 2.9812,
        "init_stance_position": 0.97386,
    }
]

search_space = {
    "init_swing": tune.uniform(3.5, 5),
    "init_stance": tune.uniform(0.5, 1),
    "speed_swing": tune.uniform(3.5, 6.5),
    "speed_stance": tune.uniform(2, 4),
    "init_stance_position": tune.uniform(0, 1),
    # "inner_inhibit": tune.uniform(-1, 1),
    # "sw_sw_con": tune.uniform(-1, 1),
    # "st_sw_con": tune.uniform(-1, 1),
    # "sw_st_con": tune.uniform(-1, 1),
    # "st_st_con": tune.uniform(-1, 1),
}

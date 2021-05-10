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
        "init_stance": 0.67401,
        "init_stance_position": 0.97386,
        "init_swing": 4.0234,
        "speed_stance": 2.9812,
        "speed_swing": 5.6165,
        "inner_inhibit": 0,
        "sw_sw_con": 0,
        "st_sw_con": 0,
        "sw_st_con": 0,
        "st_st_con": 0,
    }
]

search_space = {
    "init_stance": tune.uniform(0.6, 0.7),
    "init_stance_position": tune.uniform(0.5, 1),
    "init_swing": tune.uniform(3.5, 4.5),
    "speed_stance": tune.uniform(2.5, 3.5),
    "speed_swing": tune.uniform(4, 6.5),
    "inner_inhibit": tune.uniform(-1, 1),
    "sw_sw_con": tune.uniform(-1, 1),
    "st_sw_con": tune.uniform(-1, 1),
    "sw_st_con": tune.uniform(-1, 1),
    "st_st_con": tune.uniform(-1, 1),
}

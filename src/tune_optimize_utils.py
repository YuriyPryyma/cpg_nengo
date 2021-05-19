from datetime import datetime
from ray import tune


def ray_wrapper(func):
    def inner(params):
        error, error_phase, error_speed, error_sym1, error_sym2 = func(params)
        tune.report(error=error, error_phase=error_phase,
                    error_speed=error_speed,
                    error_symmetricity1=error_sym1,
                    error_symmetricity2=error_sym2)

    return inner


def exp_name(Alg):
    now = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    return f"{Alg}{now}"


best_params = [
    {
        "init_stance": 0.65995,
        "init_stance_position": 0.86596,
        "init_swing": 5.1540,
        "speed_stance": 3.4672,
        "speed_swing": 5.6165,
        "inner_inhibit": -0.28419,
        "sw_sw_con": -0.19935,
        "st_sw_con": 0.62068,
        "sw_st_con": -0.29972,
        "st_st_con": 0.62975,
    }
]

search_space = {
    "init_stance": tune.uniform(0, 2),
    "init_stance_position": tune.uniform(0, 1),
    "init_swing": tune.uniform(4, 6),
    "speed_stance": tune.uniform(3, 5),
    "speed_swing": tune.uniform(2, 6),
    "inner_inhibit": tune.uniform(-1, 1),
    "sw_sw_con": tune.uniform(-1, 1),
    "st_sw_con": tune.uniform(-1, 1),
    "sw_st_con": tune.uniform(-1, 1),
    "st_st_con": tune.uniform(-1, 1),
}

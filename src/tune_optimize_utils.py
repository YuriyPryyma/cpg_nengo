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
        "init_stance": 0.54503,
        "init_stance_position": 0.63792,
        "init_swing": 5.0592,
        "speed_stance": 3.6923,
        "speed_swing": 3.2043,
        "inner_inhibit": -0.45334,
        "st_st_con": 0.80108,
        "st_sw_con": -0.85135,
        "sw_st_con": -0.64631,
        "sw_sw_con": 0.10515,
    }
]

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

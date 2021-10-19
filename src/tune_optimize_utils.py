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
        "init_stance": 0.68159,
        "init_stance_position": 0.76815,
        "init_swing": 5.0656,
        "speed_stance": 3.5113,
        "speed_swing": 3.3579,
        "inner_inhibit": -0.31102,
        "st_st_con": 0.38125,
        "st_sw_con": -0.66464,
        "sw_st_con": -0.57982,
        "sw_sw_con": -0.68891,
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

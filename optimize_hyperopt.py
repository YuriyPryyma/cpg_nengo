from datetime import datetime
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from optimize import simulation_error


def ray_wrapper(func):
    def inner(params):
        error, error_phase, error_speed, error_symmetricity = func(params)
        tune.report(error=error, error_phase=error_phase,
                    error_speed=error_speed,
                    error_symmetricity=error_symmetricity)

    return inner


if __name__ == "__main__":

    ray_simulation_error = ray_wrapper(simulation_error)

    algo = HyperOptSearch()

    now = datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')

    analysis = tune.run(
        ray_simulation_error,
        name=f"Hyperout_{now}",
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=200,
        resources_per_trial={"cpu": 1},
        config={
            "init_swing": tune.uniform(0, 7),
            "init_stance": tune.uniform(0, 7),
            "speed_swing": tune.uniform(0, 7),
            "speed_stance": tune.uniform(0, 7),
            # "inner_inhibit": tune.uniform(-1, 4),
            # "sw_sw_con": tune.uniform(-1, 4),
            # "st_sw_con": tune.uniform(-1, 4),
            # "sw_st_con": tune.uniform(-1, 4),
            # "st_st_con": tune.uniform(-1, 4),
        })

    print("Best hyperparameters found were: ", analysis.best_config)

from datetime import datetime
from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
from optimize import simulation_error

def ray_wrapper(func):
    def inner(params):
        error, error_rms, error_range= func(params)
        tune.report(error=error, error_rms=error_rms, error_range=error_range)

    return inner


if __name__ == "__main__":

    ray_simulation_error = ray_wrapper(simulation_error)

    algo = HEBOSearch(
        random_state_seed=42,
        max_concurrent=1
    )

    analysis = tune.run(
        ray_simulation_error,
        name=f"HEBOSearch_{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}",
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=200,
        resources_per_trial={"cpu": 1},
        config={
            "init_swing": tune.uniform(0, 5),
            "init_stance": tune.uniform(0, 5),
            "speed_swing": tune.uniform(0, 5),
            "speed_stance": tune.uniform(0, 5),
            # "inner_inhibit": tune.uniform(-1, 4),
            # "sw_sw_con": tune.uniform(-1, 4),
            # "st_sw_con": tune.uniform(-1, 4),
            # "sw_st_con": tune.uniform(-1, 4),
            # "st_st_con": tune.uniform(-1, 4),
        })

    print("Best hyperparameters found were: ", analysis.best_config)

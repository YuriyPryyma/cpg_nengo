from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from optimize import simulation_error

def ray_wrapper(func):
    def inner(params):
        error, error_rms, error_range= func(params)
        tune.report(error=error, error_rms=error_rms, error_range=error_range)

    return inner

if __name__ == "__main__":

    ray_simulation_error = ray_wrapper(simulation_error)

    best_params = [{
        "init_swing": 2.7136,
        "init_stance": 0,
        "speed_swing": 1.1668,
        "speed_stance": 1.6596,
        # "inner_inhibit": -0.009,
        # "sw_sw_con": 0.0921,
        # "st_sw_con": -0.0636,
        # "sw_st_con": -0.0934,
        # "st_st_con": 0.01246,
    }]

    algo = HyperOptSearch(points_to_evaluate=best_params)

    analysis = tune.run(
        ray_simulation_error,
        name=f"Hyperout {datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}",
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=100,
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

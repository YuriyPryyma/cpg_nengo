from datetime import datetime
from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
from optimize import simulation_error
from optimize_hyperopt import ray_wrapper


if __name__ == "__main__":

    ray_simulation_error = ray_wrapper(simulation_error)

    previously_run_params = [
        {
            "init_swing": 4.04,
            "init_stance": 0.77,
            "speed_swing": 0.98,
            "speed_stance": 0.60,
        }
    ]

    algo = HEBOSearch(
        points_to_evaluate=previously_run_params,
        random_state_seed=42,
        max_concurrent=8
    )

    now = datetime.now().strftime('%Y_%m_%d-%H:%M:%S')

    analysis = tune.run(
        ray_simulation_error,
        name=f"HEBOSearch_{now}",
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=200,
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

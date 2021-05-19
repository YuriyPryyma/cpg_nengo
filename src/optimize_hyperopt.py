import os
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from optimize import simulation_error
import tune_optimize_utils as utils


if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = '0'
    ray_simulation_error = utils.ray_wrapper(simulation_error)

    algo = HyperOptSearch(points_to_evaluate=utils.best_params)

    analysis = tune.run(
        ray_simulation_error,
        name=utils.exp_name("Hyperout"),
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=1000,
        resources_per_trial={"cpu": 1, "gpu": 1 / 8},
        config=utils.search_space)

    print("Best hyperparameters found were: ", analysis.best_config)

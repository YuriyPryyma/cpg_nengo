import os
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from optimize import simulation_error
import tune_optimize_utils as utils


if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = '0'
    algo = HyperOptSearch(points_to_evaluate=utils.best_params)

    analysis = tune.run(
        utils.tune_error,
        name=utils.exp_name("Hyperout"),
        search_alg=algo,
        metric="error",
        mode="min",
        num_samples=1000,
        resources_per_trial={"cpu": 1},
        config=utils.search_space)

    print("Best hyperparameters found were: ", analysis.best_config)

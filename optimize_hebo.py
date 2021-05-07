from ray import tune
from ray.tune.suggest.hebo import HEBOSearch
from optimize import simulation_error
import tune_optimize_utils as utils


if __name__ == "__main__":

    print(simulation_error(utils.best_params[0], time=10))

    # ray_simulation_error = utils.ray_wrapper(simulation_error)

    # algo = HEBOSearch(points_to_evaluate=utils.best_params, max_concurrent=8)

    # analysis = tune.run(
    #     ray_simulation_error,
    #     name=utils.exp_name("Hebo"),
    #     search_alg=algo,
    #     metric="error",
    #     mode="min",
    #     num_samples=200,
    #     config=utils.search_space)

    # print("Best hyperparameters found were: ", analysis.best_config)

from ray import tune
import os
from ray.tune.suggest.hebo import HEBOSearch
from optimize import simulation_error
import tune_optimize_utils as utils

from time import time

if __name__ == "__main__":
    os.environ["PYOPENCL_CTX"] = '0'
    start = time()
    print(simulation_error(utils.best_params[0], progress_bar=True))
    print("took ", time() - start)


    """

    2000 cl took  97.32726407051086
    2000 normal took  107.81963443756104

    5000 normal took  373.71038460731506
    5000 cl took  124.63484764099121




    old
    0.38059564816101454, 0.14231079256513718, 0.46399999999999997, 0.012569711191754818)
    (0.3293509862204104, 0.14280035099839483, 0.37299999999999967, 0.00010127044403149512)

    new
    (0.6328806605704831, 0.1417859964569826, 0.34199999999999964, 0.6401893282270015)
    (0.6422056100066968, 0.14363610935198495, 0.2899999999999999, 0.7071390013094239)


    """

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

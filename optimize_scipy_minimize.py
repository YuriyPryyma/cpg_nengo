from scipy.optimize import minimize

from optimize import simulation_error


def scipy_wrapper(func):
    def inner(params, info):
        error, _, _, _ = func(params)

        print(f"step {info['Nfeval']} error {error}")
        info['Nfeval'] += 1

        return error

    return inner


if __name__ == "__main__":

    scipy_simulation_error = scipy_wrapper(simulation_error)

    x0 = [2.7136, 0, 1.1668, 1.6596, -0.009, 0.0921, -0.0636, -0.0934, 0.01246]

    res = minimize(scipy_simulation_error, x0,
                   method='nelder-mead', args=({'Nfeval': 0},))

    print(res)

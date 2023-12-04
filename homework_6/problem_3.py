import numpy as np
from ml_collections import config_dict

from morris_sampling import Parameter, MorrisSampling


def morris_config():
    config = config_dict.ConfigDict()
    config.num_runs = 20
    config.num_iterations = 100
    config.step_scale = 100
    config.reject_max = np.inf
    config.reject_min = -np.inf
    return config


def main(argv=None):

    def model_function(thetas, xdata):
        def _g(theta, a):
            return (np.abs(4 * theta - 2) + a) / (1 + a)

        a = np.array([78.0, 12.0, 0.5, 2.0, 97.0, 33.0])
        num_thetas = thetas.shape[0]
        product_vector = []
        for i in range(num_thetas):
            product_vector.append(_g(thetas[i], a[i]))

        product_vector = np.array(product_vector)
        return np.array([
            np.prod(product_vector, axis=0),
        ])

    p1 = Parameter('p1', 'O', 0, 1, [], [])
    p2 = Parameter('p2', 'O', 0, 1, [], [])
    p3 = Parameter('p3', 'O', 0, 1, [], [])
    p4 = Parameter('p4', 'O', 0, 1, [], [])
    p5 = Parameter('p5', 'O', 0, 1, [], [])
    p6 = Parameter('p6', 'O', 0, 1, [], [])

    params = [p1, p2, p3, p4, p5, p6]
    xdata = np.array([0.0])

    config = morris_config()

    sampler = MorrisSampling(model_function, params, xdata, config)
    mean, variance = sampler.sample()

    mean_indx = np.argsort(mean) + 1
    variance_indx = np.argsort(variance) + 1

    print(f"Means: {mean}")
    print(f"Ranked Means: {mean_indx}")
    print(f"Variances: {variance}")
    print(f"Ranked Variances: {variance_indx}")


if __name__ == '__main__':
    main()

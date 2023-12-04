import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

    p1 = Parameter('p1', 'U', 0, 1, [], [])
    p2 = Parameter('p2', 'U', 0, 1, [], [])
    p3 = Parameter('p3', 'U', 0, 1, [], [])
    p4 = Parameter('p4', 'U', 0, 1, [], [])
    p5 = Parameter('p5', 'U', 0, 1, [], [])
    p6 = Parameter('p6', 'U', 0, 1, [], [])

    params = [p1, p2, p3, p4, p5, p6]
    xdata = np.array([0.0])

    # Default config:
    config = morris_config()

    num_tests = 10
    num_runs = np.arange(1, 100, num_tests)
    step_scales = np.arange(1, 100, num_tests)
    means = []
    means_rank = []
    variances = []
    variances_rank = []

    for i in range(num_tests):
        for j in range(num_tests):
            config.num_runs = int(num_runs[i])
            config.step_scale = int(step_scales[j])
            sampler = MorrisSampling(model_function, params, xdata, config)
            mean, variance = sampler.sample()
            mean_indx = np.argsort(mean) + 1
            variance_indx = np.argsort(variance) + 1
            means.append(mean)
            means_rank.append(mean_indx)
            variances.append(variance)
            variances_rank.append(variance_indx)

    means = np.reshape(
        np.asarray(means), (num_tests, num_tests, len(params)),
    )
    means_rank = np.reshape(
        np.asarray(means_rank), (num_tests, num_tests, len(params)),
    )
    variances = np.reshape(
        np.asarray(variances), (num_tests, num_tests, len(params)),
    )
    variances_rank = np.reshape(
        np.asarray(variances_rank), (num_tests, num_tests, len(params)),
    )

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Morris Sampling')
    row = 0
    col = 0
    for i in range(len(params)):
        ax = axs[row, col]
        im = ax.imshow(means[:, :, i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'{params[i].param_name} Means')
        ax.set_xlabel('Step Scale')
        ax.set_ylabel('Number of Runs')

        col += 1
        if col == len(params)//2:
            col = 0
            row += 1

    fig.tight_layout()
    filename = 'means.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Morris Sampling')
    row = 0
    col = 0
    for i in range(len(params)):
        ax = axs[row, col]
        im = ax.imshow(means_rank[:, :, i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'{params[i].param_name} Ranked Means')
        ax.set_xlabel('Step Scale')
        ax.set_ylabel('Number of Runs')

        col += 1
        if col == len(params)//2:
            col = 0
            row += 1

    fig.tight_layout()
    filename = 'means_rank.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Morris Sampling')
    row = 0
    col = 0
    for i in range(len(params)):
        ax = axs[row, col]
        im = ax.imshow(variances[:, :, i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'{params[i].param_name} Variances')
        ax.set_xlabel('Step Scale')
        ax.set_ylabel('Number of Runs')

        col += 1
        if col == len(params)//2:
            col = 0
            row += 1

    fig.tight_layout()
    filename = 'variances.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle('Morris Sampling')
    row = 0
    col = 0
    for i in range(len(params)):
        ax = axs[row, col]
        im = ax.imshow(variances_rank[:, :, i])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title(f'{params[i].param_name} Ranked Variances')
        ax.set_xlabel('Step Scale')
        ax.set_ylabel('Number of Runs')

        col += 1
        if col == len(params)//2:
            col = 0
            row += 1

    fig.tight_layout()
    filename = 'variances_rank.pdf'
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename,
    )
    fig.savefig(filepath)


if __name__ == '__main__':
    main()

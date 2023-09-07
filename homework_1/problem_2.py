import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt


def model_function(theta, x):
    return theta[0] * x


def sum_squares(theta, data):
    y_data = np.squeeze(data.ydata[0])
    x_data = np.squeeze(data.xdata[0])
    y_model = model_function(theta, x_data)
    error = y_model - y_data
    residual = np.sum(error ** 2)
    return residual


def main(argv=None):
    # Set up figure path:
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')
    path = pathlib.Path(figure_path)
    path.mkdir(parents=True, exist_ok=True)

    # Generate data:
    ef = 5000e-6
    de = ef / 1000
    epsilon = np.arange(0, ef, de)
    E_mean = 1e9
    sig_E = E_mean * epsilon + 0e4 * E_mean * epsilon ** 3
    sig_noise = np.max(sig_E) * np.random.randn(1, sig_E.shape[0]) / 50
    sig_data = sig_E + sig_noise

    data_x = np.squeeze(epsilon)
    data_y = np.squeeze(sig_data)

    # Initial Model Params:
    E = 1.16e9
    theta = np.array([E])

    # Run Model:
    model_output = model_function(theta, data_x)

    # Plot:
    fig, ax = plt.subplots()
    ax.plot(data_x, data_y / 1e6, 'bo:', markersize=3, linewidth=2)
    ax.plot(data_x, model_output / 1e6, 'r-', linewidth=3)
    ax.set_xlabel('epsilon (m/m)')
    ax.set_ylabel('sigma (MPa)')
    ax.legend(['Data', 'Model'], loc='upper left')

    # Save figure:
    figure_name = os.path.join(figure_path, 'problem_2.png')
    fig.savefig(figure_name)


if __name__ == "__main__":
    main()

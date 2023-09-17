import os
import pathlib

import numpy as np
import scipy

import matplotlib.pyplot as plt

from pymcmcstat.MCMC import MCMC
from pymcmcstat import propagation
from pymcmcstat import mcmcplot


def square_wave(x, scale=1.15, period=2.0, random_scale=0.1):
    square_wave = scale * scipy.signal.square(period * x, duty=0.5)
    random_noise = (random_scale) * np.random.randn(x.shape[0])
    return square_wave + random_noise


def actual_model(x, half_period=1.0, accuracy=100):
    def vectorized_function(x):
        return np.sum(np.sin(n * np.pi * x / half_period) / n, axis=0)
    n = np.arange(1, accuracy, 2)
    return (4 / np.pi) * np.vectorize(vectorized_function)(x)


def model_function(theta, x):
    def vectorized_function(x):
        return np.sum(k * np.sin(a * x), axis=0)
    k, a = np.split(theta, 2)
    return np.vectorize(vectorized_function)(x)


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

    # Generate Square Wave Data:
    x_data = np.linspace(0, 2 * np.pi, 1000)
    period = 2.0
    y_data = square_wave(x_data, period=period)

    # Fourier Series Approximation:
    half_period = (np.pi/4)*period
    y_model = actual_model(x_data, half_period=half_period, accuracy=5)

    # Plot Data and Model:
    fig, ax = plt.subplots()
    ax.plot(x_data, y_data, 'bo:', markersize=3, linewidth=2)
    ax.plot(x_data, y_model, 'r-', linewidth=3)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend(['Data', 'Model'], loc='upper left')

    figure_name = os.path.join(figure_path, 'problem_1_data_and_model.png')
    fig.savefig(figure_name, dpi=300)

    # Initial Guess:
    k1 = 1
    k2 = 1/3
    k3 = 1/5
    a1 = 1 * np.pi / half_period
    a2 = 3 * np.pi / half_period
    a3 = 5 * np.pi / half_period
    theta = np.array([k1, k2, k3, a1, a2, a3], dtype=np.float64)

    # Set up MCMC:
    mcstat = MCMC()

    # Add data
    mcstat.data.add_data_set(x_data, y_data)

    # Define simulation options and model settings
    mcstat.simulation_options.define_simulation_options(
        nsimu=10000,
        method='dram',
        updatesigma=True,
    )
    mcstat.model_settings.define_model_settings(
        sos_function=sum_squares,
        sigma2=1e-4,
        S20=1e-4,
        N0=1,
        N=x_data.shape[0],
    )
    mcstat.parameters.add_model_parameter(
        name='k1',
        theta0=k1,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='k2',
        theta0=k2,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='k3',
        theta0=k3,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='a1',
        theta0=a1,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='a2',
        theta0=a2,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='a3',
        theta0=a3,
        minimum=0,
        maximum=np.inf,
    )

    # Run simulation
    mcstat.run_simulation()

    # Extract results
    results = mcstat.simulation_results.results
    chain = results['chain']
    s2chain = results['s2chain']
    names = results['names']

    # Define burnin
    burnin = int(results['nsimu'] / 2)

    # Display chain statistics
    mcstat.chainstats(chain[burnin:, :], results)

    density_plot = mcmcplot.plot_density_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_1_density.png')
    density_plot.savefig(figure_name)

    chain_plot = mcmcplot.plot_chain_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_1_chain.png')
    chain_plot.savefig(figure_name)

    pair_plot = mcmcplot.plot_pairwise_correlation_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_1_pairwise.png')
    pair_plot.savefig(figure_name)

    # Calculate Prediction Intervals
    out = propagation.calculate_intervals(
        chain=chain[burnin:, :],
        results=results,
        data=x_data,
        model=model_function,
        s2chain=s2chain[burnin:],
        waitbar=True,
    )

    fig, ax = propagation.plot_intervals(out, x_data, ydata=y_data, xdata=x_data)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend(
        ['95% Prediction Interval', '95% Credible Interval', 'Model Fit', 'Simulated Data'],
        loc='lower right',
    )
    figure_name = os.path.join(figure_path, 'problem_1_prediction_intervals.png')
    fig.savefig(figure_name)


if __name__ == "__main__":
    main()

import os
import pathlib

import numpy as np
import scipy

from pymcmcstat.MCMC import MCMC
from pymcmcstat import propagation
from pymcmcstat import mcmcplot


def model_function(theta, x):
    return theta[0] * x + theta[1] * x ** 3


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

    # Load mat file:
    filepath = os.path.join(os.path.dirname(__file__), 'data/stress_strain_randn.mat')
    data = scipy.io.loadmat(filepath)

    data_y = np.squeeze(data['sig_data'])
    data_x = np.squeeze(data['epsilon'])

    # Initial Guess:
    E = 1.0e9
    E2 = 1.0e13
    theta = np.array([E, E2])

    # Set up MCMC:
    mcstat = MCMC()

    # Add data
    mcstat.data.add_data_set(data_x, data_y)

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
        N=data_x.shape[0],
    )
    mcstat.parameters.add_model_parameter(
        name='theta_1',
        theta0=E,
        minimum=0,
        maximum=np.inf,
    )
    mcstat.parameters.add_model_parameter(
        name='theta_2',
        theta0=E2,
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
    figure_name = os.path.join(figure_path, 'problem_4_density.png')
    density_plot.savefig(figure_name)

    chain_plot = mcmcplot.plot_chain_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_4_chain.png')
    chain_plot.savefig(figure_name)

    pair_plot = mcmcplot.plot_pairwise_correlation_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_4_pairwise.png')
    pair_plot.savefig(figure_name)

    # Calculate Prediction Intervals
    out = propagation.calculate_intervals(
        chain=chain[burnin:, :],
        results=results,
        data=data_x,
        model=model_function,
        s2chain=s2chain[burnin:],
        waitbar=True,
    )

    fig, ax = propagation.plot_intervals(out, data_x, ydata=data_y, xdata=data_x)
    ax.set_xlabel('epsilon (m/m)')
    ax.set_ylabel('sigma (Pa)')
    ax.legend(
        ['95% Prediction Interval', '95% Credible Interval', 'Model Fit', 'Simulated Data'],
        loc='lower right',
    )
    figure_name = os.path.join(figure_path, 'problem_4_prediction_intervals.png')
    fig.savefig(figure_name)


if __name__ == "__main__":
    main()

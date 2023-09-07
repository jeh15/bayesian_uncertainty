import os

import numpy as np
import scipy

from pymcmcstat.MCMC import MCMC
from pymcmcstat import mcmcplot


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

    # Load mat file:
    filepath = os.path.join(os.path.dirname(__file__), 'data/stress_strain_randn.mat')
    data = scipy.io.loadmat(filepath)

    data_y = np.squeeze(data['sig_data'])
    data_x = np.squeeze(data['epsilon'])

    # Initial Model Params:
    E = 1.16e9
    theta = np.array([E])

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
        name='m',
        theta0=theta[0],
        minimum=0,
        maximum=np.inf,
    )

    # Run simulation
    mcstat.run_simulation()

    # Extract results
    results = mcstat.simulation_results.results
    chain = results['chain']
    names = results['names']

    # Define burnin
    burnin = int(results['nsimu'] / 2)

    # Display chain statistics
    mcstat.chainstats(chain[burnin:, :], results)

    chain_plot = mcmcplot.plot_chain_panel(
        chains=chain[:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_3_chain_no_burnin.png')
    chain_plot.savefig(figure_name)

    chain_plot = mcmcplot.plot_chain_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_3_chain_plot_burnin.png')
    chain_plot.savefig(figure_name)

    density_plot = mcmcplot.plot_density_panel(
        chains=chain[burnin:, :],
        names=names,
    )
    figure_name = os.path.join(figure_path, 'problem_3_density_plot.png')
    density_plot.savefig(figure_name)


if __name__ == "__main__":
    main()

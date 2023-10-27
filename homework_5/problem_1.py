import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy

import MCMC


def main(argv=None):
    def model_function(q, x):
        area = q[0] * x ** 2
        return area

    # Figure Path:
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    # Generate data:
    key = jax.random.PRNGKey(42)
    sample_size = 1000
    scale = 0.1
    x = jax.random.uniform(key, shape=(sample_size,))
    actual_q = np.array([np.pi])
    y = []
    for i in range(x.shape[0]):
        y.append(model_function(actual_q, x[i]))

    _, key = jax.random.split(key)
    y = np.asarray(y) + scale * jax.random.normal(key, shape=(sample_size,))

    # Initialize Parameters:
    q = jnp.zeros((1,))

    # Initialize Random Walk:
    num_chain_elements = 1000
    random_walk = MCMC.RandomWalk(
        model_function=model_function,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=0.1,
        num_chain_elements=num_chain_elements,
        custom_initial_guess=jnp.array([2.5]),
    )

    # Run MCMC Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    actual_theta = actual_q
    final_theta = theta[-1, :]

    print(f"Actual Theta: {actual_theta} \n Initial Theta: {random_walk.q0} \n Final Theta: {final_theta}")

    # Compute Chains:
    burnin = num_chain_elements // 10
    num_std = 1.0
    theta_1_kernel = scipy.stats.gaussian_kde(theta[burnin:, 0])
    theta_1_range = (
        np.mean(theta[burnin:, 0]) - num_std * np.std(theta[burnin:, 0]),
        np.mean(theta[burnin:, 0]) + num_std * np.std(theta[burnin:, 0]),
    )
    theta_1_range = np.linspace(*theta_1_range, 1000)
    theta_1_samples = theta_1_kernel.evaluate(theta_1_range)

    # Compute New Solution:
    model_response = []
    for i in range(x.shape[0]):
        model_response.append(model_function(final_theta, x[i]))

    model_response = np.asarray(model_response)

    # Plot Results:
    fig, ax = plt.subplots(3)
    plt.subplots_adjust(hspace=1.0)
    ax[0].scatter(x, y)
    ax[0].scatter(x, model_response, marker='.')
    ax[0].set_xlabel("radius sample")
    ax[0].set_ylabel("area")
    ax[0].legend(["Data", "Model"])
    ax[0].set_title("Area Model:")
    ax[1].scatter(theta_1_range, theta_1_samples)
    ax[1].set_xlabel("theta_1")
    ax[1].set_ylabel("pdf")
    ax[1].set_title("Density Plot:")
    iteration = np.arange(num_chain_elements)
    ax[2].scatter(iteration, theta)
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("theta_1")
    ax[2].set_title("Chain Plot:")
    figure_name = os.path.join(figure_path, 'problem_1.png')
    fig.savefig(fname=figure_name, dpi=300)


if __name__ == "__main__":
    main()

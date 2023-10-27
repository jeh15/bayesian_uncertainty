import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy
import distrax

import MCMC


def main(argv=None):
    def model_function(q, x):
        probability = jnp.where(
            x == 0,
            1 - q[0],
            q[0],
        )
        distribution = distrax.Bernoulli(probs=probability)
        probability_win = distribution.prob(1.0)
        return probability_win

    # Figure Path:
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    # Generate data:
    key = jax.random.PRNGKey(42)

    # Create data:
    sample_size = 1000
    x = jax.random.bernoulli(key, shape=(sample_size,))
    actual_q = np.array([1/3])
    y = []
    for i in range(sample_size):
        y.append(model_function(actual_q, x[i]))

    _, key = jax.random.split(key)
    y = np.asarray(y)

    # Initialize Parameters:
    q = jnp.zeros((1,))

    # Initialize Random Walk:
    num_chain_elements = 10000
    random_walk = MCMC.RandomWalk(
        model_function=model_function,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=0.1,
        num_chain_elements=num_chain_elements,
        custom_initial_guess=jnp.array([0.1]),
    )

    # Run MCMC Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    actual_theta = actual_q
    final_theta = theta[-1, :]
    print(f"Actual Theta: {actual_theta} \n Initial Theta: {random_walk.q0} \n Final Theta: {final_theta}")

    # Compute Chains:
    # burnin = num_chain_elements // 10
    burnin = 0
    num_std = 1.0
    theta_1_kernel = scipy.stats.gaussian_kde(theta[burnin:, 0])
    theta_1_range = (
        np.mean(theta[burnin:, 0]) - num_std * np.std(theta[burnin:, 0]),
        np.mean(theta[burnin:, 0]) + num_std * np.std(theta[burnin:, 0]),
    )
    theta_1_range = np.linspace(*theta_1_range, 1000)
    theta_1_samples = theta_1_kernel.evaluate(theta_1_range)

    # Plot Results:
    fig, ax = plt.subplots(4)
    plt.subplots_adjust(hspace=1.5)
    ax[0].scatter(theta_1_range, theta_1_samples)
    ax[0].set_xlabel("theta_1")
    ax[0].set_ylabel("pdf")
    ax[0].set_title("Probability of Winning if you Stay")
    iteration = np.arange(num_chain_elements)
    ax[2].scatter(iteration, theta)
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("theta_1")
    ax[2].set_title("Chain Plot: Stay")

    # Probability Switch:
    actual_q = np.array([2/3])
    y = []
    for i in range(sample_size):
        y.append(model_function(actual_q, x[i]))

    _, key = jax.random.split(key)
    y = np.asarray(y)

    # Initialize Parameters:
    q = jnp.zeros((1,))

    # Initialize Random Walk:
    num_chain_elements = 10000
    random_walk = MCMC.RandomWalk(
        model_function=model_function,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=0.1,
        num_chain_elements=num_chain_elements,
        custom_initial_guess=jnp.array([0.1]),
    )

    # Run MCMC Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    actual_theta = actual_q
    final_theta = theta[-1, :]
    print(f"Actual Theta: {actual_theta} \n Initial Theta: {random_walk.q0} \n Final Theta: {final_theta}")

    # Compute Chains:
    # burnin = num_chain_elements // 10
    burnin = 0
    num_std = 1.0
    theta_1_kernel = scipy.stats.gaussian_kde(theta[burnin:, 0])
    theta_1_range = (
        np.mean(theta[burnin:, 0]) - num_std * np.std(theta[burnin:, 0]),
        np.mean(theta[burnin:, 0]) + num_std * np.std(theta[burnin:, 0]),
    )
    theta_1_range = np.linspace(*theta_1_range, 1000)
    theta_1_samples = theta_1_kernel.evaluate(theta_1_range)

    ax[1].scatter(theta_1_range, theta_1_samples)
    ax[1].set_xlabel("theta_1")
    ax[1].set_ylabel("pdf")
    ax[1].set_title("Probability of Winning if you Switch")
    iteration = np.arange(num_chain_elements)
    ax[3].scatter(iteration, theta)
    ax[3].set_xlabel("iteration")
    ax[3].set_ylabel("theta_1")
    ax[3].set_title("Chain Plot: Switch")
    figure_name = os.path.join(figure_path, 'problem_2.png')
    fig.savefig(fname=figure_name, dpi=300)


if __name__ == "__main__":
    main()

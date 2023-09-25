import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    transition_matrix = np.array(
        [
            [1/3, 2/3],
            [1, 0]
        ],
        dtype=np.float64,
    )

    # Test to see if distribution is stationary:
    n = 10000
    test_iterations = 100
    distributions = []
    for i in range(test_iterations):
        a = np.random.rand(1)
        b = 1 - a
        initial_distribution = np.array(
            [a, b],
            dtype=np.float64,
        ).flatten()
        distributions.append(initial_distribution @ np.linalg.matrix_power(transition_matrix, n))

    distributions = np.asarray(distributions)
    mean_distribution = np.mean(distributions, axis=0)

    # If all sampled distributions are close to the average distribution,
    # then most likely the Markov Chain has a unique stationary distribution.
    if np.allclose(mean_distribution * np.ones_like(distributions), distributions):
        print(f"The Markov Chain has a unqiue stationary distribution: \n {mean_distribution}.")
        print(f"As a result, the Markov Chain is irreducible and aperiodic.")
    else:
        print(f"The Markov Chain does not have a unique stationary distribution.")

    # Test to see if distribution is stationary:
    n = 100
    test_iterations = 100
    distributions = np.zeros((test_iterations, n, 2))
    for i in range(test_iterations):
        a = np.random.rand(1)
        b = 1 - a
        initial_distribution = np.array(
            [a, b],
            dtype=np.float64,
        ).flatten()
        for j in range(n):
            distributions[i, j, :] = initial_distribution @ np.linalg.matrix_power(transition_matrix, j)

    distributions = np.asarray(distributions)
    fig, ax = plt.subplots(2)
    fig.tight_layout()
    for i in range(test_iterations):
        ax[0].plot(distributions[i, :, 0])
        ax[1].plot(distributions[i, :, 1])

    ax[0].plot(mean_distribution[0] * np.ones_like(distributions[0, :, 0]), linestyle='--')
    ax[1].plot(mean_distribution[1] * np.ones_like(distributions[0, :, 1]), linestyle='--')
    ax[0].set_title("Distribution of States")
    ax[0].set_ylabel("State 1")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylim([0, 1])
    ax[1].set_ylabel("State 2")
    ax[1].set_ylim([0, 1])
    ax[1].set_xlabel(r"Iterations")
    ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

    figure_name = os.path.join(figure_path, 'problem_3.png')
    fig.savefig(figure_name)

    # Find when the condition where
    # M >=1 such that P^m > 0 for all m >= M is false
    false_indicies = []
    for i in range(1, n):
        P_m = np.linalg.matrix_power(transition_matrix, i)
        if np.any(P_m <= 0):
            false_indicies.append(i)

    false_indicies = np.asarray(false_indicies)

    print(r"The condition where M >=1")
    print(r"such that P^m > 0 for all m >= M")
    if false_indicies.shape[0] == 1:
        print(f"is satisfied where M = {false_indicies[0] + 1}")
    else:
        print(r"is not satisfied for the current transition matrix.")

    # Solve lim P^m as m -> inf:
    P = np.linalg.matrix_power(transition_matrix, n)
    print(f"Limit of P^m as m -> inf: \n {P}")


if __name__ == "__main__":
    main()

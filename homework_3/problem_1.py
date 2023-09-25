import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    transition_matrix = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1/2, 0, 0, 1/2, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=np.float64,
    )

    distribution = np.array(
        [1/6, 1/6, 1/3, 1/6, 1/6],
        dtype=np.float64,
    )

    initial_distribution = np.array(
        [1/2, 0, 0, 1/2, 0],
        dtype=np.float64,
    )

    # Test to see if distribution is stationary:
    if np.allclose(distribution @ transition_matrix, distribution):
        print(f"The distribution pi = {distribution} is a stationary distribution.")
        print(r"As verified by pi = pi @ P")

    # Check for period of Markov Chain:
    n = 10000
    period = 3
    distributions = []
    for i in range(period, n, period):
        distributions.append(
            initial_distribution @ np.linalg.matrix_power(transition_matrix, i)
        )

    distributions = np.asarray(distributions)
    mean_distribution = np.mean(distributions, axis=0)

    # Check if Markov Chain has a period of k = 3:
    condition = np.allclose(
        mean_distribution * np.ones_like(distributions),
        distributions,
    )
    print(f"The Markov Chain has a period of k = 3: {condition}")

    # Visualize Distribution:
    n = 10000
    sample = 20
    distributions = []
    for i in range(n):
        distributions.append(
            initial_distribution @ np.linalg.matrix_power(transition_matrix, i)
        )

    distributions = np.asarray(distributions)
    mean_distribution = np.mean(distributions, axis=0)

    fig, ax = plt.subplots(5)
    fig.tight_layout()
    ax[0].plot(distributions[-sample:, 0])
    ax[0].plot(mean_distribution[0] * np.ones_like(distributions[-sample:, 0]), linestyle='--')
    ax[1].plot(distributions[-sample:, 1])
    ax[1].plot(mean_distribution[1] * np.ones_like(distributions[-sample:, 1]), linestyle='--')
    ax[2].plot(distributions[-sample:, 2])
    ax[2].plot(mean_distribution[2] * np.ones_like(distributions[-sample:, 2]), linestyle='--')
    ax[3].plot(distributions[-sample:, 3])
    ax[3].plot(mean_distribution[3] * np.ones_like(distributions[-sample:, 3]), linestyle='--')
    ax[4].plot(distributions[-sample:, 4])
    ax[4].plot(mean_distribution[4] * np.ones_like(distributions[-sample:, 4]), linestyle='--')
    ax[0].set_title("Distribution of States")
    ax[0].set_ylabel("State 1")
    ax[0].xaxis.set_visible(False)
    ax[0].set_ylim([0, 1])
    ax[1].set_ylabel("State 2")
    ax[1].xaxis.set_visible(False)
    ax[1].set_ylim([0, 1])
    ax[2].set_ylabel("State 3")
    ax[2].xaxis.set_visible(False)
    ax[2].set_ylim([0, 1])
    ax[3].set_ylabel("State 4")
    ax[3].xaxis.set_visible(False)
    ax[3].set_ylim([0, 1])
    ax[4].set_ylabel("State 5")
    ax[4].set_ylim([0, 1])
    ax[4].set_xlabel(f"Last {sample} Iterations out of {n}")
    ax[4].xaxis.set_major_locator(MaxNLocator(integer=True))

    figure_name = os.path.join(figure_path, 'problem_1.png')
    fig.savefig(figure_name)


if __name__ == "__main__":
    main()

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    transition_matrix = np.array(
        [
            [0, 1, 0, 0],
            [1/3, 0, 2/3, 0],
            [0, 0, 0, 1],
            [1/2, 0, 1/2, 0],
        ],
        dtype=np.float64,
    )

    initial_distribution = np.array(
        [1/4, 1/4, 1/4, 1/4],
        dtype=np.float64,
    )

    n = 10000
    sample = 20
    distributions = []
    for i in range(n):
        distribution = initial_distribution @ np.linalg.matrix_power(transition_matrix, i)
        distributions.append(distribution)

    distributions = np.asarray(distributions)
    average_distribution = np.mean(distributions, axis=0)

    if np.allclose(
        average_distribution * np.ones_like(distributions[-sample:]),
        distributions[-sample:],
        atol=1e-3,
    ):
        print(f"The initial distribution of {initial_distribution}")
        print(f"converges to {average_distribution}")
    else:
        print(f"The initial distribution of {initial_distribution}")
        print(r"does not converge to a stationary distribution.")

    print(f"This can be verified by looking at the last {sample} distributions calculated")
    print(f"as the limit of p @ P^m as m -> inf where inf was approximated as {n}.")
    print(f"{distributions[-sample:]}")

    fig, ax = plt.subplots(4)
    fig.tight_layout()
    ax[0].plot(distributions[-sample:, 0])
    ax[0].plot(average_distribution[0] * np.ones_like(distributions[-sample:, 0]), linestyle='--')
    ax[1].plot(distributions[-sample:, 1])
    ax[1].plot(average_distribution[1] * np.ones_like(distributions[-sample:, 1]), linestyle='--')
    ax[2].plot(distributions[-sample:, 2])
    ax[2].plot(average_distribution[2] * np.ones_like(distributions[-sample:, 2]), linestyle='--')
    ax[3].plot(distributions[-sample:, 3])
    ax[3].plot(average_distribution[3] * np.ones_like(distributions[-sample:, 3]), linestyle='--')
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
    ax[3].set_ylim([0, 1])
    ax[3].set_xlabel(f"Last {sample} Iterations out of {n}")
    ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))


    figure_name = os.path.join(figure_path, 'problem_2_1.png')
    fig.savefig(figure_name)

    initial_distribution = np.array(
        [1, 0, 0, 0],
        dtype=np.float64,
    )

    distributions = []
    for i in range(n):
        distribution = initial_distribution @ np.linalg.matrix_power(transition_matrix, i)
        distributions.append(distribution)

    distributions = np.asarray(distributions)
    average_distribution = np.mean(distributions, axis=0)

    if np.allclose(
        average_distribution * np.ones_like(distributions[-sample:]),
        distributions[-sample:],
        atol=1e-3,
    ):
        print(f"The initial distribution of {initial_distribution}")
        print(f"converges to {average_distribution}")
    else:
        print(f"The initial distribution of {initial_distribution}")
        print(r"does not converge to a stationary distribution.")

    print(f"This can be verified by looking at the last {sample} distributions calculated")
    print(f"as the limit of p @ P^m as m -> inf where inf was approximated as {n}.")
    print(f"{distributions[-sample:]}")

    print(f"As seen by the last {sample} distributions calculated,")
    print(r"the Markov chain has a period of 2.")

    fig, ax = plt.subplots(4)
    fig.tight_layout()
    ax[0].plot(distributions[-sample:, 0])
    ax[0].plot(average_distribution[0] * np.ones_like(distributions[-sample:, 0]), linestyle='--')
    ax[1].plot(distributions[-sample:, 1])
    ax[1].plot(average_distribution[1] * np.ones_like(distributions[-sample:, 1]), linestyle='--')
    ax[2].plot(distributions[-sample:, 2])
    ax[2].plot(average_distribution[2] * np.ones_like(distributions[-sample:, 2]), linestyle='--')
    ax[3].plot(distributions[-sample:, 3])
    ax[3].plot(average_distribution[3] * np.ones_like(distributions[-sample:, 3]), linestyle='--')
    ax[0].set_title("Distribution of States")
    ax[0].set_ylabel("State 1")
    ax[0].set_ylim([0, 1])
    ax[0].xaxis.set_visible(False)
    ax[1].set_ylabel("State 2")
    ax[1].set_ylim([0, 1])
    ax[1].xaxis.set_visible(False)
    ax[2].set_ylabel("State 3")
    ax[2].set_ylim([0, 1])
    ax[2].xaxis.set_visible(False)
    ax[3].set_ylabel("State 4")
    ax[3].set_ylim([0, 1])
    ax[3].set_xlabel(f"Last {sample} Iterations out of {n}")
    ax[3].xaxis.set_major_locator(MaxNLocator(integer=True))

    figure_name = os.path.join(figure_path, 'problem_2_2.png')
    fig.savefig(figure_name)


if __name__ == "__main__":
    main()

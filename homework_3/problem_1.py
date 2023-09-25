import numpy as np


def main(argv=None):
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


if __name__ == "__main__":
    main()

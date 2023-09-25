import numpy as np

def main(argv=None):
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
    sample = 4
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


if __name__ == "__main__":
    main()

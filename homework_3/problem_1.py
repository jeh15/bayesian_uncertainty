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

    n = 10000
    states = []
    for i in range(n):
        state = initial_distribution @ np.linalg.matrix_power(transition_matrix, i)
        states.append(state)

    states = np.asarray(states)
    stable_distribution = np.mean(states, axis=0)
    print(f"Steady state distribution of {n} samples: {stable_distribution}")


if __name__ == "__main__":
    main()

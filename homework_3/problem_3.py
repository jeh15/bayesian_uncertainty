import numpy as np


def main(argv=None):
    transition_matrix = np.array(
        [
            [1/3, 2/3],
            [1, 0]
        ],
        dtype=np.float64,
    )

    n = 10000
    P = []
    for i in range(n):
        P_m = np.linalg.matrix_power(transition_matrix, i)
        P.append(P_m)

    P = np.asarray(P)


if __name__ == "__main__":
    main()

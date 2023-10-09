import os

import numpy as np
import matplotlib.pyplot as plt


def random_range_finder(A, l, type='uniform'):
    # Get Shape of A:
    n, m = A.shape
    # Generate Random Matrix:
    rng = np.random.default_rng()
    if type == 'uniform':
        sigma = rng.uniform(low=0, high=1, size=(m, l))
    else:
        sigma = rng.standard_normal((m, l))
    Y = A @ sigma
    # Get orthonormal basis for Y:
    Q, R = np.linalg.qr(Y)
    return Q, R


def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    # Case ii:
    n = 101
    p = 1000
    l = 75
    t = np.linspace(0, 1, n)
    A = []
    for i in range(p):
        A.append(np.sin(2 * np.pi * (i + 1) * t))
    A = np.array(A).T

    fig, ax = plt.subplots(2)
    ax[0].plot(t, A[:, 0], label='A(:, 1)')
    ax[0].plot(t, A[:, 100], label='A(:, 101)')
    ax[0].plot(t, A[:, 200], label='A(:, 201)')
    ax[0].plot(t, A[:, 900], label='A(:, 901)')
    ax[0].set_ylabel('Columns Entries of A')
    ax[0].legend(loc='upper right')

    ax[1].plot(t, A[:, 0], linestyle='--', label='A(:, 1)')
    ax[1].plot(t, A[:, 50], label='A(:, 51)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Columns Entries of A')
    ax[1].legend(loc='upper right')

    figure_name = os.path.join(figure_path, 'problem_2_case_2_6_4.png')
    fig.savefig(fname=figure_name, dpi=300)

    # Compute singular values:
    _, S_A, _ = np.linalg.svd(A)
    Q, R = random_range_finder(A, l, type='uniform')
    B = Q.T @ A
    _, S_B, _ = np.linalg.svd(B)

    distance = np.abs(S_A[:l] - S_B)

    fig, ax = plt.subplots(2)
    fig.tight_layout()
    ax[0].plot(S_A, linestyle='none', label='Deterministic Algorithm', marker='*')
    ax[0].plot(S_B, linestyle='none', label='Randomized Algorithm', marker='o')
    ax[0].set_ylabel('Singular Values')
    ax[0].legend(loc='upper right')
    ax[1].plot(distance, linestyle='none', marker='.')
    ax[1].set_ylabel('Absolute Difference in Singular Values')
    figure_name = os.path.join(figure_path, 'problem_2_case_2_6_5.png')
    fig.savefig(fname=figure_name, dpi=300)

    # Case iii:
    n = 100
    p = 10 ** 6

    A = []
    for i in range(p):
        A.append(np.sin(2 * np.pi * (i + 1) * t))
    A = np.array(A).T

    Q, R = random_range_finder(A, l, type='uniform')
    diag_R = np.diag(R)

    fig, ax = plt.subplots()
    ax.plot(diag_R, linestyle='none', marker='.')
    ax.set_ylabel('Diagonal Values of R')
    figure_name = os.path.join(figure_path, 'problem_2_case_3.png')
    fig.savefig(fname=figure_name, dpi=300)


if __name__ == '__main__':
    main()

import os

import numpy as np
import matplotlib.pyplot as plt


def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    # Defined Mean and Covariance Matrcies
    mean = np.array([1, 2])
    covariance = np.array([[25, 4], [4, 1]])
    samples = np.random.default_rng().multivariate_normal(mean, covariance, size=5000)

    fig, ax = plt.subplots()
    ax.scatter(samples[:, 0], samples[:, 1])
    ax.set_xlabel('Q_1')
    ax.set_ylabel('Q_2')
    ax.set_title('Scatter Plot of Samples: Q_1 vs Q_2')
    figure_name = os.path.join(figure_path, 'problem_1_parameter_correlation.png')
    fig.savefig(fname=figure_name, dpi=300)

    # Linear Model:
    N = 100
    samples = np.random.default_rng().multivariate_normal(mean, covariance)
    q_1 = samples[0]
    q_2 = samples[1]
    x = np.linspace(0, 1, N)
    X = np.vstack([np.ones_like(x), x]).T
    q = np.array([[q_1, q_2]]).T
    y = X @ q

    # Least Squares Fit:
    q_estimate = np.linalg.inv((X.T @ X)) @ X.T @ y
    y_fit = q_estimate[0] + q_estimate[1] * x

    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, y_fit, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Least Squares Fit:')
    ax.legend(['Samples', 'Least Squares Fit'], loc='upper right')
    figure_name = os.path.join(figure_path, 'problem_1_linear_model.png')
    fig.savefig(fname=figure_name, dpi=300)

    # Find condition number:
    condition_number = np.linalg.cond(X.T @ X)
    print(f'Condition Number of X.T @ X: {condition_number}')


if __name__ == '__main__':
    main()

import os

import numpy as np
import scipy
from pydrake.solvers import MathematicalProgram, Solve

import matplotlib.pyplot as plt


def square_wave(x, scale=1.15, period=2.0, random_scale=0.1):
    square_wave = scale * scipy.signal.square(period * x, duty=0.5)
    random_noise = (random_scale) * np.random.randn(x.shape[0])
    return square_wave + random_noise


def model_function(theta, x):
    def vectorized_function(x):
        return np.sum(k * np.sin(a * x), axis=0)
    k, a = np.split(theta, 2)
    return np.vectorize(vectorized_function)(x)


def main(argv=None):
    # Set up figure path:
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')

    # Generate Square Wave Data:
    x_data = np.linspace(0, 2 * np.pi, 1000)
    period = 2.0
    y_data = square_wave(x_data, period=period)

    # Fourier Series Approximation:
    half_period = (np.pi/4)*period

    # Defined Optimization Variables and constants:
    approximation_resolution = 3
    optimization_variables = np.ones((approximation_resolution,), dtype=np.float64)
    stop = approximation_resolution + approximation_resolution - 1
    n = np.linspace(1, stop, approximation_resolution, dtype=np.float64)
    a = n * np.pi / half_period
    period_approximation = np.array([
        np.sin(a[0] * x_data),
        np.sin(a[1] * x_data),
        np.sin(a[2] * x_data),
    ]).T

    # Optimization Problem:
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(6)
    # Variable Bounds:
    for i in range(len(x)):
        prog.AddConstraint(x[i] >= 0)

    # Objective Function:
    y = x[0] * np.sin(x[3] * x_data) + x[1] * np.sin(x[4] * x_data) + x[2] * np.sin(x[5] * x_data)
    prog.AddCost(np.sum((y - y_data) ** 2))

    # Set Initial Guess:
    # Informed Guess:
    # prog.SetInitialGuess(x, np.array([1.46, 0.49, 0.29, a[0], a[1], a[2]]))
    # Random Guess:
    prog.SetInitialGuess(x, np.random.rand(x.shape[0]))

    # Solve Optimization Problem:
    results = Solve(prog)
    print(f"Optimization Results: {results.is_success()}")

    sol = results.GetSolution(x)
    k_sol = sol[:approximation_resolution]
    a_sol = sol[approximation_resolution:]

    print(r"Optimal Values:")
    for i in range(approximation_resolution):
        print(f"k_{int(n[i])} = {k_sol[i]} \t a_{int(n[i])} = {a_sol[i]}")

    # Comparison Plot:
    fig, ax = plt.subplots()
    optimization_output = model_function(
        sol, x_data,
    )
    ax.plot(x_data, y_data, 'bo:', markersize=3, linewidth=2)
    ax.plot(x_data, optimization_output, 'r-', linewidth=3)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend(['Data', 'Optimization'], loc='upper left')
    figure_name = os.path.join(figure_path, 'problem_3_nonlinear_optimization.png')
    fig.savefig(figure_name, dpi=300)


if __name__ == '__main__':
    main()

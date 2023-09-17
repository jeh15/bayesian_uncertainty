import os
import pathlib

import numpy as np
import scipy
import jax
import jax.numpy as jnp
import osqp

import matplotlib.pyplot as plt

jax.config.update('jax_enable_x64', True)


def square_wave(x, scale=1.15, period=2.0, random_scale=0.1):
    square_wave = scale * scipy.signal.square(period * x, duty=0.5)
    random_noise = (random_scale) * np.random.randn(x.shape[0])
    return square_wave + random_noise


def model_function(theta, x):
    def vectorized_function(x):
        return np.sum(k * np.sin(a * x), axis=0)
    k, a = np.split(theta, 2)
    return np.vectorize(vectorized_function)(x)


def objective_function(
    q: jax.typing.ArrayLike,
    a: jax.typing.ArrayLike,
    y_data: jax.typing.ArrayLike,
) -> jnp.ndarray:
    def model(q, a):
        return q @ a
    # VMAP over the data:
    y = jax.vmap(model, in_axes=(None, 0), out_axes=0)(q, a)
    residual = jnp.sum((y - y_data) ** 2)
    return residual


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
    optimization_variables = jnp.ones((approximation_resolution,), dtype=jnp.float64)
    stop = approximation_resolution + approximation_resolution - 1
    n = jnp.linspace(1, stop, approximation_resolution, dtype=jnp.float64)
    a = n * jnp.pi / half_period
    period_approximation = jnp.array([
        jnp.sin(a[0] * x_data),
        jnp.sin(a[1] * x_data),
        jnp.sin(a[2] * x_data),
    ]).T

    # Compute Hessian and Gradient functions:
    H_fn = jax.jacfwd(jax.jacrev(objective_function))
    f_fn = jax.jacfwd(objective_function)

    # Set up optimization problem:
    H = scipy.sparse.csc_matrix(
        np.asarray(
            H_fn(optimization_variables, period_approximation, y_data),
        ),
    )

    f = np.asarray(
        f_fn(optimization_variables, period_approximation, y_data),
    )
    A = scipy.sparse.csc_matrix(
        np.eye(approximation_resolution),
    )
    lb = np.zeros((approximation_resolution,))
    ub = np.inf * np.ones((approximation_resolution,))

    # Solve optimization problem:
    prob = osqp.OSQP()

    prob.setup(
        P=H,
        q=f,
        A=A,
        l=lb,
        u=ub,
        rho=1e-2,
        max_iter=10000,
        eps_abs=1e-10,
        eps_rel=1e-10,
        eps_prim_inf=1e-12,
        eps_dual_inf=1e-12,
        check_termination=25,
        polish=True,
        polish_refine_iter=3,
    )

    results = prob.solve()

    print(r"Optimal Values:")
    for i in range(approximation_resolution):
        print(f"k_{int(n[i])} = {results.x[i]}")

    # Parameter Estimator:
    a1 = 1 * np.pi / half_period
    a2 = 3 * np.pi / half_period
    a3 = 5 * np.pi / half_period
    x = np.array([np.sin(a1 * x_data), np.sin(a2 * x_data), np.sin(a3 * x_data)]).T
    q_estimate = np.linalg.inv((x.T @ x)) @ x.T @ y_data

    print(r"Parameter Estimate:")
    for i in range(q_estimate.shape[0]):
        print(f"k_{int(n[i])} = {q_estimate[i]}")

    # Variance Estimate:
    residual_estimate = y_data - x @ q_estimate
    variance_estimate = (residual_estimate.T @ residual_estimate) / (x.shape[0] - x.shape[1])

    print(f"Variance Estimate: {variance_estimate}")

    # Covariance Estimate:
    covariance_estimate = variance_estimate * np.linalg.inv(x.T @ x)

    print(f"Covariance Estimate: {covariance_estimate}")

    # Comparison Plot:
    fig, ax = plt.subplots()
    optimization_output = model_function(
        np.concatenate([results.x, a]), x_data,
    )
    estimator_output = model_function(
        np.concatenate([q_estimate, a]), x_data,
    )
    ax.plot(x_data, y_data, 'bo:', markersize=3, linewidth=2)
    ax.plot(x_data, optimization_output, 'r-', linewidth=3)
    ax.plot(x_data, estimator_output, 'g-', linewidth=3)
    ax.set_xlabel('t')
    ax.set_ylabel('y')
    ax.legend(['Data', 'Optimization', 'Estimator'], loc='upper left')
    figure_name = os.path.join(figure_path, 'problem_2_comparison.png')
    fig.savefig(figure_name, dpi=300)


if __name__ == '__main__':
    main()

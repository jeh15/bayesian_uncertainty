from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import scipy
import osqp


def compute_initial_values(
    fun: Callable,
    data: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    q: jax.typing.ArrayLike,
) -> jnp.ndarray:
    # Sum of Squares Objective Function:
    def _sum_of_squares(q, y, x):
        def _error(q, y, x):
            return y - fun(q, x)
        residual = jax.vmap(
            _error, in_axes=(None, 0, 0), out_axes=(0),
        )(q, y, x)
        return jnp.sum(residual ** 2)

    # Unpack data:
    y, x = data

    hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(_sum_of_squares)))
    hessian = hessian_fn(q, y, x)
    gradient_fn = jax.jit(jax.jacfwd(_sum_of_squares))
    gradient = gradient_fn(q, y, x)

    A = scipy.sparse.csc_matrix(
        np.eye(q.shape[0]),
    )
    lb = -np.inf * np.ones_like(q)
    ub = np.inf * np.ones_like(q)

    H = scipy.sparse.csc_matrix(hessian)
    f = np.asarray(gradient)

    # Setup OSQP program:
    program = osqp.OSQP()

    program.setup(
        P=H,
        q=f,
        A=A,
        l=lb,
        u=ub,
        verbose=False,
        warm_start=True,
        polish=True,
        rho=1e-2,
        max_iter=4000,
        eps_abs=1e-4,
        eps_rel=1e-4,
        eps_prim_inf=1e-6,
        eps_dual_inf=1e-6,
        check_termination=10,
        delta=1e-6,
        polish_refine_iter=5,
    )

    # Solve for parameters:
    result = program.solve()
    q0 = result.x

    # Compute sum of squares at q0:
    ss_q0 = _sum_of_squares(q0, y, x)

    # Computer initial variance estimate:
    variance_squared = ss_q0 / (y.shape[0] - q.shape[0])

    # Compute initial covariance estimate:
    sensitivity_matrix = np.expand_dims(
        np.asarray(gradient_fn(q0, y, x)),
        axis=0,
    )
    covariance = variance_squared * np.linalg.inv(
        sensitivity_matrix.T @ sensitivity_matrix,
    )
    R = np.linalg.cholesky(covariance)

    return q0


def compute_sum_of_squares_q0(
    fun: Callable,
    data: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
    q0: jax.typing.ArrayLike,
) -> jnp.ndarray:
    # Sum of Squares Objective Function:
    def _sum_of_squares(q, y, x):
        def _error(q, y, x):
            return y - fun(q, x)
        residual = jax.vmap(
            _error, in_axes=(None, 0, 0), out_axes=(0),
        )(q, y, x)
        return jnp.sum(residual ** 2)

    # Unpack data:
    y, x = data

    return _sum_of_squares(q0, y, x)


def main(argv=None):
    # Random Walk Metropolis:

    # Set number of chain elements and design parameters nu_s and sigma_s:
    alpha, beta = 1, 1
    M = 10
    nu_s = 2 * alpha
    sigma_s_sq = beta / alpha
    sigma_s = np.sqrt(beta / alpha)

    # Determine q^0 = argmin_q sum_{i=1}^M (y_i - f_i(q))^2:
    # y_i is data, f_i(q) is model

    def fun(q, x):
        y = q[0] * x + q[1]
        return y

    # Generate data:
    key = jax.random.PRNGKey(42)
    random_data = jax.random.uniform(key, shape=(10,2))
    x = random_data[:, 0]
    y = random_data[:, 1]

    # Number of parameters:
    q = jnp.zeros((2,))

    # Compute initial q:
    q0 = compute_initial_values(fun, (y, x), q)
    print(q0)

    # # Compute sum of squares at q0:
    # sum_of_squares_q0 = compute_sum_of_squares_q0(fun, (y, x), q0)
    # print(sum_of_squares_q0)


if __name__ == "__main__":
    main()

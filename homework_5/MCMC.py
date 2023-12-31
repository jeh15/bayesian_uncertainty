from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import osqp

# For plotting:
import matplotlib.pyplot as plt
import scipy

# Type hints:
PRNGKey = jax.random.PRNGKey


class RandomWalk():
    def __init__(
        self,
        model_function: Callable,
        parameters: jax.typing.ArrayLike,
        data: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike],
        *,
        rng_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
        num_observations: float = 1,
        num_chain_elements: int = 1000,
        custom_initial_guess: jax.typing.ArrayLike = None,
    ) -> None:
        # Initialize RNG Key:
        self.rng_key = rng_key

        # Number of loop iterations:
        self.num_chain_elements = num_chain_elements

        # Bind Model Function:
        self.model_function = model_function
        self.gradient_fn = jax.jit(jax.jacfwd(self.model_function))

        # Create Parameter vector:
        self.q = jnp.zeros_like(parameters)
        self.num_parameters = self.q.shape[0]

        # Unpack Data:
        self.y, self.x = data
        self.num_samples = self.y.shape[0]

        # Override initial guess:
        self.custom_q0 = custom_initial_guess

        # Isollate functions:
        self.sum_of_squares = lambda q, y, x: self._sum_of_squares(
            q, y, x, self.model_function,
        )
        self.compute_candidate = lambda q, R, rng_key: self._compute_candidate(
            q, R, rng_key, self.q.shape,
        )
        self.compute_sensitivity_matrix = lambda q, x: self._compute_sensitivity_matrix(
            q, x, self.model_function,
        )

        # Compute initial values:
        self.compute_initial_values()

        # Hyperparameters:
        # (mean_squared_error is estimated via compute_initial_values)
        self.num_observations = num_observations
        self.alpha = self.num_observations / 2
        self.beta = self.mean_squared_error * self.alpha

    def compute_initial_values(self) -> None:
        if self.custom_q0 is None:
            # Compute hessian and gradient functions:
            _hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(self.sum_of_squares)))
            hessian = _hessian_fn(
                self.q,
                self.y,
                self.x,
            )
            _gradient_fn = jax.jit(jax.jacfwd(self.sum_of_squares))
            gradient = _gradient_fn(
                self.q,
                self.y,
                self.x,
            )

            # Setup OSQP program:
            A = scipy.sparse.csc_matrix(
                np.eye(self.num_parameters),
            )
            lb = -np.inf * np.ones_like(self.q)
            ub = np.inf * np.ones_like(self.q)

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

            # Assert that program is solved:
            assert result.info.status == "solved"

            # Extract parameters:
            self.q0 = result.x
        else:
            self.q0 = self.custom_q0

        # Compute sum of squares at q0:
        self.ss_q0 = self.sum_of_squares(
            self.q0,
            self.y,
            self.x,
        )

        # Computer initial variance estimate:
        self.mean_squared_error = self.ss_q0 / (
            self.num_samples - self.num_parameters
        )
        self.variance_squared = self.ss_q0 / (
            self.num_samples - self.num_parameters
        )

        # Compute initial covariance estimate:
        sensitivity_matrix = self.compute_sensitivity_matrix(
            self.q0,
            self.x,
        )
        self.covariance = self.variance_squared * np.linalg.inv(
            sensitivity_matrix.T @ sensitivity_matrix,
        )
        self.R = np.linalg.cholesky(self.covariance)

    def loop(self):
        def _loop_iteration(carry, data):
            # Unpack carry:
            q, ss_q, variance_squared, alpha, beta, rng_key = carry

            # Compute candidate and Sum of Squares:
            q_candidate = self.compute_candidate(
                q,
                self.R,
                rng_key,
            )
            ss_candidate = self.sum_of_squares(
                q_candidate,
                self.y,
                self.x,
            )
            ss_q = self.sum_of_squares(
                q,
                self.y,
                self.x,
            )

            # Compute acceptance probability:
            ratio = jnp.exp(
                -(ss_candidate - ss_q) / (2 * variance_squared)
            )
            minimum_array = jnp.asarray([1.0, ratio])
            alpha = jnp.min(minimum_array)
            u = jax.random.uniform(rng_key)

            # Update q and ss_q:
            q, ss_q = jax.lax.cond(
                u < alpha,
                lambda x: (x[0], x[1]),
                lambda x: (x[2], x[3]),
                (q_candidate, ss_candidate, q, ss_q),
            )

            # Update variance:
            variance = self._sample_inverse_gamma_distribution(
                rng_key,
                alpha,
                beta
            )
            variance_squared = variance ** 2

            # Update alpha and beta:
            alpha = 0.5 * (self.num_observations + self.num_samples)
            beta = 0.5 * (
                self.num_observations * self.mean_squared_error + ss_q
            )

            # Generate new RNG Key for next iteration:
            _, rng_key = jax.random.split(rng_key)

            carry = (q, ss_q, variance_squared, alpha, beta, rng_key)
            data = (q, ss_q, variance_squared)

            return carry, data

        # Initialize carry:
        initial_carry = (
            self.q0,
            self.ss_q0,
            self.variance_squared,
            self.alpha,
            self.beta,
            self.rng_key,
        )

        carry, data = jax.lax.scan(
            _loop_iteration,
            initial_carry,
            (),
            self.num_chain_elements,
        )

        return data

    @partial(jax.jit, static_argnames=("self", "shape",))
    def _compute_candidate(
        self,
        q: jax.typing.ArrayLike,
        R: jax.typing.ArrayLike,
        rng_key: jax.random.PRNGKey,
        shape: tuple[int, ...],
    ) -> jnp.ndarray:
        _, rng_key = jax.random.split(rng_key)
        return q + R @ jax.random.normal(rng_key, shape)

    @partial(jax.jit, static_argnames=("self", "fun",))
    def _sum_of_squares(self, q, y, x, fun):
        def _error(q, y, x):
            return y - fun(q, x)
        residual = jax.vmap(
            _error, in_axes=(None, 0, 0), out_axes=(0),
        )(q, y, x)
        return jnp.sum(residual ** 2)

    @partial(jax.jit, static_argnames=("self", "fun",))
    def _compute_sensitivity_matrix(self, q, x, fun):
        return jax.vmap(
            jax.jacfwd(fun), in_axes=(None, 0), out_axes=(0),
        )(q, x)

    @partial(jax.jit, static_argnames=("self",))
    def _sample_inverse_gamma_distribution(self, rng_key, alpha, beta):
        # Random sample:
        x = jax.random.uniform(rng_key, minval=0, maxval=jnp.inf)
        # Probability density function: Inverse Gamma at x
        gamma_value = jax.scipy.special.gamma(alpha)
        inverse_gamma_sample = (
            (beta ** alpha) / (gamma_value)
        ) * (1 / x) ** (alpha + 1) * jnp.exp(-beta / x)
        return inverse_gamma_sample


def main(argv=None):
    def fun(q, x):
        y = q[0] * x + q[1]
        return y

    # Generate data:
    key = jax.random.PRNGKey(42)
    sample_size = 1000
    param_size = 2
    scale = 0.1
    x = jax.random.uniform(key, shape=(sample_size,))
    random_q = jax.random.uniform(key, shape=(param_size,))
    print("Actual theta:", random_q)
    y = []
    for i in range(x.shape[0]):
        y.append(fun(random_q, x[i]))

    _, key = jax.random.split(key)
    y = np.asarray(y) + scale * jax.random.normal(key, shape=(sample_size,))

    # Actual Function:
    x_plot = np.linspace(0, 1, 100)
    y_plot = []
    for i in range(x_plot.shape[0]):
        y_plot.append(fun(random_q, x_plot[i]))
    y_plot = np.asarray(y_plot)

    # Plot data:
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x_plot, y_plot, color="black", linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Number of parameters:
    q = jnp.zeros((2,))

    # Initialize Random Walk:
    num_chain_elements = 5000
    random_walk = RandomWalk(
        model_function=fun,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=0.1,
        num_chain_elements=num_chain_elements,
    )

    # Test Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    print("Final theta:", theta[-1, :])

    approximated_y = []
    for i in range(x_plot.shape[0]):
        approximated_y.append(fun(theta[-1, :], x_plot[i]))
    approximated_y = np.asarray(approximated_y)
    ax.plot(x_plot, approximated_y, color="red", linestyle='--')
    plt.show()

    burnin = num_chain_elements // 10
    theta_1_range = (np.min(theta[burnin:, 0]), np.max(theta[burnin:, 0]))
    theta_2_range = (np.min(theta[burnin:, 1]), np.max(theta[burnin:, 1]))

    fig, ax = plt.subplots(2)
    num_std = 1.0
    theta_1_kernel = scipy.stats.gaussian_kde(theta[burnin:, 0])
    theta_2_kernel = scipy.stats.gaussian_kde(theta[burnin:, 1])
    theta_1_range = (
        np.mean(theta[burnin:, 0]) - num_std * np.std(theta[burnin:, 0]),
        np.mean(theta[burnin:, 0]) + num_std * np.std(theta[burnin:, 0]),
    )
    theta_1_range = np.linspace(*theta_1_range, 1000)
    theta_2_range = (
        np.mean(theta[burnin:, 1]) - num_std * np.std(theta[burnin:, 1]),
        np.mean(theta[burnin:, 1]) + num_std * np.std(theta[burnin:, 1]),
    )
    theta_2_range = np.linspace(*theta_2_range, 1000)
    theta_1_samples = theta_1_kernel.evaluate(theta_1_range)
    theta_2_samples = theta_2_kernel.evaluate(theta_2_range)
    ax[0].scatter(theta_1_range, theta_1_samples)
    ax[1].scatter(theta_2_range, theta_2_samples)
    plt.show()


if __name__ == "__main__":
    main()

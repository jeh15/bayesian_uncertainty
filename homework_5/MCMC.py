from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import osqp

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
        num_observations: int = 1,
        num_chain_elements: int = 1000,
    ) -> None:
        # Initialize RNG Key:
        self.rng_key = rng_key

        # Number of loop iterations:
        self.num_chain_elements = num_chain_elements

        # Bind Model Function:
        self.model_function = model_function

        # Create Parameter vector:
        self.q = jnp.zeros_like(parameters)
        self.num_parameters = self.q.shape[0]

        # Unpack Data:
        self.y, self.x = data
        self.num_samples = self.y.shape[0]

        # Isollate functions:
        self.sum_of_squares = lambda q, y, x: self._sum_of_squares(
            q, y, x, self.model_function,
        )
        self.compute_candidate = lambda q, R, rng_key: self._compute_candidate(
            q, R, rng_key, self.q.shape,
        )

        # Compute initial values:
        self.compute_initial_values()

        # Hyperparameters:
        # (mean_squared_error is estimated via compute_initial_values)
        self.num_observations = num_observations
        self.alpha = self.num_observations / 2
        self.beta = self.mean_squared_error * self.alpha

    def compute_initial_values(self) -> None:
        # Compute hessian and gradient functions:
        self.hessian_fn = jax.jit(jax.jacfwd(jax.jacrev(self.sum_of_squares)))
        hessian = self.hessian_fn(
            self.q,
            self.y,
            self.x,
        )
        self.gradient_fn = jax.jit(jax.jacfwd(self.sum_of_squares))
        gradient = self.gradient_fn(
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

        self.q0 = result.x

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
        sensitivity_matrix = np.expand_dims(
            np.asarray(
                self.gradient_fn(self.q0, self.y, self.x)),
            axis=0,
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
            q = jnp.where(u < alpha, q_candidate, q)
            ss_q = jnp.where(u < alpha, ss_candidate, ss_q)

            # Update variance:
            variance = jax.random.gamma(
                rng_key,
                alpha,
            ) / beta
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


def main(argv=None):
    def fun(q, x):
        y = q[0] * x + q[1]
        return y

    # Generate data:
    key = jax.random.PRNGKey(42)
    random_data = jax.random.uniform(key, shape=(10, 2))
    x = random_data[:, 0]
    y = random_data[:, 1]

    # Number of parameters:
    q = jnp.zeros((2,))

    # Initialize Random Walk:
    random_walk = RandomWalk(
        model_function=fun,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=1,
        num_chain_elements=1000,
    )

    # Get Values from CLass:
    print("q0:", random_walk.q0)
    print("R:", random_walk.R)

    # Test Candidate:
    candidate = random_walk.compute_candidate(
        random_walk.q0,
        random_walk.R,
        random_walk.rng_key,
    )
    print("candidate:", candidate)

    # Test Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    print("Initial theta:", theta[0, :])
    print("Final theta:", theta[-1, :])


if __name__ == "__main__":
    main()

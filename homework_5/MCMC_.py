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
        parameters: jax.typing.ArrayLike,
        data: tuple[jax.typing.ArrayLike, jax.typing.ArrayLike, jax.random.PRNGKeyArray],
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
        def model_function(q, x, key):
            win_percentage = jnp.where(x < q, 2/3, 1/3)
            a = jnp.array([0, 1]).flatten()
            p = jnp.array([1-win_percentage, win_percentage]).flatten()
            result = jax.random.choice(
                key,
                a=a,
                p=p,
            )
            return result
        self.model_function = model_function
        self.gradient_fn = jax.jit(jax.jacfwd(self.model_function))

        # Create Parameter vector:
        self.q = jnp.zeros_like(parameters)
        self.num_parameters = self.q.shape[0]

        # Unpack Data:
        self.y, self.x, self.keys = data
        self.num_samples = self.y.shape[0]

        # Override initial guess:
        self.custom_q0 = custom_initial_guess

        # Isollate functions:
        self.sum_of_squares = lambda q, y, x, keys: self._sum_of_squares(
            q, y, x, keys, self.model_function,
        )
        self.compute_candidate = lambda q, R, rng_key: self._compute_candidate(
            q, R, rng_key, self.q.shape,
        )
        self.compute_sensitivity_matrix = lambda q, x, keys: self._compute_sensitivity_matrix(
            q, x, keys, self.model_function,
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
                self.keys,
            )
            _gradient_fn = jax.jit(jax.jacfwd(self.sum_of_squares))
            gradient = _gradient_fn(
                self.q,
                self.y,
                self.x,
                self.keys,
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
            self.keys,
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
            self.keys,
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
                self.keys,
            )
            ss_q = self.sum_of_squares(
                q,
                self.y,
                self.x,
                self.keys,
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
    def _sum_of_squares(self, q, y, x, keys, fun):
        def _error(q, y, x, keys):
            return y - fun(q, x, keys)
        residual = jax.vmap(
            _error, in_axes=(None, 0, 0, 0), out_axes=(0),
        )(q, y, x, keys)
        return jnp.sum(residual ** 2)

    @partial(jax.jit, static_argnames=("self", "fun",))
    def _compute_sensitivity_matrix(self, q, x, keys, fun):
        return jax.vmap(
            jax.jacfwd(fun), in_axes=(None, 0, 0), out_axes=(0),
        )(q, x, keys)

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
    
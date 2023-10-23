import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy
import distrax

import MCMC


def main(argv=None):
    def model_function(q, x):
        stay_probability = q[0] * (q[1] * 1/3 + q[2] * 1/3) / (q[1] * 1/3)
        # switch_probability = q[1] * (q[2] * 1/3 + q[3] * 1/3) / (q[3] * 1/3)
        # probability = jnp.where(x == 0, stay_probability, switch_probability)
        return stay_probability

    # Generate data:
    key = jax.random.PRNGKey(42)

    # Create data:
    sample_size = 1000
    x = jax.random.bernoulli(key, shape=(sample_size,))
    actual_q = np.array([1/3, 1/2, 1])
    y = []
    for i in range(sample_size):
        y.append(model_function(actual_q, x[i]))

    _, key = jax.random.split(key)
    y = np.asarray(y)

    # Initialize Parameters:
    q = jnp.zeros((3,))

    # Initialize Random Walk:
    num_chain_elements = 10000
    random_walk = MCMC.RandomWalk(
        model_function=model_function,
        parameters=q,
        data=(y, x),
        rng_key=key,
        num_observations=0.1,
        num_chain_elements=num_chain_elements,
        custom_initial_guess=jnp.array([0.5, 0.5, 0.5]),
    )

    # Run MCMC Loop:
    data = random_walk.loop()
    theta, ss, variance = data
    actual_theta = actual_q
    final_theta = theta[-1, :]


if __name__ == "__main__":
    main()

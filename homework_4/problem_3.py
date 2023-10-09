import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def model(m, c, k, z_0, z_1, t):
    c_1 = z_0
    c_2 = (c / (jnp.sqrt(4 * m * k - c ** 2))) * z_0 + (2 * m / (jnp.sqrt(4 * m * k - c ** 2))) * z_1 
    A = jnp.exp(-(c / (2*m)) * t)
    B = c_1 * jnp.cos(jnp.sqrt(4 * m * k - c ** 2) * t / (2 * m))
    C = c_2 * jnp.sin(jnp.sqrt(4 * m * k - c ** 2) * t / (2 * m))
    z = A * (B + C)
    return z

def main(argv=None):
    figure_path = os.path.join(os.path.dirname(__file__), 'figures')
    q = jnp.array([1, 0.5, 10])
    z_0 = 1
    z_1 = 0
    t = 1

    model_fn = lambda m, c, k: model(m, c, k, z_0, z_1, t)

    # Take the derivative of the model function:
    grad_fn_m = jax.grad(model_fn, argnums=0)
    grad_fn_c = jax.grad(model_fn, argnums=1)
    grad_fn_k = jax.grad(model_fn, argnums=2)

    dz_dm = grad_fn_m(*q)
    dz_dc = grad_fn_c(*q)
    dz_dk = grad_fn_k(*q)

    print(f'dz_dm: {dz_dm}')
    print(f'dz_dc: {dz_dc}')
    print(f'dz_dk: {dz_dk}')


if __name__ == '__main__':
    main()

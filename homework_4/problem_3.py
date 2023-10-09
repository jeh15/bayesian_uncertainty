import os

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from problem_2 import random_range_finder


def model(m, c, k, t, z_0, z_1):
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
    t = jnp.linspace(0, 1, 101)

    model_fn = lambda m, c, k, t: model(m, c, k, t, z_0, z_1)

    # Take the derivative of the model function:
    grad_fn_m = jax.grad(model_fn, argnums=0)
    grad_fn_c = jax.grad(model_fn, argnums=1)
    grad_fn_k = jax.grad(model_fn, argnums=2)

    dz_dm = jax.vmap(
        grad_fn_m, in_axes=(None, None, None, 0), out_axes=0,
    )(*q, t)
    dz_dc = jax.vmap(
        grad_fn_c, in_axes=(None, None, None, 0), out_axes=0,
    )(*q, t)
    dz_dk = jax.vmap(
        grad_fn_k, in_axes=(None, None, None, 0), out_axes=0,
    )(*q, t)

    print(f'dz_dm: {dz_dm}')
    print(f'dz_dc: {dz_dc}')
    print(f'dz_dk: {dz_dk}')

    X = jnp.vstack([dz_dm, dz_dc, dz_dk])

    # Fisher Information Matrix:
    F = X.T @ X
    _, S, _ = np.linalg.svd(F)
    _, R = np.linalg.qr(F)
    _, S_R, _ = np.linalg.svd(R)
    Q_r, R_r = random_range_finder(F, 101, type='uniform')
    B = Q_r.T @ F
    _, S_B, _ = np.linalg.svd(B)

    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.plot(S, linestyle='none', label='Deterministic Algorithm', marker='*')
    ax.plot(S_R, linestyle='none', label='QR Factorization', marker='.')
    ax.plot(S_B, linestyle='none', label='Random Range Finder', marker='o')
    ax.set_ylabel('Singular Values')
    ax.legend(loc='upper right')

    figure_name = os.path.join(figure_path, 'problem_3.png')
    fig.savefig(fname=figure_name, dpi=300)


if __name__ == '__main__':
    main()

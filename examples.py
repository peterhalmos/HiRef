import jax
import jax.numpy as jnp
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchdyn.datasets import generate_moons

# -----------------------------
# 8-Gaussians (vectorized)
# -----------------------------
def eight_normal_sample(key, n, dim=2, scale=1.0, var=1.0, dtype=jnp.float32):
    """
    Sample n points from a mixture of 8 isotropic Gaussians placed
    on the axes and diagonals at radius `scale`, variance `var`.
    """
    key_noise, key_cat = jax.random.split(key)

    # Isotropic noise
    noise = jnp.sqrt(var) * jax.random.normal(key_noise, (n, dim), dtype=dtype)

    # Centers on unit circle axes + diagonals (then scaled)
    centers = jnp.array([
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (0.0, -1.0),
        (1.0 / jnp.sqrt(2.0),  1.0 / jnp.sqrt(2.0)),
        (1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)),
        (-1.0 / jnp.sqrt(2.0),  1.0 / jnp.sqrt(2.0)),
        (-1.0 / jnp.sqrt(2.0), -1.0 / jnp.sqrt(2.0)),
    ], dtype=dtype) * scale

    # Uniform categorical over 8 components (logits=0 => uniform)
    idx = jax.random.categorical(key_cat, jnp.zeros((8,), dtype=dtype), shape=(n,))

    # Gather centers and add noise
    data = centers[idx] + noise
    return data.astype(dtype)

def sample_8gaussians(key, n, scale=5.0, var=0.2, dtype=jnp.float32):
    return eight_normal_sample(key, n, dim=2, scale=scale, var=var, dtype=dtype)
    
def sample_moons(n):
    x0, _ = generate_moons(n, noise=0.5)
    x0 = torch_to_jax(x0)
    return x0 * 3 - 1

# -----------------------------
# Two Moons (scikit-learn style)
# -----------------------------
def torch_to_jax(t: torch.Tensor):
    return jnp.asarray(t.detach().cpu().numpy())

def plot_alignment(frontier, X, Y, *, 
                   max_lines_per_leaf=20,   
                   sample_frac=None,       
                   linewidth=0.4,
                   point_size=8,
                   title="HiRef-LR alignment (subset of lines)"):
    """
    frontier: list of (idxX, idxY) leaves from hiref_lr
    X, Y: (n,2) arrays (JAX or NumPy). Will be viewed as NumPy.
    """
    Xnp = np.asarray(X)
    Ynp = np.asarray(Y)

    plt.figure(figsize=(7, 7))
    plt.scatter(Xnp[:, 0], Xnp[:, 1], s=point_size, label="X")
    plt.scatter(Ynp[:, 0], Ynp[:, 1], s=point_size, marker="x", label="Y")

    rng = np.random.default_rng(0)
    for idxX, idxY in frontier:
        ix = np.asarray(idxX)
        iy = np.asarray(idxY)

        if sample_frac is not None and 0 < sample_frac < 1:
            k = max(1, int(len(ix) * sample_frac))
            sel = rng.choice(len(ix), size=k, replace=False)
            ix = ix[sel]; iy = iy[sel]

        if max_lines_per_leaf is not None:
            k = min(len(ix), max_lines_per_leaf)
            ix = ix[:k]; iy = iy[:k]

        # draw lines
        for i, j in zip(ix, iy):
            plt.plot([Xnp[i, 0], Ynp[j, 0]], 
                     [Xnp[i, 1], Ynp[j, 1]], 
                     linewidth=linewidth, alpha=0.5)

    plt.legend()
    plt.axis("equal")
    plt.title(title)
    plt.show()
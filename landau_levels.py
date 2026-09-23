"""Plot the wavefunctions of the first several Landau levels.

Two complementary pictures of a charged particle in a uniform magnetic field
B = B z-hat, in units where the magnetic length l_B = sqrt(hbar/(eB)) = 1:

1. Landau gauge A = (0, Bx, 0).  Eigenstates factorize as
       psi_{n,k}(x, y) = e^{i k y} phi_n(x - k l_B^2),
   where phi_n is the n-th harmonic-oscillator eigenfunction,

       phi_n(u) = (1 / sqrt(2^n n! sqrt(pi))) H_n(u) exp(-u^2 / 2),

   with energy E_n = hbar omega_c (n + 1/2).  The transverse profile phi_n
   and its density are what the 1D panels show.

2. Symmetric gauge A = B/2 (-y, x).  The lowest-Landau-level states with
   angular momentum m are the familiar rings

       psi_m(z) ~ z^m exp(-|z|^2 / 4),   z = x + i y,

   and more generally the (n, m) states are built from Laguerre polynomials,

       psi_{n,m} ~ r^{|m|} L_n^{|m|}(r^2 / 2) exp(-r^2 / 4) e^{i m theta}.

   The 2D panels show |psi_{n,m}|^2 for the first few n at fixed m.

Usage:
    python landau_levels.py                  # writes landau_levels_1d.png
                                             #    and landau_levels_2d.png
    python landau_levels.py --levels 6 --show
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_genlaguerre, factorial, hermite


# --------------------------------------------------------------------------
# Landau gauge: 1D transverse profiles
# --------------------------------------------------------------------------
def landau_gauge_profile(n: int, u: np.ndarray) -> np.ndarray:
    """n-th harmonic-oscillator eigenfunction phi_n(u), l_B = 1."""
    norm = 1.0 / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi))
    return norm * hermite(n)(u) * np.exp(-(u**2) / 2.0)


def plot_landau_gauge(n_levels: int, x_max: float = 6.0, n_points: int = 1200):
    """Wavefunctions and densities stacked on the Landau energy ladder."""
    x = np.linspace(-x_max, x_max, n_points)
    # Vertical scale so the wiggles are visible but levels stay separated.
    amp = 0.7

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, n_levels))

    for n in range(n_levels):
        phi = landau_gauge_profile(n, x)
        energy = n + 0.5  # in units of hbar * omega_c
        for ax, data, scale in (
            (axes[0], phi, amp / np.abs(phi).max()),
            (axes[1], phi**2, amp / (phi**2).max()),
        ):
            ax.axhline(energy, color="0.8", lw=0.8, zorder=0)
            ax.plot(x, energy + scale * data, color=colors[n], lw=1.8)
            ax.fill_between(
                x, energy, energy + scale * data,
                color=colors[n], alpha=0.25, lw=0,
            )
        axes[0].text(
            -x_max + 0.1, energy + 0.12, f"n = {n}",
            color=colors[n], fontsize=10, va="bottom",
        )

    for ax, title in (
        (axes[0], r"wavefunction  $\varphi_n(x-x_0)$"),
        (axes[1], r"density  $|\varphi_n(x-x_0)|^2$"),
    ):
        ax.set_title(title)
        ax.set_xlabel(r"$(x - x_0)\,/\,\ell_B$")
        ax.set_xlim(-x_max, x_max)

    axes[0].set_ylabel(r"energy  $E_n / \hbar\omega_c = n + 1/2$")
    axes[0].set_ylim(-0.4, n_levels + 0.4)
    fig.suptitle(
        "Landau levels in the Landau gauge "
        r"($\psi_{n,k} = e^{iky}\varphi_n(x-k\ell_B^2)$)",
        fontsize=13,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Symmetric gauge: 2D densities
# --------------------------------------------------------------------------
def symmetric_gauge_state(n: int, m: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """psi_{n,m}(x, y) in the symmetric gauge, l_B = 1 (up to normalization)."""
    r2 = x**2 + y**2
    theta = np.arctan2(y, x)
    radial = (
        r2 ** (abs(m) / 2.0)
        * eval_genlaguerre(n, abs(m), r2 / 2.0)
        * np.exp(-r2 / 4.0)
    )
    return radial * np.exp(1j * m * theta)


def plot_symmetric_gauge(n_levels: int, m: int = 2, r_max: float = 8.0,
                         n_points: int = 400):
    """|psi_{n,m}|^2 for the first n_levels Landau levels at fixed m."""
    grid = np.linspace(-r_max, r_max, n_points)
    xx, yy = np.meshgrid(grid, grid)

    fig, axes = plt.subplots(
        1, n_levels, figsize=(3.0 * n_levels, 4.1), squeeze=False,
    )
    for n, ax in enumerate(axes[0]):
        density = np.abs(symmetric_gauge_state(n, m, xx, yy)) ** 2
        density /= density.max()
        ax.imshow(
            density, origin="lower", cmap="magma",
            extent=(-r_max, r_max, -r_max, r_max),
        )
        ax.set_title(rf"$n={n},\ m={m}$" + "\n" + rf"$E={n + 0.5:.1f}\,\hbar\omega_c$")
        ax.set_xlabel(r"$x/\ell_B$")
        ax.set_xticks([-r_max, 0, r_max])
        ax.set_yticks([-r_max, 0, r_max])
    axes[0][0].set_ylabel(r"$y/\ell_B$")

    fig.suptitle(
        r"Symmetric gauge densities $|\psi_{n,m}(x,y)|^2$ "
        f"(angular momentum m = {m})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, default=5,
                        help="number of Landau levels to draw (default: 5)")
    parser.add_argument("-m", type=int, default=2,
                        help="angular momentum for the symmetric-gauge panels")
    parser.add_argument("--show", action="store_true",
                        help="open the figures instead of only saving them")
    args = parser.parse_args()

    if args.levels < 1:
        parser.error("--levels must be at least 1")

    fig1 = plot_landau_gauge(args.levels)
    fig2 = plot_symmetric_gauge(args.levels, m=args.m)

    fig1.savefig("landau_levels_1d.png", dpi=150)
    fig2.savefig("landau_levels_2d.png", dpi=150)
    print("wrote landau_levels_1d.png and landau_levels_2d.png")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

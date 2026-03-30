import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import re
from pathlib import Path

def fftfreq(n):
    return np.fft.fftfreq(n)

def gaussian_plateau_filter(kx_mat, ky_mat, q_filter, sigma_fraction=0.15):
    if sigma_fraction <= 0:
        raise ValueError("sigma_fraction must be positive.")
    k = np.sqrt(kx_mat**2 + ky_mat**2)
    k_start = q_filter
    sigma = sigma_fraction * q_filter

    F = np.ones_like(k)
    tail = k > k_start
    if np.any(tail):
        normalized = (k[tail] - k_start) / sigma
        F[tail] = np.exp(-(normalized**2))
    return F


def raised_cosine_filter(kx_mat, ky_mat, q_filter, rolloff_fraction=0.66):
    """
    Match Julia-style raised-cosine low-pass:
      k_start = (1 - rolloff_fraction) * q_filter
      k_cut   = q_filter
      F=1 below k_start, smooth cosine rolloff to 0 at k_cut.
    """
    if rolloff_fraction <= 0 or rolloff_fraction >= 1:
        raise ValueError("rolloff_fraction must be in (0, 1).")
    k = np.sqrt(kx_mat**2 + ky_mat**2)
    k_cut = q_filter
    k_start = (1.0 - rolloff_fraction) * k_cut
    width = k_cut - k_start

    F = np.ones_like(k)
    F[k >= k_cut] = 0.0
    if width > 0:
        in_band = (k > k_start) & (k < k_cut)
        s = (k[in_band] - k_start) / width
        F[in_band] = 0.5 * (1.0 + np.cos(np.pi * s))
    return F

def build_k_grids(nxy, num_crit):
    L_dom = 2.0 * np.pi * num_crit
    hxy = L_dom / nxy
    kx = fftfreq(nxy) * 2.0 * np.pi / hxy
    ky = kx.copy()
    kx_mat = np.tile(kx, (nxy, 1))
    ky_mat = np.tile(ky.reshape(-1, 1), (1, nxy))
    return L_dom, hxy, kx, ky, kx_mat, ky_mat

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot instability balloons, and show effect of Fourier masks."
    )
    parser.add_argument("-f", "--filename", required=True,
                        help="Simulation folder name under inputs/ (contains params.jl).")
    parser.add_argument("--input-root", default="inputs",
                        help="Root directory containing input folders (default: inputs).")
    return parser.parse_args()

def read_params_jl(sim_name, input_root):
    params_path = Path(input_root) / sim_name / "params.jl"
    if not params_path.exists():
        raise FileNotFoundError(f"Could not find params file: {params_path}")

    values = {}
    const_re = re.compile(r"const\s+(?P<name>\w+)\s*=\s*(?P<val>.+)")
    for line in params_path.read_text().splitlines():
        m = const_re.search(line)
        if not m:
            continue
        name = m.group("name")
        raw = m.group("val").split("#", 1)[0].strip().rstrip(";")
        if "::" in raw:
            raw = raw.split("::", 1)[0].strip()
        try:
            values[name] = float(raw)
        except ValueError:
            pass
    return values

def calc_pth(qbar, wbar_r, b0, R):
    """
    """
    arg = 0.5 * np.pi * qbar**2
    s = np.sin(arg)

    pth = np.full_like(qbar, np.inf, dtype=float)
    good = s > 0
    # Avoid division by tiny numbers close to 0 -> huge threshold (still fine).
    pth[good] = (2.0 * wbar_r * qbar[good]**2) / (b0 * R * s[good])
    return pth

def calc_growth(qbar, wbar_r, p0, pth):
    """
    gbar = wbar_r qbar^2 sqrt(p0/pth - 1) for p0 > pth, else 0
    """
    g = np.zeros_like(qbar, dtype=float)
    unstable = np.isfinite(pth) & (p0 > pth)
    g[unstable] = wbar_r * qbar[unstable]**2 * np.sqrt(p0 / pth[unstable] - 1.0)
    return g


def threshold_growth_like_julia(wbar_r, b0, R, p0, points=100000):
    """
    Match threshold_growth.jl:
      q2 = range(1e-3, 11, points)
      thresh = 2*omega_r/(b0*R) * q2 / sin(pi*q2/2)
      thresh[thresh > 90*omega_r/(b0*R)] = NaN
      p0_over_thresh = (p0/thresh - 1) for valid unstable points, else 0
      g = omega_r * q2 * sqrt(p0_over_thresh)
    """
    q2 = np.linspace(1.0e-3, 11.0, int(points))
    thresh = (2.0 * wbar_r / (b0 * R)) * q2 / np.sin(0.5 * np.pi * q2)
    thresh = thresh.astype(float)

    disc = thresh > (90.0 * wbar_r / (b0 * R))
    thresh[disc] = np.nan

    valid = np.isfinite(thresh) & (thresh > 1.0e-15) & (p0 > thresh)
    p0_over_thresh = np.zeros_like(thresh)
    p0_over_thresh[valid] = p0 / thresh[valid] - 1.0
    g = wbar_r * q2 * np.sqrt(np.maximum(p0_over_thresh, 0.0))
    return q2, thresh, g

def main():
    args = parse_args()
    params = read_params_jl(args.filename, args.input_root)

    # Required grid params
    if "nxy" not in params or "num_crit" not in params:
        raise ValueError("params.jl must define both `nxy` and `num_crit`.")
    nxy = int(params["nxy"])
    num_crit = float(params["num_crit"])

    b0 = params.get("b0")
    p0 = params.get("p0")
    R = params.get("R", params.get("ref"))
    wbar_r = params.get("wbar_r", params.get("omega_r"))

    missing = []
    if b0 is None:
        missing.append("b0")
    if p0 is None:
        missing.append("p0")
    if R is None:
        missing.append("R/ref")
    if wbar_r is None:
        missing.append("wbar_r/omega_r")
    if missing:
        raise ValueError(
            "ERROR"
        )

    b0 = float(b0)
    p0 = float(p0)
    R = float(R)
    wbar_r = float(wbar_r)

    # Mask params (defaults aligned to Julia file)
    q_filter = float(params.get("q_filter", np.sqrt(2)))
    opt_sigma_frac = float(params.get("OPTICAL_GAUSS_SIGMA_FRACTION", 0.85))
    dens_sigma_frac = float(params.get("DENS_GAUSS_SIGMA_FRACTION", 4.0))
    rolloff_frac = float(params.get("OPTICAL_FILTER_ROLLOFF", 0.66))

    L_dom, hxy, kx, ky, kx_mat, ky_mat = build_k_grids(nxy, num_crit)

    # qbar grid (assumes qc = 1 in these nondimensional units)
    qbar = np.sqrt(kx_mat**2 + ky_mat**2)

    # Threshold and growth (no mask)
    pth = calc_pth(qbar, wbar_r=wbar_r, b0=b0, R=R)
    g = calc_growth(qbar, wbar_r=wbar_r, p0=p0, pth=pth)

    # Masks
    M_halfgauss = gaussian_plateau_filter(kx_mat, ky_mat, q_filter, sigma_fraction=opt_sigma_frac)
    M_rollcos = raised_cosine_filter(kx_mat, ky_mat, q_filter, rolloff_fraction=rolloff_frac)
    M_den = gaussian_plateau_filter(kx_mat, ky_mat, q_filter, sigma_fraction=dens_sigma_frac)

    # Thresholds with masks applied (for stability plots only)
    pth_halfgauss = pth / np.clip(M_halfgauss**2, 1e-300, None)
    pth_rollcos = pth / np.clip(M_rollcos**2, 1e-300, None)

    # Plot on centered Fourier axes for readability.
    kx_plot = np.fft.fftshift(kx)
    ky_plot = np.fft.fftshift(ky)
    extent = [kx_plot[0], kx_plot[-1], ky_plot[0], ky_plot[-1]]

    unstable = ((p0 > pth) & np.isfinite(pth)).astype(float)
    unstable_halfgauss = ((p0 > pth_halfgauss) & np.isfinite(pth_halfgauss)).astype(float)
    unstable_rollcos = ((p0 > pth_rollcos) & np.isfinite(pth_rollcos)).astype(float)
    unstable_plot = np.fft.fftshift(unstable)
    unstable_halfgauss_plot = np.fft.fftshift(unstable_halfgauss)
    unstable_rollcos_plot = np.fft.fftshift(unstable_rollcos)
    g_plot = np.fft.fftshift(g)
    gmax = float(np.max(g_plot))
    gmin_pos = 1e-8 * gmax if gmax > 0 else 1e-16
    g_norm = colors.LogNorm(vmin=gmin_pos, vmax=max(gmax, gmin_pos * 10.0))

    # --- Plots ---

    # Stability balloons (no mask vs masks)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    panels = [
        (unstable_plot, "No mask"),
        (unstable_halfgauss_plot, "Half-Gaussian mask"),
        (unstable_rollcos_plot, "Rolling-cosine mask"),
    ]
    for ax, (img, title) in zip(axs, panels):
        im = ax.imshow(
            img,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
    fig.colorbar(im, ax=axs, shrink=0.9, label="unstable (0/1)")
    fig.suptitle("Stability balloons from Eq. (20)")

    # 2D growth-rate map (no mask)
    plt.figure()
    plt.imshow(
        np.maximum(g_plot, gmin_pos),
        origin="lower",
        extent=extent,
        norm=g_norm,
        cmap="inferno",
        interpolation="nearest",
    )
    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$k_y$")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(r"Growth rate $\bar g(q)$ (Eq. 22) — no mask")
    plt.colorbar(label=r"$\bar g$")

    # 1D growth curve (no mask) matching threshold_growth.jl
    q2_1d, pth_1d, g_1d = threshold_growth_like_julia(wbar_r, b0, R, p0)

    # 1D threshold curve (no mask) matching threshold_growth.jl
    plt.figure()
    plt.plot(q2_1d, pth_1d, label=r"$p_{th}$ (no mask)")
    plt.axhline(p0, linestyle="--", label=r"$p_0$")
    plt.xlabel(r"$\bar q^2$")
    plt.ylabel(r"$p_{th}$")
    plt.title("Threshold curve (matches threshold_growth.jl)")
    y_max = 40.0 * wbar_r / (b0 * R)
    if not np.isfinite(y_max) or y_max <= 0:
        y_max = max(4.0 * p0, 1.0)
    plt.ylim(0.0, y_max)
    plt.xlim(float(np.min(q2_1d)), float(np.max(q2_1d)))
    plt.legend()

    # 1D growth curve (no mask) matching threshold_growth.jl
    plt.figure()
    plt.plot(q2_1d, g_1d, label=r"$\bar g$ (no mask)")
    plt.xlabel(r"$\bar q^2$")
    plt.ylabel(r"$\bar g$")
    plt.title("Growth curve (matches threshold_growth.jl)")
    g_max = float(np.nanmax(g_1d)) if np.any(np.isfinite(g_1d)) else 1.0
    plt.ylim(0.0, 1.1 * g_max if g_max > 0 else 1.0)
    plt.xlim(float(np.min(q2_1d)), float(np.max(q2_1d)))
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()

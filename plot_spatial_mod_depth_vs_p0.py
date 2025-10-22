import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import fourier_utils as ft_utils
import standard_data_utils as stand_utils


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Plot spatial modulation depth versus p0.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="filename",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. ivan_diss_params).",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        metavar="x_position",
        required=True,
        help="Spatial location to initialise the tracked peak (in simulation units).",
    )
    return parser.parse_args()


def read_input(seed_dir):
    """Read simulation metadata from the seed file."""
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int),  # plotnum
    )


def extract_float(filename):
    """Extract the floating point value from a psi filename."""
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_data(output_dir):
    """Load and sort psi files from the output directory."""
    data_files = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(data_files, key=extract_float)


def calc_spatial_mod_depth_vs_p0(sorted_files, nodes, x_index, freq_vals):
    """
    Compute spatial modulation depth and Fourier peak amplitude for every psi file.

    For each snapshot the tracked peak is followed in time, the time of its
    maximum amplitude is located, and a spatial cut at that instant is used to
    evaluate the modulation depth and corresponding Fourier amplitude.
    """
    if not sorted_files:
        raise FileNotFoundError("No psi files found in the specified output directory.")

    mod_depth_vals = []
    for file in sorted_files:
        psi_data = np.loadtxt(file)
        temporal_cut = stand_utils.find_temporal_cut_of_x_peaks(psi_data, x_index)

        peak_time_index = int(np.argmax(temporal_cut))
        spatial_cut = psi_data[peak_time_index, 1:]  # drop the time column

        print(f'The max peak occurs at t = {psi_data[peak_time_index, 0]}')

        max_val = np.max(spatial_cut)
        min_val = np.min(spatial_cut)

        denom = max_val + min_val
        mod_depth = (max_val - min_val) / denom if denom != 0 else 0.0
        mod_depth_vals.append(mod_depth)

    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)
    ft_data = ft_utils.analyse_fourier_data(
        sorted_files,
        freq_vals,
        norm_factor=1,
        is_temporal_ft=False,
        cut_index=x_index,
        time_index_strategy="tracked_peak_max",
    )
    first_mode_amp = np.array(ft_data["first_mode_ft_peaks_amp"], dtype=float)

    return np.array(p0_vals), np.array(mod_depth_vals), first_mode_amp


def create_plot(p0_vals, mod_depth_vals, ft_amp_vals, x, p0_samples, m_max_vals, p_th):
    """Create the modulation depth vs p0 scatter plot."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(
        p0_vals,
        mod_depth_vals,
        color="black",
        edgecolor="black",
        marker="x",
        linewidth=0.6,
        s=30,
        label="Simulation",
        zorder=3,
    )

    if ft_amp_vals.size:
        ax.plot(
            p0_vals,
            ft_amp_vals,
            color="tab:blue",
            marker="o",
            markersize=4,
            linewidth=1,
            label="FFT first harmonic",
            zorder=4,
        )

    if m_max_vals.size:
        ax.plot(
            p0_samples,
            m_max_vals,
            color="black",
            linewidth=1,
            label=r"$M_{max}$ analytic",
            zorder=5,
        )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, label=r"$p_{th}$", zorder=2)

    ax.set_xlabel(r"Pump strength $p_0$", fontsize=13)
    ax.set_ylabel(r"Modulation depth $m[|\psi|^2]$", fontsize=13)
    ax.set_title(rf"Spatial modulation depth at $x = {x}$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(which="major", linestyle=":", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.15)
    ax.legend(frameon=False, loc="lower right")

    if p0_vals.size:
        ax.set_xlim(p0_vals.min() * 0.98, p0_vals.max() * 1.02)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def main():
    """Main entry point."""
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_dir = os.path.join(input_dir, "seed.in")

    (
        nodes,
        maxt,
        ht,
        width_psi,
        p0,
        Delta,
        gambar,
        b0,
        num_crit,
        R,
        gbar,
        v0,
        plotnum,
    ) = read_input(seed_dir)

    x = float(args.xpos)
    frac = (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
    x_index = int(np.clip(frac, 0, nodes - 1))

    x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)
    dx = x_vals[1] - x_vals[0] if nodes > 1 else 1.0
    k_vals = np.fft.fftfreq(len(x_vals), dx)

    sorted_files = load_data(output_dir)
    p0_vals, mod_depth_vals, ft_amp_vals = calc_spatial_mod_depth_vs_p0(sorted_files, nodes, x_index, k_vals)

    p_th = (2 * gambar) / (b0 * R)

    if p0_vals.size:
        p0_min, p0_max = p0_vals.min(), p0_vals.max()
        if p0_min == p0_max:
            p0_samples = np.array([p0_min])
        else:
            p0_samples = np.linspace(p0_min, p0_max, max(len(p0_vals), 200))
        with np.errstate(divide="ignore", invalid="ignore"):
            m_max_vals = np.sqrt(2 * p_th * np.maximum(p0_samples - p_th, 0) / (p0_samples**2))
            m_max_vals = np.where(p0_samples > 0, m_max_vals, np.nan)
    else:
        p0_samples = np.array([])
        m_max_vals = np.array([])

    create_plot(p0_vals, mod_depth_vals, ft_amp_vals, x, p0_samples, m_max_vals, p_th)
    plt.show()


if __name__ == "__main__":
    main()

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

import standard_data_utils as stand_utils


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot the interval between the last two peaks versus p0.")
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
        help="Spatial location to analyse (in the same units used by the simulation).",
    )
    return parser.parse_args()


def read_input(seed_dir):
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int),  # plotnum
    )


def extract_float(filename):
    match = re.search(r"psi\d+_([\d.e-]+)", filename)
    if match:
        float_str = match.group(1).rstrip(".")
        return float(float_str)
    return 0.0


def load_data(output_dir):
    data_files = glob.glob(os.path.join(output_dir, "psi*"))
    return sorted(data_files, key=extract_float)


def compute_last_peak_intervals(sorted_files, p0_vals, x_index):
    intervals = []
    valid_p0 = []

    for file, p0 in zip(sorted_files, p0_vals):
        data = np.loadtxt(file)
        if data.ndim != 2 or data.shape[1] < 2:
            continue

        times = data[:, 0]
        spatial_data = data[:, 1:]
        spatial_idx = int(np.clip(x_index, 0, spatial_data.shape[1] - 1))
        trace = spatial_data[:, spatial_idx]

        amp_range = np.max(trace) - np.min(trace)
        if amp_range <= 0:
            continue

        prominence = amp_range / stand_utils.PEAK_PROMINENCE_FACTOR
        prominence = prominence if prominence > 0 else None

        peak_kwargs = {"prominence": prominence} if prominence is not None else {}
        peaks, _ = find_peaks(trace, **peak_kwargs)

        if peaks.size < 2:
            continue

        last_two_indices = peaks[-2:]
        interval = times[last_two_indices[1]] - times[last_two_indices[0]]
        if interval <= 0:
            continue

        valid_p0.append(p0)
        intervals.append(interval)

    return np.array(valid_p0, dtype=float), np.array(intervals, dtype=float)


def create_plot(p0_vals, intervals, p_th, x):
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        p0_vals,
        intervals,
        color="C0",
        linewidth=1.5,
        zorder=2,
        label="Interval between last peaks",
    )
    ax.scatter(
        p0_vals,
        intervals,
        color="black",
        marker="x",
        s=55,
        linewidths=1.1,
        zorder=3,
    )

    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, label=r"$p_{th}$", zorder=1)

    ax.set_xlabel(r"Pump strength $p_0$", fontsize=13)
    ax.set_ylabel(r"Interval $\Delta (\Gamma t)$", fontsize=13)
    ax.set_title(rf"Time interval between final peaks at $x = {x}$", fontsize=14)
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax.grid(which="major", linestyle=":", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.15)
    ax.legend(frameon=False, loc="best")

    if p0_vals.size:
        ax.set_xlim(p0_vals.min() * 0.9, p0_vals.max() * 1.02)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig, ax


def main():
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

    sorted_files = load_data(output_dir)
    all_p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    p_th = (2 * gambar) / (b0 * R)

    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(all_p0_vals, sorted_files, p_th)

    p0_vals, intervals = compute_last_peak_intervals(sorted_files_above_th, p0_above_th_vals, x_index)

    if p0_vals.size == 0:
        raise RuntimeError("No valid peak intervals were found for the provided data.")

    create_plot(p0_vals, intervals, p_th, x)
    plt.show()


if __name__ == "__main__":
    main()

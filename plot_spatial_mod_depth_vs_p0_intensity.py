import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

import standard_data_utils as stand_utils

S_FILE_PATTERN = re.compile(r"^s\d+_[\d.eE+-]+\.out$")

PUBLICATION_FONTS = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot optical intensity first-harmonic amplitude versus p0."
    )
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
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int),  # plotnum
    )


def extract_float(filename):
    match = re.search(r"s\d+_([\d.eE+-]+)\.out$", os.path.basename(filename))
    if not match:
        raise ValueError(f"Unable to parse p0 from filename: {filename}")
    return float(match.group(1).rstrip("."))


def load_data(output_dir):
    candidates = glob.glob(os.path.join(output_dir, "s*"))
    data_files = [
        path
        for path in candidates
        if S_FILE_PATTERN.match(os.path.basename(path))
    ]
    if not data_files:
        raise FileNotFoundError(
            "No intensity files matching sN_<p0>.out found in the specified output directory."
        )
    return sorted(data_files, key=extract_float)


def calc_first_harmonic_vs_p0_intensity(sorted_files):
    if not sorted_files:
        raise FileNotFoundError("No s files found in the specified output directory.")

    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)
    first_mode_amp_max_over_time = []

    for file in sorted_files:
        s_data = np.loadtxt(file)
        if s_data.ndim != 2 or s_data.shape[1] < 3:
            first_mode_amp_max_over_time.append(0.0)
            continue

        spatial_data = s_data[:, 1:]
        max_amp = 0.0
        for row in spatial_data:
            fft_vals = np.abs(np.fft.fft(row, norm="forward")[: len(row) // 2])
            if fft_vals.size <= 1:
                continue
            # Exclude the DC component and track the strongest harmonic at this time.
            row_max = float(np.max(fft_vals[1:]))
            if row_max > max_amp:
                max_amp = row_max

        first_mode_amp_max_over_time.append(max_amp)

    return np.array(p0_vals), np.array(first_mode_amp_max_over_time, dtype=float)


def create_plot(p0_vals, ft_amp_vals, x, p_th):
    fig, ax = plt.subplots(figsize=(9, 6))

    # shift_p0 = [p0 - 2e-10 if p0 > 2e-10 else 0 for p0 in p0_vals]
    # log_p0 = np.log(shift_p0)

    # if ft_amp_vals.size:
    #     ax.plot(
    #         log_p0,
    #         np.log(ft_amp_vals),
    #         color="tab:blue",
    #         marker="o",
    #         markersize=6,
    #         linewidth=1.5,
    #         label="FFT first harmonic",
    #         zorder=4
    #     )

    if ft_amp_vals.size:
        ax.plot(
            p0_vals,
            ft_amp_vals,
            color="tab:blue",
            marker="o",
            markersize=6,
            linewidth=1.5,
            label="FFT first harmonic",
            zorder=4
        )
    
    ax.axvline(p_th, color="k", linestyle="--", linewidth=1.2, label=r"$p_{th}$", zorder=2)

    ax.set_xlabel(r"Pump strength $p_0$", fontsize=16)
    ax.set_ylabel("First-harmonic amplitude", fontsize=16)
    ax.set_title(rf"Optical intensity first harmonic at $x = {x}$", fontsize=18)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=14)
    ax.grid(which="major", linestyle=":", alpha=0.4)
    ax.grid(which="minor", linestyle=":", alpha=0.15)
    ax.legend(frameon=False, loc="lower right", fontsize=14)

    if p0_vals.size:
        ax.set_xlim(p0_vals.min() * 0.98, p0_vals.max() * 1.02)

    fig.tight_layout()
    return fig, ax


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"
    plt.rcParams.update(PUBLICATION_FONTS)

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
    sorted_files = load_data(output_dir)
    p0_vals, ft_amp_vals = calc_first_harmonic_vs_p0_intensity(sorted_files)

    p_th = (2 * gambar) / (b0 * R)
    create_plot(p0_vals, ft_amp_vals, x, p_th)
    plt.show()


if __name__ == "__main__":
    main()

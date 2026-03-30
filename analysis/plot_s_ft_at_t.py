from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["ps.usedistiller"] = "xpdf"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot spatial Fourier transform of S(x, t) at a chosen time."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of patt1d input/output directory pair.",
    )
    parser.add_argument(
        "-i",
        "--frame_index",
        type=int,
        required=True,
        help="Frame/file index N for sN_<p0>.out.",
    )
    parser.add_argument(
        "-t",
        "--time",
        type=float,
        required=True,
        help="Target time value (Gamma t); nearest available row is used.",
    )
    return parser.parse_args()


def read_input(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    nodes = int(data[0])
    num_crit = float(data[8])
    return nodes, num_crit


def load_s_data(output_dir, frame_index):
    pattern = os.path.join(output_dir, f"s{frame_index}_*.out")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files match {pattern}; expected exactly one.")
    return matches[0], np.loadtxt(matches[0])


def main():
    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_path = os.path.join(input_dir, "seed.in")

    nodes, num_crit = read_input(seed_path)
    s_path, s_data = load_s_data(output_dir, args.frame_index)
    print(f"Loaded S file: {s_path}")

    if s_data.ndim != 2 or s_data.shape[1] < 2:
        raise ValueError("Loaded S data has invalid shape.")

    t_vals = s_data[:, 0]
    spatial_vals = s_data[:, 1:]

    if spatial_vals.shape[1] != nodes:
        raise ValueError(
            f"S data spatial width ({spatial_vals.shape[1]}) does not match nodes ({nodes})."
        )

    t_index = int(np.argmin(np.abs(t_vals - args.time)))
    t_actual = float(t_vals[t_index])
    print(f"Requested time: {args.time:.6e}, using nearest time: {t_actual:.6e} (row {t_index}).")

    s_at_t = spatial_vals[t_index, :]

    x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes, endpoint=False)
    dx = x_vals[1] - x_vals[0] if nodes > 1 else 1.0
    k_vals = np.fft.fftfreq(nodes, d=dx)
    fft_vals = np.abs(np.fft.fft(s_at_t, norm="forward"))

    pos_mask = k_vals >= 0
    k_pos = k_vals[pos_mask]
    fft_pos = fft_vals[pos_mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(k_pos, fft_pos, linewidth=1.5)
    ax.set_xlabel(r"$\Gamma k$")
    ax.set_ylabel(r"$|\mathcal{F}[s(x)]|$")
    ax.set_title(
        rf"Spatial FT of $s(x)$ at $\Gamma t={t_actual:.3e}$ (frame {args.frame_index})"
    )
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

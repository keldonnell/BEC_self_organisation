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
import re

import matplotlib.pyplot as plt
import numpy as np


S_FILE_PATTERN = re.compile(r"^s(\d+)_([\d.eE+-]+)\.out$")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "For each intensity file sN_<p0>.out: compute first-harmonic amplitude "
            "vs time, find its maximum, print the corresponding time, and plot "
            "max first-harmonic amplitude vs p0."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of patt1d input/output directory pair (e.g. ivan_diss_params).",
    )
    return parser.parse_args()


def parse_s_filename(path):
    base = os.path.basename(path)
    match = S_FILE_PATTERN.match(base)
    if not match:
        raise ValueError(f"Filename does not match sN_<p0>.out pattern: {base}")
    frame_index = int(match.group(1))
    p0 = float(match.group(2).rstrip("."))
    return frame_index, p0


def load_sorted_s_files(output_dir):
    candidates = glob.glob(os.path.join(output_dir, "s*.out"))
    parsed = []
    for path in candidates:
        base = os.path.basename(path)
        if not S_FILE_PATTERN.match(base):
            continue
        frame_index, p0 = parse_s_filename(path)
        parsed.append((p0, frame_index, path))

    if not parsed:
        raise FileNotFoundError(
            f"No files matching sN_<p0>.out found in {output_dir}."
        )

    parsed.sort(key=lambda x: (x[0], x[1]))
    return parsed


def first_non_dc_local_peak_amplitude(fft_vals):
    if fft_vals.size <= 1:
        return 0.0

    non_dc = fft_vals[1:]
    if non_dc.size == 0:
        return 0.0

    if non_dc.size >= 3:
        local_max_mask = (non_dc[1:-1] > non_dc[:-2]) & (non_dc[1:-1] >= non_dc[2:])
        local_max_indices = np.where(local_max_mask)[0] + 1
        if local_max_indices.size:
            return float(non_dc[local_max_indices[0]])

    return float(np.max(non_dc))


def compute_first_harmonic_amp_trace(s_data):
    if s_data.ndim != 2 or s_data.shape[1] < 3:
        raise ValueError("Intensity data must be 2D with time + at least 2 spatial points.")

    times = s_data[:, 0]
    spatial_data = s_data[:, 1:]
    half_nodes = spatial_data.shape[1] // 2

    amp_trace = np.zeros(spatial_data.shape[0], dtype=float)
    for i, spatial_cut in enumerate(spatial_data):
        row = spatial_cut - np.mean(spatial_cut)
        fft_vals = np.abs(np.fft.fft(row, norm="forward")[:half_nodes])
        amp_trace[i] = first_non_dc_local_peak_amplitude(fft_vals)

    return times, amp_trace


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"
    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    parsed_files = load_sorted_s_files(output_dir)

    p0_vals = []
    max_amp_vals = []

    print("# index p0 t_at_max first_harmonic_amp_max")
    for p0, frame_index, path in parsed_files:
        s_data = np.loadtxt(path)
        times, amp_trace = compute_first_harmonic_amp_trace(s_data)
        max_idx = int(np.argmax(amp_trace))
        t_at_max = float(times[max_idx])
        amp_max = float(amp_trace[max_idx])

        p0_vals.append(p0)
        max_amp_vals.append(amp_max)
        print(f"{frame_index:4d}  {p0:.8e}  {t_at_max:.8e}  {amp_max:.8e}")

    p0_vals = np.array(p0_vals, dtype=float)
    max_amp_vals = np.array(max_amp_vals, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p0_vals, max_amp_vals, marker="o", linewidth=1.5, markersize=5)
    ax.set_xlabel(r"Pump strength $p_0$")
    ax.set_ylabel("Max 1st-harmonic amplitude")
    ax.set_title("Max intensity 1st-harmonic amplitude vs p0")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()

    out_base = "intensity_first_harmonic_max_vs_p0"
    plt.savefig(f"{out_base}.pdf", dpi=300)
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot the first-harmonic Fourier amplitude of intensity over time."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. ivan_diss_params).",
    )
    parser.add_argument(
        "-i",
        "--frame_index",
        type=int,
        required=True,
        help="Index of the intensity file sN_<p0>.out to analyze.",
    )
    return parser.parse_args()


def read_input(seed_dir):
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "num_crit": float(data[8]),
        "plotnum": int(data[12]),
    }


def load_s_data(output_dir, frame_index):
    s_paths = glob.glob(os.path.join(output_dir, f"s{frame_index}_*.out"))
    if not s_paths:
        raise FileNotFoundError(
            f"No intensity s file found for index {frame_index} in {output_dir}"
        )
    if len(s_paths) > 1:
        raise RuntimeError(
            f"Multiple intensity files found for index {frame_index} in {output_dir}"
        )
    return np.loadtxt(s_paths[0])


def build_spatial_freqs(num_nodes, num_crit):
    domain_len = 2 * np.pi * num_crit
    dx = domain_len / num_nodes
    freq_vals = np.fft.fftfreq(num_nodes, d=dx) * 2 * np.pi
    return np.abs(freq_vals[: num_nodes // 2])


def first_non_dc_local_peak_amplitude(fft_vals):
    """
    Return the first non-DC local-maximum amplitude in a 1D FFT magnitude array.

    If no strict local maximum exists, fall back to the strongest non-DC bin.
    """
    if fft_vals.size <= 1:
        return 0.0

    non_dc = fft_vals[1:]
    if non_dc.size == 0:
        return 0.0

    # Strict local maxima among non-DC bins.
    if non_dc.size >= 3:
        local_max_mask = (non_dc[1:-1] > non_dc[:-2]) & (non_dc[1:-1] >= non_dc[2:])
        local_max_indices = np.where(local_max_mask)[0] + 1
        if local_max_indices.size:
            return float(non_dc[local_max_indices[0]])

    # Fallback when spectrum is monotonic/noisy and no local-peak test succeeds.
    return float(np.max(non_dc))


def compute_first_harmonic_amplitude_over_time(s_data, freq_vals):
    spatial_data = s_data[:, 1:]
    num_nodes = spatial_data.shape[1]
    half_nodes = num_nodes // 2

    amp_trace = np.zeros(spatial_data.shape[0], dtype=float)
    for i, spatial_cut in enumerate(spatial_data):
        # Remove DC bias from the row before FFT to improve numerical robustness
        # when the harmonic amplitude is much smaller than the mean intensity.
        row = spatial_cut - np.mean(spatial_cut)
        fft_vals = np.abs(np.fft.fft(row, norm="forward")[:half_nodes])
        amp_trace[i] = first_non_dc_local_peak_amplitude(fft_vals)

    return amp_trace


def main():
    plt.rcParams["ps.usedistiller"] = "xpdf"

    args = parse_arguments()
    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_dir = os.path.join(input_dir, "seed.in")

    params = read_input(seed_dir)
    s_data = load_s_data(output_dir, args.frame_index)

    num_nodes = s_data.shape[1] - 1
    freq_vals = build_spatial_freqs(num_nodes, params["num_crit"])

    times = s_data[:, 0]
    amp_trace = compute_first_harmonic_amplitude_over_time(s_data, freq_vals)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, amp_trace, color="tab:blue", linewidth=1.5)
    ax.set_xlabel(r"Time $\Gamma t$", fontsize=14)
    ax.set_ylabel("1st harmonic amplitude", fontsize=14)
    ax.set_title(
        f"Intensity 1st harmonic amplitude vs time (index {args.frame_index})",
        fontsize=14,
    )
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out_base = f"intensity_first_harmonic_amp_vs_time_idx{args.frame_index}"
    plt.savefig(f"{out_base}.pdf", dpi=300)
    plt.savefig(f"{out_base}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()

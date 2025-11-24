import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from analytic_predictors import analytic_delay_time, pump_threshold
import fourier_utils as ft_utils
import standard_data_utils as stand_utils

FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?e[-+]\d+)", re.IGNORECASE)


def non_negative_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer.")
    return ivalue


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot analytic omega/Gamma versus pump strength.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="name",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. high_res).",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        metavar="x_position",
        default=0.0,
        type=float,
        help="Spatial position (in simulation units) to track when extracting observed peaks.",
    )
    parser.add_argument(
        "--s-diagnostic-index",
        metavar="index",
        type=non_negative_int,
        default=None,
        help=(
            "If provided, plots the averaged S(x) profile and its FFT magnitude for the given index in the"
            " sorted S-file list (0-based)."
        ),
    )
    return parser.parse_args()


def read_seed_metadata(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "omega_r": data[6],
        "gamma_bar": data[6],
        "b0": data[7],
        "num_crit": data[8],
        "R": data[9],
        "seed": data[13],
    }


def compute_x_index(x_pos, nodes, num_crit):
    """
    Convert a spatial position (in simulation units) into the column index used for traces.
    """
    if nodes <= 1:
        return 1

    if num_crit <= 0:
        return max(1, min(nodes // 2, nodes))

    domain = 2 * np.pi * num_crit
    frac = (x_pos + np.pi * num_crit) / domain
    frac = np.clip(frac, 0.0, 1.0)
    column = int(np.clip(frac * nodes, 0, nodes))
    return max(1, min(column, nodes))


def _collect_field_files(output_dir, prefix):
    pattern = os.path.join(output_dir, f"{prefix}*.out")
    files = glob.glob(pattern)
    filtered = [path for path in files if FLOAT_PATTERN.search(os.path.basename(path))]
    if not filtered:
        raise FileNotFoundError(f"No {prefix}*.out files with embedded p0 values found in {output_dir}")
    return sorted(filtered, key=lambda f: float(FLOAT_PATTERN.search(f).group(1)))


def collect_psi_files(output_dir):
    return _collect_field_files(output_dir, "psi")


def collect_s_files(output_dir):
    return _collect_field_files(output_dir, "s")


def extract_peak_profile(psi_path, nodes):
    data = np.loadtxt(psi_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{psi_path} does not look like a psi output table.")

    centre_idx = max(1, min(nodes // 2, data.shape[1] - 1))
    time_index = int(np.argmax(data[:, centre_idx]))
    return data[time_index, 1:]  # discard the time column


def compute_analytic_omega_gamma(psi_files, params):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    if p0_vals.size == 0:
        raise ValueError("Unable to extract p0 values from psi filenames.")

    omega_gamma_vals = []
    nqc_vals = []

    for file_path, p0 in zip(psi_files, p0_vals):
        psi_snapshot = extract_peak_profile(file_path, params["nodes"])
        n_qc = ft_utils.calculate_nqc_amplitude(psi_snapshot, params["num_crit"])
        print(f'n_qc = {np.abs(n_qc)}')
        # Analytic omega/Gamma definition: sqrt(b0 * R * p0 * |n^{q_c}| * omega_r)
        omega_gamma = np.sqrt(params["b0"] * params["R"] * p0 * n_qc * params["omega_r"])
        nqc_vals.append(n_qc)
        omega_gamma_vals.append(omega_gamma)

    return p0_vals, np.array(omega_gamma_vals), np.array(nqc_vals)


def plot_results(
    p0_vals,
    omega_vals,
    observed_p0,
    observed_omega,
    folder_name,
    alt_curve_avg=None,
):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p0_vals, omega_vals, marker="o", linewidth=1.2, label="Analytic")

    if observed_p0.size:
        ax.plot(
            observed_p0,
            observed_omega,
            marker="x",
            linestyle="--",
            linewidth=1.2,
            label="Observed",
        )
    else:
        print("No observed omega/Gamma values passed to plotting routine.")

    if alt_curve_avg is not None and alt_curve_avg[0].size:
        ax.plot(
            alt_curve_avg[0],
            alt_curve_avg[1],
            marker="d",
            linestyle="-",
            linewidth=1.2,
            label=r"$|<S(x, t)>_{t_0 -> T}|_{q_c}$",
        )

    ax.set_xlabel(r"$p_0$")
    ax.set_ylabel(r"$\omega / \Gamma$")
    ax.set_title(rf"Analytic vs observed $\omega / \Gamma$ for {folder_name}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    plt.show()
    return fig, ax


def compute_first_fourier_amplitude(spatial_profile, num_crit):
    """
    Compute the modulus of the k=1 spatial Fourier component for a real profile.
    Matches the diagnostic FFT scaling (raw |FFT|, no normalization).
    """
    profile = np.asarray(spatial_profile, dtype=float)
    if profile.ndim != 1 or profile.size == 0:
        return np.nan

    N = profile.size
    L = 2 * np.pi * num_crit if num_crit > 0 else float(N)
    dx = L / max(N, 1)

    fft_vals = np.fft.fft(profile)
    k_vals = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    target_k = 1.0
    idx = np.argmin(np.abs(k_vals - target_k))
    return np.abs(fft_vals[idx])


def compute_analytic_curve_from_amplitude(p0_vals, amp_vals, params):
    """
    Substitute alternative amplitudes into the analytic omega/Gamma formula.
    """
    if p0_vals.size == 0 or amp_vals.size == 0:
        return np.array([]), np.array([])

    p0_vals = np.asarray(p0_vals)
    amp_vals = np.asarray(amp_vals)
    mask = np.isfinite(p0_vals) & np.isfinite(amp_vals) & (amp_vals >= 0)
    if not np.any(mask):
        return np.array([]), np.array([])

    p0_sel = p0_vals[mask]
    amp_sel = amp_vals[mask]
    factor = params["b0"] * params["R"] * params["omega_r"]
    omega_vals = np.sqrt(factor * p0_sel) * amp_sel
    omega_vals = amp_sel #REMOVE THIS LATER. THIS IS JUST TO TEST
    return p0_sel, omega_vals


def _compute_fft_profile(profile, num_crit):
    """
    Return positive-k FFT magnitudes and wavenumbers for a real spatial profile.
    """
    profile = np.asarray(profile, dtype=float)
    if profile.ndim != 1 or profile.size == 0:
        return np.array([]), np.array([])

    N = profile.size
    L = 2 * np.pi * num_crit if num_crit > 0 else float(N)
    dx = L / max(N, 1)
    fft_vals = np.fft.fft(profile)
    k_vals = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    pos_mask = k_vals >= 0
    return k_vals[pos_mask], np.abs(fft_vals[pos_mask])


def plot_sqrt_average_diagnostics(mean_profile, sqrt_of_mean, mean_of_sqrt, num_crit, p0, diag_index):
    """
    Plot <S>, sqrt(<S>), <sqrt(S)> and their FFT magnitudes for inspection.
    """
    profiles = [
        (r"$\langle S \rangle$", np.asarray(mean_profile, dtype=float)),
        (r"$\sqrt{\langle S \rangle}$", np.asarray(sqrt_of_mean, dtype=float)),
        (r"$\langle \sqrt{S} \rangle$", np.asarray(mean_of_sqrt, dtype=float)),
    ]
    if any(p.ndim != 1 or p.size == 0 for _, p in profiles):
        return

    N = profiles[0][1].size
    L = 2 * np.pi * num_crit if num_crit > 0 else float(N)
    x_coords = np.linspace(-L / 2, L / 2, N, endpoint=False)

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    for col, (label, profile) in enumerate(profiles):
        axes[0, col].plot(x_coords, profile)
        axes[0, col].set_title(f"{label} profile")
        axes[0, col].set_xlabel("x")
        axes[0, col].set_ylabel(label)

        k_vals, fft_mag = _compute_fft_profile(profile, num_crit)
        axes[1, col].plot(k_vals, fft_mag)
        axes[1, col].set_title(f"FFT({label}) magnitude")
        axes[1, col].set_xlabel("k")
        axes[1, col].set_ylabel(r"$|S_k|$")

    fig.suptitle(f"S diagnostics for p0={p0:.3e} (index {diag_index})")
    fig.tight_layout()
    plt.show()


def compute_observed_omega_gamma(psi_files, params, x_index):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    observed_p0 = []
    observed_omega = []

    for file_path, p0 in zip(psi_files, p0_vals):
        data = np.loadtxt(file_path)

        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping p0={p0:.3e} ({os.path.basename(file_path)}): data shape invalid.")
            continue

        times = data[:, 0]
        spatial_idx = int(np.clip(x_index, 1, data.shape[1] - 1))
        trace = data[:, spatial_idx]

        amp_range = np.max(trace) - np.min(trace)
        if amp_range <= 0:
            print(f"Skipping p0={p0:.3e}: trace is flat, cannot detect peaks.")
            continue

        prominence = amp_range / stand_utils.PEAK_PROMINENCE_FACTOR
        peak_kwargs = {"prominence": prominence} if prominence > 0 else {}
        peaks, _ = find_peaks(trace, **peak_kwargs)

        if peaks.size < 2:
            print(f"Skipping p0={p0:.3e}: fewer than two peaks detected.")
            continue

        first_two = peaks[:2]
        gamma_t = times[first_two[1]] - times[first_two[0]]
        if gamma_t <= 0:
            print(f"Skipping p0={p0:.3e}: invalid peak spacing (gamma_t={gamma_t}).")
            continue

        omega_gamma = (2 * np.pi) / gamma_t
        observed_p0.append(p0)
        observed_omega.append(omega_gamma)

    return np.array(observed_p0), np.array(observed_omega)


def compute_s_potential_statistics(s_files, params, diag_index=None):
    p0_vals = stand_utils.find_p0_vals_from_filenames(s_files)
    if p0_vals.size == 0:
        raise ValueError("Unable to extract p0 values from s filenames.")

    gamma_bar = params["gamma_bar"]
    b0 = params["b0"]
    reflectivity = params["R"]
    seed = params["seed"]
    num_crit = params["num_crit"]
    p_th = pump_threshold(gamma_bar, b0, reflectivity)
    t0_vals = analytic_delay_time(p0_vals, p_th, gamma_bar, seed)

    valid_p0 = []
    fft_amp_vals = []

    diag_plotted = False

    for idx, (s_path, p0, t0) in enumerate(zip(s_files, p0_vals, t0_vals)):
        if not np.isfinite(t0):
            print(f"Skipping p0={p0:.3e}: analytic t0 is not finite.")
            continue

        data = np.loadtxt(s_path)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping p0={p0:.3e}: {os.path.basename(s_path)} has invalid dimensions.")
            continue

        times = data[:, 0]
        after_t0 = times >= t0
        if not np.any(after_t0):
            print(f"Skipping p0={p0:.3e}: no S samples beyond analytic t0={t0:.3e}.")
            continue

        s_values = data[after_t0, 1:]
        if s_values.size == 0:
            print(f"Skipping p0={p0:.3e}: no spatial data beyond t0.")
            continue

        time_avg_profile = np.mean(s_values, axis=0)
        if diag_index is not None and idx == diag_index:
            sqrt_of_mean_profile = np.sqrt(np.abs(time_avg_profile))
            mean_of_sqrt_profile = np.mean(np.sqrt(np.abs(s_values)), axis=0)
            plot_sqrt_average_diagnostics(time_avg_profile, sqrt_of_mean_profile, mean_of_sqrt_profile, num_crit, p0, idx)
            diag_plotted = True
        amp = compute_first_fourier_amplitude(time_avg_profile, num_crit)
        if not np.isfinite(amp):
            print(f"Skipping p0={p0:.3e}: unable to compute FFT amplitude after averaging.")
            continue

        valid_p0.append(p0)
        fft_amp_vals.append(amp)

    if diag_index is not None and not diag_plotted:
        print(f"Diagnostic index {diag_index} was not plotted (out of range or filtered).")

    return np.array(valid_p0), np.array(fft_amp_vals)


def main():
    args = parse_arguments()

    output_dir = os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_path = os.path.join(input_dir, "seed.in")

    params = read_seed_metadata(seed_path)
    psi_files = collect_psi_files(output_dir)
    try:
        s_files = collect_s_files(output_dir)
    except FileNotFoundError:
        print(f"No S output files found in {output_dir}; skipping S potential averages.")
        s_p0_vals = np.array([])
        fft_amp_vals = np.array([])
    else:
        s_p0_vals, fft_amp_vals = compute_s_potential_statistics(
            s_files, params, diag_index=args.s_diagnostic_index
        )

    alt_curve_avg = (
        compute_analytic_curve_from_amplitude(s_p0_vals, fft_amp_vals, params) if s_p0_vals.size else None
    )

    x_index = compute_x_index(float(args.xpos), params["nodes"], params["num_crit"])

    p0_vals, omega_vals, _ = compute_analytic_omega_gamma(psi_files, params)
    observed_p0, observed_omega = compute_observed_omega_gamma(psi_files, params, x_index)

    plot_results(
        p0_vals,
        omega_vals,
        observed_p0,
        observed_omega,
        args.filename,
        alt_curve_avg=alt_curve_avg,
    )


if __name__ == "__main__":
    main()

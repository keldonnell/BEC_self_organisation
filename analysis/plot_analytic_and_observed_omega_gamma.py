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
from scipy.signal import find_peaks

from analytic_predictors import analytic_delay_time, pump_threshold
import fourier_utils as ft_utils
import standard_data_utils as stand_utils

FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?e[-+]\d+)", re.IGNORECASE)
PUBLICATION_FONTS = {
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
}


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
    parser.add_argument(
        "--s-avg-index",
        metavar="index",
        type=non_negative_int,
        default=None,
        help=(
            "If provided, plots the time-averaged S(x) profile versus x for the given index in the sorted"
            " S-file list (0-based)."
        ),
    )
    return parser.parse_args()


def read_seed_metadata(seed_path):
    data = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "omega_r": data[6],
        "Delta": data[5],
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


def extract_max_nqc_amplitude(psi_path, num_crit):
    """
    Compute the maximum |n^{q_c}| across all time frames in a psi output file.
    """
    data = np.loadtxt(psi_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{psi_path} does not look like a psi output table.")

    psi_values = data[:, 1:]
    if psi_values.size == 0:
        return np.nan

    max_amp = -np.inf
    for row in psi_values:
        amp = ft_utils.calculate_nqc_amplitude(row, num_crit)
        if amp > max_amp:
            max_amp = amp
    return max_amp if np.isfinite(max_amp) else np.nan


def compute_max_nqc_omega_gamma(psi_files, params):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    if p0_vals.size == 0:
        raise ValueError("Unable to extract p0 values from psi filenames.")

    omega_gamma_vals = []
    nqc_vals = []

    for file_path, p0 in zip(psi_files, p0_vals):
        n_qc = extract_max_nqc_amplitude(file_path, params["num_crit"])
        print(f"n_qc (max over time) = {np.abs(n_qc)}")
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
    alt_curve_avg=None,
    delta_curve=None,
    density_delta_curve=None,
    mmax_curve=None,
):
    fig, ax = plt.subplots()
    if p0_vals.size > 1:
        p0_dense = np.linspace(np.min(p0_vals), np.max(p0_vals), 500)
        omega_dense = np.interp(p0_dense, p0_vals, omega_vals)
        ax.plot(
            p0_dense,
            omega_dense,
            color="black",
            linewidth=1,
            label=r"$\sqrt{b_0 R p_0 \max_t(|n|_{q_c}) (\omega_r / \Gamma)}$",
        )
    else:
        ax.plot(
            p0_vals,
            omega_vals,
            color="black",
            linewidth=1,
            label=r"$\sqrt{b_0 R p_0 \max_t(|n|_{q_c}) (\omega_r / \Gamma)}$",
        )

    if observed_p0.size:
        ax.plot(
            observed_p0,
            observed_omega,
            color="tab:red",
            marker="x",
            linestyle="-",
            markersize=5,
            linewidth=1,
            label="Numerically observed",
        )
    else:
        print("No observed omega/Gamma values passed to plotting routine.")

    # if alt_curve_avg is not None and alt_curve_avg[0].size:
    #     ax.plot(
    #         alt_curve_avg[0],
    #         alt_curve_avg[1],
    #         marker="d",
    #         linestyle="-",
    #         linewidth=1.2,
    #         label=r"$|<S(x, t)>_{t_0 -> T}|_{q_c}$",
    #     )

    if delta_curve is not None and delta_curve[0].size:
        ax.plot(
            delta_curve[0],
            delta_curve[1],
            color="tab:blue",
            marker=".",
            linestyle="-",
            markersize=4,
            linewidth=1,
            label=r"$\sqrt{(\omega_r/\Gamma)\,\Delta\,|\langle S \rangle_t|_{q_c}}$",
        )

    if density_delta_curve is not None and density_delta_curve[0].size:
        ax.plot(
            density_delta_curve[0],
            density_delta_curve[1],
            color="#E68600",
            marker="s",
            linestyle="-",
            markersize=4,
            linewidth=1,
            label=r"$\sqrt{b_0 R p_0 |\langle n \rangle_t|_{q_c} (\omega_r/\Gamma)}$",
        )

    if mmax_curve is not None and mmax_curve[0].size:
        ax.plot(
            mmax_curve[0],
            mmax_curve[1],
            color="tab:green",
            linestyle="-",
            linewidth=1.2,
            label=r"$\sqrt{b_0 R p_0 M(t_0) (\omega_r/\Gamma)}$",
        )

    ax.set_xlabel(r"Pump strength $p_0$")
    ax.set_ylabel(r"$\omega / \Gamma$")
    ax.legend(loc="upper left", frameon=False)
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

    fft_vals = np.fft.fft(profile, norm="forward")
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
    #omega_vals = amp_sel #REMOVE THIS LATER. THIS IS JUST TO TEST
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


def compute_mmax_analytic_curve(p0_vals, params):
    """
    Compute the analytic M_max curve and map it through the omega/Gamma formula.
    Uses the same M_max expression as plot_spatial_mod_depth_vs_p0.py.
    """
    p0_vals = np.asarray(p0_vals, dtype=float)
    if p0_vals.size == 0:
        return np.array([]), np.array([])

    p0_min, p0_max = np.nanmin(p0_vals), np.nanmax(p0_vals)
    if not np.isfinite(p0_min) or not np.isfinite(p0_max):
        return np.array([]), np.array([])

    if p0_min == p0_max:
        p0_samples = np.array([p0_min])
    else:
        p0_samples = np.linspace(p0_min, p0_max, max(len(p0_vals), 500))

    gamma_bar = params.get("gamma_bar", np.nan)
    b0 = params.get("b0", np.nan)
    reflectivity = params.get("R", np.nan)
    omega_r = params.get("omega_r", np.nan)
    if not np.isfinite(gamma_bar) or not np.isfinite(b0) or not np.isfinite(reflectivity):
        return np.array([]), np.array([])

    p_th = (2 * gamma_bar) / (b0 * reflectivity)
    with np.errstate(divide="ignore", invalid="ignore"):
        m_max_vals = np.sqrt(2 * p_th * np.maximum(p0_samples - p_th, 0) / (p0_samples**2))
        m_max_vals = np.where(p0_samples > 0, m_max_vals, np.nan)

    factor = b0 * reflectivity * omega_r
    omega_vals = np.sqrt(factor * p0_samples * m_max_vals)
    mask = np.isfinite(p0_samples) & np.isfinite(omega_vals)
    return p0_samples[mask], omega_vals[mask]


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


def plot_time_averaged_profile(time_avg_profile, num_crit, p0, idx):
    """
    Plot the intermediate time-averaged S(x, t) profile used for FFT-based diagnostics.
    """
    profile = np.asarray(time_avg_profile, dtype=float)
    if profile.ndim != 1 or profile.size == 0:
        return

    N = profile.size
    L = 2 * np.pi * num_crit if num_crit > 0 else float(N)
    x_coords = np.linspace(-L / 2, L / 2, N, endpoint=False)

    plt.figure(figsize=(7, 3.5))
    plt.plot(x_coords, profile, linewidth=1.2)
    plt.title(f"Time-averaged S(x) for p0={p0:.3e} (index {idx})")
    plt.xlabel("x")
    plt.ylabel(r"$\langle S(x, t) \rangle_t$")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
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


def compute_s_potential_statistics(s_files, params, diag_index=None, avg_profile_index=None):
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
    avg_profile_plotted = False

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
        if avg_profile_index is not None and idx == avg_profile_index:
            plot_time_averaged_profile(time_avg_profile, num_crit, p0, idx)
            avg_profile_plotted = True

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
    if avg_profile_index is not None and not avg_profile_plotted:
        print(f"Average-profile index {avg_profile_index} was not plotted (out of range or filtered).")

    return np.array(valid_p0), np.array(fft_amp_vals)


def compute_density_statistics(psi_files, params):
    p0_vals = stand_utils.find_p0_vals_from_filenames(psi_files)
    if p0_vals.size == 0:
        raise ValueError("Unable to extract p0 values from psi filenames.")

    gamma_bar = params["gamma_bar"]
    b0 = params["b0"]
    reflectivity = params["R"]
    seed = params["seed"]
    num_crit = params["num_crit"]
    p_th = pump_threshold(gamma_bar, b0, reflectivity)
    t0_vals = analytic_delay_time(p0_vals, p_th, gamma_bar, seed)

    valid_p0 = []
    fft_amp_vals = []

    for psi_path, p0, t0 in zip(psi_files, p0_vals, t0_vals):
        if not np.isfinite(t0):
            print(f"Skipping p0={p0:.3e}: analytic t0 is not finite.")
            continue

        data = np.loadtxt(psi_path)
        if data.ndim != 2 or data.shape[1] < 2:
            print(f"Skipping p0={p0:.3e}: {os.path.basename(psi_path)} has invalid dimensions.")
            continue

        times = data[:, 0]
        after_t0 = times >= t0
        if not np.any(after_t0):
            print(f"Skipping p0={p0:.3e}: no density samples beyond analytic t0={t0:.3e}.")
            continue

        psi_values = data[after_t0, 1:]
        if psi_values.size == 0:
            print(f"Skipping p0={p0:.3e}: no spatial data beyond t0.")
            continue

        density_values = np.abs(psi_values) ** 2
        time_avg_profile = np.mean(density_values, axis=0)
        amp = compute_first_fourier_amplitude(time_avg_profile, num_crit)
        if not np.isfinite(amp):
            print(f"Skipping p0={p0:.3e}: unable to compute density FFT amplitude after averaging.")
            continue

        valid_p0.append(p0)
        fft_amp_vals.append(amp)

    return np.array(valid_p0), np.array(fft_amp_vals)

def main():
    args = parse_arguments()
    plt.rcParams["ps.usedistiller"] = "xpdf"
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
        }
    )

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
            s_files,
            params,
            diag_index=args.s_diagnostic_index,
            avg_profile_index=args.s_avg_index,
        )

    alt_curve_avg = (
        compute_analytic_curve_from_amplitude(s_p0_vals, fft_amp_vals, params) if s_p0_vals.size else None
    )
    delta_curve = None
    if s_p0_vals.size:
        omega_ratio = params.get("omega_r", np.nan)
        delta_param = params.get("Delta", np.nan)
        combined = omega_ratio * delta_param * fft_amp_vals
        mask = np.isfinite(s_p0_vals) & np.isfinite(combined) & (combined >= 0)
        if np.any(mask):
            delta_curve = (s_p0_vals[mask], np.sqrt(combined[mask]))

    density_delta_curve = None
    density_p0_vals, density_fft_amp_vals = compute_density_statistics(psi_files, params)
    if density_p0_vals.size:
        omega_ratio = params.get("omega_r", np.nan)
        combined = (
            params.get("b0", np.nan)
            * params.get("R", np.nan)
            * density_p0_vals
            * density_fft_amp_vals
            * omega_ratio
        )
        mask = np.isfinite(density_p0_vals) & np.isfinite(combined) & (combined >= 0)
        if np.any(mask):
            density_delta_curve = (density_p0_vals[mask], np.sqrt(combined[mask]))

    x_index = compute_x_index(float(args.xpos), params["nodes"], params["num_crit"])

    p0_vals, omega_vals, _ = compute_max_nqc_omega_gamma(psi_files, params)
    observed_p0, observed_omega = compute_observed_omega_gamma(psi_files, params, x_index)
    mmax_curve = compute_mmax_analytic_curve(p0_vals, params)

    plot_results(
        p0_vals,
        omega_vals,
        observed_p0,
        observed_omega,
        alt_curve_avg=alt_curve_avg,
        delta_curve=delta_curve,
        density_delta_curve=density_delta_curve,
        mmax_curve=mmax_curve,
    )


if __name__ == "__main__":
    main()

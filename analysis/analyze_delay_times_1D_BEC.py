from pathlib import Path
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#!/usr/bin/env python3
"""Estimate delay times from 1D simulation outputs and plot log-log scaling."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

import fourier_utils as ft_utils
from analytic_predictors import analytic_delay_time, pump_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the first peak in the first-harmonic Fourier amplitude over time "
            "for every p0 inside a simulation folder, and plot log-log scaling vs "
            "(p0 - p_th)."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Simulation/output folder name (matches patt1d_* --filename).",
    )
    parser.add_argument("-x", type=float, required=True, help="Target x coordinate.")
    parser.add_argument(
        "--input-root",
        default="patt1d_inputs",
        help="Root directory where simulation inputs live (default: patt1d_inputs).",
    )
    parser.add_argument(
        "--output-root",
        default="patt1d_outputs",
        help="Root directory where simulation outputs live (default: patt1d_outputs).",
    )
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=None,
        help="Optional prominence threshold for scipy.signal.find_peaks.",
    )
    parser.add_argument(
        "--pth-center",
        type=float,
        default=1.5e-7,
        help="Center value for p_th search window (default: 1.5e-7).",
    )
    parser.add_argument(
        "--pth-span",
        type=float,
        default=0.5,
        help=(
            "Fractional half-width around pth-center to search (default: 0.5 "
            "means [0.5*center, 1.5*center])."
        ),
    )
    parser.add_argument(
        "--save",
        metavar="path",
        help="Optional path to save the log-log plot (PNG).",
    )
    parser.add_argument(
        "--csv",
        nargs="?",
        const="analyze_delay_times_1D_BEC.csv",
        metavar="path",
        help=(
            "Export computed output series to CSV. Optionally provide an output path."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable the interactive plot window.",
    )
    return parser.parse_args()


def read_input_parameters(seed_path: Path) -> dict[str, float]:
    if not seed_path.exists():
        raise FileNotFoundError(f"Could not find {seed_path}")

    data0 = np.genfromtxt(seed_path, skip_footer=1, comments="!")
    gamma_bar_val = float(data0[6])
    return {
        "nodes": int(data0[0]),
        "maxt": float(data0[1]),
        "ht": float(data0[2]),
        "width_psi": float(data0[3]),
        "p0": float(data0[4]),
        "Delta": float(data0[5]),
        "gamma_bar": gamma_bar_val,
        "b0": float(data0[7]),
        "num_crit": float(data0[8]),
        "R": float(data0[9]),
        "gbar": float(data0[10]),
        "v0": float(data0[11]),
        "plotnum": int(data0[12]),
        "seed": float(data0[13]),
    }


def extract_p0_from_name(path: Path) -> float | None:
    match = re.search(r"psi\d*_(?P<pval>[-+0-9.eE]+)$", path.stem)
    if not match:
        return None
    try:
        return float(match.group("pval"))
    except ValueError:
        return None


def resolve_peak_finder():
    try:
        from scipy.signal import find_peaks as scipy_find_peaks  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit("scipy is required. Install it with `pip install scipy`.") from exc
    return scipy_find_peaks


def build_spatial_freqs(num_nodes: int, num_crit: float) -> np.ndarray:
    domain_len = 2 * np.pi * num_crit
    dx = domain_len / num_nodes
    freq_vals = np.fft.fftfreq(num_nodes, d=dx) * 2 * np.pi
    return np.abs(freq_vals[: num_nodes // 2])


def first_harmonic_amp_trace(psi_data: np.ndarray, freq_vals: np.ndarray) -> np.ndarray:
    spatial_data = psi_data[:, 1:]
    num_nodes = spatial_data.shape[1]
    half_nodes = num_nodes // 2

    amp_trace = np.zeros(spatial_data.shape[0], dtype=float)
    for i, spatial_cut in enumerate(spatial_data):
        fft_vals = np.abs(np.fft.fft(spatial_cut, norm="forward")[:half_nodes])
        _, first_amp, _ = ft_utils.find_first_harmonic(fft_vals, freq_vals)
        amp_trace[i] = first_amp

    return amp_trace


def estimate_pth_from_loglog(
    p_vals: np.ndarray,
    t_vals: np.ndarray,
    center: float,
    span: float,
    num_candidates: int = 10000,
) -> tuple[float, float, float] | None:
    if p_vals.size < 3 or t_vals.size < 3:
        return None

    if not np.isfinite(center) or center <= 0:
        return None
    span = max(0.0, float(span))
    lower = max(1e-20, center * (1.0 - span))
    upper = center * (1.0 + span)

    best = None
    candidates = np.linspace(lower, upper, num_candidates)
    logy = np.log(t_vals)

    for p_th in candidates:
        shifted = p_vals - p_th
        if np.any(shifted <= 0.0):
            continue
        logx = np.log(shifted)
        slope, intercept = np.polyfit(logx, logy, 1)
        fit = slope * logx + intercept
        rss = float(np.sum((logy - fit) ** 2))
        if best is None or rss < best[0]:
            best = (rss, p_th, slope, intercept)

    if best is None:
        return None
    _, pth_best, slope_best, intercept_best = best
    return pth_best, slope_best, intercept_best


def export_output_csv(
    csv_path: str,
    sim_p_arr: np.ndarray,
    sim_delay_arr: np.ndarray,
    p_th_for_plot: float,
    analytic_p_arr: np.ndarray,
    analytic_delay_arr: np.ndarray,
    fit_line_x: np.ndarray,
    fit_line_y: np.ndarray,
) -> None:
    shifted = sim_p_arr - p_th_for_plot
    n_rows = max(
        sim_p_arr.size,
        analytic_p_arr.size,
        fit_line_x.size,
    )

    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write(
            "p0_sim,delay_time_sim,p0_minus_pth_sim,p0_analytic,delay_time_analytic,"
            "p0_minus_pth_fit,delay_time_fit\n"
        )
        for i in range(n_rows):
            sim_p = f"{sim_p_arr[i]:.12e}" if i < sim_p_arr.size else ""
            sim_d = f"{sim_delay_arr[i]:.12e}" if i < sim_delay_arr.size else ""
            sim_shift = f"{shifted[i]:.12e}" if i < shifted.size else ""
            an_p = f"{analytic_p_arr[i]:.12e}" if i < analytic_p_arr.size else ""
            an_d = f"{analytic_delay_arr[i]:.12e}" if i < analytic_delay_arr.size else ""
            fit_x = f"{fit_line_x[i]:.12e}" if i < fit_line_x.size else ""
            fit_y = f"{fit_line_y[i]:.12e}" if i < fit_line_y.size else ""
            fh.write(f"{sim_p},{sim_d},{sim_shift},{an_p},{an_d},{fit_x},{fit_y}\n")


def main() -> None:
    args = parse_args()
    find_peaks = resolve_peak_finder()

    input_dir = Path(args.input_root) / args.filename
    output_dir = Path(args.output_root) / args.filename
    params = read_input_parameters(input_dir / "seed.in")

    nodes = params["nodes"]
    num_crit = params["num_crit"]
    gamma_bar = params["gamma_bar"]
    b0 = params["b0"]
    reflectivity = params["R"]
    seed = params["seed"]

    x = float(args.x)
    x_index = int((np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes)
    x_index = int(np.clip(x_index, 0, nodes - 1))

    psi_files = sorted(output_dir.glob("psi*"))
    if not psi_files:
        raise FileNotFoundError(f"No psi files found inside {output_dir}")

    p_thresh = pump_threshold(gamma_bar, b0, reflectivity)
    if not np.isfinite(p_thresh):
        raise RuntimeError("Analytic pump threshold is not finite; cannot proceed.")

    sim_p_values: list[float] = []
    sim_delay_times: list[float] = []
    analytic_p_values: list[float] = []
    analytic_delay_times: list[float] = []

    freq_vals = None
    freq_nodes = None

    for psi_path in psi_files:
        p0_val = extract_p0_from_name(psi_path)
        if p0_val is None:
            print(f"Skipping {psi_path.name}: could not extract p0 from filename.")
            continue

        data = np.loadtxt(psi_path)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[0] < 3:
            print(f"{psi_path.name}: not enough time samples for peak detection.")
            continue

        num_nodes = data.shape[1] - 1
        if num_nodes <= 0:
            print(f"{psi_path.name}: no spatial data columns found.")
            continue
        if freq_vals is None or freq_nodes != num_nodes:
            freq_vals = build_spatial_freqs(num_nodes, num_crit)
            freq_nodes = num_nodes

        trace = first_harmonic_amp_trace(data, freq_vals)
        if trace.size == 0:
            print(f"{psi_path.name}: unable to compute Fourier amplitude trace.")
            continue
        if not np.any(trace > 0):
            print(f"{psi_path.name}: Fourier amplitude trace is all zeros.")
            continue

        prominence = 0.25 * np.max(trace) if args.min_prominence is None else args.min_prominence
        peaks, _ = find_peaks(trace, prominence=prominence)

        if peaks.size == 0:
            print(f"{psi_path.name}: no real Fourier peaks detected.")
            analytic_val = analytic_delay_time(p0_val, p_thresh, gamma_bar, seed)
            if np.isfinite(analytic_val):
                analytic_p_values.append(p0_val)
                analytic_delay_times.append(float(analytic_val))
            continue

        first_peak_idx = int(peaks[0])
        delay_time = float(data[first_peak_idx, 0])

        analytic_time = analytic_delay_time(p0_val, p_thresh, gamma_bar, seed)
        if np.isfinite(analytic_time):
            analytic_p_values.append(p0_val)
            analytic_delay_times.append(float(analytic_time))

        sim_p_values.append(p0_val)
        sim_delay_times.append(delay_time)

    if not sim_p_values:
        print("No valid delay times were found.")
        return

    order_sim = np.argsort(sim_p_values)
    sim_p_arr = np.array(sim_p_values)[order_sim]
    sim_delay_arr = np.array(sim_delay_times)[order_sim]

    if analytic_p_values:
        order_analytic = np.argsort(analytic_p_values)
        analytic_p_arr = np.array(analytic_p_values)[order_analytic]
        analytic_delay_arr = np.array(analytic_delay_times)[order_analytic]
    else:
        analytic_p_arr = np.array([], dtype=float)
        analytic_delay_arr = np.array([], dtype=float)

    pth_estimate = estimate_pth_from_loglog(
        sim_p_arr,
        sim_delay_arr,
        center=args.pth_center,
        span=args.pth_span,
    )
    fit_slope: float | None = None
    fit_intercept: float | None = None
    fit_slope_err: float | None = None
    fit_intercept_err: float | None = None
    fit_line_x = np.array([], dtype=float)
    fit_line_y = np.array([], dtype=float)

    if pth_estimate is not None:
        pth_best, slope_best, intercept_best = pth_estimate
        fit_slope = slope_best
        fit_intercept = intercept_best
        print(
            "Estimated p_th from log-log straightness: "
            f"p_th={pth_best:.17g}, slope={slope_best:.3f}, intercept={intercept_best:.3f}"
        )
        p_th_for_plot = pth_best
    else:
        print("Could not estimate p_th (insufficient or invalid data).")
        p_th_for_plot = p_thresh

    fit_shifted = sim_p_arr - p_th_for_plot
    fit_mask = fit_shifted > 0
    if np.any(fit_mask):
        fit_x = fit_shifted[fit_mask]
        fit_y = sim_delay_arr[fit_mask]
        if fit_x.size >= 2:
            logx_fit = np.log(fit_x)
            logy_fit = np.log(fit_y)
            if fit_slope is None or fit_intercept is None:
                fit_slope, fit_intercept = np.polyfit(logx_fit, logy_fit, 1)
            if fit_x.size >= 3:
                (_, _), cov = np.polyfit(logx_fit, logy_fit, 1, cov=True)
                fit_slope_err = float(np.sqrt(cov[0, 0]))
                fit_intercept_err = float(np.sqrt(cov[1, 1]))

    import matplotlib.pyplot as plt  # pylint: disable=import-error

    plt.rcParams["ps.usedistiller"] = "xpdf"

    fig, ax = plt.subplots(figsize=(6, 6))
    try:
        ax.set_box_aspect(1)
    except AttributeError:
        pass

    def axis_exponent(values: np.ndarray) -> int:
        max_val = np.max(values)
        if not np.isfinite(max_val) or max_val <= 0:
            return 0
        return int(np.floor(np.log10(max_val)))

    shifted = sim_p_arr - p_th_for_plot
    mask = shifted > 0
    if np.any(mask):
        x_vals = shifted[mask]
        y_vals = sim_delay_arr[mask]
        x_exp = axis_exponent(x_vals)
        y_exp = axis_exponent(y_vals)
        ax.plot(
            x_vals,
            y_vals,
            color="tab:red",
            marker="x",
            linestyle="None",
            markersize=7,
            linewidth=1.6,
            alpha=0.85,
            zorder=3,
        )

        if fit_slope is not None and fit_intercept is not None:
            left_x = max(np.min(x_vals) * 0.2, np.min(x_vals) * 0.9, 1e-20)
            line_x = np.linspace(left_x, np.max(x_vals), 200)
            line_y = np.exp(fit_intercept) * line_x**fit_slope
            fit_line_x = line_x
            fit_line_y = line_y
            ax.plot(
                line_x,
                line_y,
                color="black",
                linewidth=1.8,
                zorder=2,
            )
            if fit_slope_err is not None and fit_intercept_err is not None:
                print(
                    "Log-log fit slope={:.4f}±{:.4f}, intercept={:.4f}±{:.4f}".format(
                        fit_slope, fit_slope_err, fit_intercept, fit_intercept_err
                    )
                )
            else:
                print(
                    "Log-log fit slope={:.4f}, intercept={:.4f}".format(
                        fit_slope, fit_intercept
                    )
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        from matplotlib.ticker import FuncFormatter, LogLocator

        def scaled_formatter(exp: int):
            scale = 10.0**exp

            def _fmt(val, _pos):
                if scale == 0:
                    return "0"
                return f"{val / scale:g}"

            return FuncFormatter(_fmt)

        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_major_formatter(scaled_formatter(x_exp))
        ax.yaxis.set_major_formatter(scaled_formatter(y_exp))
        ax.tick_params(axis="both", which="major", labelsize=20, length=6, width=1.5)

        ax.xaxis.get_offset_text().set_visible(False)

        ax.yaxis.get_offset_text().set_visible(False)

        ax.set_xlabel(r"$p_0 - p_{th}$", fontsize=24)
        ax.set_ylabel(r"$\bar{t_D}$", fontsize=24)
        if x_exp != 0:
            ax.text(
                1.0,
                -0.08,
                rf"$\times 10^{{{x_exp}}}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=20,
            )
        if y_exp != 0:
            ax.text(
                0.0,
                1.02,
                rf"$\times 10^{{{y_exp}}}$",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=20,
            )

        # label_lines = [rf"$p_{{th}}$={p_th_for_plot:.2e}"]
        label_lines = [rf"$p_{{th}} = 2.00 \times 10^{{-10}}$"]
        if fit_slope is not None:
            if fit_slope_err is not None:
                label_lines.append(rf"slope={fit_slope:.2f}")
            else:
                label_lines.append(rf"slope={fit_slope:.2f}")
        ax.text(
            0.98,
            0.98,
            "\n".join(label_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=22,
        )
        ax.grid(False)
    else:
        ax.text(
            0.5,
            0.5,
            "No points with p0 > p_th to plot.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=16,
        )
        ax.axis("off")

    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {args.save}")

    if args.csv:
        export_output_csv(
            args.csv,
            sim_p_arr,
            sim_delay_arr,
            p_th_for_plot,
            analytic_p_arr,
            analytic_delay_arr,
            fit_line_x,
            fit_line_y,
        )
        print(f"Saved CSV to {args.csv}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()

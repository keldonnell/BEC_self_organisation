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

import numpy as np

from analytic_predictors import analytic_delay_time, pump_threshold

FLOAT_PATTERN = re.compile(r"(\d+(?:\.\d+)?e[-+]\d+)", re.IGNORECASE)


def positive_int(value):
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer.")
    return ivalue


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Export averaged sqrt(S) variants after analytic delay for a specific p0 index."
        )
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Name of the patt1d input/output directory pair (e.g. high_res).",
    )
    parser.add_argument(
        "-i",
        "--index",
        required=True,
        type=positive_int,
        help="Zero-based index into the sorted list of S files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to write CSV files to (defaults to the chosen patt1d_outputs/<filename> folder).",
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


def _collect_field_files(output_dir, prefix):
    pattern = os.path.join(output_dir, f"{prefix}*.out")
    files = glob.glob(pattern)
    filtered = [path for path in files if FLOAT_PATTERN.search(os.path.basename(path))]
    if not filtered:
        raise FileNotFoundError(f"No {prefix}*.out files with embedded p0 values found in {output_dir}")
    return sorted(filtered, key=lambda f: float(FLOAT_PATTERN.search(f).group(1)))


def collect_s_files(output_dir):
    return _collect_field_files(output_dir, "s")


def find_p0_vals_from_filenames(sorted_files):
    pattern = r"(\d+(?:\.\d+)?e[-+]\d+)"
    return np.array([float(re.findall(pattern, file)[0]) for file in sorted_files])


def export_arrays_to_csv(path, values, header):
    np.savetxt(path, values, delimiter=",", header=header, comments="", fmt="%.10e")


def compute_x_positions(n_points, num_crit):
    L = 2 * np.pi * num_crit if num_crit > 0 else float(n_points)
    return np.linspace(-L / 2, L / 2, n_points, endpoint=False)


def main():
    args = parse_arguments()

    output_dir = args.output_dir or os.path.join("patt1d_outputs", args.filename)
    input_dir = os.path.join("patt1d_inputs", args.filename)
    seed_path = os.path.join(input_dir, "seed.in")

    params = read_seed_metadata(seed_path)
    s_files = collect_s_files(output_dir)
    if args.index >= len(s_files):
        raise IndexError(f"Index {args.index} is out of range for {len(s_files)} S files.")

    p0_vals = find_p0_vals_from_filenames(s_files)
    p0 = float(p0_vals[args.index])

    p_th = pump_threshold(params["gamma_bar"], params["b0"], params["R"])
    t0 = analytic_delay_time(p0, p_th, params["gamma_bar"], params["seed"])
    if not np.isfinite(t0):
        raise ValueError(f"Analytic delay time t0 is not finite for p0={p0:.3e}.")

    s_path = s_files[args.index]
    data = np.loadtxt(s_path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"{s_path} does not look like an S output table.")

    times = data[:, 0]
    after_t0_mask = times >= t0
    if not np.any(after_t0_mask):
        raise ValueError(f"No samples at times >= t0={t0:.3e} for p0={p0:.3e}.")

    filtered_data = data[after_t0_mask]
    s_values = filtered_data[:, 1:]

    # Use absolute values to avoid NaNs from small negative noise before square roots
    abs_s = np.abs(s_values)

    # Compute <sqrt(|S|)> by averaging sqrt across time samples
    avg_sqrt_s = np.mean(np.sqrt(abs_s), axis=0)

    # Compute sqrt(<|S|>) by averaging first, then taking the square root
    sqrt_avg_s = np.sqrt(np.mean(abs_s, axis=0))

    base_name = f"p0_{p0:.3e}_idx_{args.index}"
    combined_out_path = os.path.join(output_dir, f"s_profiles_{base_name}.csv")

    x_positions = compute_x_positions(s_values.shape[1], params["num_crit"])
    combined = np.column_stack((x_positions, avg_sqrt_s, sqrt_avg_s))
    export_arrays_to_csv(
        combined_out_path,
        combined,
        header="x_position,avg_sqrt_s,sqrt_avg_s",
    )

    print(f"Wrote profiles (t >= t0={t0:.3e}) to {combined_out_path}")


if __name__ == "__main__":
    main()

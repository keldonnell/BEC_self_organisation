# -*- coding: utf-8 -*-
"""
Data analysis and visualization script for pattern formation in one-dimensional systems.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import integrate, stats
import fourier_utils as fourier_utils
import standard_data_utils as stand_utils

# Set rcParams for matplotlib
plt.rcParams["ps.usedistiller"] = "xpdf"  # improves quality of .eps figures for use with LaTeX

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        args: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Data analysis for 1D pattern formation")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="filename",
        required=True,
        help="The name of the file to save to",
    )
    return parser.parse_args()

def read_input(seed_dir):
    """Read input data from seed file."""
    data = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return {
        "nodes": int(data[0]),
        "maxt": data[1],
        "ht": data[2],
        "width_psi": data[3],
        "p0": data[4],
        "Delta": data[5],
        "gambar": data[6],
        "b0": data[7],
        "num_crit": data[8],
        "R": data[9],
        "gbar": data[10],
        "v0": data[11],
        "plotnum": int(data[12]),
    }

def extract_float(filename):
    """
    Extract a floating point number from the filename.

    Args:
        filename (str): The filename from which to extract the number.
        
    Returns:
        float: Extracted floating point number or 0 if no number found.
    """
    match = re.search(r'psi\d+_([\d.e-]+)', filename)
    if match:
        # Remove trailing period if it exists
        float_str = match.group(1).rstrip('.')
        return float(float_str)
    return 0

def load_data(output_dir, frame_index):
    """
    Load data files and sort them based on the extracted floating point numbers.

    Args:
        output_dir (str): Directory containing the output data files.
        
    Returns:
        list: Sorted list of data files.
    """
    data1_files = glob.glob(output_dir + "psi*")
    sorted_files = sorted(data1_files, key=extract_float)
    return sorted_files

def main():

    # Parse command-line arguments
    args = parse_arguments()
    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = f"{input_dir}seed.in"

    # Read input data
    params = read_input(seed_dir)

    # Load and sort data files
    sorted_files = load_data(output_dir, 1)
    print(sorted_files)

    x_vals = np.linspace(-np.pi * params['num_crit'], np.pi * params['num_crit'], params['nodes'])
    k_vals = np.fft.fftfreq(len(x_vals), np.diff(x_vals)[0])

    p_th = (2 * params['gambar']) / (params['b0'] * params['R'])
    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)

    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
    p0_shift_vals = (p0_above_th_vals / p_th) - 1

    analysed_fourier_data = fourier_utils.analyse_fourier_data(sorted_files_above_th, k_vals, 1, False)
    analytic_M_vals = stand_utils.calc_analytic_M_at_t0(p0_above_th_vals, p_th)

    # Plotting the graph
    fig, ax = plt.subplots(2, 4, figsize=(30, 14))
    fig.subplots_adjust(wspace=0.3, hspace=0.45)

    # First harmonic amplitude
    ax[0, 0].set_title(r"1st harmonic amplitude for different $p_0$", fontsize=8)
    ax[0, 0].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
    ax[0, 0].set_ylabel(r"1st harmonic amplitude", fontsize=14)
    ax[0, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"])
    ax[0, 0].scatter(p0_shift_vals, analytic_M_vals, label="M", marker="D", s=8)
    opt_params, fit_M_vals = stand_utils.fit_y_data(
        analysed_fourier_data["first_mode_ft_peaks_amp"], analytic_M_vals, [1.19, -0.013]
    )
    ax[0, 0].scatter(
        p0_shift_vals, fit_M_vals, label=f"y = {opt_params[0]:.3} * M + {opt_params[1]:.3} (Fitted analytical M)",
        color="black", marker="x"
    )
    ax[0, 0].legend(fontsize=6)

    # First harmonic frequency
    ax[0, 1].set_title(r"1st harmonic frequency for different $p_0$", fontsize=8)
    ax[0, 1].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
    ax[0, 1].set_ylabel(r"1st harmonic frequency", fontsize=14)
    ax[0, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"])

    # Area under first harmonic of |ψ|²
    ax[0, 2].set_title(r"Area under first harmonic of $|\psi|^2$", fontsize=8)
    ax[0, 2].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
    ax[0, 2].set_ylabel(r"Area under first harmonic (energy)", fontsize=14)
    ax[0, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"])
    opt_params, fit_M_vals = stand_utils.fit_y_data(
        analysed_fourier_data["first_mode_ft_peak_area"], analytic_M_vals, [0.0309, -0.0003]
    )
    ax[0, 2].scatter(
        p0_shift_vals, fit_M_vals, label=f"y = {opt_params[0]:.3} * M + {opt_params[1]:.3} (Fitted analytical M)",
        color="black", marker="x"
    )
    ax[0, 2].legend(fontsize=6)

    # Area under higher harmonics of |ψ|²
    ax[0, 3].set_title(r"Area under all higher harmonics of $|\psi|^2$", fontsize=8)
    ax[0, 3].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
    ax[0, 3].set_ylabel(r"Area under all higher harmonics (energy)", fontsize=14)
    ax[0, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"])

    # Logarithmic regression of first harmonic amplitude
    final_fit_index = 10
    exponent, coefficient, r_srd, conf_interval = stand_utils.find_log_exponent(
        p0_shift_vals[:final_fit_index], analysed_fourier_data["first_mode_ft_peaks_amp"][:final_fit_index]
    )
    print(f"1st mode ft peak: exponent = {exponent} ± {conf_interval}, coefficient = {coefficient}, r-squared = {r_srd}")
    ax[1, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"], label='1st harmonic amp')
    x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), len(p0_shift_vals))[:final_fit_index]
    y_smooth = coefficient * x_smooth**exponent
    ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    exponent, coefficient, r_srd, conf_interval = stand_utils.find_log_exponent(
        p0_shift_vals[:final_fit_index], analytic_M_vals[:final_fit_index]
    )
    print(f"analytic M: exponent = {exponent} ± {conf_interval}, coefficient = {coefficient}, r-squared = {r_srd}")
    ax[1, 0].scatter(p0_shift_vals, analytic_M_vals, label='Analytic M', marker="D", s=8)
    y_smooth = coefficient * x_smooth**exponent
    ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    ax[1, 0].set_xscale('log')
    ax[1, 0].set_yscale('log')
    ax[1, 0].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
    ax[1, 0].set_ylabel(r'1st harmonic amplitude')
    ax[1, 0].set_title(
        f'Logarithmic Regression of 1st harmonic amplitude \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8
    )
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Linear regression of first harmonic frequency
    ax[1, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"], label='Data')
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"]
    )
    y_smooth = slope * x_smooth + intercept
    print(f"1st mode ft frequency: slope = {slope}, intercept = {intercept}, r-value = {r_value}")
    ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    ax[1, 1].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
    ax[1, 1].set_ylabel(r'1st harmonic frequency')
    ax[1, 1].set_title(f'Linear Regression of 1st harmonic frequency \n (y = {slope} * x + {intercept})', fontsize=8)
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    # Logarithmic regression of area under first harmonic
    final_fit_index = 7
    exponent, coefficient, r_srd, conf_interval = stand_utils.find_log_exponent(
        p0_shift_vals[:final_fit_index], analysed_fourier_data["first_mode_ft_peak_area"][:final_fit_index]
    )
    print(f"1st mode ft peak: exponent = {exponent} ± {conf_interval}, coefficient = {coefficient}, r-squared = {r_srd}")
    ax[1, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"], label='Data')
    y_smooth = coefficient * x_smooth**exponent
    ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    ax[1, 2].set_xscale('log')
    ax[1, 2].set_yscale('log')
    ax[1, 2].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
    ax[1, 2].set_ylabel(r'Area under first harmonic (energy)')
    ax[1, 2].set_title(
        f'Logarithmic regression of the area under the first harmonic \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8
    )
    ax[1, 2].legend()
    ax[1, 2].grid(True)

    # Linear regression of area under higher harmonics
    ax[1, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"], label='Data')
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"]
    )
    y_smooth = slope * x_smooth + intercept
    print(f"1st mode ft frequency: slope = {slope}, intercept = {intercept}, r-value = {r_value}")
    ax[1, 3].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    ax[1, 3].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
    ax[1, 3].set_ylabel(r'Area under higher harmonics (energy)')
    ax[1, 3].set_title(f'Linear Regression of area under higher harmonics \n (y = {slope} * x + {intercept})', fontsize=8)
    ax[1, 3].legend()
    ax[1, 3].grid(True)

    plt.show()

if __name__ == "__main__":
    main()
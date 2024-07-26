
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import integrate, stats
import fourier_utils as fourier_utils
import standard_data_utils as stand_utils

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze Fourier transforms of patt1d outputs.")
    parser.add_argument(
        "-f",
        "--filename",
        metavar="filename",
        required=True,
        help="The name of the file to save to",
    )
    parser.add_argument(
        "-x",
        "--xpos",
        metavar="x_position",
        required=True,
        help="The x co-ord to inspect the fourier transform at",
    )
    return parser.parse_args()

def read_input(seed_dir):
    """
    Read input data from the seed file.
    
    Args:
        seed_dir (str): Path to the seed file
    
    Returns:
        tuple: Various parameters read from the seed file
    """
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    return (
        data0[0].astype(int),  # nodes
        *data0[1:12],
        data0[12].astype(int)  # plotnum
    )

def extract_float(filename):
    """
    Extract float value from filename.
    
    Args:
        filename (str): Name of the file
    
    Returns:
        float: Extracted float value or 0 if not found
    """
    match = re.search(r'psi\d+_([\d.e-]+)', filename)
    if match:
        float_str = match.group(1).rstrip('.')
        return float(float_str)
    return 0

def load_data(output_dir):
    """
    Load and sort data files from the output directory.
    
    Args:
        output_dir (str): Path to the output directory
    
    Returns:
        list: Sorted list of file paths
    """
    data1_files = glob.glob(output_dir + "psi*")
    return sorted(data1_files, key=extract_float)

def calculate_values(sorted_files, nodes, p_th, x_vals, Delta, R, b0, gambar, x_index, freq_vals):
    """
    Calculate various values based on the sorted files and input parameters.
    
    Args:
        sorted_files (list): List of sorted file paths
        nodes (int): Number of nodes
        p_th (float): Threshold value
        x_vals (np.array): Array of x values
        Delta (float): Delta parameter
        R (float): R parameter
        b0 (float): b0 parameter
        gambar (float): gambar parameter
        x_index (int): Index of x position to analyze
        freq_vals (np.array): Array of frequency values
    
    Returns:
        tuple: Various calculated values
    """
    # Extract p0 values from filenames
    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)
    
    # Find values above threshold
    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
    p0_shift_vals = (p0_above_th_vals / p_th) - 1

    # Analyze Fourier data
    analysed_fourier_data = fourier_utils.analyse_fourier_data(sorted_files_above_th, freq_vals, 1, True, x_index)

    return p0_shift_vals, analysed_fourier_data

def create_plots(p0_shift_vals, analysed_fourier_data, x, plot_data):
    """
    Create initial plots for various calculated values.
    
    Args:
        p0_shift_vals (np.array): Shifted p0 values
        analysed_fourier_data (dict): Dictionary containing analyzed Fourier data
        x (float): X position being analyzed
        plot_data (list): List of tuples containing plot information
    
    Returns:
        tuple: Figure and axes objects
    """
    fig, ax = plt.subplots(2, 4, figsize=(30, 14))
    fig.subplots_adjust(wspace=0.3, hspace=0.45)

    # Create scatter plots for the first row
    for ax_plot, title, ylabel, data in plot_data:
        ax_plot.set_title(title, fontsize=8)
        ax_plot.set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
        ax_plot.set_ylabel(ylabel, fontsize=14)
        ax_plot.scatter(p0_shift_vals, data)

    return fig, ax

def fit_and_plot(ax, x_vals, y_vals, xlabel, ylabel, title, final_fit_index):
    """
    Perform logarithmic regression and plot the results.
    
    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        x_vals (np.array): x values
        y_vals (np.array): y values
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        title (str): Title of the plot
        final_fit_index (int): Index up to which to perform the fit
    """
    # Perform logarithmic regression
    exponent, coefficient, r_srd, conf_interval = stand_utils.find_log_exponent(x_vals[:final_fit_index], y_vals[:final_fit_index])
    print(f"{title}: exponent = {exponent} +- {conf_interval}, coefficient = {coefficient}, r-squared = {r_srd}")

    # Plot scatter of original data
    ax.scatter(x_vals, y_vals, label='Data')

    # Generate smooth curve for the fit
    x_smooth = np.linspace(x_vals.min(), x_vals.max(), len(x_vals))[:final_fit_index]
    y_smooth = coefficient * x_smooth**exponent

    # Plot the fitted curve
    ax.plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
    
    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Logarithmic Regression of {title}\n(Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8)
    
    ax.legend()
    ax.grid(True)

def main():
    """
    Main function to run the script.
    """
    # Set matplotlib parameters
    plt.rcParams["ps.usedistiller"] = "xpdf"
    
    # Parse command-line arguments
    args = parse_arguments()

    # Define directories
    output_dir = f"patt1d_outputs/{args.filename}/"
    input_dir = f"patt1d_inputs/{args.filename}/"
    seed_dir = f"{input_dir}seed.in"

    # Read input data
    nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = read_input(seed_dir)

    # Calculate x index
    x = float(args.xpos)
    x_index = int((np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes)

    # Load and sort data files
    sorted_files = load_data(output_dir)

    # Calculate time and frequency values
    t_vals = np.linspace(0, maxt, plotnum)
    freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])

    # Calculate threshold value
    p_th = (2 * gambar) / (b0 * R)

    # Calculate various values
    p0_shift_vals, analysed_fourier_data = calculate_values(
        sorted_files, nodes, p_th, None, Delta, R, b0, gambar, x_index, freq_vals
    )

    # Create initial plots
    fig, ax = plt.subplots(2, 4, figsize=(30, 14))
    fig.subplots_adjust(wspace=0.3, hspace=0.45)

    # Define plot data for the first row of subplots
    plot_data = [
        (ax[0, 0], f"1st harmonic amplitude for different p0 at x = {x}", "1st harmonic amplitude", analysed_fourier_data["first_mode_ft_peaks_amp"]),
        (ax[0, 1], f"1st harmonic frequency for different p0 at x = {x}", "1st harmonic frequency", analysed_fourier_data["first_mode_ft_peaks_freq"]),
        (ax[0, 2], f"Area under first harmonic of $|\\psi|^2$ at x = {x}", "Area under first harmonic (energy)", analysed_fourier_data["first_mode_ft_peak_area"]),
        (ax[0, 3], f"Area under all higher harmonics of $|\\psi|^2$ at x = {x}", "Area under all higher harmonics (energy)", analysed_fourier_data["higher_modes_ft_peak_area"]),
    ]

    # Create initial plots
    for ax_plot, title, ylabel, data in plot_data:
        ax_plot.set_title(title, fontsize=8)
        ax_plot.set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
        ax_plot.set_ylabel(ylabel, fontsize=14)
        ax_plot.scatter(p0_shift_vals, data)

    # Define data for logarithmic regression plots
    fit_data = [
        (ax[1, 0], p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"], r'$\frac{p_0 - p_{th}}{p_{th}}$', '1st harmonic amplitude', '1st harmonic amplitude', 15),
        (ax[1, 1], p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"], r'$\frac{p_0 - p_{th}}{p_{th}}$', '1st harmonic frequency', '1st harmonic frequency', 15),
        (ax[1, 2], p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"], r'$\frac{p_0 - p_{th}}{p_{th}}$', 'Area under first harmonic (energy)', 'Area under first harmonic', 15),
        (ax[1, 3], p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"], r'$\frac{p_0 - p_{th}}{p_{th}}$', 'Area under all higher modes (energy)', 'Area under higher modes', 15),
    ]

    # Perform logarithmic regression and plot results
    for ax_plot, x_vals, y_vals, xlabel, ylabel, title, final_fit_index in fit_data:
        fit_and_plot(ax_plot, x_vals, y_vals, xlabel, ylabel, title, final_fit_index)

    # Display the plots
    plt.show()

if __name__ == "__main__":
    main()
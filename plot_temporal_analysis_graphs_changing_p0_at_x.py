import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import stats
import standard_data_utils as stand_utils

def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Analyze temporal statistics of patt1d outputs.")
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
        help="The x co-ord to inspect the temporal statistics at",
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

def calculate_values(sorted_files, nodes, p_th, x_index):
    """
    Calculate various statistical values based on the sorted files and input parameters.
    
    Args:
        sorted_files (list): List of sorted file paths
        nodes (int): Number of nodes
        p_th (float): Threshold value
        x_index (int): Index of x position to analyze
    
    Returns:
        tuple: Various calculated values
    """
    p0_vals = stand_utils.find_p0_vals_from_filenames(sorted_files)
    p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
    p0_shift_vals = (p0_above_th_vals / p_th) - 1 

    sd_vals = stand_utils.calc_standard_deviation(sorted_files_above_th, nodes, True, x_index)
    mod_depth_vals = stand_utils.calc_modulation_depth(sorted_files_above_th, nodes, True, x_index)
    span_vals = stand_utils.calc_span(sorted_files_above_th, nodes, True, x_index)

    return p0_shift_vals, p0_above_th_vals, sd_vals, mod_depth_vals, span_vals

def create_plots(p0_shift_vals, sd_vals, mod_depth_vals, span_vals, x):
    """
    Create initial plots for various calculated values.
    
    Args:
        p0_shift_vals (np.array): Shifted p0 values
        sd_vals (np.array): Standard deviation values
        mod_depth_vals (np.array): Modulation depth values
        span_vals (np.array): Span values
        x (float): X position being analyzed
    
    Returns:
        tuple: Figure and axes objects
    """
    fig, ax = plt.subplots(2, 3, figsize=(22, 14))
    fig.subplots_adjust(wspace=0.55, hspace=0.3)

    # Plot data for the first row
    plot_data = [
        (ax[0, 0], r"Temporal standard deviation of $|\psi|^2$ at x = " + str(x), r"$\sigma[|\psi|^2]$", sd_vals),
        (ax[0, 1], r"Modulation depth of $|\psi|^2$ at x = " + str(x), r"Modulation depth m$[|\psi|^2]$", mod_depth_vals),
        (ax[0, 2], r"Span of $|\psi|^2$ at x = " + str(x), r"Span $[|\psi|^2]$", span_vals),
    ]

    for ax_plot, title, ylabel, data in plot_data:
        ax_plot.set_title(title)
        ax_plot.set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
        ax_plot.set_ylabel(ylabel, fontsize=14)
        ax_plot.scatter(p0_shift_vals, data)

    return fig, ax

def create_linear_plots(p0_vals, sd_vals, mod_depth_vals, span_vals, p_th, x):
    """
    Create linear-scale plots of the temporal metrics versus p0.

    Args:
        p0_vals (np.array): p0 values above threshold
        sd_vals (np.array): Standard deviation values
        mod_depth_vals (np.array): Modulation depth values
        span_vals (np.array): Span values
        p_th (float): Threshold p0 value
        x (float): X position being analyzed

    Returns:
        tuple: Figure and axes objects
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))
    fig.subplots_adjust(wspace=0.35)

    plot_data = [
        (axs[0], r"Temporal standard deviation of $|\psi|^2$ at x = " + str(x), r"$\sigma[|\psi|^2]$", sd_vals),
        (axs[1], r"Modulation depth of $|\psi|^2$ at x = " + str(x), r"Modulation depth m$[|\psi|^2]$", mod_depth_vals),
        (axs[2], r"Span of $|\psi|^2$ at x = " + str(x), r"Span $[|\psi|^2]$", span_vals),
    ]

    for i, (ax_plot, title, ylabel, data) in enumerate(plot_data):
        ax_plot.set_title(title)
        ax_plot.set_xlabel(r"$p_0$", fontsize=14)
        ax_plot.set_ylabel(ylabel, fontsize=14)
        ax_plot.scatter(p0_vals, data)
        if i == 0:
            ax_plot.axvline(p_th, color="red", linestyle="--", label=r"$p_{th}$")
            ax_plot.legend()
        else:
            ax_plot.axvline(p_th, color="red", linestyle="--")
        ax_plot.grid(True, alpha=0.3)

    return fig, axs

def fit_and_plot(ax, x_vals, y_vals, xlabel, ylabel, title, final_fit_index=None):
    """
    Perform logarithmic regression and plot the results.
    
    Args:
        ax (matplotlib.axes.Axes): Axes object to plot on
        x_vals (np.array): x values
        y_vals (np.array): y values
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        title (str): Title of the plot
        final_fit_index (int, optional): Index up to which to perform the fit
    """
    if final_fit_index is None:
        final_fit_index = len(x_vals)

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
    ax.set_title(f'Logarithmic Regression of {title}\n(Fit: y = {coefficient:.2e} * x^{exponent:.2f})')
    
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

    # Calculate threshold value
    p_th = (2 * gambar) / (b0 * R)

    # Calculate various values
    p0_shift_vals, p0_vals_above_th, sd_vals, mod_depth_vals, span_vals = calculate_values(sorted_files, nodes, p_th, x_index)

    # Create initial plots
    fig, ax = create_plots(p0_shift_vals, sd_vals, mod_depth_vals, span_vals, x)
    fig_linear, _ = create_linear_plots(p0_vals_above_th, sd_vals, mod_depth_vals, span_vals, p_th, x)

    # Define data for logarithmic regression plots
    fit_data = [
        (ax[1, 0], p0_shift_vals, sd_vals, r'$\frac{p_0 - p_{th}}{p_{th}}$', r'$\sigma[|\psi^2|]$', 'Standard Deviation', None),
        (ax[1, 1], p0_shift_vals, mod_depth_vals, r'$\frac{p_0 - p_{th}}{p_{th}}$', r'Modulation Depth $m[|\psi^2|]$', 'Modulation Depth', 8),
        (ax[1, 2], p0_shift_vals, span_vals, r'$\frac{p_0 - p_{th}}{p_{th}}$', r'Span$[|\psi^2|]$', 'Span', None),
    ]

    # Perform logarithmic regression and plot results
    for ax_plot, x_vals, y_vals, xlabel, ylabel, title, final_fit_index in fit_data:
        fit_and_plot(ax_plot, x_vals, y_vals, xlabel, ylabel, title, final_fit_index)

    # Display the plots
    plt.show()

if __name__ == "__main__":
    main()

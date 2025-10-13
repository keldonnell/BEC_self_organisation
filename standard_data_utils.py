import numpy as np
from scipy.signal import find_peaks
from scipy import integrate, stats, constants, optimize
import fourier_utils as ft_utils
import re

# Constants
MIN_ABOVE_THRESHOLD_SCALING = 1e-3 
PEAK_PROMINENCE_FACTOR = 3
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_INITIAL_PARAMS = [1, 0]
FIRST_HARMONIC_PROMINENCE = 0.00043

def find_vals_above_th(p0_vals, sorted_files, p_th):
    """
    Find values above a threshold in p0_vals and return corresponding files.

    Args:
    p0_vals (array): Array of p0 values.
    sorted_files (array): Array of sorted file names.
    p_th (float): Threshold value.
    min_above_th (float): Minimum value above threshold to consider.

    Returns:
    tuple: (p0 values above threshold, corresponding sorted files)
    """
    min_above_th = p_th * MIN_ABOVE_THRESHOLD_SCALING
    min_ind_above_th = next((i for i, val in enumerate(p0_vals) if val > (p_th + min_above_th)), len(p0_vals))
    print(f"p_th is {p_th}")
    print("THE MIN IS " + str(min_ind_above_th))

    return p0_vals[min_ind_above_th:], sorted_files[min_ind_above_th:]

def find_temporal_cut_of_x_peaks(psi_data, starting_x_index):
    """
    Find temporal cut of x peaks in psi_data.

    Args:
    psi_data (2D array): Data containing psi values.
    starting_x_index (int): Starting index for x.

    Returns:
    array: Temporal cut of x peaks.
    """
    max_amp = np.max(psi_data[:, starting_x_index])
    print(f"max_amp = {max_amp}")

    x_centre_indices = [starting_x_index]
    for i in range(1, psi_data.shape[0]):
        x_cut_data = psi_data[i]
        peak_indices, _ = find_peaks(x_cut_data, prominence=(max_amp / PEAK_PROMINENCE_FACTOR))
        if peak_indices.size:
            current_x_centre_index = peak_indices[np.argmin(np.abs(peak_indices - x_centre_indices[-1]))]
        else:
            current_x_centre_index = x_centre_indices[-1]
        x_centre_indices.append(current_x_centre_index)

    temporal_cut = np.array([psi_data[i, x_centre_indices[i]] for i in range(len(x_centre_indices))])
    print(temporal_cut)
    return temporal_cut

def calc_standard_deviation(sorted_files, nodes, is_temporal, x_index=None):
    """
    Calculate standard deviation for data in sorted_files.

    Args:
    sorted_files (list): List of sorted file names.
    nodes (int): Number of nodes.
    is_temporal (bool): Whether to use temporal cut.
    x_index (int, optional): Index for temporal cut.

    Returns:
    list: Standard deviation values.
    """
    if is_temporal and x_index is None:
        raise ValueError("You must specify an x_index for a temporal cut")

    sd_vals = []
    for file in sorted_files:
        data = np.loadtxt(file)
        cut_data = find_temporal_cut_of_x_peaks(data, x_index) if is_temporal else data[np.argmax(data[:, nodes // 2]), 1:]
        sd_vals.append(np.std(cut_data))

    return sd_vals

def calc_modulation_depth(sorted_files, nodes, is_temporal, x_index=None):
    """
    Calculate modulation depth for data in sorted_files.

    Args:
    sorted_files (list): List of sorted file names.
    nodes (int): Number of nodes.
    is_temporal (bool): Whether to use temporal cut.
    x_index (int, optional): Index for temporal cut.

    Returns:
    list: Modulation depth values.
    """
    if is_temporal and x_index is None:
        raise ValueError("You must specify an x_index for a temporal cut")

    mod_depth_vals = []
    for file in sorted_files:
        data = np.loadtxt(file)
        cut_data = find_temporal_cut_of_x_peaks(data, x_index) if is_temporal else data[np.argmax(data[:, nodes // 2]), 1:]
        max_val, min_val = np.max(cut_data), np.min(cut_data)
        mod_depth = (max_val - min_val) / (max_val + min_val) * 100
        mod_depth_vals.append(mod_depth)

    return mod_depth_vals

def find_log_exponent(x, y, confidence_level=DEFAULT_CONFIDENCE_LEVEL):
    """
    Find log exponent for given x and y data.

    Args:
    x (array-like): x values.
    y (array-like): y values.
    confidence_level (float): Confidence level for interval calculation.

    Returns:
    tuple: (exponent, coefficient, r_squared, confidence_interval)
    """
    x, y = np.array(x), np.array(y)
    mask = (x > 0) & (y > 0)
    x, y = x[mask], y[mask]

    x_log, y_log = np.log(x), np.log(y)
    slope, intercept, r_value, _, std_err = stats.linregress(x_log, y_log)

    n = len(x)
    dof = n - 2
    t_stat = stats.t.ppf((1 + confidence_level) / 2, dof)
    ci = t_stat * std_err

    return slope, np.exp(intercept), r_value**2, ci

def find_p0_vals_from_filenames(sorted_files):
    """
    Extract p0 values from filenames.

    Args:
    sorted_files (list): List of sorted file names.

    Returns:
    array: Extracted p0 values.
    """
    pattern = r"(\d+(?:\.\d+)?e[-+]\d+)"
    return np.array([float(re.findall(pattern, file)[0]) for file in sorted_files])

def calc_span(sorted_files, nodes, is_temporal, x_index=None):
    """
    Calculate span for data in sorted_files.

    Args:
    sorted_files (list): List of sorted file names.
    nodes (int): Number of nodes.
    is_temporal (bool): Whether to use temporal cut.
    x_index (int, optional): Index for temporal cut.

    Returns:
    list: Span values.
    """
    if is_temporal and x_index is None:
        raise ValueError("You must specify an x_index for a temporal cut")

    span_vals = []
    for file in sorted_files:
        data = np.loadtxt(file)
        cut_data = find_temporal_cut_of_x_peaks(data, x_index) if is_temporal else data[np.argmax(data[:, nodes // 2]), 1:]
        span_vals.append(np.abs(np.max(cut_data) - np.min(cut_data)))

    return span_vals

def calc_legett(psi_data, x_vals, nodes, t_index=None):
    """
    Calculate Legett value for given psi_data.

    Args:
    psi_data (2D array): Data containing psi values.
    x_vals (array): x values.
    nodes (int): Number of nodes.
    t_index (int, optional): Time index.

    Returns:
    float: Calculated Legett value.
    """
    row_data = psi_data[t_index, 1:] if t_index is not None else psi_data[np.argmax(psi_data[:, nodes // 2]), 1:]
    recip_psi_vals_at_t = 1 / row_data
    L = np.abs(x_vals[-1] - x_vals[0])
    N = integrate.simps(row_data, x_vals)
    Q_0 = 1 / (integrate.simps(recip_psi_vals_at_t, x_vals) * (N / L**2))
    return Q_0

def calc_legett_chng_p0(sorted_files, x_vals, nodes):
    """
    Calculate Legett values for changing p0.

    Args:
    sorted_files (list): List of sorted file names.
    x_vals (array): x values.
    nodes (int): Number of nodes.

    Returns:
    list: Legett values.
    """
    return [calc_legett(np.loadtxt(file), x_vals, nodes) for file in sorted_files]

def calc_oscill_omega(psi_data, x_vals, nodes, delta, R, p0, b0, gambar, decay_rate, t_index=None):
    """
    Calculate oscillation omega.

    Args:
    psi_data (2D array): Data containing psi values.
    x_vals (array): x values.
    nodes (int): Number of nodes.
    delta, R, p0, b0, gambar, decay_rate: Physical parameters.
    t_index (int, optional): Time index.

    Returns:
    float: Calculated omega value.
    """
    row_data = psi_data[t_index, 1:] if t_index is not None else psi_data[np.argmax(psi_data[:, nodes // 2]), 1:]
    fft_psi_vals = np.abs(np.fft.fft(row_data, norm="forward")[:len(row_data) // 2])
    k_vals = np.fft.fftfreq(len(x_vals), np.diff(x_vals)[0])
    abs_n_qc = ft_utils.find_first_harmonic(fft_psi_vals, k_vals, prominence=FIRST_HARMONIC_PROMINENCE)[1]
    return np.sqrt((decay_rate**2 * R * p0 * b0 * gambar * abs_n_qc) / constants.hbar)

def calc_oscill_omega_vals(sorted_files, sorted_p0_vals, x_vals, nodes, delta, R, b0, gambar, decay_rate):
    """
    Calculate oscillation omega values for multiple files.

    Args:
    sorted_files (list): List of sorted file names.
    sorted_p0_vals (array): Sorted p0 values.
    x_vals (array): x values.
    nodes (int): Number of nodes.
    delta, R, b0, gambar, decay_rate: Physical parameters.

    Returns:
    list: Calculated omega values.
    """
    return [calc_oscill_omega(np.loadtxt(file), x_vals, nodes, delta, R, p0, b0, gambar, decay_rate)
            for file, p0 in zip(sorted_files, sorted_p0_vals)]

def calc_analytic_M_at_t0(p0_vals, p_th):
    """
    Calculate analytic M at t=0.

    Args:
    p0_vals (array): p0 values.
    p_th (float): Threshold value.

    Returns:
    array: Calculated M values.
    """
    return (np.sqrt(2) * p_th / p0_vals) * np.sqrt((p0_vals / p_th) - 1)

def scale_and_shift(params, y_to_scale):
    """Apply scaling and shifting to y_to_scale."""
    scale, shift = params
    return scale * y_to_scale + shift

def error_function(params, y_reference, y_to_scale):
    """Calculate the sum of squared differences between y_reference and scaled y_to_scale."""
    y_scaled = scale_and_shift(params, y_to_scale)
    return np.sum((y_reference - y_scaled) ** 2)

def fit_y_data(y_reference, y_to_scale, initial_params=DEFAULT_INITIAL_PARAMS):
    """
    Fit y_to_scale to y_reference by finding the optimal scaling factor and shift.

    Args:
    y_reference (array-like): Reference y-values.
    y_to_scale (array-like): Y-values to be scaled and shifted.
    initial_params (list): Initial parameters for optimization.

    Returns:
    tuple: (optimal_params, y_fitted)
    """
    result = optimize.minimize(
        error_function,
        initial_params,
        args=(y_reference, y_to_scale),
        method="Nelder-Mead",
    )
    optimal_params = result.x
    y_fitted = scale_and_shift(optimal_params, y_to_scale)
    return optimal_params, y_fitted
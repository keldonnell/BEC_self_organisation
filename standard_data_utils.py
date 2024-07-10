
import numpy as np
from scipy.signal import find_peaks
from scipy import integrate, stats, constants
import fourier_utils as ft_utils


def find_vals_above_th(p0_vals, sorted_files, p_th, min_above_th = 0.05e-7):
    
    min_ind_above_th = np.min([i for i, val in enumerate(p0_vals) if val > (p_th + min_above_th)])

    p0_above_th_vals = p0_vals[min_ind_above_th:]
    sorted_files_above_th = sorted_files[min_ind_above_th:]

    return p0_above_th_vals, sorted_files_above_th

def calc_standard_deviation(sorted_files, nodes, is_temporal, x_index = None):

    if is_temporal and x_index == None:
        raise Exception("You must specify a x index for a temporal cut")

    if is_temporal:
        sd_vals = [
            np.sqrt(
                np.mean(
                    (
                        np.loadtxt(sorted_files[i])[:, x_index]
                        - np.mean(np.loadtxt(sorted_files[i])[:, x_index])
                    )
                    ** 2
                )
            )
            for i in range(len(sorted_files))
        ]
    else:
        sd_vals = [
            np.sqrt(
                np.mean(
                    (
                        np.loadtxt(sorted_files[i])[
                            np.argmax(np.loadtxt(sorted_files[i])[:, nodes // 2]), 1:
                        ]
                        - np.mean(
                            np.loadtxt(sorted_files[i])[
                                np.argmax(np.loadtxt(sorted_files[i])[:, nodes // 2]), 1:
                            ]
                        )
                    )
                    ** 2
                )
            )
            for i in range(len(sorted_files))
        ]

    return sd_vals


def calc_modulation_depth(sorted_files, nodes, is_temporal, x_index = None):

    if is_temporal and x_index == None:
        raise Exception("You must specify a x index for a temporal cut")

    mod_depth_vals = []
        
    for file in sorted_files:
        # Load the file only once
        data = np.loadtxt(file)
        
        if is_temporal:
            row_data = data[:, x_index]
        else:
            # Find the index of maximum value in the middle column
            max_index = np.argmax(data[:, nodes // 2])
            # Extract the relevant row
            row_data = data[max_index, 1:]
        
        # Calculate max and min
        max_val = np.max(row_data)
        min_val = np.min(row_data)
        
        # Calculate modulation depth
        mod_depth = (max_val - min_val) / (max_val + min_val) * 100
        
        mod_depth_vals.append(mod_depth)

    return mod_depth_vals


def find_log_exponent(x, y):
    # Ensure x and y are positive (domain of log function)
    x = np.array(x)
    y = np.array(y)
    mask = (x > 0) & (y > 0)
    x = x[mask]
    y = y[mask]

    # Transform both x and y data
    x_log = np.log(x)
    y_log = np.log(y)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)

    # The exponent is the slope of the linear regression
    exponent = slope

    # Calculate the coefficient (e^intercept)
    coefficient = np.exp(intercept)

    return exponent, coefficient, r_value**2


def calc_span(sorted_files, nodes, is_temporal, x_index = None):

    if is_temporal and x_index == None:
        raise Exception("You must specify a x index for a temporal cut")

    if is_temporal:
        span_vals = [
            np.abs(
                (
                    np.max(np.loadtxt(sorted_files[i])[:, x_index])
                    - np.min(np.loadtxt(sorted_files[i])[:, x_index])
                )
            )
            for i in range(len(sorted_files))
        ]
    else:
        span_vals = [
            np.abs(
                (
                    np.max(
                        np.loadtxt(sorted_files[i])[
                            np.argmax(np.loadtxt(sorted_files[i])[:, nodes // 2]), 1:
                        ]
                    )
                    - np.min(
                        np.loadtxt(sorted_files[i])[
                            np.argmax(np.loadtxt(sorted_files[i])[:, nodes // 2]), 1:
                        ]
                    )
                )
            )
            for i in range(len(sorted_files))
        ]
        

    return span_vals

def calc_legett(psi_data, x_vals, nodes, t_index = None):

    if t_index == None:
        # Find the index of maximum value in the middle column
        max_index = np.argmax(psi_data[:, nodes // 2])
        # Extract the relevant row
        row_data = psi_data[max_index, 1:]
    else:
        row_data = psi_data[t_index, 1:]

    recip_psi_vals_at_t = 1 / row_data
    L = np.abs(x_vals[-1] - x_vals[0]) 
    N = integrate.simps(row_data, x_vals)
    Q_0 = 1 / (integrate.simps(recip_psi_vals_at_t, x_vals) * (N / L**2))

    return Q_0 

def calc_legett_chng_p0(sorted_files, x_vals, nodes):

    legett_vals = [calc_legett(np.loadtxt(sorted_files[i]), x_vals, nodes) for i in range(len(sorted_files))]    

    return legett_vals 


def calc_oscill_omega(psi_data, x_vals, nodes, delta, R, p0, b0, gambar, decay_rate, t_index = None):

    if t_index == None:
        # Find the index of maximum value in the middle column
        max_index = np.argmax(psi_data[:, nodes // 2])
        # Extract the relevant row
        row_data = psi_data[max_index, 1:]
    else:
        row_data = psi_data[t_index, 1:]

    fft_psi_vals = np.abs(np.fft.fft(row_data, norm="forward")[:len(row_data)//2])
    k_vals = np.fft.fftfreq(len(x_vals), np.diff(x_vals)[0])

    abs_n_qc = ft_utils.find_first_harmonic(fft_psi_vals, k_vals)[1]

    omega = np.sqrt((decay_rate**2 * R * p0 * b0 * gambar) / constants.hbar)

    return omega


def calc_oscill_omega_vals(sorted_files, sorted_p0_vals, x_vals, nodes, delta, R, b0, gambar, decay_rate):

    oscill_omega_vals = []
        
    for i in range(len(sorted_files)):
        data = np.loadtxt(sorted_files[i])
        p0 = sorted_p0_vals[i]
        
        omega = calc_oscill_omega(data, x_vals, nodes, delta, R, p0, b0, gambar, decay_rate)
        oscill_omega_vals.append(omega)

    return oscill_omega_vals
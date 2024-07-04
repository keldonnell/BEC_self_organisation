
import numpy as np
from scipy.signal import find_peaks
from scipy import integrate, stats


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
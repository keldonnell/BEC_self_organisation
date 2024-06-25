
# Plots graphs relating to different aspects of the temproal ft for different pump parameters at a specific x coord of the density (|psi|^2) data

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import stats
from scipy.signal import find_peaks


# fname = raw_input("Enter filename: ")
plt.rcParams["ps.usedistiller"] = (
    "xpdf"  # improves quality of .eps figures for use with LaTeX
)

parser = argparse.ArgumentParser(description="")

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


args = parser.parse_args()


output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s.out"
psi_dir = output_dir + "psi.out"
seed_dir = input_dir + "seed.in"


# Read input data from file
def readinput():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")  # load input data file

    nodes = data0[0].astype(int)
    maxt = data0[1]
    ht = data0[2]
    width_psi = data0[3]
    p0 = data0[4]
    Delta = data0[5]
    gambar = data0[6]
    b0 = data0[7]
    num_crit = data0[8]
    R = data0[9]
    gbar = data0[10]
    v0 = data0[11]
    plotnum = data0[12].astype(int)

    return (
        nodes,
        maxt,
        ht,
        width_psi,
        p0,
        Delta,
        gambar,
        b0,
        num_crit,
        R,
        gbar,
        v0,
        plotnum,
    )


nodes, maxt, ht, width_psi, p0, Delta, gambar, b0, num_crit, R, gbar, v0, plotnum = (
    readinput()
)


def find_log_exponent(x, y):

    # Ensure x is positive (domain of log function)
    x = np.array(x)
    y = np.array(y)
    x = x[x > 0]
    y = y[:len(x)] 

    # Transform x data
    x_log = np.log(x)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y)

    # The exponent is the slope of the linear regression
    exponent = slope

    # Calculate the base of the logarithm (e^intercept)
    base = np.exp(intercept)


    return exponent, base, r_value**2

def find_first_harmonic(fft_data, fft_freq_data):
    
    # Find all peaks
    peaks, _ = find_peaks(fft_data)
    
    # Get peak amplitudes
    peak_amplitudes = fft_data[peaks]
    
    # Sort peak indices based on their amplitudes in descending order
    sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]
    
    # The second highest peak should be the 1st harmonic
    first_harmonic_index = peaks[sorted_peak_indices[1]]
    first_harmonic_amplitude = fft_data[first_harmonic_index]
    first_harmonic_frequency = fft_freq_data[first_harmonic_index]
    
    return first_harmonic_frequency, first_harmonic_amplitude



x = float(args.xpos)
x_index = int(
    (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
)

data1_files = glob.glob(output_dir + "psi*")

t_vals = np.linspace(0, maxt, plotnum)
freq_vals = np.abs(np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0]))
freq_vals = freq_vals[:len(freq_vals//2)]

first_mode_ft_peaks_amp = []
first_mode_ft_peaks_freq = []
    
for file in data1_files:
    # Load the file
    data = np.loadtxt(file)

    psi_vals_at_x = data[:, x_index]
    fft_psi_vals = np.abs(np.fft.fft(psi_vals_at_x))
    fft_psi_vals = fft_psi_vals[:len(fft_psi_vals//2)]
    
    first_harmonic_frequency, first_harmonic_amplitude = find_first_harmonic(fft_psi_vals, freq_vals)
    first_mode_ft_peaks_amp.append(first_harmonic_amplitude)
    first_mode_ft_peaks_freq.append(first_harmonic_frequency)


p0_vals = np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", data1_files[j])[0])
        for j in range(len(data1_files))
    ]
)

# Plotting the graph
fig, ax = plt.subplots(2, 3, figsize=(22, 14))
ax[0, 0].set_title(r"1st harmonic amplitude for differnent p0 at x = " + str(x))
ax[0, 0].set_xlabel(r"$p_0$", fontsize=14)
ax[0, 0].set_ylabel(r"1st harmonic amplitude", fontsize=14)
ax[0, 0].scatter(p0_vals, first_mode_ft_peaks_amp)

ax[0, 1].set_title(r"1st harmonic frequency for differnent p0 at x = " + str(x))
ax[0, 1].set_xlabel(r"$p_0$", fontsize=14)
ax[0, 1].set_ylabel(r"1st harmonic frequency", fontsize=14)
ax[0, 1].scatter(p0_vals, first_mode_ft_peaks_freq)

""" 
ax[0, 2].set_title(r"Span of $|\psi|^2$ at x = " + str(x))
ax[0, 2].set_xlabel(r"$p_0$", fontsize=14)
ax[0, 2].set_ylabel(r"Span $[|\psi|^2]$", fontsize=14)
ax[0, 2].scatter(p0_vals, span_vals)
"""


exponent, base, r_srd = find_log_exponent(p0_vals, first_mode_ft_peaks_amp)
print(f"1st mode ft peak: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 0].scatter(p0_vals, first_mode_ft_peaks_amp, label='Data')

x_smooth = np.linspace(p0_vals.min(), p0_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel(r'$p_0$')
ax[1, 0].set_ylabel(r'1st harmonic amplitude')
ax[1, 0].set_title(f'Logarithmic Regression of 1st harmonic amplitude (y = {slope:.2f} * log(x) + {intercept:.2f})')
ax[1, 0].legend()
ax[1, 0].grid(True)


ax[1, 1].scatter(p0_vals, first_mode_ft_peaks_freq, label='Data')

x_smooth = np.linspace(p0_vals.min(), p0_vals.max(), 200)
slope, intercept, r_value, p_value, std_err = stats.linregress(p0_vals, first_mode_ft_peaks_freq)
y_smooth = slope * x_smooth + intercept

print(f"1st mode ft frequency: slope = {slope}, intercept = {intercept}, r-value = {r_value}")

ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 1].set_xlabel(r'$p_0$')
ax[1, 1].set_ylabel(r'1st harmonic frequency')
ax[1, 1].set_title(f'Linear Regression of 1st harmonic amplitude (y = {slope:.2f} * x + {intercept:.2f})')
ax[1, 1].legend()
ax[1, 1].grid(True)

"""
exponent, base, r_srd = find_log_exponent(p0_vals, span_vals)
print(f"Standard Deviation: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 2].scatter(p0_vals, span_vals, label='Data')

x_smooth = np.linspace(p0_vals.min(), p0_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 2].set_xscale('log')
ax[1, 2].set_xlabel(r'$p_0$')
ax[1, 2].set_ylabel(r'Span$[|\psi^2|]$')
ax[1, 2].set_title(f'Logarithmic Regression of span (y = {slope:.2f} * log(x) + {intercept:.2f})')
ax[1, 2].legend()
ax[1, 2].grid(True) """

plt.show()


# Plots graphs relating to different aspects of the temproal ft for different pump parameters at a specific x coord of the density (|psi|^2) data

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import integrate, stats
import fourier_utils as fourier_utils
import standard_data_utils as stand_utils


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

def extract_float(filename):
    match = re.search(r'psi\d+_([\d.e-]+)', filename)
    if match:
        # Remove trailing period if it exists
        float_str = match.group(1).rstrip('.')
        return float(float_str)
    return 0

data1_files = glob.glob(output_dir + "psi*")
sorted_files = sorted(data1_files, key=extract_float)
x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)
k_vals = np.fft.fftfreq(len(x_vals), np.diff(x_vals)[0])


p_th = (2 * gambar) / (b0 * R)

p0_vals = np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", sorted_files[j])[0])
        for j in range(len(sorted_files))
    ]
)

p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
p0_shift_vals = (p0_above_th_vals / p_th) - 1 

analysed_fourier_data = fourier_utils.analyse_fourier_data(sorted_files_above_th, k_vals, (2 * np.pi * num_crit), False)



# Plotting the graph
fig, ax = plt.subplots(2, 4, figsize=(30, 14))
fig.subplots_adjust(wspace=0.3, hspace=0.45)
ax[0, 0].set_title(r"1st harmonic amplitude for differnent p0", fontsize=8)
ax[0, 0].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 0].set_ylabel(r"1st harmonic amplitude", fontsize=14)
ax[0, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"])

ax[0, 1].set_title(r"1st harmonic frequency for differnent p0", fontsize=8)
ax[0, 1].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 1].set_ylabel(r"1st harmonic frequency", fontsize=14)
ax[0, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"])

ax[0, 2].set_title(r"Area under first harmonic of $|\psi|^2$", fontsize=8)
ax[0, 2].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 2].set_ylabel(r"Area under first harmonic (energy)", fontsize=14)
ax[0, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"])

ax[0, 3].set_title(r"Area under all higher harmonics of $|\psi|^2$", fontsize=8)
ax[0, 3].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 3].set_ylabel(r"Area under all higher harmonics (energy)", fontsize=14)
ax[0, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"])


exponent, base, r_srd = fourier_utils.find_log_exponent(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"])
print(f"1st mode ft peak: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"], label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 0].set_ylabel(r'1st harmonic amplitude')
ax[1, 0].set_title(f'Logarithmic Regression of 1st harmonic amplitude \n (y = {slope:.2f} * log(x) + {intercept:.2f})', fontsize=8)
ax[1, 0].legend()
ax[1, 0].grid(True)


ax[1, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"], label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope, intercept, r_value, p_value, std_err = stats.linregress(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"])
y_smooth = slope * x_smooth + intercept

print(f"1st mode ft frequency: slope = {slope}, intercept = {intercept}, r-value = {r_value}")

ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 1].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 1].set_ylabel(r'1st harmonic frequency')
ax[1, 1].set_title(f'Linear Regression of 1st harmonic frequency \n (y = {slope} * x + {intercept})', fontsize=8)
ax[1, 1].legend()
ax[1, 1].grid(True)


exponent, base, r_srd = fourier_utils.find_log_exponent(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"])
print(f"1st mode ft peak: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"], label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 2].set_xscale('log')
ax[1, 2].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 2].set_ylabel(r'Area under first harmonic (energy)')
ax[1, 2].set_title(f'Logarithmic regression of the area under the first harmonic \n (y = {slope:.2f} * log(x) + {intercept:.2f})', fontsize=8)
ax[1, 2].legend()
ax[1, 2].grid(True)


ax[1, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"], label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope, intercept, r_value, p_value, std_err = stats.linregress(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"])
y_smooth = slope * x_smooth + intercept

print(f"1st mode ft frequency: slope = {slope}, intercept = {intercept}, r-value = {r_value}")

ax[1, 3].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 3].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 3].set_ylabel(r'Area under higher harmonics (energy)')
ax[1, 3].set_title(f'Linear Regression of area under higher harmonics \n (y = {slope} * x + {intercept})', fontsize=8)
ax[1, 3].legend()
ax[1, 3].grid(True)


plt.show()
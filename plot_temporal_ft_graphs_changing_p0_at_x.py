
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


def extract_float(filename):
    match = re.search(r'psi\d+_([\d.e-]+)', filename)
    if match:
        # Remove trailing period if it exists
        float_str = match.group(1).rstrip('.')
        return float(float_str)
    return 0

x = float(args.xpos)
x_index = int(
    (np.abs(x + np.pi * num_crit) / (2 * np.pi * num_crit)) * nodes
)

data1_files = glob.glob(output_dir + "psi*")
sorted_files = sorted(data1_files, key=extract_float)
t_vals = np.linspace(0, maxt, plotnum)
freq_vals = np.fft.fftfreq(len(t_vals), np.diff(t_vals)[0])

print(len(np.loadtxt(sorted_files[0])[:, 10]))

p_th = (2 * gambar) / (b0 * R)

p0_vals = np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", sorted_files[j])[0])
        for j in range(len(sorted_files))
    ]
)

p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
p0_shift_vals = (p0_above_th_vals / p_th) - 1 

analysed_fourier_data = fourier_utils.analyse_fourier_data(sorted_files_above_th, freq_vals, 1, True, x_index)


# Plotting the graph
fig, ax = plt.subplots(2, 4, figsize=(30, 14))
fig.subplots_adjust(wspace=0.3, hspace=0.45)
ax[0, 0].set_title(r"1st harmonic amplitude for differnent p0 at x = " + str(x), fontsize=8)
ax[0, 0].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 0].set_ylabel(r"1st harmonic amplitude", fontsize=14)
ax[0, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"])

ax[0, 1].set_title(r"1st harmonic frequency for differnent p0 at x = " + str(x), fontsize=8)
ax[0, 1].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 1].set_ylabel(r"1st harmonic frequency", fontsize=14)
ax[0, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"])

ax[0, 2].set_title(r"Area under first harmonic of $|\psi|^2$ at x = " + str(x), fontsize=8)
ax[0, 2].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 2].set_ylabel(r"Area under first harmonic (energy)", fontsize=14)
ax[0, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"])

ax[0, 3].set_title(r"Area under all higher harmonics of $|\psi|^2$ at x = " + str(x), fontsize=8)
ax[0, 3].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 3].set_ylabel(r"Area under all higher harmonics (energy)", fontsize=14)
ax[0, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"])


final_fit_index = 15
exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals[:final_fit_index], analysed_fourier_data["first_mode_ft_peaks_amp"][:final_fit_index])
print(f"1st mode ft peak: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 0].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_amp"], label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), len(p0_shift_vals))[:final_fit_index]
y_smooth = coefficient * x_smooth**exponent

ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 0].set_ylabel(r'1st harmonic amplitude')
ax[1, 0].set_title(f'Logarithmic Regression of 1st harmonic amplitude \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8)
ax[1, 0].legend()
ax[1, 0].grid(True)


final_fit_index = 15
exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals[:final_fit_index], analysed_fourier_data["first_mode_ft_peaks_freq"][:final_fit_index])
print(f"1st mode ft frequency: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 1].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peaks_freq"], label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), len(p0_shift_vals))[:final_fit_index]
y_smooth = coefficient * x_smooth**exponent

ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 1].set_xscale('log')
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 1].set_ylabel(r'1st harmonic frequency')
ax[1, 1].set_title(f'Logarithmic Regression of 1st harmonic frequency \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8)
ax[1, 1].legend()
ax[1, 1].grid(True)


final_fit_index = 15
exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals[:final_fit_index], analysed_fourier_data["first_mode_ft_peak_area"][:final_fit_index])
print(f"Area under first mode: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 2].scatter(p0_shift_vals, analysed_fourier_data["first_mode_ft_peak_area"], label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), len(p0_shift_vals))[:final_fit_index]
y_smooth = coefficient * x_smooth**exponent


ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 2].set_xscale('log')
ax[1, 2].set_yscale('log')
ax[1, 2].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 2].set_ylabel(r'Area under first harmonic (energy)')
ax[1, 2].set_title(f'Lograithmic Regression of the area under the first harmonic \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8)
ax[1, 2].legend()
ax[1, 2].grid(True)



final_fit_index = 15
exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals[:final_fit_index], analysed_fourier_data["higher_modes_ft_peak_area"][:final_fit_index])
print(f"1st mode ft peak: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 3].scatter(p0_shift_vals, analysed_fourier_data["higher_modes_ft_peak_area"], label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), len(p0_shift_vals))[:final_fit_index]
y_smooth = coefficient * x_smooth**exponent

ax[1, 3].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 3].set_xscale('log')
ax[1, 3].set_yscale('log')
ax[1, 3].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 3].set_ylabel(r'Area under all higher modes (energy)')
ax[1, 3].set_title(f'Logarithmic Regression of area under higher modes \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})', fontsize=8)
ax[1, 3].legend()
ax[1, 3].grid(True)


plt.show()

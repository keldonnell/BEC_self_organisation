# Plots the spatial standard deviation for different pump parameters at a specific time/peak of the density (|psi|^2) data

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import stats
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

p_th = (2 * gambar) / (b0 * R)

p0_vals = np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", sorted_files[j])[0])
        for j in range(len(sorted_files))
    ]
)
x_vals = np.linspace(-np.pi * num_crit, np.pi * num_crit, nodes)

p0_above_th_vals, sorted_files_above_th = stand_utils.find_vals_above_th(p0_vals, sorted_files, p_th)
p0_shift_vals = (p0_above_th_vals / p_th) - 1 

sd_vals = stand_utils.calc_standard_deviation(sorted_files_above_th, nodes, False)
mod_depth_vals = stand_utils.calc_modulation_depth(sorted_files_above_th, nodes, False)
span_vals = stand_utils.calc_span(sorted_files_above_th, nodes, False)
legett_vals = stand_utils.calc_legett_chng_p0(sorted_files_above_th, x_vals, nodes)

# Plotting the graph
fig, ax = plt.subplots(2, 4, figsize=(24, 14))
fig.subplots_adjust(wspace=0.55, hspace=0.3)
ax[0, 0].set_title(r"Spatial standard deviation of $|\psi|^2$")
ax[0, 0].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 0].set_ylabel(r"$\sigma[|\psi|^2]$", fontsize=14)
ax[0, 0].scatter(p0_shift_vals, sd_vals)

ax[0, 1].set_title(r"Spatial modulation depth of $|\psi|^2$")
ax[0, 1].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 1].set_ylabel(r"Modulation depth m$[|\psi|^2]$", fontsize=14)
ax[0, 1].scatter(p0_shift_vals, mod_depth_vals)

ax[0, 2].set_title(r"Spatial span of $|\psi|^2$")
ax[0, 2].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 2].set_ylabel(r"Span $[|\psi|^2]$", fontsize=14)
ax[0, 2].scatter(p0_shift_vals, span_vals)

ax[0, 3].set_title(r"Legett criteria $Q_0$")
ax[0, 3].set_xlabel(r"$\frac{p_0 - p_{th}}{p_{th}}$", fontsize=14)
ax[0, 3].set_ylabel(r"$Q_0$", fontsize=14)
ax[0, 3].scatter(p0_shift_vals, legett_vals)


exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals, sd_vals)
print(f"Standard Deviation: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 0].scatter(p0_shift_vals, sd_vals, label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
y_smooth = coefficient * x_smooth**exponent

ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 0].set_xscale('log')
ax[1, 0].set_yscale('log')
ax[1, 0].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 0].set_ylabel(r'$\sigma[|\psi^2|]$')
ax[1, 0].set_title(f'Logarithmic Regression of s.d. \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})')
ax[1, 0].legend()
ax[1, 0].grid(True)


exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals, mod_depth_vals)
print(f"Modulation Depth: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 1].scatter(p0_shift_vals, mod_depth_vals, label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
y_smooth = coefficient * x_smooth**exponent

ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 1].set_xscale('log')
ax[1, 1].set_yscale('log')
ax[1, 1].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 1].set_ylabel(r'Modulation Depth $m[|\psi^2|]$')
ax[1, 1].set_title(f'Logarithmic Regression of modulation depth \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})')
ax[1, 1].legend()
ax[1, 1].grid(True)


exponent, coefficient, r_srd = stand_utils.find_log_exponent(p0_shift_vals, span_vals)
print(f"Span: exponent = {exponent}, coefficient = {coefficient}, r-squared = {r_srd}")

ax[1, 2].scatter(p0_shift_vals, span_vals, label='Data')

#y = coeff * x^exp
# Generate smooth data for plotting
x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
y_smooth = coefficient * x_smooth**exponent

ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
# Set log scale for both axes
ax[1, 2].set_xscale('log')
ax[1, 2].set_yscale('log')
ax[1, 2].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 2].set_ylabel(r'Span$[|\psi^2|]$')
ax[1, 2].set_title(f'Logarithmic Regression of span \n (Fit: y = {coefficient:.2e} * x^{exponent:.2f})')
ax[1, 2].legend()
ax[1, 2].grid(True)

plt.show()

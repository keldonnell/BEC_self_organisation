# Plots the spatial standard deviation for different pump parameters at a specific time/peak of the density (|psi|^2) data

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import glob
import re
from scipy import stats

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

data1_files = glob.glob(output_dir + "psi*")

sd_vals = [
    np.sqrt(
        np.mean(
            (
                np.loadtxt(data1_files[i])[
                    np.argmax(np.loadtxt(data1_files[i])[:, nodes // 2]), 1:
                ]
                - np.mean(
                    np.loadtxt(data1_files[i])[
                        np.argmax(np.loadtxt(data1_files[i])[:, nodes // 2]), 1:
                    ]
                )
            )
            ** 2
        )
    )
    for i in range(len(data1_files))
]

mod_depth_vals = []
    
for file in data1_files:
    # Load the file only once
    data = np.loadtxt(file)
    
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

span_vals = [
    np.abs(
        (
            np.max(
                np.loadtxt(data1_files[i])[
                    np.argmax(np.loadtxt(data1_files[i])[:, nodes // 2]), 1:
                ]
            )
            - np.min(
                np.loadtxt(data1_files[i])[
                    np.argmax(np.loadtxt(data1_files[i])[:, nodes // 2]), 1:
                ]
            )
        )
    )
    for i in range(len(data1_files))
]

p_th = (2 * gambar) / (b0 * R)

#(p0 - pth) / pth
p0_shift_vals = (np.array(
    [
        float(re.findall(r"(\d+\.\d+e[-+]\d+)", data1_files[j])[0])
        for j in range(len(data1_files))
    ]
) / p_th) - 0.98

# Plotting the graph
fig, ax = plt.subplots(2, 3, figsize=(24, 14))
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



exponent, base, r_srd = find_log_exponent(p0_shift_vals, sd_vals)
print(f"Standard Deviation: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 0].scatter(p0_shift_vals, sd_vals, label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 0].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 0].set_xscale('log')
ax[1, 0].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 0].set_ylabel(r'$\sigma[|\psi^2|]$')
ax[1, 0].set_title(f'Logarithmic Regression of s.d. (y = {slope:.2f} * log(x) + {intercept:.2f})')
ax[1, 0].legend()
ax[1, 0].grid(True)


exponent, base, r_srd = find_log_exponent(p0_shift_vals, mod_depth_vals)
print(f"Standard Deviation: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 1].scatter(p0_shift_vals, mod_depth_vals, label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 1].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 1].set_xscale('log')
ax[1, 1].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 1].set_ylabel(r'Modulation Depth $m[|\psi^2|]$')
ax[1, 1].set_title(f'Logarithmic Regression of modulation depth (y = {slope:.2f} * log(x) + {intercept:.2f})')
ax[1, 1].legend()
ax[1, 1].grid(True)


exponent, base, r_srd = find_log_exponent(p0_shift_vals, span_vals)
print(f"Standard Deviation: exponent = {exponent}, base = {base}, r-squared = {r_srd}")

ax[1, 2].scatter(p0_shift_vals, span_vals, label='Data')

x_smooth = np.linspace(p0_shift_vals.min(), p0_shift_vals.max(), 200)
slope = exponent
intercept = np.log(base)
y_smooth = slope * np.log(x_smooth) + intercept

ax[1, 2].plot(x_smooth, y_smooth, 'r', label='Fitted Curve')
ax[1, 2].set_xscale('log')
ax[1, 2].set_xlabel(r'$\frac{p_0 - p_{th}}{p_{th}}$')
ax[1, 2].set_ylabel(r'Span$[|\psi^2|]$')
ax[1, 2].set_title(f'Logarithmic Regression of span (y = {slope:.2f} * log(x) + {intercept:.2f})')
ax[1, 2].legend()
ax[1, 2].grid(True)

plt.show()

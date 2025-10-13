# -*- coding: utf-8 -*-
"""
Created on Thu Oct 8 09:45:14 2020

@author: Gordon Robb
"""
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-f",
    "--filename",
    metavar="filename",
    required=True,
    help="The name of the file to save to",
)

parser.add_argument(
    "-n",
    "--num_pump_frames",
    metavar="num_pump_frames",
    required=True,
    help="The number of different pump saturation param generated frames",
)

parser.add_argument(
    "-s",
    "--start_pump_param",
    metavar="start_pump_param",
    required=False,
    help="The starting pump saturation param",
)

parser.add_argument(
    "-e",
    "--end_pump_param",
    metavar="end_pump_param",
    required=False,
    help="The ending pump saturation param",
)

parser.add_argument(
    "-i",
    "--index",
    metavar="index",
    required=False,
    help="The index of the p_0 value. This is used for SLURM",
)

args = parser.parse_args()

if int(args.num_pump_frames) > 1 and (
    args.start_pump_param == None or args.end_pump_param == None
):
    raise Exception(
        "If you specify more than one frame you must specify a start and end pump parameter"
    )

output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s"
psi_dir = output_dir + "psi"
seed_dir = input_dir + "seed.in"


import os

if not(os.path.exists(output_dir)) and os.path.exists(input_dir):
	os.mkdir(output_dir)
elif os.path.exists(output_dir) and args.index == None:
	raise Exception("That filename already exits")

	
	


""" # Open new output data files
def openfiles():

    f_s = open(s_dir, "w")
    f_s.close()

    f_psi = open(psi_dir, "w")
    f_psi.close() """


# Read input data from file
def readinput():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")  # load input data file

    nodes = data0[0].astype(int)
    maxt = data0[1]
    ht = data0[2]
    width_psi = data0[3]
    p0 = data0[4]
    Delta = data0[5]
    omega_r = data0[6]
    b0 = data0[7]
    num_crit = data0[8]
    R = data0[9]
    gbar = data0[10]
    v0 = data0[11]
    plotnum = data0[12].astype(int)
    seed = data0[13]
    noise1 = data0[14]
    noise2 = data0[15]
    noise3 = data0[16]

    return (
        nodes,
        maxt,
        ht,
        width_psi,
        p0,
        Delta,
        omega_r,
        b0,
        num_crit,
        R,
        gbar,
        v0,
        plotnum,
        seed,
        noise1,
        noise2,
        noise3
    )


# Initialise variables
def initvars():

    shift = np.pi / 2.0
    L_dom = 2.0 * np.pi * num_crit
    hx = L_dom / np.float32(nodes)
    tperplot = maxt / np.float32(plotnum - 1)
    x = np.linspace(0, L_dom - hx, nodes) - L_dom / 2.0
    y0 = np.complex64(np.exp(-(x**2) / (2.0 * width_psi**2)))
    y0 = y0 * np.exp(1j * v0 * x)

    noise1_vals = (
        np.random.uniform(-1, 1, nodes) * noise1 
    )  
    noise2_vals = (
        np.random.uniform(-1, 1, nodes) * noise2 
    )  
    noise3_vals = (
        np.random.uniform(-1, 1, nodes) * noise3 
    )  
    
    seed_vals = seed * np.cos(x)

    y0 = y0 * (np.ones(nodes) + seed_vals + noise1_vals + noise2_vals)
    norm = hx * np.sum(np.abs(y0) ** 2)
    y0 = y0 / np.sqrt(norm) * np.sqrt(L_dom)


    kx = np.fft.fftfreq(nodes, d=hx) * 2.0 * np.pi

    return shift, L_dom, hx, tperplot, x, y0, kx, noise3_vals


# Write data to output files
def output(t, y, p0, counter):
    name_modifier = ""
    if int(args.num_pump_frames) > 1 or args.index != None:
        name_modifier = str(counter) + "_" + str(p0)

    psi = y

    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * np.abs(psi) ** 2)
    )

    B = calc_B(F, shift)
    s = p0 + np.abs(B) ** 2
    error = hx * np.sum((np.abs(psi)) ** 2) - L_dom
    mod = np.max(s) - np.min(s)

    f_s = open(s_dir + name_modifier + ".out", "a+")
    data = np.concatenate(([t], s))
    np.savetxt(f_s, data.reshape((1, nodes + 1)), fmt="%1.3E", delimiter=" ")
    f_s.close()

    f_psi = open(psi_dir + name_modifier + ".out", "a+")
    data = np.concatenate(([t], np.abs(psi) ** 2))
    np.savetxt(f_psi, data.reshape((1, nodes + 1)), fmt="%1.3E", delimiter=" ")
    f_psi.close()

    progress = np.int32(t / maxt * 100)
    print(
        "Completed "
        + str(progress)
        + " % :  mod = "
        + str(mod)
        + ",  Error ="
        + str(error)
    )

    return t, mod, error


# Integrate kinetic energy part of Schrodinger equation
def propagate_bec(y, tstep):
    psi = y
    psi_k = np.fft.fft(psi)
    psi_k = psi_k * np.exp(-1j * omega_r * kx**2 * tstep)
    psi = np.fft.ifft(psi_k)

    return psi


# Propagate optical field in free space to calculate backward field (B)
def calc_B(F, shift):
    Fk = np.fft.fft(F)
    Bk = np.sqrt(R) * Fk * np.exp(-1j * kx**2 * shift)
    B = np.fft.ifft(Bk)

    return B


# 2nd order Runge-Kutta algorithm
def rk2(t, y, p0):
    yk1 = ht * dy(t, y, p0)
    tt = t + 0.5 * ht
    yt = y + 0.5 * yk1
    yk2 = ht * dy(tt, yt, p0)
    newt = t + ht
    newy = y + yk2

    return newt, newy


# RHS of ODEs for integration of potential energy part of Schrodinger equation
def dy(t, y, p0):
    psi = y
    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * (np.abs(psi)) ** 2)
    )
    B = calc_B(F, shift)
    return -1j * Delta / 4.0 * (p0 + np.abs(B) ** 2) * psi


##########

""" openfiles() """
(
    nodes,
    maxt,
    ht,
    width_psi,
    p0,
    Delta,
    omega_r,
    b0,
    num_crit,
    R,
    gbar,
    v0,
    plotnum,
    seed,
    noise1,
    noise2,
    noise3
) = readinput()
shift, L_dom, hx, tperplot, x, y0, kx, noise3_vals = initvars()

if int(args.num_pump_frames) > 1:
    pump_params = np.linspace(
        float(args.start_pump_param),
        float(args.end_pump_param),
        int(args.num_pump_frames),
    )
else:
    pump_params = [p0]


if args.index == None:
    counter = 0
else:
    counter = int(args.index)


for pump_param in pump_params:
    p0 = pump_param
    y = y0
    t = 0.0
    nextt = tperplot
    ind = 0
    output(t, y, p0, counter)

    while t < maxt:
        
        noise2_vals = (
            np.random.uniform(-1, 1, nodes) * noise2 
        )  
        y = y * (np.ones(nodes) + noise2_vals)

        y = propagate_bec(y, 0.5 * ht)
        t, y = rk2(t, y, p0)
        y = propagate_bec(y, 0.5 * ht)
        if t >= nextt:
            output(t, y, p0, counter)
            ind = ind + 1
            nextt = nextt + tperplot
    counter += 1

    print("Finished " + str(counter) + "/" + str(args.num_pump_frames) + " frames")
print("Finished all frames!")

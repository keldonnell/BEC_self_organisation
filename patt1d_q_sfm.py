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

args = parser.parse_args()

output_dir = "patt1d_outputs/" + args.filename + "/"
input_dir = "patt1d_inputs/" + args.filename + "/"
s_dir = output_dir + "s.out"
psi_dir = output_dir + "psi.out"
seed_dir = input_dir + "seed.in"

if os.path.exists(output_dir):
    raise Exception("That filename already exists")
else:
    if os.path.exists(input_dir):
        os.mkdir(output_dir)
    else:
        raise Exception("There is no input seed file associated with that filename")


# Open new output data files
def openfiles():

    f_s = open(s_dir, "w")
    f_s.close()

    f_psi = open(psi_dir, "w")
    f_psi.close()


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
    noise_val = data0[13]

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
        noise_val,
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
    y0 = y0 * (np.ones(nodes) + 1.0e-6 * np.cos(x))
    norm = hx * np.sum(np.abs(y0) ** 2)
    y0 = y0 / np.sqrt(norm) * np.sqrt(L_dom)
    noise = (
        np.random.random_sample(nodes) * noise_val
    )  # When is it useful to remove the *0.0?
    kx = np.fft.fftfreq(nodes, d=hx) * 2.0 * np.pi

    return shift, L_dom, hx, tperplot, x, y0, noise, kx


# Write data to output files
def output(t, y):
    psi = y
    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * np.abs(psi) ** 2)
        * (np.ones(nodes) + noise)  # Why do we need to add 'noise'?
    )
    B = calc_B(F, shift)
    s = p0 + np.abs(B) ** 2
    error = hx * np.sum((np.abs(psi)) ** 2) - L_dom
    mod = np.max(s) - np.min(s)

    f_s = open(s_dir, "a+")
    data = np.concatenate(([t], s))
    np.savetxt(f_s, data.reshape((1, nodes + 1)), fmt="%1.3E", delimiter=" ")
    f_s.close()

    f_psi = open(psi_dir, "a+")
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
def rk2(t, y):
    yk1 = ht * dy(t, y)
    tt = t + 0.5 * ht
    yt = y + 0.5 * yk1
    yk2 = ht * dy(tt, yt)
    newt = t + ht
    newy = y + yk2

    return newt, newy


# RHS of ODEs for integration of potential energy part of Schrodinger equation
def dy(t, y):
    psi = y
    F = (
        np.sqrt(p0)
        * np.exp(-1j * b0 / (2.0 * Delta) * (np.abs(psi)) ** 2)
        * (np.ones(nodes) + noise)
    )
    B = calc_B(F, shift)
    return -1j * Delta / 4.0 * (p0 + np.abs(B) ** 2) * psi


##########

openfiles()
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
    noise_val,
) = readinput()
shift, L_dom, hx, tperplot, x, y0, noise, kx = initvars()

y = y0
t = 0.0
nextt = tperplot
ind = 0
output(t, y)

while t < maxt:
    y = propagate_bec(y, 0.5 * ht)
    t, y = rk2(t, y)
    y = propagate_bec(y, 0.5 * ht)
    if t >= nextt:
        output(t, y)
        ind = ind + 1
        nextt = nextt + tperplot

print("Finished.")

# q: what does this code do?

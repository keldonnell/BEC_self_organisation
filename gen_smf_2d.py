import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-f", "--filename", metavar="filename", required=True, help="The name of the file to save to")
args = parser.parse_args()

output_dir = "patt2d_outputs/" + args.filename + "/"
input_dir = "patt2d_inputs/" + args.filename + "/"
s_dir = output_dir + "s"
psi_dir = output_dir + "psi"
seed_dir = input_dir + "seed.in"

if os.path.exists(output_dir):
    raise Exception("That filename already exists")
else:
    if os.path.exists(input_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise Exception("There is no input seed file associated with that filename")

def readinput():
    data0 = np.genfromtxt(seed_dir, skip_footer=1, comments="!")
    nodes_x, nodes_y = data0[0].astype(int), data0[1].astype(int)
    maxt = data0[2]
    ht = data0[3]
    width_psi_x, width_psi_y = data0[4], data0[5]
    p0 = data0[6]
    Delta = data0[7]
    omega_r = data0[8]
    b0 = data0[9]
    num_crit = data0[10]
    R = data0[11]
    gbar = data0[12]
    v0_x, v0_y = data0[13], data0[14]
    plotnum = data0[15].astype(int)
    noise_val = data0[16]
    
    return (nodes_x, nodes_y, maxt, ht, width_psi_x, width_psi_y, p0, Delta, omega_r, b0, num_crit, R, gbar, v0_x, v0_y, plotnum, noise_val)

def initvars():
    shift = np.pi / 2.0
    L_dom_x, L_dom_y = 2.0 * np.pi * num_crit, 2.0 * np.pi * num_crit
    hx, hy = L_dom_x / np.float32(nodes_x), L_dom_y / np.float32(nodes_y)
    tperplot = maxt / np.float32(plotnum - 1)
    x = np.linspace(0, L_dom_x - hx, nodes_x) - L_dom_x / 2.0
    y = np.linspace(0, L_dom_y - hy, nodes_y) - L_dom_y / 2.0
    X, Y = np.meshgrid(x, y)
    
    psi_0 = np.complex64(np.exp(-(X**2)/(2.0*width_psi_x**2) - (Y**2)/(2.0*width_psi_y**2)))
    psi_0 = psi_0 * np.exp(1j * (v0_x * X + v0_y * Y))
    psi_0 = psi_0 * (np.ones((nodes_y, nodes_x)) + 1.0e-6 * np.cos(X))
    norm = hx * hy * np.sum(np.abs(psi_0) ** 2)
    psi_0 = psi_0 / np.sqrt(norm) * np.sqrt(L_dom_x * L_dom_y)
    noise = np.random.random((nodes_y, nodes_x)) * noise_val
    kx = np.fft.fftfreq(nodes_x, d=hx) * 2.0 * np.pi
    ky = np.fft.fftfreq(nodes_y, d=hy) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    
    return shift, L_dom_x, L_dom_y, hx, hy, tperplot, X, Y, psi_0, noise, KX, KY

def output(t, psi):
    psi_out = psi
    F = np.sqrt(p0) * np.exp(-1j * b0 / (2.0 * Delta) * np.abs(psi_out) ** 2) * (np.ones((nodes_y, nodes_x)) + noise)
    B = calc_B(F, shift)
    s = p0 + np.abs(B) ** 2
    error = np.sum((np.abs(psi_out)) ** 2) * hx * hy - L_dom_x * L_dom_y
    mod = np.max(s) - np.min(s)

    # Save s and |psi|^2 as 2D arrays for each time step
    np.save(f"{s_dir}_{t:.3f}.npy", s)
    np.save(f"{psi_dir}_{t:.3f}.npy", np.abs(psi_out) ** 2)

    progress = np.int32(t / maxt * 100)
    print(f"Completed {progress}%: mod = {mod}, Error = {error}")

    return t, mod, error

def propagate_bec(psi, tstep):
    psi_k = np.fft.fft2(psi)
    psi_k = psi_k * np.exp(-1j * omega_r * (KX**2 + KY**2) * tstep)
    psi_out = np.fft.ifft2(psi_k)
    return psi_out

def calc_B(F, shift):
    Fk = np.fft.fft2(F)
    Bk = np.sqrt(R) * Fk * np.exp(-1j * (KX**2 + KY**2) * shift)
    B = np.fft.ifft2(Bk)
    return B

def rk2(t, psi):
    yk1 = ht * dy(t, psi)
    tt = t + 0.5 * ht
    yt = psi + 0.5 * yk1
    yk2 = ht * dy(tt, yt)
    newt = t + ht
    newy = psi + yk2
    return newt, newy

def dy(t, psi):
    psi_out = psi
    F = np.sqrt(p0) * np.exp(-1j * b0 / (2.0 * Delta) * (np.abs(psi_out)) ** 2) * (np.ones((nodes_y, nodes_x)) + noise)
    B = calc_B(F, shift)
    return -1j * Delta / 4.0 * (p0 + np.abs(B) ** 2) * psi_out

# Main simulation
(nodes_x, nodes_y, maxt, ht, width_psi_x, width_psi_y, p0, Delta, omega_r, b0, num_crit, R, gbar, v0_x, v0_y, plotnum, noise_val) = readinput()
shift, L_dom_x, L_dom_y, hx, hy, tperplot, X, Y, psi_0, noise, KX, KY = initvars()

psi = psi_0
t = 0.0
nextt = tperplot
ind = 0
output(t, psi)

while t < maxt:
    psi = propagate_bec(psi, 0.5 * ht)
    t, psi = rk2(t, psi)
    psi = propagate_bec(psi, 0.5 * ht)
    if t >= nextt:
        output(t, psi)
        ind = ind + 1
        nextt = nextt + tperplot
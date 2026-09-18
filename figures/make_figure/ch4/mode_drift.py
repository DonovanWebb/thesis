import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import graph_style

graph_style.set_graph_style(1.0)


def unpack_data(path):
    """Download data from Grafana with "Formatted data" disabled."""
    df = pd.read_csv(path, header=[0])
    df = df[df["artiq.mean"].notna()]
    df["Time (s)"] = (df["Time"] - df["Time"].min()) * 1e-3

    t = df["Time (s)"].values
    f = df["artiq.mean"].values * 1e6
    return t, f


def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def format_with_uncertainty(value, error):
    error_magnitude = int(np.floor(np.log10(abs(error))))
    error_digit = int(round(error / 10**error_magnitude))
    if error_digit == 10:
        error_digit = 1
        error_magnitude += 1
    value_rounded = round(value / 10**error_magnitude) * 10**error_magnitude
    precision = max(0, -error_magnitude)

    if error_magnitude > 0:
        error_display = error_digit * 10**error_magnitude
    else:
        error_display = error_digit

    return f"{value_rounded:.{precision}f}({error_display})"


# path = r"../make_figure/Tickle mode frequency-data-2025-05-15 19_19_54.csv"
# date_reference = "2025-05-10"
# t_scale = 15

path = r"../make_figure/Tickle mode frequency-data-2025-05-28 21_01_20.csv"
date_reference = "2025-05-28"
t_scale = 10

t, f = unpack_data(path)
t = t[50:] - 50 * t_scale
f = f[50:]


"""Each point takes t_scale seconds. Assume we calibrate every 10 minutes"""
calib_time = 600 / t_scale
pfit = np.array([f[int(x // calib_time * calib_time)] for x in range(len(t))])

"""Fourier transform to see spectrum"""
F = np.fft.fft(f)
freqs = np.fft.fftfreq(len(t), d=t[1] - t[0])  # frequency bins
"""
plt.figure("FFT")

mask = freqs >= 0
freqs_pl = freqs[mask]
plt.plot(freqs_pl, np.abs(F)[mask])
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Amplitude")
plt.xlabel("Frequency / Hz")
"""

# Low-pass filter: zero out frequencies above cutoff
cutoff = 1.67e-3  # Hz # Need an estimate of thermal frequency cutoff
F_lowpass = F.copy()
F_lowpass[np.abs(freqs) > cutoff] = 0

# Inverse FFT to get lowpass signal
f_lowpass = np.fft.ifft(F_lowpass).real
fig = plt.figure("Time scans")
fig_width = graph_style.get_fig_width()
fig_height = 1.5 * fig_width / graph_style.get_phi()
fig.set_size_inches(fig_width, fig_height)


ax0 = plt.subplot2grid(shape=(100, 100), loc=(0, 10), colspan=87, rowspan=38)
ax1 = plt.subplot2grid(shape=(100, 100), loc=(56, 10), colspan=40, rowspan=40)
ax2 = plt.subplot2grid(shape=(100, 100), loc=(56, 57), colspan=40, rowspan=40)

ax = [ax0, ax1, ax2]

ax[0].plot(t, f * 1e-6, label="Data", alpha=0.9, ls="-")
ax[0].plot(
    t[10:-10],
    f_lowpass[10:-10] * 1e-6,
    label="Low passed",
    alpha=graph_style.get_alpha(),
)
ax[0].plot(t, pfit * 1e-6, alpha=graph_style.get_alpha(), label="10 min calibration")
ax[0].set_xlabel("Time stamp (s)")
ax[0].set_ylabel("Frequency (MHz)")
ax[0].set_xlim(0, t[-1])
ax[0].legend(fontsize=8, loc="upper left")

green = graph_style.get_color(1)
red = graph_style.get_color(2)


#  residues
def get_residue(residue, axis):
    residue = residue / 1000  # convert to kHz
    rang = np.max(np.abs(residue)) * np.array([-1, 1])
    mu = np.mean(residue)
    sigma = np.std(residue)
    A = 1 / (sigma * np.sqrt(2 * np.pi))

    n = len(residue)
    mu_err = sigma / np.sqrt(n)
    sigma_err = sigma / np.sqrt(2 * (n - 1))

    print(f"sigma = {sigma:.2f} +/- {sigma_err:.2f} Hz")
    mu_formatted = format_with_uncertainty(mu, mu_err)
    sigma_formatted = format_with_uncertainty(sigma, sigma_err)

    x = np.linspace(rang[0], rang[1], 1000)
    y = gaussian(x, A, mu, sigma)

    color = green if axis == 2 else red

    nbins = 40
    ax[axis].hist(
        residue,
        bins=nbins,
        range=rang,
        density=True,
        alpha=0.8,
        label=f"Residuals \nnbins={nbins}\n",
    )
    ax[axis].plot(
        x,
        y,
        alpha=0.7,
        label=f"Gaussian fit\n$\mu$={mu_formatted} Hz\n$\sigma$={sigma_formatted} Hz",
        color=color,
    )
    # ax[axis].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax[axis].set_xlabel("Residual (kHz)")


ax2.yaxis.tick_right()
ax[1].set_ylabel("Frequency Density")

residue_calib = f - pfit
residue_thermal = f - f_lowpass

ax1.set_title("Calibration every 10 min", fontsize=10)
ax2.set_title("Thermal drift (10 min cutoff)", fontsize=10)
# add text to the top right corner with residual width
textstr = f"$\sigma =$ {format_with_uncertainty(np.std(residue_calib), np.std(residue_calib) / np.sqrt(len(residue_calib)))} Hz"
ax1.text(
    0.95,
    0.95,
    textstr,
    transform=ax1.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)

textstr = f"$\sigma =$ {format_with_uncertainty(np.std(residue_thermal), np.std(residue_thermal) / np.sqrt(len(residue_thermal)))} Hz"
ax2.text(
    0.95,
    0.95,
    textstr,
    transform=ax2.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)


print("Calibration mode residuals:")
get_residue(residue_calib, 1)
print("Thermal mode residuals:")
get_residue(residue_thermal, 2)

# label top left corner with "a", "b", "c"
ax[0].text(
    -0.10,
    1.05,
    "a",
    transform=ax[0].transAxes,
    fontweight="bold",
    fontsize=14,
)
ax[1].text(
    -0.19,
    1.05,
    "b",
    transform=ax[1].transAxes,
    fontweight="bold",
    fontsize=14,
)
ax[2].text(
    -0.08,
    1.05,
    "c",
    transform=ax[2].transAxes,
    fontweight="bold",
    fontsize=14,
)

# plt.show()
plt.savefig("mode_drift.pdf")

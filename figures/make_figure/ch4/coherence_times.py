import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter

graph_style.set_graph_style(1.0)


def gaussian_func(x, a, sigma):
    return a * np.exp(-((x) ** 2) / (sigma**2))


def exp_func(x, a, sigma):
    return a * np.exp(-(x) / (sigma))


date = "2025-06-05"
rid_dict = {36415: "FNC", 36419: "No FNC"}

fig = plt.figure()
for rid in rid_dict.keys():
    dict = graph_style.load_data(rid, date)
    delays = (
        np.array(dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"]) * 1e3
    )  # convert to us
    contrast = np.array(
        dict["datasets"][f"ndscan.rid_{rid}.points.channel_scan_ion_phase_contrast"]
    )
    contrast_err = np.array(
        dict["datasets"][f"ndscan.rid_{rid}.points.channel_scan_ion_phase_contrast_err"]
    )
    argsort = np.argsort(delays)
    # remove second last point, which is a large outlier
    argsort = np.delete(argsort, -2)
    delays = delays[argsort]
    contrast = contrast[argsort]
    contrast_err = contrast_err[argsort]

    # ----------- gauss func
    initial_guess = [1, 2]

    popt_ramsey, pcov_ramsey = curve_fit(
        gaussian_func, delays, contrast, sigma=contrast_err, p0=initial_guess
    )
    p_err = np.sqrt(np.diag(pcov_ramsey))

    xplot = np.linspace(0, np.max(delays), 1000)
    num = rid % 3 - 1
    c = graph_style.get_color(num)
    plt.errorbar(delays, contrast, yerr=contrast_err, fmt="^", color=c, elinewidth=1.0)
    plt.plot(xplot, gaussian_func(xplot, *popt_ramsey), color=c, alpha=0.5)
    # add v line at coherence time, that goes to height of the fit
    height = gaussian_func(popt_ramsey[1], *popt_ramsey)
    plt.vlines(
        popt_ramsey[1],
        0,
        height,
        color=c,
        linestyle=":",
        label=f"{rid_dict[rid]} $\\tau$ = {popt_ramsey[1]:.1f}({p_err[1]*10:.0f}) ms",
        alpha=0.75,
    )
    # horizontal line at 1/e that ends at the fit line

    plt.hlines(
        1 / np.e,
        0,
        popt_ramsey[1],
        color="black",
        linestyle=":",
        alpha=0.5,
    )


plt.ylabel("Contrast")
plt.xlabel("Ramsey delay $t_{\\rm RAM}$ (ms)")
plt.xlim(0, np.max(delays))
plt.ylim(0, 1)
plt.grid()
plt.legend()
# label figure text "a" in the top left corner
plt.text(
    -0.10,
    1.0,
    "a",
    transform=plt.gca().transAxes,
    fontsize=14,
    fontweight="bold",
)
plt.savefig("fnc_coherence.pdf")

# ----------------------
""" repeat for mag field"""

date = "2025-06-05"
rid_dict = {36418: "+3/2", 36412: "-5/2", 36415: "-3/2"}

fig, ax1 = plt.subplots()

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.65, 0.45, 0.25, 0.2]
ax2 = fig.add_axes([left, bottom, width, height])


for rid in rid_dict.keys():
    dict = graph_style.load_data(rid, date)
    delays = (
        np.array(dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"]) * 1e3
    )  # convert to us
    contrast = np.array(
        dict["datasets"][f"ndscan.rid_{rid}.points.channel_scan_ion_phase_contrast"]
    )
    contrast_err = np.array(
        dict["datasets"][f"ndscan.rid_{rid}.points.channel_scan_ion_phase_contrast_err"]
    )
    argsort = np.argsort(delays)
    if rid == 36412:
        argsort = np.delete(argsort, 1)
    elif rid == 36415:
        argsort = np.delete(argsort, -2)
    delays = delays[argsort]
    contrast = contrast[argsort]
    contrast_err = contrast_err[argsort]

    # ----------- gauss func
    initial_guess = [1, 2]

    popt_ramsey, pcov_ramsey = curve_fit(
        gaussian_func, delays, contrast, sigma=contrast_err, p0=initial_guess
    )
    p_err = np.sqrt(np.diag(pcov_ramsey))

    xplot = np.linspace(0, np.max(delays), 1000)
    num = (rid - 3) % 4
    c = graph_style.get_color(num)
    ax1.errorbar(delays, contrast, yerr=contrast_err, fmt="^", color=c, elinewidth=1.0)
    ax1.plot(xplot, gaussian_func(xplot, *popt_ramsey), color=c, alpha=0.5)
    # add v line at coherence time, that goes to height of the fit
    height = gaussian_func(popt_ramsey[1], *popt_ramsey)
    ax1.vlines(
        popt_ramsey[1],
        0,
        height,
        color=c,
        linestyle=":",
        label=r"$|3D_{5/2},~m_j =$ "
        + f"{rid_dict[rid]}$\\rangle$ $\\tau$ = {popt_ramsey[1]:.1f}({p_err[1]*10:.0f}) ms",
        alpha=0.75,
    )
    # horizontal line at 1/e that ends at the fit line

    ax1.hlines(
        1 / np.e,
        0,
        popt_ramsey[1],
        color="black",
        linestyle=":",
        alpha=0.5,
    )


ax1.set_ylabel("Contrast")
ax1.set_xlabel("Ramsey delay $t_{\\rm RAM}$ (ms)")
ax1.set_xlim(0, np.max(delays) + 2)
ax1.set_ylim(0, 1)
ax1.grid()
ax1.legend()

# -------
"""
Plot coherence time vs magnetic field sensitivity
"""
mag_sens = [+0.624, -0.446, -0.178]
mag_sens = np.abs(mag_sens)

coherence_times = [1.568, 2.257, 6.179]
coherence_times_err = [0.166, 0.301, 0.298]

c = "black"
ax2.errorbar(
    mag_sens,
    coherence_times,
    yerr=coherence_times_err,
    fmt="^",
    color=c,
    elinewidth=1.0,
)


# linear fit
def linear_fit(x, a, b):
    return a * x + b


p_fit, p_cov = curve_fit(
    linear_fit,
    mag_sens,
    coherence_times,
    sigma=coherence_times_err,
    p0=[-1, 10],
)
p_err = np.sqrt(np.diag(p_cov))
ax2.plot(
    mag_sens,
    linear_fit(mag_sens, *p_fit),
    color=c,
    alpha=0.75,
)

ax2.set_xlabel("B-field sensitivity (MHz/G)", fontsize=9)
ax2.set_ylabel("$\\tau$ (ms)", fontsize=9)
ax1.text(
    -0.10,
    0.98,
    "b",
    transform=ax1.transAxes,
    fontsize=14,
    fontweight="bold",
)
plt.savefig("mag_coherence.pdf")

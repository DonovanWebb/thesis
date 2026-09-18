import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter

graph_style.set_graph_style(0.75)


def gaussian_func(x, a, sigma):
    return a * np.exp(-((x) ** 2) / (sigma**2))


def exp_func(x, a, sigma):
    return a * np.exp(-(x) / (sigma))


date = "2025-06-06"
rid = 36458

fig = plt.figure()
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
delays = delays[argsort]
contrast = contrast[argsort]
contrast_err = contrast_err[argsort]

c = graph_style.get_color(0)
plt.errorbar(delays, contrast, yerr=contrast_err, fmt="^", color=c, elinewidth=1.0)


# ----------- exp_func func
def exp_plot():
    initial_guess = [1, 2]

    popt_ramsey, pcov_ramsey = curve_fit(
        exp_func, delays, contrast, sigma=contrast_err, p0=initial_guess
    )
    p_err = np.sqrt(np.diag(pcov_ramsey))

    xplot = np.linspace(0, np.max(delays), 1000)
    c = graph_style.get_color(1)
    plt.plot(
        xplot,
        exp_func(xplot, *popt_ramsey),
        color=c,
        alpha=0.5,
        label="Exponential fit",
    )
    # add v line at coherence time, that goes to height of the fit
    height = exp_func(popt_ramsey[1], *popt_ramsey)
    plt.vlines(
        popt_ramsey[1],
        0,
        height,
        color=c,
        linestyle=":",
        label=f"$\\tau$ = {popt_ramsey[1]:.1f}({p_err[1]*10:.0f}) ms",
        alpha=0.75,
    )
    # horizontal line at 1/e that ends at the fit line

    plt.hlines(
        popt_ramsey[0] / np.e,
        0,
        popt_ramsey[1],
        color="black",
        linestyle=":",
        alpha=0.5,
    )
    residuals = contrast - exp_func(delays, *popt_ramsey)
    sigma_residuals = np.std(residuals)

    return sigma_residuals


def gauss_plot():
    initial_guess = [1, 2]

    popt_ramsey, pcov_ramsey = curve_fit(
        gaussian_func, delays, contrast, sigma=contrast_err, p0=initial_guess
    )
    p_err = np.sqrt(np.diag(pcov_ramsey))

    xplot = np.linspace(0, np.max(delays), 1000)
    c = graph_style.get_color(2)
    plt.plot(
        xplot,
        gaussian_func(xplot, *popt_ramsey),
        color=c,
        alpha=0.5,
        label="Gaussian fit",
    )
    residuals = contrast - gaussian_func(delays, *popt_ramsey)
    # get stadard deviation of residuals
    sigma_residuals = np.std(residuals)

    return sigma_residuals


print(f"Gaussian residuals: {gauss_plot()}")
print(f"Exponential residuals: {exp_plot()}")


plt.ylabel("Contrast")
plt.xlabel("Ramsey delay $t_{\\rm RAM}$ (ms)")
plt.xlim(-0.1, np.max(delays) + 0.1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
# plt.show()
plt.savefig("motional_coherence.pdf")

import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


def decaying_sine(t, A, lambda_, W):
    return 1 / 2 + A * np.exp(-lambda_ * t) * np.cos(W * t / 2) / 2


graph_style.set_graph_style(1.0)

date = "2025-02-04"
rid = 16238

dict = graph_style.load_data(rid, date)
# for x in dict["datasets"].keys():
# print(x)

duration = np.array(dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"])
pop = np.array(
    dict["datasets"][f"ndscan.rid_{rid}.points.channel_measurement_camera_readout_p"]
)
pop_err = np.array(
    dict["datasets"][
        f"ndscan.rid_{rid}.points.channel_measurement_camera_readout_p_err"
    ]
)
argsort = np.argsort(duration)
duration = duration[argsort]
pop = pop[argsort]
pop_err = pop_err[argsort]

p_fit, p_cov = curve_fit(
    decaying_sine,
    duration,
    pop,
    p0=[1, 1e6, 2 * np.pi * 1e5],
    bounds=([0, 0, 0], [1.1, 2 * np.pi * 2e5, 2 * np.pi * 1e7]),
    sigma=pop_err,
)

p_err = np.sqrt(np.diag(p_cov))
print(
    f"Omega = 2pi * { p_fit[2] / (2 * np.pi*1e6):.4f} +/- {p_err[2] / (2 * np.pi*1e6):.4f} MHz"
)
print(f"Initial Amp = {p_fit[0]:.2f} +/- {p_err[0]:.2f}")
print(f"Decay rate = {p_fit[1]/1e6:.4f} +/- {p_err[1]/1e6:.4f} 1/$\\mu$ s")

fig = plt.figure()

fig_width = graph_style.get_fig_width()
fig_height = fig_width / graph_style.get_phi() / 2
fig.set_size_inches(fig_width, fig_height)

c = graph_style.get_color(0)
plt.errorbar(
    duration * 1e6, pop, yerr=pop_err, fmt="^", color=c, zorder=11, elinewidth=1.0
)
linrange = np.linspace(
    0,
    np.max(duration * 1e6),
    5000,
)
plt.plot(
    duration * 1e6,
    decaying_sine(duration, *p_fit),
    color=c,
    zorder=10,
    alpha=0.5,
    label="Fit: A=%.2f, lambda=%.2f, W=%.2f" % tuple(p_fit),
)
plt.xlabel("729-nm Pulse Duration $t$ ($\mu$s)")
plt.ylabel("$P_\\downarrow$")
# plt.show()
plt.savefig("long_flop.pdf")

import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


graph_style.set_graph_style(0.75)

date = "2024-12-13"
rid = 11740

data_dict = graph_style.load_data(rid, date)
# for x in data_dict["datasets"].keys():
# print(x)

delays0 = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"]) * 1000
freqs = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.axis_1"])
flop_durations = np.array(
    data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__axis_0"]
)
nbar = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__nbar"])
nbar_err = np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__nbar_err"])
np.array(data_dict["datasets"][f"ndscan.rid_{rid}.points.channel__spec"])

# Chose all entries with delays1 = f_ax
f_ax = 1.07
delays0 = delays0[freqs == f_ax]
nbar = nbar[freqs == f_ax]
nbar_err = nbar_err[freqs == f_ax]

p_fit, p_cov = curve_fit(graph_style.linear_func, delays0, nbar)

ndot = p_fit[0] * 1000
ndot_err = np.sqrt(p_cov[0][0]) * 1000
print(ndot)
print(ndot_err)

fig, ax = plt.subplots()
c = graph_style.get_color(0)
ax.errorbar(delays0, nbar, yerr=nbar_err, zorder=11, fmt="^", color=c, elinewidth=1.0)
ax.plot(
    delays0,
    graph_style.linear_func(delays0, *p_fit),
    label="linear heating fit",
    alpha=graph_style.get_alpha(),
    color=c,
    zorder=10,
)
textstr = f"Heating rate = {ndot:.0f}({ndot_err:.0f}) q/s"

ax.text(
    0.05,
    0.95,
    textstr,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="left",
    fontsize=12,
)
ax.set_xlabel("Delay $t_{\\rm delay}$ (ms)")
ax.set_ylabel("Avg. Fock state $\\bar{n}$")

plt.savefig("heating_rate.pdf")

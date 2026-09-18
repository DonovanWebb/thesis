import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


def sinusoid_func(x, a, b, c, d):
    return a * np.sin(b * x + c) + d


graph_style.set_graph_style(1.0)

date = "2025-01-31"
rid_pop = 15960
rid_parity = 15984
rid_pop_f = 15985

pop_dict = graph_style.load_data(rid_pop, date)
popf_dict = graph_style.load_data(rid_pop_f, date)
parity_dict = graph_style.load_data(rid_parity, date)
for x in popf_dict["datasets"].keys():
    print(x)

duration = np.array(pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.axis_0"]) * 1e6
p00 = np.array(pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_00"])
p0110 = np.array(pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_01_10"])
p11 = np.array(pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_11"])
p_err_00 = np.array(
    pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_err_00"]
)
p_err_0110 = np.array(
    pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_err_01_10"]
)
p_err_11 = np.array(
    pop_dict["datasets"][f"ndscan.rid_{rid_pop}.points.channel_p_err_11"]
)

parity = np.array(
    parity_dict["datasets"][f"ndscan.rid_{rid_parity}.points.channel_parity"]
)
parity_err = np.array(
    parity_dict["datasets"][f"ndscan.rid_{rid_parity}.points.channel_parity_err"]
)
phase = np.array(parity_dict["datasets"][f"ndscan.rid_{rid_parity}.points.axis_0"])

p00f = np.array(popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_00"])
p0110f = np.array(popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_01_10"])
p11f = np.array(popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_11"])
p_err_00f = np.array(popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_err_00"])
p_err_0110f = np.array(
    popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_err_01_10"]
)
p_err_11f = np.array(popf_dict["datasets"][f"ndscan.rid_{rid_pop_f}.point.p_err_11"])

arg_sort = np.argsort(phase)
phase = phase[arg_sort]
parity = parity[arg_sort]
parity_err = parity_err[arg_sort]

p_fit, p_cov = curve_fit(
    sinusoid_func,
    phase,
    parity,
    p0=[1, 1, 0, 0],
    sigma=parity_err,
    absolute_sigma=True,
)

fit_err = np.sqrt(np.diag(p_cov))
print("Amplitude: ", p_fit[0])
print("Amplitude err: ", fit_err[0])
print("Population: ", p0110f)
print("Population err: ", p_err_0110f)

# creating grid for subplots
fig = plt.figure()

fig_width = graph_style.get_fig_width()
fig_height = 1.5 * fig_width / graph_style.get_phi()
fig.set_size_inches(fig_width, fig_height)

ax0 = plt.subplot2grid(shape=(100, 100), loc=(0, 5), colspan=100, rowspan=45)
ax1 = plt.subplot2grid(shape=(100, 100), loc=(65, 5), colspan=80, rowspan=35)
ax2 = plt.subplot2grid(shape=(100, 100), loc=(65, 95), colspan=10, rowspan=35)
ax2.yaxis.tick_right()

ax0.errorbar(
    duration,
    p00,
    label="$P_{00}$",
    fmt="^",
    yerr=p_err_00,
    elinewidth=1.0,
)
ax0.errorbar(
    duration,
    p11,
    label="$P_{11}$",
    fmt="^",
    yerr=p_err_11,
    elinewidth=1.0,
)
ax0.errorbar(
    duration,
    p0110,
    label="$(P_{01} + P_{10})/2$",
    fmt="^",
    yerr=p_err_0110,
    elinewidth=1.0,
)
ax0.legend(loc="upper right", fontsize=8)
ax0.set_ylabel("Populations")
ax0.set_xlabel("Duration $t_g$ (us)")
ax0.set_ylim(0.0, 1.0)
ax0.set_xlim(0.0, 300.0)

ofs = 0.01
ax2.errorbar(
    70 - ofs,
    p00f,
    label="p00 final",
    fmt="^",
    yerr=p_err_00f,
    alpha=0.8,
    capsize=3,
    elinewidth=1.0,
)
ax2.errorbar(
    70 + ofs,
    p11f,
    label="p11 final",
    fmt="^",
    yerr=p_err_11f,
    alpha=0.8,
    capsize=3,
    elinewidth=1.0,
)
ax2.errorbar(
    70,
    p0110f,
    label="p01_10 final",
    fmt="^",
    yerr=p_err_0110f,
    capsize=3,
    elinewidth=1.0,
)
ax2.set_ylabel("Populations")
ax2.set_ylim(0.0, 0.5)
ax2.set_xlim(70 - 2 * ofs, 70 + 2 * ofs)
ax2.set_xticks([])
ax2.set_yticks(np.arange(0.0, 0.7, 0.1))


c = graph_style.get_color(0)
ax1.errorbar(
    phase,
    parity,
    yerr=parity_err,
    color=c,
    zorder=11,
    fmt="^",
    elinewidth=1.0,
)
linrange = np.linspace(
    np.min(phase),
    np.max(phase),
    1000,
)
ax1.plot(
    linrange,
    sinusoid_func(linrange, *p_fit),
    color=c,
    alpha=graph_style.get_alpha(),
    zorder=10,
    label="Parity fit",
)
ax1.set_ylim(-1.0, 1.0)
ax1.set_ylabel("Parity")
ax1.set_xlabel("Phase turns $\\phi$")

ax0.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax2.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
plt.tight_layout()

plt.savefig("ms_gate.pdf")

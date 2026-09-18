import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit
from matplotlib.ticker import FormatStrFormatter


def thermal_split(x, W):
    """
    nbar is initial thermal phonon number
    W is the SDF Rabi frequency
    """
    nbar = 0.03  # measured temperature
    a = W * x / 2
    return 1 / 2 * (1 + np.exp(-4 * (nbar + 1 / 2) * a**2))


def detune_sdf(x, W):
    """
    x is detuning in radians/s
    nbar is initial thermal phonon number
    W is the SDF Rabi frequency in radians/s
    """
    nbar = 0.03  # measured temperature
    t = 100e-6
    a = W * np.sinc(x * t / (2 * np.pi)) * t / 2  # numpy normalised sinc
    return 1 / 2 * (1 + np.exp(-4 * (nbar + 1 / 2) * a**2))


graph_style.set_graph_style(1.0)

date = "2025-05-03"
rid_det = 23502  # detune LS 3.5 kHz
rid_dur = 23503  # duration
rid_tb = 23501  # LS 4 kHz

det_dict = graph_style.load_data(rid_det, date)
dur_dict = graph_style.load_data(rid_dur, date)
tb_dict = graph_style.load_data(rid_tb, date)

# for x in det_dict["datasets"].keys():
# print(x)

duration = np.array(dur_dict["datasets"][f"ndscan.rid_{rid_dur}.points.axis_0"])
pop_dur = np.array(
    dur_dict["datasets"][
        f"ndscan.rid_{rid_dur}.points.channel_measurement_camera_readout_p"
    ]
)
pop_err_dur = np.array(
    dur_dict["datasets"][
        f"ndscan.rid_{rid_dur}.points.channel_measurement_camera_readout_p_err"
    ]
)
argsort_dur = np.argsort(duration)
duration = duration[argsort_dur]
pop_dur = pop_dur[argsort_dur]
pop_err_dur = pop_err_dur[argsort_dur]

detuning = np.array(det_dict["datasets"][f"ndscan.rid_{rid_det}.points.axis_0"])
total_pts = len(detuning)
detuning = detuning[:total_pts]
pop_det = np.array(
    det_dict["datasets"][
        f"ndscan.rid_{rid_det}.points.channel_measurement_camera_readout_p"
    ]
)
pop_det = pop_det[:total_pts]
pop_err_det = np.array(
    det_dict["datasets"][
        f"ndscan.rid_{rid_det}.points.channel_measurement_camera_readout_p_err"
    ]
)
pop_err_det = pop_err_det[:total_pts]
argsort_dur = np.argsort(detuning)
detuning = detuning[argsort_dur]
pop_det = pop_det[argsort_dur]
pop_err_det = pop_err_det[argsort_dur]

detuning_tb = np.array(tb_dict["datasets"][f"ndscan.rid_{rid_tb}.points.axis_0"])
pop_tb = np.array(
    tb_dict["datasets"][
        f"ndscan.rid_{rid_tb}.points.channel_measurement_camera_readout_p"
    ]
)
pop_err_tb = np.array(
    tb_dict["datasets"][
        f"ndscan.rid_{rid_tb}.points.channel_measurement_camera_readout_p_err"
    ]
)

# Fit duration data
p_fit, p_cov = curve_fit(
    thermal_split,
    duration,
    pop_dur,
    p0=[2 * np.pi * 6.37e3],
    sigma=pop_err_dur,
    bounds=([2 * np.pi * 0e3], [2 * np.pi * 100e3]),
)

Wsdf = p_fit[0] / 1000
p_errs = np.sqrt(np.diag(p_cov))
Wsdf_err = p_errs[0] / 1000
print(f"Omega_sdf: 2pi {Wsdf / (2 * np.pi):.3f} +/- {Wsdf_err / (2 * np.pi):.3f} kHz")

# Fit detuning data
p_fit_det, p_cov_det = curve_fit(
    detune_sdf,
    detuning * 2 * np.pi,
    pop_det,
    p0=[2 * np.pi * 6.37e3],
    sigma=pop_err_det,
    bounds=([2 * np.pi * 0e3], [2 * np.pi * 100e3]),
)
Wsdf_det = p_fit_det[0] / 1000
p_errs_det = np.sqrt(np.diag(p_cov_det))
Wsdf_det_err = p_errs_det[0] / 1000
print(
    f"Omega_sdf_det: 2pi {Wsdf_det / (2 * np.pi):.3f} +/- {Wsdf_det_err / (2 * np.pi):.3f} kHz"
)


# creating grid for subplots
fig = plt.figure()

fig_width = graph_style.get_fig_width()
fig_height = 1.5 * fig_width / graph_style.get_phi()
fig.set_size_inches(fig_width, fig_height)

ax0 = plt.subplot2grid(shape=(100, 100), loc=(0, 5), colspan=43, rowspan=45)
ax1 = plt.subplot2grid(shape=(100, 100), loc=(0, 57), colspan=43, rowspan=45)
ax2 = plt.subplot2grid(shape=(100, 100), loc=(65, 5), colspan=95, rowspan=33)

ax0.errorbar(
    detuning / 1000,
    pop_det,
    yerr=pop_err_det,
    fmt="^",
    label="Detuning",
    zorder=11,
    alpha=graph_style.get_alpha(),
    elinewidth=1.0,
)
ax0.set_xlabel("Detuning $\\delta_g$ (kHz)")
ax0.set_xlim(-20, 20)
ax0.set_xticks(
    np.arange(-20, 21, 10),
)
ax0.set_ylabel("Pop. $P_\\downarrow$")
ax0.set_ylim(0.25, 1.0)
ax0.set_yticks(
    np.arange(0.25, 1.25, 0.25),
)
ax0.plot(
    detuning / 1000,
    detune_sdf(detuning * 2 * np.pi, *p_fit_det),
    label="Fit",
    zorder=10,
    alpha=graph_style.get_alpha(),
    ls="-",
)
ax1.errorbar(
    detuning_tb / 1000,
    pop_tb,
    yerr=pop_err_tb,
    fmt="^",
    zorder=11,
    label="Detuning Tone Balance",
    alpha=graph_style.get_alpha(),
    elinewidth=1.0,
)
ax1.plot(
    detuning / 1000,
    detune_sdf(detuning * 2 * np.pi, *p_fit_det),
    label="Fit",
    zorder=11,
    alpha=graph_style.get_alpha(),
    ls="-",
)
ax1.set_xlabel("Detuning $\\delta_g$ (kHz)")
ax1.set_xlim(-20, 20)
ax1.set_xticks(
    np.arange(-20, 21, 10),
)
ax1.set_ylim(0.25, 1.0)
ax1.set_yticks(
    np.arange(0.25, 1.25, 0.25),
)
ax1.yaxis.tick_right()

c = graph_style.get_color(0)
ax2.errorbar(
    duration * 1e6,
    pop_dur,
    yerr=pop_err_dur,
    fmt="^",
    zorder=11,
    label="Duration",
    alpha=graph_style.get_alpha(),
    color=c,
    elinewidth=1.0,
)
ax2.set_ylim(0.25, 1.0)
ax2.set_yticks(
    np.arange(0.25, 1.25, 0.25),
)
ax2.set_xlim(0, 200)
ax2.set_xlabel("Duration $t$ (us)")
ax2.set_ylabel("Pop. $P_\\downarrow$")
ax2.plot(
    duration * 1e6,
    thermal_split(duration, *p_fit),
    label="Fit",
    zorder=10,
    alpha=graph_style.get_alpha(),
    color=c,
)

textstr = (
    "$\Omega_{\\rm SDF}$ = 2$\pi \\times $ "
    + f"{Wsdf / (2 * np.pi):.1f} ({10*Wsdf_err / (2 * np.pi):.0f}) kHz"
)

ax2.text(
    0.95,
    0.95,
    textstr,
    transform=ax2.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)
ax0.text(-0.25, 1.01, "a", transform=ax0.transAxes, size=16, fontweight="bold")
ax1.text(-0.12, 1.01, "b", transform=ax1.transAxes, size=16, fontweight="bold")
ax2.text(-0.11, 1.01, "c", transform=ax2.transAxes, size=16, fontweight="bold")

plt.savefig("sdf.pdf")

import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit


def linear_fit(t, a, b):
    return a * t + b


graph_style.set_graph_style(0.75)

date = "2025-06-05"
rid = 36400

dict = graph_style.load_data(rid, date)
# for x in dict["datasets"].keys():
# print(x)

num_pulses = np.array(dict["datasets"][f"ndscan.rid_{rid}.points.axis_0"])
pop = np.array(
    dict["datasets"][f"ndscan.rid_{rid}.points.channel_measurement_camera_readout_p"]
)
pop_err = np.array(
    dict["datasets"][
        f"ndscan.rid_{rid}.points.channel_measurement_camera_readout_p_err"
    ]
)
argsort = np.argsort(num_pulses)
num_pulses = num_pulses[argsort]
pop = 1 - pop  # convert to excitation probability
pop = pop[argsort]
pop_log = np.log10(pop)
pop_err = pop_err[argsort]
pop_err_log = pop_err / (pop * np.log(10))
# pop_err[0] = 1e-4  # v. large error for 0 pulses

n_lin = 7
p_fit, p_cov = curve_fit(
    linear_fit,
    num_pulses[:n_lin],
    pop_log[:n_lin],
    sigma=pop_err_log[:n_lin],
)

p_err = np.sqrt(np.diag(p_cov))


fig = plt.figure()

c = graph_style.get_color(0)
plt.errorbar(
    num_pulses, pop_log, yerr=pop_err_log, fmt="^", color=c, zorder=11, elinewidth=1.0
)
linrange = np.linspace(
    0,
    np.max(num_pulses),
    5000,
)
# plt.plot(
# num_pulses,
# linear_fit(num_pulses, *p_fit),
# color=c,
# zorder=10,
# alpha=graph_style.get_alpha(),
# )
plt.xlabel("Repeats $N$")
plt.ylabel("State preparation error")
# relabel y-axis ticks
y_range = np.arange(-5, +1, 1)
plt.yticks(
    y_range,
    [r"$10^{%d}$" % x for x in y_range],
)
plt.xlim([0, np.max(num_pulses)])
plt.ylim([-5, 0])

plt.savefig("state_prep.pdf")

import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit


graph_style.set_graph_style(0.75)

date = "2025-05-27"
# rid = 31565  # 20,000 counts
rid = 31590  # 100,000 counts

data_dict = graph_style.load_data(rid, date)
# for x in data_dict["datasets"].keys():
# print(x)

# camera_readout.1_ions.mean_bright
# camera_readout.1_ions.mean_dark
# camera_readout.1_ions.threshold
# camera_readout.1_ions.width_bright
# camera_readout.1_ions.width_dark
# camera_readout.mean_bright
# camera_readout.mean_dark
thresh = np.array(data_dict["datasets"][f"camera_readout.mean_threshold"])
counts = np.array(data_dict["datasets"][f"data.camera_readout.counts"])
mean_bright = np.array(data_dict["datasets"][f"camera_readout.1_ions.mean_bright"])
mean_dark = np.array(data_dict["datasets"][f"camera_readout.1_ions.mean_dark"])
width_bright = np.array(data_dict["datasets"][f"camera_readout.1_ions.width_bright"])
width_dark = np.array(data_dict["datasets"][f"camera_readout.1_ions.width_dark"])

min_tot = np.min(counts)
counts = counts - min_tot  # shift counts to start at zero
thresh = thresh - min_tot  # shift threshold to match counts
mean_bright = mean_bright - min_tot  # shift mean bright to match counts
mean_dark = mean_dark - min_tot  # shift mean dark to match counts

print(f"Mean bright: {mean_bright[0]:.0f}, Mean dark: {mean_dark[0]:.0f}")
print(f"Width bright: {width_bright[0]:.0f}, Width dark: {width_dark[0]:.0f}")
print(f"Threshold: {thresh:.0f}")

counts_dark = counts[1::2]
counts_bright = counts[0::2]

# Get false positive and false negative rates:
# 20,000 counts total
false_bright = 10**6 * np.sum(counts_dark > thresh) / counts_dark.shape[0]
false_dark = 10**6 * np.sum(counts_bright < thresh) / counts_bright.shape[0]
print(f"False bright rate: {false_bright:.0f}ppm, False dark rate: {false_dark:.0f}ppm")

max_tot = np.max(counts)

bins = np.arange(0, max_tot)

fig, ax = plt.subplots()
c_dark = graph_style.get_color(0)
c_bright = graph_style.get_color(2)

ax.hist(
    counts_dark,
    bins=bins,
    density=True,
    color=c_dark,
    label="dark counts",
    zorder=3,
    alpha=graph_style.get_alpha(),
)

ax.hist(
    counts_bright,
    bins=bins,
    density=True,
    color=c_bright,
    label="bright counts",
    zorder=3,
    alpha=graph_style.get_alpha(),
)

ax.axvline(
    thresh,
    color="black",
    linestyle="--",
    label="threshold",
    zorder=12,
)

textstr = f""

ax.legend(
    loc="lower right",
)

ax.text(
    0.95,
    0.85,
    textstr,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)

ax.set_xlabel("Camera Counts")
ax.set_ylabel("Probability")
ax.set_xlim(0, max_tot)
ax.set_yscale("log")
# plt.show()
plt.savefig("readout_hist.pdf")

"""
# Two ions:

date = "2025-05-27"
rid = 31587

data_dict = graph_style.load_data(rid, date)
# for x in data_dict["datasets"].keys():
# print(x)

# camera_readout.1_ions.mean_bright
# camera_readout.1_ions.mean_dark
# camera_readout.1_ions.threshold
# camera_readout.1_ions.width_bright
# camera_readout.1_ions.width_dark
# camera_readout.mean_bright
# camera_readout.mean_dark
thresh = np.array(data_dict["datasets"][f"camera_readout.mean_threshold"])
counts = np.array(data_dict["datasets"][f"data.camera_readout.counts"])
# split counts into ion 1 and ion 2:
counts_one = counts[:, 0]
counts_two = counts[:, 1]
print(counts_one)

min_tot = np.min(counts_one)
max_tot = np.max(counts_one)
bins = np.linspace(min_tot - 0.5, max_tot + 0.5, 100)


c_dark = graph_style.get_color(0)
c_bright = graph_style.get_color(2)

for i, c in enumerate([counts_one, counts_two]):
    fig, ax = plt.subplots()
    counts_dark = c[1::2]
    counts_bright = c[0::2]

    ax.hist(
        counts_dark,
        bins=bins,
        color=c_dark,
        label="dark counts",
        zorder=11,
        alpha=graph_style.get_alpha(),
    )

    ax.hist(
        counts_bright,
        bins=bins,
        color=c_bright,
        label="bright counts",
        zorder=11,
        alpha=graph_style.get_alpha(),
    )

    ax.axvline(
        thresh,
        color="black",
        linestyle="--",
        label="threshold \n expected error = 1$\\times 10^{-3}$",
        zorder=12,
    )

textstr = f""

ax.legend(
    loc="upper right",
)

ax.text(
    0.95,
    0.85,
    textstr,
    transform=ax.transAxes,
    verticalalignment="top",
    horizontalalignment="right",
    fontsize=12,
)

ax.set_xlabel("Camera Counts")
ax.set_ylabel("Occurrences")
plt.show()

"""

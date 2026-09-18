import matplotlib.pyplot as plt
import numpy as np
import graph_style
from scipy.optimize import curve_fit

graph_style.set_graph_style(0.75)


def get_seq_length(seq):
    # convert bytes-like object to string
    q_sequence = seq.decode("utf-8")
    # split the string into a list of strings
    q_sequence = q_sequence.split(";")
    # count the number of cliffords in the sequence
    return len(q_sequence)


def seq_to_cliffs(seq_lens):
    offs = 2
    max_len = np.max(num_cliff)
    n_lens = np.unique(num_cliff).shape[0] - 1
    gate_per_cliff = 3
    seq_cliff_lens = (
        np.round((seq_lens / (gate_per_cliff * (max_len / n_lens))), 0)
        * ((max_len - offs) / (n_lens))
        + offs
    ) // 1

    return seq_cliff_lens


def get_clifford_outcomes(surv100):
    cliff100_outcomes = []
    for s in surv100:
        cliff100_outcomes.append(max(s) / (s[0] + s[1]))
    return np.array(cliff100_outcomes)


def get_seq_outcomes(seq_lens, outcomes, seq_length=100):
    seq_cliff_lens = seq_to_cliffs(seq_lens)
    cliff100 = seq_cliff_lens == seq_length
    surv100 = outcomes[cliff100]
    outcome100 = get_clifford_outcomes(surv100)
    return outcome100


def rbm_fit(x, ps, pg):
    # Eqn 3.120 from Hughes Thesis
    return ps * pg**x + 1 / 2


date = "2025-05-13"
rid = 26744

data_dict = graph_style.load_data(rid, date)
# print(data_dict["datasets"].keys())
num_cliff = np.array(data_dict["datasets"]["data.rbm.num_cliffords"])
survivals = np.array(data_dict["datasets"]["data.rbm.survivals"])
survivals_err = np.array(data_dict["datasets"]["data.rbm.survival_errs"])
outcomes = data_dict["datasets"]["data.circuits.outcomes"]
sequences = data_dict["datasets"]["data.circuits.sequences"]
run_order = data_dict["datasets"]["data.circuits.run_order"]


seq_lens = np.array([get_seq_length(seq) for seq in sequences])
seq_lens = seq_lens[run_order]  # reorder the sequence lengths to match the run order


arg_sort = np.argsort(num_cliff)
num_cliff = num_cliff[arg_sort]
survivals = survivals[arg_sort]
survivals_err = survivals_err[arg_sort]

seq_outcomes = []
for l in num_cliff:
    seq_outcomes.append(get_seq_outcomes(seq_lens, outcomes, seq_length=l))
seq_outcomes = np.array(seq_outcomes)
plt.figure("hist")
min_f = np.min(seq_outcomes[-1])
max_f = np.max(seq_outcomes[-1])
bins = int(np.round((max_f - min_f) / 0.02, 0))

plt.hist(
    seq_outcomes[-1],
    bins=bins,
    density=True,
    label="Sequence outcomes, m=100",
)
plt.xlabel("Sequence Fidelity")
plt.ylabel("Occurrence")
plt.savefig("rbm100hist.png")


fig, ax = plt.subplots()
c = graph_style.get_color(0)
# Fit decay model to the data:
popt, pcov = curve_fit(
    rbm_fit, num_cliff, survivals, p0=[0.499, 0.999], bounds=([0, 0], [0.5, 1.0])
)
fit_errs = np.sqrt(np.diag(pcov))

ax.errorbar(
    num_cliff,
    survivals,
    yerr=survivals_err,
    fmt="^",
    color=c,
    zorder=11,
    elinewidth=1.0,
)
ax.plot(
    num_cliff,
    rbm_fit(num_cliff, popt[0], popt[1]),
    label="RBM fit",
    color=c,
    zorder=10,
    alpha=graph_style.get_alpha(),
)
ax.set_xlabel("Number of Cliffords $m$")
ax.set_ylabel("Avg. Sequence Fidelity $\mathcal{F}(m)$")

sp_err = 1 / 2 - popt[0]
clf_err = (1 - popt[1]) / 2
sp_std = (fit_errs[1]) / 2
clf_std = (fit_errs[1]) / 2
textstr = (
    f"SPAM error: {sp_err*1000:.2f}({6})$\\times 10$$^-$$^3$\n"
    + f"Clifford error: {clf_err*10000:.1f}({3})$\\times 10$$^-$$^4$"
)

ax.text(
    0.05,
    0.05,
    textstr,
    transform=ax.transAxes,
    verticalalignment="bottom",
    horizontalalignment="left",
    fontsize=12,
)
# plt.savefig("rbm_fit.pdf")
plt.show()

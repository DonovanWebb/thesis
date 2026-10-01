from atomic_physics.ions.ca40 import Ca40
import numpy as np
import matplotlib.pyplot as plt
import graph_style

graph_style.set_graph_style(0.65)

B_experiment = 5.4e-4  ## 5.4 G measured in the lab
ion = Ca40(B=B_experiment)
transition = ion.transitions["729"]


def get_rabi_frequency(lower_, upper_, gamma_):
    """Calculates Rabi frequency for transition lower_ <--> upper_
    for angle gamma_ between B-field and polarisation.
    I assume k is perpendicular to both B and polarisation.
    gamma: angle between B-field and polarisation vector
    phi_: angle between k-vector and B-field
    Full angular dependence can be found in link:
    https://quantumoptics.at/images/publications/dissertation/gulde_diss.pdf
    """

    I0 = ion.I0("729")
    I = 1.0
    phi_ = np.pi / 2

    q = ion.M[upper_] - ion.M[lower_]

    if np.abs(q) == 1:
        g = (
            1
            / np.sqrt(6)
            * np.abs(
                (np.cos(gamma_) * np.cos(2 * phi_))
                + 1j * (np.sin(gamma_) * np.cos(phi_))
            )
        )
    elif np.abs(q) == 2:
        g = (
            1
            / np.sqrt(6)
            * np.abs(
                1 / 2 * np.cos(gamma_) * np.sin(2 * phi_)
                + 1j * np.sin(gamma_) * np.sin(phi_)
            )
        )
    elif np.abs(q) == 0:
        g = 1 / 2 * np.abs(np.cos(gamma_) * np.sin(2 * phi_))
    else:
        raise ValueError("q must be 0, 1 or 2, but is {}".format(q))

    # Extracting Rabi frequency using definition in the atomic physics docs
    # https://github.com/OxfordIonTrapGroup/atomic_physics/pull/10 (could contain mistakes)
    Rabi_peak_sqrd = ion.GammaJ[upper_] * ion.Gamma[lower_, upper_] * I / I0 / 4

    Rabi = np.sqrt(Rabi_peak_sqrd) * g
    return Rabi


def get_zeeman_spliting(lower_, upper_, B_):
    """Calculates energy splitting of  lower_ and upper_ for non-zero
    magnetic field strength, relative to splitting at B=0
    """
    dfreq = (Ca40(B=B_).E[upper_] - Ca40(B=B_).E[lower_]) / (2 * np.pi)  # [Hz]
    return dfreq


def plot_QP_transition_spectrum(gamma, label=""):
    """ " Plots spectrum of QP transitions (S12 <--> D52) with heights proportional
    to the Rabi frequency of the transition for a given angle between B and polarisation
    """
    # Find largest Rabi frequency to normalise against
    max_rabi = 0
    for M_l_sign in [1, -1]:
        M_l = M_l_sign * 1 / 2
        lower = ion.index(transition.lower, M_l)
        for M_u in M_l_sign * np.array([5 / 2, 3 / 2, 1 / 2, -1 / 2, -3 / 2]):
            upper = ion.index(transition.upper, M_u)
            if get_rabi_frequency(lower, upper, gamma) > max_rabi:
                max_rabi = get_rabi_frequency(lower, upper, gamma)

    fig, ax = plt.subplots(1, 1)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for M_l_sign in [1, -1]:
        M_l = M_l_sign * 1 / 2
        lower = ion.index(transition.lower, M_l)

        if M_l_sign == -1:
            ls = "solid"
            fl = True
        else:
            ls = "dashed"
            fl = False

        for M_u in M_l_sign * np.array([5 / 2, 3 / 2, 1 / 2, -1 / 2, -3 / 2]):
            upper = ion.index(transition.upper, M_u)

            relative_splitting = (
                get_zeeman_spliting(lower, upper, B_=B_experiment) * 1e-6
            )  # MHz

            normalized_rabi = get_rabi_frequency(lower, upper, gamma) / max_rabi

            # set color based on the index in the array
            if M_u != M_l:
                c = cycle[int(M_u * M_l_sign - 5 / 2)]
                ax.bar(
                    relative_splitting * np.ones(2),
                    [0, normalized_rabi],
                    width=1.5,
                    fill=fl,
                    color=c,
                    edgecolor=c,
                    label="S({:.0f}/2)$\leftrightarrow$D({:.0f}/2)".format(
                        2 * M_l, 2 * M_u
                    ),
                    zorder=2,
                )
    # Shrink current axis's height by 20% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 - box.height * 0.05, box.width, box.height * 0.55])
    plt.text(-0.15, 1.01, label, transform=ax.transAxes, size=16, fontweight="bold")
    # put text top center of the chosen gamma angle
    ax.text(
        0,
        0.86,
        r"$\gamma = \frac{\pi}{%d}$" % (np.pi / gamma),
        transform=ax.transData,
        size=12,
        horizontalalignment="center",
        verticalalignment="bottom",
    )

    # Put a legend below current axis
    """
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        fancybox=True,
        ncol=4,
        fontsize=9,
    )
    """

    ax.set_ylim([0, 1.0])
    plt.xlabel("Relative splitting (MHz)")
    plt.ylabel("Relative Rabi \n frequency $\Omega$")
    plt.savefig(f"qp_transition_spectrum_{gamma/np.pi:.2f}.pdf")
    return


plot_QP_transition_spectrum(gamma=np.pi / 2, label="b")
plot_QP_transition_spectrum(gamma=np.pi / 4, label="c")

width = 0.65 * graph_style.get_fig_width()
figsize = (width, width * 0.75)
fig, (axu, ax) = plt.subplots(
    2, 1, gridspec_kw={"height_ratios": [1, 3]}, figsize=figsize
)

lower = ion.index(transition.lower, -1 / 2)
upper = ion.index(transition.upper, -5 / 2)
max_rabi = get_rabi_frequency(lower, upper, np.pi / 2)

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for M_l_sign in [-1]:
    M_l = M_l_sign * 1 / 2
    lower = ion.index(transition.lower, M_l)

    if M_l_sign == -1:
        ls = "solid"
    else:
        ls = "dashed"

    for M_u in M_l_sign * np.array([5 / 2, 3 / 2, 1 / 2, -1 / 2, -3 / 2]):
        upper = ion.index(transition.upper, M_u)

        relative_splitting = (
            get_zeeman_spliting(lower, upper, B_=B_experiment) * 1e-6
        )  # MHz

        gamma_scan = np.linspace(0, np.pi, 100)
        rabi_arr = []

        for gamma in gamma_scan:
            rabi = get_rabi_frequency(lower, upper, gamma) / max_rabi
            rabi_arr.append(rabi)
        c = cycle[int(M_u * M_l_sign - 5 / 2)]
        axu.plot(
            0,
            0,
            color=c,
            ls=ls,
            label="S({:.0f}/2)$\leftrightarrow$D({:.0f}/2)".format(2 * M_l, 2 * M_u),
        )
        ax.plot(
            gamma_scan,
            rabi_arr,
            color=c,
            ls=ls,
            label="S({:.0f}/2)$\leftrightarrow$D({:.0f}/2)".format(2 * M_l, 2 * M_u),
        )
# plot pi/2 dashed line
ax.axvline(
    np.pi / 2,
    color="black",
    ls="dashed",
    label="Chosen polarisation",
)
axu.axvline(
    np.pi / 2,
    color="black",
    ls="dashed",
    label="Chosen $\gamma$",
)


# Shrink current axis's height by 20% on the bottom
box = ax.get_position()
# ax.set_position([box.x0, box.y0 - box.height * 0.05, box.width, box.height * 0.55])

# Put a legend above grid
axu.legend(
    loc="upper center",
    fancybox=True,
    ncol=2,
    fontsize=9,
)
# Hide everything from upper plot apart from the legend
axu.set_xticks([])
axu.set_yticks([])
axu.spines["top"].set_visible(False)
axu.spines["right"].set_visible(False)
axu.spines["left"].set_visible(False)
axu.spines["bottom"].set_visible(False)
axu.tick_params(axis="both", which="both", length=0, labelsize=0, labelcolor="none")


# change x ticks to radians in pi
plt.xticks(
    np.linspace(0, np.pi, 5),
    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"],
)
plt.xlabel("Angle $\gamma$ (rad)")
plt.ylabel("Relative Rabi \n frequency $\Omega$")
plt.ylim([0, 1.0])
plt.xlim([0, np.pi])
plt.text(-0.15, 1.03, "a", transform=ax.transAxes, size=16, fontweight="bold")

plt.savefig("qp_gamma.pdf")

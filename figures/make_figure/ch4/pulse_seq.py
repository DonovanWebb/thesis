"""
A pulse sequence figure for state preparation of a qubit.
This state preparation uses pi-pulses on the state m+1/2 to m-3/2
followed by deshelving pulses with the 854 laser.
This is repeated N times
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
import graph_style

# Set the style for the plot
graph_style.set_graph_style(1.0)


def color(i):
    return graph_style.get_color(i)


blue = color(0)
green = color(1)
red = color(2)
purple = color(3)


# Constants
block_height = 0.7
block_width = 1.5
spacing = 0.1
rounding_radius = 0.2
y_base = 0

width = graph_style.get_fig_width()
height = width / 4
fig, ax = plt.subplots(figsize=(width, height))
x = 0


def getAxSize(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width * fig.dpi, bbox.height * fig.dpi
    return width, height


def curlyBrace(fig, ax, p1, p2, k_r=0.1, bool_auto=True, **kwargs):
    pt1 = [p1[0], p1[1]]
    pt2 = [p2[0], p2[1]]
    ax_width, ax_height = getAxSize(fig, ax)
    ax_xlim = list(ax.get_xlim())
    ax_ylim = list(ax.get_ylim())

    xscale = ax_width / abs(ax_xlim[1] - ax_xlim[0])
    yscale = ax_height / abs(ax_ylim[1] - ax_ylim[0])
    if not bool_auto:
        xscale = yscale = 1.0

    pt1[0] = (pt1[0] - ax_xlim[0]) * xscale
    pt1[1] = (pt1[1] - ax_ylim[0]) * yscale
    pt2[0] = (pt2[0] - ax_xlim[0]) * xscale
    pt2[1] = (pt2[1] - ax_ylim[0]) * yscale

    theta = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    r = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]) * k_r

    x11 = pt1[0] + r * np.cos(theta)
    y11 = pt1[1] + r * np.sin(theta)
    x22 = (pt1[0] + pt2[0]) / 2.0 - 2 * r * np.sin(theta) - r * np.cos(theta)
    y22 = (pt1[1] + pt2[1]) / 2.0 + 2 * r * np.cos(theta) - r * np.sin(theta)
    x33 = (pt1[0] + pt2[0]) / 2.0 - 2 * r * np.sin(theta) + r * np.cos(theta)
    y33 = (pt1[1] + pt2[1]) / 2.0 + 2 * r * np.cos(theta) + r * np.sin(theta)
    x44 = pt2[0] - r * np.cos(theta)
    y44 = pt2[1] - r * np.sin(theta)

    q = np.linspace(theta, theta + np.pi / 2.0, 50)
    t = q[::-1]
    arc1x = r * np.cos(t + np.pi / 2.0) + x11
    arc1y = r * np.sin(t + np.pi / 2.0) + y11
    arc2x = r * np.cos(q - np.pi / 2.0) + x22
    arc2y = r * np.sin(q - np.pi / 2.0) + y22
    arc3x = r * np.cos(q + np.pi) + x33
    arc3y = r * np.sin(q + np.pi) + y33
    arc4x = r * np.cos(t) + x44
    arc4y = r * np.sin(t) + y44

    # back to axis coordinates
    arcs = []
    for arcx, arcy in [(arc1x, arc1y), (arc2x, arc2y), (arc3x, arc3y), (arc4x, arc4y)]:
        arcx = arcx / xscale + ax_xlim[0]
        arcy = arcy / yscale + ax_ylim[0]
        arcs.append((arcx, arcy))
        ax.plot(arcx, arcy, **kwargs)

    # draw lines between arcs
    ax.plot([arcs[0][0][-1], arcs[1][0][1]], [arcs[0][1][-1], arcs[1][1][1]], **kwargs)
    ax.plot([arcs[2][0][-1], arcs[3][0][1]], [arcs[2][1][-1], arcs[3][1][1]], **kwargs)


# Rounded top-left and top-right corners only
def rounded_top_rect(x, y, width, height, radius):
    verts = [
        (x, y),  # bottom-left
        (x, y + height - radius),
        (x, y + height),
        (x + radius, y + height),  # top-left corner
        (x + width - radius, y + height),
        (x + width, y + height),
        (x + width, y + height - radius),  # top-right corner
        (x + width, y),  # down to bottom-right
        (x, y),  # close
    ]

    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]

    return Path(verts, codes)


# Draw labeled pulse block
def draw_block(label, color, x_pos, ax, text_color="black"):
    path = rounded_top_rect(x_pos, y_base, block_width, block_height, rounding_radius)
    patch = patches.PathPatch(path, facecolor=color, edgecolor="black", lw=1.5)
    ax.add_patch(patch)
    ax.text(
        x_pos + block_width / 2,
        y_base + block_height / 2,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=text_color,
    )


# Draw sequence
x = spacing / 2  # Start with initial spacing
draw_block("Doppler\nCooling", blue, x, ax)
x += block_width + spacing
draw_block("State\nPrep.$\\times 7$", green, x, ax)
x += block_width + spacing
draw_block("RSB Pulse\n$t(N)$", red, x, ax)
x1 = x
x += block_width + spacing
draw_block("Deshelve\n854-nm", purple, x, ax)
x += block_width + spacing
draw_block("Deshelve\n866-nm", purple, x, ax)
x += block_width + spacing
draw_block("State\nPrep.$\\times 1$", green, x, ax)
x += block_width + spacing
draw_block("Exp.", "lightgrey", x, ax)
x += block_width + spacing
draw_block("Readout", blue, x, ax)
x2 = x + block_width
x += block_width + spacing
x += spacing * 2

# Shared baseline
# ax.plot([0, x], [y_base, y_base], color="black", lw=1)
# add arrow head to the end of the line
ax.annotate(
    "",
    xy=(x, y_base),
    xytext=(0, y_base),
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
)

# Format
ax.axis("off")
ax.set_xlim(0, x)
ax.set_ylim(-0.5, 2)


# Add curly brace around SBC
x_start = (block_width + spacing) * 2
x_end = (block_width + spacing) * 5
y_brace = 0.65
curlyBrace(
    fig, ax, [x_start, y_brace], [x_end, y_brace], k_r=0.04, color="black", lw=1.5
)

# Optional annotation above curly brace
ax.text(
    (x_start + x_end) / 2, y_brace + 0.36, "Repeat$\\times 5$", ha="center", fontsize=8
)

# Add curly brace around last 3 pulses (P4 to P6)
x_start = (block_width + spacing) * 2
x_end = (block_width + spacing) * 6
y_brace = 1.05
curlyBrace(
    fig, ax, [x_start, y_brace], [x_end, y_brace], k_r=0.04, color="black", lw=1.5
)

# Optional annotation above curly brace
ax.text(
    (x_start + x_end) / 2,
    y_brace + 0.51,
    "Repeat$\\times N/5$",
    ha="center",
    fontsize=8,
)
# add text "a" to the top left corner
ax.text(
    -0.02,
    0.75,
    "a",
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold",
    transform=ax.transAxes,
)

plt.savefig("pulse_sequence.pdf")

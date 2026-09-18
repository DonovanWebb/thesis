import matplotlib
import matplotlib.pyplot as plt
import graph_style

graph_style.set_graph_style()
fig_size = graph_style.set_size(345)
plt.rcParams.update(
    {
        "font.family": "serif",  # use serif/main font for text elements
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    }
)


matplotlib.use("pgf")
fig, ax = plt.subplots(1, 1, figsize=fig_size)
plt.scatter([0, 1], [0, 1], label="test")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Test PGF Figure")
plt.legend()
plt.savefig("test.pgf", format="pgf")

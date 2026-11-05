"""Plot styling shared by every figure. `pca_rows` are rows read from a PCA CSV file."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Default categorical palette, fixed order
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#7d5bd6", "#2fb3c7", "#8c6d3f"]
PCA_COLOR = "0.45"
# One blue ramp, light -> dark = simplest -> most complex quartile
QUARTILE_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def style_auroc_axis(ax):
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color="0.9", lw=0.8)


def style_error_axis(ax):
    """Log error on the x-axis, with better reconstruction to the right."""
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("mean reconstruction error on normal test beats (MSE, log)")


def plot_pca_curve(ax, pca_rows, label):
    ax.plot([float(r["normal_err"]) for r in pca_rows], [float(r["auroc"]) for r in pca_rows], "s--",
            color=PCA_COLOR, lw=2, ms=6, label=label)


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

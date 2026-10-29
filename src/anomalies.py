"""Synthetic anomalies matched on distance (E3, E4).

Each generator takes a normal beat and a strength dial `a` (0 = unchanged).
Simpler types remove structure, complex types add it:

    smoothed : Gaussian low-pass filter, a = filter width (samples)     simpler
    flatline : a segment covering fraction a of the beat set to its mean simpler
    clipped  : main peak clipped at the (1 - a) quantile                simpler
    noise    : additive Gaussian noise, a = noise scale                 complex
    spikes   : 3 narrow pulses of height a                              complex

For ECG5000 every anomaly is renormalized to its source beat's mean and std,
so it can't be detected by scale. For each beat we bisect on `a` until the
distance d = ||x_anom - x|| hits a target in DISTANCES, so every type covers
the same distances. Random parts (noise vector, spike positions, flatline
centre) are drawn once per beat and type, so the distance levels differ only
in strength.

    python src/anomalies.py    # generate from ECG5000 test normals and summarize
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

DISTANCES = (2.0, 4.0, 6.0, 8.0)
N_SPIKES = 3
TOL = 0.02  # max |achieved - target| distance; otherwise the beat is skipped


def smoothed(x, a, aux):
    return gaussian_filter1d(x, a, mode="nearest") if a > 0 else x.copy()


def flatline(x, a, aux):
    # Segment of length a * L centred at aux; edge samples are partly blended
    # so the distance changes continuously with a.
    i = np.arange(len(x))
    w = np.clip(a * len(x) / 2 + 0.5 - np.abs(i - aux), 0, 1)
    if w.sum() == 0:
        return x.copy()
    m = (w * x).sum() / w.sum()
    return (1 - w) * x + w * m


def clipped(x, a, aux):
    # The main peak is the sample furthest from the median; clip on its side.
    if x[np.argmax(np.abs(x - np.median(x)))] > np.median(x):
        return np.minimum(x, np.quantile(x, 1 - a))
    return np.maximum(x, np.quantile(x, a))


def noise(x, a, aux):
    return x + a * aux


def spikes(x, a, aux):
    positions, signs = aux
    t = np.arange(len(x))[:, None]
    return x + a * (signs * np.exp(-0.5 * (t - positions) ** 2)).sum(axis=1)


# name -> (generator, direction, max strength, draw random aux from (rng, L))
TYPES = {
    "smoothed": (smoothed, "simpler", 40.0, lambda rng, L: None),
    "flatline": (flatline, "simpler", 0.8, lambda rng, L: rng.uniform(0, L - 1)),
    "clipped": (clipped, "simpler", 0.5, lambda rng, L: None),
    "noise": (noise, "complex", 20.0, lambda rng, L: rng.standard_normal(L)),
    "spikes": (spikes, "complex", 50.0,
               lambda rng, L: (rng.choice(np.arange(3, L - 3), N_SPIKES, replace=False),
                               rng.choice([-1.0, 1.0], N_SPIKES))),
}


def renormalize(x_anom, x):
    """Give the anomaly the source beat's mean and std."""
    s = x_anom.std()
    if s < 1e-8:
        return np.full_like(x, np.nan)
    return (x_anom - x_anom.mean()) / s * x.std() + x.mean()


def match_distance(x, gen, aux, target, a_max, renorm=True, iters=50):
    """Bisect on strength a so that ||x_anom - x|| = target. Returns (x_anom, a) or None."""
    def make(a):
        x_anom = gen(x, a, aux)
        return renormalize(x_anom, x) if renorm else x_anom

    def dist(a):
        return np.linalg.norm(make(a) - x)

    if not dist(a_max) >= target:  # also False for nan
        return None
    lo, hi = 0.0, a_max
    for _ in range(iters):
        mid = (lo + hi) / 2
        if dist(mid) < target:
            lo = mid
        else:
            hi = mid
    if abs(dist(hi) - target) > TOL:  # distance wasn't monotone in a
        return None
    return make(hi), hi


def generate(X, types=tuple(TYPES), distances=DISTANCES, renorm=True, seed=0):
    """One anomaly per (source beat, type, distance) where the distance is reachable.

    Returns a dict of arrays: X, source (row index into the input), anomaly_type,
    direction, distance (target), dial (matched strength).
    """
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["X", "source", "anomaly_type", "direction", "distance", "dial"]}
    for i, x in enumerate(X):
        for name in types:
            gen, direction, a_max, draw = TYPES[name]
            aux = draw(rng, len(x))
            for d in distances:
                match = match_distance(x, gen, aux, d, a_max, renorm)
                if match is None:
                    continue
                out["X"].append(match[0])
                out["source"].append(i)
                out["anomaly_type"].append(name)
                out["direction"].append(direction)
                out["distance"].append(d)
                out["dial"].append(match[1])
    return {k: np.array(v) for k, v in out.items()}


def plot_examples(X, anoms, path, source=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(TYPES), figsize=(3 * len(TYPES), 2.8), sharey=True)
    for ax, name in zip(axes, TYPES):
        ax.plot(X[source], color="black", lw=1.5, label="original")
        for d in DISTANCES:
            m = (anoms["source"] == source) & (anoms["anomaly_type"] == name) & (anoms["distance"] == d)
            if m.any():
                ax.plot(anoms["X"][m][0], lw=0.9, label=f"d={d:g}")
        ax.set_title(f"{name} ({TYPES[name][1]})", fontsize=10)
        ax.set_xticks([])
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    import complexity
    import data

    splits = data.load_splits()
    X = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = generate(X)

    # How many source beats reached each distance?
    print(f"{len(X)} source beats (test normals)")
    print(f"{'type':>9} " + " ".join(f"{'d=' + format(d, 'g'):>6}" for d in DISTANCES))
    for name in TYPES:
        counts = [((anoms["anomaly_type"] == name) & (anoms["distance"] == d)).sum() for d in DISTANCES]
        print(f"{name:>9} " + " ".join(f"{c:>6}" for c in counts))

    # Are the "simpler" types actually simpler than their source beats?
    # Median change in complexity (anomaly minus its source beat), at d = 4.
    src = complexity.compute_all(X)
    anom = complexity.compute_all(anoms["X"])
    names = list(src)
    print("\nmedian complexity change vs. source beat at d=4 (negative = simpler)")
    print(f"{'type':>9} " + " ".join(f"{n.removeprefix('complexity_'):>9}" for n in names))
    for name in TYPES:
        m = (anoms["anomaly_type"] == name) & (anoms["distance"] == 4.0)
        cells = " ".join(f"{np.median(anom[n][m] - src[n][anoms['source'][m]]):>9.3f}" for n in names)
        print(f"{name:>9} {cells}")

    path = Path(__file__).resolve().parent.parent / "results" / "figures" / "synthetic_examples.png"
    plot_examples(X, anoms, path)
    print(f"\nsaved {path.name}")

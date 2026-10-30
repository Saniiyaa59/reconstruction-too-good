"""Why does PCA miss smoothed beats at high k? Split energy and error by frequency.

Each signal's energy is split at an FFT cutoff into slow (bins < cutoff, the
overall beat shape) and fast (bins >= cutoff, small wiggles). Bin j means j
cycles per beat. The FFT is only a measurement: PCA is trained on and
reconstructs the full, unfiltered beats.

    1. slow / fast energy of the input beats
    2. slow / fast energy of the PCA error (x - x_hat), per k
    3. share of each PCA component's energy that is fast
    4. Spearman correlation between PCA error and the input's fast energy

    python src/frequency_check.py                       # cutoffs 5 10 15 20, d = 4
    python src/frequency_check.py --cutoffs 10 --distance 6
"""
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

import anomalies
import data

KS = [2, 20, 50, 120]
COMPONENT_GROUPS = [(0, 5), (5, 20), (20, 50), (50, 120)]
TYPES = ["smoothed", "noise"]


def band_energy(X, cutoff):
    """(slow, fast) energy per row; slow + fast = sum of squares (Parseval)."""
    n = X.shape[1]
    power = np.abs(np.fft.rfft(X, axis=1)) ** 2
    power[:, 1:(n + 1) // 2] *= 2  # count the dropped negative frequencies (not DC or Nyquist)
    power /= n
    return power[:, :cutoff].sum(axis=1), power[:, cutoff:].sum(axis=1)


def run_checks(X_train, groups, cutoff):
    """groups: dict name -> beats; the first is the reference (normal) group."""
    print(f"\n===== cutoff = bin {cutoff} (slow: bins 0-{cutoff - 1}, fast: bins {cutoff}+) =====")

    print("1) input energy, median per beat (slow / fast)")
    for name, X in groups.items():
        slow, fast = band_energy(X, cutoff)
        print(f"   {name:>9}: {np.median(slow):7.2f} / {np.median(fast):6.3f}")

    print("2) PCA error energy, median per beat (slow / fast)")
    pcas = {k: PCA(n_components=k).fit(X_train) for k in KS}
    for k, pca in pcas.items():
        cells = []
        for name, X in groups.items():
            slow, fast = band_energy(X - pca.inverse_transform(pca.transform(X)), cutoff)
            cells.append(f"{name}: {np.median(slow):.3f} / {np.median(fast):.3f}")
        print(f"   k={k:>3} | " + " | ".join(cells))

    print("3) PCA components (k=120): median share of energy that is fast")
    slow, fast = band_energy(pcas[120].components_, cutoff)
    share = fast / (slow + fast)
    for lo, hi in COMPONENT_GROUPS:
        print(f"   components {lo + 1:>3}-{hi:<3}: {np.median(share[lo:hi]):.2f}")

    # Beats are z-normalized, so total energy is fixed and the slow-energy
    # correlation is the same number with the opposite sign.
    print("4) spearman(PCA error, input fast energy) over normal + smoothed beats")
    X = np.r_[groups["normal"], groups["smoothed"]]
    _, fast_in = band_energy(X, cutoff)
    for k, pca in pcas.items():
        err = ((X - pca.inverse_transform(pca.transform(X))) ** 2).sum(axis=1)
        print(f"   k={k:>3}: {spearmanr(err, fast_in).statistic:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--distance", type=float, default=4.0)
    args = parser.parse_args()

    splits = data.load_splits()
    X_normal = splits["X_test"][splits["y_test"] == data.NORMAL_CLASS]
    anoms = anomalies.generate(X_normal)  # all types, so the random draws match E3
    groups = {"normal": X_normal}
    for name in TYPES:
        m = (anoms["anomaly_type"] == name) & (anoms["distance"] == args.distance)
        groups[name] = anoms["X"][m]

    for cutoff in args.cutoffs:
        run_checks(splits["X_train"], groups, cutoff)


if __name__ == "__main__":
    main()

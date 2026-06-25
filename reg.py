import os
BASE = "/Users/nwang40/Library/CloudStorage/Dropbox/pr1_zuco/analysis/a3_comp/results"
os.chdir(BASE)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, t as t_dist
from scipy.ndimage import gaussian_filter1d

betas = np.load('coef3_nsubj_ridge_time_channel.npy')# shape: (12, 5, 801, 105) subject, regressor and intercept, time, channel


def find_clusters_1d(mask):
    """Return list of (start, end) inclusive indices for contiguous True segments."""
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts, ends))


def cluster_perm_1samp(X_subj_time, alpha=0.05, n_perm=2000, tail=1, rng=None):
    """One-sample sign-flip cluster permutation on (n_subj, n_time)."""
    if rng is None:
        rng = np.random.default_rng(0)

    n_subj, n_time = X_subj_time.shape
    df = n_subj - 1

    t_obs, _ = ttest_1samp(X_subj_time, 0.0, axis=0, nan_policy="omit")

    if tail == 1:
        t_thr = t_dist.ppf(1 - alpha, df)
        supra = t_obs > t_thr
    elif tail == -1:
        t_thr = t_dist.ppf(alpha, df)
        supra = t_obs < t_thr
    else:
        t_thr = t_dist.ppf(1 - alpha / 2, df)
        supra = np.abs(t_obs) > t_thr

    clust_ranges = find_clusters_1d(supra)
    clust_masses = []
    for (s, e) in clust_ranges:
        if tail == 0:
            mass = e - s + 1
        else:
            mass = np.sum(t_obs[s:e + 1])
        clust_masses.append(mass)

    max_masses = np.zeros(n_perm, dtype=float)
    for b in range(n_perm):
        flips = rng.choice([-1.0, 1.0], size=(n_subj, 1))
        Xp = X_subj_time * flips
        t_p, _ = ttest_1samp(Xp, 0.0, axis=0, nan_policy="omit")

        if tail == 1:
            supra_p = t_p > t_thr
        elif tail == -1:
            supra_p = t_p < t_thr
        else:
            supra_p = np.abs(t_p) > t_thr

        ranges_p = find_clusters_1d(supra_p)
        if not ranges_p:
            max_masses[b] = 0.0
        else:
            masses_p = [
                (np.sum(np.abs(t_p[s:e + 1])) if tail == 0 else np.sum(t_p[s:e + 1]))
                for (s, e) in ranges_p
            ]
            max_masses[b] = np.max(masses_p)

    clusters = []
    sig_mask = np.zeros(n_time, dtype=bool)
    for (s, e), mass in zip(clust_ranges, clust_masses):
        p_clust = (np.sum(max_masses >= mass) + 1) / (n_perm + 1)
        clusters.append({"start": s, "end": e, "mass": mass, "p": p_clust})
        if p_clust < 0.05:
            sig_mask[s:e + 1] = True
    

    return t_obs, clusters, sig_mask, t_thr, max_masses


def plot_roi_clusters(X_sel, roi, labels, colors, t_axis,
                      alpha_cluster, n_perm, tail, sigma_plot, rng,
                      sig_alpha=0.05, y_sig=-0.035, save_dir="fig"):
    """Run cluster-perm and plot one ROI for the given predictors."""
    n_subj = X_sel.shape[0]
    df = n_subj - 1

    fig, ax = plt.subplots(figsize=(8, 6))

    for k, lab in enumerate(labels):
        X = X_sel[:, k, :]
        t_obs, clusters, sig_mask, t_thr, _ = cluster_perm_1samp(
            X, alpha=alpha_cluster, n_perm=n_perm, tail=tail, rng=rng
        )

        mean_tc = X.mean(axis=0)
        sem_tc = X.std(axis=0, ddof=1) / np.sqrt(n_subj)
        mean_plot = gaussian_filter1d(mean_tc, sigma=sigma_plot)
        sem_plot = gaussian_filter1d(sem_tc, sigma=sigma_plot)
        peak_idx = int(np.argmax(mean_tc))
        print(f"[{roi} | {lab}] peak = {mean_plot[peak_idx]:.4f} at t = {t_axis[peak_idx]:.1f} ms")


        ax.plot(t_axis, mean_plot, label=lab, color=colors[k])
        ax.fill_between(t_axis, mean_plot - sem_plot, mean_plot + sem_plot,
                        color=colors[k], alpha=0.1, linewidth=0)
                        

        for c in clusters:
            if c["p"] < sig_alpha:
                ax.axvspan(t_axis[c["start"]], t_axis[c["end"]],
                           color=colors[k], alpha=0.05)
                ax.plot([t_axis[c["start"]], t_axis[c["end"]]],
                        [y_sig, y_sig], color=colors[k], linewidth=5)

        if not clusters:
            print(f"\n[{roi} | {lab}] No supra-threshold clusters at alpha={alpha_cluster}")
        else:
            print(f"\n[{roi} | {lab}] t-thr={t_thr:.3f} (df={df}), perms={n_perm}")
            for c in clusters:
                tag = " *SIG*" if c["p"] < sig_alpha else ""
                t0, t1 = t_axis[c["start"]], t_axis[c["end"]]
                print(f"  cluster idx {c['start']:>4d}..{c['end']:<4d}  "
                    f"time {t0:7.1f}..{t1:7.1f} ms  "
                    f"mass={c['mass']:7.2f}  p={c['p']:.4f}{tag}")

            # one-line summary of just the significant windows
            sig = [c for c in clusters if c["p"] < sig_alpha]
            if sig:
                wins = [f"{t_axis[c['start']]:.0f}–{t_axis[c['end']]:.0f} ms (p={c['p']:.4f})"
                        for c in sig]
                print(f"  >> Significant windows [{lab}]: " + "; ".join(wins))

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_title(roi)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{save_dir}/regr-roi{roi}-regr{labels[0]}-"
                f"cluthre{alpha_cluster}-perm{n_perm}.png", dpi=300)
    plt.show()
    plt.close(fig)



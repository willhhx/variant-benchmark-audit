"""Shared helpers for the variant-interpretation benchmark.

Design choices (documented in the manuscript Methods):
  * variants are split by chromosome group (held-out chromosomes), never at random,
    so neighbouring / linked variants cannot straddle train and test;
  * the same variants and the same bootstrap resamples are used for every model, so
    model comparisons are paired;
  * every random step is seeded.
"""
import os
import numpy as np
import pandas as pd

SEED = 42

# model key (as in model_embeddings.MODEL_REGISTRY) -> display name
MODELS = {
    "dnabert1": "DNABERT",
    "dnabert2": "DNABERT-2",
    "hyenadna": "HyenaDNA",
    "nucleotide_transformer_v2": "NT-v2",
    "gena_lm": "GENA-LM",
    "grover": "GROVER",
}
BASELINE = {"kmer4": "4-mer baseline"}
DISPLAY = {**MODELS, **BASELINE}

# Same three chromosome groups as the upstream benchmark (Feng et al. 2025)
CHROM_GROUPS = {
    1: [f"chr{c}" for c in [3, 6, 9, 12, 16, 18, 19, 21]],
    2: [f"chr{c}" for c in [2, 5, 11, 14, 17, 20, 22]] + ["chrX"],
    3: [f"chr{c}" for c in [1, 4, 7, 8, 10, 13, 15]],
}
CHROM2GROUP = {c: g for g, cs in CHROM_GROUPS.items() for c in cs}

VARIANT_KEY = ["chromosome", "pos", "ref", "alt"]


CROP = 501  # central window (bp) given to every model; variant sits at the exact centre


def tag_for(n_per_class, seed=SEED, crop=CROP):
    return f"n{n_per_class}_s{seed}_w{crop}"


def paths(outdir, tag):
    return {
        "meta": os.path.join(outdir, f"meta_{tag}.csv"),
        "emb": os.path.join(outdir, "emb"),
        "res": os.path.join(outdir, "results"),
        "fig": os.path.join(outdir, "figures"),
    }


def emb_file(outdir, tag, model, which):
    return os.path.join(outdir, "emb", f"{model}_{which}_{tag}.npy")


def sample_meta(csv_path, n_per_class, seed=SEED):
    """Deterministic, class-stratified sample of variants (metadata only, no sequences)."""
    cols = ["class", "chromosome", "ref", "pos", "alt", "MAF", "split", "label"]
    df = pd.read_csv(csv_path, usecols=cols)
    df["row_id"] = np.arange(len(df))
    df = df[df["chromosome"].isin(CHROM2GROUP)].copy()
    df = df.drop_duplicates(VARIANT_KEY)
    if n_per_class and n_per_class > 0:
        parts = []
        for cls, sub in df.groupby("class"):
            k = min(n_per_class, len(sub))
            parts.append(sub.sample(n=k, random_state=seed))
        df = pd.concat(parts)
    df = df.sort_values("row_id").reset_index(drop=True)
    df["group"] = df["chromosome"].map(CHROM2GROUP)
    return df


def load_meta(outdir, tag):
    return pd.read_csv(paths(outdir, tag)["meta"])


def load_features(outdir, tag, model):
    ref = np.load(emb_file(outdir, tag, model, "ref")).astype(np.float64)
    alt = np.load(emb_file(outdir, tag, model, "alt")).astype(np.float64)
    if ref.ndim > 2:
        ref = ref.reshape(len(ref), -1)
        alt = alt.reshape(len(alt), -1)
    return ref, alt


# ----------------------------------------------------------------------------
# Bootstrap utilities
# ----------------------------------------------------------------------------
def strat_boot_indices(y, B, seed=SEED):
    """(B, n) index matrix; resamples positives and negatives separately (keeps class
    counts fixed). The same matrix is reused for every model => paired resamples."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    out = np.empty((B, len(y)), dtype=np.int32)
    for b in range(B):
        out[b, : len(pos)] = rng.choice(pos, len(pos), replace=True)
        out[b, len(pos):] = rng.choice(neg, len(neg), replace=True)
    return out


def cluster_boot_indices(clusters, B, seed=SEED):
    """List of B index arrays from resampling whole clusters (e.g. chromosomes)."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {u: np.where(clusters == u)[0] for u in uniq}
    out = []
    for _ in range(B):
        pick = rng.choice(uniq, len(uniq), replace=True)
        out.append(np.concatenate([members[u] for u in pick]))
    return out


def pct_ci(x, alpha=0.05):
    x = np.asarray(x)
    return float(np.nanpercentile(x, 100 * alpha / 2)), float(np.nanpercentile(x, 100 * (1 - alpha / 2)))


def boot_pvalue(diffs):
    """Two-sided bootstrap p-value for H0: difference = 0 (percentile / sign method,
    with +1 smoothing so p is never exactly 0)."""
    diffs = np.asarray(diffs)
    B = len(diffs)
    p_le = (np.sum(diffs <= 0) + 1) / (B + 1)
    p_ge = (np.sum(diffs >= 0) + 1) / (B + 1)
    return float(min(1.0, 2 * min(p_le, p_ge)))


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ----------------------------------------------------------------------------
# DeLong test for two correlated ROC curves (Sun & Xu 2014 fast algorithm)
# ----------------------------------------------------------------------------
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1)
        i = k
    out = np.empty(n)
    out[j] = t + 1
    return out


def delong_two(y, s1, s2):
    """Returns (auc1, auc2, z, two-sided p) for H0: AUC1 == AUC2 (paired)."""
    from scipy.stats import norm

    y = np.asarray(y)
    order = np.argsort(-y, kind="stable")  # positives first
    preds = np.vstack([np.asarray(s1)[order], np.asarray(s2)[order]])
    m = int(np.sum(y == 1))
    n = preds.shape[1] - m
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(p) for p in pos])
    ty = np.vstack([_midrank(p) for p in neg])
    tz = np.vstack([_midrank(p) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), float("nan")
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


# ----------------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------------
def fmt_ci(v, lo, hi, nd=3):
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_macros(path, macros, append=False):
    """macros: dict name -> value. Names must contain letters only."""
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for k, v in macros.items():
            assert k.isalpha(), f"macro name must be letters only: {k}"
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

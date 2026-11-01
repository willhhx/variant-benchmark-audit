"""Shared helpers for the leakage analyses (reuses common.py conventions: seed 42,
three held-out chromosome groups, StandardScaler + L2 logistic regression C=1)."""
import os, re, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
warnings.filterwarnings("ignore")

SEED = 42
from pathlib import Path
DATA = str(Path(__file__).resolve().parents[1] / "data") + "/"   # repo-relative; see data/README.md
RUN = DATA                       # meta_*.csv and emb/*.npy live directly under data/
VEP = DATA + "vep_pathogenic_coding.csv"
AG_REF = DATA + "alphagenome/patho_ref.npy"
AG_ALT = DATA + "alphagenome/patho_alt.npy"
AG_META = DATA + "alphagenome/alphagenome_test_meta.csv"
KEY = ["chromosome", "pos", "ref", "alt"]
CHROM_GROUPS = {
    1: [f"chr{c}" for c in [3, 6, 9, 12, 16, 18, 19, 21]],
    2: [f"chr{c}" for c in [2, 5, 11, 14, 17, 20, 22]] + ["chrX"],
    3: [f"chr{c}" for c in [1, 4, 7, 8, 10, 13, 15]],
}
CHROM2GROUP = {c: g for g, cs in CHROM_GROUPS.items() for c in cs}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(y, s, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s = np.asarray(s)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)
    v = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        v.append(auroc(y[i], s[i]))
    return tuple(np.percentile(v, [2.5, 97.5]))


def paired_boot_diff(y, s1, s2, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    d = []
    for _ in range(B):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(auroc(y[i], s1[i]) - auroc(y[i], s2[i]))
    d = np.array(d)
    p = min(1.0, 2 * min((np.sum(d <= 0) + 1) / (B + 1), (np.sum(d >= 0) + 1) / (B + 1)))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(p)


def oof_scores(X, y, groups, C=1.0, seed=SEED):
    """Leave-chromosome-group-out out-of-fold probabilities."""
    out = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g; tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, random_state=seed))
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    return out


def consequence_flags(cons):
    """Raw terms and harmonised flags from the CONSEQUENCE string."""
    c = cons.fillna("")
    terms = c.str.split(",")
    allterms = sorted({t for ts in terms for t in ts if t})
    raw = pd.DataFrame({f"term::{t}": terms.apply(lambda ts, t=t: int(t in ts)) for t in allterms}, index=cons.index)
    nc = c.str.contains(r"non[-_]coding", regex=True).astype(int)
    pure = (c == "missense_variant").astype(int)
    return raw, nc, pure


def load_main():
    meta = pd.read_csv(RUN + "meta_n3000_s42_w501.csv")
    vep = pd.read_csv(VEP, usecols=["CHROM", "POS", "REF", "ALT", "CONSEQUENCE", "REVIEW_STATUS"], low_memory=False)
    vep = vep.rename(columns={"CHROM": "chromosome", "POS": "pos", "REF": "ref", "ALT": "alt"}).drop_duplicates(KEY)
    m = meta.merge(vep, on=KEY, how="left")
    assert len(m) == len(meta)
    return m


def subst_type(df):
    return (df["ref"] + ">" + df["alt"])

"""Leakage analyses on the existing n=3000/class, 501-bp embeddings (NO new extraction).
Outputs (../results): metadata_vs_embedding.csv, stratified_auroc.csv, incremental.csv, shuffle_null.csv"""
import sys, json
import numpy as np, pandas as pd
import leakage_lib as L

OUT = "../results/"
m = L.load_main()
y = m["class"].values.astype(int); g = m["group"].values
raw, nc, pure = L.consequence_flags(m["CONSEQUENCE"])
sub = pd.get_dummies(L.subst_type(m)).astype(int)

feat = {
    "meta: pure-missense flag only": pure.values.reshape(-1, 1),
    "meta: harmonised flags (non-coding, pure-missense)": np.c_[nc.values, pure.values],
    "meta: raw consequence terms": raw.values,
    "meta: raw terms + substitution type": np.c_[raw.values, sub.values],
}
rows = []; oof = {}
for name, X in feat.items():
    s = L.oof_scores(X.astype(float), y, g)
    oof[name] = s
    lo, hi = L.boot_ci(y, s)
    rows.append(dict(source="metadata", model="-", features=name, auroc=L.auroc(y, s), lo=lo, hi=hi, n=len(y)))

for mod, disp in [("hyenadna", "HyenaDNA"), ("dnabert2", "DNABERT-2"), ("kmer4", "4-mer")]:
    ref = np.load(f"{L.RUN}emb/{mod}_ref_n3000_s42_w501.npy").astype(float)
    alt = np.load(f"{L.RUN}emb/{mod}_alt_n3000_s42_w501.npy").astype(float)
    ref = ref.reshape(len(ref), -1); alt = alt.reshape(len(alt), -1)
    for fname, X in [("reference-only", ref), ("delta (alt-ref)", alt - ref)]:
        s = L.oof_scores(X, y, g)
        oof[f"{disp}|{fname}"] = s
        lo, hi = L.boot_ci(y, s)
        rows.append(dict(source="embedding", model=disp, features=fname, auroc=L.auroc(y, s), lo=lo, hi=hi, n=len(y)))
res = pd.DataFrame(rows)
res.to_csv(OUT + "metadata_vs_embedding.csv", index=False)
print(res.round(3).to_string())
pd.DataFrame(oof).assign(class_=y, group=g, row_id=m.row_id).to_csv(OUT + "oof_scores_main.csv", index=False)

# --- stratified AUROC (a) using full-sample OOF scores; (b) probes trained+tested inside the stratum
strata = {
    "pure missense (no other consequence)": (pure == 1).values,
    "non-coding-transcript flagged": (nc == 1).values,
    "other annotated (neither)": ((pure == 0) & (nc == 0)).values,
}
srows = []
for sname, mask in strata.items():
    n1 = int(y[mask].sum()); n0 = int(mask.sum() - n1)
    for key in ["HyenaDNA|reference-only", "HyenaDNA|delta (alt-ref)", "DNABERT-2|reference-only",
                "DNABERT-2|delta (alt-ref)", "4-mer|reference-only", "4-mer|delta (alt-ref)",
                "meta: raw terms + substitution type"]:
        s = oof[key][mask]
        lo, hi = L.boot_ci(y[mask], s)
        srows.append(dict(stratum=sname, n_pathogenic=n1, n_common=n0, scores="full-sample OOF", key=key,
                          auroc=L.auroc(y[mask], s), lo=lo, hi=hi))
    # (b) within-stratum training
    if min(n1, n0) >= 30:
        for mod, disp in [("hyenadna", "HyenaDNA"), ("dnabert2", "DNABERT-2")]:
            ref = np.load(f"{L.RUN}emb/{mod}_ref_n3000_s42_w501.npy").astype(float); ref = ref.reshape(len(ref), -1)
            s = L.oof_scores(ref[mask], y[mask], g[mask])
            ok = ~np.isnan(s)
            lo, hi = L.boot_ci(y[mask][ok], s[ok])
            srows.append(dict(stratum=sname, n_pathogenic=n1, n_common=n0, scores="trained within stratum",
                              key=f"{disp}|reference-only", auroc=L.auroc(y[mask][ok], s[ok]), lo=lo, hi=hi))
sres = pd.DataFrame(srows); sres.to_csv(OUT + "stratified_auroc.csv", index=False)
print(sres.round(3).to_string())

# --- incremental value of the reference-only score beyond metadata
irows = []
Mname = "meta: raw terms + substitution type"
for disp in ["HyenaDNA", "DNABERT-2", "4-mer"]:
    r = oof[f"{disp}|reference-only"]
    base = oof[Mname]
    X2 = np.c_[raw.values, sub.values, r]
    comb = L.oof_scores(X2.astype(float), y, g)
    d, lo, hi, p = L.paired_boot_diff(y, comb, base)
    irows.append(dict(model=disp, auroc_meta=L.auroc(y, base), auroc_ref=L.auroc(y, r), auroc_meta_plus_ref=L.auroc(y, comb),
                      gain=d, gain_lo=lo, gain_hi=hi, p_boot=p))
    d2, lo2, hi2, p2 = L.paired_boot_diff(y, base, r)
    irows[-1].update(meta_minus_ref=d2, mmr_lo=lo2, mmr_hi=hi2, mmr_p=p2)
ires = pd.DataFrame(irows); ires.to_csv(OUT + "incremental.csv", index=False)
print(ires.round(3).to_string())

# --- label-shuffle null (sanity check of the probe pipeline)
rng = np.random.default_rng(L.SEED); nrows = []
for mod, disp in [("hyenadna", "HyenaDNA"), ("dnabert2", "DNABERT-2")]:
    ref = np.load(f"{L.RUN}emb/{mod}_ref_n3000_s42_w501.npy").astype(float); ref = ref.reshape(len(ref), -1)
    a = []
    for _ in range(20):
        ys = rng.permutation(y)
        a.append(L.auroc(ys, L.oof_scores(ref, ys, g)))
    nrows.append(dict(model=disp, features="reference-only", perms=20, mean=np.mean(a), sd=np.std(a), max=np.max(a),
                      observed=L.auroc(y, oof[f"{disp}|reference-only"])))
pd.DataFrame(nrows).to_csv(OUT + "shuffle_null.csv", index=False)
print(pd.DataFrame(nrows).round(3).to_string())

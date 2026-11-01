"""(a) Cross-tab of consequence terms by class (full VEP table + sample); (b) AlphaGenome 1,018-variant subset, reported separately."""
import numpy as np, pandas as pd
import leakage_lib as L
OUT = "../results/"

# (a) full table
vep = pd.read_csv(L.VEP, usecols=["CHROM", "POS", "REF", "ALT", "LABEL", "CONSEQUENCE"], low_memory=False)
terms = vep.assign(t=vep.CONSEQUENCE.str.split(",")).explode("t")
ct = terms.groupby(["t", "LABEL"]).size().unstack(fill_value=0)
ct["n_total"] = ct.sum(axis=1)
tot = vep.LABEL.value_counts()
ct["frac_common"] = ct.get("Common", 0) / tot["Common"]
ct["frac_pathogenic"] = ct.get("Pathogenic", 0) / tot["Pathogenic"]
ct = ct.sort_values("n_total", ascending=False)
ct.to_csv(OUT + "consequence_term_crosstab_full_table.csv")
print(ct.head(25).to_string())
print("pure missense:", (vep.CONSEQUENCE == "missense_variant").groupby(vep.LABEL).agg(["sum", "size", "mean"]).to_string())
# within non-coding stratum: which spelling does each class use?
nc = vep[vep.CONSEQUENCE.str.contains(r"non[-_]coding")]
spell = nc.assign(hyphen=nc.CONSEQUENCE.str.contains("non-coding_transcript_variant"),
                  exon=nc.CONSEQUENCE.str.contains("non_coding_transcript_exon_variant"))
print(spell.groupby("LABEL")[["hyphen", "exon"]].agg(["sum", "size"]).to_string())

# (b) AlphaGenome subset
t = pd.read_csv(L.AG_META)   # upstream test split rows, in the order of the cached embeddings
ref = np.load(L.AG_REF).astype(float)
alt = np.load(L.AG_ALT).astype(float)
assert len(t) == len(ref) == len(alt), (len(t), len(ref))
vep2 = vep.rename(columns={"CHROM": "chromosome", "POS": "pos", "REF": "ref", "ALT": "alt"}).drop_duplicates(L.KEY)
t = t.merge(vep2[L.KEY + ["CONSEQUENCE"]], on=L.KEY, how="left")
print("VEP match:", t.CONSEQUENCE.notna().mean(), len(t))
y = t["class"].values.astype(int); # Feng et al.'s test split is entirely chr8, so chromosome-group CV is impossible here.
# Use 5 contiguous position blocks on chr8 (limits, but does not remove, linkage between folds).
order = np.argsort(t.pos.values); g = np.empty(len(t), int); g[order] = np.arange(len(t)) * 5 // len(t)
keep = (ref.std(0) > 1e-8) & (alt.std(0) > 1e-8); ref = ref[:, keep]; alt = alt[:, keep]; print("AG dims kept:", int(keep.sum()))
raw, ncf, pure = L.consequence_flags(t["CONSEQUENCE"]); sub = pd.get_dummies(L.subst_type(t)).astype(int)
print("relative perturbation (median ||d||/||ref||):", float(np.median(np.linalg.norm(alt - ref, axis=1) / np.linalg.norm(ref, axis=1))))
rows = []
def add(name, X):
    s = L.oof_scores(X.astype(float), y, g); lo, hi = L.boot_ci(y, s)
    rows.append(dict(features=name, auroc=L.auroc(y, s), lo=lo, hi=hi, n=len(y), n_path=int(y.sum()))); return s
sm = add("meta: raw terms + substitution type", np.c_[raw.values, sub.values])
sr = add("AlphaGenome reference-only", ref); sd = add("AlphaGenome delta", alt - ref)
res = pd.DataFrame(rows); res.to_csv(OUT + "alphagenome_subset.csv", index=False); print(res.round(3).to_string())
mask = (pure == 1).values
print("AG pure-missense stratum n:", int(y[mask].sum()), int(mask.sum() - y[mask].sum()))
if min(y[mask].sum(), mask.sum() - y[mask].sum()) >= 20:
    for nm, s in [("ref", sr), ("delta", sd), ("meta", sm)]:
        print("AG stratum", nm, round(L.auroc(y[mask], s[mask]), 3), [round(v, 3) for v in L.boot_ci(y[mask], s[mask])])

"""Metadata probe restricted to consequence terms observed in BOTH classes (removes pipeline-specific vocabulary), plus Figure 1."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import leakage_lib as L
OUT = "../results/"; FIG = "../figures/"
m = L.load_main(); y = m["class"].values.astype(int); g = m["group"].values
raw, nc, pure = L.consequence_flags(m["CONSEQUENCE"])
# vocabulary from the FULL table (not the sample), terms with >=1% prevalence in each class
vep = pd.read_csv(L.VEP, usecols=["LABEL", "CONSEQUENCE"])
tt = vep.assign(t=vep.CONSEQUENCE.str.split(",")).explode("t")
prev = tt.groupby(["t", "LABEL"]).size().unstack(fill_value=0) / vep.LABEL.value_counts()
shared = [t for t in prev.index if prev.loc[t].min() >= 0.01]
print("shared terms (>=1% in both classes):", shared)
cols = [c for c in raw.columns if c.replace("term::", "") in shared]
s = L.oof_scores(raw[cols].values.astype(float), y, g); lo, hi = L.boot_ci(y, s)
sv = dict(features="meta: consequence terms shared by both classes", auroc=L.auroc(y, s), lo=lo, hi=hi, n_terms=len(cols))
pd.DataFrame([sv]).to_csv(OUT + "shared_vocab_metadata.csv", index=False); print(sv)

res = pd.read_csv(OUT + "metadata_vs_embedding.csv"); strat = pd.read_csv(OUT + "stratified_auroc.csv")
bars = [("Metadata: consequence terms\n+ substitution (all)", res.iloc[3]), ("Metadata: shared-vocabulary\nterms only", pd.Series(sv)),
        ("Metadata: harmonised\nflags (2)", res.iloc[1])]
for mod in ["DNABERT-2", "HyenaDNA", "4-mer"]:
    bars.append((f"{mod}\nreference-only", res[(res.model == mod) & (res.features == "reference-only")].iloc[0]))
for mod in ["DNABERT-2", "HyenaDNA", "4-mer"]:
    bars.append((f"{mod}\nvariant (delta)", res[(res.model == mod) & (res.features == "delta (alt-ref)")].iloc[0]))
col = ["#b5483a"] * 3 + ["#2f6f8f"] * 3 + ["#9aa5ad"] * 3
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
a = ax[0]
for i, (n, r) in enumerate(bars):
    a.bar(i, r["auroc"] - 0.5, bottom=0.5, color=col[i], width=0.7)
    a.errorbar(i, r["auroc"], yerr=[[r["auroc"] - r["lo"]], [r["hi"] - r["auroc"]]], color="k", lw=1, capsize=2)
a.axhline(0.5, color="k", lw=0.8, ls="--"); a.set_ylim(0.4, 1.0); a.set_xticks(range(len(bars)))
a.set_xticklabels([b[0] for b in bars], rotation=60, ha="right", fontsize=7.5); a.set_ylabel("AUROC (leave-chromosome-group-out; 95% CI)")
a.set_title("A. Annotation alone vs. sequence embeddings (n = 6,000)", fontsize=10, loc="left")
b = ax[1]; sub = strat[(strat.scores == "full-sample OOF") & (strat.stratum.str.startswith("pure"))]
labs = []; k = 0
for mod in ["DNABERT-2", "HyenaDNA", "4-mer"]:
    for f, c in [("reference-only", "#2f6f8f"), ("delta (alt-ref)", "#9aa5ad")]:
        r = sub[sub.key == f"{mod}|{f}"].iloc[0]
        b.bar(k, r.auroc - 0.5, bottom=0.5, color=c, width=0.7); b.errorbar(k, r.auroc, yerr=[[r.auroc - r.lo], [r.hi - r.auroc]], color="k", lw=1, capsize=2)
        labs.append(f"{mod}\n{'ref-only' if f.startswith('ref') else 'delta'}"); k += 1
b.axhline(0.5, color="k", lw=0.8, ls="--"); b.set_ylim(0.4, 1.0); b.set_xticks(range(k)); b.set_xticklabels(labs, rotation=60, ha="right", fontsize=7.5)
n1, n0 = int(sub.n_pathogenic.iloc[0]), int(sub.n_common.iloc[0])
b.set_title(f"B. Plain-missense stratum only ({n1:,} pathogenic, {n0} common)", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(FIG + "fig_leakage_main.pdf"); plt.savefig(FIG + "fig_leakage_main.png", dpi=200)

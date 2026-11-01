"""Relative perturbation (dilution) macros, plain-missense-only main table, fold-level supplement table."""
import numpy as np, pandas as pd, leakage_lib as L
R = "../results/"; G = "../manuscript/generated/"
rel = {}
for mod, nm in [("hyenadna", "Hyn"), ("dnabert2", "Dnb"), ("kmer4", "Km")]:
    ref = np.load(f"{L.RUN}emb/{mod}_ref_n3000_s42_w501.npy").astype(float); alt = np.load(f"{L.RUN}emb/{mod}_alt_n3000_s42_w501.npy").astype(float)
    ref = ref.reshape(len(ref), -1); alt = alt.reshape(len(alt), -1)
    rel[nm] = float(np.median(np.linalg.norm(alt - ref, axis=1) / np.linalg.norm(ref, axis=1)) * 100)
ref = np.load(L.AG_REF).astype(float); alt = np.load(L.AG_ALT).astype(float)
rel["Ag"] = float(np.median(np.linalg.norm(alt - ref, axis=1) / np.linalg.norm(ref, axis=1)) * 100)
print(rel)
with open(G + "macros_leakage.tex", "a") as f:
    for k, v in rel.items(): f.write(f"\\newcommand{{\\Rel{k}}}{{{v:.3g}}}\n")
st = pd.read_csv(R + "stratified_auroc.csv"); p = st[st.stratum.str.startswith("pure")]
nice = {"meta: raw terms + substitution type": "Annotation (substitution type only here)", "DNABERT-2|reference-only": "DNABERT-2 reference-only", "DNABERT-2|delta (alt-ref)": "DNABERT-2 variant ($\\Delta$)",
        "HyenaDNA|reference-only": "HyenaDNA reference-only", "HyenaDNA|delta (alt-ref)": "HyenaDNA variant ($\\Delta$)", "4-mer|reference-only": "4-mer reference-only", "4-mer|delta (alt-ref)": "4-mer variant ($\\Delta$)"}
order = list(nice)
with open(G + "tab_strata_pure.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\\toprule\nFeature source & Scores from probe trained on & AUROC (95\\% CI)\\\\\\midrule\n")
    for k in order:
        for sc in ["full-sample OOF", "trained within stratum"]:
            r = p[(p.key == k) & (p.scores == sc)]
            if len(r): r = r.iloc[0]; f.write(f"{nice[k]} & {'all 6,000 variants' if sc.startswith('full') else 'plain-missense variants only'} & {r.auroc:.3f} ({r.lo:.3f}-{r.hi:.3f})\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
tf = pd.read_csv(R + "main_with_cluster_ci_and_folds.csv")
with open(G + "tab_folds.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\\toprule\nFeature source & Group 1 & Group 2 & Group 3\\\\\\midrule\n")
    for _, r in tf.iterrows(): f.write(f"{r['name']} & {r.f1:.3f} & {r.f2:.3f} & {r.f3:.3f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
ct = pd.read_csv(R + "consequence_term_crosstab_full_table.csv").head(20)
with open(G + "tab_crosstab_full.tex", "w") as f:
    f.write("\\begin{tabular}{lrrr}\\toprule\nConsequence term & Pathogenic (n) & Common (n) & Total\\\\\\midrule\n")
    for _, r in ct.iterrows(): f.write(f"\\texttt{{{str(r.t).replace('_', chr(92) + '_').replace('&', chr(92) + '&')}}} & {int(r.Pathogenic):,} & {int(r.Common):,} & {int(r.n_total):,}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")

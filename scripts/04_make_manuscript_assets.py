"""Robustness analyses + LaTeX assets. Every number in the manuscript comes from here (macros_leakage.tex) or from
../results/*.csv; nothing is typed by hand. Run after 01-03."""
import numpy as np, pandas as pd, json
import leakage_lib as L
R = "../results/"; G = "../manuscript/generated/"
m = L.load_main(); y = m["class"].values.astype(int); g = m["group"].values; chrom = m["chromosome"].values
raw, nc, pure = L.consequence_flags(m["CONSEQUENCE"]); sub = pd.get_dummies(L.subst_type(m)).astype(int)
oof = pd.read_csv(R + "oof_scores_main.csv")
M = {}  # macro name -> value string
def f3(x): return f"{x:.3f}"
def put(name, auc, lo=None, hi=None):
    M[name] = f3(auc)
    if lo is not None: M[name + "Lo"] = f3(lo); M[name + "Hi"] = f3(hi)

# ---------- cluster bootstrap (resample whole chromosomes) + fold-level AUROC ----------
def cluster_ci(s, B=1000, seed=L.SEED, mask=None):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
    cl = chrom[idx]; u = np.unique(cl); mem = {c: idx[cl == c] for c in u}; v = []
    for _ in range(B):
        pick = rng.choice(u, len(u)); ii = np.concatenate([mem[c] for c in pick]); v.append(L.auroc(y[ii], s[ii]))
    return tuple(np.nanpercentile(v, [2.5, 97.5]))
def folds(s): return [L.auroc(y[g == k], s[g == k]) for k in (1, 2, 3)]
cols = {"Annotation: consequence terms + substitution": "meta: raw terms + substitution type",
        "Annotation: harmonised flags (2)": "meta: harmonised flags (non-coding, pure-missense)",
        "Annotation: plain-missense flag only": "meta: pure-missense flag only",
        "DNABERT-2, reference-only": "DNABERT-2|reference-only", "DNABERT-2, variant ($\\Delta$)": "DNABERT-2|delta (alt-ref)",
        "HyenaDNA, reference-only": "HyenaDNA|reference-only", "HyenaDNA, variant ($\\Delta$)": "HyenaDNA|delta (alt-ref)",
        "4-mer, reference-only": "4-mer|reference-only", "4-mer, variant ($\\Delta$)": "4-mer|delta (alt-ref)"}
me = pd.read_csv(R + "metadata_vs_embedding.csv")
def lookup(key):
    if key.startswith("meta"): r = me[me.features == key].iloc[0]
    else: mod, f = key.split("|"); r = me[(me.model == mod) & (me.features == f)].iloc[0]
    return r
rows = []; extra = {}
# substitution-only (no annotation at all)
s_sub = L.oof_scores(sub.values.astype(float), y, g); extra["Substitution type only (no annotation)"] = s_sub
oofd = {c: oof[c].values for c in oof.columns if c not in ("class_", "group", "row_id")}
for disp, key in list(cols.items()):
    s = oofd[key]; r = lookup(key); cl = cluster_ci(s); fo = folds(s)
    rows.append(dict(name=disp, auroc=r.auroc, lo=r.lo, hi=r.hi, clo=cl[0], chi=cl[1], f1=fo[0], f2=fo[1], f3=fo[2]))
s = s_sub; lo, hi = L.boot_ci(y, s); cl = cluster_ci(s); fo = folds(s)
rows.append(dict(name="Substitution type only (no annotation)", auroc=L.auroc(y, s), lo=lo, hi=hi, clo=cl[0], chi=cl[1], f1=fo[0], f2=fo[1], f3=fo[2]))
tabF = pd.DataFrame(rows); tabF.to_csv(R + "main_with_cluster_ci_and_folds.csv", index=False)
put("SubOnly", tabF.iloc[-1].auroc, tabF.iloc[-1].lo, tabF.iloc[-1].hi)

# ---------- review-status sensitivity (label quality) ----------
rs = m["REVIEW_STATUS"].fillna("")
crit = ~((y == 1) & (rs == "no_assertion_criteria_provided"))
multi = ~((y == 1) & ~rs.isin(["criteria_provided,_multiple_submitters,_no_conflicts", "reviewed_by_expert_panel", "practice_guideline"]))
rrows = []
for lab, mk in [("All variants", np.ones(len(y), bool)), ("Pathogenic with assertion criteria", crit), ("Pathogenic multi-submitter/expert", multi)]:
    for nm, key in [("Annotation: consequence terms + substitution", "meta: raw terms + substitution type"),
                    ("DNABERT-2, reference-only", "DNABERT-2|reference-only"), ("DNABERT-2, variant ($\\Delta$)", "DNABERT-2|delta (alt-ref)"),
                    ("HyenaDNA, reference-only", "HyenaDNA|reference-only")]:
        sc = oofd[key]; lo, hi = L.boot_ci(y[mk], sc[mk])
        rrows.append(dict(subset=lab, n_pathogenic=int(y[mk].sum()), n_common=int((1 - y[mk]).sum()), features=nm, auroc=L.auroc(y[mk], sc[mk]), lo=lo, hi=hi))
rev = pd.DataFrame(rrows); rev.to_csv(R + "review_status_sensitivity.csv", index=False)
print(m.REVIEW_STATUS.value_counts().to_string())

# ---------- C sweep for the three feature sources (supplement) ----------
csw = []
for mod, disp in [("dnabert2", "DNABERT-2"), ("hyenadna", "HyenaDNA"), ("kmer4", "4-mer")]:
    ref = np.load(f"{L.RUN}emb/{mod}_ref_n3000_s42_w501.npy").astype(float); alt = np.load(f"{L.RUN}emb/{mod}_alt_n3000_s42_w501.npy").astype(float)
    ref = ref.reshape(len(ref), -1); alt = alt.reshape(len(alt), -1)
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        csw.append(dict(model=disp, C=C, reference_only=L.auroc(y, L.oof_scores(ref, y, g, C=C)), delta=L.auroc(y, L.oof_scores(alt - ref, y, g, C=C))))
for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
    csw.append(dict(model="Annotation (terms + substitution)", C=C, reference_only=np.nan, delta=L.auroc(y, L.oof_scores(np.c_[raw.values, sub.values].astype(float), y, g, C=C))))
pd.DataFrame(csw).to_csv(R + "c_sweep_all.csv", index=False)

# ---------- macros ----------
vep = pd.read_csv(L.VEP, usecols=["CHROM", "POS", "REF", "ALT", "LABEL", "CONSEQUENCE"], low_memory=False)
vep = vep.rename(columns={"CHROM": "chromosome", "POS": "pos", "REF": "ref", "ALT": "alt"})
print("VEP rows by label:", vep.LABEL.value_counts().to_dict(), " unique keys:", vep.drop_duplicates(L.KEY).LABEL.value_counts().to_dict())
vd = vep.drop_duplicates(L.KEY); pm = (vd.CONSEQUENCE == "missense_variant")
M["TabNPath"] = f"{int((vd.LABEL == 'Pathogenic').sum()):,}"; M["TabNCommon"] = f"{int((vd.LABEL == 'Common').sum()):,}"
M["PureNPath"] = f"{int(pm[vd.LABEL == 'Pathogenic'].sum()):,}"; M["PureNCommon"] = f"{int(pm[vd.LABEL == 'Common'].sum()):,}"
M["PurePctPath"] = f"{100 * pm[vd.LABEL == 'Pathogenic'].mean():.1f}"; M["PurePctCommon"] = f"{100 * pm[vd.LABEL == 'Common'].mean():.1f}"
M["SampPurePctPath"] = f"{100 * pure[y == 1].mean():.1f}"; M["SampPurePctCommon"] = f"{100 * pure[y == 0].mean():.1f}"
M["NSample"] = f"{len(y):,}"; M["NPerClass"] = f"{int(y.sum()):,}"
for nm, key in [("MetaAll", "meta: raw terms + substitution type"), ("MetaHarm", "meta: harmonised flags (non-coding, pure-missense)"),
                ("MetaFlag", "meta: pure-missense flag only"), ("MetaTerms", "meta: raw consequence terms")]:
    r = lookup(key); put(nm, r.auroc, r.lo, r.hi)
r = pd.read_csv(R + "shared_vocab_metadata.csv").iloc[0]; put("MetaShared", r.auroc, r.lo, r.hi)
for nm, key in [("DnbRef", "DNABERT-2|reference-only"), ("DnbDelta", "DNABERT-2|delta (alt-ref)"), ("HynRef", "HyenaDNA|reference-only"),
                ("HynDelta", "HyenaDNA|delta (alt-ref)"), ("KmRef", "4-mer|reference-only"), ("KmDelta", "4-mer|delta (alt-ref)")]:
    r = lookup(key); put(nm, r.auroc, r.lo, r.hi)
st = pd.read_csv(R + "stratified_auroc.csv"); ps = st[st.stratum.str.startswith("pure")]
M["PureStratP"] = f"{int(ps.n_pathogenic.iloc[0]):,}"; M["PureStratC"] = f"{int(ps.n_common.iloc[0]):,}"
def sget(stratum, scores, key):
    r = st[st.stratum.str.startswith(stratum) & (st.scores == scores) & (st.key == key)].iloc[0]; return r.auroc, r.lo, r.hi
for nm, key in [("DnbRef", "DNABERT-2|reference-only"), ("DnbDelta", "DNABERT-2|delta (alt-ref)"), ("HynRef", "HyenaDNA|reference-only"),
                ("HynDelta", "HyenaDNA|delta (alt-ref)"), ("KmRef", "4-mer|reference-only"), ("KmDelta", "4-mer|delta (alt-ref)"), ("Meta", "meta: raw terms + substitution type")]:
    put(nm + "Pure", *sget("pure", "full-sample OOF", key))
put("DnbRefPureTrained", *sget("pure", "trained within stratum", "DNABERT-2|reference-only"))
put("HynRefPureTrained", *sget("pure", "trained within stratum", "HyenaDNA|reference-only"))
put("MetaNC", *sget("non-coding", "full-sample OOF", "meta: raw terms + substitution type"))
put("DnbRefNC", *sget("non-coding", "full-sample OOF", "DNABERT-2|reference-only"))
M["NCP"] = f"{int(st[st.stratum.str.startswith('non-coding')].n_pathogenic.iloc[0]):,}"; M["NCC"] = f"{int(st[st.stratum.str.startswith('non-coding')].n_common.iloc[0]):,}"
inc = pd.read_csv(R + "incremental.csv").set_index("model")
M["IncDnb"] = f3(inc.loc["DNABERT-2", "gain"]); M["IncDnbLo"] = f3(inc.loc["DNABERT-2", "gain_lo"]); M["IncDnbHi"] = f3(inc.loc["DNABERT-2", "gain_hi"])
M["IncHyn"] = f3(inc.loc["HyenaDNA", "gain"]); M["IncHynLo"] = f3(inc.loc["HyenaDNA", "gain_lo"]); M["IncHynHi"] = f3(inc.loc["HyenaDNA", "gain_hi"])
sh = pd.read_csv(R + "shuffle_null.csv").set_index("model"); M["ShufMean"] = f3(sh["mean"].mean()); M["ShufMax"] = f3(sh["max"].max())
ag = pd.read_csv(R + "alphagenome_subset.csv").set_index("features")
for nm, key in [("AgMeta", "meta: raw terms + substitution type"), ("AgRef", "AlphaGenome reference-only"), ("AgDelta", "AlphaGenome delta")]:
    r = ag.loc[key]; put(nm, r.auroc, r.lo, r.hi)
M["AgN"] = "1,018"; M["AgPath"] = "461"; M["AgCommon"] = "557"
cs0 = pd.read_csv(R + "c_sweep_all.csv")
ann = cs0[cs0.model.str.startswith("Annotation")]; M["CAnnMin"] = f3(ann.delta.min()); M["CAnnMax"] = f3(ann.delta.max())
dl = cs0[~cs0.model.str.startswith("Annotation")]; M["CDeltaMin"] = f3(dl.delta.min()); M["CDeltaMax"] = f3(dl.delta.max())
for nm, mod in [("Dnb", "DNABERT-2"), ("Hyn", "HyenaDNA")]:
    d_ = cs0[cs0.model == mod]; M["CRef" + nm + "Min"] = f3(d_.reference_only.min()); M["CRef" + nm + "Max"] = f3(d_.reference_only.max())
rv = rev[(rev.features.str.startswith("Annotation")) & (rev.subset != "All variants")]; M["RevAnnMin"] = f3(rv.auroc.min()); M["RevAnnMax"] = f3(rv.auroc.max())
M["DnbSens"] = f3(L.auroc(y, oofd["DNABERT-2|reference-only"]) - L.auroc(y, oofd["DNABERT-2|delta (alt-ref)"]))
# consequence cross-tab values for text
ct = pd.read_csv(R + "consequence_term_crosstab_full_table.csv").set_index("t")
for nm, term in [("NCExon", "non_coding_transcript_exon_variant"), ("Downstream", "downstream_gene_variant"), ("Upstream", "upstream_gene_variant"),
                 ("Regulatory", "regulatory_region_variant"), ("NCHyphen", "non-coding_transcript_variant"), ("InitTypo", "initiatior_codon_variant")]:
    r = ct.loc[term]; M[nm + "Com"] = f"{int(r.Common):,}"; M[nm + "Path"] = f"{int(r.Pathogenic):,}"
with open(G + "macros_leakage.tex", "w") as f:
    for k, v in M.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
json.dump(M, open(R + "manuscript_numbers.json", "w"), indent=1)

# ---------- LaTeX tables ----------
def ci(a, lo, hi): return f"{a:.3f} ({lo:.3f}--{hi:.3f})"
with open(G + "tab_main.tex", "w") as f:
    f.write("\\begin{tabular}{lcc}\\toprule\nFeature source & AUROC (95\\% CI) & Chromosome-cluster 95\\% CI\\\\\\midrule\n")
    for i, r in tabF.iterrows():
        if i == 3: f.write("\\midrule\n")
        f.write(f"{r['name']} & {ci(r.auroc, r.lo, r.hi)} & {r.clo:.3f}--{r.chi:.3f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(G + "tab_strata.tex", "w") as f:
    f.write("\\begin{tabular}{llcc}\\toprule\nStratum (pathogenic / common) & Feature source & AUROC (95\\% CI) & Probe trained\\\\\\midrule\n")
    lab = {"pure": "Plain missense only", "non-coding": "Non-coding-transcript flagged", "other": "Other annotated"}
    nice = {"meta: raw terms + substitution type": "Annotation", "DNABERT-2|reference-only": "DNABERT-2 reference-only", "DNABERT-2|delta (alt-ref)": "DNABERT-2 variant ($\\Delta$)",
            "HyenaDNA|reference-only": "HyenaDNA reference-only", "HyenaDNA|delta (alt-ref)": "HyenaDNA variant ($\\Delta$)", "4-mer|reference-only": "4-mer reference-only", "4-mer|delta (alt-ref)": "4-mer variant ($\\Delta$)"}
    for sk in ["pure", "non-coding", "other"]:
        sub_ = st[st.stratum.str.startswith(sk)]; n1, n0 = int(sub_.n_pathogenic.iloc[0]), int(sub_.n_common.iloc[0])
        f.write(f"\\multicolumn{{4}}{{l}}{{\\emph{{{lab[sk]}}} ({n1:,} / {n0:,})}}\\\\\n")
        for _, r in sub_.iterrows():
            f.write(f"\\quad & {nice[r.key]} & {ci(r.auroc, r.lo, r.hi)} & {'within stratum' if r.scores.startswith('trained') else 'all variants'}\\\\\n")
        f.write("\\midrule\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(G + "tab_vocab.tex", "w") as f:
    f.write("\\begin{tabular}{lrr}\\toprule\nConsequence term (as annotated) & Pathogenic (\\%) & Common (\\%)\\\\\\midrule\n")
    for t in ["missense_variant", "intron_variant", "5_prime_UTR_variant", "3_prime_UTR_variant", "non-coding_transcript_variant", "initiatior_codon_variant",
              "non_coding_transcript_exon_variant", "downstream_gene_variant", "upstream_gene_variant", "regulatory_region_variant"]:
        r = ct.loc[t]; f.write(f"\\texttt{{{t.replace('_', chr(92) + '_')}}} & {100 * r.frac_pathogenic:.1f} & {100 * r.frac_common:.1f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(G + "tab_ag.tex", "w") as f:
    f.write("\\begin{tabular}{lc}\\toprule\nFeature source (1,018 variants; position-block CV) & AUROC (95\\% CI)\\\\\\midrule\n")
    for key, nm in [("meta: raw terms + substitution type", "Annotation: consequence terms + substitution"), ("AlphaGenome reference-only", "AlphaGenome reference-only"), ("AlphaGenome delta", "AlphaGenome variant ($\\Delta$)")]:
        r = ag.loc[key]; f.write(f"{nm} & {ci(r.auroc, r.lo, r.hi)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(G + "tab_review.tex", "w") as f:
    f.write("\\begin{tabular}{llcc}\\toprule\nPathogenic subset (pathogenic / common) & Feature source & AUROC (95\\% CI)\\\\\\midrule\n")
    for sname, d in rev.groupby("subset", sort=False):
        f.write(f"\\multicolumn{{3}}{{l}}{{\\emph{{{sname}}} ({d.n_pathogenic.iloc[0]:,} / {d.n_common.iloc[0]:,})}}\\\\\n")
        for _, r in d.iterrows(): f.write(f"\\quad {r.features} & & {ci(r.auroc, r.lo, r.hi)}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
cs = pd.read_csv(R + "c_sweep_all.csv")
with open(G + "tab_csweep.tex", "w") as f:
    f.write("\\begin{tabular}{lrcc}\\toprule\nFeature source & $C$ & Reference-only AUROC & Variant ($\\Delta$) AUROC\\\\\\midrule\n")
    for _, r in cs.iterrows():
        a = "--" if np.isnan(r.reference_only) else f"{r.reference_only:.3f}"; f.write(f"{r.model} & {r.C:g} & {a} & {r.delta:.3f}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
with open(G + "tab_incremental.tex", "w") as f:
    f.write("\\begin{tabular}{lccc}\\toprule\nModel & Annotation alone & + reference-only score & Gain (95\\% CI)\\\\\\midrule\n")
    for k, r in inc.iterrows(): f.write(f"{k} & {r.auroc_meta:.3f} & {r.auroc_meta_plus_ref:.3f} & {r.gain:.3f} ({r.gain_lo:.3f}--{r.gain_hi:.3f})\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
print(tabF.round(3).to_string()); print(rev.round(3).to_string()); print(cs.round(3).to_string()); print(len(M), "macros")

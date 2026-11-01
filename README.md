# eujph-context-vs-variant

Audit of annotation shortcuts in a ClinVar-pathogenic vs gnomAD-common missense benchmark, and what they imply for computational evidence in variant interpretation
(manuscript: *When Accuracy Is Not Understanding: Auditing Annotation Shortcuts in a ClinVar-gnomAD Benchmark for AI-Assisted Variant Interpretation*).

**Status: unpublished research code and manuscript (private).**

## Headline result
Annotation metadata alone (consequence terms + substitution, no sequence) reaches AUROC 0.968 (95% CI 0.963-0.972) under leave-chromosome-group-out evaluation,
versus 0.681 for the best sequence embedding (DNABERT-2 reference-only); variant-specific (alt-ref) embeddings are near chance (0.504-0.524).
All numbers in the manuscript are generated from `results/` by the scripts; see `manuscript/generated/macros_leakage.tex`.

## Layout
```
data/          all inputs (see data/README.md for origin and licensing notes)
scripts/       01 leakage probes, strata, incremental, shuffle null
               02 vocabulary cross-tab + AlphaGenome 1,018-variant subset
               03 shared-vocabulary probe + Figure 2
               04 robustness analyses + manuscript numbers/tables
               05 relative-perturbation macros + extra tables
               build_manuscript.sh
results/       every result table (CSV/JSON), incl. per-variant out-of-fold scores
figures/       Figure 2 (PDF/PNG)
manuscript/    main.tex, supplement.tex, refs.bib, generated/ (macros and tables), built PDFs
```

## Reproduce
```bash
pip install -r requirements.txt
cd scripts
python 01_leakage_main.py && python 02_crosstab_and_alphagenome.py && python 03_shared_vocab_and_figure.py \
  && python 04_make_manuscript_assets.py && python 05_extra_assets.py
./build_manuscript.sh        # needs pdflatex, bibtex, latexmk
```
Run in this order: 04 rewrites the macro file and 05 appends to it. Runtime is a few minutes on a laptop CPU; no GPU and no model download are needed because all embeddings are included.
The pipeline was rerun end to end from a clean copy of this repository and reproduced every result table and manuscript macro exactly (seed 42; scikit-learn 1.9.1, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, Matplotlib 3.11.2).

## Related resources
- Embeddings for DNABERT-2, HyenaDNA, AlphaGenome and the 4-mer baseline are included, so the full analysis runs without any model weights or extraction code.
- The full 2-kb/6-kb/100-kb sequence tables of the upstream benchmark are available at ChongWuLab/dna_foundation_benchmark.

## Citations for the AlphaGenome run
AlphaGenome: Avsec et al., *Nature* 649(8099):1206-1218 (2026). PyTorch port: `alphagenome-pytorch` (https://github.com/genomicsxai/alphagenome-pytorch; Apache-2.0; model parameters and outputs remain subject to the Google DeepMind AlphaGenome Model Terms).

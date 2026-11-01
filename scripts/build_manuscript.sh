#!/bin/bash
# Build manuscript/main.pdf and manuscript/supplement.pdf (run from eujph_context_vs_variant/manuscript)
set -e
cd "$(dirname "$0")/../manuscript"
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplement.tex

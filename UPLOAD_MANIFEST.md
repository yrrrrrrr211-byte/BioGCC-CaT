# GitHub upload manifest

Recommended directories/files to upload:

```text
figure code/
results tables/
example data structure/
```

These complement the existing repository files:

```text
README.md
LICENSE
requirements.txt
.gitignore
train code/
processing code/
compute_subtype_attr_gxi.py
run_holdout_gene_attribution.py
```

Do not upload these local-only folders/files:

```text
.ipynb_checkpoints/
__pycache__/
data(cell-level)/
data_with_meta/
holdout_out/
baseline/
the local biology-interpretation working folder
Journal_Manuscript*.docx
Journal_Manuscript*.pdf
paper_content.txt
```

Optional README addition after uploading:

```markdown
## Figure generation and result tables

The `figure code/` directory contains scripts used to generate the main and supplementary figures.  
The `results tables/` directory contains summarized performance metrics, ablation results, attribution results, enrichment analyses, and pathway-level correlation results used in the manuscript.
```

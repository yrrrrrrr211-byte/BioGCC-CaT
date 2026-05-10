# BioGCC-CaT

BioGCC-CaT is an interpretable graph-augmented capsule network for disease-state classification in single-cell RNA-seq data.

## Overview

BioGCC-CaT integrates gene co-expression graph encoding, capsule representation learning, task-guided routing, supervised contrastive learning, and a norm-based evidence head to support both disease-state classification and traceable interpretation.

## Dataset

The scRNA-seq dataset used in the manuscript is publicly available from the Gene Expression Omnibus under accession number **GSE176269**.

Raw and processed expression matrices are not included in this repository due to file size considerations. Users can download the original data from GEO and preprocess it following the scripts provided in this repository.

## Main files

- `train code/BIOtrain.py`: Model training and evaluation (includes ablation M0-M5)
- `train code/cap.py`, `train code/gcn.py`, `train code/trans.py`: Model architecture modules
- `train code/losses.py`: Loss functions (contrastive loss, cross-entropy)
- `train code/tgr_caps.py`: Task-guided dynamic routing implementation
- `pocessing code/datapocessing.py`: Data preprocessing and feature construction
- `compute_subtype_attr_gxi.py`: Gene-level attribution analysis
- `run_holdout_gene_attribution.py`: Held-out test set analysis
- `requirements.txt`: Python dependencies

## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

Python 3.8+ required.

## Data Preparation

1. Download scRNA-seq data from GEO (GSE176269)
2. Place raw expression matrix and sample metadata in `./raw_data/`
3. Run preprocessing:

```bash
python "pocessing code/datapocessing.py"
```

This will generate preprocessed data in `./data/`

## Model Training

Train the full BioGCC-CaT model (M5):

```bash
python "train code/BIOtrain.py" --mode M5 --epochs 80 --batch_size 64 --data_dir ./data
```

To train ablation variants (M0-M4), specify `--mode`:

```bash
python "train code/BIOtrain.py" --mode M0  # Baseline
python "train code/BIOtrain.py" --mode M1  # +GCN
python "train code/BIOtrain.py" --mode M2  # +TGR
python "train code/BIOtrain.py" --mode M3  # +SupCon
python "train code/BIOtrain.py" --mode M4  # +Norm head
```

## Gene Attribution & Interpretation

After model training, compute gene-level attribution scores:

```bash
python compute_subtype_attr_gxi.py
```

This generates:
- Gene attribution rankings by evidence channel
- Top positive/negative contributor genes
- Output saved to `./output/`

## Performance

On independent held-out test set:

| Metric | M5 (Full Model) |
|--------|-----------------|
| Accuracy | 0.9446 |
| Balanced Accuracy | 0.9201 |
| Macro F1 | 0.9273 |
| Macro AUPRC | 0.9725 |

Per-class F1 scores:
- Healthy: 0.9289
- FluA: 0.9651
- COVID-19: 0.8880

## Ablation Results

| Model | Description | ACC | BACC | F1-macro |
|-------|-------------|-----|------|----------|
| M0 | scCaT baseline | 0.9152 | 0.9091 | 0.8938 |
| M1 | +GCN | 0.9268 | 0.9000 | 0.9024 |
| M2 | +TGR | 0.9213 | 0.9185 | 0.9012 |
| M3 | +SupCon | 0.9241 | 0.9134 | 0.9025 |
| M4 | +Norm head | 0.9244 | 0.8954 | 0.8982 |
| M5 | Full model | **0.9446** | **0.9201** | **0.9273** |

## Key Findings

**FluA Evidence Patterns:**
- Broadly distributed across multiple cell types
- Associated with antiviral and antigen-presentation processes

**COVID-19 Evidence Patterns:**
- Concentrated in specific cell populations (B cells, epithelial subtypes)
- Involves inflammatory activation and epithelial stress responses

## Citation

If you use this code, please cite our manuscript:

```
BioGCC-CaT: An Interpretable Graph-Augmented Capsule Network for Disease-State Classification in Single-Cell RNA-seq Data.
[Citation details to be added upon publication]
```

## License

MIT License

---

**Note**: This repository contains source code and scripts for reproducibility. Detailed biological interpretation analysis and figure generation code are available in the supplementary materials of the published manuscript.

import os
import gc
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import fdrcorrection
from scipy import sparse
from collections import Counter

# ---------------------- 模块1：环境与日志 ----------------------
def fprint(txtt):
    with open(r"dp_your_data.txt", "a+", encoding="utf-8") as f:
        f.write(str(txtt) + "\n")
    print(txtt)

if os.path.exists("dp_your_data.txt"):
    os.remove("dp_your_data.txt")

fprint("start123")

# File paths - relative to repository root
# Users should download raw data from GEO (GSE176269) and place in ./raw_data
datapath = "./raw_data"
sample_f = os.path.join(datapath, "expression", "CovidStudy_rawCounts_061721.txt")

# phenotype 文件位置自动兜底：优先 metadata，其次 phenotype
label_candidates = [
    os.path.join(datapath, "metadata", "CovidStudy_phenotype_061721.txt"),
    os.path.join(datapath, "phenotype", "CovidStudy_phenotype_061721.txt"),
]
label_f = None
for p in label_candidates:
    if os.path.exists(p):
        label_f = p
        break
if label_f is None:
    raise FileNotFoundError(
        "Cannot find CovidStudy_phenotype_061721.txt in metadata/ or phenotype/"
    )

# 输出目录：单独存，避免覆盖旧 data
OUT_DIR = "data_with_meta"
os.makedirs(OUT_DIR, exist_ok=True)

# 参数（保持原逻辑）
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
TEST_FRAC  = 0.1
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

RANDOM_STATE = 42
FDR_THRESH = 0.005
CELL_MIN_GENES = 200
GENE_MAX_ZERO_RATIO = 0.97
ANOVA_BATCH = 300

# ---------------------- 模块2：稀疏化读取 ----------------------
fprint("Reading data...")
df_head = pd.read_csv(sample_f, sep="\t", nrows=0)
samples_name_list = list(df_head.columns[1:])

gene_names_list, data, rows, cols = [], [], [], []
chunk_size, row_offset = 2000, 0

for chunk in pd.read_csv(sample_f, sep="\t", chunksize=chunk_size, index_col=0):
    genes = chunk.index.tolist()
    gene_names_list.extend(genes)
    coo = sparse.coo_matrix(chunk.values.astype(np.float32))
    data.extend(coo.data)
    rows.extend(coo.row + row_offset)
    cols.extend(coo.col)
    row_offset += len(genes)
    del chunk, coo
    gc.collect()

samples = sparse.csr_matrix(
    (data, (rows, cols)),
    shape=(len(gene_names_list), len(samples_name_list)),
    dtype=np.float32
).T

rna_name = np.array(gene_names_list)
samples_name = np.array(samples_name_list)

del data, rows, cols, df_head, gene_names_list, samples_name_list
gc.collect()

fprint("First!")
fprint(f"Samples shape: {samples.shape}")

# ---------------------- 模块3：读取标签并转数字（保持旧逻辑） ----------------------
fprint("Processing labels...")

# 这里保持原脚本口径：skiprows=1 + 取第1列和第4列
meta_df = pd.read_csv(label_f, sep="\t", skiprows=1)
meta_map = {str(row[0]): str(row[3]) for _, row in meta_df.iterrows()}
del meta_df
gc.collect()

label_dict = {"healthy": 0, "COVID-19": 1, "fluA": 2}
labels_temp = []
valid_idx = []
unknown_cid = []

for i, cid in enumerate(samples_name):
    status = meta_map.get(cid, "Unknown")
    if status in label_dict:
        labels_temp.append(label_dict[status])
        valid_idx.append(i)
    else:
        unknown_cid.append(cid)

if unknown_cid:
    fprint(f"Warning! No label for {len(unknown_cid)} cells, e.g.: {unknown_cid[:3]}...")

# 在旧逻辑这里保留 cell_id
cell_ids = samples_name[valid_idx].astype(str)

samples = samples[valid_idx]
labels = np.array(labels_temp, dtype=np.int32)

fprint("Converted to label in numbers")
fprint(f"Labels shape: {labels.shape}, Label distribution: {Counter(labels)}")

del valid_idx, labels_temp, meta_map, unknown_cid, samples_name
gc.collect()

# ---------------------- 模块4：随机划分 8:1:1 ----------------------
fprint("Splitting Train/Val/Test = 8:1:1 (NO stratify)...")

idx_all = np.arange(samples.shape[0])

train_idx, tmp_idx = train_test_split(
    idx_all,
    test_size=(1.0 - TRAIN_FRAC),
    random_state=RANDOM_STATE,
    shuffle=True
)

val_idx, test_idx = train_test_split(
    tmp_idx,
    test_size=0.5,
    random_state=RANDOM_STATE + 1,
    shuffle=True
)

x_train = samples[train_idx]
y_train = labels[train_idx]
cell_ids_train = cell_ids[train_idx]

x_val   = samples[val_idx]
y_val   = labels[val_idx]
cell_ids_val = cell_ids[val_idx]

x_test  = samples[test_idx]
y_test  = labels[test_idx]
cell_ids_test = cell_ids[test_idx]

del samples, labels, idx_all, train_idx, tmp_idx, val_idx, test_idx, cell_ids
gc.collect()

def dist_info(y):
    c = Counter(y.tolist())
    n = len(y)
    pct = {k: round(v / n * 100, 2) for k, v in c.items()}
    return c, pct

train_c, train_pct = dist_info(y_train)
val_c,   val_pct   = dist_info(y_val)
test_c,  test_pct  = dist_info(y_test)

fprint("===== SPLIT SUMMARY (NO stratify) =====")
fprint(f"Train: cells={x_train.shape[0]:,} genes={x_train.shape[1]:,} | dist={train_c} | %={train_pct}")
fprint(f"Val:   cells={x_val.shape[0]:,}   genes={x_val.shape[1]:,}   | dist={val_c}   | %={val_pct}")
fprint(f"Test:  cells={x_test.shape[0]:,}  genes={x_test.shape[1]:,}  | dist={test_c}  | %={test_pct}")
fprint("=======================================")

u = np.unique(y_train)
if len(u) < 3:
    raise ValueError(f"Train set missing classes (got {u}). Because NO stratify, try changing RANDOM_STATE.")

# ---------------------- 模块5：ANOVA & FDR ----------------------
fprint("Calculating ANOVA and FDR...")

def batch_anova(matrix, y_lbl, batch=ANOVA_BATCH):
    n_genes = matrix.shape[1]
    pvals = np.ones(n_genes, dtype=np.float64)
    for start in range(0, n_genes, batch):
        end = min(start + batch, n_genes)
        chunk = matrix[:, start:end].toarray()
        for i in range(chunk.shape[1]):
            if np.all(chunk[:, i] == 0):
                continue
            _, p = stats.f_oneway(
                chunk[y_lbl == 0, i],
                chunk[y_lbl == 1, i],
                chunk[y_lbl == 2, i]
            )
            pvals[start + i] = p
        del chunk
        gc.collect()
        if (start // batch) % 5 == 0:
            fprint(f"  ANOVA progress: {end}/{n_genes} genes")
    return pvals

P_fdr_raw = batch_anova(x_train, y_train)
rejected, P_fdr = fdrcorrection(np.nan_to_num(P_fdr_raw, nan=1.0), alpha=0.05)
del P_fdr_raw, rejected
gc.collect()

keep_fdr_mask = P_fdr <= FDR_THRESH

x_train = x_train[:, keep_fdr_mask]
x_val   = x_val[:, keep_fdr_mask]
x_test  = x_test[:, keep_fdr_mask]
rna_name = rna_name[keep_fdr_mask]
P_fdr = P_fdr[keep_fdr_mask]

fprint(f"Filter successful (FDR <= {FDR_THRESH})")
fprint(f"After FDR: x_train {x_train.shape}, x_val {x_val.shape}, x_test {x_test.shape}")
gc.collect()

# ---------------------- 模块6：剔除稀疏细胞（同步过滤 cell_id） ----------------------
fprint(f"Deleting sparse data (cell >= {CELL_MIN_GENES} genes, gene zero ratio <= {GENE_MAX_ZERO_RATIO})...")

def get_cell_mask_by_nnz(mat, min_genes=200, tag=""):
    nnz = mat.getnnz(axis=1)
    mask = nnz >= min_genes
    fprint(f"  [{tag}] Cells before: {mat.shape[0]}, after: {mask.sum()}")
    return mask

m = get_cell_mask_by_nnz(x_train, min_genes=CELL_MIN_GENES, tag="train")
x_train = x_train[m]
y_train = y_train[m]
cell_ids_train = cell_ids_train[m]

m = get_cell_mask_by_nnz(x_val, min_genes=CELL_MIN_GENES, tag="val")
x_val = x_val[m]
y_val = y_val[m]
cell_ids_val = cell_ids_val[m]

m = get_cell_mask_by_nnz(x_test, min_genes=CELL_MIN_GENES, tag="test")
x_test = x_test[m]
y_test = y_test[m]
cell_ids_test = cell_ids_test[m]

del m
gc.collect()

fprint(f"After cell filter: x_train {x_train.shape}, x_val {x_val.shape}, x_test {x_test.shape}")

# ---------------------- 模块6.5：class_weight ----------------------
fprint("Computing class_weight (balanced)...")
counts = Counter(y_train.tolist())
n_classes = len(counts)
n_samples = len(y_train)

class_weight = {cls: (n_samples / (n_classes * cnt)) for cls, cnt in counts.items()}
class_weight_pretty = {k: float(f"{v:.6f}") for k, v in class_weight.items()}

fprint(f"Train counts: {counts} | n_samples={n_samples}, n_classes={n_classes}")
fprint(f"class_weight = {class_weight_pretty}")

# ---------------------- 模块6.6：基因零比例过滤（只用 train 计算 gene_mask） ----------------------
fprint("Filtering genes by zero ratio (TRAIN-only, then apply to val/test)...")

nnz_train = x_train.getnnz(axis=0)
total_train_cells = x_train.shape[0]
zero_ratio_train = (total_train_cells - nnz_train) / total_train_cells

gene_mask = zero_ratio_train <= GENE_MAX_ZERO_RATIO

x_train = x_train[:, gene_mask]
x_val   = x_val[:, gene_mask]
x_test  = x_test[:, gene_mask]
rna_name = rna_name[gene_mask]
P_fdr    = P_fdr[gene_mask]

del nnz_train, total_train_cells, zero_ratio_train, gene_mask
gc.collect()

fprint("Deleted col sparse (TRAIN-only mask).")
fprint(f"Final data shape: x_train {x_train.shape}, x_val {x_val.shape}, x_test {x_test.shape}")
fprint(f"Final gene count: {x_train.shape[1]}")
fprint(f"Final label dist: train {Counter(y_train)}, val {Counter(y_val)}, test {Counter(y_test)}")

# ---------------------- 模块6.7：事后按 cell_id 挂 metadata（不参与建模，只为解释分析导出） ----------------------
fprint("Attaching metadata by cell_id (post hoc, for export only)...")

pheno_full = pd.read_csv(label_f, sep="\t")
pheno_full = pheno_full.rename(columns={pheno_full.columns[0]: "cell_id"})
pheno_full["cell_id"] = pheno_full["cell_id"].astype(str)
pheno_full = pheno_full.drop_duplicates(subset=["cell_id"]).reset_index(drop=True)

def build_meta(cell_ids_arr):
    return pd.DataFrame({"cell_id": cell_ids_arr.astype(str)}).merge(
        pheno_full, on="cell_id", how="left"
    )

meta_train = build_meta(cell_ids_train)
meta_val   = build_meta(cell_ids_val)
meta_test  = build_meta(cell_ids_test)

fprint(f"Meta attached: train {meta_train.shape}, val {meta_val.shape}, test {meta_test.shape}")

# ---------------------- 模块7：保存（不压缩） ----------------------
fprint(f"Saving files to {OUT_DIR} (NO compression)...")

# gene / fdr
pd.DataFrame(rna_name).to_csv(os.path.join(OUT_DIR, "rna_name.csv"), index=False, header=False)
pd.DataFrame(P_fdr).to_csv(os.path.join(OUT_DIR, "P_fdr2.csv"), index=False, header=False)

# X / y
pd.DataFrame(x_train.toarray()).to_csv(os.path.join(OUT_DIR, "training_sample.csv"), index=False, header=False)
pd.DataFrame(y_train).to_csv(os.path.join(OUT_DIR, "training_label.csv"), index=False, header=False)

pd.DataFrame(x_val.toarray()).to_csv(os.path.join(OUT_DIR, "validation_sample.csv"), index=False, header=False)
pd.DataFrame(y_val).to_csv(os.path.join(OUT_DIR, "validation_label.csv"), index=False, header=False)

pd.DataFrame(x_test.toarray()).to_csv(os.path.join(OUT_DIR, "testing_sample.csv"), index=False, header=False)
pd.DataFrame(y_test).to_csv(os.path.join(OUT_DIR, "testing_label.csv"), index=False, header=False)

# class_weight
pd.DataFrame(
    sorted(class_weight_pretty.items(), key=lambda x: x[0]),
    columns=["class", "weight"]
).to_csv(os.path.join(OUT_DIR, "class_weight.csv"), index=False)

# 新增：cell_id
pd.DataFrame(cell_ids_train).to_csv(os.path.join(OUT_DIR, "training_cell_id.csv"), index=False, header=False)
pd.DataFrame(cell_ids_val).to_csv(os.path.join(OUT_DIR, "validation_cell_id.csv"), index=False, header=False)
pd.DataFrame(cell_ids_test).to_csv(os.path.join(OUT_DIR, "testing_cell_id.csv"), index=False, header=False)

# 新增：完整 metadata
meta_train.to_csv(os.path.join(OUT_DIR, "training_meta.csv"), index=False)
meta_val.to_csv(os.path.join(OUT_DIR, "validation_meta.csv"), index=False)
meta_test.to_csv(os.path.join(OUT_DIR, "testing_meta.csv"), index=False)

# 新增：核心 metadata（后面 merge / 作图更方便）
core_cols = [c for c in ["cell_id", "sampID", "plateID", "status", "donorID", "group", "cellType", "SARS.CoV.2", "fluA"] if c in meta_train.columns]
meta_train[core_cols].to_csv(os.path.join(OUT_DIR, "training_meta_core.csv"), index=False)
meta_val[core_cols].to_csv(os.path.join(OUT_DIR, "validation_meta_core.csv"), index=False)
meta_test[core_cols].to_csv(os.path.join(OUT_DIR, "testing_meta_core.csv"), index=False)

del x_train, y_train, x_val, y_val, x_test, y_test, rna_name, P_fdr
del cell_ids_train, cell_ids_val, cell_ids_test, meta_train, meta_val, meta_test, pheno_full
gc.collect()

fprint(f"Finish! All files saved to {OUT_DIR}/")

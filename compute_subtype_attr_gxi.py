import os
import sys
import shutil
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 0. Repository paths (relative)
# =========================
CODE_DIR    = "."
HOLDOUT_DIR = os.path.join(CODE_DIR, "model_output", "M5_20260324_211757")
DATA_DIR    = os.path.join(CODE_DIR, "data")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# =========================
# 1. 导入模型依赖
# =========================
from cap import CapsuleLayer, squash
from trans import TransformerBlock
from gcn import GCNEncoder
from tgr_caps import TaskGuidedCapsuleLayer

# =========================
# 2. 文件路径
# =========================
META_FILE   = os.path.join(HOLDOUT_DIR, "test_main_with_celltype.csv")
GENE_FILE   = os.path.join(DATA_DIR, "rna_name.csv")
XTEST_FILE  = os.path.join(DATA_DIR, "testing_sample.csv")
ADJ_FILE    = os.path.join(DATA_DIR, "adj_holdout_train.npy")
MODEL_FILE  = os.path.join(HOLDOUT_DIR, "best_model.keras")

OUT_DIR = os.path.join(HOLDOUT_DIR, "subtype_attr_gxi")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# 3. 超参（与 M5 一致）
# =========================
N_CLASS = 3
GCN_HIDDEN_DIM = 32
DROPOUT = 0.1

# =========================
# 4. 模型定义（与训练时一致）
# =========================
class AblationModel(tf.keras.Model):
    def __init__(self, use_gcn, use_tgr, use_supcon, head, adj_matrix=None, n_class=3, dropout_rate=0.1, **kwargs):
        super(AblationModel, self).__init__(**kwargs)
        self.use_gcn = use_gcn
        self.use_tgr = use_tgr
        self.use_supcon = use_supcon
        self.head = head
        self.n_class = n_class
        self.tgr_alpha_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="tgr_alpha")

        if use_gcn:
            self.gcn_encoder = GCNEncoder(
                adj_matrix=adj_matrix,
                hidden_dim=GCN_HIDDEN_DIM,
                dropout_rate=dropout_rate,
                name="gcn_encoder"
            )

        self.primary_conv = tf.keras.layers.Conv1D(
            filters=8, kernel_size=1, strides=1,
            padding="valid", name="primary_cap_conv"
        )

        if use_tgr:
            self.caps_layer = TaskGuidedCapsuleLayer(
                num_capsule=n_class,
                dim_capsule=16,
                n_class=n_class,
                routings=3,
                name="tgr_caps"
            )
        else:
            self.caps_layer = CapsuleLayer(
                num_capsule=n_class,
                dim_capsule=16,
                routings=3,
                name="groupcaps"
            )

        self.transformer = TransformerBlock(embed_dim=16, num_heads=4, ff_dim=256)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        if head == "flatten":
            self.flatten_layer = tf.keras.layers.Flatten()
            self.dense1 = tf.keras.layers.Dense(150, activation="relu", name="dense1")
            self.cls = tf.keras.layers.Dense(n_class, activation="softmax", name="cls")

    def call(self, x, labels=None, training=False, return_all=False):
        if self.use_gcn:
            x = self.gcn_encoder(x, training=training)

        h = squash(self.primary_conv(x))

        if self.use_tgr:
            alpha = self.tgr_alpha_var if training else 0.0
            caps_out = self.caps_layer(h, labels=labels, alpha=alpha, training=training)
        else:
            caps_out, _ = self.caps_layer(h)

        x_t = self.transformer(caps_out)

        if self.head == "norm":
            evidence = tf.norm(x_t, axis=-1)
            logits = tf.nn.softmax(evidence, axis=-1)
        else:
            x_f = self.dense1(self.dropout(self.flatten_layer(x_t), training=training))
            evidence = None
            logits = self.cls(x_f)

        if return_all:
            return caps_out, x_t, evidence, logits
        return caps_out, logits

# =========================
# 5. 读数据
# =========================
meta = pd.read_csv(META_FILE)
gene_names = pd.read_csv(GENE_FILE, header=None).iloc[:, 0].astype(str).to_numpy()
x_test = pd.read_csv(XTEST_FILE, header=None).to_numpy(dtype=np.float32)
x_test_in = np.expand_dims(x_test, -1)

print("meta shape :", meta.shape)
print("x_test shape:", x_test.shape)
print("gene count  :", len(gene_names))

# =========================
# 6. 建模
# =========================
adj_matrix = np.load(ADJ_FILE)

model = AblationModel(
    use_gcn=True,
    use_tgr=True,
    use_supcon=True,
    head="norm",
    adj_matrix=adj_matrix,
    n_class=N_CLASS,
    dropout_rate=DROPOUT
)

_ = model(
    tf.zeros((1, x_test.shape[1], 1), dtype=tf.float32),
    labels=None, training=False
)

# =========================
# 7. 加载权重（最终兼容补丁）
# =========================
_orig_getitem = h5py._hl.attrs.AttributeManager.__getitem__

def _patched_getitem(self, name):
    val = _orig_getitem(self, name)

    if isinstance(val, str):
        return val.encode("utf-8")

    if isinstance(val, np.ndarray):
        if val.dtype.kind in ("U", "O"):
            fixed = []
            for x in val.tolist():
                if isinstance(x, str):
                    fixed.append(x.encode("utf-8"))
                else:
                    fixed.append(x)
            return np.array(fixed, dtype=object)
        return val

    if isinstance(val, (list, tuple)):
        fixed = []
        for x in val:
            if isinstance(x, str):
                fixed.append(x.encode("utf-8"))
            else:
                fixed.append(x)
        return type(val)(fixed)

    return val

h5py._hl.attrs.AttributeManager.__getitem__ = _patched_getitem

MODEL_WEIGHTS = os.path.join(HOLDOUT_DIR, "best_model.weights.h5")
if not os.path.exists(MODEL_WEIGHTS):
    shutil.copy2(MODEL_FILE, MODEL_WEIGHTS)

print("Loading weights from:", MODEL_WEIGHTS)
model.load_weights(MODEL_WEIGHTS)
print("Weights loaded successfully.")

# =========================
# 8. 任务定义
# =========================
TASKS = [
    {
        "name": "G5c_naive_fluA",
        "cellType": "G5c_naive",
        "status": "fluA",
        "y_true": 2,
        "y_pred": 2,
        "channel_idx": 2
    },
    {
        "name": "hillock_covid",
        "cellType": "hillock",
        "status": "COVID-19",
        "y_true": 1,
        "y_pred": 1,
        "channel_idx": 1
    },
    {
        "name": "ABC_covid",
        "cellType": "ABC",
        "status": "COVID-19",
        "y_true": 1,
        "y_pred": 1,
        "channel_idx": 1
    },
    {
        "name": "unk_epi_fluA",
        "cellType": "unk_epi",
        "status": "fluA",
        "y_true": 2,
        "y_pred": 2,
        "channel_idx": 2
    }
]

# =========================
# 9. Gradient × Input
# =========================
def gradient_x_input(model, x_batch, channel_idx):
    x_tensor = tf.convert_to_tensor(x_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        _, _, evidence, _ = model(x_tensor, labels=None, training=False, return_all=True)
        target = evidence[:, channel_idx]

    grads = tape.gradient(target, x_tensor)
    gxi = grads * x_tensor
    return gxi.numpy().squeeze(-1), evidence.numpy()

# =========================
# 10. 逐 subtype 运行并保存
# =========================
for task in TASKS:
    print("\n==============================")
    print("Running:", task["name"])
    print("==============================")

    sub = meta[
        (meta["cellType"] == task["cellType"]) &
        (meta["status"] == task["status"]) &
        (meta["y_true"] == task["y_true"]) &
        (meta["y_pred"] == task["y_pred"])
    ].copy()

    idx = sub.index.to_numpy()
    print("selected cells:", len(idx))

    if len(idx) == 0:
        print("Skip: no cells selected.")
        continue

    x_sub = x_test_in[idx]

    gxi, evidence = gradient_x_input(model, x_sub, task["channel_idx"])
    np.save(os.path.join(OUT_DIR, f"{task['name']}_gxi.npy"), gxi)

    mean_attr = gxi.mean(axis=0)
    mean_pos_attr = np.where(mean_attr > 0, mean_attr, 0)
    mean_neg_attr = np.where(mean_attr < 0, mean_attr, 0)

    rank_df = pd.DataFrame({
        "gene": gene_names,
        "mean_attr": mean_attr,
        "mean_pos_attr": mean_pos_attr,
        "mean_neg_attr": mean_neg_attr
    }).sort_values("mean_attr", ascending=False)

    rank_df.to_csv(os.path.join(OUT_DIR, f"{task['name']}_ranked_genes.csv"), index=False)
    rank_df.head(50).to_csv(os.path.join(OUT_DIR, f"{task['name']}_top_positive_genes.csv"), index=False)
    rank_df.sort_values("mean_attr", ascending=True).head(50).to_csv(
        os.path.join(OUT_DIR, f"{task['name']}_top_negative_genes.csv"), index=False
    )

    cell_df = sub[[
        "cell_id", "sampID", "plateID", "status",
        "donorID", "group", "cellType", "y_true", "y_pred"
    ]].copy()
    cell_df["target_evidence"] = evidence[:, task["channel_idx"]]
    cell_df.to_csv(os.path.join(OUT_DIR, f"{task['name']}_cell_metadata.csv"), index=False)

    print("saved:", f"{task['name']}_top_positive_genes.csv")

print("\nAll done.")
print("Output dir:", OUT_DIR)
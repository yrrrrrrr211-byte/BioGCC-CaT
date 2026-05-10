import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tensorflow.keras import layers, models, callbacks, optimizers

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import label_binarize

from cap import CapsuleLayer, squash, PrimaryCap
from trans import TransformerBlock
from gcn import build_gene_graph, GCNEncoder
from tgr_caps import TaskGuidedCapsuleLayer
from losses import supervised_contrastive_loss

# =========================
# 基本超参
# =========================
SEED        = 42
N_CLASS     = 3
LABEL_NAME  = {0: "healthy", 1: "COVID-19", 2: "fluA"}
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 1e-3  
PATIENCE    = 4
DROPOUT     = 0.1
ALPHA_COVID = 0.98

# BioGCC-CaT 超参
GCN_THRESHOLD  = 0.6
GCN_HIDDEN_DIM = 32
LAMBDA_SUPCON  = 0.5
SUPCON_TEMP    = 0.07

# Default data directory (can be overridden by command line argument)
# Users should download data from GEO GSE176269 and preprocess using data_processing.py
DEFAULT_DATA_DIR = "./data"

# M0 = 现在能跑的纯 scCaT baseline
# M1-M5 = 当前能跑通的冠军 5fold 脚本配置
MODE_CONFIG = {
    #         use_gcn  use_tgr  use_supcon  head
    "M0":  (  False,   False,   False,      "flatten" ),  # 纯 CaT baseline
    "M1":  (  True,    False,   False,      "flatten" ),
    "M2":  (  False,   True,    False,      "flatten" ),
    "M3":  (  False,   False,   True,       "flatten" ),
    "M4":  (  False,   False,   False,      "norm"    ),
    "M5":  (  True,    True,    True,       "norm"    ),
}


# =========================
# 工具函数
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def load_x(path):
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def load_y(path):
    return pd.read_csv(path, header=None).to_numpy().squeeze().astype(np.int32)


def compute_class_weight(y_train):
    counts = Counter(y_train.tolist())
    n = len(y_train)
    cw = {c: n / (N_CLASS * counts.get(c, 1)) for c in range(N_CLASS)}
    cw[1] *= ALPHA_COVID
    return cw


def maybe_find_gene_names(data_dir, n_genes):
    candidates = [
        os.path.join(data_dir, "rna_name.csv"),
        os.path.join(data_dir, "gene_names.csv"),
        os.path.join(data_dir, "selected_genes.csv"),
        os.path.join(os.path.dirname(data_dir), "rna_name.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            arr = pd.read_csv(p, header=None).to_numpy().squeeze().astype(str)
            if len(arr) == n_genes:
                return arr
    return np.array(["gene_%d" % i for i in range(n_genes)], dtype=str)


# =========================
# M0：沿用当前能跑的纯scCaT baseline
# =========================
def build_CaT_3class(input_shape, n_class=3, routings=5):
    inputs = layers.Input(shape=input_shape)

    primarycaps = PrimaryCap(
        inputs,
        dim_capsule=8,
        n_channels=1,
        kernel_size=1,
        strides=1,
        padding="valid"
    )

    groupcaps, _ = CapsuleLayer(
        num_capsule=n_class,
        dim_capsule=16,
        routings=routings,
        name="groupcaps"
    )(primarycaps)

    transformer_block = TransformerBlock(embed_dim=16, num_heads=2, ff_dim=128)
    x = transformer_block(groupcaps)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(150, activation="relu")(x)
    outputs = layers.Dense(n_class, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="CaT_3class_M0")
    return model


# =========================
# M1-M5：沿用当前能跑通的统一模型类
# =========================
class AlphaDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, initial_alpha=0.05, decay_rate=0.95, warm_up_epochs=8):
        super(AlphaDecayCallback, self).__init__()
        self.initial_alpha = float(initial_alpha)
        self.decay_rate = float(decay_rate)
        self.warm_up_epochs = int(warm_up_epochs)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warm_up_epochs:
            new_alpha = 0.0
        else:
            new_alpha = self.initial_alpha * (self.decay_rate ** (epoch - self.warm_up_epochs))

        self.model.tgr_alpha_var.assign(float(new_alpha))

        if epoch == self.warm_up_epochs:
            print("\n[TGR 觉醒] Epoch %d: 当前 Alpha = %.4f" % (epoch + 1, new_alpha))
        elif epoch > self.warm_up_epochs and (epoch - self.warm_up_epochs) % 10 == 0:
            print("  [TGR 衰减] Epoch %d: Alpha = %.4f" % (epoch + 1, new_alpha))


class AblationModel(tf.keras.Model):
    def __init__(self, use_gcn, use_tgr, use_supcon, head,
                 adj_matrix=None, n_class=3, dropout_rate=0.1, **kwargs):
        super(AblationModel, self).__init__(**kwargs)

        self.use_gcn = use_gcn
        self.use_tgr = use_tgr
        self.use_supcon = use_supcon
        self.head = head
        self.n_class = n_class

        self.tgr_alpha_var = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="tgr_alpha")

        if use_gcn:
            assert adj_matrix is not None, "use_gcn=True 时必须传入 adj_matrix"
            self.gcn_encoder = GCNEncoder(
                adj_matrix=adj_matrix,
                hidden_dim=GCN_HIDDEN_DIM,
                dropout_rate=dropout_rate,
                name="gcn_encoder"
            )

        self.primary_conv = layers.Conv1D(
            filters=8, kernel_size=1, strides=1, padding="valid",
            name="primary_cap_conv"
        )

        if use_tgr:
            self.caps_layer = TaskGuidedCapsuleLayer(
                num_capsule=n_class, dim_capsule=16,
                n_class=n_class, routings=3, name="tgr_caps"
            )
        else:
            self.caps_layer = CapsuleLayer(
                num_capsule=n_class, dim_capsule=16,
                routings=3, name="groupcaps"
            )

        self.transformer = TransformerBlock(embed_dim=16, num_heads=4, ff_dim=256)

        self.dropout = layers.Dropout(dropout_rate)
        if head == "flatten":
            self.flatten_layer = layers.Flatten()
            self.dense1 = layers.Dense(150, activation="relu", name="dense1")
            self.cls = layers.Dense(n_class, activation="softmax", name="cls")
        elif head == "norm":
            pass
        else:
            raise ValueError("未知 head: %s" % head)

    def call(self, x, labels=None, training=False, return_all=False):
        if self.use_gcn:
            x = self.gcn_encoder(x, training=training)

        h = self.primary_conv(x)
        h = squash(h)

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
            x_f = self.flatten_layer(x_t)
            x_f = self.dropout(x_f, training=training)
            x_f = self.dense1(x_f)
            evidence = None
            logits = self.cls(x_f)

        if return_all:
            return caps_out, x_t, evidence, logits
        return caps_out, logits

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            caps_out, logits = self(x, labels=y, training=True)
            loss_ce = self.compiled_loss(
                y, logits,
                sample_weight=sample_weight,
                regularization_losses=self.losses
            )

            if self.use_supcon:
                caps_norm = tf.norm(caps_out, axis=-1)
                loss_supcon = supervised_contrastive_loss(
                    caps_norm, y, temperature=SUPCON_TEMP
                )
                total_loss = loss_ce + LAMBDA_SUPCON * loss_supcon
            else:
                total_loss = loss_ce
                loss_supcon = tf.constant(0.0)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, logits, sample_weight=sample_weight)

        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = total_loss
        results["loss_ce"] = loss_ce
        results["loss_supcon"] = loss_supcon
        return results

    def test_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        _, logits = self(x, labels=None, training=False)

        loss = self.compiled_loss(
            y, logits,
            sample_weight=sample_weight,
            regularization_losses=self.losses
        )
        self.compiled_metrics.update_state(y, logits, sample_weight=sample_weight)

        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = loss
        return results

    def predict_proba(self, X, batch_size=256):
        all_proba = []
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size]
            _, logits = self(xb, labels=None, training=False)
            all_proba.append(logits.numpy())
        return np.concatenate(all_proba, axis=0)


# =========================
# 评估与导出
# =========================
def get_proba(model, X, batch_size=256):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X, batch_size=batch_size)
    return model.predict(X, batch_size=batch_size, verbose=0)


def eval_split(model, name, X, y, out_dir):
    proba = get_proba(model, X, batch_size=256)
    pred = np.argmax(proba, axis=1)

    acc = accuracy_score(y, pred)
    bacc = balanced_accuracy_score(y, pred)
    f1_macro = f1_score(y, pred, average="macro")
    f1_weighted = f1_score(y, pred, average="weighted")

    try:
        auroc_macro = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    except Exception:
        auroc_macro = float("nan")

    y_bin = label_binarize(y, classes=[0, 1, 2])
    try:
        auprc_macro = average_precision_score(y_bin, proba, average="macro")
    except Exception:
        auprc_macro = float("nan")

    cm = confusion_matrix(y, pred)

    report_text = classification_report(
        y, pred,
        labels=[0, 1, 2],
        target_names=[LABEL_NAME[i] for i in range(N_CLASS)],
        digits=4,
        zero_division=0
    )

    report_dict = classification_report(
        y, pred,
        labels=[0, 1, 2],
        target_names=[LABEL_NAME[i] for i in range(N_CLASS)],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T

    print("\n===== %s METRICS =====" % name.upper())
    print("ACC        =", round(acc, 6))
    print("BACC       =", round(bacc, 6))
    print("F1(macro)  =", round(f1_macro, 6))
    print("F1(weight) =", round(f1_weighted, 6))
    print("AUROC(macro, ovr) =", auroc_macro)
    print("AUPRC(macro)      =", auprc_macro)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(cm)
    print("\nPer-class report:")
    print(report_text)

    with open(os.path.join(out_dir, "%s_metrics.txt" % name), "w", encoding="utf-8") as f:
        f.write("===== %s METRICS =====\n" % name.upper())
        f.write("ACC        = %.6f\n" % acc)
        f.write("BACC       = %.6f\n" % bacc)
        f.write("F1(macro)  = %.6f\n" % f1_macro)
        f.write("F1(weight) = %.6f\n" % f1_weighted)
        f.write("AUROC(macro, ovr) = %s\n" % str(auroc_macro))
        f.write("AUPRC(macro)      = %s\n\n" % str(auprc_macro))
        f.write("Confusion Matrix (rows=true, cols=pred):\n")
        f.write(str(cm) + "\n\n")
        f.write("Per-class report:\n")
        f.write(report_text + "\n")

    pd.DataFrame(cm).to_csv(os.path.join(out_dir, "%s_confusion_matrix.csv" % name), index=False)
    report_df.to_csv(os.path.join(out_dir, "%s_classification_report.csv" % name))
    pd.DataFrame(proba, columns=["p_%s" % LABEL_NAME[i] for i in range(N_CLASS)]).to_csv(
        os.path.join(out_dir, "%s_proba.csv" % name), index=False
    )
    pd.DataFrame({"y_true": y, "y_pred": pred}).to_csv(
        os.path.join(out_dir, "%s_pred.csv" % name), index=False
    )

    return {
        "ACC": acc,
        "BACC": bacc,
        "F1_macro": f1_macro,
        "F1_weighted": f1_weighted,
        "AUROC_macro": auroc_macro,
        "AUPRC_macro": auprc_macro,
    }


def export_intermediate_outputs(model, X, split_name, out_dir):
    if not isinstance(model, AblationModel):
        return

    caps_all = []
    xt_all = []
    evidence_all = []
    logits_all = []

    batch_size = 256
    for start in range(0, len(X), batch_size):
        xb = X[start:start + batch_size]
        caps_out, x_t, evidence, logits = model(xb, labels=None, training=False, return_all=True)
        caps_all.append(caps_out.numpy())
        xt_all.append(x_t.numpy())
        logits_all.append(logits.numpy())
        if evidence is not None:
            evidence_all.append(evidence.numpy())

    np.save(os.path.join(out_dir, "%s_caps_out.npy" % split_name), np.concatenate(caps_all, axis=0))
    np.save(os.path.join(out_dir, "%s_x_t.npy" % split_name), np.concatenate(xt_all, axis=0))
    np.save(os.path.join(out_dir, "%s_logits.npy" % split_name), np.concatenate(logits_all, axis=0))
    if len(evidence_all) > 0:
        np.save(os.path.join(out_dir, "%s_evidence.npy" % split_name), np.concatenate(evidence_all, axis=0))


# =========================
# 主函数
# =========================
def main(args):
    enable_gpu_memory_growth()
    set_seed(SEED)

    data_dir = args.data_dir if args.data_dir else DEFAULT_DATA_DIR
    mode = args.mode
    use_gcn, use_tgr, use_supcon, head = MODE_CONFIG[mode]

    print("=" * 70)
    print("Holdout main model | MODE=%s | GCN=%s TGR=%s SupCon=%s Head=%s" % (
        mode, use_gcn, use_tgr, use_supcon, head
    ))
    print("=" * 70)

    x_train = load_x(os.path.join(data_dir, "training_sample.csv"))
    y_train = load_y(os.path.join(data_dir, "training_label.csv"))
    x_val   = load_x(os.path.join(data_dir, "validation_sample.csv"))
    y_val   = load_y(os.path.join(data_dir, "validation_label.csv"))
    x_test  = load_x(os.path.join(data_dir, "testing_sample.csv"))
    y_test  = load_y(os.path.join(data_dir, "testing_label.csv"))

    print("===== DATA SHAPES =====")
    print("x_train:", x_train.shape, "y_train:", y_train.shape, "dist:", Counter(y_train))
    print("x_val:  ", x_val.shape,   "y_val:  ", y_val.shape,   "dist:", Counter(y_val))
    print("x_test: ", x_test.shape,  "y_test: ", y_test.shape,  "dist:", Counter(y_test))
    print("=======================")

    n_genes = x_train.shape[1]
    gene_names = maybe_find_gene_names(data_dir, n_genes)

    adj_matrix = None
    if use_gcn:
        adj_path = os.path.join(data_dir, "adj_holdout_train.npy")
        if os.path.exists(adj_path):
            print("[GCN] 加载邻接矩阵:", adj_path)
            adj_matrix = np.load(adj_path)
        else:
            print("[GCN] 基于 x_train 构建邻接矩阵 ...")
            adj_matrix = build_gene_graph(x_train.astype(np.float32), threshold=GCN_THRESHOLD)
            np.save(adj_path, adj_matrix)

    x_train_in = np.expand_dims(x_train, axis=-1)
    x_val_in   = np.expand_dims(x_val, axis=-1)
    x_test_in  = np.expand_dims(x_test, axis=-1)

    print("Input reshaped to:", x_train_in.shape)

    out_dir = os.path.join("holdout_out", "%s_%s" % (mode, time.strftime("%Y%m%d_%H%M%S")))
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(gene_names).to_csv(os.path.join(out_dir, "final_gene_list.csv"), index=False, header=False)

    if mode == "M0":
        class_weight = compute_class_weight(y_train)
        print("===== CLASS_WEIGHT (balanced + tuned) =====")
        print("counts:", Counter(y_train.tolist()))
        print("ALPHA_COVID:", ALPHA_COVID)
        print("class_weight =", {k: float("%.6f" % v) for k, v in class_weight.items()})
        print("===========================================")

        model = build_CaT_3class(input_shape=(n_genes, 1), n_class=N_CLASS, routings=5)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.keras"), monitor="val_loss", save_best_only=True),
            callbacks.CSVLogger(os.path.join(out_dir, "train_log.csv")),
        ]

        history = model.fit(
            x_train_in, y_train,
            validation_data=(x_val_in, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=cbs,
            verbose=1
        )
    else:
        model = AblationModel(
            use_gcn=use_gcn, use_tgr=use_tgr, use_supcon=use_supcon, head=head,
            adj_matrix=adj_matrix, n_class=N_CLASS, dropout_rate=DROPOUT, name="HoldoutModel_%s" % mode
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )

        cw = compute_class_weight(y_train)
        sample_weight_tr = np.array([cw[int(yi)] for yi in y_train], dtype=np.float32)
        print("训练分布：%s | class_weight=%s" % (Counter(y_train.tolist()), cw))

        cbs = [
            callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
            callbacks.ModelCheckpoint(os.path.join(out_dir, "best_model.keras"), monitor="val_loss", save_best_only=True),
            callbacks.CSVLogger(os.path.join(out_dir, "train_log.csv")),
        ]
        if use_tgr:
            cbs.append(AlphaDecayCallback(initial_alpha=0.05, decay_rate=0.95, warm_up_epochs=5))

        history = model.fit(
            x_train_in, y_train,
            sample_weight=sample_weight_tr,
            validation_data=(x_val_in, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=cbs,
            verbose=1,
        )

    pd.DataFrame(history.history).to_csv(os.path.join(out_dir, "history.csv"), index=False)

    eval_split(model, "val", x_val_in, y_val, out_dir)
    eval_split(model, "test", x_test_in, y_test, out_dir)

    export_intermediate_outputs(model, x_val_in, "val", out_dir)
    export_intermediate_outputs(model, x_test_in, "test", out_dir)

    print("\nAll outputs saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["M0", "M1", "M2", "M3", "M4", "M5"],
                        help="主模型模式")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="固定 train/val/test 所在目录")
    args = parser.parse_args()
    main(args)

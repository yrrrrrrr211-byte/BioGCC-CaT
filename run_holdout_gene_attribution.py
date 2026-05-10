#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Held-out attribution for the final M5 model.

功能：
1. 重建 held-out 的最终 M5 模型
2. 加载 best_model.keras 权重
3. 读取 testing_sample.csv / testing_label.csv
4. 对 healthy / covid / flua 三个 evidence channel 做 Grad × Input attribution
5. 导出：
   - heldout_healthy_gene_attr.csv
   - heldout_covid_gene_attr.csv
   - heldout_flua_gene_attr.csv
   - *_top300_positive_genes.csv
   - *_top300_negative_genes.csv

注意：
- 这是给最终主模型 M5 用的，不是给 5-fold 用的
- 当前脚本默认用最新的 holdout_out/M5* 目录
"""

import os
import sys
import glob
import gc
import importlib.util
import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# [0] 路径配置
# ============================================================
CODE_DIR = "/root/autodl-tmp/scCAT/code"
DATA_DIR = "/root/autodl-tmp/scCAT/code/data(cell-level)"
RESULT_ROOT = "/root/autodl-tmp/scCAT/code/holdout_out"

AUTO_FIND_LATEST_M5 = True
MANUAL_RUN_DIR = "/root/autodl-tmp/scCAT/code/holdout_out/M5_20260324_211757"

MODE = "M5"
BATCH_SIZE = 64
TOP_N = 300

TARGET_CLASSES = {
    "healthy": 0,
    "covid": 1,
    "flua": 2,
}


# ============================================================
# [1] 工具函数
# ============================================================
def enable_gpu_memory_growth():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def find_latest_m5_run(result_root):
    cands = sorted(glob.glob(os.path.join(result_root, "M5*")))
    cands = [p for p in cands if os.path.isdir(p)]
    if len(cands) == 0:
        raise FileNotFoundError(f"在 {result_root} 下没有找到 M5 目录")
    return sorted(cands, key=lambda x: os.path.getmtime(x))[-1]


def patch_h5py_for_legacy_tf():
    import h5py
    try:
        attr_cls = h5py._hl.attrs.AttributeManager
    except Exception:
        return None

    old_getitem = attr_cls.__getitem__

    def _patched_getitem(self, name):
        out = old_getitem(self, name)
        if name in ("keras_version", "backend") and isinstance(out, str):
            return out.encode("utf8")
        if name in ("layer_names", "weight_names"):
            try:
                if hasattr(out, "__len__") and len(out) > 0 and isinstance(out[0], str):
                    return np.array([x.encode("utf8") for x in out])
            except Exception:
                pass
        return out

    attr_cls.__getitem__ = _patched_getitem
    return old_getitem


def unpatch_h5py(old_getitem):
    if old_getitem is None:
        return
    try:
        import h5py
        h5py._hl.attrs.AttributeManager.__getitem__ = old_getitem
    except Exception:
        pass


def load_x(path):
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


def load_y(path):
    return pd.read_csv(path, header=None).to_numpy().squeeze().astype(np.int32)


def load_gene_names(data_dir, run_dir, n_genes):
    # 优先用 run_dir/final_gene_list.csv
    fp1 = os.path.join(run_dir, "final_gene_list.csv")
    if os.path.exists(fp1):
        df = pd.read_csv(fp1, header=None)
        genes = df.iloc[:, 0].astype(str).tolist()
        if len(genes) == n_genes:
            print(f"[INFO] 使用 run_dir/final_gene_list.csv，gene 数={len(genes)}")
            return np.array(genes, dtype=str)

    # 其次尝试 data_dir 下的 gene 名文件
    candidates = [
        os.path.join(data_dir, "rna_name.csv"),
        os.path.join(data_dir, "gene_names.csv"),
        os.path.join(data_dir, "final_gene_list.csv"),
    ]
    for fp in candidates:
        if os.path.exists(fp):
            df = pd.read_csv(fp, header=None)
            genes = df.iloc[:, 0].astype(str).tolist()
            if len(genes) == n_genes:
                print(f"[INFO] 使用 {fp}，gene 数={len(genes)}")
                return np.array(genes, dtype=str)

    raise FileNotFoundError(
        f"无法找到与 n_genes={n_genes} 对应的 gene name 文件。"
    )


def import_bio_module(code_dir):
    bio_path = os.path.join(code_dir, "BIOtrain.py")
    if not os.path.exists(bio_path):
        raise FileNotFoundError(f"没找到 BIOtrain.py: {bio_path}")

    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    spec = importlib.util.spec_from_file_location("bio_holdout_module", bio_path)
    bio = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bio)
    return bio


def rebuild_model_from_bio(bio, mode, input_shape, data_dir):
    if mode not in bio.MODE_CONFIG:
        raise ValueError(f"未知 mode: {mode}")

    use_gcn, use_tgr, use_supcon, head = bio.MODE_CONFIG[mode]
    n_genes = input_shape[0]

    adj_matrix = None
    if use_gcn:
        adj_path = os.path.join(data_dir, "adj_holdout_train.npy")
        if not os.path.exists(adj_path):
            raise FileNotFoundError(f"需要 GCN 邻接矩阵，但没找到: {adj_path}")
        print(f"[GCN] 加载邻接矩阵: {adj_path}")
        adj_matrix = np.load(adj_path)

    if mode == "M0":
        model = bio.build_CaT_3class(
            input_shape=(n_genes, 1),
            n_class=bio.N_CLASS,
            routings=5
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=bio.LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )
    else:
        model = bio.AblationModel(
            use_gcn=use_gcn,
            use_tgr=use_tgr,
            use_supcon=use_supcon,
            head=head,
            adj_matrix=adj_matrix,
            n_class=bio.N_CLASS,
            dropout_rate=bio.DROPOUT,
            name="HeldoutAttribution_%s" % mode
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=bio.LR),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
        )

    return model


def _call_model(model, xb):
    """
    当前 BIOtrain.py 的 AblationModel:
      return_all=False -> (caps_out, logits)
      return_all=True  -> (caps_out, x_t, evidence, logits)

    这里必须显式 return_all=True，否则拿不到 evidence。
    """
    out = model(xb, labels=None, training=False, return_all=True)

    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise ValueError("当前模型未按预期返回 (caps_out, x_t, evidence, logits)")

    caps_out, x_t, evidence, logits = out
    return caps_out, x_t, evidence, logits


def load_trained_model(bio, model_path, mode, input_shape, data_dir):
    print("[INFO] 重建模型结构并加载权重...")
    model = rebuild_model_from_bio(bio, mode=mode, input_shape=input_shape, data_dir=data_dir)

    dummy = tf.zeros((1,) + tuple(input_shape), dtype=tf.float32)
    if mode == "M0":
        _ = model(dummy, training=False)
    else:
        try:
            _ = model(dummy, labels=None, training=False, return_all=True)
        except TypeError:
            _ = model(dummy, labels=None, training=False)

    old_getitem = patch_h5py_for_legacy_tf()
    try:
        model.load_weights(model_path)
    finally:
        unpatch_h5py(old_getitem)

    print(f"[INFO] 权重加载完成: {model_path}")
    return model


def compute_gradxinput_attr(model, X_input, target_class_idx, batch_size=64):
    """
    X_input: (N, G, 1)
    对 class-specific evidence channel 做 Grad × Input attribution
    """
    n_cells, n_genes, _ = X_input.shape

    signed_sum = np.zeros(n_genes, dtype=np.float64)
    abs_sum = np.zeros(n_genes, dtype=np.float64)
    pos_count = np.zeros(n_genes, dtype=np.float64)
    total_count = 0

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        xb = tf.convert_to_tensor(X_input[start:end], dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(xb)
            caps_out, x_t, evidence, logits = _call_model(model, xb)

            if evidence is None:
                raise RuntimeError("当前模型 return_all=True 后 evidence 仍为 None。")

            target_score = evidence[:, target_class_idx]
            objective = tf.reduce_sum(target_score)

        grads = tape.gradient(objective, xb)
        grad_x_input = grads * xb
        gxi = tf.squeeze(grad_x_input, axis=-1).numpy()  # (B, G)

        signed_sum += gxi.sum(axis=0)
        abs_sum += np.abs(gxi).sum(axis=0)
        pos_count += (gxi > 0).sum(axis=0)
        total_count += gxi.shape[0]

        del xb, caps_out, x_t, evidence, logits, grads, grad_x_input, gxi
        gc.collect()

    attr_mean_signed = signed_sum / max(total_count, 1)
    attr_mean_abs = abs_sum / max(total_count, 1)
    attr_pos_frac = pos_count / max(total_count, 1)

    return attr_mean_signed, attr_mean_abs, attr_pos_frac


def save_attr_outputs(out_dir, target_name, gene_names, attr_mean_signed, attr_mean_abs, attr_pos_frac, n_cells):
    df = pd.DataFrame({
        "target_class": target_name,
        "gene": gene_names,
        "attr_mean_signed": attr_mean_signed,
        "attr_mean_abs": attr_mean_abs,
        "attr_pos_frac": attr_pos_frac,
        "n_cells": n_cells
    })

    full_fp = os.path.join(out_dir, f"heldout_{target_name}_gene_attr.csv")
    pos_fp = os.path.join(out_dir, f"{target_name}_top{TOP_N}_positive_genes.csv")
    neg_fp = os.path.join(out_dir, f"{target_name}_top{TOP_N}_negative_genes.csv")

    df.sort_values("attr_mean_signed", ascending=False).to_csv(full_fp, index=False)
    df.sort_values("attr_mean_signed", ascending=False).head(TOP_N).to_csv(pos_fp, index=False)
    df.sort_values("attr_mean_signed", ascending=True).head(TOP_N).to_csv(neg_fp, index=False)

    print(f"[OK] 保存: {full_fp}")
    print(f"[OK] 保存: {pos_fp}")
    print(f"[OK] 保存: {neg_fp}")


# ============================================================
# [2] 主流程
# ============================================================
def main():
    enable_gpu_memory_growth()

    run_dir = find_latest_m5_run(RESULT_ROOT) if AUTO_FIND_LATEST_M5 else MANUAL_RUN_DIR
    model_path = os.path.join(run_dir, "best_model.keras")
    out_dir = os.path.join(run_dir, "heldout_attribution")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("Held-out attribution for final M5 model")
    print("RUN_DIR :", run_dir)
    print("MODEL   :", model_path)
    print("DATA_DIR:", DATA_DIR)
    print("=" * 80)

    x_test = load_x(os.path.join(DATA_DIR, "testing_sample.csv"))
    y_test = load_y(os.path.join(DATA_DIR, "testing_label.csv"))
    x_test = np.expand_dims(x_test.astype(np.float32), axis=-1)

    print(f"[INFO] x_test shape: {x_test.shape}")
    print(f"[INFO] y_test shape: {y_test.shape}")
    print(f"[INFO] test label counts: {pd.Series(y_test).value_counts().sort_index().to_dict()}")

    gene_names = load_gene_names(DATA_DIR, run_dir, n_genes=x_test.shape[1])

    bio = import_bio_module(CODE_DIR)
    model = load_trained_model(
        bio=bio,
        model_path=model_path,
        mode=MODE,
        input_shape=x_test.shape[1:],
        data_dir=DATA_DIR
    )

    summary_rows = []

    for target_name, target_idx in TARGET_CLASSES.items():
        X_target = x_test[y_test == target_idx]
        print(f"\n[ATTR] target={target_name} | n_cells={len(X_target)}")

        attr_mean_signed, attr_mean_abs, attr_pos_frac = compute_gradxinput_attr(
            model=model,
            X_input=X_target,
            target_class_idx=target_idx,
            batch_size=BATCH_SIZE
        )

        save_attr_outputs(
            out_dir=out_dir,
            target_name=target_name,
            gene_names=gene_names,
            attr_mean_signed=attr_mean_signed,
            attr_mean_abs=attr_mean_abs,
            attr_pos_frac=attr_pos_frac,
            n_cells=len(X_target)
        )

        top_pos = pd.DataFrame({
            "gene": gene_names,
            "score": attr_mean_signed
        }).sort_values("score", ascending=False).head(10)

        for _, row in top_pos.iterrows():
            summary_rows.append({
                "target_class": target_name,
                "gene": row["gene"],
                "attr_mean_signed": row["score"]
            })

        gc.collect()

    summary_fp = os.path.join(out_dir, "top10_summary_for_quick_check.csv")
    pd.DataFrame(summary_rows).to_csv(summary_fp, index=False)
    print(f"\n[OK] 保存快速检查文件: {summary_fp}")

    print(f"\n✅ 完成。输出目录: {out_dir}")


if __name__ == "__main__":
    main()
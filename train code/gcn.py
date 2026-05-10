"""
gcn.py —— 基因共表达图卷积预编码器

职责：
    在 PrimaryCap 之前，用图卷积让每个基因聚合邻居基因的表达信息。
    输出与输入 shape 完全一致：(batch, n_genes, 1)
    通过残差连接保证不会比原模型更差。

使用方式：
    from gcn import build_gene_graph, GCNEncoder

    # Step1：训练前跑一次，保存邻接矩阵（用全量数据）
    adj = build_gene_graph(X_all)          # X_all: (N, n_genes)
    np.save("adj.npy", adj)

    # Step2：模型内部使用
    gcn = GCNEncoder(adj=adj, name="gcn")
    h = gcn(x)                             # (batch, n_genes, 1) → (batch, n_genes, 1)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：构建归一化邻接矩阵
# 输入：X_train shape (N_cells, N_genes)，表达矩阵（未加 channel 维）
# 输出：adj_norm shape (N_genes, N_genes)，对称归一化稀疏邻接矩阵
# ─────────────────────────────────────────────────────────────────────────────
def build_gene_graph(X, threshold=0.6):
    """
    基于皮尔逊相关系数构建基因共表达图，返回归一化邻接矩阵。

    参数
    ----
    X         : np.ndarray, shape (N_cells, N_genes)，基因表达矩阵
    threshold : float，相关系数绝对值阈值，默认 0.6

    返回
    ----
    adj_norm : np.ndarray, shape (N_genes, N_genes), float32
               经过 D^{-1/2} A D^{-1/2} 对称归一化
    """
    print(f"[GCN] 正在计算基因相关矩阵，X shape = {X.shape}，阈值 = {threshold}")

    # 转成 float64 避免精度问题，再算相关矩阵
    X64 = X.astype(np.float64)
    # 用 numpy corrcoef（比 scipy spearmanr 快得多，样本多时推荐）
    corr = np.corrcoef(X64.T)          # (n_genes, n_genes)
    corr = np.nan_to_num(corr)         # 防止常数基因导致 NaN

    # 阈值裁剪 → 0/1 邻接矩阵
    adj = (np.abs(corr) >= threshold).astype(np.float32)
    np.fill_diagonal(adj, 1.0)         # 自环

    # 对称归一化：D^{-1/2} A D^{-1/2}
    deg = adj.sum(axis=1)              # (n_genes,)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    # 保险：任何数值异常都清掉，避免后续 einsum 出现 NaN
    adj_norm = np.nan_to_num(adj_norm, nan=0.0, posinf=0.0, neginf=0.0)

    n_edges = int(adj.sum() - len(adj))  # 减去自环
    density = n_edges / (len(adj) * (len(adj) - 1))
    print(f"[GCN] 图构建完成：节点数 = {len(adj)}，边数 = {n_edges}，"
          f"图密度 = {density:.4f}")

    return adj_norm.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 单层图卷积
# H' = ReLU( A_hat @ H @ W )
# ─────────────────────────────────────────────────────────────────────────────
class GCNLayer(layers.Layer):
    """
    单层图卷积层。

    输入：(batch, n_genes, in_features)
    输出：(batch, n_genes, out_features)
    """

    def __init__(self, out_features, adj_matrix, use_relu=True, **kwargs):
        super().__init__(**kwargs)
        # 邻接矩阵固定不训练（生物先验知识），shape (n_genes, n_genes)
        self.A = tf.constant(adj_matrix, dtype=tf.float32)
        self.out_features = out_features
        self.use_relu = use_relu

    def build(self, input_shape):
        in_features = int(input_shape[-1])
        self.W = self.add_weight(
            name="W",
            shape=(in_features, self.out_features),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="bias",
            shape=(self.out_features,),
            initializer="zeros",
            trainable=True
        )

    def call(self, H):
        # H: (batch, n_genes, in_features)
        # Step1: 图平滑 A_hat @ H  →  (batch, n_genes, in_features)
        AH = tf.einsum("ij,bjf->bif", self.A, H)
        # Step2: 线性投影 AH @ W + b  →  (batch, n_genes, out_features)
        out = tf.matmul(AH, self.W) + self.b
        if self.use_relu:
            out = tf.nn.relu(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"out_features": self.out_features, "use_relu": self.use_relu})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 两层 GCN 编码器（带残差连接）
# 输入输出 shape 完全一致：(batch, n_genes, 1) → (batch, n_genes, 1)
# ─────────────────────────────────────────────────────────────────────────────
class GCNEncoder(layers.Layer):
    """
    两层图卷积编码器。

    结构：
        Layer1: (batch, n_genes, 1)  → (batch, n_genes, hidden_dim)  [ReLU]
        Layer2: (batch, n_genes, hidden_dim) → (batch, n_genes, 1)   [Linear]
        残差:   output = LayerNorm(GCN(x) + x)

    残差连接保证即使 GCN 什么都没学到，也不比 baseline 差。
    """

    def __init__(self, adj_matrix, hidden_dim=32, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gcn1     = GCNLayer(hidden_dim, adj_matrix, use_relu=True,  name="gcn_l1")
        self.gcn2     = GCNLayer(1,          adj_matrix, use_relu=False, name="gcn_l2")
        self.dropout  = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(axis=-1)

    
    def call(self, x, training=False):
        # x: (batch, n_genes, 1)

        # 1) 稳定输入尺度：对 count-like 数据非常关键
        x0 = tf.math.log1p(tf.nn.relu(x))  # 保证非负再 log1p，避免极端值

        # 2) 两层 GCN
        h = self.gcn1(x0)
        h = self.dropout(h, training=training)
        h = self.gcn2(h)

        # 3) clip + 小比例残差（让它是“微调”，不是“重写输入”）
        h = tf.clip_by_value(h, -1.0, 1.0)
        out = x0 + 0.1 * h

        # 4) 返回和原始输入同尺度的东西（可选：再 expm1 回去，但不建议）
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N_GENES = 200
    BATCH   = 8

    # 造一个假邻接矩阵
    fake_adj = np.random.rand(N_GENES, N_GENES).astype(np.float32)
    fake_adj = (fake_adj > 0.7).astype(np.float32)
    np.fill_diagonal(fake_adj, 1.0)

    # 造假输入
    x = tf.random.normal([BATCH, N_GENES, 1])

    gcn = GCNEncoder(adj_matrix=fake_adj, hidden_dim=32)
    out = gcn(x, training=False)
    print(f"GCNEncoder 输出 shape: {out.shape}")
    assert out.shape == (BATCH, N_GENES, 1)
    print("GCN 自测通过 ✓")

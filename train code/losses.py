"""
losses.py —— 监督对比损失（Supervised Contrastive Loss）

作用：
    在胶囊范数表示空间上，显式拉近同类样本、推远异类样本。
    对 COVID-19 vs fluA 这种相似类别区分效果显著。

总损失：
    L_total = L_CE + lambda_supcon * L_SupCon

参考：
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
"""

import tensorflow as tf


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """
    监督对比损失。

    参数
    ----
    features    : Tensor, shape (batch, n_capsule)
                  胶囊范数向量，即 tf.norm(capsule_out, axis=-1)
                  函数内部会自动做 L2 归一化
    labels      : Tensor, shape (batch,) int32/int64
    temperature : float，温度超参，越小对比越尖锐，默认 0.07

    返回
    ----
    loss : scalar Tensor
    """
    # L2 归一化：把每个样本的特征向量投到单位球面上
    features = tf.math.l2_normalize(features, axis=1)  # (batch, n_capsule)

    batch_size = tf.shape(features)[0]
    batch_f    = tf.cast(batch_size, tf.float32)

    # ── 计算相似度矩阵 ────────────────────────────────────────────────────
    # sim[i,j] = features[i] · features[j] / temperature
    sim_matrix = tf.matmul(features, features, transpose_b=True)  # (batch, batch)
    sim_matrix = sim_matrix / temperature

    # ── 构建正样本 mask ───────────────────────────────────────────────────
    # mask_pos[i,j] = 1 if labels[i] == labels[j] AND i != j
    labels_col = tf.reshape(labels, (-1, 1))                # (batch, 1)
    labels_row = tf.reshape(labels, (1, -1))                # (1, batch)
    mask_same  = tf.equal(labels_col, labels_row)           # (batch, batch) bool
    mask_same  = tf.cast(mask_same, tf.float32)

    mask_self  = 1.0 - tf.eye(batch_size)                   # 对角线置0（排除自身）
    mask_pos   = mask_same * mask_self                      # 正样本 mask

    # ── 数值稳定的 log-softmax ────────────────────────────────────────────
    # 减去最大值防止 exp 溢出
    logits_max = tf.stop_gradient(
        tf.reduce_max(sim_matrix, axis=1, keepdims=True)
    )
    logits = sim_matrix - logits_max                        # 数值稳定

    # 分母：所有非自身样本的 exp 之和
    exp_logits = tf.exp(logits) * mask_self                 # (batch, batch)
    log_prob   = logits - tf.math.log(
        tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8
    )                                                        # (batch, batch)

    # ── 只对正样本对计算损失 ──────────────────────────────────────────────
    # 每个样本的正样本数量
    n_pos = tf.reduce_sum(mask_pos, axis=1)                 # (batch,)

    # 避免某个样本在 batch 内没有同类样本（此时直接跳过该样本）
    has_pos = tf.cast(n_pos > 0, tf.float32)                # (batch,)

    mean_log_prob_pos = tf.reduce_sum(mask_pos * log_prob, axis=1) / (
        n_pos + 1e-8
    )                                                        # (batch,)

    # 只对有正样本的样本取平均
    loss = -tf.reduce_sum(has_pos * mean_log_prob_pos) / (
        tf.reduce_sum(has_pos) + 1e-8
    )

    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Prototype Diversity Loss
# 防止同一类的 K 个原型塌缩成同一个向量
# ─────────────────────────────────────────────────────────────────────────────
def prototype_diversity_loss(proto_caps):
    """
    鼓励同一类别内的 K 个原型胶囊尽量不相似（余弦相斥）。

    参数
    ----
    proto_caps : Tensor, shape (batch, n_class, K, dim_capsule)
                 多原型胶囊层的输出（reshape 之后）

    返回
    ----
    loss : scalar，值越小说明原型越相似（越塌缩），我们要最大化多样性
           即最小化 -diversity = 最小化原型间余弦相似度均值
    """
    # L2 归一化每个原型向量
    # (batch, n_class, K, dim)
    V = tf.math.l2_normalize(proto_caps, axis=-1)

    # 计算同类内原型两两余弦相似度
    # V: (batch, n_class, K, dim)
    # V^T: (batch, n_class, dim, K)
    # sim: (batch, n_class, K, K)
    sim = tf.matmul(V, V, transpose_b=True)

    # 只取上三角（避免重复计算 i==j 的自相似）
    K = tf.shape(proto_caps)[2]
    mask = tf.linalg.band_part(tf.ones([K, K]), 0, -1)  # 上三角为1
    mask = mask - tf.eye(K)                              # 去掉对角线

    # 平均非对角线上的余弦相似度
    sim_off_diag = sim * mask
    n_pairs      = tf.reduce_sum(mask)                   # K*(K-1)/2

    # loss = 平均相似度（越高越塌缩，我们要最小化它）
    loss = tf.reduce_mean(
        tf.reduce_sum(sim_off_diag, axis=[-2, -1]) / (n_pairs + 1e-8)
    )
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    BATCH, N_CAPS = 32, 3

    # 造一个有明显类内聚合的 feature（期望 loss 比随机 feature 更小）
    features_good = tf.constant(
        np.array([[1, 0, 0]] * 10 + [[0, 1, 0]] * 10 + [[0, 0, 1]] * 12,
                 dtype=np.float32)
    )
    features_rand = tf.random.normal([BATCH, N_CAPS])
    labels = tf.constant([0]*10 + [1]*10 + [2]*12, dtype=tf.int32)

    loss_good = supervised_contrastive_loss(features_good, labels)
    loss_rand = supervised_contrastive_loss(features_rand, labels)

    print(f"类内聚合 features 的 SupCon loss: {loss_good:.4f}")
    print(f"随机 features 的 SupCon loss:     {loss_rand:.4f}")
    assert loss_good < loss_rand, "聚合 features 的 loss 应该更小！"
    print("SupCon 自测通过 ✓")

    # Diversity loss 自测
    # 塌缩的原型（K个都一样）diversity loss 应该高
    collapsed = tf.tile(
        tf.random.normal([BATCH, 3, 1, 16]), [1, 1, 4, 1]
    )
    # 分散的原型（K个互相正交）diversity loss 应该低
    ortho_base = tf.eye(16)[:4]   # 4个正交向量
    diverse = tf.tile(
        tf.reshape(ortho_base, [1, 1, 4, 16]), [BATCH, 3, 1, 1]
    )

    loss_collapsed = prototype_diversity_loss(collapsed)
    loss_diverse   = prototype_diversity_loss(diverse)
    print(f"塌缩原型的 Diversity loss:  {loss_collapsed:.4f}  (应该接近 1.0)")
    print(f"分散原型的 Diversity loss:  {loss_diverse:.4f}   (应该接近 0.0)")
    assert loss_collapsed > loss_diverse, "Diversity loss 方向错误！"
    print("Diversity loss 自测通过 ✓")

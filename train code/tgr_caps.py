# tgr_caps.py —— 任务引导动态路由胶囊层（Task-Guided Routing Capsule Layer）
# 完整最终版：只对真实类别那一列加引导偏置（避免 M2 过偏）
# 兼容 tf.function / graph 模式（不使用 tensor 作为 Python if 条件）
# 推理时 alpha=0，退化为标准无监督路由（无标签泄漏）

import tensorflow as tf
from tensorflow.keras import layers


def squash(s, axis=-1):
    """胶囊激活函数，与 cap.py 口径一致。"""
    norm_sq = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
    scale = norm_sq / (1.0 + norm_sq) / (tf.sqrt(norm_sq + 1e-8))
    return scale * s


class TaskGuidedCapsuleLayer(layers.Layer):
    """
    任务引导动态路由胶囊层。

    相比原版 CapsuleLayer 的区别：
        - 增加可学习的类别原型向量 proto: (n_class, dim_capsule)
        - 训练时：只对真实类别 j=y 那一列加路由偏置 b（温和引导）
        - 推理时：alpha=0，纯无监督路由（无标签泄漏）

    参数
    ----
    num_capsule : 输出胶囊数量（=n_class）
    dim_capsule : 每个胶囊的维度（16）
    n_class     : 类别数（用于构建原型向量）
    routings    : 动态路由迭代次数（建议 2~3）
    """

    def __init__(self, num_capsule, dim_capsule, n_class, routings=3, **kwargs):
        super().__init__(**kwargs)
        self.num_capsule = int(num_capsule)
        self.dim_capsule = int(dim_capsule)
        self.n_class = int(n_class)
        self.routings = int(routings)

    def build(self, input_shape):
        """
        input_shape: (batch, num_input_caps, dim_input_caps)
        """
        self.num_input_caps = int(input_shape[1])
        self.dim_input_caps = int(input_shape[2])

        # 变换矩阵：每个输入 capsule i 到输出 capsule j 一个矩阵
        # shape: (num_input_caps, num_capsule, dim_input_caps, dim_capsule)
        self.W = self.add_weight(
            name="W",
            shape=(self.num_input_caps, self.num_capsule,
                   self.dim_input_caps, self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True
        )

        # 类别原型向量（可学习）
        # shape: (n_class, dim_capsule)
        self.proto = self.add_weight(
            name="class_prototype",
            shape=(self.n_class, self.dim_capsule),
            initializer="glorot_uniform",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs, labels=None, alpha=0.0, training=False):
        """
        inputs   : (batch, num_input_caps, dim_input_caps)
        labels   : (batch,) int32，训练时传入；推理时 None
        alpha    : float / scalar Tensor，引导强度；推理时应为 0
        training : bool
        return   : v (batch, num_capsule, dim_capsule)
        """
        batch_size = tf.shape(inputs)[0]

        # ── Step1: 计算预测向量 u_hat ────────────────────────────────────
        # inputs_tiled: (batch, i, j, 1, dim_in)
        inputs_tiled = tf.tile(
            tf.expand_dims(tf.expand_dims(inputs, 2), 3),
            [1, 1, self.num_capsule, 1, 1]
        )
        # W_tiled: (batch, i, j, dim_in, dim_capsule)
        W_tiled = tf.tile(tf.expand_dims(self.W, 0), [batch_size, 1, 1, 1, 1])
        # u_hat: (batch, i, j, dim_capsule)
        u_hat = tf.squeeze(tf.matmul(inputs_tiled, W_tiled), axis=3)

        # ── Step2: 初始化路由偏置 b ──────────────────────────────────────
        # b: (batch, i, j)
        b = tf.zeros([batch_size, self.num_input_caps, self.num_capsule], dtype=tf.float32)

        # ── Task-Guided 初始化：只对真实类别加偏置 ───────────────────────
        # 不在 Python if 中判断 alpha>0（graph 模式会出坑）
        # alpha=0 时乘上去就是 0，不影响结果
        if training and (labels is not None):
            # u_norm: (batch, i, j, dim)
            u_norm = tf.nn.l2_normalize(u_hat, axis=-1)

            # proto_y: (batch, dim) -> (batch, 1, 1, dim)
            proto_y = tf.gather(self.proto, labels)
            proto_y = tf.nn.l2_normalize(proto_y, axis=-1)
            proto_y = tf.reshape(proto_y, [batch_size, 1, 1, self.dim_capsule])

            # onehot: (batch, j) -> (batch, 1, j)
            onehot = tf.one_hot(labels, depth=self.num_capsule, dtype=tf.float32)
            onehot = tf.expand_dims(onehot, axis=1)  # (batch,1,j)

            # 只取真实类对应的 u_hat：u_y (batch, i, 1, dim)
            onehot_4d = tf.expand_dims(onehot, axis=3)  # (batch,1,j,1)
            u_y = tf.reduce_sum(u_norm * onehot_4d, axis=2, keepdims=True)  # (batch,i,1,dim)

            # cos_y: (batch, i, 1)
            cos_y = tf.reduce_sum(u_y * proto_y, axis=-1)  # (batch,i,1)

            # bias_add: (batch, i, j) = (batch,i,1) * (batch,1,j) -> broadcast
            b = b + alpha * cos_y * onehot

        # ── Step3: 动态路由迭代 ──────────────────────────────────────────
        for r in range(self.routings):
            # c: (batch, i, j)
            c = tf.nn.softmax(b, axis=2)

            # s: (batch, j, dim)
            s = tf.reduce_sum(tf.expand_dims(c, axis=3) * u_hat, axis=1)

            # v: (batch, j, dim)
            v = squash(s, axis=-1)

            # 更新 b（最后一轮不更新）
            if r < self.routings - 1:
                v_expand = tf.expand_dims(v, axis=1)  # (batch,1,j,dim)
                b = b + tf.reduce_sum(u_hat * v_expand, axis=-1)  # (batch,i,j)

        return v

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "n_class": self.n_class,
            "routings": self.routings,
        })
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 自测
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    BATCH, N_CAPS_IN, DIM_IN = 8, 100, 8
    N_CLASS, DIM_CAP = 3, 16

    layer = TaskGuidedCapsuleLayer(N_CLASS, DIM_CAP, N_CLASS, routings=3)
    x = tf.random.normal([BATCH, N_CAPS_IN, DIM_IN])
    labels = tf.constant(np.random.randint(0, N_CLASS, BATCH), dtype=tf.int32)

    # 训练模式（alpha>0）
    v_train = layer(x, labels=labels, alpha=tf.constant(0.1), training=True)
    print("train out:", v_train.shape)
    assert v_train.shape == (BATCH, N_CLASS, DIM_CAP)

    # 推理模式（alpha=0）
    v_inf = layer(x, labels=None, alpha=tf.constant(0.0), training=False)
    print("infer out:", v_inf.shape)
    assert v_inf.shape == (BATCH, N_CLASS, DIM_CAP)

    # 无泄漏：alpha=0 时 labels 不影响输出
    fake_labels = tf.zeros([BATCH], dtype=tf.int32)
    v_inf2 = layer(x, labels=fake_labels, alpha=tf.constant(0.0), training=False)
    diff = tf.reduce_max(tf.abs(v_inf - v_inf2)).numpy()
    print("diff:", diff)
    assert diff < 1e-6
    print("TGR self-test passed ✓")

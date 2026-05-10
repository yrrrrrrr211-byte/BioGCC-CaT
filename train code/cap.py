import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D, BatchNormalization
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import *
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from PIL import Image
from tensorflow.keras import initializers, layers
from datetime import date
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.python.client import device_lib


class Length(layers.Layer):
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=tf.shape(x)[1])
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


@tf.keras.utils.register_keras_serializable(package="scCaT")
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

        # 关键：保存“可序列化”的 initializer 名字（不要直接存对象）
        self.kernel_initializer = kernel_initializer
        self._kernel_initializer_obj = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = int(input_shape[1])
        self.input_dim_capsule = int(input_shape[2])

        self.W = self.add_weight(
            shape=(self.num_capsule, self.input_num_capsule,
                   self.dim_capsule, self.input_dim_capsule),
            initializer=self._kernel_initializer_obj,
            name='W'
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 1)
        inputs_tiled  = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled  = tf.expand_dims(inputs_tiled, 4)

        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled)

        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule,
                            self.input_num_capsule, 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            c = layers.Softmax(axis=1)(b)
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)  # [None, num_capsule, 1, dim_capsule, 1]

            if i < self.routings - 1:
                outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)

        outputs = tf.squeeze(outputs, [2, 4])
        return outputs, c

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

    def get_config(self):
        # 关键：让 Keras 能保存/加载这个层
        config = super().get_config()
        config.update({
            "num_capsule": self.num_capsule,
            "dim_capsule": self.dim_capsule,
            "routings": self.routings,
            "kernel_initializer": self.kernel_initializer,  # 存字符串
        })
        return config
    


@tf.keras.utils.register_keras_serializable(package="scCaT")  
class Weightlayer(layers.Layer):
    def __init__(self, dim_capsule, n_channels, **kwargs):
        super(Weightlayer, self).__init__(**kwargs)

        # 关键：把 init 参数存起来（用于保存/加载）
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels

        self.filters = dim_capsule * n_channels

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(1, input_shape[1], self.filters),
            initializer='glorot_uniform',
            name='weight'
        )
        super().build(input_shape)  

    def call(self, inputs):
        return tf.multiply(inputs, self.weight), self.weight

    def get_config(self):
        # 关键：告诉 Keras 如何序列化这个层
        config = super().get_config()
        config.update({
            "dim_capsule": self.dim_capsule,
            "n_channels": self.n_channels,
        })
        return config  
    
def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    inputs, layerweight = Weightlayer(dim_capsule, n_channels)(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(inputs)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 创建输入数据
x = np.linspace(-7, 7, 666)  # (-7,7)之间的等间隔的180个点


# 激活函数的实现
def sigmoid(inputs):
    y = [1 / float(1 + np.exp(-x)) for x in inputs]
    return y


def relu(inputs):
    y = [x * (x > 0) for x in inputs]  # (x > 0)返回0或1
    return y


def tanh(inputs):
    y = [(np.exp(x) - np.exp(-x)) / float(np.exp(x) + np.exp(-x)) for x in inputs]
    return y


def softplus(inputs):
    y = [np.log(1 + np.exp(x)) for x in inputs]
    return y


# TensorFlow激活函数处理后的Y的值
y_sigmoid = tf.nn.sigmoid(x)
y_relu = tf.nn.relu(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)

# 创建会话
with tf.Session() as sess:
    y_sigmoid, y_relu, y_tanh, y_softplus = sess.run([y_sigmoid, y_relu, y_tanh, y_softplus])
    # 创建各个激活函数的图像
    # Sigmoid
    plt.subplot(221)
    plt.plot(x, y_sigmoid, 'r', label="Sigmoid")
    plt.ylim(-0.2, 1.2)
    plt.legend(loc="best")

    # Relu
    plt.subplot(222)
    plt.plot(x, y_relu, 'r', label="Relu")
    plt.ylim(-2, 6)
    plt.legend(loc="best")

    # Tanh
    plt.subplot(223)
    plt.plot(x, y_tanh, 'r', label="Tanh")
    plt.ylim(-1.3, 1.3)
    plt.legend(loc="best")

    # Softplus
    plt.subplot(224)
    plt.plot(x, y_softplus, 'r', label="Softplus")
    plt.ylim(-1, 6)
    plt.legend(loc="best")

    # 显示图像
    plt.show()
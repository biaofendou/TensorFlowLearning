import tensorflow as tf
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np

batch_size = 8

# 两个输入节点，2是指的数据的尺寸，None指的batch size的大小，所以可以是任何数
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义一个单层的神经网络前向传播的过程，就是简单加权和
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y - y_) * loss_more, (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
# 加入噪音
Y = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in X]
print(X)
print(Y)
plt.scatter([x[0] for x in X], Y, color='red')

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    step = 5000
    for i in range(step):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        print(sess.run(w1))
    x = np.linspace(0, 1, 10)
    plt.plot(x, x * sess.run(w1[0]) + sess.run(w1[1]), color='blue')
    plt.show()
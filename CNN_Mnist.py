# -*_ coding: UTF-8 -*-
import numpy as np
import tensorflow as tf

# 下载并载入MNIST手写数字库，共55000张28*28图片
from tensorflow.examples.tutorials.mnist import input_data

'''
one_hot是独热码的编码格式
0123456789
0:100000000
1:010000000
2:001000000
3:000100000
……
'''
# 将数据集放入指定目录
mnist = input_data.read_data_sets('data/mnist_data', one_hot=True)

input_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255  # 图片大小为28*28,None表示Tensor张量的第一个维度可以是任何长度
output_y = tf.placeholder(tf.int32, [None, 1 * 1 * 10])  # 输出10个标签
# tf.nn.con2d和tf.layers.con2d有所区别
input_image = tf.reshape(input_x, [-1, 28, 28, 1])  # 输入图像的集合，重构矩阵,-1表示根据其他位推断这位

# 从Test数据集中选取3000图片和对应标签
test_x = mnist.test.images[:3000]  # 图片
test_y = mnist.test.labels[:3000]  # 标签

# 构建卷积网络
# 第一层卷积
conv1 = tf.layers.conv2d(
    inputs=input_image,  # 形状[28,28,1]
    filters=32,  # 过滤器数量
    kernel_size=[5, 5],  # 过滤器大小
    strides=1,  # 步长为1
    padding='same',  # same表示输出大小不变，会自动在图片外围补零
    activation=tf.nn.relu  # 激活函数为Relu
)  # 形状[28,28,32]
# 第一层池化（亚采样）
pool1 = tf.layers.max_pooling2d(
    inputs=conv1,  # 将上面的结果作为输入，形状为[28,28,32]
    pool_size=[2, 2],  # 过滤器大小
    strides=2,  # 步长为2
)  # 形状[14,14,32],过滤器数量为1，不会改变深度
# 第二层卷积
conv2 = tf.layers.conv2d(
    inputs=pool1,  # 形状[14,14,32]
    filters=64,  # 过滤器数量
    kernel_size=[5, 5],  # 过滤器大小
    strides=1,  # 步长为1
    padding='same',  # same表示输出大小不变，会自动在图片外围补零
    activation=tf.nn.relu  # 激活函数为Relu
)  # 形状[14,14,64]

# 第二层池化（亚采样）
pool2 = tf.layers.max_pooling2d(
    inputs=conv2,  # 将上面的结果作为输入，形状为[14,14,64]
    pool_size=[2, 2],  # 过滤器大小
    strides=2,  # 步长为2
)  # 形状[7,7,64],过滤器数量为1，不会改变深度

# 平坦化（flat）
flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 形状[7 * 7 * 64]

# 1024个神经元的全连接层
dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

# Dropout:丢弃50%，rate=0.5
dropout = tf.layers.dropout(inputs=dense, rate=0.5)

# 构建10个神经元的全连接层，这里不用激活函数来做非线性化
logits = tf.layers.dense(inputs=dropout, units=10)  # 输出，形状[1,1,10]

# 计算误差（计算Cross entropy（交叉熵）），再用Softmax计算百分比概率
loss = tf.losses.softmax_cross_entropy(onehot_labels=output_y, logits=logits)

# 用Adam优化器最小化误差，学习率为0.001
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# 计算预测值和实际标签的匹配程度
# 返回[accuracy,update_op],会创建两个局部变量，输出两个值,第一个值为上几步的平均精度,第二值是上几步与该步的精度的平均值.
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(output_y, axis=1),  # tf.argmax就是返回最大的那个数值所在的下标，axis 0：按列计算，1：行计算
    predictions=tf.argmax(logits, axis=1))[1]

# 创建会话
with tf.Session() as sess:
    # 初始化全局变量和局部变量
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(50)  # 从train数据集中取下一个50个样本
        train_loss, train_op_ = sess.run([loss, train_op], {input_x: batch[0], output_y: batch[1]})
        if i % 100 == 0:
            test_accuracy = sess.run(accuracy, {input_x: test_x, output_y: test_y})
            print("Step=%d,Train loss = %f,[Test accuracy=%f]" % (i, train_loss, test_accuracy))

    # 打印20个预测值和真实值对
    test_output = sess.run(logits, {input_x: test_x[:20]})
    inferenced_y = np.argmax(test_output, 1)
    print(inferenced_y, 'Inferenced numbers')  # 推测的数字
    print(np.argmax(test_y[:20], 1), 'Real numbers')  # 真实的数字

# -*- coding:UTF-8 -*-

'''
用梯度下降方法解决线性回归问题
'''

import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

# 构建数据
point_num = 100
vectors = []
# 用numpy的正态随机分布函数生成100个点
# 点的x,y坐标值对应y = 0.1 * x +0.2,权重0.1，偏差0.2
for i in range(point_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]  # 将x坐标赋值
y_data = [v[1] for v in vectors]  # 将y坐标赋值

# 图像1展示所有的随机点
plt.plot(x_data, y_data, 'r*', label="Original Data")
plt.title("Linear Regression")
plt.legend()  # 显示数据标题label
plt.show()

# 构建线性回归模型
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 初始化权重
b = tf.Variable(tf.zeros([1]))  # 初始化偏差
y = W * x_data + b  # 模型计算出来的y

# 定义损失函数（loss function/cost function）
# 对TensorFlow的所有维度计算((y-y_data)^2)之和/N
loss = tf.reduce_mean(tf.square(y - y_data))

# 用梯度下降的优化器来优化loss function
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 设置学习率0.5
train = optimizer.minimize(loss)  # 让损失结果最小

# 创建会话
with tf.Session() as sess:
    # 初始化变量
    init = tf.global_variables_initializer()
    sess.run(init)
    # 训练20步
    for step in range(20):
        # 优化每一步
        sess.run(train)
        # 打印每一步的损失，权重和偏差
        print("Step=%d,Loss=%f,[Weight=%f Bias=%f]" % (step, sess.run(loss), sess.run(W), sess.run(b)))

    # 图像2：绘制出所有的点并且绘制出最佳的拟合直线
    plt.plot(x_data, y_data, 'b*', label="Original Data")
    plt.title("Linear Regression")
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="Fitted Line")  # 拟合线
    plt.legend()  # 显示数据标题label
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

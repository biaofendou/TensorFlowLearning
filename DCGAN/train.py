# -*- coding:UTF-8 -*-
'''训练DCGAN'''

import glob
import numpy as np
from scipy import misc
import tensorflow as tf

from DCGAN.network import *


def train():
    # 获取训练数据
    data = []
    for image in glob.glob("../data/images/*"):
        image_data = misc.imread(image)  # imread利用PIL来读取图片数据
        data.append(image_data)

    input_data = np.array(data)
    # 将数据标准化成[-1,1]的取值，这也是Tanh激活函数的输出范围
    input_data = (input_data.astype(np.float32) - 127.5) / 127.5

    # 构造生成器和判别器
    g = generator_model()
    d = discriminator_model()

    # 构造生成器和判别器组成的网络模型
    d_on_g = generator_containing_discriminator(g, d)
    # 优化器用Adam Optimizer
    g_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)
    d_optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1)
    # 配置生成器和判别器，在优化生成器的时候判别器不动，反之同理
    g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d_on_g.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    d.trainable = True
    d.compile(loss="binary_crossentropy", optimizer=g_optimizer)
    # 开始训练
    for epoch in range(epochs):
        # input_data.shape[0] 是数据的个数
        for index in range(int(input_data.shape[0] / batch_size)):
            input_batch = input_data[index * batch_size:(index + 1) * batch_size]
            # 生成连续型均匀分布的随机数据（噪声）,输出样本数目，为int或元组(tuple)类型，例如，size=(m,n,k), 则输出m*n*k个样本，缺省时输出1个值。
            random_data = np.random.uniform(-1, 1, size=(batch_size, 100))
            # 生成器生成的图片数据
            generated_images = g.predict(random_data, verbose=0)
            input_batch = np.concatenate((input_batch, generated_images))
            output_batch = [1] * batch_size + [0] * batch_size
            # 训练判别器，让其具备识别不合格生成图片的能力
            d_loss = d.train_on_batch(input_batch, output_batch)
            # 当训练生成器时让判别器不可被训练
            d.trainable = False

            random_data = np.random.uniform(-1, 1, size=(batch_size, 100))
            # 训练生成器，让其能通过判别器的判别
            g_loss = d_on_g.train_on_batch(random_data, [1] * batch_size)

            # 恢复判别器可被训练
            d.trainable = True

            # 打印损失
            print("Step %d Generator Loss: %f Discriminator Loss: %f" % (index, g_loss, d_loss))
        # 保存生成器和判别器的参数
        if epoch % 10 == 0:
            g.save_weights("../DCGAN/checkpoints/generator_weight", True)
            d.save_weights("../DCGAN/checkpoints/discriminator_weight", True)


if __name__ == "__main__":
    train()

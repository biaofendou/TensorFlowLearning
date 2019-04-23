# -*- coding:UTF-8 -*-
''' 用DCGAN的生成器模型和生成得到的生成器参数文件来生成图片'''

import numpy as np
from PIL import Image
import tensorflow as tf

from DCGAN.network import *


def generator():
    # 构造生成器
    g = generator_model()
    # 配置生成器
    g.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=learning_rate, beta_1=beta_1))
    # 加载训练好的生成器参数
    g.load_weights("../DCGAN/checkpoints/generator_weight")
    # 连续型均匀分布的随记数据（噪声）
    random_data = np.random.uniform(-1, 1, size=(batch_size, 100))
    # 用随机数据作为输入，生成器 生成图片数据
    images = g.predict(random_data, verbose=1)
    # 用生成的图片数据生成PNG图片
    for i in range(batch_size):
        image = images[i] * 127.5 + 127.5
        Image.fromarray(image.astype(np.uint8)).save("../DCGAN/generate/image-%s.png" % i)


if __name__ == '__main__':
    generator()

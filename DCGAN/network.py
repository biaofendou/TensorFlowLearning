# -*- coding:UTF-8 -*-
'''DCGAN 深层卷积对抗网络'''
import tensorflow as tf

# Hyperparameters 超参数
epochs = 100
batch_size = 128
learning_rate = 0.0002
beta_1 = 0.5


# 定义判别器模型（卷积神经网络）
def discriminator_model():
    model = tf.keras.Sequential()
    # 添加第一个卷积层
    model.add(tf.keras.layers.Conv2D(
        64,  # 64个过滤器
        (5, 5),  # 过滤器大小
        padding='same',  # 图片大小不变
        input_shape=(64, 64, 3)  # 输入形状，3表示rgb
    ))
    # 激活函数层
    model.add(tf.keras.layers.Activation('tanh'))
    # 添加pooling层
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 添加第二个卷积层，不需要指定input_shape
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 添加第三个卷积层，不需要指定input_shape
    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    # 扁平化
    model.add(tf.keras.layers.Flatten())
    # 1024个神经元的全连接层
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation('tanh'))
    # 1个神经元的全连接层
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))

    return model


# 定义生成器模型,从随机数生成一张图片
def generator_model():
    model = tf.keras.models.Sequential()
    # units是输出维度，是神经元个数为1024的全连接层，input_dim指输入的维度是100
    model.add(tf.keras.layers.Dense(input_dim=100, units=1024))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Dense(128 * 8 * 8))
    model.add(tf.keras.layers.BatchNormalization())  # 批标准化，对激活函数进行转换，让其标准差和平均数接近1和0
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.Reshape((8, 8, 128), input_shape=(128 * 8 * 8,)))  # 变成8*8的图片
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 是池化的反操作，上采样,变成16*16的图片
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 是池化的反操作，上采样,变成32*32的图片
    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('tanh'))
    model.add(tf.keras.layers.UpSampling2D(size=(2, 2)))  # 是池化的反操作，上采样,变成64*64的图片
    model.add(tf.keras.layers.Conv2D(3, (5, 5), padding='same'))  # 让卷积层变成彩色图片的深度3
    model.add(tf.keras.layers.Activation('tanh'))

    return model


# 构造一个Sequential对象，包含一个生成器和一个判别器
# 输入->生成器->判别器->输出
def generator_containing_discriminator(generator, discriminator):
    model = tf.keras.models.Sequential()
    model.add(generator)
    discriminator.trainable = False  # 初始时判别器是不可被训练的
    model.add(discriminator)
    return model


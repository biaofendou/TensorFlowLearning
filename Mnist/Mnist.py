import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 数据集相关的常数
input_node = 784  # 输入层的节点数，等于图片的像素
output_node = 10  # 输出层的节点数，等于类别数，因为数字识别，所以为10

# 配置神经网络的参 数
layer1_node = 500  # 隐藏层节点数

batch_size = 100  # 数字越小训练越接近随机梯度下降，数字越大训练越接近梯度下降
learning_rate_base = 0.8  # 学习率
learning_rate_decay = 0.99  # 学习率的衰减率
regularization_rate = 0.0001  # 正则化系数
training_steps = 30000  # 训练轮数
moving_average_decay = 0.99  # 滑动平均衰减率


# 定义激活函数ReLU
def inference(input_tensor, avg_class, weightsl, biases1, weights2, biases2):
    # 没有提供滑动平均类时直接使用参数当前取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weightsl) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 先使用avg_class.average函数来计算出变量的滑动平均值，再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weightsl)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练过程
def train(mnist):
    x = tf.placeholder(tf.float32, [None, input_node], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, output_node], name='y-output')
    # 生成隐藏层的参数
    weights1 = tf.Variable(tf.truncated_normal([input_node, layer1_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    # 生成输出层的参数
    weights2 = tf.Variable(tf.truncated_normal([layer1_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))
    # 函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)
    # 定义存储训练轮数的变量，此为不可训练变量所以（trainable=False）
    global_step = tf.Variable(0, trainable=False)
    # 给定训练轮数的变量可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算使用了滑动平均之后的前向传播结果
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    # 计算正则化的损失
    regularization = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        learning_rate_base,  # 基础学习率
        global_step,  # 当前迭代轮数
        mnist.train.num_examples / batch_size,  # 过完所有的训练数据需要迭代的次数
        learning_rate_decay  # 学习率衰减速度
    )
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 同时更新参数和每个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 检验是用来滑动平均模型的神经网络前向传播结果是否正确
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 这个计算就是将一个布尔型的数值转换为实数型，然后计算平均值，这个平均值就是明星在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                         y_: mnist.validation.labels}
        # 准备测试数据
        test_feed = {x: mnist.test.images,
                     y_: mnist.test.labels}
        # 迭代训练神经网络
        for i in range(training_steps):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("第%d次训练后，验证准确率为：%g" % (i, validate_acc))
            # 产生这一轮使用的一个batch的训练数据，并且运行训练过程
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        # 训练后，模型最终准确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("%d次训练后，测试准确率为：%g" % (training_steps, test_acc))


# 程序主入口
def main(argv=None):
    mnist = input_data.read_data_sets('../data', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

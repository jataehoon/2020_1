# 얼마전에 나온 Least Squares Generative Adversarial Networks를 구현해봤습니다.
# 복습할겸 한줄한줄 설명을 달아보겠습니다.
# Tensorflow implementation of the paper "Least Squares Generative Adversarial Networks"
# https://arxiv.org/abs/1611.04076
# Coded by GunhoChoi 170305
# https://github.com/GunhoChoi/LSGAN-TF/blob/master/LSGAN/LSGAN_TF.ipynb



# 우선 필요한 라이브러리와 데이터를 불러옵니다.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)



# 학습때 필요한 Hyperparameter를 설정해놓습니다.
batch_size = 512
learning_rate = 1e-3
epoch = 10000



# 이미지 데이터와 제너레이터의 z를 받아올 placeholder를 생성해 놓습니다.
# 그냥 불러오면 784개의 한줄짜리 데이터이기 때문에 28x28x1로 모양을 바꿔줍니다.
# 또한 variable initializer도 설정해줍니다.
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
z_in = tf.placeholder(tf.float32, shape=[batch_size, 100])
initializer = tf.truncated_normal_initializer(stddev=0.02)



# discriminator가 더 잘 학습되도록 그냥 relu 대신 leaky relu를 구현했습니다.
# tf.maximum(x, a*x)로 하는 방법도 있지만 메모리를 두배로 쓰기 때문에 아래와 같이 구현했습니다.
# 해당 issue가 텐서플로우 깃헙에 올라와 있는데 링크 첨부합니다.
# https://github.com/tensorflow/tensorflow/issues/4079
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)



# Generator를 만들어줍니다. 
# tf.variable_scope를 사용하면 나중에 제너레이터 부분만 업데이트하는데 편리합니다.
# conv2d_transpose 대신 아래와 같이 conv2d -> reshape 를 사용한 이유는 아래 링크에 있습니다.
# http://distill.pub/2016/deconv-checkerboard/ 
# 근데 텐서플로우 1.0에서 이 부분을 감안해서 짰는지는 확인 못했습니다.
def generator(z):
    with tf.variable_scope("generator"):
        fc1 = tf.contrib.layers.fully_connected(inputs=z, num_outputs=7*7*128, 
                                                activation_fn=tf.nn.relu, 
                                                normalizer_fn=tf.contrib.layers.batch_norm,
                                                weights_initializer=initializer,
                                                scope="g_fc1")
        fc1 = tf.reshape(fc1, shape=[batch_size, 7, 7, 128])
        conv1 = tf.contrib.layers.conv2d(fc1, num_outputs=4*64, kernel_size=5, 
                                         stride=1, padding="SAME",activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm, 
                                         weights_initializer=initializer,scope="g_conv1")
        conv1 = tf.reshape(conv1, shape=[batch_size,14,14,64])
        conv2 = tf.contrib.layers.conv2d(conv1, num_outputs=4*32, kernel_size=5,
                                         stride=1, padding="SAME", activation_fn=tf.nn.relu,
                                         normalizer_fn=tf.contrib.layers.batch_norm, 
                                         weights_initializer=initializer,
                                         scope="g_conv2")
        conv2 = tf.reshape(conv2, shape=[batch_size,28,28,32])
        conv3 = tf.contrib.layers.conv2d(conv2, num_outputs=1, kernel_size=5, 
                                         stride=1, padding="SAME", 
                                         activation_fn=tf.nn.tanh,
                                         scope="g_conv3")
        return conv3



# Discriminator도 만들어줍니다.
# Variable reuse에 관한 설명은 아래 링크로 대신합니다.
# https://www.tensorflow.org/programmers_guide/variable_scope
def discriminator(tensor,reuse=False):
    with tf.variable_scope("discriminator"):
        conv1 = tf.contrib.layers.conv2d(inputs=tensor, num_outputs=32,
                                         kernel_size=5, stride=2, padding="SAME", 
                                        reuse=reuse, activation_fn=lrelu,
                                         weights_initializer=initializer,
                                         scope="d_conv1")
        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, 
                                         kernel_size=5, stride=2, padding="SAME", 
                                         reuse=reuse, activation_fn=lrelu,
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         weights_initializer=initializer,
                                         scope="d_conv2")
        fc1 = tf.reshape(conv2, shape=[batch_size, 7*7*64])
        fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512,reuse=reuse,
                                                activation_fn=lrelu, 
                                                normalizer_fn=tf.contrib.layers.batch_norm, 
                                                weights_initializer=initializer,
                                                scope="d_fc1")
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse,
                                                activation_fn=tf.nn.sigmoid,
                                                weights_initializer=initializer,
                                                scope="d_fc2")
        return fc2



# 학습을 시키기 위해서는 D(G(z))와 D(x)가 필요하기 때문에 아래처럼 그래프를 만들어줍니다.
g_out = generator(z_in)
d_out_fake = discriminator(g_out)
d_out_real = discriminator(x_image,reuse=True)



# loss는 논문에 나온대로 구현합니다.
disc_loss = tf.reduce_sum(tf.square(d_out_real-1) + tf.square(d_out_fake))/2
gen_loss = tf.reduce_sum(tf.square(d_out_fake-1))/2



# 여기부터가 제가 좀 헷갈렸던 부분인데 gen_loss는 generator만 업데이트하고
# disc_loss는 discriminator만 업데이트하도록 하기 위해서 
# 각 name_scope에서 variable을 불러옵니다.
# https://www.tensorflow.org/api_docs/python/tf/GraphKeys
# https://www.tensorflow.org/api_docs/python/tf/get_collection
gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator") 
dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")



# 그 다음엔 loss에 대한 해당 variable의 gradient를 구해 이를 업데이트 합니다.
d_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
g_optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
d_grads = d_optimizer.compute_gradients(disc_loss,dis_variables) 
g_grads = g_optimizer.compute_gradients(gen_loss,gen_variables) 
update_D = d_optimizer.apply_gradients(d_grads)
update_G = g_optimizer.apply_gradients(g_grads)



# 위에 설정한 epoch만큼 반복하면서 다음 batch와 z를 생성해 feed_dict로 전달합니다.
# 이때 discriminator 한번에 generator 한번으로 학습하면 generator가 학습을 포기하기 때문에
# 적당한 비율로 돌려야하는데 보통은 연산 횟수에 맞춰 d:g=1:2로 한다고 합니다.
# 저는 generator에게 더 많은 기회를 주기로 결정하여 이를 늘려서 학습하였습니다.
# 그리고 변화를 시각적으로 확인하기 위해 10번마다 generator의 output을 저장했습니다. 
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        batch = mnist.train.next_batch(batch_size)
        z_input = np.random.uniform(0,1.0,size=[batch_size,100]).astype(np.float32)
        _, d_loss = sess.run([update_D,disc_loss],feed_dict={x: batch[0], z_in: z_input})
        for j in range(4):
            _, g_loss = sess.run([update_G,gen_loss],feed_dict={z_in: z_input})
        print("i: {} / d_loss: {} / g_loss: {}".format(i,np.sum(d_loss)/batch_size, np.sum(g_loss)/batch_size))
        if i % 10 == 0:
            gen_o = sess.run(g_out,feed_dict={z_in: z_input})
            #result = plt.imshow(gen_o[0][:, :, 0], cmap="gray")
            plt.imsave("{}.png".format(i),gen_o[0][:, :, 0], cmap="gray")



''' RESULT
i: 0 / d_loss: 0.2535995543003082 / g_loss: 0.130204439163208
i: 1 / d_loss: 0.25267529487609863 / g_loss: 0.10431566834449768
i: 2 / d_loss: 0.2529243230819702 / g_loss: 0.1455080509185791
i: 3 / d_loss: 0.25341713428497314 / g_loss: 0.10555155575275421
i: 4 / d_loss: 0.25059691071510315 / g_loss: 0.13938385248184204
i: 5 / d_loss: 0.24953433871269226 / g_loss: 0.10983732342720032
'''
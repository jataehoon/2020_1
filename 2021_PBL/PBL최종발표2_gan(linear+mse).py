import time
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
import numpy as np
import matplotlib.pyplot as plt

(train_x, _), (_, _) = mnist.load_data()

train_x = train_x / 127.5 - 1

train_x = train_x.reshape(-1, 784)
train_x.shape

# gan에 입력되는 noise에 대한 dimension
NDIM = 10

gen = tf.keras.Sequential()
gen.add(layers.Dense(256, input_dim=NDIM))
gen.add(layers.LeakyReLU(0.2)) 
gen.add(layers.Dense(512))
gen.add(layers.LeakyReLU(0.2))
gen.add(layers.Dense(1024)) 
gen.add(layers.LeakyReLU(0.2))
gen.add(layers.Dense(28*28, activation='tanh'))

gen.summary()

dis = tf.keras.Sequential()
dis.add(layers.Dense(1024, input_shape=(784,), kernel_initializer=RandomNormal(stddev=0.02)))
dis.add(layers.LeakyReLU(0.2))
dis.add(layers.Dropout(0.3))
dis.add(layers.Dense(512))
dis.add(layers.LeakyReLU(0.2))
dis.add(layers.Dropout(0.3))
dis.add(layers.Dense(256))
dis.add(layers.LeakyReLU(0.2))
dis.add(layers.Dropout(0.3)) 
dis.add(layers.Dense(1, activation='sigmoid'))

dis.summary()

dis.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# discriminator 학습하지 않게, generative 모델에서 generator만 학습
dis.trainable = False
g_input = Input(shape=(NDIM,))
x = gen(inputs=g_input)
output = dis(x)

gan = Model(g_input, output)

gan.summary()

gan.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

def def_batches(train_data, batch_size):
    list_of_batches = []
    for i in range(int(train_data.shape[0] // batch_size)):
        single_batch = train_data[i * batch_size: (i + 1) * batch_size]
        list_of_batches.append(single_batch)
    return np.asarray(list_of_batches)

def train_visualize(epoch, d_losses, g_losses):
  
    #샘플 데이터 생성 후 시각화
    N = np.random.normal(0, 1, size=(24, NDIM))
    g_img = gen.predict(N)
    g_img = g_img.reshape(-1, 28, 28)
    
    plt.figure(figsize=(8, 4))
    for i in range(g_img.shape[0]):
        plt.subplot(4, 6, i+1)
        plt.imshow(g_img[i], interpolation='nearest', cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

BATCH_SIZE = 256
EPOCHS = 20

# discriminator와 gan 모델의 loss 측정 위한 list 생성
list_d_loss = []
list_g_loss = []

for epoch in range(1, EPOCHS + 1):
    start = time.time()

    # 각 배치별 학습
    for real_img in def_batches(train_x, BATCH_SIZE):
        # 랜덤 노이즈 생성
        input_noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NDIM])
        
        # 가짜 이미지 데이터 생성
        g_img = gen.predict(input_noise)
        
        # Gan에 학습할 X 데이터 정의
        dis_x = np.concatenate([real_img, g_img])
        
        # Gan에 학습할 Y 데이터 정의
        dis_y = np.zeros(2 * BATCH_SIZE)
        dis_y[:BATCH_SIZE] = 0.9
        
        # Discriminator 훈련
        dis.trainable = True
        d_loss = dis.train_on_batch(dis_x, dis_y)
        
        # Gan 훈련
        noise = np.random.uniform(-1, 1, size=[BATCH_SIZE, NDIM])
        gan_y = np.ones(BATCH_SIZE)
        
        # Discriminator의 판별 학습을 방지
        dis.trainable = False
        g_loss = gan.train_on_batch(noise, gan_y)
        
    list_d_loss.append(d_loss)
    list_g_loss.append(g_loss)
    
    if epoch == 1:
        train_visualize(epoch, list_d_loss, list_g_loss)

    print('에포크 {} 에서 소요된 시간은 {} 초'.format(epoch, time.time()-start))

    if epoch % 20 == 0:
        train_visualize(epoch, list_d_loss, list_g_loss)


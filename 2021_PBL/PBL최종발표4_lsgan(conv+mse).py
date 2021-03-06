import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers
import time

(train_imgs, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_imgs = train_imgs.reshape(train_imgs.shape[0], 28, 28, 1).astype('float32')
train_imgs = (train_imgs - 127.5) / 127.5 # 이미지 [-1, 1]로 정규화
BUFFER_SIZE = 60000
BATCH_SIZE = 256
# 데이터 배치를 만들고 섞음
train_data = tf.data.Dataset.from_tensor_slices(train_imgs).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def create_generator_model():
    gen = tf.keras.Sequential()
    gen.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU())

    gen.add(layers.Reshape((7, 7, 256)))
    assert gen.output_shape == (None, 7, 7, 256) # Batch Size가 None이 주어짐

    gen.add(layers.Conv2DTranspose(128, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    assert gen.output_shape == (None, 7, 7, 128)
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU(0.2))

    gen.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert gen.output_shape == (None, 14, 14, 64)
    gen.add(layers.BatchNormalization())
    gen.add(layers.LeakyReLU(0.2))

    gen.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert gen.output_shape == (None, 28, 28, 1)

    return gen

generator = create_generator_model()

noise = tf.random.normal([1, 100])
generated_img = generator(noise, training=False)

def create_discriminator_model():
    dis = tf.keras.Sequential()
    dis.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    dis.add(layers.ReLU())
    dis.add(layers.Dropout(0.3))

    dis.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    dis.add(layers.ReLU())
    dis.add(layers.Dropout(0.3))
    dis.add(layers.Conv2D(256, (4, 4), strides=(1, 1), padding='valid'))
    dis.add(layers.ReLU())
    dis.add(layers.Dropout(0.3))

    dis.add(layers.Flatten())
    dis.add(layers.Dense(1))

    return dis

discriminator = create_discriminator_model()

generator.summary()

discriminator.summary()

# 크로스 엔트로피 손실함수 계산 위해 헬퍼 함수를 반환합니다.
cross_entropy_loss = tf.keras.losses.MeanSquaredError()

def discriminator_loss(output_real, output_fake):
    loss_real = cross_entropy_loss(tf.ones_like(output_real), output_real)
    loss_fake = cross_entropy_loss(tf.zeros_like(output_fake), output_fake)
    loss_total = loss_real + loss_fake
    return loss_total

def generator_loss(output_fake):
    return cross_entropy_loss(tf.ones_like(output_fake), output_fake)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 20
NDIM = 100
NUMBER_OF_EXAMPLES = 16
seed = tf.random.normal([NUMBER_OF_EXAMPLES, NDIM])

# 함수를 컴파일 합니다
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NDIM])

    with tf.GradientTape() as gt, tf.GradientTape() as dt:
      generated_imgs = generator(noise, training=True)

      output_real = discriminator(images, training=True)
      output_fake = discriminator(generated_imgs, training=True)

      loss_gen = generator_loss(output_fake)
      loss_disc = discriminator_loss(output_real, output_fake)

    gradients_generator = gt.gradient(loss_gen, generator.trainable_variables)
    gradients_discriminator = dt.gradient(loss_disc, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # 첫번째 이미지를 바로 생성합니다
    if epoch == 0:
      generate_save_images(generator,
                              epoch+1,
                              seed)

    # 에포크에서 걸린 시간은 다음과 같습니다
    print('에포크 {} 에서 소요된 시간은 {} 초'.format(epoch+1, time.time()-start))

  # 마지막 에포크를 끝낸 후 생성합니다.
  generate_save_images(generator,
                           epochs,
                           seed)

def generate_save_images(model, epoch, input):
  # training이 False, 즉 Batch Normalization 및 모든 충돌은 추론 모드로 실행
  predict = model(input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predict.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predict[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

train(train_data, EPOCHS)


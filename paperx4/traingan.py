import os
import random
import numpy as np
import keras.backend as K
from keras import layers as L
from keras.models import Model, load_model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import tqdm
import cv2


epochs = 1000
batch_size = 64
dpi = 1
gpi = 1
steps_per_epoch = 400

try:
    os.mkdir('/kaggle/working/X_val')
    os.mkdir('/kaggle/working/y_val')
    os.mkdir('/kaggle/working/epoch_gen_imgs')
    os.mkdir('/kaggle/working/checkpoint_models')
except OSError as e:
    pass

# load val
X_val = []
for i, file in enumerate(sorted(os.listdir('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'))):
    img_path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + \
        str(file)
    pic = cv2.imread(img_path)
    xpic = cv2.resize(pic, (32, 32))
    ypic = cv2.resize(pic, (128, 128))
    X_val.append(xpic)
    cv2.imwrite('/kaggle/working/X_val/'+str(file), xpic)
    cv2.imwrite('/kaggle/working/y_val/'+str(file), ypic)
    if i == 9:
        break

X_val = np.array(X_val)


vgg = VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
vgg.trainable = False
for l in vgg.layers:
    l.trainable = False
vggmodel = Model(inputs=vgg.input,
                 outputs=vgg.get_layer('block5_conv4').output)
vggmodel.trainable = False


def perceptual_loss(y_true, y_pred):
    p_loss = K.mean(K.square(vggmodel(y_true)-vggmodel(y_pred)))
    return p_loss


def ResBlock(x, filters):
    res = x
    x = L.Conv2D(filters=filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.PReLU(shared_axes=[1, 2])(x)
    x = L.Conv2D(filters=filters, kernel_size=(
        3, 3), strides=(1, 1), padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Add()([res, x])
    return x


def disc_conv_block(x, filters, kernel_size, strides):
    x = L.Conv2D(filters=filters, kernel_size=kernel_size,
                 strides=strides, padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.LeakyReLU()(x)
    return x


def create_generator():
    img = L.Input(shape=(32, 32, 3))
    x = L.Conv2D(filters=64, kernel_size=(9, 9),
                 strides=(1, 1), padding='same')(img)
    x = L.PReLU(shared_axes=[1, 2])(x)
    res = x
    for i in range(16):
        x = ResBlock(x, 64)
    x = L.Conv2D(filters=64, kernel_size=(3, 3),
                 strides=(1, 1), padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Add()([res, x])
    x = L.Conv2D(filters=256, kernel_size=(3, 3),
                 strides=(1, 1), padding='same')(x)
    x = L.UpSampling2D()(x)
    x = L.PReLU(shared_axes=[1, 2])(x)
    x = L.Conv2D(filters=256, kernel_size=(3, 3),
                 strides=(1, 1), padding='same')(x)
    x = L.UpSampling2D()(x)
    x = L.PReLU(shared_axes=[1, 2])(x)
    x = L.Conv2D(filters=3, kernel_size=(9, 9),
                 strides=(1, 1), padding='same',activation='relu')(x)
    gen = Model(inputs=img, outputs=x)
    return gen


def create_discriminator():
    img = L.Input(shape=(128, 128, 3))
    x = L.Conv2D(filters=64, kernel_size=(3, 3),
                 strides=(1, 1), padding='same')(img)
    x = disc_conv_block(x, filters=64, kernel_size=(3, 3), strides=(2, 2))
    x = disc_conv_block(x, filters=128, kernel_size=(3, 3), strides=(1, 1))
    x = disc_conv_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2))
    x = disc_conv_block(x, filters=256, kernel_size=(3, 3), strides=(1, 1))
    x = disc_conv_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2))
    x = disc_conv_block(x, filters=512, kernel_size=(3, 3), strides=(1, 1))
    x = disc_conv_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2))
    x = L.Flatten()(x)
    x = L.Dense(1024)(x)
    x = L.LeakyReLU()(x)
    x = L.Dense(1)(x)
    x = L.Activation('sigmoid')(x)
    dis = Model(inputs=img, outputs=x)
    dis.compile(loss=binary_crossentropy,
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return dis


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = L.Input(shape=(32, 32, 3))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x, gan_output])
    gan.compile(loss=[perceptual_loss, binary_crossentropy], loss_weights=[1, 1e-3],
                optimizer=Adam(0.0002, beta_1=0.5, epsilon=1e-8))
    return gan


generator = load_model('/kaggle/input/checkpoint-x4sr-20/x4srpgeneratorep20.hdf5') #load_model('/kaggle/input/pretrained-x4-gen/x4.hdf5') #create_generator()
discriminator = load_model('/kaggle/input/checkpoint-x4sr-20/x4srpdiscriminatorep20.hdf5') #create_discriminator()
gan = create_gan(discriminator, generator)

print(gan.summary())

outputs_list = sorted(os.listdir(
    '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'))[10:]
for e in range(21, epochs+1):
    discriminator_loss = 0
    gan_loss = [0, 0, 0]
    print("Epoch %d" % e)
    for step in tqdm.tqdm_notebook(range(steps_per_epoch)):
        for _ in range(dpi):
            image_batch_X = []
            image_batch_y = []
            ls = random.sample(outputs_list, k=batch_size)
            for file in ls:
                img_path1 = str(
                    '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + str(file))
                pic1 = cv2.imread(img_path1)
                xpic = cv2.resize(pic1, (32, 32))
                ypic = cv2.resize(pic1, (128, 128))
                image_batch_X.append(xpic)
                image_batch_y.append(ypic)
            image_batch_X = np.array(image_batch_X)
            image_batch_y = np.array(image_batch_y)
            generated_images = generator.predict(image_batch_X)
            generated_images = generated_images.astype(int)
            X = np.concatenate([image_batch_y, generated_images])
            y_dis = np.zeros(2 * batch_size)
            y_dis[batch_size:] = y_dis[batch_size:] + \
                np.random.random_sample(batch_size)*0.2
            y_dis[:batch_size] = 1
            y_dis[:batch_size] = y_dis[:batch_size] - \
                np.random.random_sample(batch_size)*0.2  # label smoothing
            discriminator.trainable = True
            discriminator_loss += discriminator.train_on_batch(X, y_dis)
        for _ in range(gpi):
            ls = random.sample(outputs_list, k=batch_size)
            image_batch_X = []
            image_batch_y = []
            for file in ls:
                img_path1 = str(
                    '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + str(file))
                pic1 = cv2.imread(img_path1)
                xpic = cv2.resize(pic1, (32, 32))
                ypic = cv2.resize(pic1, (128, 128))
                image_batch_X.append(xpic)
                image_batch_y.append(ypic)
            image_batch_X = np.array(image_batch_X)
            image_batch_y = np.array(image_batch_y)
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan_loss_ls = gan.train_on_batch(
                image_batch_X, [image_batch_y, y_gen])
            for i in range(len(gan_loss_ls)):
                gan_loss[i] += gan_loss_ls[i]
    pr = generator.predict(X_val[random.randint(0, 9)].reshape(
        1, 32, 32, 3)).reshape(128, 128, 3)
    pr = pr.astype(int)
    cv2.imwrite('/kaggle/working/epoch_gen_imgs/x4prsrp'+str(e)+'.jpg', pr)
    print("Discriminator loss="+str(discriminator_loss/(dpi*steps_per_epoch)))
    for i in range(len(gan_loss)):
        gan_loss[i] = gan_loss[i]/(gpi*steps_per_epoch)
    print("GAN loss="+str(gan_loss))
    if e % 5 == 0:
        generator.save(
            '/kaggle/working/checkpoint_models/x4srpgeneratorep'+str(e)+'.hdf5')
        discriminator.save(
            '/kaggle/working/checkpoint_models/x4srpdiscriminatorep'+str(e)+'.hdf5')
        if os.path.exists('/kaggle/working/checkpoint_models/x4srpgeneratorep'+str(e-10)+'.hdf5'):
            os.remove(
                '/kaggle/working/checkpoint_models/x4srpgeneratorep'+str(e-10)+'.hdf5')
            os.remove(
                '/kaggle/working/checkpoint_models/x4srpdiscriminatorep'+str(e-10)+'.hdf5')
import os
import random
import numpy as np
import keras.backend as K
from keras import layers as L
from keras.models import Model, load_model
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import tqdm
import cv2


epochs = 1000
batch_size = 64
dpi = 1
gpi = 1
steps_per_epoch = 400

# load val
X_val = []
for i, file in enumerate(os.listdir('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba')):
    img_path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + \
        str(file)
    pic = cv2.imread(img_path)
    xpic = cv2.resize(pic, (32, 32))
    ypic = cv2.resize(pic, (64, 64))
    X_val.append(xpic)
    cv2.imwrite('/kaggle/working/y_val/'+str(file), ypic)
    if i == 9:
        break

X_val = np.array(X_val)


vgg = VGG19(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
vgg.trainable = False
for l in vgg.layers:
    l.trainable = False
vggmodel = Model(inputs=vgg.input, outputs=vgg.get_layer('block5_conv4').output)
vggmodel.trainable = False


def perceptual_loss(y_true, y_pred):
    p_loss = K.mean(K.square(vggmodel(y_true)-vggmodel(y_pred)))
    return p_loss


def create_generator():
    img = L.Input(shape=(32, 32, 3))
    x = L.Conv2D(filters=64, kernel_size=(5, 5),
                 padding='same', strides=(1, 1))(img)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=128, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=256, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=512, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Flatten()(x)
    x = L.Dense(units=4*4*1024)(x)
    x = L.LeakyReLU()(x)
    x = L.Reshape(target_shape=(4, 4, 1024))(x)
    x = L.Conv2DTranspose(filters=512, kernel_size=(5, 5),
                          padding='same', strides=(1, 1))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=256, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=128, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=64, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.LeakyReLU()(x)
    x = L.Conv2DTranspose(filters=3, kernel_size=(5, 5),
                          padding='same', strides=(2, 2))(x)
    x = L.Activation('tanh')(x)
    gen = Model(inputs=img, outputs=x)
    return gen


def create_discriminator():
    inp = L.Input(shape=(64, 64, 3))
    x = L.Conv2D(filters=64, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=128, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=256, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=512, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Conv2D(filters=1024, kernel_size=(5, 5),
                 padding='same', strides=(2, 2))(inp)
    x = L.LeakyReLU()(x)
    x = L.Flatten()(x)
    x = L.Dense(1)(x)
    x = L.Activation('sigmoid')(x)
    dis = Model(inputs=inp, outputs=x)
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


# load_model('/kaggle/input/checkpoint-100-facegan/facegeneratorep100.hdf5')
generator = create_generator()
# load_model('/kaggle/input/checkpoint-100-facegan/facediscriminatorep100.hdf5')
discriminator = create_discriminator()
gan = create_gan(discriminator, generator)

print(gan.summary())

outputs_list = os.listdir(
    '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba')[10:]
for e in range(1, epochs+1):
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
                ypic = cv2.resize(pic1, (64, 64))
                image_batch_X.append(xpic)
                image_batch_y.append(ypic)
            image_batch_X = np.array(image_batch_X)
            image_batch_y = np.array(image_batch_y)
            image_batch_X = (image_batch_X.astype(float)-127.5)/127.5
            image_batch_y = (image_batch_y.astype(float)-127.5)/127.5
            generated_images = generator.predict(image_batch_X)
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
            image_batch_X=[]
            image_batch_y=[]
            for file in ls:
                img_path1 = str(
                    '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + str(file))
                pic1 = cv2.imread(img_path1)
                xpic = cv2.resize(pic1, (32, 32))
                ypic = cv2.resize(pic1, (64, 64))
                image_batch_X.append(xpic)
                image_batch_y.append(ypic)
            image_batch_X = np.array(image_batch_X)
            image_batch_X = (image_batch_X.astype(float)-127.5)/127.5
            image_batch_y = np.array(image_batch_y)
            image_batch_y = (image_batch_y.astype(float)-127.5)/127.5
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan_loss_ls = gan.train_on_batch(image_batch_X, [image_batch_y,y_gen])
            for i in range(len(gan_loss_ls)):
                gan_loss[i] += gan_loss_ls[i] 
    pr = generator.predict(X_val[random.randint(0,9)].reshape(1,32,32,3)).reshape(64,64,3)
    pr = ((pr*127.5)+127.5).astype(int)
    cv2.imwrite('/kaggle/working/epoch_gen_imgs/prsr'+str(e)+'.jpg', pr)
    print("Discriminator loss="+str(discriminator_loss/(dpi*steps_per_epoch)))
    for i in range(len(gan_loss)):
        gan_loss[i] = gan_loss[i]/(gpi*steps_per_epoch) 
    print("GAN loss="+str(gan_loss))
    if e % 5 == 0:
        generator.save(
            '/kaggle/working/checkpoint_models/srgeneratorep'+str(e)+'.hdf5')
        discriminator.save(
            '/kaggle/working/checkpoint_models/srdiscriminatorep'+str(e)+'.hdf5')
generator.save('/kaggle/working/srgenerator.hdf5')
discriminator.save('/kaggle/working/srdiscriminator.hdf5')
pr = generator.predict(np.random.normal(0, 1, [1, 100]), batch_size=1)
pr = pr.reshape(64, 64, 3)
pr = ((pr*127.5)+127.5).astype(int)
print(pr)
cv2.imwrite('/kaggle/working/face.jpg', pr)

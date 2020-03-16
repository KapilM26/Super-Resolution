import os
import random
import numpy as np
from keras import layers as L
from keras.models import Model, load_model
from keras.losses import binary_crossentropy, mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import cv2


epochs = 1
batch_size = 64
dpi = 1
gpi = 1
steps_per_epoch = 1

try:
    os.mkdir('/kaggle/working/X_val')
    os.mkdir('/kaggle/working/y_val')
except OSError as e:
    pass

# load val
X_val = []
y_val = []
for i, file in enumerate(sorted(os.listdir('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'))):
    img_path = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/' + \
        str(file)
    pic = cv2.imread(img_path)
    xpic = cv2.resize(pic, (32, 32))
    ypic = cv2.resize(pic, (128, 128))
    X_val.append(xpic)
    y_val.append(ypic)
    cv2.imwrite('/kaggle/working/X_val/'+str(file), xpic)
    cv2.imwrite('/kaggle/working/y_val/'+str(file), ypic)
    if i == 9:
        break

X_val = np.array(X_val)
y_val = np.array(y_val)

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
    gen.compile(loss=mean_squared_error,
                optimizer=Adam(),metrics=['accuracy'])
    return gen



def load_inputs(path_x, path_y):
    x = []
    y = []
    while (True):
        for i in sorted(os.listdir(path_x))[10:]:
            pic_x = cv2.imread(path_x + i)
            pic_x = cv2.resize(pic_x,(32,32))
            x.append(pic_x)
            pic_y = cv2.imread(path_y + i)
            pic_y = cv2.resize(pic_y,(128,128))
            y.append(pic_y)
            if len(x) == batch_size:
                x = np.array(x)
                y = np.array(y)
                yield x, y
                x = []
                y = []


if __name__ == "__main__":
    generator = create_generator()
        
    generator.fit_generator(
        load_inputs(str('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'),str('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/')),
        steps_per_epoch=1000, 
        epochs=10,
        callbacks=[ModelCheckpoint(filepath='/kaggle/working/{epoch:02d}.hdf5')],
        validation_data=[X_val,y_val],)
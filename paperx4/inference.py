import os
import cv2
from keras.models import load_model

model = load_model('x4srpgeneratorep60.hdf5',compile=False)
for f in os.listdir('X_val'):
    img = cv2.imread('X_val\\'+str(f))
    #img = cv2.resize(img,(32,32))
    #cv2.imwrite('rim'+str(i+1)+'.jpg',img)
    img = img.reshape((1,32,32,3))
    pred = model.predict(img)
    pred = pred.reshape((128,128,3))
    cv2.imwrite('val_pred60\\'+str(f),pred)
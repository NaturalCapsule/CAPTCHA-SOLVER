import os
import numpy as np
import cv2
from tensorflow.keras.layers import *
import tensorflow as tf
captcha_list = []
img_shape = (50, 200, 1)
symbols = list(map(chr, range(97, 123))) + list(map(chr, range(48, 58)))

len_symbols = len(symbols)
nSamples = len(os.listdir('samples'))
len_captcha = 5

X = np.zeros((nSamples, 50, 200, 1), dtype=np.float32)
y = np.zeros((5, nSamples, len_symbols))

for i, captcha in enumerate(os.listdir('samples')):
    captcha_code = captcha.split(".")[0]
    captcha_list.append(captcha_code)

    captcha_cv2 = cv2.imread(os.path.join('C:/Users/sxxve/Music/Python/samples', captcha), cv2.IMREAD_GRAYSCALE)

    if captcha_cv2 is None:
        print(f"Error loading image {captcha}")
        continue

    # Reshape and normalize
    captcha_cv2 = captcha_cv2.astype(np.float32) / 255.0

    if captcha_cv2.shape != img_shape[:2]:
        print(f"Image {captcha} has shape {captcha_cv2.shape}, expected {img_shape[:2]}")
        captcha_cv2 = cv2.resize(captcha_cv2, (img_shape[1], img_shape[0]))
    captcha_cv2 = np.reshape(captcha_cv2, img_shape)

    targs = np.zeros((len_captcha, len_symbols))

    for a, b in enumerate(captcha_code):
        targs[a, symbols.index(b)] = 1

    X[i] = captcha_cv2
    y[:, i] = targs

X_train = X[:856]
y_train = y[:, :856]
X_test = X[856:]
y_test = y[:, 856:]

captcha = Input(shape=(50,200,1))
x = Conv2D(16, (3,3),padding='same',activation='relu')(captcha)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = Conv2D(32, (3,3),padding='same',activation='relu')(x)
x = MaxPooling2D((2,2) , padding='same')(x)
x = BatchNormalization()(x)

flatOutput = Flatten()(x)

dense1 = Dense(64 , activation='relu')(flatOutput)
dropout1= Dropout(0.5)(dense1)
output1 = Dense(len_symbols , activation='sigmoid' , name='char_1')(dropout1)

dense2 = Dense(64 , activation='relu')(flatOutput)
dropout2= Dropout(0.5)(dense2)
output2 = Dense(len_symbols , activation='sigmoid' , name='char_2')(dropout2)

dense3 = Dense(64 , activation='relu')(flatOutput)
dropout3= Dropout(0.5)(dense3)
output3 = Dense(len_symbols , activation='sigmoid' , name='char_3')(dropout3)

dense4 = Dense(64 , activation='relu')(flatOutput)
dropout4= Dropout(0.5)(dense4)
output4 = Dense(len_symbols , activation='sigmoid' , name='char_4')(dropout4)

dense5 = Dense(64 , activation='relu')(flatOutput)
dropout5= Dropout(0.5)(dense5)
output5 = Dense(len_symbols , activation='sigmoid' , name='char_5')(dropout5)

model = tf.keras.Model(inputs = captcha , outputs=[output1 , output2 , output3 , output4 , output5])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
earlystopping = tf.keras.callbacks.EarlyStopping(monitor ="val_loss",
                             mode ="min", patience = 5,
                             restore_best_weights = True)

history = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=60, verbose=1, validation_split=0.2, callbacks =[earlystopping])

score = model.evaluate(X_test,[y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]],verbose=1)

print('Test Loss and accuracy:', score)

# def preprocess_image(image_path, img_shape):
#     # Load the image in grayscale
#     captcha_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Resize if necessary
#     captcha_cv2 = cv2.resize(captcha_cv2, (img_shape[1], img_shape[0]))
#
#     # Normalize and reshape
#     captcha_cv2 = captcha_cv2.astype(np.float32) / 255.0
#     captcha_cv2 = np.reshape(captcha_cv2, img_shape)
#
#     # Add batch dimension (1, 50, 200, 1)
#     captcha_cv2 = np.expand_dims(captcha_cv2, axis=0)
#
#     return captcha_cv2
model.save('model.h5')
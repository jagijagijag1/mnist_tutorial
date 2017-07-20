# ==============================================================================
# This code is derived from
# https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
# and partially modified
# ==============================================================================
# The LISENCE of original code is on
# https://github.com/fchollet/keras/blob/master/LICENSE
# ==============================================================================

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


batchsize = 100
num_classes = 10
epoch = 5

# the data, shuffled and split between train and test sets
print("data loading...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("DONE")

# 画像(28x28) -> 一次元ベクトル化
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Network definition
model = Sequential()
## 1st layer
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
## 2nd layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
## 3rd layer
model.add(Dense(10, activation='softmax'))

# display constructed model
model.summary()

# Run the training
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batchsize,
                    epochs=epoch,
                    verbose=1,
                    validation_data=(x_test, y_test))

# evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

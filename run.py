import os
import shutil
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from datetime import datetime
from datagen import TSNEBatchGenerator

# Load data
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()
(X_train, y_train, X_test, y_test) = X_train[:, :, :, np.newaxis], y_train[:, np.newaxis], X_test[:, :, :, np.newaxis], y_test[:, np.newaxis]

# Normalization
X_train, X_test = X_train / 255.0, X_test / 255.0

# Define model
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1", input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_2"))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2, activation='softmax'))
model.summary()

def cost_func(y_true, y_pred):
    x_batch = y_true
    y_batch = y_pred

    # batch_size = K.int_shape(x_batch)[0]
    batch_size = 32

    x_batch = K.reshape(x_batch, (batch_size, -1))
    y_batch = K.reshape(y_batch, (batch_size, -1))

    sigma = 784 / np.sqrt(2)

    # 
    rx = K.tf.reduce_sum( x_batch * x_batch, 1)
    rx = K.reshape(rx, (-1, 1))
    Gx = rx - 2 * K.tf.matmul(x_batch, K.tf.transpose(x_batch)) + K.tf.transpose(rx)
    x_numerator = K.exp( - Gx / (2 * np.square(sigma)))
    x_denominator = K.tf.reduce_sum(x_numerator)
    
    ry = K.tf.reduce_sum( y_batch * y_batch, 1)
    ry = K.reshape(ry, (-1, 1))
    Gy = ry - 2 * K.tf.matmul(y_batch, K.tf.transpose(y_batch)) + K.tf.transpose(ry)
    y_numerator = 1 / (1 + Gy)
    y_denominator = K.tf.reduce_sum(y_numerator)

    P = x_numerator / x_denominator
    Q = y_numerator / y_denominator

    KL = K.tf.reduce_sum(P * K.log(P / Q))
    return KL

model.compile(optimizer=keras.optimizers.SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), loss=cost_func)

train_batch_generator = TSNEBatchGenerator(batch_size=32)

date_string = datetime.now().strftime('%Y%m%d %H:%M:%S')
os.mkdir('./log/'+date_string)
print("model logdir :", "./log/"+date_string)

callbacks=[]
callbacks.append(keras.callbacks.CSVLogger(filename='./log/'+date_string+'/metrics.csv'))
callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./log/'+date_string+'/bestweights.hdf5', 
                                                    monitor='loss', 
                                                    save_best_only=True))


"""
history = model.fit(x=X_train, y=X_train, 
            batch_size=8, 
            epochs=100, 
            verbose=1, 
            callbacks=None, 
            validation_split=0.0, 
            validation_data=None, 
            shuffle=True, 
            class_weight=None, 
            sample_weight=None, 
            initial_epoch=0, 
            steps_per_epoch=None, 
            validation_steps=None)


"""
history= model.fit_generator(train_batch_generator, 
                                  steps_per_epoch=train_batch_generator.__len__(), 
                                  epochs=100, 
                                  verbose=1, 
                                  callbacks=callbacks, 
                                  validation_data=None, 
                                  validation_steps=None, 
                                  class_weight=None, 
                                  max_queue_size=1, 
                                  workers=4,
                                  use_multiprocessing=False, 
                                  shuffle=False, 
                                  initial_epoch=0)

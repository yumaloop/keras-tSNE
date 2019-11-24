import os
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from datetime import datetime
from datagen import TSNEBatchGenerator
from tsne import tsne_loss

def main():
    mnist = tf.keras.datasets.mnist
    (X_train, y_train),(X_test, y_test) = mnist.load_data()

    # Define model
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_1", input_shape=(28, 28, 1)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', name="conv2d_2"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2))
    model.summary()

    model.compile(optimizer=keras.optimizers.SGD(lr=2e-2, momentum=0.9, decay=0.0, nesterov=False), loss=tsne_loss(batch_size=100))
    train_batch_generator = TSNEBatchGenerator(batch_size=100)

    date_string = datetime.now().strftime('%Y%m%d %H:%M:%S')
    os.mkdir('./log/'+date_string)
    print("model logdir :", "./log/"+date_string)

    callbacks=[]
    callbacks.append(keras.callbacks.CSVLogger(filename='./log/'+date_string+'/metrics.csv'))
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./log/'+date_string+'/bestweights.hdf5', 
                                                        monitor='loss', 
                                                        save_best_only=True))

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


if __name__ == '__main__':
    main()

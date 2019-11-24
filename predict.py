import os
import keras
import keras.backend as K
import numpy as np
from tqdm import tqdm
from datagen import cost_func, TSNEBatchGenerator

def cost_func(y_true, y_pred):
    x_batch = y_true
    y_batch = y_pred

    # batch_size = K.int_shape(x_batch)[0]
    batch_size = 1   

    x_batch = K.reshape(x_batch, (batch_size, -1))
    y_batch = K.reshape(y_batch, (batch_size, -1))

    sigma = 784 / np.sqrt(2)

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

model = keras.models.load_model("./log/base1121/bestweights.hdf5", custom_objects={"cost_func": cost_func})
model.summary()

bg = TSNEBatchGenerator(batch_size=1)


y_latent = []
for i in tqdm(range(bg.num)):
    X, _ = bg.__getitem__(i)
    y = model.predict(X)[0]
    y_latent.append(y)
    
y_latent = np.array(y_latent)
np.save( os.path.join("./log/base1121/", "y_latent.npy"), y_latent)
    


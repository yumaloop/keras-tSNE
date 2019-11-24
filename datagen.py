import os
import itertools
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np
import scipy

def cost_func(y_true, y_pred):
    x_batch = y_true
    y_batch = y_pred

    # batch_size = K.int_shape(x_batch)[0]
    batch_size = 1000

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

def gram_matrix(d_batch):
    batch_size = d_batch.shape[0]
    d_batch = d_batch.reshape(batch_size, -1)
    sigma = 1 / np.sqrt(2)

    denominator = 0
    index_list = [n for n in range(batch_size)]
    for i, j in itertools.combinations(index_list, 2):
        denominator += np.exp( - np.sum( np.square( d_batch[i] - d_batch[j] ) ) / (2 * (sigma ** 2)))

    P = np.zeros((batch_size, batch_size))
    for i in range(batch_size):
        for j in range(batch_size):
            numerator = np.exp( - np.sum( np.square( d_batch[i] - d_batch[j] ) ) / (2 * (sigma ** 2)))
            P[i][j] = numerator / denominator

    return P

class TSNEBatchGenerator(keras.utils.Sequence):

    def __init__(self, batch_size=128):
        # Load data
        mnist = tf.keras.datasets.mnist
        (X_train, y_train),(X_test, y_test) = mnist.load_data()
        (X_train, y_train, X_test, y_test) = X_train[:, :, :, np.newaxis], y_train[:, np.newaxis], X_test[:, :, :, np.newaxis], y_test[:, np.newaxis]

        # Normalization
        self.X_train = X_train.astype('float32') / 255.
        self.X_test  = X_test.astype('float32')  / 255.

        self.num = len(self.X_train)
        self.batch_size = batch_size
        self.batches_per_epoch = int((self.num - 1) / self.batch_size) + 1

    def __getitem__(self, idx):
        """
        idx: batch id
        """
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.num:
            batch_to = self.num

        x_batch = self.X_train[batch_from:batch_to]
        p_batch, _ = self.solvePS(x_batch)
        # y_batch = gram_matrix(x_batch) # P
        # print("idx :", idx, "batch_from :", batch_from, "batch_to :", batch_to, "self.num", self.num)
        
        return x_batch, p_batch

    def __len__(self):
        """
        batch length: number of batch data in one epoch
        """
        return self.batches_per_epoch

    def on_epoch_end(self):
        pass

    def calcP(self, X, S):
        """
        X: data (N x N)
        S: sigma (N x N)
        """
        N = len(X)
        def squared_sum(x, y):
            return np.linalg.norm(x - y) ** 2
        D = squared_sum(*np.meshgrid(X, X))
        P = np.exp( - D / 2 * (S ** 2))
        P = P * (1 - np.eye(N))
        return P

    def calcHi(self, P, i):
        N = P.shape[0]
        H_i = 0
        for j in range(N):
            H_i += - P[j][i] * np.log(P[j][i])
        return H_i

    def solvePS(self, X, perp=30, threshold=1e-5, iter_max=100):
        N = len(X)
        S = np.ones(N)

        for i in range(N):
            smin = -np.inf
            smax =  np.inf

            P = self.calcP(X, S)
            H_i = self.calcHi(P, i)
            H_diff = H_i - np.log2(perp)

            while H_diff > threshold and iter < iter_max:
                if H_diff > 0:
                    smin = S[i]
                    if smax == -np.inf:
                        S[i] = S[i] * 2
                    else:   
                        S[i] = (smin + smax) / 2
                else:
                    smax = S[i]
                    if smin == -np.inf:
                        S[i] = S[i] / 2
                    else:
                        S[i] = (smin + smax) / 2

                P = self.calcP(X, S)
                H_i = self.calcHi(P, i)
                H_diff = H_i - np.log2(perp)
                iter += 1
        return P, S


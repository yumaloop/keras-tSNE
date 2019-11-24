import numpy as np
import keras
import keras.backend as K
import tensorflow as tf


def calcP(X, S):
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

def calcHi(P, i):
    N = P.shape[0]
    H_i = 0
    for j in range(N):
        H_i += - P[j][i] * np.log(P[j][i])
    return H_i

def bisect(X, perp=30, threshold=1e-5, iter_max=100):
    N = len(X)
    S = np.ones(N)

    for i in range(N):
        smin = -np.inf
        smax =  np.inf

        P = calcP(X, S)
        H_i = calcHi(P, i)
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

            P = calcP(X, S)
            H_i = calcHi(P, i)
            H_diff = H_i - np.log2(perp)
            iter += 1
    return P, S



def cost_func(batch_size):
    def KLdivergence(P, Y):
        sum_Y = K.sum(K.square(Y), axis=1)
        eps = K.variable(10e-15)
        D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
        Q = K.pow(1 + D, -1)
        Q *= K.variable(1 - np.eye(batch_size))
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((P + eps) / (Q + eps))
        C = K.sum(P * C)
        return C
    return KLdivergence


def main(batch_size=3000, nb_epoch=100):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 1)
    X_test  = X_test.reshape(-1, 1)
    X_train = X_train.astype('float32')
    X_test  = X_test.astype('float32')
    X_train /= 255.
    X_test  /= 255.

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(500, input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(500))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(2000))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(2))
    model.summary()

    model.compile(loss=cost_func(batch_size=batch_size), optimizer="adam")

    N = len(X_train)
    for epoch in range(nb_epoch):
        batch_num = int(N // batch_size)
        m = batch_num * batch_size

        """
        # shuffle X_train and calculate P
        shuffle_interval = nb_epoch + 1
        if epoch % shuffle_interval == 0:
            X_train = X_train
            # X = X_train[np.random.permutation(N)[:batch_size]]
            P, S = bisect(X)
        """

        # train
        loss = 0
        for i in range(0, N, batch_size):
            X = X_train[i:i+batch_size]
            P, S = bisect(X)
            loss += model.train_on_batch(X, P)
        print("Epoch: {}/{}, loss: {}".format(epoch+1, nb_epoch, loss / batch_num))

        # visualize training process
        pred = model.predict(X_test)

if __name__ == "__main__":
    main()

import numpy as np
import keras
import keras.backend as K
from tqdm import tqdm

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

def calcS_by_bisect(X, perp=30, threshold=1e-5, iter_max=100):
    N = len(X)
    S = np.ones(N)

    print("Calcurating sigma by bisection method ...")

    for i in tqdm(range(N)):
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


def tsne_loss(batch_size=1000, perp=30):
    
    def cost(p_batch, y_batch):

        p_batch = K.reshape(p_batch, (batch_size, -1))
        y_batch = K.reshape(y_batch, (batch_size, -1))

        eps = K.variable(10e-15)
        
        ry = K.tf.reduce_sum( y_batch * y_batch, 1)
        ry = K.reshape(ry, (-1, 1))
        Gy = ry - 2 * K.tf.matmul(y_batch, K.tf.transpose(y_batch)) + K.tf.transpose(ry)
        y_numerator = (1 / (1 + Gy)) + (1 - K.eye(batch_size))
        y_denominator = K.tf.reduce_sum(y_numerator)

        P = p_batch
        Q = y_numerator / y_denominator

        KL = K.tf.reduce_sum(P * K.log((P + eps) / (Q + eps)))
        return KL
    return cost


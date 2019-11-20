import tensorflow as tf
import numpy as np

# mnistデータの読み込み
mnist = tf.keras.datasets.mnist
(X_train, y_train),(X_test, y_test) = mnist.load_data()

# npzファイルでデータを保存
np.savez_compressed('./mnist-data.npz',
                   X_train=X_train,
                   y_train=y_train,
                   X_test=X_test,
                   y_test=y_test)

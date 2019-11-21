import os
import keras
import numpy as np
from tqdm import tqdm
from datagen import cost_func, TSNEBatchGenerator

model = keras.models.load_model("./log/base_tsne/bestweights.hdf5", custom_objects={"cost_func": cost_func})
model.summary()

bg = TSNEBatchGenerator(batch_size=1)

for i in tqdm(range(bg.num)):
    X, _ = bg.__getitem__(i)
    y = model.predict(X)
    np.save( os.path.join("./log/base_tsne/y_latent", "y2d_"+str(i)+".npy"), y)
    


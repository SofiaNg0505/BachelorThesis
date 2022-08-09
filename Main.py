import numpy as np
from Valerio import VAE
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd

"""conditions"""
LEARNING_RATE = 0.0005
BATCH_SIZE = 70
EPOCHS = 200


def load_mnist():

    data = (np.load('3D_dataset.npy')) 
    x_train=data[0:6500,:,:,0]
    y_train=data[0:6500,0,0,0]
    x_test=data[6500:8000,:,:,0]
    y_test=data[6500:8000,0,0,0]
    
    x_train = x_train.astype("float32") / 59
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 59
    x_test = x_test.reshape(x_test.shape + (1,))
    

    return x_train, y_train, x_test, y_test

def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(32,32,1), 
        conv_filters=(32,64,256), 
        conv_kernels=(3,3,3),
        conv_strides=(2,2,2), 
        latent_space_dim= 2000
        )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    
    return vae


if __name__ == "__main__":
    x_train,_,_,_=load_mnist()
    vae = train(x_train[0:5715], LEARNING_RATE, BATCH_SIZE, EPOCHS)

    vae.save("model_2k#2")
    

golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))

fig, ax = plt.subplots(figsize=golden_size(6))
result =vae.fit(x_train[0:4000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
history=result.fit()
hist_df = pd.DataFrame(result.history)
hist_df.plot(ax=ax)

ax.set_ylabel('NELBO')
ax.set_xlabel('# epochs')

ax.set_ylim(.99*hist_df[1:].values.min(), 
        1.1*hist_df[1:].values.max())
plt.show()
    


    


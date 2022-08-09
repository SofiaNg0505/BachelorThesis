# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 21:57:24 2021

@author: Sofia Nguyen
"""
#Second part of how to implement VAE in Keras
#%%
import numpy as np
print("bajs")
from Valerio import VAE
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
LEARNING_RATE = 0.0005#A smaller learning rate takes closer to the minimum, but it takes more time and in case of a larger learning rate.
BATCH_SIZE = 70
EPOCHS = 200#dator=mnist.load_data()
#print('halloj',dator[1:2])  
#data.shape= 8000,32,32,32
#data = np.load('3D_dataset.npy')
#print('ok',data[0:6500,0])

def load_mnist():

    
    data = (np.load('3D_dataset.npy')) #Keras will divide it 60.000 for train and 10.000 for test    
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
        conv_filters=(32,64,256), #filters are in same dim as input with same nr. of channels, but fewer rows and columns
        conv_kernels=(3,3,3),
        conv_strides=(2,2,2), #The amount of movement between applications of the filter to the input image.Default in 2D is (1,1) for the height and the width movement. 
        latent_space_dim= 2000#filters are in same dim as input with same nr. of channels, but fewer rows and columns
         #4 nr of weights ,#32*32 pixel filter size
         #The amount of movement between applications of the filter to the input image.Default in 2D is (1,1) for the height and the width movement. 
        )
    vae.summary()
    vae.compile(learning_rate)
    vae.train(x_train, batch_size, epochs)
    
    return vae


if __name__ == "__main__":
    x_train,_,_,_=load_mnist()
    #we are really just interested in x_train 
    vae = train(x_train[0:5715], LEARNING_RATE, BATCH_SIZE, EPOCHS)

    vae.save("model_2k#2")
    

#golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
#
#fig, ax = plt.subplots(figsize=golden_size(6))
#ok=vae.fit(x_train[0:4000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
#history=ok.fit()
#hist_df = pd.DataFrame(ok.history)
#hist_df.plot(ax=ax)
# 
#ax.set_ylabel('NELBO')
#ax.set_xlabel('# epochs')
#
#ax.set_ylim(.99*hist_df[1:].values.min(), 
#        1.1*hist_df[1:].values.max())
#plt.show()
    


    

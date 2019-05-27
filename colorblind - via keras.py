
# coding: utf-8

# In[1]:


from __future__ import print_function, division

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K
import scipy

import matplotlib.pyplot as plt

import numpy as np


# In[2]:


from glob import glob
import os
from random import randint
import tensorflow as tf


# In[ ]:


class CCGAN():
    def __init__(self, batchsize = 4):
        # input image being 256 * 256 * 3&1
        self.img_rows = 256
        self.img_cols = 256
        
        #color hint RGB has 3
        self.color_channels = 3
        
        #line image has 1
        self.channels = 1
        
        self.color_img_shape = (self.img_rows, self.img_cols, self.color_channels)
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # Use standrad AdamOptimizer with 2e-4 and 0.5 beta
        optimizer = Adam(0.0002, 0.5)
        
        """
        the general keras GAN network for conv2d uses a mask to randomly slice an image of height * width from the training images; don't know if I should use in the color one"

        """
        self.mask_height = 100
        self.mask_width = 100
        
        #self.num_classes = 1

        # Number of filters in first layer of generator and discriminator
        self.gf = 64
        self.df = 64


        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes the color hint and the line image as input
        Color_hint = Input(shape=self.color_img_shape)
        Line_image = Input(shape=self.img_shape)
        Input_images = tf.concat([self.Line_image, self.Color_hint], axis=3)
        
        gen_img = self.generator(Input_images)

        # See if train discriminator or not
        #self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity - the chance of it being true images that discriminator thinks
        valid = self.discriminator(gen_img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator

        self.combined = Model(inputs=[Color_hint, Line_image], valid)
        self.combined.compile(loss=['mse'],
            optimizer=optimizer)
        
        
    def build_generator(self):
        """input:None
            output: a model object(special keras object) with attributes of defined inputs and outputs
            
            use this function to create generator"""

        def conv2d(layer_input, filters, f_size=5, bn=True):
            """Layers used during downsampling
            input: (input, # of filters, default size of filter being 5x5, batchnormalization default true)
            f_size could be 4; to be tested to find optimum
            """
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.9)(d)
                #hyper-parameter here. Tpyical conv2d GAN nets I checked used a 0.8 momentum to normalize 
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=5, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=2, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.9)(u)
            #a residual layer that concats this and previous layer
            u = Concatenate()([u, skip_input])
            return u

        color_img = Input(shape=self.color_img_shape)
        Line_img = Input(shape=self.img_shape)
        img = tf.concat([self.Line_image, self.Color_hint], axis=3)

        # Downsampling
        d1 = conv2d(img, self.gf, bn=False)
        #d1 is (128 x 128 x self.gf_dim) [height x width x #of channels]
        d2 = conv2d(d1, self.gf*2)
        #d2 is (64 x 64 x self.gf_dim*2)
        d3 = conv2d(d2, self.gf*4)
        #d3 is (32 x 32 x self.gf_dim*4)
        d4 = conv2d(d3, self.gf*8)
        #d4 is (16 x 16 x self.gf_dim*8)
        d5 = conv2d(d4, self.gf*8)
        #d5 is (8 x 8 x self.gf_dim*8)

        # Upsampling
        u1 = deconv2d(d5, d4, self.gf*8)
        #u1 is (16 x 16 x self.gf_dim*8 * 2) p.s. * 2 is because we concat deconv2d(d5) and d4, both of shape (16 x 16 x self.gf_dim*8)
        u2 = deconv2d(u1, d3, self.gf*4)
        #u2 is (32 x 32 x self.gf_dim*4 * 2)
        u3 = deconv2d(u2, d2, self.gf*2)
        #u3 is (64 x 64 x self.gf_dim*2 * 2)
        u4 = deconv2d(u3, d1, self.gf)
        #u4 is (128 x 128 x self.gf_dim * 2)
        u5 = UpSampling2D(size=2)(u4)
        #u5 is unsampled from u4 back to original image type (256 x 256 x 3)
        
        #outout_img = u5
        #here typical conv2d adds another convolutional layer to transofrm; I don't know if this is necessary as the source code in deep color didn't did so
        #also note that the convolutional size formula is (W−F+2P)/S+1; here W being 256, so F must be odd so that given Strides being 1 there exsits some P in which (W−F+2P)/S+1 = W and spatial dimensions are preserved
        output_img = Conv2D(self.color_channels, kernel_size=5, strides=1, padding='same', activation='tanh')(u5)

        return Model(inputs = [Color_hint, Line_image], output_img)
    
    def build_discriminator(self):

        img = Input(shape=self.color_img_shape)
        
        #I make batchnormalization being implemented after relu activation, a commmon sense in deep learning I believe which outperforms the opposite way, even though author in deep color uses batchnormalization before relu activation
        model = Sequential()
        model.add(Conv2D(self.df, kernel_size=5, strides=2, padding='same', input_shape=self.color_img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # here output is (128 * 128 * self.df)
        model.add(Conv2D(self.df * 2, kernel_size=5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        # here output is (64 * 64 * self.df * 2)
        model.add(Conv2D(self.df * 4, kernel_size=5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        # here output is (32 * 32 * self.df * 4)
        model.add(Conv2D(self.df * 8, kernel_size=5, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(InstanceNormalization())
        # here output is (16 * 16 * self.df * 8)
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        validity = model(img)

        return Model(img, validity)

    #just ignore this one; still working on it
    def train(self, epochs, batch_size=4, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale MNIST to 32x32
        X_train = np.array([scipy.misc.imresize(x, [self.img_rows, self.img_cols]) for x in X_train])

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 4, 4, 1))
        fake = np.zeros((batch_size, 4, 4, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Sample half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            labels = y_train[idx]

            masked_imgs = self.mask_randomly(imgs)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(masked_imgs)

            # One-hot encoding of labels
            labels = to_categorical(labels, num_classes=self.num_classes+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_classes), num_classes=self.num_classes+1)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch(masked_imgs, valid)

            # Plot the progress
            print ("%d [D loss: %f, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[4], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                # Select a random half batch of images
                idx = np.random.randint(0, X_train.shape[0], 6)
                imgs = X_train[idx]
                self.sample_images(epoch, imgs)
                self.save_model()


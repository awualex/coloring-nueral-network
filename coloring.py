# this is a psuedo code for the coloring project

import tensorflow as tf
import numpy as np
from glob import glob
import math 
import sys
import random 
#import the necessary packages above I think should be useful

'''
To present the nueral network, I want to create an object with methods and attributes. 
The attributes I think could record the values for the filter and similar weights,
and the method will be used to update the weights.
'''

class coloring_machine():
    def __init__(self, input_image_size=256, batchsize=5):
        # Hi Mr. Dartfler, the __init__ function is the same as the contructor function in javascript. And self is same as this
        self.batch_size = batchsize
        #batch is the group of training examples in one iteration to update the weights. I want to test the number of training examples in one batch from 4 ~ 10 to see which is better.
        self.image_size = input_image_size
        self.output_image_size = input_image_size

        self.line_colordim = 1 
        self.color_colordim = 3
        #here I specify the color dimension for the  line image and the color hints so that I can specify the shape and create the tensorflow values for the image

        self.line_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.line_colordim])
        self.color_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.color_colordim])
        self.real_images = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.color_colordim])
        #the second part specifies the shape of the tensors | where the line_images/ color_images represent the batch of input training examples 

        combined_preimage = tf.concat([self.line_images, self.color_images], 3)
        #concat the line images and color hints at the color dimension


        self.generated_image = self.generator(combined_preimage)
        #generator() is the generator function yet to be defined

        self.real_coloredimg = tf.concat([combined_preimage, self.real_images], 3)
        self.fake_coloredimg = tf.concat([combined_preimage, self.generated_images], 3)
        #concat at the color dimention to color the preimage for both the generated(fake) and the actual one to feed into the discriminator

        self.disc_true_logits = self.discriminator(self.real_coloredimg, realness=False)
        self.disc_fake_logits = self.discriminator(self.fake_coloredimg, realness=True)
        
        self.disc_loss_real  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_true_logits, tf.ones_like(disc_true_logits)))
        self.disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.zeros_like(disc_fake_logits)))
        '''the discriminator is desgined to return the probabilities of iamge being real for each training examples
        and the function sigmoid_cross_entropy_with_logits maps the difference between generated value and the desired one to 0~1 "input(probabilities, labels)". For the real ones it is all on; and the fake one it is all 0
        reduced mean derives the mean loss among the training examples in this batch
        '''

        self.disc_loss = self.disc_loss_real + self.disc_loss_fake
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake_logits, tf.ones_like(disc_fake_logits)))
        #set the loss function for generator and discriminator 

        t_vars = tf.trainable_variables()
        #make a list of trainable variables to devide those in generator and those in discriminator

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        #train the network

        



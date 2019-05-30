
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import cv2


# In[2]:


def get_image(image_path):
    return transform(imread(image_path))

def transform(image, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (256,256))

    return np.array(cropped_image)

def imread(path):
    readimage = cv2.imread(path, 1)
    return readimage

def merge_color(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        
        print(i,j)
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image

    return img[:,:,0]

def ims(name, img):
    print("saving img " + name)
    cv2.imwrite(name, img*255)


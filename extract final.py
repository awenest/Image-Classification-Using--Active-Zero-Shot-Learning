# !/usr/bin/env python3
#
# Copyright 2017 Zegami Ltd

"""Preprocess images using Keras pre-trained models."""

import argparse
import csv
import os

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def named_model(name):
    # include_top=False removes the fully connected layer at the end/top of the network
    # This allows us to get the feature vector as opposed to a classification
    if name == 'Xception':
        return applications.xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG16':
        return applications.vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')

    if name == 'VGG19':
        return applications.vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')

    if name == 'InceptionV3':
        return applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')

    if name == 'MobileNet':
        return applications.mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')

    return applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

'''
parser = argparse.ArgumentParser(prog='Feature extractor')
parser.add_argument('source', default=None, help='Path to the source metadata file')
parser.add_argument('model', default='VGG19', type=named_model, help='Name of the pre-trained model to use')
pargs = parser.parse_args()

source_dir = os.path.dirname(pargs.source)
'''


def get_feature(metadata):
    print('{}'.format(metadata['id']))
    if True:
        img_path = 'cat.jpg'
        if True:
            print('is file: {}'.format(img_path))
            if True:
                # load image setting the image size to 224 x 224
                img = image.load_img('cat.jpg', target_size=(224, 224))
                # convert image to numpy array
                x = image.img_to_array(img)
                # the image is now in an array of shape (3, 224, 224)
                # but we need to expand it to (1, 2, 224, 224) as Keras is expecting a list of images
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # extract the features
                features = named_model('VGG19').model.predict(x)[0]
                # convert from Numpy to a list of values
                features_arr = np.char.mod('%f', features)

                print( "id", metadata['id'], "features", ','.join(features_arr))

def start():
    if True:
        features = map(get_feature, 'cat.jpg')

        print (features)

start()
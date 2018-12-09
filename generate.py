"""
Story generation
"""
import cPickle as pkl
import numpy
import copy
import sys
import skimage.transform

import skipthoughts
import decoder
import embedding

import config

import lasagne
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
if not config.FLAG_CPU_MODE:
    from lasagne.layers.corrmm import Conv2DMMLayer as ConvLayer

from scipy import optimize, stats
from collections import OrderedDict, defaultdict, Counter
from numpy.random import RandomState
from scipy.linalg import norm

from PIL import Image, ImageFile

import tensorflow as tf
from vgg19 import VGG19

import keras
from keras.models import Model
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

ImageFile.LOAD_TRUNCATED_IMAGES = True


def story(z, im, k=100, bw=50, lyric=False):
    """
    Generate a story for an image at location image_loc
    """

    # Run image through convnet
    feats = compute_features(z['net'], im).flatten()
    feats /= norm(feats)

    # Embed image into joint space
    feats = embedding.encode_images(z['vse'], feats[None,:])

    # Compute the nearest neighbours
    scores = numpy.dot(feats, z['cvec'].T).flatten()
    sorted_args = numpy.argsort(scores)[::-1]
    sentences = [z['cap'][a] for a in sorted_args[:k]]

    print 'NEAREST-CAPTIONS: '
    for s in sentences[:5]:
        print s
    print ''

    # Compute skip-thought vectors for sentences
    svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)

    # Style shifting
    shift = svecs.mean(0) - z['bneg'] + z['bpos']

    # Generate story conditioned on shift
    passage = decoder.run_sampler(z['dec'], shift, beam_width=bw)
    print 'OUTPUT: '
    if lyric:
        for line in passage.split(','):
            if line[0] != ' ':
                print line
            else:
                print line[1:]
    else:
        print passage


def load_all():
    """
    Load everything we need for generating
    """
    print config.paths['decmodel']

    # Skip-thoughts
    print 'Loading skip-thoughts...'
    stv = skipthoughts.load_model(config.paths['skmodels'],
                                  config.paths['sktables'])

    # Decoder
    print 'Loading decoder...'
    dec = decoder.load_model(config.paths['decmodel'],
                             config.paths['dictionary'])

    # Image-sentence embedding
    print 'Loading image-sentence embedding...'
    vse = embedding.load_model(config.paths['vsemodel'])

    # VGG-19
    print 'Loading and initializing ConvNet...'

    if config.FLAG_CPU_MODE:
        net = VGG19(weights='imagenet')
    else:
        net = build_convnet(config.paths['vgg'])

    # Captions
    print 'Loading captions...'
    cap = []
    with open(config.paths['captions'], 'rb') as f:
        for line in f:
            cap.append(line.strip())

    # Caption embeddings
    print 'Embedding captions...'
    cvec = embedding.encode_sentences(vse, cap, verbose=False)

    # Biases
    print 'Loading biases...'
    bneg = numpy.load(config.paths['negbias'])
    bpos = numpy.load(config.paths['posbias'])

    # Pack up
    z = {}
    z['stv'] = stv
    z['dec'] = dec
    z['vse'] = vse
    z['net'] = net
    z['cap'] = cap
    z['cvec'] = cvec
    z['bneg'] = bneg
    z['bpos'] = bpos

    return z

def compute_features(net, im):
    """
    Compute fc7 features for im
    """
    if config.FLAG_CPU_MODE:
        from vgg19 import VGG19
        #from imagenet_utils import preprocess_input, decode_predictions
        from keras.models import Model

        base_model = VGG19(weights='imagenet')
        model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
        pred_features = model.predict(im)
        return pred_features
    else:
        fc7 = numpy.array(lasagne.layers.get_output(net['fc7'], im,
                                                    deterministic=True).eval())
        return fc7

def build_convnet(path_to_vgg):
    """
    Construct VGG-19 convnet
    """
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    print 'Loading parameters...'
    output_layer = net['prob']
    model = pkl.load(open(path_to_vgg))
    lasagne.layers.set_all_param_values(output_layer, model['param values'])

    return net



import os

"""
Configuration for the generate module
"""

#-----------------------------------------------------------------------------#
# Flags for running on CPU
#-----------------------------------------------------------------------------#
FLAG_CPU_MODE = True

#-----------------------------------------------------------------------------#
# Paths to models and biases
#-----------------------------------------------------------------------------#
paths = dict()

download_path = '/Users/bcui/dev/neural-storyteller/download/'
storyteller_path = os.path.join(os.path.sep, download_path, 'neural-storyteller')

# Skip-thoughts
paths['skmodels'] = download_path
paths['sktables'] = download_path

# Decoder
paths['decmodel'] = os.path.join(os.path.sep, storyteller_path, 'romance.npz')
paths['dictionary'] = os.path.join(os.path.sep, storyteller_path, 'romance_dictionary.pkl')

# Image-sentence embedding
paths['vsemodel'] = os.path.join(os.path.sep, storyteller_path, 'coco_embedding.npz')

# VGG-19 convnet
paths['vgg'] = os.path.join(os.path.sep, download_path, 'vgg19.pkl')



# COCO training captions
paths['captions'] = os.path.join(os.sep, storyteller_path, 'coco_train_caps.txt')

# Biases
paths['negbias'] = os.path.join(os.sep, storyteller_path, 'caption_style.npy')
paths['posbias'] = os.path.join(os.sep, storyteller_path, 'romance_style.npy')


#----------------
#Run CPU only CNN
#----------------
#paths['pycaffe'] = '/u/yukun/Projects/caffe-run/python'
#paths['vgg_proto_caffe'] = os.path.join(os.sep, download_path, 'VGG_ILSVRC_19_layers_deploy.prototxt')
#paths['vgg_model_caffe'] = os.path.join(os.sep, download_path, 'VGG_ILSVRC_19_layers.caffemodel')

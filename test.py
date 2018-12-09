
import keras
import numpy
import imagenet_utils
import generate
from keras.preprocessing import image

def load_image(file_name):
    img = image.load_img(file_name, target_size=(224,224))
    im = image.img_to_array(img)
    im = numpy.expand_dims(im, axis=0)
    im = imagenet_utils.preprocess_input(im)
    return im


image = load_image('./images/ex1.jpg')
z = generate.load_all()
generate.story(z, image)


# 2024/03/29, AIST
# loading deep trained neural classifiers for ImageNet

import utils as utl
import keras
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.resnet import ResNet50
from keras.applications.xception import Xception
from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.nasnet import NASNetLarge, NASNetMobile


# ---------------------------------
#   load pre-trained model
# ---------------------------------

def load_model(model_name, work_dir):

    if model_name == 'inception_v3':
        model = InceptionV3(weights='imagenet')

    elif model_name == 'inception_resnet_v2':
        model = InceptionResNetV2(weights='imagenet')

    elif model_name == 'resnet50':
        model = ResNet50(weights='imagenet')

    elif model_name == 'xception':
        model = Xception(weights='imagenet')

    elif model_name == 'densenet121':
        model = DenseNet121(weights='imagenet')

    elif model_name == 'densenet169':
        model = DenseNet169(weights='imagenet')

    elif model_name == 'densenet201':
        model = DenseNet201(weights='imagenet')

    elif model_name == 'vgg16':
        model = VGG16(weights='imagenet')

    elif model_name == 'vgg19':
        model = VGG19(weights='imagenet')

    elif model_name == 'nasnetlarge':
        model = NASNetLarge(weights='imagenet')

    elif model_name == 'nasnetmobile':
        model = NASNetMobile(weights='imagenet')

    else:
        fn = work_dir + '/' + model_name
        model = utl.load_model(fn)
        print('load the trained model from\n  ', fn)

    return model


# ---------------------------------
#   image size
# ---------------------------------

def set_image_size(model_name):
    if model_name == 'inception_v3':
        img_x, img_y = 299, 299
    elif model_name == 'inception_resnet_v2':
        img_x, img_y = 299, 299
    elif model_name == 'resnet50':
        img_x, img_y = 224, 224
    elif model_name == 'xception':
        img_x, img_y = 299, 299
    elif model_name == 'densenet121' \
         or model_name == 'densenet169' \
         or model_name == 'densenet201':
        img_x, img_y = 224, 224
    elif model_name == 'vgg16' or model_name == 'vgg19':
        img_x, img_y = 224, 224
    elif model_name == 'nasnetlarge':
        img_x, img_y = 331, 331
    elif model_name == 'nasnetmobile':
        img_x, img_y = 224, 224
    else:
        img_x, img_y = 256, 256

    return img_x, img_y


# ---------------------------------
#   normalize image
# ---------------------------------

def normalize_image(model_name, in_dataset):
    if model_name == 'inception_v3':
        # in_dataset = in_dataset / 127.5 - 1.0  # Normalization to [-1, 1]
        in_dataset = keras.applications.inception_v3.preprocess_input(in_dataset)
    elif model_name == 'inception_resnet_v2':
        in_dataset = keras.applications.inception_resnet_v2.preprocess_input(in_dataset)
    elif model_name == 'resnet50':
        in_dataset = keras.applications.resnet.preprocess_input(in_dataset)
    elif model_name == 'xception':
        # in_dataset = in_dataset / 127.5 - 1.0  # Normalization to [-1, 1]
        in_dataset = keras.applications.xception.preprocess_input(in_dataset)
    elif model_name == 'densenet121' \
            or model_name == 'densenet169' \
            or model_name == 'densenet201':
        # in_dataset = in_dataset / 127.5 - 1.0  # Normalization to [-1, 1]
        in_dataset = keras.applications.densenet.preprocess_input(in_dataset)
    elif model_name == 'vgg16':
        in_dataset = keras.applications.vgg16.preprocess_input(in_dataset)
    elif model_name == 'vgg19':
        in_dataset = keras.applications.vgg19.preprocess_input(in_dataset)
    elif model_name == 'nasnetlarge' or model_name == 'nasnetmobile':
        in_dataset = keras.applications.nasnet.preprocess_input(in_dataset)
    else:
        in_dataset = in_dataset / 255.0

    return in_dataset

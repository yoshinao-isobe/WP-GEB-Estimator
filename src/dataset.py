# 2024/03/29, AIST
# setting datasets

import tensorflow as tf
import numpy as np
import glob

import model as mdl

Keras_dataset = {'mnist', 'fashion_mnist', 'cifar10'}


class Dataset(object):
    def __init__(self):

        self.in_dataset = None
        self.out_dataset = None
        self.dataset_size = 0
        self.dataset_offset = 0

    def load_dataset(
            self,
            dataset_name,
            dataset_size,
            dataset_offset,
            dataset_file='',
            dataset_fmt='',
            image_width=0,
            image_height=0,
            model_dir='',
            train_flag=False):

        print('Input dataset: ', dataset_name)

        if dataset_name in Keras_dataset:
            if dataset_name == 'mnist':
                dataset_module = tf.keras.datasets.mnist
            elif dataset_name == 'fashion_mnist':
                dataset_module = tf.keras.datasets.fashion_mnist
            else:  # if dataset_name == 'cifar10':
                dataset_module = tf.keras.datasets.cifar10

            # not safe
            # ssl._create_default_https_context = ssl._create_unverified_context

            (train_in_dataset, train_out_dataset), \
            (test_in_dataset, test_out_dataset) =\
                dataset_module.load_data()

            if train_flag:
                in_dataset = train_in_dataset
                out_dataset = train_out_dataset
            else:
                in_dataset = test_in_dataset
                out_dataset = test_out_dataset

            if type(out_dataset[0]) is np.ndarray:
                out_dataset = out_dataset.flatten()

            if dataset_size < in_dataset.shape[0] or dataset_offset > 0:
                in_dataset = in_dataset[dataset_offset: dataset_offset + dataset_size]
                out_dataset = out_dataset[dataset_offset: dataset_offset + dataset_size]

            in_shape = in_dataset.shape

            width = in_shape[1]
            height = in_shape[2]
            if len(in_shape) < 4:
                color = 1
            else:
                color = in_dataset.shape[3]

            in_dataset = in_dataset.reshape(dataset_size, width, height, color)

        else:  # for loading dataset-files (e.g. 'imagenet')

            (in_dataset, out_dataset) = load_dataset_file(
                dataset_file, dataset_size, dataset_offset,
                image_width, image_height, dataset_fmt)

        in_dataset = mdl.normalize_image(model_dir, in_dataset)

        self.in_dataset = in_dataset
        self.out_dataset = out_dataset
        self.dataset_size = in_dataset.shape[0]

        return

    def separate_dataset(self, ratio1):
        # e.g. dataset1 = validation, dataset2 = training
        in_dataset1, in_dataset2 \
            = separate_dataset(dataset=self.in_dataset, ratio=ratio1)
        out_dataset1, out_dataset2 \
            = separate_dataset(dataset=self.out_dataset, ratio=ratio1)
        return (in_dataset1, out_dataset1), (in_dataset2, out_dataset2)


def load_dataset_file(
        dataset_file, dataset_size, dataset_offset,
        image_width, image_height, dataset_fmt='tfrecord'):

    def decode_images(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
                # 'image/height': tf.io.FixedLenFeature([], tf.int64),
                # 'image/width': tf.io.FixedLenFeature([], tf.int64),
            })
        image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
        image = tf.image.resize_with_crop_or_pad(image, image_width, image_height)  # crop/augment instead

        label = tf.cast(features['image/class/label'], tf.int64) - 1  # [0-999]
        return image, label

    if dataset_fmt == 'tfrecord':
        print('dataset_file = ', dataset_file)

        tfrecords = sorted(glob.glob(dataset_file))
        dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(decode_images, num_parallel_calls=tf.data.AUTOTUNE)

        in_dataset = []
        out_dataset = []
        dataset = dataset.skip(dataset_offset)  # offset

        for images, labels in dataset.take(dataset_size):
            in_dataset.append(images)
            out_dataset.append(labels)

        in_dataset = np.array(in_dataset)
        out_dataset = np.array(out_dataset)

        return in_dataset, out_dataset

    else:
        print('The dataset format \'{}\' is not supported.'.format(dataset_fmt))
        exit(0)


def separate_dataset(dataset, ratio):
    dataset_size = dataset.shape[0]
    dataset1_size = round(dataset_size * ratio)
    dataset1 = dataset[0:dataset1_size]
    dataset2 = dataset[dataset1_size:]
    return dataset1, dataset2

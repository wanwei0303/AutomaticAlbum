# coding=gbk

import numpy as np

import os

import glob

from sklearn.utils import shuffle

import cv2


def load_train(train_path, img_size, classes):
    images = []

    labels = []

    img_names = []

    cls = []

    print("��ȡѵ��ͼƬ...")

    # classes����һ���б�<class 'list'>: ['dogs', 'cats']

    for fields in classes:

        index = classes.index(fields)

        print("Now going to read {} files (Index:{})".format(fields, index))

        # ·�������ʽ��'D:/PycharmProjects/Tensorflow_2018_test/catAndDog/training_data\\dogs\\*g'

        path = os.path.join(train_path, fields, '*g')

        # file���ݣ�'D:/PycharmProjects/Tensorflow_2018_test/catAndDog/training_data\\dogs\\dog.0.jpg'

        files = glob.glob(path)
        #ƥ�����еķ����������ļ�����������list����ʽ���ء�

        print(files)

        for fl in files:
            print(fl)

            # img_size ͼƬ�Ĵ�С���� 64����ԭʼͼƬ�����Ǵ�ͼƬ��Ҳ������СͼƬ������ת��Ϊͳһ��ʽ��

            image = cv2.imread(fl)

            # ת��Ϊ��С��<class 'tuple'>: (64, 64, 3)�������ǲ�ɫͼƬRGB��ͨ����Ϊ3.

            image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)

            # ��һ�����������ݳ��� 1/255��ת��Ϊ��0��1��֮��ķ�Χ��

            image = np.multiply(image, 1.0 / 255.0)

            images.append(image)

            label = np.zeros(len(classes))

            # è����������ǩ����[1. 0.]

            label[index] = 1.0

            labels.append(label)

            flbase = os.path.basename(fl)

            img_names.append(flbase)

            cls.append(fields)

    images = np.array(images)

    labels = np.array(labels)

    img_names = np.array(img_names)

    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images

        self._labels = labels

        self._img_names = img_names

        self._cls = cls

        self._epochs_done = 0

        self._index_in_epoch = 0

    def images(self):
        return self._images

    def labels(self):
        return self._labels

    def img_names(self):
        return self._img_names

    def cls(self):
        return self._cls

    def num_examples(self):
        return self._num_examples

    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        start = self._index_in_epoch

        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_done += 1

            start = 0

            self._index_in_epoch = batch_size

            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass

    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
#images��ͼƬ����labels��[[1 0]... [0 1]]��img_names��['dog.0.jpg' ...  'cat.9999.jpg'],cls��['dogs' ... 'cats']
    # ����sklearn.utils�� shuffle��������ɢè��ͼƬ���ݡ�

    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    # ���������2002��è��ͼƬ��validation_size����0.2�������֤��validation_size Ϊ400��

    # images��<class 'tuple'>: (2002, 64, 64, 3)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

        validation_images = images[:validation_size]

        validation_labels = labels[:validation_size]

        validation_img_names = img_names[:validation_size]

        validation_cls = cls[:validation_size]

        train_images = images[validation_size:]

        train_labels = labels[validation_size:]

        train_img_names = img_names[validation_size:]

        train_cls = cls[validation_size:]

        data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)

        data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

        return data_sets

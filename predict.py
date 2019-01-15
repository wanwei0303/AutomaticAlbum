# coding=gbk
import glob

import tensorflow as tf

import numpy as np

import os, cv2

image_size = 64

num_channels = 3
count=0

for i in range(1001,1005):
    images = []
    path = "D:\IdentifyCatAndDog\catAndDog\\training_data\cats\cat.{}.jpg".format(i)
    print(path)
    image = cv2.imread(path)
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)#��ͼƬ����64*64����ͬ��С

    images.append(image)

    images = np.array(images, dtype=np.uint8) #��ͼƬ�����ÿ��ֵ��Լ����[0,255]

    images = images.astype('float32')#��ͼƬ�����ÿ��֮����Ϊfloat����

    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()

        # step1����ṹͼ

    saver = tf.train.import_meta_graph('./GoodModel/dog-cat.ckpt-43750.meta')#���������������ṹ

        # step2����Ȩ�ز���

    saver.restore(sess, './GoodModel/dog-cat.ckpt-43750')#����TensorFlow������ÿһ��������ȡֵ

        # ��ȡĬ�ϵ�ͼ

    graph = tf.get_default_graph()

    y_pred = graph.get_tensor_by_name("y_pred:0")

    x = graph.get_tensor_by_name("x:0")

    y_true = graph.get_tensor_by_name("y_true:0")

    y_test_images = np.zeros((1, 2))

    feed_dict_testing = {x: x_batch, y_true: y_test_images}

    result = sess.run(y_pred, feed_dict_testing)

    res_label = ['dog', 'cat']

    print(res_label[result.argmax()])
    if res_label[result.argmax()]=="cat":
        count=count+1
print(count/4)










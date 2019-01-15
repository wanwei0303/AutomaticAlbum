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
    image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)#将图片处理到64*64的相同大小

    images.append(image)

    images = np.array(images, dtype=np.uint8) #将图片数组的每个值都约束到[0,255]

    images = images.astype('float32')#将图片数组的每个之都变为float类型

    images = np.multiply(images, 1.0 / 255.0)

    x_batch = images.reshape(1, image_size, image_size, num_channels)

    sess = tf.Session()

        # step1网络结构图

    saver = tf.train.import_meta_graph('./GoodModel/dog-cat.ckpt-43750.meta')#导入神经网络的网络结构

        # step2加载权重参数

    saver.restore(sess, './GoodModel/dog-cat.ckpt-43750')#导入TensorFlow程序中每一个变量的取值

        # 获取默认的图

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










#-*- coding:utf-8 -*
import tensorboard as tf
import numpy as np
from sklearn.utils import graph

import image_process, image_feature, image_model
from config import *
import configparser


def main():
    #特征处理
    image_array, label = image_feature.read_train_data()
    feature = []
    for num, image in enumerate(image_array):
        feature_vec = image_feature.feature_transfer(image)
        feature.append(feature_vec)
    print(np.array(feature).shape)
    print(np.array(label).shape)

    #训练模型
    image_model.trainModel(feature, label)




if __name__ == '__main__':
    main()



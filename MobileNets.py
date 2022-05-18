import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import os
import itertools
import shutil
import random
import matplotlib.pyplot as plt

#用GPU运行
'''physical_device=tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available:",len(physical_device))
tf.config.experimental.set_memory_growth(physical_device[0],True)'''

#下载mobilenet模型
mobilenet=tf.keras.applications.mobilenet.MobileNet()

#获取文件数据集,调整图像大小并且将其放入一个数组当中，使用扩展维度设置格式
#然后mobilenet对它进行处理，然后返回处理后的图像
def prepare_image(file):
    img_path='datasample/'
    img=image.load_img(img_path+file,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array_expanded_dims=np.expand_dims(img_array,axis=0)#扩展维度img_array在axis=0(行)轴上扩展维度
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from IPython.display import Image
Image(filename='datasample/1.png',width=300,height=200)#展示图像

preprocessed_image=prepare_image('1.png')
predictions=mobilenet.predict(preprocessed_image)#预测
results=imagenet_utils.decode_predictions(predictions)#结果，将返回1000个可能图像网络类的前五个预测
print(results)

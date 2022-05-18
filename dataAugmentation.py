import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#绘制图像的函数
def plotImage(images_arr):
    fig,axes=plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
#ImageDataGenerator类：图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；对每一个批次的训练图片，适时地进行数据增强处理
#下面就是在进行图像增强，range是范围
gen=ImageDataGenerator(rotation_range=10,width_shift_range=0.1 \
                       ,height_shift_range=0.1,shear_range=0.15,zoom_range=0.1 \
                       ,channel_shift_range=10,horizontal_flip=True)

#从狗的目录中选择随机图像,os.listdir('path')是返回指定文件夹包含的文件或者文件夹的列表
#random.choice生成服从特定的概率质量函数的随机数
chosen_image=random.choice(os.listdir('dataset/training_set/dogs'))
image_path='dataset/training_set/dogs/'+chosen_image
#扩展维度,matplotlib.pyplot.imread(path)用于读取一张图片，将图像数据变成数组array
image=np.expand_dims(plt.imread(image_path),0)
plt.imshow(image[0])#imshow()其实就是将数组的值以图片的形式展示出来
#plt.show()#展示图像
#flow()接收numpy数组和标签为参数,生成经过数据提升或标准化后的batch数据,并在一个无限循环中不断的返回batch数据
#生成一批图像，产生数据流迭代器
#并且增加参数save_to_dir并将其设置为磁盘上的有效位置以用来保存增强的数据
aug_iter=gen.flow(image,save_to_dir='dataset/',save_prefix='aug-image-',save_format='jpg')
#创建10个增强的数据样本
aug_images=[next(aug_iter)[0].astype(np.uint8) for i in range(10)]
#绘图
plotImage(aug_images)

#保存增强的数据
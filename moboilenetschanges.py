import tensorflow as tf
from tensorflow import keras
import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.metrics import categorical_crossentropy

'''os.chdir('dataset/Sign-Language-Digits-Dataset/')
#创建数据集
if os.path.isdir('train/0/') is False:
    os.makedirs('train')
    os.makedirs('test')
    os.makedirs('valid')
#f’{}‘等同于format用法，格式化字符串
    for i in range(0,10):
        shutil.move(f'{i}','train')
        os.makedirs(f'valid/{i}')
        os.makedirs(f'test/{i}')
#os.listdir用于返回指定文件夹包含的文件名
        valid_samples=random.sample(os.listdir(f'train/{i}'),30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}',f'valid/{i}')

        test_samples=random.sample(os.listdir(f'train/{i}'),5)
        for j in test_samples:
            shutil.move(f'train/{i}/{j}',f'test/{i}')'''

train_path='dataset/Sign-Language-Digits-Dataset/train'
valid_path='dataset/Sign-Language-Digits-Dataset/valid'
test_path='dataset/Sign-Language-Digits-Dataset/test'

#数据预处理，使其适合这个模型
train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=train_path,target_size=(224,224),batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=valid_path,target_size=(224,224),batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=test_path,target_size=(224,224),batch_size=10,shuffle=False)

###微调模型
mobile=tf.keras.applications.mobilenet.MobileNet()
mobile.summary()#展现结构

x=mobile.layers[-6].output#将倒数第六层输出
output=Dense(units=10,activation='softmax')(x)#对x做操作得到output

model=Model(inputs=mobile.input,outputs=output)#这个模型将包含从output到x的计算的所有网络层

#冻结所有的层，除了倒数第23层以后的
for layers in model.layers[:-23]:
    layers.trainable=False
model.summary()#查看框架

model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x=train_batches,validation_data=valid_batches,epochs=10,verbose=2)

if os.path.isfile('models/predict_model_mobilechange.h5') is False:
    model.save('models/predict_model_mobilechange.h5')



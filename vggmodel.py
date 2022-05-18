import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPool2D
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.metrics import categorical_crossentropy
import datatrain as data

#在keras下载vgg16模型
vgg16_model=tf.keras.applications.vgg16.VGG16()
#vgg16_model.summary()#查看vgg16的模型框架

#我们修改就是将最后一个输出层改为仅预测对应cat，dog的输出类
#确保正确导入模型
def count_params(model):
    non_trainable_param=np.sum([np.prod(v.get_shape().as_list())for v in model.non_trainable_weights])
    trainable_param=np.sum([np.prod(v.get_shape().as_list())for v in model.trainable_weights])
    return {'non_trainable_param': non_trainable_param,'trainable_param':trainable_param}
'''params=count_params(vgg16_model)
assert  params['non_trainable_param']==0
assert params['trainable_param']==138357544'''

#查看模型类型
#print(type(vgg16_model))#查看模型类型tensorflow.python.keras.engine.training.Model

model_vgg16=Sequential()
#复制除了最后一层的vgg16所有层
for layers in vgg16_model.layers[:-1]:#[::-1]顺序相反操作,[-1]读取倒数第一个元素,[:-1]除了最后一个取全部
    model_vgg16.add(layers)
#model_vgg16.summary()
#查看是否正确导入
params=count_params(vgg16_model)
assert  params['non_trainable_param']==0
assert params['trainable_param']==138357544
#设置每个层中不可训练,冻结所有层的可训练参数或者权重和偏差在模型当中，不改变（因为vgg就是识别猫狗的）
for layers in model_vgg16.layers:
    layers.trainable=False

model_vgg16.add(Dense(units=2,activation='softmax'))#添加输出层，只有两层输出，分别为猫狗，只训练这一层
#model_vgg16.summary()#查看框架

#编译模型
model_vgg16.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model_vgg16.fit(x=data.train_batches,validation_data=data.valid_batches,epochs=5,verbose=2)
assert  model_vgg16.history.history.get('accuracy')[-1]>0.95#若正确就继续

#保存模型
if os.path.isfile('models/predict_model_vgg16.h5') is False:
    model_vgg16.save('models/predict_model_vgg16.h5')




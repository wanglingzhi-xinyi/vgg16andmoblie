from tensorflow.keras.models import load_model
import datatrain as data
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools

#调用模型
modelvgg=load_model('models/predict_model_vgg16.h5')
#预测模型
predictions=modelvgg.predict(x=data.test_batches,verbose=0)
data.test_batches.classes#查看参数标签
cm=confusion_matrix(y_true=data.test_batches.classes,y_pred=np.argmax(predictions,axis=-1))
#绘制混淆矩阵
def plot_confusion_matrix(cm,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]#[:, np.newaxis]增加一个维度
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh=cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,cm[i,j],horizontalalignment="center",
                 color="white" if cm[i,j]>thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
data.test_batches.class_indices#查看标签对应，一般是猫的下标是0，狗是1
cm_plot_image=['cat','dog']
plot_confusion_matrix(cm=cm,classes=cm_plot_image)


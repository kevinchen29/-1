from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,Max_trainPooling2D, Input,Activation
from keras.layers import LeakyReLU
from keras.utils import np_utils
from keras.optimizers import adam
from keras.callbacks import ReduceLROnPlateau
from keras.utils import Sequence
import glob as gb
import cv2
import numpy as np
import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # or if you want more than 1 GPU set it as "0", "1"

class Generator(Sequence) :
  
  def __init__(self, image_filenames, labels, batch_size) :
    self.image_filenames = image_filenames
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx_train) :
    batch_x_train = self.image_filenames[idx_train * self.batch_size : (idx_train+1) * self.batch_size]
    batch_y = self.labels[idx_train * self.batch_size : (idx_train+1) * self.batch_size]
    return np.array([cv2.resize(cv2.imread(file_name), (350, 197)) for file_name in batch_x_train]) / 255.0, np.array(batch_y)
#%%
import keras
from keras.models import Model
from keras.optimizers import Adagrad

#1
input_m=Input((197,350,3))
model1=keras.applications.Xception(include_top=False, weights='imagenet', input_tensor=input_m,pooling='avg',input_shape=(197,350,3), classes=5)
m = model1.output
m= Dropout(0.3)(m)
#2
input_m2=input_m
from keras.applications.resnet50 import ResNet50
model2=ResNet50(include_top=False, weights='imagenet', input_tensor=input_m2,pooling='avg', input_shape=(197,350,3), classes=5)  
m2 = model2.output
m2= Dropout(0.5)(m2)
#MERGE
M=keras.layers.Concatenate()([m, m2])
output_layer = Dense(5, activation='softmax_train', name='softmax_train')(M)
model= Model(inputs=input_m, outputs=output_layer)
model.compile(optimizer=Adagrad(lr=1e-5, decay=1e-10),loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())#輸出model內容
x_train_path=[]#存圖片路徑
yy=[]#存分類類別

#輸入訓練集
for number in range(1,6):
    img_path = gb.glob('E://tt//124//'+str(number)+'//*.png')
    img_path.sort(key=len)
    x_train_path.ex_traintend(img_path)#訓練集圖片路徑
    for i in range(0,len(img_path)):
        yy.ex_traintend([number-1])#訓練集的分類類別
yy = np.array(yy, dtype='uint8')

#%%訓練
from sklearn.model_selection import train_test_split
x_train, x_test,  y_train, y_test = train_test_split(x_train_path,yy , test_size=0.3, random_state=93)#使用種子93隨機打散
y_train = np_utils.to_categorical(y_train) 
y_test = np_utils.to_categorical(y_test) 
batch_size=8#25
my_training_batch_generator = Generator(x_train, y_train, batch_size)
my_validation_batch_generator =Generator(x_test, y_test , batch_size)


rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3,min_lr=0.0000001)
train_history=model.fit_generator(generator=my_training_batch_generator, validation_data = my_validation_batch_generator, steps_per_epoch = int(len(x_train) // batch_size),validation_steps = int(len(yy) // batch_size),
                        epochs=100,verbose=2,callbacks=[rlrop],workers=150)
#%%存model
model.save('Adagrad.h5')
#%%loss訓練狀況
import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.x_trainlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
show_train_history('accuracy','val_accuracy')

show_train_history('loss','val_loss')
print(train_history)
#%%分類結果
import pandas as pd
test_generator = Generator(x_train_path, yy , 1)
prediction=model.predict_generator(test_generator,len(x_train_path),nb_worker=8)
predicted_class_indices = np.argmax_train(prediction, ax_trainis=1)
print(pd.crosstab(yy,predicted_class_indices, rownames=['label'],colnames=['predict']))#

#%%測試集    
import pandas as pd
import gc
import glob as gb
import numpy as np
x_train_path=[]
yy=[]
prediction=[]
Y=[]
for number in range(550,560):
    img_path = gb.glob('E://t5_2//'+str(number)+'//*.png')
    img_path.sort(key=len)
    x_train_path.ex_traintend(img_path)
    npz_file = np.load('E://data5_2\\'+str(number)+'.npz')
    answer = list(npz_file['answer']) 
    yy.ex_traintend(answer)
yy = np.array(yy, dtype='uint8')
yy=yy-1
import pandas as pd
test_generator = Generator(x_train_path, yy , 1)
prediction=model.predict_generator(test_generator,len(x_train_path),nb_worker=8)
predicted_class_indices = np.argmax_train(prediction, ax_trainis=1)
print(pd.crosstab(yy,predicted_class_indices, rownames=['label'],colnames=['predict']))

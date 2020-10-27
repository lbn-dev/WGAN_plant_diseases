#----------------------------train CNN without classic data augmentation-------------------------------------------
from __future__ import print_function, division
import sklearn as sk
from keras import backend as K
import matplotlib.gridspec as gridspec
from sklearn.metrics import classification_report
from keras.layers import multiply
from keras.layers import Embedding
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.convolutional import UpSampling2D
from subprocess import check_output
import numpy as np
import os
import imageio
import pickle
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import ZeroPadding2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.transform import resize as imresize
from tqdm import tqdm
import pandas as pd
from keras import regularizers
from functools import partial
from keras.layers.merge import _Merge
class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((100, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


BATCH_SIZE = 100
# EPOCHS = 300
RANDOM_STATE = 11
MAX=0
ep=0
CLASS = {"c_"+str(i):i for i in range(38)}


INV_CLASS = {i:"c_"+str(i) for i in range(38)}



# Dense layers set
def dense_set(inp_layer, n, activation, drop_rate=0.):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act


# Conv. layers set
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3), strides=(1, 1), zp_flag=False):
    if zp_flag:
        zp = ZeroPadding2D((1, 1))(feature_batch)
    else:
        zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = LeakyReLU(1 / 10)(bn)
    return act

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
# simple model
def get_model():
    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(128,128,3), kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(38))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy',metrics=['accuracy',f1_m,precision_m, recall_m])
    return model



def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]


def train_model(img, target,a,b,c,d):
    #callbacks = get_callbacks(filepath='D:/2ndPlant/kaggle_seedling_classification/model_weight_SGD.hdf5', patience=6)

    gmodel = get_model()
    if d!=0:
     gmodel.load_weights(filepath='./wgan_weights.h5')

    gen = ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )

    history = gmodel.fit(img, target, batch_size=BATCH_SIZE,epochs=c, verbose=1, shuffle=True, validation_data=(a, to_categorical(b)))
    model_json = gmodel.to_json()
    with open("./wgan.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    gmodel.save_weights("./wgan_weights.h5")
    # list all data in history
    print(history.history.keys())

    global MAX,ep,result_max
    if np.max(history.history['val_acc'])>MAX:
        MAX=np.max(history.history['val_acc'])
        y_pred_max = (gmodel.predict(a)).argmax(axis=-1)
        result_max = sk.metrics.confusion_matrix(b,y_pred_max)
        ep=d


    print((MAX,ep))
    dataframe = pd.DataFrame({'acc': history.history['acc'], 'val_acc': history.history['val_acc']})
    filename="wgan_record"+str(d)+".csv"
    dataframe.to_csv( filename, index=False, sep=',')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Prediction Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



# Resize all image to 128*128
def img_reshape(img):
    img = imresize(img, (128, 128, 3))
    return img


# get image tag
def img_label(path):
    return str(str(path.split('/')[-1]))


# get plant class on image
def img_class(path):
    return str(path.split('/')[-2])


# fill train and test dict
def fill_dict(paths, some_dict):
    text = ''
    if 'train' in paths[0]:
        text = 'Start fill train_dict'
    elif 'test' in paths[0]:
        text = 'Start fill test_dict'

    for p in tqdm(paths, ascii=True, ncols=85, desc=text):
        img = imageio.imread(p)
        img = img_reshape(img)
        some_dict['image'].append(img)
        some_dict['label'].append(img_label(p))
        if 'train' in paths[0]:
            some_dict['class'].append(img_class(p))

    return some_dict


# read image from dir. and fill train and test dict
def reader():
    file_ext = []
    train_path = []
    test_path = []

    for root, dirs, files in os.walk('./'):
        if dirs != []:
            print('Root:\n' + str(root))
            print('Dirs:\n' + str(dirs))
        else:
            for f in files:
                ext = os.path.splitext(str(f))[1][1:]

                if ext not in file_ext:
                    file_ext.append(ext)

                if 'train' in root:
                    path = os.path.join(root, f)
                    train_path.append(path)
                elif 'test' in root:
                    path = os.path.join(root, f)
                    test_path.append(path)
    train_dict = {
        'image': [],
        'label': [],
        'class': []
    }
    test_dict = {
        'image': [],
        'label': []
    }
    train_dict = fill_dict(train_path, train_dict)
    return train_dict, test_dict



with open('./dataset', 'rb') as f:
    X_train, X_valid, Y_train, Y_valid=pickle.load(f)
print("original train size",X_train.shape)

train_model(X_train, to_categorical(y_train),X_valid,Y_valid,700)



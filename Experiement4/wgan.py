#----------------------------------train wgan-gp-lsr----------------------------------------------------------------
from __future__ import print_function, division
import matplotlib.gridspec as gridspec
from keras.utils import to_categorical
import sklearn as sk
from keras import backend as K
from sklearn.metrics import classification_report
from keras.layers import multiply
from keras.layers import Embedding
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.convolutional import UpSampling2D
from subprocess import check_output
import numpy as np
import argparse
import os
import imageio
import pickle
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.merge import _Merge
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
from functools import partial

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((100, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class DCGAN():
    def __init__(self):
        # Input shape
        global img_size
        global noise
        global var1
        self.epi = var1
        self.img_rows = img_size
        self.img_cols = img_size
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = noise
        self.num_classes = 38
        self.X_train, self.x_valid, self.y_train, self.y_valid = reader();
        def custom_objective(y_true, y_pred):
            epi = self.epi

            # out = -epi+(1-epi)*K.sparse_categorical_crossentropy(y_true, y_pred)
            out = -epi * (K.mean(K.log(y_pred + 0.00000001))) + (1 - epi) * K.sparse_categorical_crossentropy(y_true,
                                                                                                              y_pred)
            return out
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        self.generator = self.build_generator()
        self.critic = self.build_critic()
        self.generator.trainable = False
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc,label])

        fake,fake_label = self.critic(fake_img)
        valid, valid_label = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated, inter_label = self.critic(interpolated_img)
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, label],
                            outputs=[valid, fake, validity_interpolated,valid_label,fake_label])
        self.critic_model.compile(loss=[self.wasserstein_loss,self.wasserstein_loss, partial_gp_loss,custom_objective,custom_objective],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10, 2, 2 ])
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        label_zen=Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen,label_zen])
        # Discriminator determines validity
        valid,valid_label = self.critic(img)
        # Defines generator model
        self.generator_model = Model([z_gen,label_zen], [valid,valid_label])
        self.generator_model.compile(loss=[self.wasserstein_loss,custom_objective], optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    def build_critic(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.summary()
        img = Input(shape=self.img_shape)
        features = model(img)
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)
        return Model(img, [validity, label])

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)
        return Model([noise, label], img)

    def build_discriminator(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.summary()
        img = Input(shape=self.img_shape)
        features = model(img)
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes + 1, activation="softmax")(features)
        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Load the dataset
        X_train, x_valid, y_train, y_valid = self.X_train, self.x_valid, self.y_train, self.y_valid;
        y_train = y_train.reshape(-1, 1)
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))
        d_list=[]
        g_list=[]
        for epoch in range(epochs):
            for _ in range(self.n_critic):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                img_labels = y_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                sampled_labels =img_labels
                fake_labels = self.num_classes * np.ones(img_labels.shape)
                d_loss = self.critic_model.train_on_batch([imgs, noise, sampled_labels],
                                                      [valid, fake, dummy, sampled_labels,fake_labels])
            g_loss = self.generator_model.train_on_batch([noise,sampled_labels], [valid,sampled_labels])
            d_list.append(d_loss)
            g_list.append(g_loss)
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
                epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)
                pd.DataFrame(d_list).to_csv("d_list.csv")
                pd.DataFrame(g_list).to_csv("g_list.csv")

    def sample_images(self, epoch):
        r, c = 10, self.num_classes
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig = plt.figure()
        gs = gridspec.GridSpec(6, 7,
                               wspace=0.0, hspace=0.0)
        cnt = 0
        for i in range(6):
            for j in range(7):
                im=gen_imgs[cnt, :, :, 0:3]
                ax = plt.subplot(gs[i, j])
                ax.imshow(im)
                ax.axis('off')
                cnt += 1
        fig.savefig("/work/smryan/luningbi/images_dcgan/%d.png" % epoch)
        plt.close()

    def save_model(self):
        def save(model, model_name):
            model_path = "/work/smryan/luningbi/saved_model_wgan/%s.json" % model_name
            weights_path = "/work/smryan/luningbi/saved_model_wgan/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.critic, "discriminator")



BATCH_SIZE = 100
EPOCHS = 300
RANDOM_STATE = 11

CLASS = {"c_" + str(i): i for i in range(38)}

INV_CLASS = {i: "c_" + str(i) for i in range(38)}


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


# simple model
def get_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(38, activation='softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [lr_reduce, msave]


def train_model(img, target, a, b, c, d):
    # callbacks = get_callbacks(filepath='D:/2ndPlant/kaggle_seedling_classification/model_weight_SGD.hdf5', patience=6)

    gmodel = get_model()
    if d != 0:
        gmodel.load_weights(filepath='D:/2ndPlant/REVIEW/reg_new_model_weights.h5')

    gen = ImageDataGenerator(
        rotation_range=360.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True
    )
    history = gmodel.fit_generator(gen.flow(img, target, batch_size=BATCH_SIZE),
                                   steps_per_epoch=len(img) / BATCH_SIZE,
                                   epochs=c,
                                   verbose=1,
                                   shuffle=True,
                                   validation_data=(a, to_categorical(b)))
    # history = gmodel.fit(img, target, batch_size=BATCH_SIZE,
    #                      epochs=c, verbose=1, shuffle=True,
    #                      validation_data=(a, to_categorical(b)))
    model_json = gmodel.to_json()
    with open("/REVIEW/saved_model_wgan/reg_new_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    gmodel.save_weights("/REVIEW/saved_model_wgan/reg_new_model_weights.h5")
    # list all data in history
    print(history.history.keys())

    global MAX, ep, result_max
    if np.max(history.history['val_acc']) > MAX:
        MAX = np.max(history.history['val_acc'])
        y_pred_max = (gmodel.predict(a)).argmax(axis=-1)
        result_max = sk.metrics.confusion_matrix(b, y_pred_max)
        ep = d

    print((MAX, ep))
    dataframe = pd.DataFrame({'acc': history.history['acc'], 'val_acc': history.history['val_acc']})
    filename = "record" + str(d) + ".csv"
    dataframe.to_csv(filename, index=False, sep=',')
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


# Resize all image to 51x51
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

    for root, dirs, files in os.walk('/work/smryan/luningbi/'):
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
    # test_dict = fill_dict(test_path, test_dict)

    X_train = np.array(train_dict['image'])
    y_train = np.array([CLASS[l] for l in train_dict['class']])
    X_train = (X_train.astype(np.float32) - 0.5) * 2
    y_train = y_train.reshape(-1, 1)
    return X_train, x_valid, 0,0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--epochs',
        help='epochs',
        type=int,
        required=True
    )
    parser.add_argument(
        '--batch',
        help='batchsize',
        type=int,
        required=True
    )
    parser.add_argument(
        '--interval',
        help='interval',
        type=int,
        required=True
    )
    parser.add_argument(
        '--img',
        help='img-size',
        type=int,
        required=True
    )
    parser.add_argument(
        '--noise',
        help='latent_dim',
        type=int,
        required=True
    )
    parser.add_argument(
        '--var1',
        help='epi0.22',
        type=float,
        required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    CLASS = {"c_" + str(i): i for i in range(38)}
    RANDOM_STATE = 11
    img_size = arguments['img']
    noise = arguments['noise']
    var1 = arguments['var1']
    acgan = DCGAN()

    acgan.train(epochs=arguments['epochs'], batch_size=arguments['batch'], sample_interval=arguments['interval'])

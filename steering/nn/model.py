import os
import logging
import json
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from steering.data import Data, TrainData


class Model:
    __logger = logging.getLogger(__name__)

    def __init__(self, input_img_shape=None, dropout_prob=None):
        self.debug = False
        self.__model = self.__create_model(input_img_shape=input_img_shape, dropout_prob=dropout_prob)

    def test(data_path, input_img_shape=Data.NN_INPUT_IMAGE_SHAPE, dropout_prob=0.5):
        m = Model.for_training(training_data=TrainData(data_path), input_img_shape=input_img_shape, dropout_prob=dropout_prob)

    def for_training(training_data, input_img_shape=Data.NN_INPUT_IMAGE_SHAPE, dropout_prob=0.5, checkpoint_path="./checkpoint"):
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        m = Model(input_img_shape, dropout_prob)

        checkpoint_path = "%s/weights.{epoch:02d}-{val_loss:.2f}.hdf5" % (checkpoint_path)
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False, save_weights_only=False, mode='auto')
        m.set_fit_generator(training_data, checkpoint)

        return m

    def for_predicting(self, saved_model_path, input_img_shape=Data.NN_INPUT_IMAGE_SHAPE):
        m = Model(input_img_shape, dropout_prob=0.0)
        m.__model.load_weights(saved_model_path)
        m.__model.compile(loss='mse', optimizer='Adam')
        return m

    def save(self, saveto_model_path):
        if not os.path.exists(saveto_model_path):
            os.makedirs(saveto_model_path)

        # The model architecture.
        with open("%s/model.json" % (saveto_model_path), 'w') as o:
            json.dump(self.__model.to_json(), o)

        # The model weights.
        self.__model.save_weights(filepath="%s/model.h5" % (saveto_model_path), overwrite=True)

    def set_fit_generator(self, training_data, checkpoint):
        logging.info("Set fit generator")
        # TODO
        # self.__model.fit_generator()
        pass

    def __create_model(self, input_img_shape, dropout_prob):
        """This mode is based on NVIDIAs research paper - End to End Learning for Self-Driving Cars
           http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"""

        model = Sequential()
        model.add(Convolution2D(3, 1, 1, init=self.__layer_init, subsample=(1, 1), border_mode='same', name='conv0_1x1', input_shape=(66, 200, 3)))
        model.add(ELU())
        model.add(Convolution2D(24, 5, 5, init=self.__layer_init, subsample=(2, 2), border_mode='same', name='conv1_2x2'))
        model.add(ELU())
        model.add(Convolution2D(36, 5, 5, init=self.__layer_init, subsample=(2, 2), border_mode='same', name='conv2_2x2'))
        model.add(ELU())
        model.add(Convolution2D(48, 5, 5, init=self.__layer_init, subsample=(2, 2), border_mode='same', name='conv3_2x2'))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, init=self.__layer_init, subsample=(1, 1), border_mode='same', name='conv4_1x1'))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, init=self.__layer_init, subsample=(1, 1), border_mode='same', name='conv5_1x1'))
        model.add(ELU())
        model.add(Flatten())
        model.add(Dense(1164, init=self.__layer_init, name="dense0_1164"))
        model.add(ELU())
        model.add(Dropout(dropout_prob))
        model.add(Dense(100, init=self.__layer_init, name="dense1_100"))
        model.add(ELU())
        model.add(Dropout(dropout_prob))
        model.add(Dense(50, init=self.__layer_init, name="dense2_50"))
        model.add(ELU())
        model.add(Dropout(dropout_prob))
        model.add(Dense(10, init=self.__layer_init, name="dense3_10"))
        model.add(ELU())
        model.add(Dense(1, init=self.__layer_init, name="dense4_1"))

        model.compile(optimizer="adam", loss="mse")
        logging.info("Model created")

        return model

    def __layer_init(self, shape, name):
        logging.debug("Init model layer %s" % (name))
        return tf.truncated_normal(shape, stddev=1e-2)
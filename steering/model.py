import os
import logging
import json
import tensorflow as tf
from keras.models import Sequential, Model as KerasModel
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, ELU, Activation, Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam, Adagrad
from steering.data import Data
from keras import backend as K
from keras.applications import VGG16, ResNet50
import numpy as np
import multiprocessing

class Model:
    __logger = logging.getLogger(__name__)

    def __init__(self, config, dropout_prob=0.0):
        self.__config = config
        self.__optimizer = config.get("training", "optimizer")
        self.__input_img_shape = Data.INPUT_IMAGE_SHAPE
        self.__dropout_prob = dropout_prob
        self.__model = self.__create_model()

    def selfdiag(self):
        """This function is used for testing and debugging purposes"""

        self.restore("./saved/model.h5")

    def for_training(config):
        """Instantiates the model for running training epochs"""

        dropout_prob = float(config.get("training", "dropout_prob"))
        return Model(config, dropout_prob)

    def for_predicting(config):
        """Instantiates the model for running predictions"""

        m = Model(config, dropout_prob=0.0)
        m.__model.compile("adam", "mse")
        return m

    def predict(self, image):
        """Predicting a steering angle for a given single image"""

        image = Data.image_pre_process(image)
        image = np.expand_dims(image, axis=0)
        return self.__model.predict(image, batch_size=1, verbose=0)[0][0]

    def train(self, training_data, nb_epoch, checkpoint_path="./checkpoint"):
        """Train and save (checkpoint) weights along the way at regular intervals.
           ModelCheckPoint mode is set to 'auto' with a save_best_only value set to True.
           This will save only weights that result in a min monitored value (val_loss)"""

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint_path = "%s/weights.{epoch:02d}-{val_loss:.4f}.h5" % (checkpoint_path)
        checkpoint = ModelCheckpoint(checkpoint_path, verbose=0, save_best_only=False, save_weights_only=False, monitor='val_loss', mode='auto')
        self.fit_generator(training_data, nb_epoch, checkpoint)

    def restore(self, saved_model_path):
        """loads model weights from the given file path"""

        if not os.path.exists(saved_model_path):
            logging.warning("saved weights (%s) not found" % (saved_model_path))
        else:
            logging.info("Loading %s" % (saved_model_path))
            self.__model.load_weights(saved_model_path)
            logging.info("[Done] Loading %s" % (saved_model_path))

    def save(self, saveto_model_path):
        """saves the model as json and its trained weights (h5) to disk"""

        logging.info("saving mode & weights to %s" % (saveto_model_path))
        if not os.path.exists(saveto_model_path):
            os.makedirs(saveto_model_path)

        # The model architecture.
        with open("%s/model.json" % (saveto_model_path), 'w') as o:
            json.dump(self.__model.to_json(), o)

        # The model weights.
        self.__model.save_weights(filepath="%s/model.h5" % (saveto_model_path), overwrite=True)
        logging.info("[Done] saving mode & weights to %s" % (saveto_model_path))

    def fit_generator(self, training_data, nb_epoch, checkpoint):
        """Fits the model on data generated batch-by-batch by a Python generator. The generator is run in parallel to the model, for efficiency.
           Parallism is determined by the nb_worker parameter which is set to the number of CPU cores avaialble on the training machine"""

        logging.info("Get validtion data ...")
        valid_images, valid_angles = training_data.valid_data()

        workers = multiprocessing.cpu_count()
        logging.info("Set fit generator. workers = %s optimizer: %s" % (workers, self.__optimizer))

        self.__model.compile(loss="mse", optimizer=self.__optimizer) # , metrics=['accuracy'])
        self.__model.fit_generator(training_data.fit_generator(), validation_data=(valid_images, valid_angles), samples_per_epoch=training_data.samples_per_epoch(), nb_worker=workers, nb_epoch=nb_epoch, verbose=1, callbacks=[checkpoint], pickle_safe=True)

    def __create_model(self):
        """Create a deep neural network model based on the type specified in the settings.ini file. This version only implements the
           nvidia model and future impls may include others (eg: VGG16, ResNet50 etc."""

        model = self.__config.get("model", "type")

        if model == "nvidia":
            return self.__create_nvidia_model()
        elif model == "vgg16":
            return self.__create_vgg16_model()
        elif model == "resnet50":
            return self.__create_resnet50_model()
        else:
            raise "unknown model %s" % (model)

    def __create_nvidia_model(self):
        """This mode is based on NVIDIAs research paper - End to End Learning for Self-Driving Cars
           http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf"""

        model = Sequential()
        #model.add(Convolution2D(3, 1, 1, init=self.__layer_init, subsample=(1, 1), border_mode='valid', name='conv0_1x1', input_shape=self.__input_img_shape, activation=activation))
        model.add(Convolution2D(24, 5, 5, init=self.__layer_init(), subsample=(2, 2), border_mode='valid', name='conv1_5x5', input_shape=self.__input_img_shape))
        model.add(ELU())
        model.add(Convolution2D(36, 5, 5, init=self.__layer_init(), subsample=(2, 2), border_mode='valid', name='conv2_5x5'))
        model.add(ELU())
        model.add(Convolution2D(48, 5, 5, init=self.__layer_init(), subsample=(2, 2), border_mode='valid', name='conv3_5x5'))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, init=self.__layer_init(), subsample=(1, 1), border_mode='valid', name='conv4_3x3'))
        model.add(ELU())
        model.add(Convolution2D(64, 3, 3, init=self.__layer_init(), subsample=(1, 1), border_mode='valid', name='conv5_3x3'))
        model.add(ELU())
        model.add(Flatten())
        model.add(Dense(1164, init=self.__layer_init(), name="dense0_1164"))
        model.add(ELU())
        model.add(Dropout(self.__dropout_prob))
        model.add(Dense(100, init=self.__layer_init(), name="dense1_100"))
        model.add(ELU())
        model.add(Dropout(self.__dropout_prob))
        model.add(Dense(50, init=self.__layer_init(), name="dense2_50"))
        model.add(ELU())
        model.add(Dropout(self.__dropout_prob))
        model.add(Dense(10, init=self.__layer_init(), name="dense3_10"))
        model.add(ELU())
        model.add(Dense(1, init=self.__layer_init(), name="dense4_1"))

        assert (model.get_layer(name="conv1_5x5").input_shape == (None, Data.INPUT_IMAGE_HEIGHT, Data.INPUT_IMAGE_WIDTH, Data.INPUT_IMAGE_COLCHAN)), "The input shape is: %s" % str(model.get_layer(name="hidden1").input_shape)
        assert (model.get_layer(name="dense4_1").output_shape == (None, 1)), "The input shape is: %s" % str(model.get_layer(name="hidden1").input_shape)

        logging.info("Model created (self.__dropout_prob: %s)" % (self.__dropout_prob))
        model.summary()
        return model

    def __create_vgg16_model(self):
        """VGG16 model for future experiments. Not implemented."""
        raise "vgg16_model not implemented"

    def __create_resnet50_model(self):
        """ResNet50 model for future experiments. Not implemented."""
        raise "ResNet50_model not implemented"

    def __layer_init(self):
        """Init keras layers (https://keras.io/initializations/)"""
        return "he_normal"  # Gaussian initialization scaled by fan_in (He et al., 2014)
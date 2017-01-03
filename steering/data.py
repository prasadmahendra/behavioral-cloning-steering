import os
import csv
import logging
import cv2
import numpy as np
import math
import random
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import imagenet_utils

class Data():
    INPUT_IMAGE_WIDTH = 80
    INPUT_IMAGE_HEIGHT = 40
    INPUT_IMAGE_COLCHAN = 3
    INPUT_IMAGE_SHAPE = (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH, INPUT_IMAGE_COLCHAN)

    LEFT_CAMERA_IMAGE_IDX = 1
    CENTER_CAMERA_IMAGE_IDX = 0
    RIGHT_CAMERA_IMAGE_IDX = 2
    STEERING_ANGLE_IDX = 3
    IMAGE_PATH_PREFIX = "data/"
    CAMERA_POS = ["center", "left", "right"]
    CAMERA_TO_STEERING_ANGLE_ADJ = [0, 0.25, -0.25]

    def __init__(self):
        self.debug = False

    def image_pre_process(image):
        """Pre-processes images before they are fed in to the either the prediction or training model networks"""

        image_out = Data.__region_of_interest(image)

        image_out = Data.__pixel_normalize(image_out)

        image_out = Data.__pixel_mean_subtraction(image_out)

        image_out = Data.__resize_image(image_out, Data.INPUT_IMAGE_HEIGHT, Data.INPUT_IMAGE_WIDTH)

        return image_out

    def load_image_from_file(file_path):
        image = np.asarray(load_img(file_path, target_size=(Data.INPUT_IMAGE_HEIGHT, Data.INPUT_IMAGE_WIDTH)), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        return np.asarray(image, dtype=np.uint8)

    def __resize_image(image, height, width):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def __region_of_interest(image):
        """Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black."""

        bottom_left_x = 0
        bottom_left_y = image.shape[0] - Data.__engine_compartment_pixels()
        top_left_x = 0
        top_left_y = image.shape[0] / 3
        top_right_x = image.shape[1]
        top_right_y = top_left_y
        bottom_right_x = image.shape[1]
        bottom_right_y = image.shape[0] - Data.__engine_compartment_pixels()


        mask_vertices = np.array([[(bottom_left_x, bottom_left_y),
                                   (top_left_x, top_left_y),
                                   (top_right_x, top_right_y),
                                   (bottom_right_x, bottom_right_y)]], dtype=np.int32)

        # defining a blank mask to start with
        mask = np.zeros_like(image)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(image.shape) > 2:
            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, mask_vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def __pixel_normalize(image):
        image = cv2.normalize(image, image, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #assert (math.isclose(np.min(image), 0.0, abs_tol=0.0001) and math.isclose(np.max(image), 1.0, abs_tol=0.0001)), "__normalize failed. The range of the input data is: %.10f to %.10f" % (np.min(image), np.max(image))
        return image

    def __pixel_mean_subtraction(image):
        image = image.astype(dtype='float32')
        image -= np.mean(image, dtype='float32', axis=0)
        #assert (round(np.mean(image)) == 0), "__mean_subtraction error. The mean of the input data is: %f" % np.mean(image)
        return image

    def __engine_compartment_pixels():
        return 25

    def __hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap, y_min, line_fit=False):
        """
        `img` should be the output of a Canny transform.

        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
        self.__draw_lines(line_img, lines, y_min, line_fit)
        return line_img

    def __draw_lines(self, img, lines, y_min, line_fit=False, color=[255, 0, 0], thickness=10):
        if lines is None:
            return img

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def __weighted_img(self, img, initial_img, α=0.8, β=1., λ=0.):
        return cv2.addWeighted(initial_img, α, img, β, λ)

    def display_image(title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class TrainData(Data):
    TRAIN_BATCH_SIZE = 64

    __logger = logging.getLogger(__name__)

    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise "missing training_data file!"

        self.debug = False
        self.__csv_data_file = data_file
        self.__csv_train_data = None
        self.__csv_valid_data = None
        super(TrainData, self).__init__()
        self.__load_data()

    def test(self):
        logging.info("running selfdiag ...")

        assert (self.train_data() != None)

        line_data = self.train_data()[51]

        camera, image_file, steering_angle = self.__select_camera_at_random(line_data, select_camera_at_random=True)
        assert(camera != None)
        assert(image_file != None)
        assert(steering_angle != None)
        assert(type(steering_angle) is float)

        logging.info("loading camera: %s image: %s steering_angle: %s ..." % (camera, image_file, steering_angle))

        camera_image = Data.load_image_from_file(image_file)
        assert (camera_image.shape[0] == self.INPUT_IMAGE_HEIGHT)
        assert (camera_image.shape[1] == self.INPUT_IMAGE_WIDTH)
        assert (camera_image.shape[2] == self.INPUT_IMAGE_COLCHAN)
        assert(camera_image.dtype == np.uint8)

        camera, camera_image, steering_angle = self.__select_image_for_training(line_data)
        logging.info("loading camera: %s image: %s steering_angle: %s ..." % (camera, image_file, steering_angle))

        assert (camera_image.shape[0] == self.INPUT_IMAGE_HEIGHT)
        assert (camera_image.shape[1] == self.INPUT_IMAGE_WIDTH)
        assert (camera_image.shape[2] == self.INPUT_IMAGE_COLCHAN)
        assert (camera_image.dtype == np.float32)
        assert (type(steering_angle) is float)

    def train_data(self):
        return self.__csv_train_data

    def valid_data(self):
        X = []
        y = []
        for line in self.__csv_valid_data:
            camera, image, steering_angle = self.__select_image_for_training(line, generate_new_images=False, select_camera_at_random=False)
            X.append(image)
            y.append(steering_angle)

        X = np.array(X)
        y = np.array(y)

        assert (X.shape[0] == y.shape[0]), "The number of images is not equal to the number of labels."
        assert (X.shape[1:] == self.INPUT_IMAGE_SHAPE), "Unexpected input shape %s" % (X.shape[1:])

        return X, y

    def __select_image_for_training(self, line_data, generate_new_images=False, select_camera_at_random=False):
        camera, image_file, steering_angle = self.__select_camera_at_random(line_data, select_camera_at_random)
        image = Data.load_image_from_file(image_file)
        image = Data.image_pre_process(image)

        if generate_new_images:
            # randomly flip along x/y axis ...
            image, steering_angle = self.__flip_image_lr_at_rand(image, steering_angle)

            # adj brightness ...
            image = self.__adj_brightness_at_rand(image)

            # skew for training recovery ...
            image, steering_angle = self.__x_y_skew_at_rand(image, steering_angle)

        return camera, np.array(image), steering_angle

    def __x_y_skew_at_rand(self, image, steering_angle, both_axis=False):
        x_range = image.shape[1] / 3
        y_range = image.shape[0] / 3
        angle_multiplier = 0.5

        x_tran = x_range * np.random.uniform() - (x_range / 2)  # + or -
        new_steering_angle = steering_angle + (x_tran / x_range) * angle_multiplier

        y_tran = 0
        if both_axis:
            y_tran = y_range * np.random.uniform() - (y_range / 2)  # + or -

        translation_mat = np.float32([[1, 0, x_tran], [0, 1, y_tran]])
        new_image = cv2.warpAffine(image, translation_mat, (image.shape[1], image.shape[0]))

        logging.debug("x tran: %s y tran: %s steering adj: %s -> %s" % (x_tran, y_tran, steering_angle, new_steering_angle))
        return new_image, new_steering_angle

    def __adj_brightness_at_rand(self, image):
        adj = np.random.randint(2)
        if adj == 1:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            random_bright = .25 + np.random.uniform()
            image[:, :, 2] = image[:, :, 2] * random_bright

            return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            return image

    def __flip_image_lr_at_rand(self, image, steering_angle):
        flip = np.random.randint(2)
        if flip == 1:
            return np.fliplr(image), -steering_angle
        else:
            return image, steering_angle

    def __select_camera_at_random(self, line_data, select_camera_at_random):
        selection_idx = 0
        if select_camera_at_random:
            selection_idx = np.random.randint(3)

        camera_image = line_data[selection_idx]
        steering_angle = float(line_data[self.STEERING_ANGLE_IDX].strip()) + self.CAMERA_TO_STEERING_ANGLE_ADJ[selection_idx]

        return self.CAMERA_POS[selection_idx], str(self.IMAGE_PATH_PREFIX + camera_image.strip()), steering_angle

    def __load_data(self):
        # csv format: center,left,right,steering,throttle,brake,speed
        with open(self.__csv_data_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            csv_data = list(csv_reader)[1:]

            # train vs. validation data 80:20 split
            random.shuffle(csv_data)

            row_count = len(csv_data)
            self.__csv_train_data = csv_data[:int(row_count * 0.8)]
            self.__csv_valid_data = csv_data[-int(row_count * 0.2):]

            self.__logger.info("loaded training data points: %s" % (len(self.__csv_train_data)))
            self.__logger.info("loaded validation data points: %s" % (len(self.__csv_valid_data)))

    def fit_generator(self):
        curr_iter_idx_start = 0
        curr_iter_idx_end = self.TRAIN_BATCH_SIZE - 1
        idx_max = len(self.__csv_train_data) - 1

        if curr_iter_idx_end > idx_max:
            curr_iter_idx_end = idx_max

        while 1:
            logging.debug("train batch %s-%s (tot: %s)" % (curr_iter_idx_start, curr_iter_idx_end, len(self.__csv_train_data)))
            X = []
            y = []

            curr_iter_idx = curr_iter_idx_start

            for idx in range(self.TRAIN_BATCH_SIZE):
                line = self.__csv_train_data[curr_iter_idx]

                camera, image, steering_angle = self.__select_image_for_training(line, generate_new_images=True, select_camera_at_random=True)
                X.append(image)
                y.append(steering_angle)

                curr_iter_idx += 1
                if curr_iter_idx >  curr_iter_idx_end:
                    curr_iter_idx = curr_iter_idx_start

            curr_iter_idx_start = curr_iter_idx_start + self.TRAIN_BATCH_SIZE
            curr_iter_idx_end = curr_iter_idx_end + self.TRAIN_BATCH_SIZE

            if curr_iter_idx_start > idx_max:
                curr_iter_idx_start = 0
                curr_iter_idx_end = self.TRAIN_BATCH_SIZE - 1

            if curr_iter_idx_end > idx_max:
                curr_iter_idx_end = idx_max

            logging.debug("\t next: %s-%s (tot: %s)" % (curr_iter_idx_start, curr_iter_idx_end, len(self.__csv_train_data)))

            yield np.array(X), np.array(y)


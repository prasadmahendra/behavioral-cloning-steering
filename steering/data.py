import os
import csv
import logging
import cv2
import numpy as np
import math


class Data():
    INPUT_IMAGE_WIDTH = 64
    INPUT_IMAGE_HEIGHT = 64
    NN_INPUT_IMAGE_SHAPE = (INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, 3)

    def __init__(self):
        self.debug = False

    def pre_process(self, image):
        """Pre-processes images before they are fed in to the either the prediction or training model networks"""

        image_out = self.__region_of_interest(image)

        image_out = self.__pixel_mean_subtraction(image_out)

        image_out = self.__pixel_normalize(image_out)

        image_out = self.__resize_image(image_out, self.INPUT_IMAGE_WIDTH, self.INPUT_IMAGE_HEIGHT)

        if self.debug:
            self.display_image("Input image", image)
            self.display_image("Pre-processed image", image_out)

        return image_out

    def __resize_image(self, image, width, height):
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    def __region_of_interest(self, image):
        """Only keeps the region of the image defined by the polygon formed from `vertices`. The rest of the image is set to black."""

        bottom_left_x = 0
        bottom_left_y = image.shape[0] - self.__engine_compartment_pixels()
        top_left_x = 0
        top_left_y = image.shape[0] / 4
        top_right_x = image.shape[1]
        top_right_y = top_left_y
        bottom_right_x = image.shape[1]
        bottom_right_y = image.shape[0] - self.__engine_compartment_pixels()


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

        return image

    def __pixel_normalize(self, image):
        image = cv2.normalize(image, image, alpha=0.1, beta=0.9, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        assert (math.isclose(np.min(image), 0.1, abs_tol=0.0001) and math.isclose(np.max(image), 0.9, abs_tol=0.0001)), "__normalize failed. The range of the input data is: %.10f to %.10f" % (np.min(image), np.max(image))
        return image

    def __pixel_mean_subtraction(self, image):
        image = image.astype(dtype='float64')
        image -= np.mean(image, dtype='float64', axis=0)
        assert (round(np.mean(image)) == 0), "__mean_subtraction error. The mean of the input data is: %f" % np.mean(image)
        return image

    def __engine_compartment_pixels(self):
        return 20

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

    def display_image(self, title, image):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class TrainData(Data):
    LEFT_CAMERA_IMAGE_IDX = 1
    CENTER_CAMERA_IMAGE_IDX = 0
    RIGHT_CAMERA_IMAGE_IDX = 2
    STEERING_ANGLE_IDX = 3
    IMAGE_PATH_PREFIX = "data/"
    CAMERA_POS = ["left", "center", "right"]
    CAMERA_TO_STEERING_ANGLE_ADJ = [0.25, 0, -0.25]

    __logger = logging.getLogger(__name__)

    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise "missing training_data file!"

        self.__csv_data_file = data_file
        self.__csv_data = None
        super(TrainData, self).__init__()

    def test(self):
        logging.info("running selfdiag ...")

        self.__load_data()
        assert (self.data() != None)

        line_data = self.data()[51]

        camera, image_file, steering_angle = self.__select_camera_at_random(line_data)
        assert(camera != None)
        assert(image_file != None)
        assert(steering_angle != None)

        logging.info("loading camera: %s image: %s steering_angle: %s ..." % (camera, image_file, steering_angle))

        camera_image = self.__load_image(image_file)
        assert (camera_image.shape[0] == 160)
        assert (camera_image.shape[1] == 320)
        assert (camera_image.shape[2] == 3)

        camera, camera_image, steering_angle = self.__select_image_for_training(line_data)
        logging.info("loading camera: %s image: %s steering_angle: %s ..." % (camera, image_file, steering_angle))

        assert (camera_image.shape[0] == self.INPUT_IMAGE_HEIGHT)
        assert (camera_image.shape[1] == self.INPUT_IMAGE_WIDTH)
        assert (camera_image.shape[2] == 3)

    def data(self):
        return self.__csv_data

    def __select_image_for_training(self, line_data, generate_new_images=True):
        camera, image_file, steering_angle = self.__select_camera_at_random(line_data)
        image = self.__load_image(image_file)
        image = self.pre_process(image)

        if generate_new_images:
            # randomly flip along x/y axis ...
            image, steering_angle = self.__flip_image_lr_at_rand(image, steering_angle)

            # adj brightness ...
            image = self.__adj_brightness_at_rand(image)

            # skew for training recovery ...
            image, steering_angle = self.__x_y_skew_at_rand(image, steering_angle)

        return camera, image, steering_angle

    def __x_y_skew_at_rand(self, image, steering_angle, both_axis=False):
        x_range = image.shape[1] / 4
        y_range = image.shape[0] / 4
        angle_multiplier = 0.5

        x_tran = x_range * np.random.uniform() - (x_range / 2)  # + or -
        new_steering_angle = steering_angle + (x_tran / x_range) * angle_multiplier

        y_tran = 0
        if both_axis:
            y_tran = y_range * np.random.uniform() - (y_range / 2)  # + or -

        translation_mat = np.float32([[1, 0, x_tran], [0, 1, y_tran]])
        new_image = cv2.warpAffine(image, translation_mat, (image.shape[1], image.shape[0]))

        logging.debug("x tran: %s y tran: %s steering adj: %s -> %s" % (x_tran, y_tran, steering_angle, new_steering_angle))
        if self.debug:
            self.display_image("X-Y skew", new_image)

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

    def __select_camera_at_random(self, line_data):
        selection_idx = np.random.randint(3)
        camera_image = line_data[selection_idx]
        steering_angle = float(line_data[self.STEERING_ANGLE_IDX].strip()) + self.CAMERA_TO_STEERING_ANGLE_ADJ[selection_idx]
        return self.CAMERA_POS[selection_idx], str(self.IMAGE_PATH_PREFIX + camera_image.strip()), steering_angle

    def __load_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __load_data(self):
        # csv format: center,left,right,steering,throttle,brake,speed
        with open(self.__csv_data_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            self.__csv_data = list(csv_reader)[1:]
            self.__logger.info("loaded data points: %s" % (len(self.__csv_data)))


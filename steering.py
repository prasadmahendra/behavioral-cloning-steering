import argparse
import logging
import configparser
from steering.data import TrainData, Data
from steering.model import Model

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Self-Driving Car Steering Model')

parser.add_argument('-data', help='Train data csv file')
parser.add_argument('-cmd', help='Commands (default: selfdiag)', choices=['selfdiag', 'train', 'predict'], required=True)
parser.add_argument('-epoch', default=5, help='Training Epochs (default 5)', type=int)
parser.add_argument('-loadsaved', default=True, help='Load saved weights (default: True)', action="store_true")
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument('-file', help='Image file to use in predicting (required when cmd == predict)')
parser.add_argument('-expected_angle', type=float, help='expected steering angle')

args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def run():
    config = configparser.ConfigParser()
    config.read("settings.ini")

    logger.info("Running cmd: %s" % (args.cmd))

    if args.cmd == "selfdiag":
        data = TrainData(config, args.data)
        data.selfdiag()

        model = Model.for_training(config)
        model.selfdiag()

    elif args.cmd == "train":
        data = TrainData(config, args.data)
        model = Model.for_training(config)

        if args.loadsaved:
            logging.info("restore previously saved weights ...")
            model.restore("./saved/model.h5")

        model.train(training_data=data, nb_epoch=args.epoch)

        model.save("./saved")
    elif args.cmd == "predict":
        model = Model.for_predicting(config)

        if args.loadsaved:
            logging.info("restore previously saved weights ...")
            model.restore("./saved/model.h5")

        predicted_steering_angle = model.predict(Data.load_image_from_file(args.file))
        logging.info("predicted_steering_angle: %s expected: %s" % (predicted_steering_angle, args.expected_angle))  # todo: calc loss based on expected angle.

run()


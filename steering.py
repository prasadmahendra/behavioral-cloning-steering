import argparse
import logging
import configparser
from steering.data import TrainData, Data
from steering.nn.model import Model

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Traffic Signs Recognizer')

parser.add_argument('-data', default="data/driving_log.csv", help='Train data csv file')
parser.add_argument('-cmd', help='Commands', choices=['selfdiag', 'train', 'drive', 'predict'], required=True)
parser.add_argument('-epoch', default=6, help='Training Epochs', type=int)
parser.add_argument('-dropout_prob', help='Regularization: dropout probability', type=float)
parser.add_argument('-loadsaved', default=True, help='Load saved model/weights', action="store_true")
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
parser.add_argument('-file', default="data/IMG/left_2016_12_01_13_30_48_287.jpg", help='Image file to use in predicting (required whe cmd == predict)')

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

        model = Model.for_training(config, dropout_prob=args.dropout_prob)
        model.selfdiag()

    elif args.cmd == "train":
        data = TrainData(config, args.data)
        model = Model.for_training(config, dropout_prob=args.dropout_prob)

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
        logging.info("predicted_steering_angle: %s" % (predicted_steering_angle))

run()

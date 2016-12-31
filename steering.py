import argparse
import logging
from steering.data import TrainData
from steering.nn.model import Model

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Traffic Signs Recognizer')

parser.add_argument('-data', default="data/driving_log.csv", help='Train data csv file')
parser.add_argument('-cmd', default="selfdiag", help='Commands', choices=['selfdiag', 'train', 'drive'])
parser.add_argument("-v", "--verbose", help="Verbose output", action="store_true")
args = parser.parse_args()

if args.verbose:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

def run():
    logger.info("Running cmd: %s" % (args.cmd))

    if args.cmd == "selfdiag":
        data = TrainData(args.data)
        data.test()
        Model.test(args.data)
    elif args.cmd == "train":
        data = TrainData(args.data)
        # todo
    elif args.cmd == "drive":
        data = TrainData(args.data)


run()


import argparse

from network_runner import NetworkRunner

parser = argparse.ArgumentParser(
    prog="Traffic Sign Classifier",
    description="Deep neaural networks for classification of traffic signs",
)

parser.add_argument(
    "-m", "--model", type=str, default="InceptionV3", help="model to use"
)
parser.add_argument("-t", "--train", action="store_true")
parser.add_argument("-p", "--predict", action="store_true")
parser.add_argument("--img_width", type=int)
parser.add_argument("--img_height", type=int)
args = parser.parse_args()


if __name__ == "__main__":
    networkrunner = NetworkRunner(args.model, args.train)
    networkrunner.run()

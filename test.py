from feature_normalization import FeatureProcessor, ZNormalizer
import argparse

TRAIN_FILE = "data/train.tsv"
TEST_FILE = "data/test.tsv"
OUTPUT = "data/test_proc.tsv"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="File to be used for normalization parameters training",
                        default=TRAIN_FILE)
    parser.add_argument("-i", "--input", help="File to be processed", default=TEST_FILE)
    parser.add_argument("-o", "--output", help="Output file", default=OUTPUT)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    zn = ZNormalizer()
    zn.fit(args.train)
    pr = FeatureProcessor(zn)
    pr.process(args.input, args.output)

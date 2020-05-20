from __future__ import print_function, division
from segmenter import SegmenterNet
import os
from config import *
import pprint

cfg = fusion_config()
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.GPU)
print("creating network model using gpu " + str(cfg.GPU))

# import the network model after selecting the gpu


def main():

    pprint.pprint(cfg)
    # create and train the net
    net = SegmenterNet(cfg)
    net.train_segmenter_fusion(training_data_path=cfg.training_data_path,
                               validation_data_path=cfg.validation_data_path)


if __name__ == "__main__":
    main()

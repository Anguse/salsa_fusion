import numpy as np
from easydict import EasyDict as edict


def lidar_config():

    cfg = edict()

    """road-vehicle segmentation using 3D LIDAR point clouds"""
    # classes
    cfg.CLASSES = [
        'background',
        'road',
        'vehicle']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[255.00,  255.00,  255.00],
         [0.00,  255.00, 0.00],
         [255.00, 0.00, 0.00]])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.01,  6.03, 15.78])  # smooth_freq weight values

    cfg.GPU = 0                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # Probability to keep a node in dropout
    cfg.NUM_EPOCHS = 300           # epoch number
    cfg.BATCH_SIZE = 32            # batch size
    cfg.LEARNING_RATE = 0.01       # learning rate
    cfg.LR_DECAY_FACTOR = 0.1      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = True         # print log to console in debug mode
    cfg.DATA_AUGMENTATION = True   # Whether to do data augmentation
    cfg.CHANNEL_LABELS = ['mean',  'max', 'ref',  'den']  # channel names
    cfg.IMAGE_WIDTH = 256          # image width
    cfg.IMAGE_HEIGHT = 64          # image height
    cfg.IMAGE_CHANNEL = len(cfg.CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "../data/salsaNet_bev/salsaNet_bev_train/"
    cfg.validation_data_path = "../data/salsaNet_bev/salsaNet_bev_val/"
    cfg.log_path = "../logs/"
    cfg.log_name = ""   # additional descriptive tag name to the log file if needed

    return cfg


def depth_config():

    cfg = edict()

    """road-vehicle segmentation using depth image"""
    # classes
    cfg.CLASSES = [
        'wall',
        'road',
        'obstacle',
        'car',
        'background']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[0.0],
         [1.0],
         [2.0],
         [3.0],
         [4.0]
         ])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.73, 1.76, 11.26, 26.27, 1.73])  # smooth_freq weight values

    cfg.GPU = 0                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # probability to keep a node in dropout
    cfg.NUM_EPOCHS = 100           # epoch number
    cfg.BATCH_SIZE = 8             # batch size
    cfg.LEARNING_RATE = 0.010       # learning rate
    cfg.LR_DECAY_FACTOR = 0.06      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = False          # print log to console in debug mode
    cfg.DATA_AUGMENTATION = False   # whether to do data augmentation
    cfg.CHANNEL_LABELS = ['depth']  # channel names
    cfg.IMAGE_WIDTH = 320          # image width
    cfg.IMAGE_HEIGHT = 240          # image height
    cfg.IMAGE_CHANNEL = len(cfg.CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "/data/tmp/hl_data/dataset_newnew/depth/train"
    cfg.validation_data_path = "/data/tmp/hl_data/dataset_newnew/depth/val"
    cfg.log_path = "../logs/"
    cfg.log_name = "depth"   # additional descriptive tag name to the log file if needed

    return cfg


def rgb_config():

    cfg = edict()

    """road-vehicle segmentation using depth image"""
    # classes
    cfg.CLASSES = [
        'wall',
        'road',
        'obstacle',
        'car',
        'background']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[0.0],
         [1.0],
         [2.0],
         [3.0],
         [4.0]
         ])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.73, 1.76, 11.26, 26.27, 1.73])  # smooth_freq weight values

    cfg.GPU = 1                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # probability to keep a node in dropout
    cfg.NUM_EPOCHS = 100           # epoch number
    cfg.BATCH_SIZE = 8             # batch size
    cfg.LEARNING_RATE = 0.01       # learning rate
    cfg.LR_DECAY_FACTOR = 0.06      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = False          # print log to console in debug mode
    cfg.DATA_AUGMENTATION = False   # whether to do data augmentation
    cfg.CHANNEL_LABELS = ['r', 'g', 'b']  # channel names
    cfg.IMAGE_WIDTH = 320          # image width
    cfg.IMAGE_HEIGHT = 240          # image height
    cfg.IMAGE_CHANNEL = len(cfg.CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "/data/tmp/hl_data/dataset_newnew/rgb/train"
    cfg.validation_data_path = "/data/tmp/hl_data/dataset_newnew/rgb/val"
    cfg.log_path = "../logs/"
    cfg.log_name = "rgb"   # additional descriptive tag name to the log file if needed

    return cfg


def rgbd_config():

    cfg = edict()

    """road-vehicle segmentation using depth image"""
    # classes
    cfg.CLASSES = [
        'wall',
        'road',
        'obstacle',
        'car',
        'background']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[0.0],
         [1.0],
         [2.0],
         [3.0],
         [4.0]
         ])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.73, 1.76, 11.26, 26.27, 1.73])  # smooth_freq weight values

    cfg.GPU = 1                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # probability to keep a node in dropout
    cfg.NUM_EPOCHS = 100           # epoch number
    cfg.BATCH_SIZE = 8             # batch size
    cfg.LEARNING_RATE = 0.01       # learning rate
    cfg.LR_DECAY_FACTOR = 0.06      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = False          # print log to console in debug mode
    cfg.DATA_AUGMENTATION = False   # whether to do data augmentation
    cfg.CHANNEL_LABELS = ['depth', 'r', 'g', 'b']  # channel names
    cfg.IMAGE_WIDTH = 320          # image width
    cfg.IMAGE_HEIGHT = 240          # image height
    cfg.IMAGE_CHANNEL = len(cfg.CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "/data/tmp/hl_data/dataset_newnew/rgbd/train"
    cfg.validation_data_path = "/data/tmp/hl_data/dataset_newnew/rgbd/val"
    cfg.log_path = "../trash_logs/"
    cfg.log_name = "rgbd"   # additional descriptive tag name to the log file if needed

    return cfg


def laser_config():

    cfg = edict()

    """road-vehicle segmentation using depth image"""
    # classes
    cfg.CLASSES = [
        'wall',
        'road',
        'obstacle',
        'car',
        'unknown']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[0.0],
         [1.0],
         [2.0],
         [3.0],
         [4.0]
         ])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.73, 1.76, 11.26, 26.27, 1])  # smooth_freq weight values

    cfg.GPU = 1                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # probability to keep a node in dropout
    cfg.NUM_EPOCHS = 50           # epoch number
    cfg.BATCH_SIZE = 8             # batch size
    cfg.LEARNING_RATE = 0.01       # learning rate
    cfg.LR_DECAY_FACTOR = 0.06      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = False          # print log to console in debug mode
    cfg.DATA_AUGMENTATION = False   # whether to do data augmentation
    cfg.CHANNEL_LABELS = ['occupancy']  # channel names
    cfg.IMAGE_WIDTH = 80          # image width
    cfg.IMAGE_HEIGHT = 60          # image height
    cfg.IMAGE_CHANNEL = len(cfg.CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "/data/tmp/hl_data/dataset_newnew/laser/train"
    cfg.validation_data_path = "/data/tmp/hl_data/dataset_newnew/laser/val"
    cfg.log_path = "../trash_logs/"
    cfg.log_name = "laser"   # additional descriptive tag name to the log file if needed

    return cfg


def fusion_config():

    cfg = edict()

    """road-vehicle segmentation using depth image"""
    # classes
    cfg.CLASSES = [
        'wall',
        'road',
        'obstacle',
        'car',
        'background']

    cfg.NUM_CLASS = len(cfg.CLASSES)    # number of classes
    # dict from class name to id
    cfg.CLS_2_ID = dict(zip(cfg.CLASSES, range(len(cfg.CLASSES))))

    # rgb color for each class
    cfg.CLS_COLOR_MAP = np.array(
        [[0.0],
         [1.0],
         [2.0],
         [3.0],
         [4.0]
         ])

    cfg.CLS_LOSS_WEIGHTS = np.array(
        [1.73, 1.76, 11.26, 26.27, 1.73])  # smooth_freq weight values

    cfg.GPU = 1                    # gpu id
    cfg.DROPOUT_PROB = 0.5         # probability to keep a node in dropout
    cfg.NUM_EPOCHS = 100           # epoch number
    cfg.BATCH_SIZE = 8             # batch size
    cfg.LEARNING_RATE = 0.01       # learning rate
    cfg.LR_DECAY_FACTOR = 0.06      # multiply the learning rate by this factor
    cfg.LR_DECAY_CYCLE = 20000     # step time to decrease the learning rate
    cfg.PRINT_EVERY = 20           # print in every 20 epochs
    cfg.DEBUG_MODE = False          # print log to console in debug mode
    cfg.DATA_AUGMENTATION = False   # whether to do data augmentation

    cfg.LASER_CHANNEL_LABELS = ['occupancy']  # channel names
    cfg.LASER_IMAGE_WIDTH = 80          # image width
    cfg.LASER_IMAGE_HEIGHT = 60          # image height
    cfg.LASER_IMAGE_CHANNEL = len(cfg.LASER_CHANNEL_LABELS)  # image channel
    cfg.DEPTH_CHANNEL_LABELS = ['depth']  # channel names
    cfg.DEPTH_IMAGE_WIDTH = 320          # image width
    cfg.DEPTH_IMAGE_HEIGHT = 240          # image height
    cfg.DEPTH_IMAGE_CHANNEL = len(cfg.DEPTH_CHANNEL_LABELS)  # image channel

    # paths
    cfg.training_data_path = "/data/tmp/hl_data/dataset_newnew/fusion/train"
    cfg.validation_data_path = "/data/tmp/hl_data/dataset_newnew/fusion/val"
    cfg.log_path = "../trash_logs/"
    # additional descriptive tag name to the log file if needed
    cfg.log_name = "fusion_decoder"

    return cfg

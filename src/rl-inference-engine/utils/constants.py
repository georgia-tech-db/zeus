import os as _os

import torch

# runtime constants

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")   # use CPU or GPU
MODEL_ROOT = 'models/'
UDF_MODEL_ROOT = _os.path.join(MODEL_ROOT, 'action_reg_models')
RL_MODEL_ROOT = _os.path.join(MODEL_ROOT, 'rl_models')
# dataset contants

NORMALIZE = {}
BASE_DIR = _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

DATA_DIR = _os.path.join(BASE_DIR, 'data')


# BDD100k constants

BDD100K_BASE_DIR = _os.path.join(DATA_DIR, 'datasets/bdd100k/')
BDD100K_DATA_PATH = _os.path.join(BDD100K_BASE_DIR, 'video_frames/')
BDD100K_ANNOTATION_PATH = _os.path.join(
    BDD100K_BASE_DIR, 'labels.txt')
NORMALIZE['bdd100k'] = {'mean': [0.3588, 0.4149, 0.4291],
                        'std': [0.2114, 0.2277, 0.2461]}

# Cityscapes constants

CITYSCAPES_BASE_DIR = _os.path.join(DATA_DIR, 'cityscapes/')
CITYSCAPES_DATA_PATH = _os.path.join(
    CITYSCAPES_BASE_DIR, 'leftImg8bit_allFrames/val/frankfurt')
CITYSCAPES_ANNOTATION_PATH = _os.path.join(
    CITYSCAPES_BASE_DIR, 'temporal_labels.txt')
NORMALIZE['cityscapes'] = {'mean': [0.3588, 0.4149, 0.4291],
                           'std': [0.2114, 0.2277, 0.2461]}

# KITTI constants

KITTI_BASE_DIR = _os.path.join(DATA_DIR, 'KITTI/')
KITTI_DATA_PATH = _os.path.join(
    KITTI_BASE_DIR,
    'raw_data/2011_10_03/2011_10_03_drive_0027_sync/image_02/data/')
KITTI_ANNOTATION_PATH = _os.path.join(
    KITTI_BASE_DIR, 'temporal_labels.txt')
NORMALIZE['kitti'] = {'mean': [0.3588, 0.4149, 0.4291],
                      'std': [0.2114, 0.2277, 0.2461]}

# Thumos14 constants

THUMOS14_BASE_DIR = _os.path.join(DATA_DIR, 'thumos14/')
THUMOS14_DATA_PATH = _os.path.join(
    THUMOS14_BASE_DIR, 'video_frames/')
THUMOS14_ANNOTATION_PATH = _os.path.join(
    THUMOS14_BASE_DIR, 'annotations/')

NORMALIZE['thumos14'] = {'mean': [0.43216, 0.394666, 0.37645],
                         'std': [0.22803, 0.22145, 0.216989]}

# ActivityNet constants

ACTIVITYNET_BASE_DIR = _os.path.join(DATA_DIR, 'ActivityNet/')
ACTIVITYNET_DATA_PATH = _os.path.join(
    ACTIVITYNET_BASE_DIR, 'video_frames/')
ACTIVITYNET_RAW_DATA_PATH = _os.path.join(
    ACTIVITYNET_BASE_DIR, 'videos/v1-3/train_val/')
ACTIVITYNET_ANNOTATION_PATH = _os.path.join(
    ACTIVITYNET_BASE_DIR, 'Crawler/activity_net.v1-3.min.json')

NORMALIZE['activitynet'] = {'mean': [0.4477, 0.4209, 0.3906],
                            'std': [0.2767, 0.2695, 0.2714]}

# training parameter constants

LOG_INTERVAL = 10   # interval for displaying training info
IOU = 0.5

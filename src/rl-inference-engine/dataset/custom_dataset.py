import os
import numpy as np
import random

import logging
import pickle

# custom imports
import utils.constants as cts
from dataset.bdd100k import WindowPPDataset

np.random.seed(42)
random.seed(42)


def get_gts(gt_all):
    gt = []
    i = 0
    while i < len(gt_all):
        if gt_all[i] == 1:
            start = i
            while i < len(gt_all) and gt_all[i] == 1:
                i += 1
            end = i
            gt.append((start, end))
        i += 1
    return gt


def is_action_window(pred, gts, iou):

    label = 0

    for gt in gts:
        inter = len(range(max(pred[0], gt[0]), min(pred[1], gt[1])+1))
        if inter != 0:
            union = len(range(min(pred[0], gt[0]), max(pred[1], gt[1])+1))
        else:
            union = pred[1] - pred[0] + gt[1] - gt[0] + 2

        if inter/(pred[1] - pred[0]) >= iou:
            return 1
        elif inter/union >= iou:
            return 1
        else:
            label = 0

    return label


class TestDataset(object):

    def __init__(self, args):

        self.args = args

        all_cfg_file = os.path.join(
                cts.DATA_DIR,
                cts.UDF_MODEL_ROOT,
                args.class_name,
                'cfg_metadata.pkl')
        all_configs = pickle.load(open(all_cfg_file, 'rb'))

        # picked cfgs for paper
        if self.args.class_name == 'left':
            self.configs = all_configs[3:8]
        elif self.args.class_name == 'crossright':
            self.configs = all_configs[1:7]

        logging.info(
            "Used Configurations: (Resolution | Segment Length | Sample Rate | F1 score | Throughput (sec/frame))")
        for i, item in enumerate(self.configs):
            logging.info("%d %s", i, item)

        self.generate_ground_truth_dicts()

    def reset_video_counters(self):

        self.counter = 0
        self.video_name = list(self.gts.keys())[self.video_count]
        self.tot_frames = len(self.gts[self.video_name])
        self.total_frames += self.tot_frames
        self.new_gts = get_gts(self.gts[self.video_name])

    def get_state(self, config):

        res, cl, sr, _, _ = self.configs[config]

        if self.counter + cl*sr >= self.tot_frames - 1:
            window = (self.counter, self.tot_frames)
            gt = is_action_window(window, self.new_gts, 0.5)

        else:

            window = (self.counter, self.counter+(cl*sr))
            gt = is_action_window(window, self.new_gts, 0.5)

        # end of video
        if self.counter + cl*sr >= self.tot_frames - 1:
            logging.info("Videos processed: [%d/%d]",
                         self.video_count+1,
                         len(self.gts))
            self.video_count += 1
            # end of all videos
            if self.video_count == len(self.gts):
                self.state = None
            else:
                self.reset_video_counters()
                init_config = len(self.configs)-1
                self.state = (init_config, self.video_name, self.counter)

                next_res, next_cl, next_sr, _, _ = self.configs[init_config]
                self.counter += next_cl*next_sr

        else:
            self.state = (config, self.video_name, self.counter)
            self.counter += cl*sr
        return self.state, gt

    def reset(self):
        self.video_count = 0
        self.total_frames = 0
        self.reset_video_counters()

    def generate_ground_truth_dicts(self):

        self.gts = {}
        # dummy dataset utility to access ground truths
        dummy_dataset = WindowPPDataset(
            phase='test',
            dataset=self.args.dataset,
            class_name=self.args.class_name,
            clip_length=4,
            sample_rate=8,
            overlap=1,
            transform=None)

        self.video_list = dummy_dataset.video_list
        self.num_videos = len(self.video_list)

        for video_name in self.video_list:

            if self.args.dataset == 'activitynet':
                folder_path = os.path.join(
                    dummy_dataset.data_path, 'v_'+video_name)
            elif self.args.dataset in ['cityscapes', 'kitti']:
                folder_path = dummy_dataset.data_path
            else:
                folder_path = os.path.join(
                    dummy_dataset.data_path, video_name)

            if self.args.dataset == 'cityscapes':
                all_frames = os.listdir(folder_path)
                curr_video_frames = []
                for i in all_frames:
                    if i.startswith('frankfurt_'+video_name):
                        curr_video_frames.append(i)
                num_images = len(curr_video_frames)
            else:
                num_images = len(os.listdir(folder_path))

            gts = []

            class_names = [self.args.class_name]
            for item in dummy_dataset.ann_dict[video_name]:
                if self.args.dataset in ['bdd100k', 'cityscapes', 'kitti']:
                    for i, class_name in enumerate(class_names):
                        if item[0].lower() == class_name.lower():
                            if item[1] != -1 and item[2] != -1:
                                gts.append((item[1], item[2]))
                else:
                    gts.append((item[0], item[1]))

            self.gts[video_name] = np.zeros(num_images, dtype=int)
            for item in gts:
                self.gts[video_name][item[0]:item[1]] = 1

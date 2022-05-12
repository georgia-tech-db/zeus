"""Dataset definitions for UA-DETRAC dataset"""
import os
import logging
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch

# import custom modules

import utils.constants as cts

# helper functions


def get_train_test_split(ann_dict):
    video_list = sorted(list(ann_dict.keys()))
    train_list, test_list = train_test_split(
        video_list, test_size=0.5, random_state=50)
    return train_list, test_list


def get_annotation_dict(annotation_path, dataset, class_name):
    anns = {}

    if dataset in ['bdd100k', 'cityscapes', 'kitti']:
        with open(annotation_path) as read_file:
            lines = read_file.readlines()
            for line in lines:
                line = line.split()
                video_name = os.path.splitext(line[0])[0]
                if video_name not in anns:
                    anns[video_name] = []
                anns[video_name].append((line[1], int(line[2]), int(line[3])))

    return anns


# iou based function - does current window contain an action?

def is_action_window(pred, gt, iou):
    inter = len(range(max(pred[0], gt[0]), min(pred[1], gt[1])+1))
    if inter != 0:
        union = len(range(min(pred[0], gt[0]), max(pred[1], gt[1])+1))
    else:
        union = pred[1] - pred[0] + gt[1] - gt[0] + 2

    if inter/(pred[1] - pred[0]) >= iou:
        return True
    elif inter/union >= iou:
        return True
    else:
        return False


def window_gap(window, item):
    if window[1] > item[1]:
        return window[0] - item[1]
    elif window[1] <= item[1]:
        return item[0] - window[1]


# does current frame contain an action?

def is_action_frame(frame_no, gt):
    if frame_no >= gt[0] and frame_no <= gt[1]:
        return True
    else:
        return False


class WindowPPDataset(Dataset):

    """torch.dataset class definition for UA-DETRAC baseline clip dataset

    Arguments:
        :param phase (str): training, validation or testing
        :param class_name (str): the class for which to generate dataset
        :param clip_length (int): number of frames to select from each video
        :param transform (torchvision.transform.transform): video
            transformation function
    """

    def __init__(self,
                 phase,
                 dataset,
                 class_name,
                 clip_length,
                 sample_rate,
                 overlap,
                 transform=None):
        """
        Initialization
        """
        self.phase = phase
        self.dataset = dataset
        self.class_name = [class_name]
        self.clip_length = clip_length
        self.sample_rate = sample_rate
        self.overlap = overlap
        if dataset == 'bdd100k':
            self.data_path = cts.BDD100K_DATA_PATH
            self.ann_dict = get_annotation_dict(
                cts.BDD100K_ANNOTATION_PATH, dataset, class_name)
            train_list, test_list = get_train_test_split(self.ann_dict)
            self.video_list = test_list
        else:
            logging.error("Unsupported dataset.")
            exit(0)

        self.num_videos = len(self.video_list)
        clip_list, label_list = self.generate_window_clips()
        clip_list, label_list = np.array(
            clip_list, dtype=object), np.array(label_list, dtype=object)

        logging.info("Length of test clip list: %d", len(clip_list))

        uniques, counts = np.unique(label_list, return_counts=True)
        logging.info("Counts %d/%d: %d/%d",
                     uniques[0], uniques[1], counts[0], counts[1])

        self.clips, self.labels = clip_list, label_list
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.clips)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        video_name = self.clips[index][0]
        window = self.clips[index][1]
        folder = os.path.join(self.data_path, video_name)
        # Load data
        selected_frames = np.arange(
            window[0] - 1, window[1], self.sample_rate).tolist()
        # (input) spatial images
        X = self.read_images(folder, selected_frames, self.transform)

        # (labels) LongTensor are for int64 instead of FloatTensor
        y = torch.LongTensor([self.labels[index]])

        return X, y, (video_name, window)

    @staticmethod
    def read_images(folder, selected_frames, transform):
        """Read images from disk and apply transforms

        Arguments:
            folder {str} -- path to the folder
            selected_frames {list (int)} -- list of selected frames
            transform {torchvision.transforms} -- image transformation function

        Returns:
            video clip -- 4d tensor of shape (N, C, H, W)
        """
        video_clip = []
        for i in selected_frames:
            image_path = os.path.join(
                folder, 'frame{:06d}.jpg'.format(i))

            image = Image.open(image_path)

            if transform is not None:
                image = transform(image)
            video_clip.append(image)

        video_clip = torch.stack(video_clip, dim=0)
        video_clip = video_clip.permute(1, 0, 2, 3)

        return video_clip

    def generate_window_clips(self):
        """
        Generate train, validation and test splits from dataset

        :return:
            clips_list: list of tuples each containing (folder_name,
                                                        selected_frames)
            labels_list: list of corresponding classes for each data point
        """

        clips_list = []
        labels_list = []
        modified_vid_list = []
        self.ground_truths = {}
        self.windows_list = {}
        for i, item in enumerate(self.video_list):
            video_name = item

            curr_list, curr_labels = self.get_data_for_single_video(video_name)

            clips_list.extend(curr_list)
            labels_list.extend(curr_labels)

            if len(curr_list) > 0:
                modified_vid_list.append(video_name)
                self.windows_list[video_name] = curr_list
                self.ground_truths[video_name] = curr_labels

        self.video_list = modified_vid_list

        return clips_list, labels_list

    def get_data_for_single_video(self, video_name):
        """Get clips and labels for a single video

        Args:
            video_name ([str]): Name of the input video

        Returns:
            [list]: List of tuples (video_name, window)
            [list]: List of labels (0/1) for each tuple
        """
        gts = []
        for item in self.ann_dict[video_name]:
            if self.dataset in ['bdd100k', 'cityscapes', 'kitti']:
                for i, class_name in enumerate(self.class_name):
                    if item[0].lower() == class_name.lower():
                        if item[1] != -1 and item[2] != -1:
                            gts.append((item[1], item[2]))
            else:
                gts.append((item[0], item[1]))

        if len(gts) > 0:
            curr_list, curr_labels = self.get_clips_from_annotations(
                video_name, gts)
        else:
            curr_list, curr_labels = [], []

        return curr_list, curr_labels

    def get_clips_from_annotations(self, video_name, gts):
        """Get final training data from each video based on IOU

        Args:
            video_name (str): name of the video for which to get window data
            gts list(tuples): ground truth information
                            :tuple param 1: action start frame
                            :tuple param 2: action end frame

        Returns:
            [list]: List of tuples (video_name, window)
            [list]: List of labels (0/1) for each tuple
        """

        clip_list = []
        label_list = []
        if self.dataset == 'activitynet':
            folder_path = os.path.join(self.data_path, 'v_'+video_name)
        elif self.dataset in ['cityscapes', 'kitti']:
            folder_path = self.data_path
        else:
            folder_path = os.path.join(self.data_path, video_name)

        if self.dataset == 'cityscapes':
            all_frames = os.listdir(folder_path)
            curr_video_frames = []
            for i in all_frames:
                if i.startswith('frankfurt_'+video_name):
                    curr_video_frames.append(i)
            num_images = len(curr_video_frames)
        else:
            num_images = len(os.listdir(folder_path))

        window_size = self.clip_length*self.sample_rate
        window = (1, window_size)

        while window[1] < num_images:
            label = 0
            for item in gts:
                label = (label | is_action_window(window, item, cts.IOU))

            clip_list.append((video_name, window))
            label_list.append(label)

            window = (window[0] + window_size,
                      window[1] + window_size)

        return clip_list, label_list

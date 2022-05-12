"""Dataset definitions for UA-DETRAC dataset"""
import os
import logging
import time
import numpy as np
from tqdm import tqdm

from PIL import Image
import torch
import torchvision.transforms as transforms

# import custom modules

import utils.constants as cts
from models.window_pp import WindowPP
from utils.io import load_pytorch_model


class CustomFeatExtractor():

    """torch.dataset class definition for UA-DETRAC baseline clip dataset

    Arguments:
        :param phase (str): training, validation or testing
        :param class_name (str): the class for which to generate dataset
        :param clip_length (int): number of frames to select from each video
        :param transform (torchvision.transform.transform): video
            transformation function
    """

    def __init__(self, args, configs):
        """
        Initialization
        """
        self.args = args
        if args.dataset == 'bdd100k':
            self.data_path = cts.BDD100K_DATA_PATH
        elif args.dataset == 'cityscapes':
            self.data_path = cts.CITYSCAPES_DATA_PATH
        elif args.dataset == 'kitti':
            self.data_path = cts.KITTI_DATA_PATH
        elif args.dataset == 'thumos14':
            self.data_path = os.path.join(cts.THUMOS14_DATA_PATH, 'val')
        elif args.dataset == 'activitynet':
            self.data_path = os.path.join(cts.ACTIVITYNET_DATA_PATH, 'val')
        self.configs = configs
        self.model_catalog = self.generate_model_catalog()

    def get_observation_from_state(self, state):
        """Generate observation from state using udf"""
        # Select sample
        config, video_name, index = state
        res, cl, sr, _, _ = self.configs[config]

        if self.args.dataset == 'activitynet':
            folder = os.path.join(self.data_path, 'v_'+video_name)
        elif self.args.dataset == 'cityscapes':
            folder = os.path.join(self.data_path, 'frankfurt_' + video_name)
        elif self.args.dataset == 'kitti':
            folder = self.data_path
        else:
            folder = os.path.join(self.data_path, video_name)
        # Load data
        selected_frames = np.arange(
            index, index+cl*sr, sr).tolist()
        # (input) spatial images
        input_clip = self.read_images(
            folder, selected_frames, res, self.args.dataset)
        input_clip = input_clip.unsqueeze(0).to(cts.DEVICE)
        # (labels) LongTensor are for int64 instead of FloatTensor
        with torch.no_grad():
            start = time.time()
            obs, output = \
                self.model_catalog[config](input_clip)
            if cts.USE_CUDA:
                torch.cuda.synchronize()
            end = time.time()
        # import pdb; pdb.set_trace();
        # output_probs = torch.nn.functional.softmax(output)
        # if self.args.class_name == 'left':
        #     if output_probs[0][1].item() > 0.70:
        #         pred = 1
        #     else:
        #          pred = 0
        # else:
        pred = torch.argmax(output).item()
        return obs, pred, (end - start)

    @staticmethod
    def read_images(folder, selected_frames, res, dataset):
        """Read images from disk and apply transforms

        Arguments:
            folder {str} -- path to the folder
            selected_frames {list (int)} -- list of selected frames
            transform {torchvision.transforms} -- image transformation function

        Returns:
            video clip -- 4d tensor of shape (N, C, H, W)
        """
        mean = cts.NORMALIZE[dataset]['mean']
        std = cts.NORMALIZE[dataset]['std']
        transform = transforms.Compose([
            transforms.Resize([res, res]),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        video_clip = []
        for i in selected_frames:
            if dataset == 'cityscapes':
                image_path = folder + '_{:06d}_leftImg8bit.png'.format(i)
            elif dataset == 'kitti':
                image_path = os.path.join(folder, (str(i).zfill(10)+'.png'))
            else:
                image_path = os.path.join(
                    folder, 'frame{:06d}.jpg'.format(i))

            image = Image.open(image_path).convert('RGB')

            if transform is not None:
                image = transform(image)
            video_clip.append(image)

        video_clip = torch.stack(video_clip, dim=0)
        video_clip = video_clip.permute(1, 0, 2, 3)

        return video_clip

    def generate_model_catalog(self):

        model_catalog = {}

        # if self.args.dataset in ['bdd100k', 'cityscapes', 'kitti']:
        #     if not self.args.multi_class:
        #         exp_name = 'high_res_50_50_split'
        #     else:
        #         exp_name = 'multi_class_exps_combined'
        # elif self.args.dataset == 'thumos14':
        #     exp_name = 'thumos14_corrected_f1_iou_0.5'
        # elif self.args.dataset == 'activitynet':
        #     exp_name = 'activitynet_init'

        model_root = os.path.join(
            cts.DATA_DIR,
            cts.UDF_MODEL_ROOT,
            self.args.class_name)

        logging.info("Loading action recognition models...")
        for i, _ in enumerate(tqdm(self.configs)):
            model = WindowPP(
                num_classes=2, dataset=self.args.dataset).to(cts.DEVICE)
            model_path = os.path.join(
                    model_root,
                    'cfg_{}.pth'.format(i))

            # if self.args.dataset in ['bdd100k', 'cityscapes', 'kitti']:
            #     if not self.args.multi_class:
            #         model_path = os.path.join(
            #             model_root,
            #             'overlap_1',
            #             'clip_length_'+str(cl),
            #             'resolution_'+str(res),
            #             'model.pth')
            #     else:
            #         model_path = os.path.join(
            #             model_root,
            #             'overlap_1',
            #             'clip_length_'+str(6),
            #             'sample_rate_'+str(1),
            #             'resolution_'+str(250),
            #             'model.pth')
            # else:
            #     model_path = os.path.join(
            #         model_root,
            #         'overlap_16',
            #         'clip_length_'+str(32),
            #         'sample_rate_'+str(2),
            #         'resolution_'+str(160),
            #         'model.pth')
            model_catalog[i] = load_pytorch_model(model,
                                                  model_path,
                                                  cts.DEVICE)
            model_catalog[i].eval()
        return model_catalog


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

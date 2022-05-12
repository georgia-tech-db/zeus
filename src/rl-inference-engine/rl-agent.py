import os
import random
import json
import logging

import numpy as np

import torch
from sklearn.metrics import f1_score

# custom imports

from models.dqn import DQNModel
from utils.misc import parse_arguments, set_logger
from utils.io import load_dqn_model

import utils.constants as cts
from dataset.custom_feat_extractor import CustomFeatExtractor
from dataset.custom_dataset import TestDataset

np.random.seed(42)
random.seed(42)
torch.set_printoptions(precision=4, sci_mode=False)


class TestAgent():

    def __init__(self, args):

        self.args = args

        self.test_dataset = TestDataset(args)
        self.configs = self.test_dataset.configs
        self.n_configs = len(self.configs)

        self.model = DQNModel(
            n_actions=self.n_configs, dataset=args.dataset).to(cts.DEVICE)

        self.feat_extractor = CustomFeatExtractor(args, self.configs)

        model_path = os.path.join(
                cts.DATA_DIR,
                cts.RL_MODEL_ROOT,
                args.class_name)
        if not os.path.exists(model_path):
            logging.info("Model folder %s does not exist")
            exit(0)

        self.model = load_dqn_model(
            self.model, model_path, cts.DEVICE)

        self.model.eval()

    def test_agent_with_raw_images(self):

        self.model.eval()
        test_time = 0
        preds = []
        gts = []
        configs = []
        init_config = len(self.test_dataset.configs)-1

        self.test_dataset.reset()

        state, gt = self.test_dataset.get_state(init_config)
        obs, pred, udf_time = self.feat_extractor.get_observation_from_state(
            state)
        test_time += udf_time
        preds.append(pred)
        gts.append(gt)
        configs.append(init_config)
        while obs is not None:
            obs = obs.to(cts.DEVICE)
            q_value = self.model(obs)
            if cts.USE_CUDA:
                torch.cuda.synchronize()
            config = q_value.argmax(
                1).data.cpu().numpy().astype(int)[0]

            state, gt = self.test_dataset.get_state(config)
            if state is None:
                obs = None
                continue
            obs, pred, udf_time = \
                self.feat_extractor.get_observation_from_state(state)

            test_time += udf_time
            preds.append(pred)
            gts.append(gt)
            configs.append(config)

        test_f1 = f1_score(gts, preds)
        logging.info("Total execution time: %.2f", test_time)
        logging.info("Total frames: %d", self.test_dataset.total_frames)
        logging.info("Test F1 score: %.2f", test_f1*100)
        logging.info("FPS: %.2f", self.test_dataset.total_frames/test_time)
        logging.info("Config dist: %s", np.unique(configs, return_counts=True))


if __name__ == "__main__":

    args = parse_arguments()

    args.result_path = os.path.join(
        cts.DATA_DIR,
        'results',
        args.class_name)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    log_file = os.path.join(args.result_path,
                            'test_results.log')
    set_logger(log_file=log_file)

    logging.info("Start execution")
    logging.info("main:: Start execution with args:\n %s",
                 json.dumps(vars(args), indent=4))

    agent = TestAgent(args)

    agent.test_agent_with_raw_images()

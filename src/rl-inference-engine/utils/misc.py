import os
import sys
import logging
import argparse


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,
                        default='bdd100k',
                        help='Name of the dataset for which to run the exp')

    parser.add_argument('--class-name', type=str,
                        default='left',
                        choices=['left', 'crossright'],
                        help='Name of the class for which to run the exp')

    args = parser.parse_args()

    return args


def set_logger(log_file='', debug_mode=False):
    """Initialize the logger"""
    if log_file:
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file))
        handlers = [logging.FileHandler(
            log_file, 'w+'), logging.StreamHandler(sys.stdout)]
    else:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                        format='%(asctime)s %(filename)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)

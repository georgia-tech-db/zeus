import os
import logging
import torch


def load_pytorch_model(model, model_path, device):
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        logging.info("No model checkpoint found at %s. Exiting...", model_path)
        exit(0)

    return model


def load_dqn_model(model, model_path, device):
    model_path = os.path.join(model_path, 'model_best.pt')
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        model.load_state_dict(checkpoint)
    else:
        logging.info("No model checkpoint found at %s. Exiting...", model_path)
        exit(0)

    return model

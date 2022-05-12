import torch.nn as nn


class DQNModel(nn.Module):
    """The agent neural network
    """

    def __init__(self, n_actions=6, n_layers=3, dataset='bdd100k'):
        super(DQNModel, self).__init__()

        self.dataset = dataset
        if self.dataset in ['bdd100k', 'cityscapes', 'kitti']:
            self.fc = nn.Sequential(nn.Linear(512, n_actions))
        else:
            modules = []
            for i in range(n_layers-1):
                modules.append(nn.Linear(256, 256))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(256, n_actions))
            self.fc = nn.Sequential(*modules)

    def forward(self, obs):
        actions = self.fc(obs)
        return actions

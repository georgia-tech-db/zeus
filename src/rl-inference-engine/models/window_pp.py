import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class WindowPP(nn.Module):
    """Model class for end-to-end video classification
    using pytorch r21d network implementation
    """

    def __init__(self, num_classes, dataset):
        super(WindowPP, self).__init__()

        self.dataset = dataset
        if dataset in ['bdd100k', 'cityscapes', 'kitti']:
            self.net = models.video.r3d_18(pretrained=True)
            self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))

            self.fc = nn.Linear(in_features=512, out_features=512)
            self.fc1 = nn.Linear(in_features=512, out_features=256)
            self.fc2 = nn.Linear(in_features=256, out_features=128)
            self.fc3 = nn.Linear(in_features=128, out_features=num_classes)
        else:
            self.net = models.video.r2plus1d_18(pretrained=True)
            fc = nn.Linear(in_features=512, out_features=256)
            self.fc1 = nn.Linear(in_features=256, out_features=num_classes)
            self.net.fc = fc

    def forward(self, x_3d):

        if self.dataset in ['bdd100k', 'cityscapes', 'kitti']:
            out = self.net(x_3d)
            out = F.relu(out)
            feat = out.flatten(1)

            out = F.relu(self.fc1(feat))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
        else:
            out = F.relu(self.net(x_3d))
            feat = out.flatten(1)
            out = self.fc1(out)
        return feat, out


class SegmentPP(nn.Module):

    def __init__(self, num_classes):
        super(SegmentPP, self).__init__()

        self.conv_layer1 = self._conv_layer_set(3, 32, 3, 1)
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(5, 5, 5))
        self.fc1 = nn.Linear(5 * 5 * 5 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.LeakyReLU()

    def _conv_layer_set(self, in_c, out_c, t_dim, pad):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=pad),
            nn.LeakyReLU(),
        )
        return conv_layer

    def forward(self, x):

        out = self.conv_layer1(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out

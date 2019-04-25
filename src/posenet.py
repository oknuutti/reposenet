import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS


class PoseNet(nn.Module):
    """ Based on https://github.com/bexcite/apolloscape-loc/models/posenet.py by Pavlo Bashmakov """

    def __init__(self, arch, num_features=2048, dropout=0.5, cache_dir=None,
                 track_running_stats=False, pretrained=False):
        super(PoseNet, self).__init__()
        self.dropout = dropout
        self.track_running_stats = track_running_stats
        self.pretrained = pretrained
        self.features = num_features

        if cache_dir is not None:
            os.environ['TORCH_MODEL_ZOO'] = cache_dir
        assert arch in models.__dict__, 'invalid model name: %s' % arch
        self.feature_extractor = getattr(models, arch)(pretrained=pretrained)

        if arch.startswith('alexnet'):
            final_layer = 'classifier'
        elif arch.startswith('vgg'):
            final_layer = 'classifier'
        elif arch.startswith('resnet'):
            final_layer = 'fc'
        elif arch.startswith('squeezenet'):
            final_layer = 'classifier'
        elif arch.startswith('densenet'):
            final_layer = 'classifier'
        elif arch.startswith('inception'):
            final_layer = 'fc'
        elif arch.startswith('googlenet'):
            final_layer = 'fc'
        else:
            assert False, 'model %s not supported' % arch

        cls_in_features = getattr(self.feature_extractor, final_layer).in_features
        setattr(self.feature_extractor, final_layer, Identity())
        self.fc_feat = nn.Linear(cls_in_features, num_features)

        # Should use AdaptiveAvgPool2d between fc_feat and fc_xyz/fc_quat? Where in the article / which article says so?
        # self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

        # Translation
        self.fc_xyz = nn.Linear(num_features, 3)

        # Rotation in quaternions
        self.fc_quat = nn.Linear(num_features, 4)

        # Turns off track_running_stats for BatchNorm layers,
        # it simplifies testing on small datasets due to eval()/train() differences
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = self.track_running_stats

        # Initialization
        if self.pretrained:
            init_modules = [self.fc_feat, self.fc_xyz, self.fc_quat]
        else:
            init_modules = self.modules()

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def extract_features(self, x):
        x_features = self.feature_extractor(x)
        x_features = self.fc_feat(x_features)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return x_features

    def forward(self, x):
        # x is batch_images [batch_size x image, batch_size x image]

        #         x = self.feature_extractor(x)

        # if type(x) is list:
        #     x_features = [self.extract_features(xi) for xi in x]
        #     x_translations = [self.fc_xyz(xi) for xi in x_features]
        #     x_rotations = [self.fc_quat(xi) for xi in x_features]
        #     x_poses = [torch.cat((xt, xr), dim=1) for xt, xr in zip(x_translations, x_rotations)]
        # elif torch.is_tensor(x):
        x_features = self.extract_features(x)
        x_translations = self.fc_xyz(x_features)
        x_rotations = self.fc_quat(x_features)
        x_rotations = F.normalize(x_rotations, p=2, dim=1)
        x_poses = torch.cat((x_translations, x_rotations), dim=1)

        return x_poses


class PoseNetCriterion(nn.Module):
    def __init__(self, stereo=False, beta=512.0, learn_uncertainties=False, sx=0.0, sq=-3.0):
        super(PoseNetCriterion, self).__init__()
        self.stereo = stereo
        self.loss_fn = nn.L1Loss()
        self.learn_uncertainties = learn_uncertainties
        if learn_uncertainties:
            self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=learn_uncertainties)
            self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=learn_uncertainties)
            self.beta = 1.0
        else:
            self.sx = 0.0
            self.sq = 0.0
            self.beta = beta

    def forward(self, x, y):
        """
        Args:
            x: list(N x 7, N x 7) - prediction (xyz, quat)
            y: list(N x 7, N x 7) - target (xyz, quat)
        """

        loss = 0
        if self.stereo:
            for i in range(2):
                # Translation loss
                loss += torch.exp(-self.sx) * self.loss_fn(x[i][:, :3], y[i][:, :3]) + self.sx
                # Rotation loss
                loss += torch.exp(-self.sq) * self.beta * self.loss_fn(x[i][:, 3:], y[i][:, 3:]) + self.sq

            # Normalize per image so we can compare stereo vs no-stereo mode
            loss = loss / 2
        else:
            # Translation loss
            loss += torch.exp(-self.sx) * self.loss_fn(x[:, :3], y[:, :3]) + self.sx
            # Rotation loss
            loss += torch.exp(-self.sq) * self.beta * self.loss_fn(x[:, 3:], y[:, 3:]) + self.sq
        #         print('x = \n{}'.format(x[0]))
        #         print('y = \n{}'.format(y[0]))
        return loss


class PoseDataset(ImageFolder):
    def __init__(self, root, label_file, random_crop=False, loader=default_loader):
        self.root = root
        self.label_file = label_file
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224) if random_crop else transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = None
        self.loader = loader
        self.extensions = IMG_EXTENSIONS
        self.imgs = self.samples = self.load_samples()
        self.targets = [s[1] for s in self.samples]

    def load_samples(self):
        samples = []
        for scene_dir in os.listdir(self.root):
            label_file = os.path.join(self.root, scene_dir, self.label_file)
            if os.path.exists(label_file):
                with open(label_file, 'r') as fh:
                    for line in fh.readlines():
                        row = line.strip('\r\n').split(' ')
                        if len(row) == 8:
                            pose = list(map(float, row[1:]))
                            if np.linalg.norm(pose) < 1000:
                                samples.append((
                                    os.path.join(self.root, scene_dir, row[0]),
                                    torch.tensor(pose)))
                            else:
                                print('inexpected meta data at %s: %s' % (scene_dir, row,))

        return samples


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

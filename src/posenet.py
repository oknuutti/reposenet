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
        self.register_buffer('position_scale', torch.Tensor([1, 1, 1]))

        if cache_dir is not None:
            os.environ['TORCH_MODEL_ZOO'] = cache_dir
        assert arch in models.__dict__, 'invalid model name: %s' % arch

        if arch.startswith('alexnet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)
            repl_layers = ('classifier',)

        elif arch.startswith('vgg'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)
            repl_layers = ('classifier',)

        elif arch.startswith('resnet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)
            repl_layers = ('fc',)

        elif arch.startswith('squeezenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)
            repl_layers = ('classifier',)

        elif arch.startswith('densenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)
            repl_layers = ('classifier',)

        elif arch.startswith('inception'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)
            repl_layers = ('fc', 'AuxLogits.fc')

        elif arch.startswith('googlenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)
            repl_layers = ('fc', 'aux1.fc2', 'aux2.fc2')

        else:
            assert False, 'model %s not supported' % arch

        # replace all logit layers with identity layers, add corresponding regression layers to self
        extra_init = []
        self.aux_qty = 0
        for i, repl_layer in enumerate(repl_layers):
            np = self.feature_extractor
            for c in repl_layer.split('.'):
                p = np
                np = getattr(p, c)
            setattr(p, c, Identity())
            if i == 0:
                self.fc_feat = nn.Linear(np.in_features, num_features)
            else:
                setattr(self, 'aux_v'+str(i), nn.Linear(np.in_features, 3))
                setattr(self, 'aux_q'+str(i), nn.Linear(np.in_features, 4))
                self.aux_qty = i
                extra_init.append(p)    # auxiliary heads are not pretrained

        # Should use AdaptiveAvgPool2d between fc_feat and fc_vect/fc_quat? Where in the article / which article says so?
        # self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)

        # Translation
        self.fc_vect = nn.Linear(num_features, 3)

        # Rotation in quaternions
        self.fc_quat = nn.Linear(num_features, 4)

        # Turns off track_running_stats for BatchNorm layers,
        # it simplifies testing on small datasets due to eval()/train() differences
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = self.track_running_stats

        # Initialization (feature_extractor already initialized in its __init__ method)
        init_modules = [self.fc_feat, self.fc_vect, self.fc_quat] \
                       + [getattr(self, 'aux_v' + str(i)) for i in range(1, self.aux_qty + 1)] \
                       + [getattr(self, 'aux_q' + str(i)) for i in range(1, self.aux_qty + 1)]

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        # Cost function has trainable parameters so included here
        self.cost_fn = PoseNetCriterion(stereo=False, learn_uncertainties=True, sx=0.0, sq=-6.24)

    def set_target_scale(self, targets):
        tmp = torch.Tensor(np.array([t.data[:3].numpy() for t in targets]))
        self.position_scale.data = torch.std(tmp[:, :3], dim=0)

    def extract_features(self, x):
        x_features = self.feature_extractor(x)
        x_features, *auxs = x_features if isinstance(x_features, tuple) else (x_features, )
        x_features = self.fc_feat(x_features)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return [x_features] + list(auxs)

    def forward(self, x):
        # x is batch_images [batch_size x image, batch_size x image]

        x_features, *auxs = self.extract_features(x)
        x_translations = self.fc_vect(x_features)
        x_translations = self.position_scale.unsqueeze(0) * x_translations
        x_rotations = self.fc_quat(x_features)
        x_rotations = F.normalize(x_rotations, p=2, dim=1)
        x_poses = torch.cat((x_translations, x_rotations), dim=1)

        if self.training and len(auxs) > 0:
            aux_output = []
            for i, aux in enumerate(auxs):
                av = getattr(self, 'aux_v' + str(i + 1))(aux)
                av = self.position_scale.unsqueeze(0) * av
                aq = getattr(self, 'aux_q' + str(i + 1))(aux)
                aq = F.normalize(aq, p=2, dim=1)
                aux_output.append(torch.cat((av, aq), dim=1))
            return [x_poses] + aux_output

        return x_poses

    def cost(self, x, y):
        return self.cost_fn(x, y)


class PoseNetCriterion(nn.Module):
    def __init__(self, stereo=False, beta=500.0, learn_uncertainties=False, sx=0.0, sq=-3):
        super(PoseNetCriterion, self).__init__()
        self.stereo = stereo
        self.loss_fn = nn.L1Loss()
        self.learn_uncertainties = learn_uncertainties
        if learn_uncertainties:
            self.sx = nn.Parameter(torch.Tensor([sx]))
            self.sq = nn.Parameter(torch.Tensor([sq]))
            self.beta = 1.0
        else:
            self.sx = 0.0
            self.sq = 0.0
            self.beta = beta

    def forward(self, all_x, y):
        """
        Args:
            x: list(N x 7, N x 7) - prediction (xyz, quat)
            y: list(N x 7, N x 7) - target (xyz, quat)
        """

        if not isinstance(all_x, (list, tuple)):
            all_x = (all_x,)

        loss = 0
        for x in all_x:
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
                xq = torch.sign(x[:, 3]).unsqueeze(1) * x[:, 3:]  # map both hemispheres of the quaternions to a single one (y done already)
                loss += torch.exp(-self.sq) * self.beta * self.loss_fn(xq, y[:, 3:]) + self.sq
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
        label_file = os.path.join(self.root, self.label_file)
        for scene_dir in ([''] if os.path.exists(label_file) else os.listdir(self.root)):
            label_file = os.path.join(self.root, scene_dir, self.label_file)
            if os.path.exists(label_file):
                with open(label_file, 'r') as fh:
                    for line in fh.readlines():
                        row = line.strip('\r\n').split(' ')
                        if len(row) == 8:
                            pose = np.array(list(map(float, row[1:]))).astype('f4')
                            pose[3:] *= np.sign(pose[3])/np.linalg.norm(pose[3:])  # normalize quaternion
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

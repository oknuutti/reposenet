import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader, IMG_EXTENSIONS


class PoseNet(nn.Module):
    def __init__(self, arch, num_features=2048, dropout=0.0, cache_dir=None, pretrained=False):
        super(PoseNet, self).__init__()
        self.dropout = dropout
        self.pretrained = pretrained
        self.features = num_features
        self.register_buffer('target_mean', torch.zeros(7))
        self.register_buffer('target_std', torch.ones(7))

        if cache_dir is not None:
            os.environ['TORCH_MODEL_ZOO'] = cache_dir
        assert arch in models.__dict__, 'invalid model name: %s' % arch

        if arch.startswith('alexnet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # 61M params
            repl_layers = ('classifier',)

        elif arch.startswith('vgg'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # 11&13: 133M, 16: 138M, 19: 144M
            repl_layers = ('classifier',)

        elif arch.startswith('resnet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # 18: 12M, 34: 22M, 50: 26M, 101: 45M, 152: 60M
            repl_layers = ('fc',)

        elif arch.startswith('squeezenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # 1_0 & 1_1: 1.2M
            repl_layers = ('classifier',)

        elif arch.startswith('mobilenet'):
            assert not pretrained, 'pretrained model not available'
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # v2: 3.5M
            repl_layers = ('classifier',)

        elif arch.startswith('densenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained)  # 121: 8M, 169: 14M, 201: 20M, 161: 29M
            repl_layers = ('classifier',)

        elif arch.startswith('inception'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)  # v3: 27M params
            repl_layers = ('fc', 'AuxLogits.fc')

        elif arch.startswith('googlenet'):
            self.feature_extractor = getattr(models, arch)(pretrained=pretrained, aux_logits=True,
                                                           transform_input=False)  # 13M params
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

        # Translation
        self.fc_vect = nn.Linear(num_features, 3)

        # Rotation in quaternions
        self.fc_quat = nn.Linear(num_features, 4)

        # Initialization (feature_extractor already initialized in its __init__ method)
        init_modules = [(self.fc_feat, 0.01), (self.fc_vect, 0.5), (self.fc_quat, 0.01)] \
                       + [(getattr(self, 'aux_v' + str(i)), 0.5) for i in range(1, self.aux_qty + 1)] \
                       + [(getattr(self, 'aux_q' + str(i)), 0.01) for i in range(1, self.aux_qty + 1)]

        for (m, std) in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight.data)
                nn.init.normal_(m.weight.data, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        # Cost function has trainable parameters so included here
        self.cost_fn = PoseNetCriterion(learn_uncertainties=True, sx=0.0, sq=-3.0)

    def params_to_optimize(self, split=False, only_blank=False):
        # exclude all params from BatchNorm layers
        np = list(self.named_parameters(recurse=False))
        names, params = zip(*np) if len(np) > 0 else ([], [])
        for mn, m in self.named_modules():
            if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                np = list(m.named_parameters(recurse=False))
                n, p = zip(*np) if len(np) > 0 else ([], [])
                names.extend([mn+'.'+k for k in n])
                params.extend(p)

        if split:
            bias_params = []
            weight_params = []
            other_params = []
            for name, param in zip(names, params):
                if only_blank and (
                        'fc_feat' not in name and
                        'fc_quat' not in name and
                        'fc_vect' not in name and
                        'cost_fn' not in name and
                        'aux' not in name):
                    pass
                elif 'bias' in name:
                    bias_params.append(param)
                elif 'weight' in name:
                    weight_params.append(param)
                else:
                    other_params.append(param)

            return bias_params, weight_params, other_params

        return params

    def set_target_transform(self, mean, std):
        self.target_mean.data = torch.Tensor(mean)
        self.target_std.data = torch.Tensor(std)

    def extract_features(self, x):
        x_features = self.feature_extractor(x)
        x_features, *auxs = x_features if isinstance(x_features, tuple) else (x_features, )
        x_features = self.fc_feat(x_features)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return [x_features] + list(auxs)

    def forward(self, x):
        x_features, *auxs = self.extract_features(x)
        x_translations = self.fc_vect(x_features)
        x_rotations = self.fc_quat(x_features)
        x_poses = self.fix_poses(x_translations, x_rotations)

        if self.training and len(auxs) > 0:
            aux_output = []
            for i, aux in enumerate(auxs):
                av = getattr(self, 'aux_v' + str(i + 1))(aux)
                aq = getattr(self, 'aux_q' + str(i + 1))(aux)
                ap = self.fix_poses(av, aq)
                aux_output.append(ap)
            return [x_poses] + aux_output

        return x_poses

    def fix_poses(self, translations, rotations):
        translations = translations * self.target_std[:3] + self.target_mean[:3]
        rotations = rotations * self.target_std[3:] + self.target_mean[3:]
        rotations = F.normalize(rotations, p=2, dim=1)
        poses = torch.cat((translations, rotations), dim=1)
        return poses

    def cost(self, x, y):
        return self.cost_fn(x, y)


class PoseNetCriterion(nn.Module):
    def __init__(self, beta=500.0, aux_cost_coef=0.3, learn_uncertainties=False, sx=0.0, sq=-3.0):
        super(PoseNetCriterion, self).__init__()
        self.loss_fn = nn.MSELoss() if False else nn.L1Loss()

        self.aux_cost_coef = aux_cost_coef
        self.learn_uncertainties = learn_uncertainties
        if learn_uncertainties:
            self.sx = nn.Parameter(torch.Tensor([sx]))
            self.sq = nn.Parameter(torch.Tensor([sq]))
            self.beta = 1.0
        else:
            self.register_buffer('sq', torch.Tensor([0]))
            self.register_buffer('sx', torch.Tensor([0]))
            self.register_buffer('beta', torch.Tensor([beta]))

    def forward(self, all_x, y):
        if not isinstance(all_x, (list, tuple)):
            all_x = (all_x,)

        loss = 0
        for i, x in enumerate(all_x):
            coef = 1 if i == 0 else self.aux_cost_coef

            # Translation loss
            loss += coef * (torch.exp(-self.sx) * self.loss_fn(x[:, :3], y[:, :3]) + self.sx)

            # Rotation loss
            xq = torch.sign(x[:, 3]).unsqueeze(1) * x[:, 3:]  # map both hemispheres of the quaternions to a single one (y done already)
            yq = torch.sign(y[:, 3]).unsqueeze(1) * y[:, 3:]  # map both hemispheres of the quaternions to a single one (y done already)
            loss += coef * (torch.exp(-self.sq) * self.beta * self.loss_fn(xq, yq) + self.sq)

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
        self.samples, self.target_mean, self.target_std = self._load_samples()
        self.targets = [s[1] for s in self.samples]

    def _load_samples(self):
        samples = []
        paths = []
        poses = []
        label_file = os.path.join(self.root, self.label_file)
        for scene_dir in ([''] if os.path.exists(label_file) else os.listdir(self.root)):
            label_file = os.path.join(self.root, scene_dir, self.label_file)
            if os.path.exists(label_file):
                with open(label_file, 'r') as fh:
                    for line in fh.readlines():
                        row = line.strip('\r\n').split(' ')
                        if len(row) == 8:
                            pose = np.array(list(map(float, row[1:]))).astype('f4')
                            # pose[3:] *= np.sign(pose[3])/np.linalg.norm(pose[3:])  # normalize quaternion
                            if np.linalg.norm(pose) < 1000:
                                paths.append(os.path.join(self.root, scene_dir, row[0]))
                                poses.append(pose)
                            else:
                                print('inexpected meta data at %s: %s' % (scene_dir, row,))

        samples = [(paths[i], torch.tensor(pose)) for i, pose in enumerate(poses)]
        return samples, np.mean(poses, axis=0), np.std(poses, axis=0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

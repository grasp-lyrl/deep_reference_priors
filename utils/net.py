import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.functional import softmax

from copy import deepcopy
from collections import OrderedDict


class BasicBlock(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 stride: int,
                 drop_rate: float = 0.0,
                 bn_affine: bool = False,
                 bn_momentum: bool = False,
                 activate_before_residual: bool = False) -> None:
        nn.Module.__init__(self)
        # super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, affine=bn_affine,
                                  momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, affine=bn_affine,
                                  momentum=bn_momentum)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = nn.Conv2d(in_planes, out_planes,
                                      kernel_size=1, stride=stride,
                                      padding=0, bias=False)

        self.activate_before_residual = activate_before_residual

    def forward(self, x: Tensor) -> Tensor:
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self,
                 nb_layers: int,
                 in_planes: int,
                 out_planes: int,
                 block: BasicBlock,
                 stride: int,
                 drop_rate: float = 0.0,
                 bn_affine: bool = False,
                 bn_momentum: float = 0.1,
                 activate_before_residual: bool = False) -> None:
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes,
                                      nb_layers, stride, drop_rate,
                                      bn_affine, bn_momentum, activate_before_residual)

    def _make_layer(self,
                    block: BasicBlock,
                    in_planes: int,
                    out_planes: int,
                    nb_layers: int,
                    stride: int,
                    drop_rate: float,
                    bn_affine: bool,
                    bn_momentum: float,
                    activate_before_residual: bool) -> nn.Sequential:
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, bn_affine,
                                bn_momentum, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    """
    Wide-Resnet (https://arxiv.org/abs/1605.07146) for multiple tasks.
    This implementation assumes all tasks have the same number of classes. 
    See WideResNetMultiTask_v2 if you want to have multiple classes
    """
    def __init__(self,
                 depth: int,
                 widen_factor: int = 1,
                 drop_rate: float = 0.0,
                 num_cls: int = 2,
                 inp_channels: int = 3,
                 bn_affine: bool = False,
                 bn_momentum: float = 0.1) -> None:
        """
        Args:
            - depth: Depth of WRN
            - num_task: Number of tasks (number of classification layers)
            - num_cls: Number of classes for each task (same for all tasks)
            - widen_factor: The scaling factor for the number of channels
            - drop_rate: Dropout prob. that element is zeroed out
            - inp_channels: Number of channels in the input
        """
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock

        self.conv1 = nn.Conv2d(inp_channels, nChannels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1],
                                   block, 1, drop_rate, bn_affine,
                                   bn_momentum, True)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2],
                                   block, 2, drop_rate, bn_affine,
                                   bn_momentum)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3],
                                   block, 2, drop_rate, bn_affine,
                                   bn_momentum)

        # global average pooling to accomodate flexible input image sizes
        self.bn1 = nn.BatchNorm2d(nChannels[3], affine=bn_affine,
                                  momentum=bn_momentum)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Multiple linear layers, one for each task
        self.fc = nn.Linear(nChannels[3], num_cls)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and bn_affine:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d) and not bn_affine:
                if m.weight is None:
                   m.weight = nn.Parameter(torch.ones(
                       m.running_var.shape, dtype=m.running_var.dtype,
                       device=m.running_var.device), requires_grad=False)
                if m.bias is None:
                    m.bias = nn.Parameter(torch.zeros(
                        m.running_var.shape, dtype=m.running_var.dtype,
                        device=m.running_var.device), requires_grad=False)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pool(out)
        out = out.view(-1, self.nChannels)

        # Fill logits with zeros
        logits = self.fc(out)
        return logits


class ParticlesNet(nn.Module):

    def __init__(self, net, netargs, cfg):
        super(ParticlesNet, self).__init__()
        ngpus = torch.cuda.device_count()
        self.cfg = cfg
        self.particles = nn.ModuleList([net(**netargs) for i in range(cfg.ref.particles)])

        if torch.cuda.is_available():
            self.devs = ['cuda:%d' % (i % ngpus) for i in range(cfg.ref.particles)]
            self.dev = 'cuda:0'
            for i in range(len(self.particles)):
                self.particles[i].to('cuda:%d' % (i % ngpus))
        else:
            self.devs = ['cpu' for i in range(cfg.ref.particles)]
            self.dev = 'cpu'

        self.logprior = torch.nn.Parameter(torch.zeros(cfg.ref.particles).to(self.dev),
                                           requires_grad=False)

    def forward(self, x):
        nparts = self.cfg.ref.particles
        out = [self.particles[i](x.to(self.devs[i], non_blocking=True)) for i in range(nparts)]
        out = [pred.to(self.dev, non_blocking=True) for pred in out]
        out = torch.stack(out, dim=2)
        return out

    def get_prior(self):
        prior = softmax(self.logprior, dim=0)
        return prior

    def get_opt_params(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name == 'logprior' or ".bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': self.cfg.hp.wd}]


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()
        self.shadow.eval()

    @torch.no_grad()
    def update_ema(self):
        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        # shadow = decay*shadow + (1-decay)*param
        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        # Copy batch norm buffers exactly
        for name, buff in model_buffers.items():
            shadow_buffers[name].copy_(buff)

    def forward(self, inputs: Tensor, use_ema=False) -> Tensor:
        if use_ema:
            return self.shadow(inputs)
        else:
            return self.model(inputs)

    def get_opt_params(self, wd):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if 'logprior' in name or ".bn" in name:
                no_decay.append(param)
            else:
                decay.append(param)
        return [{'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': wd}]

    def get_prior(self):
        prior = softmax(self.model.logprior, dim=0)
        return prior


def create_net(cfg):
    # The Batch norm momentum is smaller than usuual.
    # This parameter is the rate at which running mean/std parameters are updated.
    resnet_args = {'depth': 28, 'widen_factor': 2,
                   'drop_rate': 0.0, 'num_cls': 10,
                   'inp_channels': 3, 'bn_affine': True,
                   'bn_momentum': 0.001}
    particles = ParticlesNet(WideResNet, resnet_args, cfg)
    net = EMA(particles, decay=0.999)
    return net



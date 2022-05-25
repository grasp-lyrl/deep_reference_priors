import math
import numpy as np

from PIL import Image
from torchvision import datasets
from torchvision import transforms
from utils.randaugment import RandAugmentMC
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_dataloaders(cfg):
    datasets = get_dataset(cfg)

    labeled_trainloader = DataLoader(
        datasets[0],
        sampler=RandomSampler(datasets[0]),
        batch_size=cfg.hp.bs,
        num_workers=cfg.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        datasets[1],
        sampler=RandomSampler(datasets[1]),
        batch_size=cfg.hp.bs * cfg.ref.μ,
        num_workers=cfg.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        datasets[2],
        sampler=SequentialSampler(datasets[2]),
        batch_size=cfg.hp.bs,
        num_workers=cfg.num_workers)

    return (labeled_trainloader, unlabeled_trainloader, test_loader)


def get_dataset(cfg):
    all_transforms = get_transforms()

    # Split dataset into labeled and unlabeled dataset
    base_dataset = datasets.CIFAR10("./data/", train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs = split_dataset(cfg, base_dataset.targets)

    # Get datasets
    train_labeled_dataset = CIFAR10SSL(
        "./data/", train_labeled_idxs,
        train=True, transform=all_transforms[0])
    train_unlabeled_dataset = CIFAR10SSL(
        "./data/", train_unlabeled_idxs,
        train=True, transform=all_transforms[1])
    test_dataset = datasets.CIFAR10(
        "./data/", train=False,
        transform=all_transforms[2], download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def split_dataset(cfg, labels):
    num_classes = len(set(labels))
    label_per_class = cfg.data.num_labeled // num_classes
    labels = np.array(labels)

    # Get labaled and unlabeled indices
    labeled_idx = []
    rng = np.random.default_rng(cfg.seed)
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        idx = rng.choice(idx, label_per_class, False)
        labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)

    unlabeled_idx = np.array(range(len(labels)))
    unlabeled_idx = np.setdiff1d(unlabeled_idx, labeled_idx)

    # Duplicate indices to run dataloader till eval step
    num_expand_x = math.ceil(cfg.hp.bs * cfg.steps.updates / cfg.data.num_labeled)
    labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)

    return labeled_idx, unlabeled_idx


def get_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    transform_lab = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=4,
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform_lab, TransformUnlabeled(mean=mean, std=std), transform_val


class TransformUnlabeled(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=4,
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from collections import defaultdict
from random import random

import clip
import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms

from classifiers import zeroshot_classifier


def main():
    # Path for ImageNet
    data_path = "/your/data/path"

    train_features_path = "./imagenet_f_train.pt"
    train_targets_path = "./imagenet_t_train.pt"

    test_features_path = "./imagenet_f_test.pt"
    test_targets_path = "./imagenet_t_test.pt"

    # load_train = False
    # load_test = False

    load_train = True
    load_test = True

    search = True

    k_shot = 16

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1.17)
    parser.add_argument('--train_epoch', type=float, default=20)
    parser.add_argument('--augment_epoch', type=float, default=10)
    args = parser.parse_args()
    print(args)

    clip.available_models()
    name = 'RN50'

    model, preprocess = clip.load(name)
    model.eval()

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    print(f"Model params: {np.sum([int(np.prod(p.shape)) for p in model.parameters()])}")
    print(f"Input resolution: {input_resolution}")
    print(f"Context length: {context_length}")
    print(f"Vocab size: {vocab_size}")

    random.seed(42)
    torch.manual_seed(42)

    print(f"{len(classes)} classes, {len(templates)} templates")

    images = torchvision.datasets.ImageNet(data_path, split='val', transform=preprocess)
    loader = torch.utils.data.Dataloader(images, batch_size=64, num_workers=8, shuffle=False)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224,
                                     scale=(0.5, 1),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    train_images = torchvision.datasets.ImageNet(data_path,
                                                 split='train',
                                                 transform=train_transform)

    split_by_label_dict = defaultdict(list)
    print('Dataloading finished')

    # build kshot set
    for i in range(len(train_images.imgs)):
        split_by_label_dict[train_images.targets[i]].append(train_images.imgs[i])

    imgs = []
    targets = []

    for label, items in split_by_label_dict.items():
        imgs = imgs + random.sample(items, k_shot)
        targets = targets + [label for i in range(k_shot)]

    train_images.imgs = imgs
    train_images.targets = targets
    train_images.samples = imgs
    train_loader = torch.utils.data.Dataloader(train_images,
                                               batch_size=256,
                                               num_workers=8,
                                               shuffle=False)
    train_loader_shuffle = torch.utils.data.Dataloader(train_images,
                                                       batch_size=256,
                                                       num_workers=8,
                                                       shuffle=True)

    # get the text feature
    print('Start obtaining text features')
    zeroshot_weights = zeroshot_classifier(classes, templates, model)
    print('finish getting text features. start getting image features')

if __name__ == '__main__':
    main()

# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
from collections import defaultdict
from random import random

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets
import torchvision.transforms as transforms
from tqdm import tqdm

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

    print('Start saving training image features')

    if not load_train:

        train_images_targets = []
        train_images_features_agg = []

        with torch.no_grad():
            for augment_idx in range(args.augment_epoch):
                train_images_features = []

                print(f"Augment time: {augment_idx} / {args.augment_epoch}")
                for i, (images, targets) in enumerate(tqdm(train_loader)):
                    images = images.cuda()
                    images_features = model.encode_image(images)
                    train_images_features.append(images_features)

                    if augment_idx == 0:
                        target = target.cuda()
                        train_images_targets.append(target)

                images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
                train_images_features_agg.append(images_features_cat)

        train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(0)
        train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
        train_images_features_agg = train_images_features_agg.permute(1, 0)

        train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()
        torch.save(train_images_features_agg, train_features_path)
        torch.save(train_images_targets, train_targets_path)

    else:
        train_images_features_agg = torch.load(train_features_path)
        train_images_targets = torch.load(train_targets_path)

    print("Start saving testing features")

    if not load_test:
        test_features =[]
        test_labels=[]
        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images = images.cuda()
                target = target.cuda()
                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                test_features.append(image_features)
                test_labels.append(target)
        test_features = torch.cat(test_features)
        test_labels = torch.cat(test_labels)

        torch.save(test_features, test_features_path)
        torch.save(test_labels, test_targets_path)

    else:
        test_features = torch.load(test_features_path)
        test_labels = torch.load(test_targets_path)

if __name__ == '__main__':
    main()

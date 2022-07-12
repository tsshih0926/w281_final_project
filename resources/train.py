import os
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from collections import OrderedDict

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random_split

from utils import *
from wide_resnet import WideResNet
from auto_augment import AutoAugment, Cutout


class MavocDataset(Dataset):
    """MAVOC dataset containing pairs of EO + SAR images"""

    def __init__(self, root_dir, eo_transform=None, sar_transform=None, composite_transform=None):
        """
        Constructs a dataframe reference to each sample pair and label in the MAVOC "training" dataset
        :param root_dir (string): path to the folder containing the class folders
        :param transform (callable, optional): transforms to be applied on paired EO + SAR samples
        """

        self.root_dir = root_dir
        self.img_pairs = defaultdict(lambda: dict)

        self.eo_transform = eo_transform
        self.sar_transform = sar_transform
        self.composite_transform = composite_transform

        eo_prefix = "EO" # case sensitive!
        sar_prefix = "SAR"
        class_folders = os.listdir(self.root_dir)
        if '.DS_Store' in class_folders: class_folders.remove('.DS_Store') # for macOS

        # populate a dictionary with image_id number, the eo and sar file path, and class label
        for class_dir in class_folders:
            for file in os.listdir(os.path.join(self.root_dir, class_dir)):

                id = int(re.findall("\d+", file)[0]) # grab the integer (image_id) in filename and use as key
                label = int(class_dir)
                img_path = os.path.join(self.root_dir,class_dir, file)

                if id in self.img_pairs.keys():
                    if file.startswith(eo_prefix):
                        self.img_pairs[id].update({"eo_img": img_path})
                    if file.startswith(sar_prefix):
                        self.img_pairs[id].update({"sar_img": img_path})
                else:
                    if file.startswith(eo_prefix):
                        self.img_pairs[id] = {"eo_img": img_path, "sar_img":None, "label":label}
                    if file.startswith(sar_prefix):
                        self.img_pairs[id] = {"eo_img": None,"sar_img": img_path, "label":label}

        # convert the dict to a dataframe so that we can properly index into the dataset with __getitem__
        self.img_labels_df = pd.DataFrame.from_dict(self.img_pairs, orient='index')
        self.img_labels_df.reset_index(inplace=True)
        self.img_labels_df = self.img_labels_df.rename(columns = {'index':'image_id'})

    def __getitem__(self, idx):
        df = self.img_labels_df
        eo_img_path = df.loc[df.index[idx], "eo_img"]
        sar_img_path = df.loc[df.index[idx], "sar_img"]

        eo_image = read_image(eo_img_path) # reads jpeg or png into a 3d RGB or grayscale tensor (uint8 in [0,255])
        sar_image = read_image(sar_img_path)
        # composite_image = None

        # resize EO to 31x31
        eo_image = F.resize(eo_image, (31, 31))
        # resize SAR to 55x55 (slightly vary in size)
        sar_image = F.resize(sar_image, (55, 55))

        if self.eo_transform:
            eo_image = self.eo_transform(eo_image)
        if self.sar_transform:
            sar_image = self.sar_transform(sar_image)

        # TODO: add composite_image to return statement
        # if self.composite_transform:
        #     composite_image = self.composite_transform(eo_image, sar_image)

        label = df.loc[df.index[idx], "label"]

        return eo_image, sar_image, label

    def __len__(self):
        return len(self.img_labels_df.index)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--dataset', default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--depth', default=28, type=int)
    parser.add_argument('--width', default=10, type=int)
    parser.add_argument('--cutout', default=False, type=str2bool)
    parser.add_argument('--auto-augment', default=False, type=str2bool)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float)
    parser.add_argument('--milestones', default='60,120,160', type=str)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    args = parser.parse_args()

    return args


def train(args, train_loader, model, criterion, optimizer, epoch, scheduler=None):
    losses = AverageMeter()
    scores = AverageMeter()

    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # from original paper's appendix
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc = accuracy(output, target)[0]

        losses.update(loss.item(), input.size(0))
        scores.update(acc.item(), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            scores.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc', scores.avg),
    ])

    return log


def main():
    args = parse_args()

    if args.name is None:
        args.name = '%s_WideResNet%s-%s' %(args.dataset, args.depth, args.width)
        if args.cutout:
            args.name += '_wCutout'
        if args.auto_augment:
            args.name += '_wAutoAugment'

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    # data loading code
    if args.dataset == 'Mavoc':
        transform_train = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())
        if args.cutout:
            transform_train.append(Cutout())
        transform_train.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        dataset = MavocDataset(root_dir="train_images", eo_transform=transform_train, sar_transform=transform_train)
        train_set, test_set = random_split(torch_dataset, [205640, 58754])
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=4,
            shuffle=True,
            num_workers=0)

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=8)

        num_classes = 10


    # create model
    model = WideResNet(args.depth, args.width, num_classes=num_classes)
    model = model.cuda()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer,
            milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'
    ])

    best_acc = 0
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch+1, args.epochs))

        scheduler.step()

        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        val_log = validate(args, test_loader, model, criterion)

        print('loss %.4f - acc %.4f - val_loss %.4f - val_acc %.4f'
            %(train_log['loss'], train_log['acc'], val_log['loss'], val_log['acc']))

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_log['loss'],
            train_log['acc'],
            val_log['loss'],
            val_log['acc'],
        ], index=['epoch', 'lr', 'loss', 'acc', 'val_loss', 'val_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/%s/log.csv' %args.name, index=False)

        if val_log['acc'] > best_acc:
            torch.save(model.state_dict(), 'models/%s/model.pth' %args.name)
            best_acc = val_log['acc']
            print("=> saved best model")


if __name__ == '__main__':
    main()

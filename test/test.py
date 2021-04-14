from utils.misc import AverageMeter, accuracy
import argparse

import torch
from model import resnet_generic
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys
sys.path.append('.')

parser = argparse.ArgumentParser(description='PyTorch EBConv ImageNet Testing')
parser.add_argument(
    'data',
    metavar='DIR',
    default='/data-sets/imagenet-eureka/imagenet256/',
    help='path to dataset')
parser.add_argument('--path-to-model', type=str, default=None)
args = parser.parse_args()

model = resnet_generic(
    num_experts=4,
    binary=True,
    activation=torch.nn.functional.softmax,
    use_se=True,
    groups=[
        4,
        8,
        8,
        16],
    expansion=[
        2,
        2,
        2,
        2],
    structure=[
        1,
        2,
        6,
        2],
    num_classes=1000,
    add_g_layer=True,
    decompose_downsample=4)
model.cuda()
print(model)
res = model.load_state_dict(torch.load(args.path_to_model))

# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=256, shuffle=False,
    num_workers=8, pin_memory=True)

# Validation loop
model.eval()

print(model)

top1 = AverageMeter('Acc@1', ':6.2f')
top5 = AverageMeter('Acc@5', ':6.2f')

with torch.no_grad():
    for i, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

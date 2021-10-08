
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision as tv
import numpy as np
from models.resnet import ResNet18
from models.resnet_reg import ResNet18 as ResNet18_REG
from time import time
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model, load_model, LabelDict, load_model_base
import copy
import matplotlib
import argparse
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
import seaborn as sns

import torchvision.models as models

parser = argparse.ArgumentParser(description='Frequency Analysis for normal model.')
parser.add_argument('--data_root', default='./data',
                    help='the directory to save the dataset')
# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=8./255.0,
                    help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=0.003,
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=20,
                    help='maximum iteration when generating adversarial examples')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                    help='the type of the perturbation (linf or l2)')
parser.add_argument('--checkpoint', default='./output/resnet18_std.pth',
                    help='save path of the model')


args = parser.parse_args()
# label_dict = LabelDict(args.dataset)

te_dataset = tv.datasets.CIFAR10(args.data_root,
                                train=False,
                                transform=tv.transforms.ToTensor(),
                                download=True)

te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


model_path_resnet18 = args.checkpoint

model_adv = ResNet18()
model_std = ResNet18()
model_clip = ResNet18()

load_model(model_adv, model_path_resnet18)
load_model(model_std, model_path_resnet18)
load_model(model_clip, model_path_resnet18)

feat_result_input = []
feat_result_output = []
feat_result_input_std = []
feat_result_output_std = []
grad_result = []

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.k,
                  step_size=args.alpha,
                  random=True):
    # out, _ = model(X, _eval=True)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output = model(X_pgd, _eval=True)
            loss = nn.CrossEntropyLoss()(output, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd


def get_features_hook(module, data_input, data_output):
    feat_result_input.append(data_input)
    feat_result_output.append(data_output)


def get_features_hook_std(module, data_input, data_output):
    feat_result_input_std.append(data_input)
    feat_result_output_std.append(data_output)

with torch.no_grad():
    if torch.cuda.is_available():
        model_std.cuda()
        model_adv.cuda()
        model_clip.cuda()

#########################################################################################
    # register handler
    relu_index = 0
    model_adv.layer4.register_forward_hook(get_features_hook)
    model_clip.layer4.register_forward_hook(get_features_hook_std)

###########################################################################################
    statis_results_robust = 0.
    statis_results_std = 0.
    magnitude_robust = 0.
    magnitude_std = 0.
    batch_idx = 0
    count_samples = 0
    for data, label in te_loader:
        # clear feature blobs
        feat_result_input.clear()
        feat_result_output.clear()
        feat_result_input_std.clear()
        feat_result_output_std.clear()

        data, label = tensor2cuda(data), tensor2cuda(label)
        adv_data1 = _pgd_whitebox(model_std, data, label)

        output1 = model_adv(adv_data1, _eval=True)
        output2 = model_std(adv_data1, _eval=True)
        output3 = model_clip(data, _eval=True)
        # output1 = model(data, _eval=True)
        # output2 = model_std(data, _eval=True)
        pred1 = torch.max(output1, dim=1)[1]
        pred2 = torch.max(output2, dim=1)[1]
        pred3 = torch.max(output3, dim=1)[1]

        # select adv data

        # idx = torch.tensor(np.arange(data.shape[0]))
        idx = np.where(label.cpu().numpy() == np.array([0]*data.shape[0]))[0]
        idx = torch.tensor(idx)
        count_samples += len(idx)

        test_std = 0.
        test_robust = 0.
        if len(idx) > 0:
            feat1 = feat_result_input_std[0]
            feat2 = feat_result_output_std[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape
            max_value = torch.max(feat_out, dim=1, keepdim=True)[0]
            threshold = 1e-2 * max_value
            mask = feat_out > threshold.expand(N, C)
            count_activate = torch.sum(mask, dim=0).view(C)
            feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
            for k in range(C):
                if feat_mean_magnitude[k] != 0:
                    feat_mean_magnitude[k] = feat_mean_magnitude[k] / count_activate[k].float()
            count_activate = count_activate.cpu().numpy()
            feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
            if batch_idx == 0:
                statis_results_std = count_activate
                magnitude_std = feat_mean_magnitude
            else:
                statis_results_std = statis_results_std + count_activate
                magnitude_std = (magnitude_std + feat_mean_magnitude) / 2


        # print(statis_results_std)
        if len(idx) > 0:
            feat1 = feat_result_input[0]
            feat2 = feat_result_output[0]
            feat_in = feat1[0][idx]
            feat_out = feat2[idx]
            if len(feat_out.shape) == 4:
                N, C, H, W = feat_out.shape
                feat_out = feat_out.view(N, C, H * W)
                feat_out = torch.mean(feat_out, dim=-1)
            N, C = feat_out.shape
            max_value = torch.max(feat_out, dim=1, keepdim=True)[0]
            threshold = 1e-2 * max_value
            mask = feat_out > threshold.expand(N, C)
            count_activate = torch.sum(mask, dim=0).view(C)
            feat_mean_magnitude = torch.sum(feat_out, dim=0).view(C)
            for k in range(C):
                if feat_mean_magnitude[k] != 0:
                    feat_mean_magnitude[k] = feat_mean_magnitude[k] / count_activate[k].float()
            count_activate = count_activate.cpu().numpy()
            feat_mean_magnitude = feat_mean_magnitude.cpu().numpy()
            if batch_idx == 0:
                statis_results_robust = count_activate
                magnitude_robust = feat_mean_magnitude
            else:
                statis_results_robust = (statis_results_robust + count_activate)
                magnitude_robust = (magnitude_robust + feat_mean_magnitude) / 2
        batch_idx += 1

#################################################################################
    print('Count Samples', count_samples)
    statis_results_robust = np.array(statis_results_robust)
    statis_results_std = np.array(statis_results_std)
    res = np.concatenate([statis_results_robust, statis_results_std], axis=0)
    if os.path.exists('./Frequency') == False:
        os.makedirs('./Frequency')
    np.save('./Frequency/cifar10_adv_layer4_class0_1e-2.npy', res)

    magnitude_results_robust = np.array(magnitude_robust)
    magnitude_results_std = np.array(magnitude_std)
    res = np.concatenate([magnitude_results_robust, magnitude_results_std], axis=0)
    if os.path.exists('./Magnitude') == False:
        os.makedirs('./Magnitude')
    np.save('./Magnitude/cifar10_adv_layer4_class0_1e-2.npy', res)





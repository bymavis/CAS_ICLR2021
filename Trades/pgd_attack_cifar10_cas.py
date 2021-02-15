from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from models.resnet_cas import ResNet18 as ResNet18_REG
from models.VGG16_cas import VGG16_REG
from models.wideresnet_cas import WideResNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR PGD Attack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')
parser.add_argument('--random',
                    default=True,
                    help='random initialization for PGD')
parser.add_argument('--model-path',
                    default='./checkpoint/model-resnet18-last.pt',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')
parser.add_argument('--beta', default=2.0,
                    help='regularization')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
if args.dataset == 'cifar-10':
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
if args.dataset == 'svhn':
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN('./data', split='test', download=True,
                                  transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False)


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  random=True,
                  beta=1.):
    # out, _ = model(X, _eval=True)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output, extra_output = model(X_pgd, y, _eval=True)
            loss = nn.CrossEntropyLoss()(output, y)
            extra_loss = 0.
            for output in extra_output:
                extra_loss += nn.CrossEntropyLoss()(output, y)
            extra_loss /= len(extra_output)
            loss += beta * extra_loss
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    adv_output, _ = model(X_pgd, _eval=True)
    err_pgd = (adv_output.data.max(1)[1] != y.data).float().sum()
    return err_pgd

def _cw_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size,
                  beta=args.beta):
    # out = model(X)
    # err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            output, extra_output = model(X_pgd, y, _eval=True)
            correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
            tmp1 = torch.argsort(output, dim=1)[:, -2:]
            new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
            wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
            loss = - F.relu(correct_logit-wrong_logit)
            extra_loss = 0.
            for output in extra_output:
                correct_logit = torch.sum(torch.gather(output, 1, (y.unsqueeze(1)).long()).squeeze())
                tmp1 = torch.argsort(output, dim=1)[:, -2:]
                new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
                wrong_logit = torch.sum(torch.gather(output, 1, (new_y.unsqueeze(1)).long()).squeeze())
                extra_loss += - F.relu(correct_logit - wrong_logit)
            extra_loss /= len(extra_output)
            loss += beta * extra_loss

        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    output, _ = model(X_pgd, _eval=True)
    err_pgd = (output.data.max(1)[1] != y.data).float().sum()
    return err_pgd


def _pgd_blackbox(model_target,
                  model_source,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps,
                  step_size=args.step_size):
    out = model_target(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if args.random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model_source(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    err_pgd = (model_target(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd black-box: ', err_pgd)
    return err, err_pgd


def clean_test(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        out, _ = model(X, _eval=True)
        err = (out.data.max(1)[1] != y.data).float().sum()
        err_total += err
    print('Clean Acc: ', 1 - err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_pgd20(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('PGD20 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_various_epsilon(model, device, test_loader):
    model.eval()
    test_epsilon = [2.0/255, 4.0/255, 8.0/255, 16.0/255, 32.0/255]
    for epsilon in test_epsilon:
        robust_err_total = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_robust = _pgd_whitebox(model, X, y, random=True, beta=args.beta, epsilon=epsilon, step_size=epsilon/10.0)
            robust_err_total += err_robust
        print('PGD20 epsilon %.4f robust_acc: '% (epsilon), 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_pgd100(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=args.epsilon,
                  num_steps=100, step_size=args.step_size, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('PGD100 robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_fgsm(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _pgd_whitebox(model, X, y, epsilon=8/255.0, num_steps=1,
                                step_size=8/255.0, random=True, beta=args.beta)
        robust_err_total += err_robust
    print('FGSM robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))

def eval_adv_test_whitebox_cw(model, device, test_loader):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_robust = _cw_whitebox(model, X, y, beta=args.beta)
        robust_err_total += err_robust
    print('cw robust_acc: ', 1 - robust_err_total / len(test_loader.dataset))


def eval_adv_test_blackbox(model_target, model_source, device, test_loader):
    """
    evaluate model by black-box attack
    """
    model_target.eval()
    model_source.eval()
    robust_err_total = 0
    natural_err_total = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_blackbox(model_target, model_source, X, y)
        robust_err_total += err_robust
        natural_err_total += err_natural
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)


def main():

    if args.white_box_attack:
        # white-box attack
        print('pgd white-box attack')
        # model = WideResNet().to(device)
        model = ResNet18_REG(num_classes=10).to(device)
        # model = VGG16_REG(n_classes=10).to(device)
        model.load_state_dict(torch.load(args.model_path))
        clean_test(model, device, test_loader)
        # eval_adv_various_epsilon(model, device, test_loader)
        eval_adv_test_whitebox_pgd20(model, device, test_loader)
        eval_adv_test_whitebox_fgsm(model, device, test_loader)
        eval_adv_test_whitebox_cw(model, device, test_loader)
        # eval_adv_test_whitebox_pgd100(model, device, test_loader)

    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)


if __name__ == '__main__':
    main()

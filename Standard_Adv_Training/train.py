import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision as tv
from time import time
from models.VGG16_cas import VGG16_REG
from models.resnet_cas import ResNet18 as ResNet18_REG
from models.wideresnet_cas import WideResNet
from attack import FGSM_REG
from utils import makedirs, create_logger, tensor2cuda, numpy2cuda, evaluate, save_model
import argparse
# import time

parser = argparse.ArgumentParser(description='Video Summarization')
parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
                    help='what behavior want to do: train | valid | test | visualize')
parser.add_argument('--dataset', default='cifar-10', help='use what dataset')
parser.add_argument('--data_root', default='./data',
                    help='the directory to save the dataset')
parser.add_argument('--log_root', default='log',
                    help='the directory to save the logs or other imformations (e.g. images)')
parser.add_argument('--model_root', default='checkpoint', help='the directory to save the models')
parser.add_argument('--load_checkpoint', default='./model/default/model.pth')
parser.add_argument('--affix', default='default', help='the affix for the save folder')

# parameters for generating adversarial examples
parser.add_argument('--epsilon', '-e', type=float, default=8. / 255.0,
                    help='maximum perturbation of adversaries (4/255=0.0157)')
parser.add_argument('--alpha', '-a', type=float, default=2. / 255.0,
                    help='movement multiplier per iteration when generating adversarial examples (2/255=0.00784)')
parser.add_argument('--k', '-k', type=int, default=10,
                    help='maximum iteration when generating adversarial examples')

parser.add_argument('--batch_size', '-b', type=int, default=128, help='batch size')
parser.add_argument('--max_epoch', '-m_e', type=int, default=200,
                    help='the maximum numbers of the model see a sample')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-1, help='learning rate')
parser.add_argument('--weight_decay', '-w', type=float, default=1e-4,
                    help='the parameter of l2 restriction for weights')

parser.add_argument('--gpu', '-g', default='0', help='which gpu to use')
parser.add_argument('--n_eval_step', type=int, default=100,
                    help='number of iteration per one evaluation')
parser.add_argument('--n_checkpoint_step', type=int, default=4000,
                    help='number of iteration to save a checkpoint')
parser.add_argument('--n_store_image_step', type=int, default=4000,
                    help='number of iteration to save adversaries')
parser.add_argument('--perturbation_type', '-p', choices=['linf', 'l2'], default='linf',
                    help='the type of the perturbation (linf or l2)')

parser.add_argument('--adv_train', action='store_true')

parser.add_argument('--beta', '-beta', type=float, default=2,
                    help='channel regularization')

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

args = parser.parse_args()


class Trainer():
    def __init__(self, args, logger, attack):
        self.args = args
        self.logger = logger
        self.attack = attack

    def standard_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, False)

    def adversarial_train(self, model, tr_loader, va_loader=None):
        self.train(model, tr_loader, va_loader, True)

    def train(self, model, tr_loader, va_loader=None, adv_train=False):
        args = self.args
        logger = self.logger

        best_adv_acc = 0.

        opt = torch.optim.SGD(model.parameters(), args.learning_rate,  momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                         milestones=[75, 90],
                                                         gamma=0.1)
        _iter = 0


        for epoch in range(1, args.max_epoch + 1):
            begin_time = time()
            scheduler.step()
            for data, label in tr_loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                if adv_train:
                    # When training, the adversarial example is created from a random
                    # close point to the original data point. If in evaluation mode,
                    # just start from the original data point.
                    adv_data = self.attack.perturb(data, label, 'mean', True, args.beta)
                    output, class_wise_output = model(adv_data, y=label, _eval=False)
                    loss = F.cross_entropy(output, label)
                    channel_reg_loss = 0.
                    for extra_output in class_wise_output:
                        channel_reg_loss += F.cross_entropy(extra_output, label)
                    channel_reg_loss = channel_reg_loss / len(class_wise_output)
                    loss = loss + args.beta * channel_reg_loss
                else:
                    output, class_wise_output = model(data, y=label, _eval=False)
                    loss = F.cross_entropy(output, label)
                    channel_reg_loss = 0.
                    for extra_output in class_wise_output:
                        channel_reg_loss += F.cross_entropy(extra_output, label)
                    channel_reg_loss = channel_reg_loss / len(class_wise_output)
                    loss = loss + args.beta * channel_reg_loss

                opt.zero_grad()
                loss.backward()
                opt.step()

                if _iter % args.n_eval_step == 0:

                    if adv_train:
                        with torch.no_grad():
                            stand_output, _ = model(data, _eval=True)
                        pred = torch.max(stand_output, dim=1)[1]

                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    else:

                        adv_data = self.attack.perturb(data, label, 'mean', False)

                        with torch.no_grad():
                            adv_output, _ = model(adv_data, _eval=True)
                        pred = torch.max(adv_output, dim=1)[1]
                        adv_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                        pred = torch.max(output, dim=1)[1]
                        # print(pred)
                        std_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy()) * 100

                    logger.info('epoch: %d, iter: %d, spent %.2f s, tr_loss: %.3f' % (
                        epoch, _iter, time() - begin_time, loss.item()))

                    logger.info('standard acc: %.3f %%, robustness acc: %.3f %%' % (
                        std_acc, adv_acc))

                if _iter % args.n_checkpoint_step == 0:
                    file_name = os.path.join(args.model_folder, 'checkpoint_last.pth')
                    save_model(model, file_name)

                _iter += 1
            epoch_cost = time()
            print('cost:', epoch_cost - begin_time)
            # assert False

            if va_loader is not None:
                t1 = time()
                va_acc, va_adv_acc = self.test(model, va_loader, True, False)
                va_acc, va_adv_acc = va_acc * 100.0, va_adv_acc * 100.0

                t2 = time()
                logger.info('\n' + '=' * 20 + ' evaluation at epoch: %d iteration: %d ' % (epoch, _iter) \
                            + '=' * 20)
                logger.info('test acc: %.3f %%, test adv acc: %.3f %%, spent: %.3f' % (
                    va_acc, va_adv_acc, t2 - t1))
                logger.info('=' * 28 + ' end of evaluation ' + '=' * 28 + '\n')

                if va_adv_acc > best_adv_acc:
                    file_name = os.path.join(args.model_folder, 'checkpoint_best_adv.pth')
                    save_model(model, file_name)
                    best_adv_acc = va_adv_acc



    def test(self, model, loader, adv_test=False, use_pseudo_label=False):
        # adv_test is False, return adv_acc as -1

        total_acc = 0.0
        num = 0
        total_adv_acc = 0.0

        with torch.no_grad():
            for data, label in loader:
                data, label = tensor2cuda(data), tensor2cuda(label)

                output, _ = model(data,  _eval=True)

                pred = torch.max(output, dim=1)[1]
                te_acc = evaluate(pred.cpu().numpy(), label.cpu().numpy(), 'sum')

                total_acc += te_acc
                num += output.shape[0]

                if adv_test:
                    # use predicted label as target label
                    with torch.enable_grad():
                        adv_data = self.attack.perturb(data,
                                                       pred if use_pseudo_label else label,
                                                       'mean',
                                                       False,
                                                       args.beta)

                    adv_output, _ = model(adv_data, _eval=True)

                    adv_pred = torch.max(adv_output, dim=1)[1]
                    adv_acc = evaluate(adv_pred.cpu().numpy(), label.cpu().numpy(), 'sum')
                    total_adv_acc += adv_acc
                else:
                    total_adv_acc = -num

        return total_acc / num, total_adv_acc / num


def main(args):
    save_folder = '%s_%s' % (args.dataset, args.affix)

    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, args.todo, 'info')

    print_args(args, logger)

    # model = VGG16_REG(n_classes=10)
    model = ResNet18_REG(num_classes=10)

    attack =FGSM_REG(model,
                    args.epsilon,
                    args.alpha,
                    min_val=0,
                    max_val=1,
                    max_iters=args.k,
                    _type=args.perturbation_type)

    if torch.cuda.is_available():
        model.cuda()

    trainer = Trainer(args, logger, attack)

    if args.todo == 'train':
        transform_train = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                 (4, 4, 4, 4), mode='constant', value=0).squeeze()),
            tv.transforms.ToPILImage(),
            tv.transforms.RandomCrop(32),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
        ])
        tr_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=True,
                                         transform=transform_train,
                                         download=True)

        tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # evaluation during training
        te_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=False,
                                         transform=tv.transforms.ToTensor(),
                                         download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        trainer.train(model, tr_loader, te_loader, args.adv_train)
    elif args.todo == 'test':
        te_dataset = tv.datasets.CIFAR10(args.data_root,
                                         train=False,
                                         transform=tv.transforms.ToTensor(),
                                         download=True)

        te_loader = DataLoader(te_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        checkpoint = torch.load(args.load_checkpoint)
        model.load_state_dict(checkpoint)

        std_acc, adv_acc = trainer.test(model, te_loader, adv_test=True, use_pseudo_label=False)

        print("std acc: %.4f, adv_acc: %.4f" % (std_acc * 100, adv_acc * 100))

    else:
        raise NotImplementedError


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='l_inf',
                cr_beta=1.0):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                output_adv, extraoutput_adv = model(x_adv, y, _eval=True)
                output_nat, extraoutput_nat = model(x_natural, y, _eval=True)
                loss_kl = criterion_kl(F.log_softmax(output_adv, dim=1),
                                       F.softmax(output_nat, dim=1))
                channel_reg_loss = 0.
                for i in range(len(extraoutput_adv)):
                    channel_reg_loss += criterion_kl(F.log_softmax(extraoutput_adv[i], dim=1),
                                       F.softmax(extraoutput_nat[i], dim=1))
                channel_reg_loss /= len(extraoutput_adv)
                loss_kl += cr_beta * channel_reg_loss
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits, extra_output_nat = model(x_natural, y, _eval=False)

    loss_natural = F.cross_entropy(logits, y)
    extra_loss_natural = 0.
    for output in extra_output_nat:
        extra_loss_natural += F.cross_entropy(output, y)
    extra_loss_natural /= len(extra_output_nat)
    loss_natural += cr_beta * extra_loss_natural

    output_adv, extra_output_adv = model(x_adv, y, _eval=False)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(output_adv, dim=1),
                                                    F.softmax(logits, dim=1))
    extra_loss_robust = 0.
    for i in range(len(extra_output_adv)):
        extra_loss_robust += (1.0 / batch_size) * criterion_kl(F.log_softmax(extra_output_adv[i], dim=1),
                                                        F.softmax(extra_output_nat[i], dim=1))
    extra_loss_robust /= len(extra_output_adv)
    loss_robust += cr_beta * extra_loss_robust

    loss = loss_natural + beta * loss_robust
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def mart_loss(model,
              x_natural,
              y,
              optimizer,
              step_size=0.007,
              epsilon=0.031,
              perturb_steps=10,
              beta=6.0,
              distance='l_inf',
              cr_beta=1.0):
    kl = nn.KLDivLoss(reduction='none')
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_output, extra_outputs = model(x_adv, y, _eval=True)
                loss_ce = F.cross_entropy(adv_output, y)
                extra_ce = 0.
                for output in extra_outputs:
                    extra_ce += F.cross_entropy(output, y)
                extra_ce /= len(extra_outputs)
                loss_ce += cr_beta * extra_ce
            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()

    logits, extra_outputs_nat = model(x_natural, y, _eval=False)

    logits_adv, extra_outputs_robust = model(x_adv, y, _eval=False)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y)

    extra_adv_probs = []
    extra_loss_adv = 0.
    for output in extra_outputs_robust:
        adv_probs_extra = F.softmax(output, dim=1)
        # print(adv_probs_extra)
        extra_adv_probs.append(adv_probs_extra)
        tmp1 = torch.argsort(adv_probs_extra, dim=1)[:, -2:]
        new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
        # extra_loss_adv += F.cross_entropy(output, y)
        # print(adv_probs_extra[y], adv_probs_extra[new_y])
        # extra_loss_adv += 1/len(extra_outputs_robust) * (1.0 / batch_size) * (F.cross_entropy(output, y) + F.nll_loss(torch.log(1.0001 - adv_probs_extra + 1e-12), new_y))
        extra_loss_adv +=  (F.cross_entropy(output, y) + F.nll_loss(torch.log(1.0001 - adv_probs_extra + 1e-12), new_y))
    extra_loss_adv /= len(extra_outputs_robust)
    loss_adv += cr_beta * extra_loss_adv

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    extra_loss_robust = 0.
    for idx, output in enumerate(extra_outputs_nat):
        nat_probs_extra = F.softmax(output, dim=1)
        true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()
        extra_loss_robust += (1.0 / batch_size) * torch.sum(
            torch.sum(kl(torch.log(extra_adv_probs[idx] + 1e-12), nat_probs_extra), dim=1) * (1.0000001 - true_probs))
    extra_loss_robust /= len(extra_outputs_nat)
    loss_robust += cr_beta * extra_loss_robust

    loss = loss_adv + float(beta) * loss_robust

    return loss

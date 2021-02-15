import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.Sequential(*layers)


class Conv_FC_Block(tnn.Module):

    def __init__(self, chann_in, chann_out, k_size, p_size):
        super(Conv_FC_Block, self).__init__()
        self.conv = tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size)
        self.bn = tnn.BatchNorm2d(chann_out)
        self.relu = tnn.ReLU()
        self.channel_out = chann_out
        self.fc = tnn.Linear(self.channel_out, 10)

    def forward(self, x, label=None):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        fc_in = torch.mean(out.view(out.shape[0], out.shape[1], -1), dim=-1)
        fc_out = self.fc(fc_in.view(out.shape[0], out.shape[1]))
        if self.training:
            N, C, H, W = out.shape
            mask = self.fc.weight[label, :]
            out = out * mask.view(N, C, 1, 1)
        else:
            N, C, H, W = out.shape
            pred_label = torch.max(fc_out, dim=1)[1]
            mask = self.fc.weight[pred_label, :]
            out = out * mask.view(N, C, 1, 1)
        return out, fc_out

def vgg_conv_block_channel_reg(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [Conv_FC_Block(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    # layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return tnn.ModuleList(layers)

def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


class VGG16_REG(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16_REG, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block_channel_reg([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.max_pooling = tnn.MaxPool2d(kernel_size=2, stride=2)

        # FC layers
        self.layer6 = vgg_fc_layer(1 * 1 * 512, 512)
        self.layer7 = vgg_fc_layer(512, 512)

        # Final layer
        self.layer8 = tnn.Linear(512, n_classes)

    def forward(self, x, y=None, _eval=False):
        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()
        extra_output = []

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        for layer in self.layer5:
            out, layer5_out = layer(out, y)
            extra_output.append(layer5_out)
        vgg16_features = self.max_pooling(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return out, extra_output


def get_features_hook(module, data_input, data_output):
    fmap_block.append(data_output)
    input_block.append(data_input)


def get_grads_hook(module, grad_input, grad_output):
    grad_result.append(grad_output[0].data.cpu().numpy())


# Created by 112020333002191(CH.Zhang)
# End-to-End Video-based Cocktail Causal Container for Blood Pressure Estimation and Glucose Prediction
import copy
import random
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torch.autograd import Function
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from train_detail import train_detail
import math

class ImageGenerator(object):
    def __init__(self, generator, transition):
        super(ImageGenerator, self).__init__()
        self._generator = generator.eval()
        self._transition = transition.eval()

    def __call__(self, device=device):
        low = torch.ones(1, dtype=torch.float32) * -1
        high = torch.ones(1, dtype=torch.float32)
        self._Uniform = torch.distributions.uniform.Uniform(low=low, high=high)

        loc = torch.zeros(1, dtype=torch.float32)
        scale = torch.ones(1, dtype=torch.float32)
        self._Normal = torch.distributions.normal.Normal(loc=loc, scale=scale)

        with torch.no_grad():
            s_current = self._Uniform.sample(sample_shape=(1, 7))
            s_current = torch.squeeze(s_current)
            s_current = s_current.reshape((1, ) + s_current.shape)
            s_next, _ = self._transition(s_current)
            z = self._Normal.sample(sample_shape=(1, 4))
            z = torch.squeeze(z)
            z = z.reshape((1, ) + z.shape)
            x = torch.cat((z, s_current, s_next), dim=1)
            x = torch.reshape(x, shape=x.shape + (1, 1))
            o = self._generator(x)
            return torch.chunk(o, 2, dim=1)

def gaussian_nll(x, mean, ln_var, reduce='sum'):

    x_prec = torch.exp(-ln_var)
    x_diff = x - mean
    x_power = (x_diff * x_diff) * x_prec * -0.5
    loss = (ln_var + math.log(2 * math.pi)) / 2 - x_power
    if reduce == 'sum':
        return loss.sum()
    elif reduce == 'mean':
        return loss.mean()
    else:
        return


class Classifier(nn.Module):
    def __init__(self, s_dim=7):
        super(Classifier, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(
            in_channels=2, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv5 = nn.Conv1d(
            in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv5(x)
        return F.sigmoid(x)


class Generator(nn.Module):
    def __init__(self, inchannels=7 + 7 + 4, out_color_num=1, out_frame_num=2, batch_size=32, device=device):
        super(Generator, self).__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=inchannels, out_channels=512, kernel_size=3, stride=1),
            nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True))
        out_channels = out_color_num * out_frame_num
        self.deconv5 = nn.ConvTranspose1d(in_channels=128, out_channels=out_channels, kernel_size=1, padding=1,
                                          stride=1)

    def forward(self, s_bp, s_bg, z):
        x = torch.cat((z, s_bp, s_bg), dim=1)
        x = torch.reshape(x, shape=x.shape + (1, 1))
        x = torch.unsqueeze(x.squeeze(), dim=2)
        assert x.shape[2] == 1
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv5(x)
        x = torch.tanh(x)
        o_current, o_next = torch.chunk(x, 2, dim=1)
        return o_current, o_next


class Discriminator(nn.Module):
    def __init__(self, in_color_num=1, in_frame_num=2, batch_size=32, device=device):
        super(Discriminator, self).__init__()
        in_channels = in_color_num * in_frame_num
        self.conv1 = nn.Sequential(nn.Conv1d(
            in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1, stride=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.one_labels = self._generate_one_labels(shape=(batch_size, 1), device=device)
        self.zero_labels = self._generate_zero_labels(shape=(batch_size, 1), device=device)

    def _generate_one_labels(self, shape, device=device):
        labels = torch.ones(size=shape, dtype=torch.float32)
        labels = torch.autograd.Variable(labels)
        labels = labels.to(device=device)
        return labels
    def _generate_zero_labels(self, shape, device=device):
        labels = torch.zeros(size=shape, dtype=torch.float32)
        labels = torch.autograd.Variable(labels)
        labels = labels.to(device=device)
        return labels

    def forward(self, x, **kwargs):
        if "real" in kwargs:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv5(x)
            x = self.pool(x)
            x = torch.unsqueeze(torch.squeeze(x), dim=1)
            x = F.binary_cross_entropy_with_logits(x, self.one_labels)
            return x
        if "fake" in kwargs:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv5(x)
            x = self.pool(x)
            x = torch.unsqueeze(torch.squeeze(x), dim=1)
            x = F.binary_cross_entropy_with_logits(x, self.zero_labels)
            return x


class Posterior(nn.Module):
    def __init__(self, in_color_num=1, in_frame_num=1, c_dim=7, batch_size=32, device=device):
        super(Posterior, self).__init__()
        in_channels = in_color_num * in_frame_num
        self.conv1 = nn.Sequential(nn.Conv1d(
            in_channels=in_channels, out_channels=128, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(
            in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv1d(
            in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1), nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv1d(
            in_channels=512, out_channels=128, kernel_size=1, stride=1), nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.conv_mu = nn.Conv1d(in_channels=128, out_channels=c_dim, kernel_size=1, stride=1)
        self.conv_var = nn.Conv1d(in_channels=128, out_channels=c_dim, kernel_size=1, stride=1)

    def forward(self, x, s):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv5(x)
        mu = self.conv_mu(x)
        mu = torch.squeeze(mu)
        ln_var = self.conv_var(x)
        ln_var = torch.squeeze(ln_var)
        return gaussian_nll(x=s, mean=mu, ln_var=ln_var, reduce='mean')



class Transition(nn.Module):
    def __init__(self, s_dim=7, batch_size=32, device=device):
        super(Transition, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_features=s_dim, out_features=64), nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True))
        self.linear2 = nn.Sequential(nn.Linear(in_features=64, out_features=64),  nn.BatchNorm1d(64),
                                     nn.ReLU(inplace=True))
        self.linear3 = nn.Linear(in_features=64, out_features=s_dim * 2)
        low = torch.ones(1, dtype=torch.float32) * -1
        high = torch.ones(1, dtype=torch.float32)
        self._Uniform = torch.distributions.uniform.Uniform(low=low, high=high)
        loc = torch.zeros(1, dtype=torch.float32)
        scale = torch.ones(1, dtype=torch.float32)
        self._Normal = torch.distributions.normal.Normal(loc=loc, scale=scale)
        self.s_shape = (batch_size, 7)
        self.z_shape = (batch_size, 4)
        self.device = device

    def forward(self, x=None, s=None, **kwargs):
        if "sample" in kwargs:
            s_current = self._Uniform.sample(sample_shape=self.s_shape)
            s_current = torch.squeeze(s_current.to(self.device))
            x = self.linear1(s_current)
            x = self.linear2(x)
            x = self.linear3(x)
            s_next, _ = torch.chunk(x, 2, dim=-1)
            z = self._Normal.sample(sample_shape=self.z_shape)
            z = torch.squeeze(z.to(self.device))
            return s_current, s_next, z

        if "nll" in kwargs:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            mu, ln_var = torch.chunk(x, 2, dim=-1)
            return -gaussian_nll(x=s, mean=mu, ln_var=ln_var, reduce='mean')

        if "loss" in kwargs:
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            _, ln_var = torch.chunk(x, 2, dim=-1)
            var = torch.exp(ln_var)
            x = var[0].reshape(len(var[0]), -1)
            l2_norm = (x * x).sum()
            return l2_norm.mean()
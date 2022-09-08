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
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import _cfg, Mlp, Block
import math
from model.cnn import cnn


class MHConvAttention(nn.Module):

    def __init__(self, num_heads=4, embedding_dim=64, out_dim=2048, window_size=5):
        super().__init__()
        self.nh = num_heads
        self.window_size = window_size
        self.pos_embed_dim = embedding_dim // self.nh
        self.rel_pos_embed = nn.Parameter(torch.zeros(self.pos_embed_dim, window_size, window_size))
        nn.init.normal_(self.rel_pos_embed, std=0.02)
        self.qkv_conv = nn.Conv2d(embedding_dim, 3 * embedding_dim, 1, bias=False)
        self.cpe = nn.Conv2d(embedding_dim, embedding_dim, 3, 1, 1, bias=False, groups=embedding_dim)
        self.qkv_conv.apply(weights_init_kaiming)
        self.cpe.apply(weights_init_kaiming)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.output_dim = out_dim
        self.ChannelAttention = nn.Sigmoid()
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, src):
        _, C, H, W = src.shape
        scaling_factor = (C // self.nh) ** -0.5
        feature_raw = src
        src = self.cpe(src) + src
        qkv = self.qkv_conv(src)
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        k = rearrange(k, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        v = rearrange(v, "b (nh hd) h w -> (b nh) hd h w", nh=self.nh)
        content_lambda = torch.einsum("bin, bon -> bio", k.flatten(-2).softmax(-1), v.flatten(-2))
        content_output = torch.einsum("bin, bio -> bon", q.flatten(-2) * scaling_factor, content_lambda)
        content_output = rearrange(content_output, "bnh hd (h w) -> bnh hd h w", h=H)
        position_lambda = F.conv2d(
            v,
            weight=rearrange(self.rel_pos_embed, "D Mx My -> D 1 Mx My"),
            padding=self.window_size // 2,
            groups=self.pos_embed_dim,
        )
        position_output = q * position_lambda
        result = content_output + position_output
        result1 = rearrange(result, "(b nh) hd h w -> b (nh hd) h w", nh=self.nh)
        X_1 = feature_raw
        X_1 = self.avg(X_1)
        X_1 = self.conv1d(X_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        X_1 = self.ChannelAttention(X_1)
        X_1 = X_1.expand_as(feature_raw)
        result2 = feature_raw.mul(X_1)
        result = self.out(torch.cat([result1, result2], dim=1))
        return result

class Fnet(nn.Module):
    def __init__(self, N, dhidden, sig_size=32, patch_size=8, in_chans=1536, embed_dim=1024):
        super().__init__()
        self.dense = nn.Linear(dhidden, dhidden)
        self.conv = nn.Conv2d(1, 128, 1)
        self.LayerNorm1 = nn.LayerNorm(dhidden)
        self.feedforward = nn.Linear(dhidden, dhidden)
        self.LayerNorm2 = nn.LayerNorm(dhidden)


    def forward(self, x):
        x = rearrange(x, 's b c->b s c')
        x_fft = torch.real(torch.fft.fft(torch.fft.fft(x).T).T)
        x = self.LayerNorm1(x + x_fft)
        x_ff = self.feedforward(x)
        x = self.LayerNorm2(x + x_ff)
        x = self.dense(x)
        x = rearrange(x, 'b s c->s b c')
        return x


class Subtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(Subtractor, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(
            in_channels=in_channels, out_channels=512, kernel_size=1), nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv1d(
            in_channels=512, out_channels=128, kernel_size=1), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Conv1d(
            in_channels=128, out_channels=1, kernel_size=1), nn.BatchNorm1d(1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.classifier(x)
        x = F.tanh(x)
        return x

def CrossEntropy(outputs, targets):
    log_softmax_outputs = F.log_softmax(outputs, dim=1)
    softmax_targets = F.softmax(targets, dim=1)

    return -(log_softmax_outputs*softmax_targets).sum(dim=1).mean()


class ccdn_bg(nn.Module):
    def __init__(self, backbone1d, backbone2d, backbone1d_bp, backbone2d_bp, num_classes, num_classes_bp,
                 task_input_1d, task_input_2d, in_channel1d, in_channel2d):
        super(ccdn_bg, self).__init__()

        self.input_features_1d = in_channel1d
        self.input_features_2d = in_channel2d
        self.input_size_2d = task_input_2d
        self.input_size_1d = task_input_1d
        self.num_classes = num_classes
        self.num_classes_bp = num_classes_bp
        self.substractor_feature_1d = 1536
        self.classifier_feature_img = 380
        self.classifier_feature = 192
        self.middle_feature = 100
        self.last_feature = 64
        self.fc_bg = nn.Sequential(nn.Linear(self.classifier_feature, self.num_classes))
        self.fc_soft_bg = nn.Sequential(nn.Linear(self.input_features_1d, self.num_classes)                                 )
        self.fc_bp = nn.Sequential(nn.Linear(self.classifier_feature, self.num_classes_bp))
        self.fc_bp_img = nn.Linear(self.middle_feature, self.num_classes_bp)
        self.fc_bp_sig = nn.Linear(self.last_feature, self.num_classes_bp)
        self.fc_bg_img = nn.Linear(self.middle_feature, self.num_classes)
        self.fc_bg_sig = nn.Linear(self.last_feature, self.num_classes)
        self.fc_soft_bp = nn.Sequential(nn.Linear(self.input_features_1d, self.num_classes_bp)
                                        )
        self.subtractor = Subtractor(in_channels=self.substractor_feature_1d)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.avg_fuse = nn.AdaptiveMaxPool2d(1)
        self.bi_classifier_img = nn.Sequential(nn.Linear(self.input_features_2d, self.middle_feature),
                      nn.BatchNorm1d(self.middle_feature, affine=True),
                      nn.ReLU(inplace=True))
        self.bi_classifier_sig = nn.Sequential(
            nn.Linear(self.input_features_1d, self.last_feature),
                                nn.BatchNorm1d(self.last_feature, affine=True),
                                nn.ReLU(inplace=True))
        self.bi_classifier_img_bp = nn.Sequential(nn.Linear(self.input_features_2d, self.middle_feature),
                                                  nn.BatchNorm1d(self.middle_feature, affine=True),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(self.middle_feature, self.num_classes_bp))
        self.bi_classifier_sig_bp = nn.Sequential(
                      nn.Linear(self.input_features_1d, self.last_feature),
                      nn.BatchNorm1d(self.last_feature, affine=True),
                      nn.ReLU(inplace=True),
                      nn.Linear(self.last_feature, self.num_classes_bp),
                      )

        self.transformerEncoder_bp = Fnet(1, self.classifier_feature)
        self.transformerEncoder_bg = Fnet(1, self.classifier_feature)
        self.classifier_bp = nn.Sequential(nn.Conv1d(self.substractor_feature_1d, self.substractor_feature_1d, 1, 1),
                                           nn.BatchNorm1d(self.substractor_feature_1d), nn.ReLU(inplace=True))
        self.classifier_bg = nn.Sequential(nn.Conv1d(self.substractor_feature_1d, self.substractor_feature_1d, 1, 1),
                                           nn.BatchNorm1d(self.substractor_feature_1d), nn.ReLU(inplace=True))
        self.drop_bg = nn.Dropout(0.1)
        self.drop_bp = nn.Dropout(0.1)
        self.MHSA_img_fuse_bp = nn.Sequential(nn.Conv2d(self.input_features_2d, self.classifier_feature, 1, 1),
                                           nn.BatchNorm2d(self.classifier_feature), nn.ReLU(inplace=True))
        self.MHSA_img_fuse_bg = nn.Sequential(nn.Conv2d(self.input_features_2d, self.classifier_feature, 1, 1),
                                           nn.BatchNorm2d(self.classifier_feature), nn.ReLU(inplace=True))

        self.MHSA_img_bp = MHConvAttention(embedding_dim=self.input_features_2d, out_dim=self.input_features_2d)
        self.MHSA_img_bg = MHConvAttention(embedding_dim=self.input_features_2d, out_dim=self.input_features_2d)
        self.transformer_bp = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.classifier_feature, nhead=3), 2)
        self.transformer_bg = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.classifier_feature, nhead=3), 2)
        self.feature_extractor_2d = backbone2d.encoder
        self.feature_extractor_1d = backbone1d.encoder
        self.feature_extractor_1d_de = backbone1d.decoder
        self.feature_extractor_bp_1d_de = backbone1d_bp.decoder


    def forward(self, x_img, x_sig, **kwargs):
        x_img_original = self.feature_extractor_2d(x_img)
        if 'val' in kwargs:
            x_sig_feature = self.feature_extractor_1d(x_sig)
        else:
            x_sig_feature = self.feature_extractor_1d(x_sig, aug=True)
        x_sig_original = self.feature_extractor_1d_de(self.classifier_bg(x_sig_feature))
        x_sig_out = self.bi_classifier_sig(x_sig_original)
        x_img_out = self.avg(self.MHSA_img_bg(x_img_original))
        x_img_out = torch.flatten(x_img_out, 1)
        x_img_out = self.bi_classifier_img(x_img_out)
        x_img_original_bp = x_img_original
        x_sig_feature_bp = x_sig_feature
        x_sig_original_bp = self.feature_extractor_bp_1d_de(self.classifier_bp(x_sig_feature_bp))
        x_bp_sig_out = self.bi_classifier_sig_bp(x_sig_original_bp)
        x_bp_img_out = self.avg(self.MHSA_img_bp(x_img_original_bp))
        x_bp_img_out = torch.flatten(x_bp_img_out, 1)
        x_bp_img_out = self.bi_classifier_img_bp(x_bp_img_out)
        gan_1 = self.subtractor(F.normalize(x_sig_feature))
        gan_2 = self.subtractor(F.normalize(x_sig_feature_bp))
        x_sig = F.normalize(x_sig_original)
        x_sig = F.sigmoid(torch.flatten(x_sig, 1))
        x_img = self.avg_fuse(self.MHSA_img_fuse_bg(F.normalize(x_img_original)))
        x_img = F.sigmoid(torch.flatten(x_img, 1))
        out = torch.cat((x_img.unsqueeze(0), x_sig.unsqueeze(0)), dim=0)
        out = torch.mean(self.transformer_bg(self.transformerEncoder_bg(out)), dim=0)
        pseudo_label = self.fc_soft_bg(out)
        x_bp_fusion = self.fc_bg(x_sig_original)
        ce_bg = 0.01 * torch.clamp(F.relu(CrossEntropy(x_bp_fusion, pseudo_label)), min=0., max=10.)
        x_sig = F.normalize(x_sig_original_bp)
        x_sig = F.sigmoid(torch.flatten(x_sig, 1))
        x_img = self.avg_fuse(self.MHSA_img_fuse_bp(F.normalize(x_img_original_bp)))
        x_img = F.sigmoid(torch.flatten(x_img, 1))
        out = torch.cat((x_img.unsqueeze(0), x_sig.unsqueeze(0)), dim=0)
        out = torch.mean(self.transformer_bp(self.transformerEncoder_bp(out)), dim=0)
        pseudo_label = self.fc_soft_bp(out)
        x_bp_fusion = self.fc_bp(x_sig_original_bp)
        ce_bp = 0.01 * torch.clamp(F.relu(CrossEntropy(x_bp_fusion, pseudo_label)), min=0., max=10.)
        return (ce_bg, ce_bp), (x_img_out, x_sig_out), (x_bp_img_out, x_bp_sig_out), (gan_1, gan_2)


def CCDN(
        backbone1d,
        backbone2d,
        backbone1d_bp,
        backbone2d_bp,
        num_classes=28,
        num_classes_bp=2,
        task_input_1d=1800,
        task_input_2d=4,
        in_channel1d=192,
        in_channel2d=512):
    model_bg = ccdn_bg(backbone1d, backbone2d, backbone1d_bp, backbone2d_bp, num_classes, num_classes_bp, task_input_1d, task_input_2d,
                       in_channel1d, in_channel2d)

    return model_bg

import timm
from torchvision import models
from model.parnet import *
from model.stnet import *
from model.cnn import cnn
from model.model import CCDN
from model.gan import Discriminator, Generator, Posterior, Transition


def weights_init_kaiming(m,  scale=1):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0.0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
# Network_name = train_opt_2.model

def select_network_2D(network, num_classes=28, img_size=None, pre_train=True):
    if network == 'Res18':
        model = timm.create_model('resnet18', num_classes=0, global_pool='', pretrained=pre_train, in_chans=3)
        return model
    elif network == 'STnet':
        model = swin_tiny_patch4_window7_224(num_classes, in_chans=18)
        return model
    elif network == 'Rexnet':
        model = timm.create_model('rexnet_100', num_classes=0, global_pool='', pretrained=pre_train, in_chans=3)
        return model
    elif network == 'Contra':
        model = CNNEncoder(num_class=num_classes, n_dim=256)
        return model

def select_network_1D(network, num_classes=28):
    if network == 'cnn':
        model = cnn(n_class=num_classes, in_channels=6, channels=512, embd_dim=192)
        return model


def select_network(network1d, network2d, num_classes=28, num_classes_bp=3, batch_size=1,
                   pre_train=False, weigh_1d=None,weigh_2d=None):
    network1d_bg = select_network_1D(network1d, num_classes)
    network2d_bg = select_network_2D(network2d, num_classes)
    network1d_bp = select_network_1D(network1d, num_classes_bp)
    network2d_bp = select_network_2D(network2d, num_classes_bp)
    model = CCDN(network1d_bg, network2d_bg, network1d_bp, network2d_bp,  num_classes=num_classes, num_classes_bp=num_classes_bp)
    discriminator = Discriminator(batch_size=batch_size)
    generator = Generator(batch_size=batch_size)
    posterior = Posterior(batch_size=batch_size)
    transition = Transition(batch_size=batch_size)
    return model, discriminator, generator, posterior, transition

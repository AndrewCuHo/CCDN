import cv2
from datasets import VideoDataSet
import os
import random
import time
from model.utils import select_network_1D, select_network_2D, select_network, weights_init_kaiming
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import pickle
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from common.misc import mkdir_p
from sklearn.preprocessing import OneHotEncoder
from loss import *
from train_detail import train_detail
from torch.nn import functional as F
from tqdm import tqdm
import warnings
from common.ema import ModelEMA
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import torchaudio

train_opt = train_detail().parse()
Weight_path_1d = train_opt.weight_path_1d
Weight_path_2d = train_opt.weight_path_2d
Num_classes = train_opt.num_classes
Size_height = train_opt.input_height
Size_weight = train_opt.input_weight
Model1d = train_opt.model1d
Model2d = train_opt.model2d
Checkpoint = train_opt.checkpoints
Resume = train_opt.resume
Loss = train_opt.loss
Num_epochs = train_opt.num_epochs
Batch_size = train_opt.batch_size
Freeze = train_opt.freeze
Init_lr = train_opt.init_lr
Lr_scheduler = train_opt.lr_scheduler
Step_size = train_opt.step_size
Multiplier = train_opt.multiplier
Total_epoch = train_opt.total_epoch
Alpha = train_opt.alpha
Gamma = train_opt.gamma
Re = train_opt.re
ManualSeed = train_opt.manualSeed
torch.manual_seed(ManualSeed)
torch.cuda.manual_seed_all(ManualSeed)
np.random.seed(ManualSeed)
random.seed(ManualSeed)
torch.backends.cudnn.deterministic = True
OutpuDir = train_opt.out
UnP = train_opt.UnlabeledPercent / 100
PThreshold = train_opt.Distrib_Threshold
Un_lamda = train_opt.Balance_loss
IF_GPU = train_opt.IF_GPU
IF_TRAIN = train_opt.IF_TRAIN
SelectData = train_opt.SelectData
Is_Visual = train_opt.Is_Visual
Is_EMA = train_opt.use_ema
Is_Mix = train_opt.use_mixup
Is_Fintuning = train_opt.use_fintuning
Is_public = train_opt.is_public
Iteration_max = train_opt.iteration
K_folds = train_opt.k_folds
Num_classes_bp = train_opt.num_classes_bp
Saved_root = 'I:/PPG/checkpoints_no_kd'
OutpuDir = 'I:/PPG/log_no_kd'
filename=OutpuDir + '/training.log'

class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)
        self.num_matri = num
        self.positi = nn.ReLU()

    def forward(self, *x):
        loss_sum = 0
        loss_indentify = torch.ones(self.num_matri)
        for i, loss in enumerate(x):
            if i == 0:
                loss_sum += self.positi(0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
                loss_indentify[i] = self.positi(0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
            else:
                loss_sum += self.positi(0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))
                loss_indentify[i] = self.positi(0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2))

class CB_loss(nn.Module):

    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma, device):
        super(CB_loss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def forward(self, preds, truth):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(truth, self.no_of_classes).float()

        weights = torch.tensor(weights, device=self.device).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == "mse":
            pred = preds.softmax(dim=1)
            cb_loss = (F.smooth_l1_loss(input=pred, target=labels_one_hot, reduce=False, reduction='sum',
                                        beta=1.25) * weights).sum()
        return cb_loss





def train():

    model, discriminator, generator, posterior, transition = \
        select_network(Model1d, Model2d, num_classes=Num_classes, num_classes_bp=Num_classes_bp, batch_size = Batch_size,
                           pre_train=False, weigh_1d=Weight_path_1d, weigh_2d=Weight_path_2d)

    train_dataset = VideoDataSet(
            root_path=videos_root,
            sequence_path=sequence_root,
            list_file=train_annotation_file,
            come_file=com_root,
            transform=None,
            mode='1D+2D',
            combine=True,
            phase="Train"
        )

    val_dataset = VideoDataSet(
        root_path=videos_root,
        sequence_path=sequence_root,
        list_file=val_annotation_file,
        come_file=com_root,
        transform=None,
        mode='1D+2D',
        combine=True,
        phase="Val"
    )

    awl = AutomaticWeightedLoss(2)

    lr_list = []
    cnt = 0
    train_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=Batch_size,
        sampler=train_subsampler,
        drop_last=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        sampler=val_subsampler
    )

    for epoch in tqdm(range(start_epoch + 1, Num_epochs)):
        for ims, labels, ims_1d, labels_bp in train_dataloader:
            model.train()
            discriminator.train()
            generator.train()
            posterior.train()
            transition.train()

            if IF_GPU:
                target = labels.cuda()
                target_ohe = (F.one_hot(target.to(torch.int64), num_classes=Num_classes)).float()
                target = target.long()

                target_bp = labels_bp.cuda()
                target_ohe_bp = (F.one_hot(target_bp.to(torch.int64), num_classes=Num_classes_bp)).float()
                target_bp = target_bp.long()
            else:
                target = labels.cpu()
                target_ohe = (F.one_hot(target.to(torch.int64), num_classes=Num_classes)).float()
                target = target.long()

                target_bp = labels_bp.cpu()
                target_ohe_bp = (F.one_hot(target_bp.to(torch.int64), num_classes=Num_classes_bp)).float()
                target_bp = target_bp.long()
            if IF_GPU:
                input_ob = ims.cuda()
                input_ob_1d = ims_1d.cuda()
            else:
                input_ob = ims.cpu()
                input_ob_1d = ims_1d.cpu()

            kd_loss, output_bg, output_bp, gan_out = model(input_ob, input_ob_1d)

            real_loss = discriminator(torch.cat((gan_out[0], gan_out[1]), dim=1),
                                      real=True)
            s_bp, s_bg, z = transition(sample=True)
            fake_o_bp, fake_o_bg = generator(s_bp, s_bg, z)
            fake_d_loss = discriminator(
                torch.cat((torch.autograd.Variable(fake_o_bg), torch.autograd.Variable(fake_o_bp)),
                          dim=1), fake=True)
            d_loss = real_loss + fake_d_loss
            fake_g_loss = discriminator(torch.cat((fake_o_bg, fake_o_bp), dim=1), real=True)
            q_bp_nll = posterior(fake_o_bp, s_bp)
            q_bg_nll = posterior(fake_o_bg, s_bg)
            t_pll = transition(s_bp, s_bg, nll=True)
            transition_loss = transition(s_bp, loss=True)
            mutual_information_loss = q_bp_nll + q_bg_nll + t_pll
            gpt_loss = fake_g_loss + 0.1 * mutual_information_loss + 0.1 * transition_loss
            loss_gan = 0.01 * (d_loss + gpt_loss)
            _, pred_img = output_bg[0].max(1)
            _, pred_sig = output_bg[1].max(1)
            criterion_bp_img = AM_Softmax(d=model.middle_feature, num_classes=Num_classes_bp, use_gpu=IF_GPU)
            criterion_bp_sig = AM_Softmax(d=model.last_feature, num_classes=Num_classes_bp, use_gpu=IF_GPU)
            criterion_correlation = Neg_Pearson()
            loss_weight_bg = np.asarray([10, 10, 10, 10, 10, 10, 10, 10, 10,
                                         20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                                         10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                         5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                         2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            criterion = CB_loss(samples_per_cls=loss_weight_bg, no_of_classes=61, loss_type='mse', beta=0.75,
                                gamma=2.0, device=torch.device('cuda'))
            loss_img_bg = (criterion(output_bg[0], target))
            loss_sig_bg = (criterion(output_bg[1], target))
            correlation_loss_bg = torch.clamp(0.01 * criterion_correlation(output_bg[0], output_bg[1]),
                                              max=0.1, min=0.)
            loss_bg = (loss_img_bg + loss_sig_bg) + correlation_loss_bg
            loss_img_bp = criterion_bp_img(output_bp[0], target_bp, model.fc_bp_img)
            loss_sig_bp = criterion_bp_sig(output_bp[1], target_bp, model.fc_bp_sig)
            loss_bp = (loss_img_bp + loss_sig_bp).mean()
            loss, loss_identify_weight = awl(loss_bg, loss_bp)
            loss = loss_gan + loss
            loss_identify = np.asarray([(kd_loss[0] + kd_loss[1]).data.cpu().numpy(),
                                        loss_bg.data.cpu().numpy(),
                                        loss_bp.data.cpu().numpy(),
                                        correlation_loss_bg,
                                        0])
            optimizer.zero_grad()
            optimizer_gan.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_gan.step()



if __name__ == '__main__':

    # args = parser.parse_args()
    if not os.path.isdir(OutpuDir):
        mkdir_p(OutpuDir)
    if not os.path.isdir(Saved_root):
        mkdir_p(Saved_root)
    train()
# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
from core.utils import angular_error_degress_np, import_shortcut
import torchvision
from torchvision.models.vgg import make_layers, VGG
try:
    from torchvision.models.vgg import cfgs
except:
    from torchvision.models.vgg import cfg as cfgs
import numpy as np

import time

# Multi-Hypothesis model

#Multi-head attention
class MHSA(nn.Module):
    def __init__(self, n_dims=1, width=14, height=14):
        super(MHSA, self).__init__()

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        # self.mutihead = nn.MultiheadAttention(,1)

        self.rel_h = nn.Parameter(torch.randn([1, n_dims, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, n_dims, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, C, -1)
        k = self.key(x).view(n_batch, C, -1)
        v = self.value(x).view(n_batch, C, -1)

        content_content = torch.bmm(q.permute(0, 2, 1), k)

        content_position = (self.rel_h + self.rel_w).view(1, C, -1).permute(0, 2, 1)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(n_batch, C, width, height)

        return out

class MHAT_block(nn.Module):
    def __init__(self, input_channel, n_features):
        super(MHAT_block, self).__init__()

        BoTNet_layers = []
        BoTNet_layers.append(nn.Conv2d(input_channel, n_features, kernel_size=1))
        # BoTNet_layers.append(nn.BatchNorm2d(n_features))
        BoTNet_layers.append(nn.ReLU(inplace=True))

        BoTNet_layers.append(MHSA(n_dims=n_features))
        # BoTNet_layers.append(nn.BatchNorm2d(n_features))
        BoTNet_layers.append(nn.ReLU(inplace=True))
        
        BoTNet_layers.append(nn.Conv2d(n_features, input_channel, kernel_size=1))
        # BoTNet_layers.append(nn.BatchNorm2d(input_channel))
        # BoTNet_layers.append(nn.ReLU(inplace=True))

        self.MHAT_block_layer = nn.Sequential(*BoTNet_layers)

    def forward(self, x):
        return F.relu(x + self.MHAT_block_layer(x))



class VggMHATFfn(VGG):

    def __init__(self, conf, pretrained=False, dropout = 0.5,
                fix_base_network = False,
                final_affine_transformation = False,
                n_fc_layers = 2, n_pointwise_conv = 1, n_features = 16,
                **kwargs):
        """Multi-Hypothesis model constructor.

        Args:
            conf (dict): Dictionary of the configuration file (json).
            pretrained (:obj:`bool`, optional): Whether used pretrained model
                on ImageNet.
            dropout (:obj:`float`): Probability of dropout.
            fix_base_network (:obj:`bool`): If true, we don't learn the first
                convolution, just use the weights from pretrained model.
            final_affine_transformation (:obj:`bool`): Whether to learn an
                affine transformation of the output of the network.
            n_fc_layers (:obj:`int`): Number of FC layers after conv layers.
            n_pointwise_conv (:obj:`int`): Number of 1x1 conv layers after
                first 3x3 convolution.
            n_features (:obj:`int`): Number of feature for the final 1x1 conv.
        """
        # super(VggMHAT, self).__init__()
        arch = conf['network']['subarch']
        # vgg11 or vgg11 with batch norm
        if arch == 'vgg11':
            super(VggMHATFfn, self).__init__(make_layers(cfgs['A']), **kwargs)#调用VGG中的构造函数，使用继承重写（定义不执行）
        elif arch == 'vgg11_bn':
            super(VggMHATFfn, self).__init__(make_layers(cfgs['A'], batch_norm=True), **kwargs)
        else:
            raise Exception('Wrong architecture')

        if pretrained:
            self.load_state_dict(model_zoo.load_url(torchvision.models.vgg.model_urls[arch]))#加载预训练参数并赋值给VGG

        # we keep only the first VGG convolution!
        self.conv1 = self.features[0]
        self.relu1 = self.features[1]

        self.amp1 = nn.AdaptiveMaxPool2d((14,14))
        self.amp2 = nn.AdaptiveAvgPool2d((14,14))

        # remove VGG features and classifier (FCs of the net)
        del self.features
        del self.classifier

        # this is used to remove all cameras not included in this setting,
        # if defined in the conf file.
        if 'cameras' in conf:
            self._cameras = conf['cameras']
        else:
            self._cameras = None

        # load candidate selection method:
        # see folder modules/candidate_selection/ for available methods.
        class_obj = import_shortcut('modules.candidate_selection',
                                    conf['candidate_selection']['name'])#python文件所在目录名，方法名（利用function转为驼峰的python文件名）
        self.candidate_selection = class_obj(conf, **conf['candidate_selection']['params'])#给参数进kmeans.__init__(),实例化
        self._final_affine_transformation = final_affine_transformation

        self.downsample_conv = nn.Conv2d(128, 64, 1)

        # then, we have N="n_pointwise_conv" number of 1x1 convs
        # 改变n_pointwise_conv，会让中间层全都是1x1卷积，并且channel（in&out）=64,最后一层输出为n_features指定的
        # BoTNet_num = 1
        BoTNet_layers = []
        for _ in range(n_pointwise_conv):
            BoTNet_layers.append(MHAT_block(64, n_features))

        self.pointwise_conv = nn.Sequential(*BoTNet_layers)



        # if this option is enabled, we don't learn the first conv weights,
        # they are copied from VGG pretrained on Imagenet (from pytorch),
        # the weights: /torch_model_zoo/vgg11-bbd30ac9.pth
        if fix_base_network:
            # do not learn any parameters from VGG
            included = ['conv1.bias', 'conv1.weight']
            for name, param in self.named_parameters():
                if name in included:
                    #print('name',name, param.shape)
                    param.requires_grad = False

        # probability of dropout
        self.dropout = nn.Dropout(dropout)

        # final FC layers: from N to 1 (probability for illuminant)
        # final_n = 1

        # N: initial feature size
        # n_fc_layers: number of FC layers
        # final_n: size of final prediction
        self.fc = self._fc_layers(64, 2, 1)
        # self.conv_out = nn.Conv2d(64,1,1)
        # self.avg_out = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(1)#按行计算
        self.logsoftmax = nn.LogSoftmax(1)#对softmax结果取对数，<0

    def _fc_layers(self, n_curr, n_fc_layers, n_final):
        # generate "n_fc_layers" FC layers,
        # initially, we have "n_features" features and,
        # we convert that to n_final
        # we reduce the number of features //2 every time
        fc_layers = []
        for fc_i in range(n_fc_layers):
            next_n = n_curr // 2
            if fc_i == n_fc_layers-1:
                next_n = n_final

            fc_layers.append(nn.Linear(n_curr, next_n, bias=True))
            if fc_i != n_fc_layers-1:
                fc_layers.append(nn.ReLU(inplace=True))

            n_curr = next_n

        return nn.Sequential(*fc_layers)

    def _fc_layers_affine(self, n_features, n_fc_layers, n_layers):
        # same as previous function but we specify the size
        # of intermediate features
        #n_layers->list, 存放中间层输出维度
        fc_layers = []
        n_curr = n_features
        for fc_i in range(n_fc_layers):
            N = n_layers[fc_i]
            fc_layers.append(nn.Linear(n_curr, N, bias=True))
            if fc_i != n_fc_layers-1:
                fc_layers.append(nn.ReLU(inplace=True))

            n_curr = N

        return nn.Sequential(*fc_layers)

    def initialize(self, illuminants = None):
        # when running the constructor, we don't have the
        # training set illuminants yet, so, we run initialize
        # to do the candidate selection
        if illuminants is not None:
            # here we save one candidate set per camera（key）
            for key in illuminants.keys():
                candidates = self.candidate_selection.initialize(illuminants[key])#将该相机的所有光源值 给到candidate)selection.k_means.py中，调用该python文件中定义的initialize方法，对可见的训练集的379个光源进行聚类
                self.register_buffer('clusters_'+key, candidates)#向模块添加持久缓冲区,例如，生成类变量self.clusters_ShiGehler,在后面forward中被调用
        else:
            # if illuminants is None, we init the candidate set with
            # default values (zeros). This is used for inference scripts,
            # we don't have the illuminants when we call this function,
            # but it does not matter because we will load it from
            # the checkpoint file.
            for key in self._cameras:
                candidates = self.candidate_selection.initialize(None)
                self.register_buffer('clusters_'+key, candidates)

        # Number of candidate illuminants
        self.n_output = candidates.shape[1]#3

        # Final affine transformation (B and G in the paper)用于调整模型对不同设备的适应性
        if self._final_affine_transformation:
            self.mult = nn.Parameter(torch.ones(1, self.n_output))#1,120
            self.add = nn.Parameter(torch.zeros(1, self.n_output))

    def _inference(self, input, gt_illuminant, sensor, candidates, do_dropout = True):
        # output logics and confs
        logits = torch.zeros(input.shape[0], candidates.shape[1]).type(input.type())#32,120

        # loop illuminants
        for i in range(candidates.shape[1]):#120
            # get illuminant "i"
            illuminant = candidates[:, i, :].unsqueeze(-1).unsqueeze(-1)#1,3,1,1
            # generate candidate image! image / illuminant
            input2 = input / illuminant#对角模型，直接除接完事了

            # avoid log(0)
            input2 = torch.log(input2 + 0.000001)#取对数值

            # first conv (from VGG)
            x = self.conv1(input2)
            x = self.relu1(x)

            x1 = self.amp1(x)
            x2 = self.amp2(x)

            x = torch.cat((x1,x2),1)

            x = self.downsample_conv(x)

            # point-wise convs (1x1)
            x = self.pointwise_conv(x)

            # x = self.conv_out(x) #64x64x1

            # global average pooling, from 64x64x128 to 1x1x128
            x = F.adaptive_avg_pool2d(x, (1, 1))

            # dropout (in paper=0.5)
            x = self.dropout(x)#32,128,1,1

            # reshape 1x1x128 -> 128
            x = x.view(x.size(0), -1)#32,128
            # FC layers: 128 -> 1
            x = self.fc(x)

            # save log-likelihood for this illuminant
            logits[:, i] = x[:, 0]#一组batch的图的第一个光源的可能性，第一列

        # apply affine transformation (G and B)
        if self._final_affine_transformation:
            # just get B and G
            mult = self.mult#1,120
            add = self.add#1,120

            # save softmax of the pre-affine output (for visualization)
            bin_probability_pre_affine = self.softmax(logits)
            # apply aff. trans.
            logits = mult*logits + add#32,120

        # softmax!
        bin_probability = self.softmax(logits)#按行求softmax，32,120

        # do linear comb. in RGB
        # candidates(1,120,3)
        illuminant_r = torch.sum(bin_probability*candidates[:, :, 0], 1, keepdim=True)#32,1
        illuminant_g = torch.sum(bin_probability*candidates[:, :, 1], 1, keepdim=True)
        illuminant_b = torch.sum(bin_probability*candidates[:, :, 2], 1, keepdim=True)
        illuminant = torch.cat([illuminant_r, illuminant_g, illuminant_b], 1)#32,3

        # l2-normalization of the vector
        norm = torch.norm(illuminant, p=2, dim=1, keepdim=True)#32,1
        illuminant = illuminant.div(norm.expand_as(illuminant))

        # save output to dictionary
        # illuminant: 最终光源输出值        bin_probability2:网络模型输出的softmax结果
        # logits:不做softmax的逻辑回归结果  candidates:该数据集的候选光源
        d = {'illuminant': illuminant, 'bin_probability2': bin_probability,
            'logits': logits, 'candidates': candidates}

        if self._final_affine_transformation:
            d["bias2"] = add
            d["gain2"] = mult
            d["bin_probability_preaffine"] = bin_probability_pre_affine#不做B,G的设备相关调整的softmax结果

        return d

    def forward(self, data, image_index=0):
        input = data['rgb']#data_key:'rgb'(32,1,3,64,64),'mask'(32,1,64,64),'illuminant'(32,3),'sensor'字典（key:'camera_name'(32),'black_level'(32),'saturation'(32),'ccm'(32,3,3)）,'path'(32),'epoch'
        if 'illuminant' in data:
            gt_illuminant = Variable(data['illuminant'])
        else:
            gt_illuminant = None
        sensor = data['sensor']#'sensor'字典,[{}],（key:'camera_name'(32),'black_level'(32),'saturation'(32),'ccm'(32,3,3)）

        # only one image
        input = input[:, image_index, ...]#32,3,64,64

        # get candidates
        candidates_cam = getattr(self, 'clusters_' + sensor[0]['camera_name'][0])#1,120,3,即，self.clusters_ShiGehler，在initial中注入到缓存中的

        # run image specific tunning of the candidate set
        # this only applies for Minkowski candidate selection
        # (modules/candidate_selection/minkowski.py)
        candidates = self.candidate_selection.run(input, candidates_cam)#do nothing: no candidate tunning for each image

        # do inference
        return self._inference(input, gt_illuminant, sensor, candidates, self.training)

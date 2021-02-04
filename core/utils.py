# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import torch
import math
import cv2
import numpy as np
import importlib
import re
import torchvision
from collections import OrderedDict

# differentiable batch log-determinant for a 2x2 matrix for pytorch
def batch_logdet2x2(matrix):
    a = matrix[:, 0, 0]
    b = matrix[:, 0, 1]
    c = matrix[:, 1, 0]
    d = matrix[:, 1, 1]
    det = a*d - b*c
    if torch.sum(det <= 10**-40) > 0:
        print('WARNING: determinant very small or negative: ', det)
    logdet = torch.log(det)
    return logdet

# angular error in degrees for numpy
def angular_error_degress_np(ill1, ill2):
    err = 180.0*angular_error_np(ill1, ill2)/math.pi#最终用于评价的角度误差，为角度制
    return err

# normalize illuminant (numpy)，光源聚类结果L2范数归一化
def normalize_illuminant(illuminant):
    norm = math.sqrt((illuminant*illuminant).sum())
    illuminant = illuminant / norm

    return illuminant

# angular error for numpy
def angular_error_np(ill1, ill2):
    dot = (ill1*ill2).sum()
    norm1 = math.sqrt((ill1*ill1).sum())
    norm2 = math.sqrt((ill2*ill2).sum())
    norm = norm1*norm2
    dot = dot / norm
    if dot > 1:
        dot = 1
    if dot < 0:
        dot = 0

    return math.acos(dot)

# angular error in degrees for numpy
def angular_error_degrees(outputs, labels):
    err = 180.0*angular_error(outputs, labels)/math.pi
    return err

# differentiable angular error for pytorch，用于训练
def angular_error_gradsafe(pred, labels, compute_acos=True, vectordim=1):
    # avoid 0 vector
    pred = pred.clamp(1e-10)#clamp上下限处理
    dot = torch.sum(torch.mul(pred, labels), dim=vectordim, keepdim=True)#对应位置元素相乘
    norm_pred = torch.norm(pred, dim=vectordim, keepdim=True)
    norm_labels = torch.norm(labels, dim=vectordim, keepdim=True)
    norm = norm_pred * norm_labels

    if compute_acos:
        ddn = dot.div(norm.expand_as(dot))
        ddn = ddn.clamp(-0.999999, 0.999999)
        err = torch.acos(ddn)#弧度制[0,π]
    else:
        err = norm - dot

    return err

# angular error for pytorch (NOT GRADIENT SAFE!!! DON'T USE DURING TRAINING)
def angular_error(outputs, labels):
    dot = torch.sum(torch.mul(outputs, labels), dim=1, keepdim=True)
    norm_outputs = torch.norm(outputs, dim=1, keepdim=True)
    norm_labels = torch.norm(labels, dim=1, keepdim=True)
    norm = norm_outputs * norm_labels
    ddn = dot.div(norm.expand_as(dot))
    ddn = ddn.clamp(0, 1)
    err = torch.acos(ddn)

    return err

# convert to numpy axis for images (numpy)
def axis_numpy(im):
    return np.moveaxis(im, 0, 2)

## convert to pytorch axis for images (numpy)
def axis_pytorch(im):
    return np.moveaxis(im, 2, 0)

# do all processing for image visualization: (numpy)
# 1. white balance
# 2. apply CCM
# 3. apply gamma correction
def convert_for_display(im_src, illum, ccm):
    im = np.copy(im_src)
    if illum is not None:
        im = apply_whitebalance(im, illum)
    im = apply_ccm(im, ccm)
    im = apply_gamma_correction(im)
    return im

# do white balance: we clip the image [0, 1] after WB (numpy)
def apply_whitebalance(im, illum):
    for channel in range(im.shape[2]):
        im[:, :, channel] /= max(illum[channel], 0.00001)

    return np.clip(im, 0.0, 1.0)

# Apply CCM to the image (numpy)
def apply_ccm(im, ccm):
    pixels = np.reshape(im, [im.shape[0]*im.shape[1], im.shape[2]])
    pixels_converted = np.matmul(pixels, ccm.T)
    im = np.reshape(pixels_converted, im.shape)
    return im

# apply SRGB gamma correction: (numpy)
# https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
def apply_gamma_correction(im):
    a = 0.055
    b = 1.055
    t = 0.0031308
    im[im < 0] = 0
    im[im > 1] = 1
    #im = (im_linear * 12.92) .* (im_linear <= t) ...
    #   + ((1 + a)*im_linear.^(1/2.4) - a) .* (im_linear > t);
    im = np.multiply(im * 12.92, im < t) + np.multiply(b*np.power(im, (1/2.4)) - a, im >= t)
    im[im < 0] = 0
    im[im > 1] = 1
    return im

# get statistics (mean, median, ...) for a vector of angular errors (numpy)
def summary_angular_errors(errors):
    errors = sorted(errors)

    #取误差的特定百分比误差
    def g(f):
        return np.percentile(errors, f * 100)

    median = np.median(errors)
    mean = np.mean(errors)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    results = OrderedDict()
    results['mean'] = mean
    results['med'] = median
    results['tri'] = trimean
    results['25'] = np.mean(errors[:int(0.25 * len(errors))])#取前25%的均值
    results['75'] = np.mean(errors[int(0.75 * len(errors)):])
    results['95'] = g(0.95)

    return results

# convert from log [g/r, g/b] to RGB (numpy)
def uv_to_rgb(uv):
    u = uv[0]
    v = uv[1]

    r = np.exp(-u)
    b = np.exp(-v)

    rgb = [r, 1.0, b]
    rgb = rgb / np.linalg.norm(rgb)

    return rgb

# convert from log [g/r, g/b] to RGB (pytorch)
def uv_to_rgb_torch(uv, inverse_uv):
    if inverse_uv:
        ill_rgb_u = torch.exp(uv[:, 0]).unsqueeze(1)
        ill_rgb_v = torch.exp(uv[:, 1]).unsqueeze(1)
    else:
        ill_rgb_u = torch.exp(-uv[:, 0]).unsqueeze(1)
        ill_rgb_v = torch.exp(-uv[:, 1]).unsqueeze(1)
    ones = torch.ones(uv.shape[0]).type(uv.type()).unsqueeze(1)

    illuminant = torch.cat([ill_rgb_u, ones, ill_rgb_v], 1)

    norm = torch.norm(illuminant, p=2, dim=1, keepdim=True)
    illuminant = illuminant.div(norm.expand_as(illuminant))

    return illuminant

# convert from RGB to log [g/r, g/b]
def rgb_to_uv(rgb, conf = None):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    log_r = np.log(r)
    log_g = np.log(g)
    log_b = np.log(b)

    if conf is not None and 'inverse_uv' in conf and conf['inverse_uv']:
        u = log_r - log_g
        v = log_b - log_g
    else:
        u = log_g - log_r
        v = log_g - log_b

    if conf is not None:
        wrapped_u = np.mod(np.round((u - conf['starting_uv']) / conf['bin_size']), conf['num_bins'])
        wrapped_v = np.mod(np.round((v - conf['starting_uv']) / conf['bin_size']), conf['num_bins'])
        wrapped = np.round(np.array([wrapped_u, wrapped_v])).astype(np.int)

        return np.array([u, v]), wrapped

    return np.array([u, v])

# check whether values for FFCC/CCC histograms are valid
def log_uv_histogram_checks(conf, illuminants):
    starting_uv = conf['starting_uv']
    bin_size = conf['bin_size']
    num_bins = conf['num_bins']

    final_uv = starting_uv + bin_size*(num_bins+1)
    min_uv = 10000
    max_uv = 0
    str_out_of_hist = 'ERROR'
    for ill in illuminants:
        uv_illuminant = rgb_to_uv(ill)
        if uv_illuminant[0] < starting_uv or uv_illuminant[0] > final_uv or uv_illuminant[1] < starting_uv or uv_illuminant[1] > final_uv:
            print(str_out_of_hist,': The illuminant ', uv_illuminant, ' is outside the histogram range')

        min_uv = min(min_uv, uv_illuminant[0])
        min_uv = min(min_uv, uv_illuminant[1])

        max_uv = max(max_uv, uv_illuminant[0])
        max_uv = max(max_uv, uv_illuminant[1])

    uv_center = (min_uv + max_uv) / 2
    gt_starting_uv = uv_center - bin_size*num_bins/2
    gt_starting_uv = round(gt_starting_uv/bin_size)*bin_size
    if abs(gt_starting_uv - starting_uv) > 0.001:
        print('INFO: Difference GT_starting_position (', gt_starting_uv, ') - starting_uv (', starting_uv, ') = ', gt_starting_uv - starting_uv)

# import class function
def import_class(module, class_name):
    module = importlib.import_module(module)#动态导入对象，此处获得的是model文件夹下vgg_classification.py中定义的类，即py文件中定义的类
    return getattr(module, class_name)#返回一个对象属性值，参数（对象，属性（字符串））

# convert CamelCase to snake_case
#转换至标准文件命名方式，例k_means,vgg_classification
def camel_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)#正则表达式，替换；变成以大写字母分割的形式，如Blacklevel_AndSaturation
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()#变成全部小写且再次分割，如blacklevel_and_saturation

# import module function
def import_shortcut(base_module, class_name):
    return import_class(base_module+'.'+camel_to_snake(class_name), class_name)#base_module为transforms等目录名，class_name为BalcklevelAndSaturation等文件名（未处理）

# create list of transforms
def create_all_transforms(worker, transform_conf, return_list = False):
    t = []
    for transform in transform_conf:#transform_conf->list,transform->字典
        name = list(transform.keys())[0]#取该字典所有key的第一个key，如'BlacklevelAndSaturation'
        args = transform[name]#取该key值对应的value，如{'saturation_scale':0.95}
        t.append(create_transform(name, worker, args))

    if return_list:
        return t

    return torchvision.transforms.Compose(t)

# create optimizer
def create_optimizer(t, model_params, opt_params = None):
    cl = import_class('torch.optim', t)

    if opt_params is None:
        return cl(model_params)
    else:
        return cl(model_params, **opt_params)

# create transform
#对transform内容，加载对应的类，并传入参数
def create_transform(t, worker, arguments):
    try:
        cl = import_shortcut('transforms', t)
    except AttributeError as e:
        cl = import_class('torchvision.transforms', t)

    if arguments is None:
        return cl(worker)
    else:
        return cl(worker, **arguments)

# find FFCC histogram configuration from transforms
def find_loguv_warp_conf(transforms):
    if transforms is None:
        return None

    from transforms.log_uv_histogram_wrap import LogUvHistogramWrap
    for t in transforms.transforms:
        if isinstance(t, LogUvHistogramWrap):#判断一个对象是否是一个已知的类型
            return t._conf

    return None

# apply black level subtraction (numpy)
def blacklevel_saturation_correct(im, sensor, saturation_scale = 1.0):
    original_dtype = im.dtype
    max_value = np.iinfo(original_dtype).max

    im = im.astype(np.float)
    im = im - sensor.black_level
    im[im < 0] = 0
    saturation = sensor.saturation - sensor.black_level
    im = im / (saturation_scale * saturation)
    im[im > 1] = 1.0
    im = (max_value*im)
    im = im.astype(original_dtype)

    return im

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import torch
from torch.utils.data.dataset import Dataset
import cv2
import numpy as np
from datasets.sensor import Sensor
import os
import sys
import time
import math

TWO_CAMERAS = False

# os.path.realpath(__file__)获取当前执行脚本的绝对路径
with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'canon5d.txt'), 'r') as f:#即打开当前脚本所在目录下的ccm目录下的canon5d.txt文件
    # numpy.loadtxt(fname, dtype=, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
    CCM_Canon5D = torch.FloatTensor(np.loadtxt(f))

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ccm', 'canon1d.txt'), 'r') as f:
    CCM_Canon1D = torch.FloatTensor(np.loadtxt(f))

class Canon5DSensor(Sensor):
    def __init__(self):
        if TWO_CAMERAS:
            super(Canon5DSensor, self).__init__(129, 3692, CCM_Canon5D, 'Canon5D')#black_level, saturation, ccm, camera_name
        else:
            super(Canon5DSensor, self).__init__(129, 3692, CCM_Canon5D, 'ShiGehler')

class Canon1DSensor(Sensor):
    def __init__(self):
        if TWO_CAMERAS:
            super(Canon1DSensor, self).__init__(0, 3692, CCM_Canon1D, 'Canon1D')
        else:
            super(Canon1DSensor, self).__init__(0, 3692, CCM_Canon1D, 'ShiGehler')

# Shi Gehler data loader: https://www2.cs.sfu.ca/~colour/data/shi_gehler/
#参数列表：
#   _rgbs                   图片完整路径列表
#   _illuminants            数据集所有光源值
#   _x/_y                   mask坐标
#   _illuminants_by_sensor  分相机存储光源值
#   _base_path              所有图像的基础路径，即paths.json中对应的'shi_gehler_all'的路径
class ShiGehler(Dataset):
    def __init__(self, subdataset, data_conf, file, cache):
        # subdataset:'all'  data_conf:paths.json中的所有内容    file:'data/shi_gehler/splits/fold1.txt'
        self._rgbs = []
        self._illuminants = []
        if TWO_CAMERAS:
            self._illuminants_by_sensor = {'Canon5D': [], 'Canon1D': []}
        else:
            self._illuminants_by_sensor = {'ShiGehler': []}
        self._x = {}
        self._y = {}
        self._base_path = data_conf['shi_gehler_'+subdataset]

        if type(file) is list:
            for f in file:
                self._read_list(os.path.join(data_conf['base'], f))
        else:
            self._read_list(os.path.join(data_conf['base'], file))

        self._cache = cache

    def get_filename(self, index):
        return self._rgbs[index]

    def get_illuminants_by_sensor(self):
        return self._illuminants_by_sensor

    def get_illuminants(self):
        return self._illuminants

    def _read_list(self, file):
        with open(file, 'r') as f:
            content = f.readlines()#foldX.txt中的所有内容
            for line in content:#取每一行
                elements = line.strip().split(' ')#元素变列表整理
                filename = elements[0]#图片名
                rgb_path = os.path.join(self._base_path, filename)#完整图片路径
                self._rgbs.append(rgb_path)#存储图片路径
                illuminant = elements[1:4]#光源r,g,b值
                illuminant = [float(e) for e in illuminant]
                length = math.sqrt(sum([e*e for e in illuminant]))#与原点的欧氏距离
                illuminant = [e / length for e in illuminant]#使用Frobenius范数归一化
                #利用图片文件名区分两种相机
                if 'IMG' in filename:
                    if TWO_CAMERAS:
                        sensor = 'Canon5D'
                    else:
                        sensor = 'ShiGehler'
                else:
                    if TWO_CAMERAS:
                        sensor = 'Canon1D'
                    else:
                        sensor = 'ShiGehler'
                self._illuminants_by_sensor[sensor].append(illuminant)#区分相机的光源值
                self._illuminants.append(illuminant)#所有光源值
                x = elements[4:8]#取出四个坐标值
                x = [float(e) for e in x]
                self._x[rgb_path] = x#坐标的存储，key值采用完整图片路径
                y = elements[8:]
                y = [float(e) for e in y]
                self._y[rgb_path] = y

    def get_rgb_by_path(self, filename):
        if 'IMG' in filename:
            sensor = Canon5DSensor()
        else:
            sensor = Canon1DSensor()
        #若已经加载过了，就直接从缓存中读取
        if self._cache.is_cached(filename):
            im, mask = self._cache.read(filename)
        else:
            im = cv2.imread(filename, -1)#(h,w,c)例(1460,2193,3),uint16
            if im is None:
                raise Exception('File not found: ' + filename)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)#颜色通道转换

            # get mask of valid pixels
            mask = np.ones(im.shape[:2], dtype = np.float32)#初始化二维mask，与输入图像等大(h,w)例(1460,2193)，初始化为1
            rc = np.array((self._x[filename], self._y[filename])).T#(4,2)
            ctr = rc.reshape((-1,1,2)).astype(np.int32)#(4,1,2)
            cv2.drawContours(mask, [ctr], 0, 0, -1)

            # set CCB pixels to zero
            # TODO: ideally, downsampling should consider the mask
            # and then, apply the mask as a final step
            im[mask == 0] = [0, 0, 0]

            # rotate 90 degrees so that all images have the same resolution，逆时针
            if im.shape[0] == 2193 or im.shape[0] == 2041:
                im = cv2.transpose(im)
                im = cv2.flip(im, flipCode=0)#沿x轴翻转
                mask = cv2.transpose(mask)
                mask = cv2.flip(mask, flipCode=0)

            #剪裁其他尺寸，舍弃最下方一行、最右5列
            if im.shape[0] == 1359:
                # 1359x2041 -> 1358x2036
                # remove last row to make it divisible by 2
                # and remove some columns to have the same
                # aspect ratio for all the dataset
                im = im[:-1, :-5, :]
                mask = mask[:-1, :-5]

            if im.shape[0] == 1460:
                # 1460x2193 -> 1460x2190
                # remove last row to make it divisible by 2
                # and remove some columns to have the same
                # aspect ratio for all the dataset
                im = im[:, :-3, :]
                mask = mask[:, :-3]

            self._cache.save(filename, (im, mask))

        im = im[np.newaxis, ...]
        mask = mask[np.newaxis, ...]
        sensor = [sensor]

        return im, mask, sensor

    def get_rgb(self, index):
        path = self._rgbs[index]

        return self.get_rgb_by_path(path)

    def __getitem__(self, index):
        filename = self._rgbs[index]

        im, mask, sensor = self.get_rgb(index)

        illuminant = np.array(self._illuminants[index], dtype=np.float32)

        dict = {'rgb': im, 'sensor': sensor, 'mask': mask,
                'illuminant': illuminant, 'path': filename}

        return dict

    def __len__(self):
        return len(self._rgbs)


if __name__ == '__main__':
    import scipy
    import scipy.io

    path = '/kgx_nfs2/daniel/src/ffcc/data/shi_gehler'
    illum = scipy.io.loadmat(os.path.join(path, 'real_illum_568.mat'))['real_rgb']
    split = scipy.io.loadmat(os.path.join(path, 'threefoldCVsplit.mat'))
    te_split = split['te_split']
    Xfiles = split['Xfiles']

    im_path = os.path.join(path, 'images')
    images = os.listdir(im_path)
    coord_dir = os.path.join(path, 'coordinates')

    images = sorted(images)

    save_path = '/kgx_nfs2/daniel/src/color_constancy/data/shi_gehler/splits/'
    for fold in range(te_split.shape[1]):
        with open(os.path.join(save_path, 'fold'+str(fold+1)+'.txt'), 'w') as fold_f:
            for i in range(te_split[0, fold].shape[1]):
                fid = te_split[0, fold][0, i]-1
                filename = Xfiles[fid, 0][0][0]
                filename = filename[:-3]+'png'
                norm_ill = illum[fid, :]
                index = 0
                good_index = -1
                for im_p in images:
                    if im_p == filename:
                        good_index = index
                    index += 1
                assert(good_index == fid)
                #norm_ill = illum[fid, :] / np.linalg.norm(illum[fid, :])
                fnme = filename[:-4]

                image_path = os.path.join(im_path, filename)
                im = cv2.imread(image_path,-1)

                with open(os.path.join(coord_dir, fnme+'_macbeth.txt')) as f:
                    coords = np.loadtxt(f)

                scale = np.flip(coords[0, :], axis=0) / im.shape[:2]

                x = np.array([coords[1, 0], coords[3, 0], coords[4, 0], coords[2, 0]]) / scale[0]
                y = np.array([coords[1, 1], coords[3, 1], coords[4, 1], coords[2, 1]]) / scale[1]

                fold_f.write("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n".format(
                            filename,
                            norm_ill[0],
                            norm_ill[1],
                            norm_ill[2],
                            x[0], x[1], x[2], x[3],
                            y[0], y[1], y[2], y[3]))

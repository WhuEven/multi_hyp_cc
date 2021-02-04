# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved. THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import json

import torch
import torch.backends.cudnn as cudnn

from core.print_logger import PrintLogger
from core.cache_manager import CacheManager
from core.crossvalidation import Fold, Crossvalidation

parser = argparse.ArgumentParser(description='Color Constancy: Cross validation')

parser.add_argument('configurationfile',
                    help='path to configuration file')
parser.add_argument('datasetsplits', help='path to text file containing all the dataset description')

parser.add_argument('-gpu', type=int, help='GPU id to use.')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--resume', action='store_true',
                    help='resume from previous execution')
parser.add_argument('--pretrainedmodel', help='path to model pretrained model file')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_fullres', action='store_true',
                    help='save full resolution prediction images')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--outputfolder', default='./output/', type=str,
                    help='path for the ouput folder. ')
parser.add_argument('--datapath', default='data/paths.json', type=str,
                    help='path to json file that specifies the directories of the datasets. ')

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # load configuration file: epochs, loss function, etc... for this experiment
    with open(args.configurationfile, 'r') as f:
        conf = json.load(f)

    # load datapath file: paths specific to the current machine
    with open(args.datapath, 'r') as f:
        data_conf = json.load(f)

    # load all splits/folds from the text file
    # 'datasetsplits' is a text file that contains a list of text files,
    # each line corresponds to one fold
    splits = None
    with open(args.datasetsplits, 'r') as f:
        splits = f.readlines()

    # the first line of 'datasetsplits', is the class name of the
    # dataset (if it is 'Nus', it will correspond to datasets/nus.py)
    splits = [e.strip() for e in splits]
    args.dataset = splits[0] # dataset: e.g. Nus, Cube, Mate20...
    splits = splits[1:] # CV folds

    # get subdataset from filename
    subdataset = os.path.basename(args.datasetsplits).replace('.txt', '')
    args.subdataset = subdataset

    # remove previous results
    output_dir = os.path.join(args.outputfolder, 'checkpoint', args.dataset, subdataset, conf['name'])
    if not args.evaluate and not args.resume:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

    # create output folder
    os.makedirs(output_dir, exist_ok=True)
    args.outputfolder = output_dir

    # copy configuration file to output folder
    shutil.copy(args.configurationfile, os.path.join(output_dir, os.path.basename(args.configurationfile)))

    # we overwrite the stdout and stderr (standard output and error) to
    # files in the output directory
    sys.stdout = PrintLogger(os.path.join(output_dir, 'stdout.txt'), sys.stdout)
    sys.stderr = PrintLogger(os.path.join(output_dir, 'stderr.txt'), sys.stderr)

    # init the cache manager: this caches images from the dataset
    # to avoid reading them more than once
    cache = CacheManager(conf)

    folds = []#存放多个FoldX.txt分割的Fold类，n个txt则有n个Fold对象
    #下方len==3
    for fold in range(len(splits)):
        # This is the format for cross validation files:
        # DatasetClass
        # Fold1.txt [ValFold1.txt] [TestFold1.txt]
        # Fold2.txt [ValFold2.txt] [TestFold2.txt]
        # ...
        # 1. if there's only FoldX.txt, we treat this as independent folds (they should be disjoint),
        #    for each CV fold, the training will be the amalgamation of the rest of the folds and FoldX.txt
        #    will be the validation fold. Example: data/p30/noccb.txt
        # 2. if there's FoldX.txt and ValFoldX.txt, the training fold will be FoldX.txt and ValFoldX.txt will
        #    be the validation folds.
        # 3. TestFoldX.txt is assumed to be the same as the validation fold unless it is specified.
        #    Validation results are shown during training, and Test results are shown at the end.
        split_list = splits[fold].split(' ')#取第一行，即单个FoldX.txt，下方len==1
        if len(split_list) == 1:
            valfile = splits[fold]#当前行fold做验证集
            trainfiles = list(splits)
            trainfiles.remove(valfile)#剩余行fold做训练集
            testfile = valfile
        elif len(split_list) == 2:
            trainfiles, valfile = split_list
            testfile = valfile
        elif len(split_list) == 3:
            trainfiles, valfile, testfile = split_list
        else:
            raise Exception('Wrong number of files in splits')
        # We prepare the folds with all details for processing in
        # the Crossvalidation code (core/crossvalidation.py)
        folds.append(Fold(trainfiles, valfile, testfile))

    # Go to core/crossvalidation.py for more details
    cv = Crossvalidation(cache, conf, data_conf, args, folds)
    cv.run()#执行训练、验证、测试，并返回单个epoch的results字典（已整理，包含mean，25%，medium等）

if __name__ == '__main__':
    main()

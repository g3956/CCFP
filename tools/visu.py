#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
import pandas as pd
import numpy as np
import numpy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import glob
import os
from sklearn.manifold import TSNE
import collections
import re
import torchvision.transforms as T
import random
from scipy.stats import multivariate_normal
from fastreid.modeling.meta_arch.baseline import Baseline
import torch
from fastreid.data.transforms import ToTensor
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.common import CommDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


def Gaussian_Distribution(N=2, M=50, m=0, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(m, sigma, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


'''
colors = ['#FFB08E', '#A9D18E', '#FFC000', '#E88492']
for color, m,sigma in zip(colors, [numpy.array([0,0]), numpy.array([0,5]), numpy.array([-4,-3]),numpy.array([4,-3])],
    [numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]]),numpy.array([[1.2,0],[0,1.2]])]):
    plt.axis('off')
    print(m,sigma)
    data,_ = Gaussian_Distribution(N=2, M=50,m=m,sigma=sigma)
    x,y = data.T
    plt.scatter(x,y,color=color, s=7)

plt.axis([-8, 8, -8, 8])
plt.show()

plt.savefig('./augmentation.png')
'''


res = []
res.append(T.Resize((256,128), interpolation=3))
res.append(ToTensor())
trans = T.Compose(res)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

args = default_argument_parser().parse_args()

cfg = setup(args)


cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
model = DefaultTrainer.build_model(cfg)

Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

data = []
pattern = re.compile(r'([-\d]+)_c(\d)')
test_path = '/home/wenhang/data/Market-1501/bounding_box_train'
img_paths = glob.glob(os.path.join(test_path, '*.jpg'))
cid2pid = collections.defaultdict(list)
pid2index = [collections.defaultdict(list) for _ in range(6)]

for index,img_path in enumerate(img_paths):
    pid, camid = map(int, pattern.search(img_path).groups())
    camid = camid-1
    if pid == -1 or pid == 0:
        continue  # junk images are just ignored
    if not pid in cid2pid[camid]:
        cid2pid[camid].append(pid)
    pid2index[int(camid)][int(pid)].append(img_path)


choice_pids = [707,948,842,390,475,430,354,998,519,901]
train_cid = {707:2,948:4,842:4,390:4,475:4,430:3,354:4,998:2,519:2,901:4}


for cid,pids in cid2pid.items():
    for pid in choice_pids:
        img_paths_temp = pid2index[cid][pid]
        for img_path_temp in img_paths_temp:
            data.append((img_path_temp,pid,cid,0,0))
print(len(data))

visu_data = CommDataset(data,trans,relabel=False)
visu_loader = torch.utils.data.DataLoader(visu_data,batch_size=32,shuffle=False,drop_last=False)
visu_data = collections.defaultdict(list)
fake_feature = collections.defaultdict(list)
model.eval()
with torch.no_grad():
    for i,batched_inputs in enumerate(visu_loader):
        features,fake_feature_list = model(batched_inputs)
        targets = batched_inputs["targets"]
        camids = batched_inputs['camids']
        for j,(feature,pid,cid) in enumerate(zip(features,targets,camids)):
            visu_data['features'].append(feature)
            visu_data['pid'].append(pid)
            visu_data['cid'].append(cid)
            '''
            for k in range(6):
                visu_data['features'].append(fake_feature_list[k][j])
                visu_data['pid'].append(pid)
                visu_data['cid'].append(torch.tensor(100))
            '''

'''

features = visu_data['features']
features = torch.stack(features).cpu()
print(features.size())


y = visu_data.pop('cid')
y = LabelEncoder().fit(y).transform(y)

X = StandardScaler().fit(features).transform(features)


n_components = 2
tsne = TSNE(n_components=2)
X_pca = tsne.fit_transform(X)
colors = ['darkgrey', 'r', 'dodgerblue', 'limegreen', 'pink', 'darkorange','blue']
plt.figure()#(figsize=(8, 8))
for color, i in zip(colors, [0, 1, 2, 3, 4, 5]):
    plt.axis('off')

    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=color, s=5)
#plt.axis([-60, 60, -100, 100])
plt.show()
plt.savefig('./AGW_all.pdf')


'''
labels = torch.stack(visu_data['cid'])
labels = torch.unique(labels)

pids = torch.stack(visu_data['pid'])
pids_uniq = torch.unique(pids)


features = visu_data['features']
features = torch.stack(features).cpu()

y = visu_data.pop('cid')
y = LabelEncoder().fit(y).transform(y)


# standardize the data by setting the mean to 0 and std to 1
standardize = True
X = StandardScaler().fit(features).transform(features)

n_components = 2
tsne = TSNE(n_components=2)
X_pca = tsne.fit_transform(X)

colors = [ 'dodgerblue', 'darkorange','darkgrey', 'r', 'limegreen','pink']
          #'yellow', 'red', 'pink', 'palegoldenrod', 'navy', 'turquoise', 'darkorange', 'blue', 'purple', 'green',]

plt.figure()#(figsize=(8, 8))
for pid,marker in zip(choice_pids,['o','v','^','<','>','*','x','+','3','p']):
    print(pid)
    pid = torch.tensor(pid)
    cid = train_cid[int(pid)]
    index = torch.where(pid == pids)[0]
    X_pca_single_pid = X_pca[index]
    y_single_cid = y[index]
    print(y_single_cid)

    '''
    for color, i in zip(colors, [0, 1, 2, 3, 4, 5]):
        plt.axis('off')

        plt.scatter(X_pca_single_pid[y_single_cid == i, 0], X_pca_single_pid[y_single_cid == i, 1],
                    color=color,s=10,marker=marker,linewidths=1)
    '''
    plt.axis('off')

    plt.scatter(X_pca_single_pid[y_single_cid == cid, 0], X_pca_single_pid[y_single_cid == cid, 1],
                color='dodgerblue', s=20, marker=marker, linewidths=2)
    plt.scatter(X_pca_single_pid[y_single_cid != cid, 0], X_pca_single_pid[y_single_cid != cid, 1],
                color='darkorange', s=20, marker=marker, linewidths=2,alpha=0.8)
    '''
    plt.scatter(X_pca_single_pid[y_single_cid == 6, 0], X_pca_single_pid[y_single_cid == 6, 1],
                color='limegreen', s=5, marker=marker, linewidths=1)
    '''



#plt.axis([-30, 30, -30, 30])
plt.show()
plt.savefig('./base.pdf')



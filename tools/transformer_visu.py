
import sys

sys.path.append('.')

from fastreid.utils.visualizer import AttentionVisualizer
import random
from scipy.stats import multivariate_normal
from fastreid.modeling.meta_arch.baseline import Baseline
import torch
from fastreid.data.transforms import ToTensor
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.common import CommDataset
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
import collections
import re
import torchvision.transforms as T


res = []
res.append(T.Resize(800))
res.append(ToTensor())
res.append(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
transforms = T.Compose(res)

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
#model = DefaultTrainer.build_model(cfg)
model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
img_path = '/home/wenhang/data/Market-1501/market_sct/0002_c2s1_000301_01.jpg'
#Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model

model.eval()
with torch.no_grad():
    visualizer = AttentionVisualizer(model,transforms,img_path)
    visualizer.run()




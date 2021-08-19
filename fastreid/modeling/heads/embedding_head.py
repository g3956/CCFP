# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from torch import nn
import torch
from fastreid.layers import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import REID_HEADS_REGISTRY
import collections

@REID_HEADS_REGISTRY.register()
class EmbeddingHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # fmt: off
        feat_dim      = cfg.MODEL.BACKBONE.FEAT_DIM
        num_classes   = cfg.MODEL.HEADS.NUM_CLASSES
        neck_feat     = cfg.MODEL.HEADS.NECK_FEAT
        pool_type     = cfg.MODEL.HEADS.POOL_LAYER
        cls_type      = cfg.MODEL.HEADS.CLS_LAYER
        norm_type     = cfg.MODEL.HEADS.NORM
        self.num_cameras = cfg.DATASETS.NUM_CAMERAS

        if pool_type == 'fastavgpool':   self.pool_layer = FastGlobalAvgPool2d()
        elif pool_type == 'avgpool':     self.pool_layer = nn.AdaptiveAvgPool2d(1)
        elif pool_type == 'gempoolP':    self.pool_layer = GeneralizedMeanPoolingP()
        elif pool_type == 'gempool':     self.pool_layer = GeneralizedMeanPooling()
        else:                            raise KeyError(f"{pool_type} is not supported!")
        # fmt: on

        self.neck_feat = neck_feat

        self.bottleneck = get_norm(norm_type, feat_dim, bias_freeze=True)

        #camera-specific batch normalization
        self.bottleneck_camera = nn.ModuleList(torch.nn.BatchNorm2d(2048) for _ in range(self.num_cameras))
        self.bottleneck_camera_map = nn.ModuleList(torch.nn.BatchNorm2d(2048) for _ in range(self.num_cameras))

        # classification layer
        # fmt: off
        if cls_type == 'linear':          self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        else:                             raise KeyError(f"{cls_type} is not supported!")
        # fmt: on

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck_map.apply(weights_init_kaiming)
        for bn in self.bottleneck_camera:
            bn.apply(weights_init_kaiming)
        for bn in self.bottleneck_camera_map:
            bn.apply(weights_init_kaiming)

    def forward(self, features, targets=None, camids=None):
        """
        See :class:`ReIDHeads.forward`.
        """
        global_feat = self.pool_layer(features)

        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        feature_after_bn_list = []
        feature_map_list = []
        feature_dict_out = collections.defaultdict(list)
        feature_map_dict_out = collections.defaultdict(list)
        global_feat_map = self.bottleneck(features)
        feature_map_out = []
        fake_feature_out = []

        if not self.training:
            return bn_feat, global_feat_map
        else:
            uniq = torch.unique(camids)
            for c in uniq:
                index = torch.where(c == camids)[0]
                feature_after_bn_list.append(global_feat[index])
                feature_map_list.append(features[index])

            for cid, feature_cid_per,feature_map in zip(uniq,feature_after_bn_list, feature_map_list):
                for i in range(self.num_cameras):
                    if i == int(cid):
                        feature_dict_out[int(cid)].append(self.bottleneck_camera[i](feature_cid_per)[..., 0, 0])
                        feature_map_dict_out[int(cid)].append(self.bottleneck_camera_map[i](feature_map))
                    else:
                        self.bottleneck_camera[i].eval()
                        self.bottleneck_camera_map[i].eval()
                        feature_dict_out[int(cid)].append(self.bottleneck_camera[i](feature_cid_per)[..., 0, 0])
                        feature_map_dict_out[int(cid)].append(self.bottleneck_camera_map[i](feature_map))
                        self.bottleneck_camera[i].train()
                        self.bottleneck_camera_map[i].train()

            for i, (key, values) in enumerate(feature_map_dict_out.items()):
                if i == 0:
                    for value in values:
                        feature_map_out.append(value)
                if i == 1:
                    for j, value in enumerate(values):
                        feature_map_out[j] = torch.cat((feature_map_out[j], value))

            for i, (key, values) in enumerate(feature_dict_out.items()):
                if i == 0:
                    for value in values:
                        fake_feature_out.append(value)
                if i == 1:
                    for j, value in enumerate(values):
                        fake_feature_out[j] = torch.cat((fake_feature_out[j], value))

        # Evaluation
        # fmt: off
        if not self.training: return bn_feat
        # fmt: on
        # Training
        if self.classifier.__class__.__name__ == 'Linear':
            cls_outputs = self.classifier(bn_feat)
            pred_class_logits = F.linear(bn_feat, self.classifier.weight)
        else:
            cls_outputs = self.classifier(bn_feat, targets)
            pred_class_logits = self.classifier.s * F.linear(F.normalize(bn_feat),
                                                             F.normalize(self.classifier.weight))

        # fmt: off
        if self.neck_feat == "before":  feat = global_feat[..., 0, 0]
        elif self.neck_feat == "after": feat = bn_feat
        else:                           raise KeyError(f"{self.neck_feat} is invalid for MODEL.HEADS.NECK_FEAT")
        # fmt: on

        return {
            "cls_outputs": cls_outputs,
            "pred_class_logits": pred_class_logits,
            "features": feat,
            'bn_feat': bn_feat,
            'feature_dict_bn': feature_dict_out,
            'feature_map_dict': feature_map_dict_out,
            'feature_map_out': feature_map_out,
            'global_feature_map': global_feat_map,
        }

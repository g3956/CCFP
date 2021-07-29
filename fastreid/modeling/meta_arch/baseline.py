# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from fastreid.modeling.self_module import *
from fastreid.utils.my_tools import *
from fastreid.utils.misc import *

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        # backbone
        self.backbone = build_backbone(cfg)

        #transformer
        self.transformer = build_transformer(cfg)

        #positional encoder
        self.positional_encoder = build_position_encoding(cfg)

        #transformer_related
        self.class_embed = nn.Linear(cfg.MODEL.Transformer.hidden_dim, cfg.MODEL.Transformer.num_patch)
        self.query_embed = nn.Embedding(cfg.MODEL.Transformer.num_patch, cfg.MODEL.Transformer.hidden_dim)
        self.input_proj = nn.Conv2d(2048, cfg.MODEL.Transformer.hidden_dim, kernel_size=1)

        # head
        self.heads = build_heads(cfg)

        self.distance_loss = build_distanceloss()

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, class2cid=None, iter=None):

        images = self.preprocess_image(batched_inputs)

        if isinstance(images,(List, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        features = self.backbone(images.tensors)

        if not self.training:
            outputs, global_feat_map_bn = self.heads(features)
            return outputs,global_feat_map_bn

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"].to(self.device)
            camids = batched_inputs['camids']

            outputs = self.heads(features, targets=targets, camids=camids)

            global_feat_map_bn = outputs['global_feature_map']

            patch_features_anchor,outp_class_anchor = forward_transformer(images, global_feat_map_bn,
                 self.positional_encoder,self.transformer, self.input_proj, self.query_embed, self.class_embed)
            
            patch_feature_list = []
            feature_map_list = outputs['feature_map_out']
            with torch.no_grad():
                for feature_map in feature_map_list:
                    patch_feature_list.append(forward_transformer(images, feature_map, self.positional_encoder,
                                 self.transformer, self.input_proj, self.query_embed, self.class_embed)[0])

            losses = self.losses(outputs, targets, camids, class2cid, iter, patch_feature_list,
                                 patch_features_anchor,outp_class_anchor)

            return losses

    def preprocess_image(self, batched_inputs):
        r"""
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs["images"].to(self.device)
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs.to(self.device)
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images


    def losses(self, outputs, gt_labels, camids, class2cid, iter, patch_feature_list,patch_features_anchor,outp_class_anchor):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off

        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        bn_features       = outputs['bn_feat']
        feature_dict_bn    = outputs['feature_dict_bn']

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}

        #CaCE loss for classification
        loss_dict['CaCE'] = cross_entropy_loss_ca(
            cls_outputs, gt_labels,
            self._cfg.MODEL.LOSSES.CE.EPSILON,
            class2cid,
            self._cfg.MODEL.LOSSES.CE.ALPHA)


        #MCNL loss for metric learning
        loss_dict["mcnl-G_loss"] = self.distance_loss(pred_features, gt_labels, camids)[0]

        #global-level self-supervised learning loss
        loss_dict['GSL_loss'] = bn_self_super_loss(bn_features, feature_dict_bn, gt_labels, camids)
		

        #local-level self-supervised learning loss
        loss_list = patch_loss(patch_feature_list, patch_features_anchor, gt_labels,camids,outp_class_anchor)

        for i, loss in enumerate(loss_list):
            if i == 0:
                if iter >= 1000:
                    loss_dict['LSL_loss'] = loss
            elif i == 1:
                loss_dict['SSRC_loss'] = loss
            else:
                if iter >= 500:
                    loss_dict['MCNL-L_loss'] = loss

        return loss_dict

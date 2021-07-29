# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F

from fastreid.modeling.losses import *

distance_loss  = build_distanceloss()

def D(p, z):
    z = z.detach()
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()

def patch_loss(patch_feature_list, patch_features_anchor, gt_classes, camids, outp_class_anchor):

    bs = patch_features_anchor.size(0)
    num_patch = patch_features_anchor.size(1)

    loss = []

    loss_items = []
    patch_features_anchor_trans = patch_features_anchor.transpose(0,1)

    for patch_feature in patch_feature_list:
        for i in range(patch_features_anchor_trans.size(0)):
            patch_feature_ = patch_feature.transpose(0,1)[i]
            loss_items.append(D(patch_features_anchor_trans[i], patch_feature_))
    loss_patch = torch.mean(torch.stack(loss_items))

    loss_exclusive = 0

    for i in range(bs):

        patch_features_single_img = patch_features_anchor[i]
        patch_features_single_img_norm = F.normalize(patch_features_single_img, p=2, dim=1)
        cosine_similarity = torch.mm(patch_features_single_img_norm, patch_features_single_img_norm.t())
        logit = F.log_softmax(cosine_similarity, dim=1)
        target = torch.arange(num_patch).cuda()  # each patch belongs to a exlusive class
        loss_exclusive += F.nll_loss(logit, target) / bs

        outp_class_single = outp_class_anchor[i]
        target = torch.arange(num_patch).cuda()
        outp_class_single_log = F.log_softmax(outp_class_single, dim=1)
        zeros = torch.zeros(outp_class_single_log.size())
        targets = zeros.scatter_(1, target.unsqueeze(1).data.cpu(), 1).cuda()
        loss_exclusive += (-targets * outp_class_single_log).mean(0).sum() / bs

    features_patch_conca = patch_features_anchor.reshape((bs, -1))
    loss_discri = distance_loss(features_patch_conca, gt_classes, camids)[0]

    loss.append(loss_patch)
    loss.append(loss_exclusive)
    loss.append(loss_discri)


    return loss

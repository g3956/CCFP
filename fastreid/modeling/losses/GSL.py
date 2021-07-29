import torch
import torch.nn.functional as F
from .MCNL import build_distanceloss

def bn_self_super_loss(features, feature_dict_bn, pids, camids):

    uniq = torch.unique(camids)

    loss = 0
    count = 0

    #only consider positive pairs
    for c in uniq:
        index = torch.where(c == camids)[0]
        pids_camera = pids[index]
        uniq_pid = torch.unique(pids_camera)
        features_camera = features[index]
        contrastive_feature_list = feature_dict_bn[int(c)]
        for features_contrastive in contrastive_feature_list:
            for pid in uniq_pid:
                index_same_id = torch.where(pid == pids_camera)[0]
                features_same_id = features_camera[index_same_id]
                features_same_id_c = features_contrastive[index_same_id]
                distance_matrix = -torch.mm(F.normalize(features_same_id,p=2,dim=1),
                                            F.normalize(features_same_id_c,p=2,dim=1).t().detach())
                loss += torch.mean(distance_matrix)
                count += 1

    GSL = loss / count

    return GSL





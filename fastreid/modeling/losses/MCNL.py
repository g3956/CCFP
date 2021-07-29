
import torch
from torch import nn

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def distance_mining(dist_mat, labels, cameras):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)

    N = dist_mat.size(0)

    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())  # & cameras.expand(N,N).eq(cameras.expand(N,N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())  # | cameras.expand(N,N).ne(cameras.expand(N,N).t())
    d1 = torch.max(dist_mat * is_pos.float().detach(), 1, keepdim=True)[0]
    d1 = d1.squeeze(1)
    # dist_neg=dist_mat[is_neg].contiguous().view(N,-1)
    d2 = d1.new().resize_as_(d1).fill_(0)
    d3 = d1.new().resize_as_(d1).fill_(0)
    d2ind = []
    for i in range(N):
        sorted_tensor, sorted_index = torch.sort(dist_mat[i])
        cam_id = cameras[i]
        B, C = False, False
        for ind in sorted_index:
            if labels[ind] == labels[i]:
                continue
            if B == False and cam_id == cameras[ind]:
                d3[i] = dist_mat[i][ind]
                B = True
            if C == False and cam_id != cameras[ind]:
                d2[i] = dist_mat[i][ind]
                C = True
                d2ind.append(ind)
            if B and C:
                break
    return d1, d2, d3, d2ind


class DistanceLoss(object):
    """Multi-camera negative loss
        In a mini-batch,
       d1=(A,A'), A' is the hardest true positive.
       d2=(A,C), C is the hardest negative in another camera.
       d3=(A,B), B is the hardest negative in same camera as A.
       let d1<d2<d3
    """
    def __init__(self,loader=None,margin=None):
        self.margin=margin
        self.texture_loader=loader
        if margin is not None:
            self.ranking_loss1=nn.MarginRankingLoss(margin=margin[0],reduction="mean")
            self.ranking_loss2=nn.MarginRankingLoss(margin=margin[1],reduction="mean")
            self.ranking_loss3 = nn.MarginRankingLoss(margin=margin[2], reduction="mean")
        else:
            self.ranking_loss=nn.SoftMarginLoss(reduction="mean")

    def __call__(self,feat,labels,cameras,model=None,paths=None,epoch=0,normalize_feature=False):
        if normalize_feature: # default: don't normalize , distance [0,1]
            feat=normalize(feat,axis=-1)
        dist_mat=euclidean_dist(feat,feat)

        d1,d2,d3,d2ind= distance_mining(dist_mat,labels,cameras)

        y=d1.new().resize_as_(d1).fill_(1)
        if self.margin is not None:
            l1=self.ranking_loss1(d2,d1,y)
            l2=self.ranking_loss2(d3,d2,y)

        else:
            l1=self.ranking_loss(d2-d1,y)
            l2=self.ranking_loss(d3-d2,y)

        loss=l2+l1
        accuracy1=torch.mean((d1<d2).float())
        accuracy2=torch.mean((d2<d3).float())
        return loss,accuracy1,accuracy2

def build_distanceloss():
    return DistanceLoss(margin=(0.1,0.1,0.1))
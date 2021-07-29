from torch.nn import Parameter
import torch
import numpy as np
from torchvision.utils import save_image
import os
import time
import torch
import torch.nn as nn
from fastreid.utils.misc import *
import torch.nn.functional as F
from fastreid.utils.compute_dist import *

def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            # print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    print(missing)

    return model

def preprocess_image(batched_inputs):
    r"""
    Normalize and batch the input images.
    """
    if isinstance(batched_inputs, dict):
        images = batched_inputs["images"].cuda()
    elif isinstance(batched_inputs, torch.Tensor):
        images = batched_inputs.cuda()
    else:
        raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))
    pixel_mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).view(1, -1, 1, 1).cuda()
    pixel_std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).view(1, -1, 1, 1).cuda()

    images.sub_(pixel_mean).div_(pixel_std)
    return images


def forward_transformer(images, features, positional_encoder, transformer, input_proj, query_embed, class_embed, require_grad=True):
    m = images.mask
    mask = F.interpolate(m[None].float(), size=features.shape[-2:]).to(torch.bool)[0]
    out = NestedTensor(features, mask)

    # positional encoder
    if require_grad is False:
        for key, param in positional_encoder.named_parameters():
            param.require_grad = False
        pos = positional_encoder(out).to(out.tensors.dtype)

        # transformer encoder
        src, mask = out.decompose()
        for key, param in transformer.named_parameters():
            param.require_grad = False
        hs = transformer(input_proj(src), mask, query_embed.weight, pos)[0]  # [6, 72, 8, 256]
        for key, param in positional_encoder.named_parameters():
            param.require_grad = True
        for key, param in transformer.named_parameters():
            param.require_grad = True
        outputs_class = class_embed(hs)
    else:
        pos = positional_encoder(out).to(out.tensors.dtype)

        # transformer encoder
        src, mask = out.decompose()
        hs = transformer(input_proj(src), mask, query_embed.weight, pos)[0]  # [6, bs, 6, 256]
        outputs_class = class_embed(hs[-1])#[128,6,6]


    return hs[-1],outputs_class

def get_class2camera(model, data_loader, num_classes):
    class2cid = {}
    pid2cid = {}
    cid_sorted = []
    model.eval()
    data_loader_iter = iter(data_loader)
    with torch.no_grad():
        while True:
            data = next(data_loader_iter)
            pids = data['targets']
            cids = data['camids']
            for pid, cid in zip(pids, cids):
                pid2cid[int(pid)] = int(cid)
            if len(pid2cid) == num_classes:
                break
        for idx in sorted(pid2cid.keys()):
            cid_sorted.append(pid2cid[idx])
        while True:
            data = next(data_loader_iter)
            pids = data['targets']
            cids = data['camids']
            for pid, cid in zip(pids, cids):
                class2cid[int(pid)] = (cid != torch.tensor(cid_sorted)).long()
            if len(class2cid) == num_classes:
                break
    model.train()
    return class2cid
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image


def conv_layer(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)

def fc_layer(in_channels, out_channels, bias=True, activation=nn.ReLU, batch_norm=False):
    layers = [nn.Linear(int(in_channels), int(out_channels), bias=bias)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))
    if activation:
        layers.append(activation())
    return nn.Sequential(*layers)

def slicing(grid, guide):
    N, C, H, W = guide.shape
    device = grid.get_device()
    if device >= 0:
        hh, ww = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device)) # H, W
    else:
        hh, ww = torch.meshgrid(torch.arange(H), torch.arange(W)) # H, W
    # To [-1, 1] range for grid_sample
    hh = hh / (H - 1) * 2 - 1
    ww = ww / (W - 1) * 2 - 1
    guide = guide * 2 - 1
    hh = hh[None, :, :, None].repeat(N, 1, 1, 1) # N, H, W, C=1
    ww = ww[None, :, :, None].repeat(N, 1, 1, 1)  # N, H, W, C=1
    guide = guide.permute(0, 2, 3, 1) # N, H, W, C=1

    guide_coords = torch.cat([ww, hh, guide], dim=3) # N, H, W, 3
    # unsqueeze because extra D dimension
    guide_coords = guide_coords.unsqueeze(1) # N, Dout=1, H, W, 3
    sliced = F.grid_sample(grid, guide_coords, align_corners=False, padding_mode="border") # N, C=12, Dout=1, H, W
    sliced = sliced.squeeze(2) # N, C=12, H, W

    return sliced

def apply(sliced, fullres):
    # r' = w1*r + w2*g + w3*b + w4
    rr = fullres * sliced[:, 0:3, :, :] # N, C=3, H, W
    gg = fullres * sliced[:, 4:7, :, :] # N, C=3, H, W
    bb = fullres * sliced[:, 8:11, :, :] # N, C=3, H, W
    rr = torch.sum(rr, dim=1) + sliced[:, 3, :, :] # N, H, W
    gg = torch.sum(gg, dim=1) + sliced[:, 7, :, :] # N, H, W
    bb = torch.sum(bb, dim=1) + sliced[:, 11, :, :] # N, H, W
    output = torch.stack([rr, gg, bb], dim=1) # N, C=3, H, W
    return output
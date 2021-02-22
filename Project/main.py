import os
import sys
import time

sys.path.extend([item[0] for item in os.walk(os.path.join(os.path.dirname(__file__), "DeepHomography"))])

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import imageio
from DeepHomography.OnelineDLTv1.torch_homography_model import build_model
from DeepHomography.OnelineDLTv1.dataset import *
from DeepHomography.OnelineDLTv1.utils import transformer as trans
import numpy as np

from collections import OrderedDict

exp_name = os.path.join(os.path.abspath(os.path.dirname("__file__")), 'DeepHomography')
work_dir = os.path.join(exp_name, 'Data')
pair_list = list(open(os.path.join(work_dir, 'Test_List.txt')))
npy_path = os.path.join(work_dir, 'Coordinate/')


def build_CAUDHE(args):
    """
    Function to build CAUDHE network model from pretrained weights.
    This network is composed by ...
    TODO: rellenar descripción
    :param model_name: resnet model name. 'resnet18', 'resnet34', 'resnet50',
        'resnet101' & 'resnet152' are available from URL. 'resnet34' by default.
    :param pretrained: True to load weigths from URL.
    :param finetune: True to load weights for CAUDHE from file.
    """

    # Weights load
    net = build_model(args.model_name, pretrained=True)
    model_path = os.path.join(exp_name, 'models/freeze-mask-first-fintune.pth')
    state_dict = torch.load(model_path, map_location='cpu')

    new_state_dict = OrderedDict()
    for k, v in state_dict.state_dict().items():
        namekey = k[7:]  # Remove word 'module' from namekeys
        new_state_dict[namekey] = v
    # load params
    net = build_model(args.model_name)
    model_dict = net.state_dict()
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict.keys()}
    model_dict.update(new_state_dict)
    net.load_state_dict(model_dict)

    # For parallel processing
    net = torch.nn.DataParallel(net)
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def load_dataset(name='MyDataset', patch_w=1, patch_h=1, rho=0, WIDTH=1, HEIGHT=1, batch_size=1):
    data = eval(name)(data_path=exp_name, patch_w=patch_w, patch_h=patch_h,
                      WIDTH=WIDTH, HEIGHT=HEIGHT)

    data_loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=0,
                             shuffle=False, drop_last=True)
    return data_loader


def test(net, dataset):
    net.eval()
    for i, batch_val in enumerate(dataset):

        org_imges = batch_val[0].float()
        input_tensors = batch_val[1].float()
        patch_indices = batch_val[2].float()
        h4p = batch_val[3].float()

        if torch.cuda.is_available():
            input_tensors = input_tensors.cuda()
            patch_indices = patch_indices.cuda()
            h4p = h4p.cuda()

        print(f"Computing homography (i = {i})")
        ts = time.time()
        batch_out = net(org_imges, input_tensors, h4p, patch_indices)
        H_mat = batch_out['H_mat']  # Homography matrix
        print(H_mat)
        tf = time.time() - ts
        print(f"Time spent: {tf} seg\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-9, help='learning_rate')

    parser.add_argument('--model_name', type=str, default='resnet34')
    print('<======================= Loading data =======================>')

    args = parser.parse_args()
    print(args)

    net = build_CAUDHE(args)
    print(net)

    dataset = load_dataset(name='MyDataset', patch_w=args.img_w, patch_h=args.img_h, rho=0,
                           WIDTH=args.img_w, HEIGHT=args.img_h, batch_size=args.batch_size)

    test(net, dataset)

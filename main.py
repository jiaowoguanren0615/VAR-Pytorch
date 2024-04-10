import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path
import random
import os

import os.path as osp
import torch, torchvision
import PIL.Image as PImage
import matplotlib.pyplot as plt

from models import VQVAE, build_VAR

import util.dist_utils as utils
from util.engine import generate_image


def get_args_parser():
    parser = argparse.ArgumentParser('VAR ImageGeneration script', add_help=False)
    parser.add_argument('--model_depth', default=16, type=int, choices=[12, 16, 20, 24, 30])
    parser.add_argument('--image_name', default='demo.png', type=str,
                        help="set image's name")
    parser.add_argument('--output_dir', default='./save_images', type=str,
                        help='create a directory to save images, set it null for no saving')
    parser.add_argument('--hf_home', default='https://huggingface.co/FoundationVision/var/resolve/main', type=str)

    # build vae, var
    parser.add_argument('--V', default=4096, type=int)
    parser.add_argument('--ch', default=160, type=int)
    parser.add_argument('--Cvae', default=4, type=int)
    parser.add_argument('--share_quant_resi', default=4, type=int)
    parser.add_argument('--patch_nums', default=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), type=tuple)

    # training parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--tf32', default=True, type=bool)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cfg', default=4, type=int,
                        help='@param {type:"slider", min:1, max:10, step:0.1}')
    parser.add_argument('--class_labels', default=(980, 980, 437, 437, 22, 22, 562, 562), type=tuple,
                        help='@param {type:"raw"}')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser


def main(args):
    print(args)

    utils.initialize()

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    # run faster
    tf32 = args.tf32
    cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    assert args.model_depth in [12, 16, 20, 24, 30], 'You must set the model-depth in [12, 16, 20, 24, 30]'

    # download checkpoints
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{args.model_depth}.pth'
    if not osp.exists(vae_ckpt): os.system(f'wget {args.hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt): os.system(f'wget {args.hf_home}/{var_ckpt}')

    # build models
    vae = VQVAE(vocab_size=args.V, z_channels=args.Cvae, ch=args.ch, test_mode=True, share_quant_resi=4,
                v_patch_nums=args.patch_nums).to(args.device)
    var = build_VAR(vae=vae, depth=args.model_depth, patch_nums=args.patch_nums).to(args.device)

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)

    if args.distributed:
        vae = torch.nn.parallel.DistributedDataParallel(vae, device_ids=[args.gpu])
        var = torch.nn.parallel.DistributedDataParallel(var, device_ids=[args.gpu])
    vae.eval(), var.eval()

    # freeze_layers
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)

    print(f'prepare finished.')

    B = len(args.class_labels)
    label_B: torch.LongTensor = torch.tensor(args.lass_labels, device=args.device)

    recon_B3HW = generate_image(model=var, B=B, label_B=label_B, cfg=args.cfg, seed=seed, top_k=900, top_p=0.95)
    chw = torchvision.utils.make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))

    chw.save(f'{args.output_dir}/{args.image_name}')

    # Saving image using matplotlib
    # plt.axis('off')
    # plt.imshow(chw)
    # plt.savefig(f'{args.output_dir}/{args.image_name}', bbox_inches='tight', pad_inches=0)
    # plt.show()

    # Open image with PIL
    # image = PImage.open(f'{args.output_dir}/{args.image_name}')
    # plt.axis('off')
    # plt.imshow(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VAR ImageGeneration script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

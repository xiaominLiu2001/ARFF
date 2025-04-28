import os
import json
import argparse
import openslide
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from torch import nn
import torchvision.transforms as torch_trans
from torchvision import models
from torch.nn import DataParallel
import cv2
from PIL import Image
from resnet50_pretrained_TCGA import resnet50_TCGA
import pandas as pd


def transform_CLAHE(img):
    img = np.array(img)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y_channel, cr, cb = cv2.split(ycrcb_img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_y = clahe.apply(y_channel)
    merged_channels = cv2.merge([clahe_y, cr, cb])
    final_img = cv2.cvtColor(merged_channels, cv2.COLOR_YCrCb2RGB)
    final_tensor = torch.tensor(final_img, dtype=torch.float32).permute(2, 0, 1)
    final_tensor /= 255.0
    return final_tensor


def create_encoder(args):
    print(f"Info: Creating extractor {args.image_encoder}")
    if args.image_encoder == 'resnet50_TCGA':
        encoder = resnet50_TCGA(pretrained=True).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
    elif args.image_encoder == 'resnet50':
        encoder = models.resnet50(pretrained=True).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
    elif args.image_encoder == 'resnet18':
        encoder = models.resnet18(pretrained=True).to(args.device)
        layers = list(encoder.children())[:-1]
        layers.append(nn.Flatten(1))
        encoder = nn.Sequential(*layers)
    else:
        raise ValueError(f"image_encoder's name error!")

    ## Adapt model for multi-GPU if necessary
    if args.device != torch.device('cpu') and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)

    print(f"{args.image_encoder}:\n{encoder}")
    return encoder


def extract(args, image, encoder, transform):
    with torch.no_grad():
        if transform is True:
            image = transform_CLAHE(image).unsqueeze(dim=0).to(args.device)
        else:
            to_tensor = torch_trans.ToTensor()
            image = to_tensor(image).unsqueeze(dim=0).to(args.device)
        feat = encoder(image).cpu().numpy()
        return feat


def extract_features(args, encoder, save_dir):

    coord_dir = Path(args.patch_dir) / 'coord_true'
    if not coord_dir.exists():
        print(f"{str(coord_dir)} doesn't exist! ")
        return
    coord_list = sorted(list(coord_dir.glob('*.json')))
    print(f"num of coord: {len(coord_list)}")

    with torch.no_grad():
        encoder.eval()
        for i, coord_filepath in enumerate(coord_list):
            filename = coord_filepath.stem
            npz_filepath = save_dir / f'{filename}.npz'
            if npz_filepath.exists() and not args.exist_ok:
                print(f"{npz_filepath.name} is already exist, skip!")
                continue

            # obtain the parameters needed for feature extraction from `coord` file
            with open(coord_filepath) as fp:
                coord_dict = json.load(fp)
            num_patches = coord_dict['num_patches']
            if num_patches == 0:
                print(f"{filename}'s num_patches is {num_patches}, skip!")
                continue
            num_row, num_col = coord_dict['num_row'], coord_dict['num_col']
            coords = coord_dict['coords']
            patch_size_level0 = coord_dict['patch_size_level0']
            patch_size = coord_dict['patch_size']
            slide = openslide.open_slide(coord_dict['slide_filepath'])

            coords_bar = tqdm(coords)
            features, cds = [], []
            for c in coords_bar:
                img = slide.read_region(
                    location=(c['x'], c['y']),
                    level=0,
                    size=(patch_size_level0, patch_size_level0)
                ).convert('RGB').resize((patch_size, patch_size))

                feat = extract(args, img, encoder, transform=False)
                features.append(feat)
                cds.append(np.array([c['row'], c['col']], dtype=np.int))

                coords_bar.set_description(f"{i + 1:3}/{len(coord_list):3} | filename: {filename}")
                coords_bar.update()

            img_features = np.concatenate(features, axis=0)
            cds = np.stack(cds, axis=0)
            # save all the patch features of a WSI as a .npz file
            np.savez(file=npz_filepath,
                     filename=filename,
                     num_patches=num_patches,
                     num_row=num_row,
                     num_col=num_col,
                     img_features=img_features,
                     coords=cds)


def run(args):
    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    if args.save_dir is not None:
        save_dir = Path(args.save_dir) / args.image_encoder
    else:  # if the save directory is not specified, `patch_dir/features/${image_encoder}` is used by default
        save_dir = Path(args.patch_dir) / 'features' / args.image_encoder
    save_dir.mkdir(parents=True, exist_ok=True)

    if Path(save_dir).exists():
        print(f"{save_dir} is already exists. ")

    encoder = create_encoder(args)
    extract_features(args, encoder, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_dir', type=str, default='/home/ARFF/CLS_model/data_FTD/',
                        help='Directory containing `coord` files')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--image_encoder', type=str, default='resnet50') #resnet50_TCGA
    parser.add_argument('--device', default='0')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

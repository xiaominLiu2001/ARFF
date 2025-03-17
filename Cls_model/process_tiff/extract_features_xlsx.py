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
import shutil
import logging


def linear_scale(channel, min_val, max_val):
    if max_val != min_val:
        scale = 255.0 / (max_val - min_val)
    else:
        scale = 1.0
    scaled = (channel - min_val) * scale
    scaled = np.clip(scaled, 0, 255)
    return scaled.astype(np.uint8)

def apply_gamma(image_array, gamma=1.0):
    if gamma <= 0:
        gamma = 1.0
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype('uint8')
    return table[image_array]

def process_image_with_excel(image, image_name, excel_data, gamma_value=1.0):
    image_np = np.array(image)
    image_name_str = str(image_name).strip()
    matched_rows = excel_data[excel_data['Slide ID'] == image_name_str]
    if matched_rows.empty:
        return image

    row = matched_rows.iloc[0]
    min_val = row['min value']
    max_val = row['max value']

    if pd.isna(min_val) or pd.isna(max_val):
        return image

    if min_val >= max_val:
        return image

    for channel in range(3):  # 0: Red, 1: Green, 2: Blue
        try:
            image_np[:, :, channel] = linear_scale(image_np[:, :, channel], min_val, max_val)
        except Exception as e:
            return image

    if gamma_value != 1.0:
        try:
            image_np = apply_gamma(image_np, gamma=gamma_value)
        except Exception as e:
            logging.error(f"Error applying Gamma correction to image {image_name_str}: {e}")
            return image

    try:
        processed_image = Image.fromarray(image_np)
    except Exception as e:
        logging.error(f"Error converting processed NumPy array to PIL image: {e}")
        return image
    return processed_image



def create_encoder(args):
    print(f"Info: Creating extractor {args.image_encoder}")
    if args.image_encoder == 'vgg16':
        encoder = models.vgg16(pretrained=True).to(args.device)
        encoder.classifier = nn.Sequential(*list(encoder.classifier.children())[:-3])
    elif args.image_encoder == 'resnet50_TCGA':
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
    if args.device != torch.device('cpu') and torch.cuda.device_count() > 1:
        encoder = nn.DataParallel(encoder)
    print(f"{args.image_encoder}:\n{encoder}")
    return encoder


def extract(args, image, encoder):
    with torch.no_grad():
        to_tensor = torch_trans.ToTensor()
        image = to_tensor(image).unsqueeze(dim=0).to(args.device)
        feat = encoder(image).cpu().numpy()
        return feat


def extract_features(args, encoder, save_dir):
    excel_path = Path(args.excel_file)
    if not excel_path.exists():
        print(f"{str(excel_path)} doesn't exist!")
        return
    excel_data = pd.read_excel(excel_path)
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
                img = process_image_with_excel(img, filename, excel_data, gamma_value=args.gamma)

                feat = extract(args, img, encoder)
                features.append(feat)
                cds.append(np.array([c['row'], c['col']], dtype=np.int))

                coords_bar.set_description(f"{i + 1:3}/{len(coord_list):3} | filename: {filename}")
                coords_bar.update()

            img_features = np.concatenate(features, axis=0)
            cds = np.stack(cds, axis=0)
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
    else:
        save_dir = Path(args.patch_dir) / 'features' / args.image_encoder / 'true'
    save_dir.mkdir(parents=True, exist_ok=True)

    if Path(save_dir).exists():
        print(f"{save_dir} is already exists. ")

    encoder = create_encoder(args)
    extract_features(args, encoder, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_file', type=str, help='Directory containing `xlsx` files')
    parser.add_argument('--patch_dir', type=str, help='Directory containing `coord` files')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--image_encoder', type=str, default='resnet50') #resnet50_TCGA
    parser.add_argument('--device', default='3')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--gamma', type=float, default=1.0)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

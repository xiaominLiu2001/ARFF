import cv2
import json
import argparse
import openslide
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import os
from utils import get_three_points, keep_patch, out_of_bound
import logging
import pandas as pd
import concurrent.futures

def tiling(slide_filepath, magnification, patch_size, scale_factor=32, tissue_thresh=0.5,
           overview_level=-1, coord_dir=None, thumbnail_dir=None, overview_dir=None, mask_dir=None, filename=None, device=None):

    torch.cuda.set_device(device)
    slide = openslide.open_slide(str(slide_filepath))
    if 'aperio.AppMag' in slide.properties.keys():
        level0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level0_magnification = 40
    if level0_magnification < magnification:
        print(f"{level0_magnification}<{magnification}? magnification should <= level0_magnification.")
        return
    patch_size_level0 = int(patch_size * (level0_magnification / magnification))

    if overview_dir is not None and thumbnail_dir is not None:
        thumbnail_path = str(thumbnail_dir / f'{filename}.jpg')
        thumbnail = cv2.imread(thumbnail_path)
    elif overview_dir is not None and thumbnail_dir is None:
        thumbnail = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
    else:
        print("thumbnail = None")
        thumbnail = None

    mask_path = str(mask_dir / f'{filename}.png') if mask_dir is not None else None
    if not os.path.exists(mask_path):
        return
    mask = Image.open(mask_path)

    color_bg = np.array([0, 0, 0])
    mask_w, mask_h = mask.size
    if mask.mode == 'RGBA':
        mask = mask.convert('RGB')

    mask_array = np.asarray(mask)

    if len(mask_array.shape) == 2 or (len(mask_array.shape) == 3 and mask_array.shape[2] == 1):
        mask = cv2.cvtColor(mask_array, cv2.COLOR_GRAY2BGR)
    else:
        mask = mask_array

    mask_patch_size = int(((patch_size_level0 // scale_factor) * 2 + 1) // 2)
    num_step_x = int(mask_w // mask_patch_size)
    num_step_y = int(mask_h // mask_patch_size)

    coord_list = []
    print(f"Processing {filename}...")
    for row in range(num_step_y):
        for col in range(num_step_x):
            # get the patch image at mask
            points_mask = get_three_points(col, row, mask_patch_size)
            row_start, row_end = points_mask[0][1], points_mask[1][1]
            col_start, col_end = points_mask[0][0], points_mask[1][0]
            patch_mask = mask[row_start:row_end, col_start:col_end]
            if keep_patch(patch_mask, tissue_thresh, color_bg):  # decide keep or drop the patch by `tissue_thresh`
                points_level0 = get_three_points(col, row, patch_size_level0)
                if out_of_bound(slide.dimensions[0], slide.dimensions[1], points_level0[1][0], points_level0[1][1]):
                    continue
                coord_list.append({'row': row, 'col': col, 'x': points_level0[0][0], 'y': points_level0[0][1]})
                if overview_dir is not None:
                    points_thumbnail = get_three_points(col, row,
                                                        patch_size_level0 / slide.level_downsamples[overview_level])
                    cv2.rectangle(thumbnail, points_thumbnail[0], points_thumbnail[1], color=(0, 0, 255), thickness=3)

    coord_dict = {
        'slide_filepath': str(slide_filepath),
        'magnification': magnification,
        'magnification_level0': level0_magnification,
        'num_row': num_step_y,
        'num_col': num_step_x,
        'patch_size': patch_size,
        'patch_size_level0': patch_size_level0,
        'num_patches': len(coord_list),
        'coords': coord_list
    }
    with open(coord_dir / f'{filename}.json', 'w', encoding='utf-8') as fp:
        json.dump(coord_dict, fp)
    if thumbnail is not None:
        cv2.imwrite(str(overview_dir / f'{filename}.png'), thumbnail)
    print(f"{filename} | mag0: {level0_magnification} | (rows, cols): {num_step_y}, {num_step_x} | "
          f"patch_size: {patch_size} | num_patches: {len(coord_list)}")


def run(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    coord_dir = save_dir / 'coord_true'
    coord_dir.mkdir(parents=True, exist_ok=True)
    if args.overview:
        thumbnail_dir = Path(args.thumbnail_dir)
        overview_dir = save_dir / 'overview'
        overview_dir.mkdir(parents=True, exist_ok=True)
    else:
        thumbnail_dir = None
        overview_dir = None
    if args.save_mask:
        mask_dir = Path(args.mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        mask_dir = None


    try:
        df = pd.read_excel(args.excel_path)
    except Exception as e:
        logging.error(f"Error: {e}")
        return

    required_columns = ['path1', 'patient', 'Slide ID']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"not find '{col}' in Excel")
            return

    base_path_tiff = Path(args.base_path_tiff)
    tiff_filepaths = []
    missing_files = []

    for idx, row in df.iterrows():
        path1 = str(row['path1']).strip()
        patient = str(row['patient']).strip()
        slide_id = str(row['Slide ID']).strip()
        tiff_filename = f"{slide_id}.tiff"
        tiff_filepath = base_path_tiff / path1 / patient / tiff_filename

        if tiff_filepath.is_file():
            tiff_filepaths.append(tiff_filepath)
        else:
            missing_files.append(str(tiff_filepath))

    num_slide = len(tiff_filepaths)
    logging.info(f"Read {num_stlide} slide files from the Excel file.")

    if missing_files:
        logging.warning(f"The following {len (missing_files)} TIFF files were not found:")
        for file in missing_files:
            logging.warning(f" - {file}")

    if num_slide == 0:
        logging.error("There are no TIFF files available for processing. Please check if the Excel file and base path are correct.")
        return

    logging.info("Start slicing processing...")

    def process_slide(slide_tuple):
        slide_idx, slide_filepath = slide_tuple
        filename = slide_filepath.stem
        coord_file = coord_dir / f'{filename}.json'
        if coord_file.exists() and not args.exist_ok:
            logging.info(f"{coold_file} already exists, skip!")
            return
        try:
            logging.info(f"{slide_idx + 1:3}/{num_slide}, processing {filename}...")
            tiling(
                slide_filepath=slide_filepath,
                magnification=args.magnification,
                patch_size=args.patch_size,
                scale_factor=args.scale_factor,
                tissue_thresh=args.tissue_thresh,
                overview_level=args.overview_level,
                coord_dir=coord_dir,
                thumbnail_dir=thumbnail_dir,
                overview_dir=overview_dir,
                mask_dir=mask_dir,
                filename=filename,
                device=device
            )
            logging.info(f"{filename} processing completed！\n")
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_slide, enumerate(tiff_filepaths))

    logging.info("All slides have been processed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_mask', action='store_true', default=True)
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--magnification', type=int, default=20, choices=[40, 20, 10, 5])
    parser.add_argument('--scale_factor', type=int, default=32,
                        help="scale wsi to down-sampled image for judging tissue percent of each patch.")
    parser.add_argument('--tissue_thresh', type=float, default=0.5,
                        help="the ratio of tissue region of a patch")
    parser.add_argument('--overview', action='store_true', default=False,
                        help="save the overview image after tiling if True")
    parser.add_argument('--overview_level', type=int, default=-1,
                        help="the down-sample level of overview image")
    parser.add_argument('--device', type=int, default=0, help="GPU device number to use")
    parser.add_argument('--mask_dir', type=str, default='/home/ARFF/Dataset_FTD/FTD_ANN')
    parser.add_argument('--save_dir', type=str, default='/home/ARFF/data_FTD')
    parser.add_argument('--thumbnail_dir', type=str, default='/home/ARFF/Dataset_FTD/FTD_thumbs')
    parser.add_argument('--wsi_format', type=str, default='.tiff', choices=['.svs', '.tif', '.tiff'])
    parser.add_argument('--excel_path', type=str,  default='/homeARFF/Dataset_FTD/FTD.xlsx')
    parser.add_argument('--base_path_tiff', type=str,  default='/home/Dataset')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()

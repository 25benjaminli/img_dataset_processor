"""
#### This assumes that you have a dataset with the following structure
dataset
|--> train
<br>
    |--> images
    |--> labels

|--> valid
<br>
    |--> images
    |--> labels

|--> test
    |--> images
    |--> labels
data.yaml
"""
import os
import argparse
import utils.restructure_obj
import utils.misc
import shutil
from dataset import Dataset

# get command line arguments



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, help='path to old dataset - within the datasets folder', required=True)
    parser.add_argument('--output', type=str, help='path to new dataset - within the datasets folder', required=True)
    parser.add_argument('--detection_type', type=str, default='object', help='type of detection', choices=['object', 'segmentation', 'classification'])
    parser.add_argument('--format', type=str, default='yolo', help='format of dataset', choices=['yolo', 'coco'])

    parser.add_argument('--split_type', type=str, default='images', help='type of split', choices=['images', 'objects'])
    parser.add_argument('--train_split', type=float, default=0.8, help='train ratio')
    parser.add_argument('--valid_split', type=float, default=0.1, help='valid ratio')
    parser.add_argument('--test_split', type=float, default=0.1, help='test ratio')

    parser.add_argument('--resize_to', type=str, default='416x416', help='resize images in dataset - format: widthxheight')
    parser.add_argument('--contrast', type=str, default='CLAHE', help='contrast type of images in dataset', choices=['CLAHE', 'AHE', ''])
    parser.add_argument('--grayscale', type=bool, default=False, help='grayscale or not')
    # parser.add_argument('--custom_preprocess', type=str, default="", help='path to file with custom preprocessing techniques')

    parser.add_argument('--backgrounds', type=int, default=0, help='maximum ratio of background to total dataset size')
    parser.add_argument('--b_delimiter', type=str, default='_background', help='delimiter for background images')

    # comma delimited list of classes to exclude
    parser.add_argument('--exclude', type=str, default="", help='comma delimited list of classes to exclude, no spaces')

    # only for offline augmentation
    parser.add_argument('--augment_by', type=bool, default=False, help='augment by (multiplier)')

    parser.add_argument('--augment_options', type=str, default="", help='comma delimited list of augmentations to use, no spaces. Available: flip, rotate, blur, noise, brightness, contrast, sharpness, saturation, hue, mosaic')

    parser.add_argument('--synth_aug', type=bool, default=False, help='synth augment or not')
    
    args = parser.parse_args()
    
    ds = Dataset(args)
import requests
import json
import os
import cv2
from PIL import Image
import shutil
import pandas as pd
import random
import numpy as np
from utils import global_vars

def remove_existing_backgrounds(symbol='background'):
    # remove all background images from {global_vars.ds_path}
    for split in os.listdir(global_vars.ds_path):
        if split == 'train' or split == 'valid' or split == 'test':
            for filename in os.listdir(f'{global_vars.ds_path}/{split}/images'):
                first_part, ext = os.path.splitext(filename)
                if symbol in filename:
                    os.remove(f'{global_vars.ds_path}/{split}/images/{filename}')
                    os.remove(f'{global_vars.ds_path}/{split}/labels/{first_part}.txt')

    print(len(os.listdir(f'{global_vars.ds_path}/train/images')) + len(os.listdir(f'{global_vars.ds_path}/valid/images')) + len(os.listdir(f'{global_vars.ds_path}/test/images')))


def clean_roboflow_dataset():
    # get index of "_jpg" 
    for split in os.listdir():
        if split == 'test' or split == 'train' or split == 'valid':
            print(split)
            for image in os.listdir(f'{global_vars.ds_path}/{split}/images'):
                # get index of "_jpg"
                image_name, ext = os.path.splitext(image)
                new_image = image[:image.rfind('_')] + ext
                # replace current image with new image
                # get image path
                new_image_name  = image[:image.rfind('_')]

                os.rename(f'{global_vars.ds_path}/{split}/images/{image}', f'{global_vars.ds_path}/{split}/images/{new_image}')
                os.rename(f'{global_vars.ds_path}/{split}/labels/{image_name}.txt', f'{global_vars.ds_path}/{split}/labels/{new_image_name}.txt')

                found = False
                for comp_name in global_vars.names:
                    if comp_name.lower() in new_image.lower():
                        found = True
                        break
                
                if not found:
                    # retrieve label from corresponding txt file
                    if os.path.exists(f'{global_vars.ds_path}/{split}/labels/{new_image_name}.txt'):   
                        with open(f'{global_vars.ds_path}/{split}/labels/{new_image_name}.txt', 'r') as f:
                            label = f.read()
                            if len(label) > 0:
                                label = int(label.split(' ')[0])
                                # copy image to corresponding folder
                                os.rename(f'{global_vars.ds_path}/{split}/images/{new_image}', f'{global_vars.ds_path}/{split}/images/{global_vars.names[label]}-{new_image}')
                                # copy label to corresponding folder
                                os.rename(f'{global_vars.ds_path}/{split}/labels/{new_image_name}.txt', f'{global_vars.ds_path}/{split}/labels/{global_vars.names[label]}-{new_image_name}.txt')



def check_for_incorrect_labels():
    for split in os.listdir(f'{global_vars.ds_path}'):
        if split == 'train' or split == 'valid' or split == 'test':
            for image in os.listdir(f'{global_vars.ds_path}/{split}/images'):
                matches = False
                for name in global_vars.names:
                    if name.lower() in image.lower():
                        if not os.path.exists(f'{global_vars.ds_path}/{split}/labels/{image[:-4]}.txt'):
                            continue
                        with open(f'{global_vars.ds_path}/{split}/labels/{image[:-4]}.txt', 'r') as f:
                            data = f.read()
                            if len(data) == 0:
                                continue
                            data = int(data.split(' ')[0])

                            if global_vars.names[data] == name:
                                matches = True
                                break
                if not matches:
                    print(f'{global_vars.ds_path}/{split}/images/{image}')


# def rebalance_dataset(**kwargs):
#     print("deleting folders")
#     send_to = kwargs.get('send_to', "final_ds")
#     useCLAHE = kwargs.get('useCLAHE', False)
#     augmentImages = kwargs.get('augment', 0)
#     backgrounds = kwargs.get('backgrounds', 0.1)
#     grayscale = kwargs.get('grayscale', False)
#     data = kwargs.get('data', None)

#     def delete_all_folders():
#         shutil.rmtree("/combined_ds")
#         shutil.rmtree("/org_ds")
#         shutil.rmtree(f"/{send_to}")
#     try:
#         delete_all_folders()
#     except:
#         pass

#     os.mkdir(f"/{send_to}")

#     print("moving everything to combined")
#     move_to_combined()
    
#     print("organizing to global_vars.names")

#     organize_to_global_vars.names(backgrounds, augment=augmentImages)
    
#     ftrain, fval, ftest = get_train_val_test_splits(include_backgrounds = (backgrounds > 0))
#     # check_freqs(ftrain, fval, ftest)
#     print(fval)
#     print(ftest)
    
#     reorganize_to_final(ftrain, fval, ftest, base = send_to, data=data)

#     print("preprocessing")
#     preprocess(useCLAHE = useCLAHE, grayscale = grayscale, base = send_to)

#     # send to zip
#     shutil.make_archive(path + f"/{send_to}", 'zip', path + f"/{send_to}")


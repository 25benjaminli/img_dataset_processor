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

def remove_existing_backgrounds():
    # remove all background images from {global_vars.ds_path}
    """
    remove all background images from {global_vars.ds_path}
    global_vars.ds_path, global_vars.args.b_delimiter
    
    """
    for split in os.listdir(global_vars.ds_path):
        if split == 'train' or split == 'valid' or split == 'test':
            for filename in os.listdir(f'{global_vars.ds_path}/{split}/images'):
                first_part, ext = os.path.splitext(filename)
                if global_vars.args.b_delimiter in filename:
                    os.remove(f'{global_vars.ds_path}/{split}/images/{filename}')
                    os.remove(f'{global_vars.ds_path}/{split}/labels/{first_part}.txt')

    print(len(os.listdir(f'{global_vars.ds_path}/train/images')) + len(os.listdir(f'{global_vars.ds_path}/valid/images')) + len(os.listdir(f'{global_vars.ds_path}/test/images')))


def clean_roboflow_dataset():
    """
    Cleans roboflow dataset to remove .rf extension and preserve filenames
    global_vars.ds_path
    """
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
    """
    If there are labels that do not match the proper image name in the filename, print it
    """
    incorrects = []
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
                    incorrects.append(image)
    
    return incorrects

def delete_all_folders():
    try:
        shutil.rmtree("/combined_ds")
        shutil.rmtree("/org_ds")
        shutil.rmtree(f"{global_vars.send_to}")
    except:
        pass

def move_to_combined():
    for split in os.listdir(f"{global_vars.ds_path}"):
        if split == 'train' or split == 'valid' or split == 'test':
            for image in os.listdir(f"{global_vars.ds_path}/{split}/images"):
                img_name, ext = os.path.splitext(image)
                if len(global_vars.exclude) == 0:
                    shutil.copy(f"{global_vars.ds_path}/{split}/images/{image}", "/combined_ds/images")
                    shutil.copy(f"{global_vars.ds_path}/{split}/labels/{img_name}.txt", "/combined_ds/labels")
                else:
                    for name in global_vars.exclude:
                        if name.lower() in image.lower():
                            shutil.copy(f"{global_vars.ds_path}/{split}/images/{image}", "/combined_ds/images")
                            shutil.copy(f"{global_vars.ds_path}/{split}/labels/{img_name}.txt", "/combined_ds/labels")
                            break
def rm_and_make(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

def organize_to_names():

    names_with_freqs = [0 for i in range(len(global_vars.names))]
    x=0
    rm_and_make("datasets/org_ds")
    for name in global_vars.names:
        os.makedirs(f"datasets/org_ds/{name}")
    
    os.makedirs(f"datasets/org_ds/labels")

    if global_vars.args.backgrounds > 0:
        os.makedirs(f"datasets/org_ds/backgrounds")
    
    for label in os.listdir("datasets/combined_ds/labels"):
        with open(f"datasets/combined_ds/labels/{label}") as file:
            # read first line
            asdf = file.readline()
            if len(asdf) == 0:
                print(label, "is empty")
                continue
            
            asdf = asdf.split(" ")[0]
            # print(asdf)
            numval = int(asdf)
            real_name = global_vars.names[numval]
            
            # move image and label to folder
            names_with_freqs[numval] += 1
            
            name, extension = os.path.splitext(label)
            # assume that the file is already in jpg form

            if not name in global_vars.exclude:
                shutil.copy(f"/combined_ds/images/{label[:-4]}.jpg", f"/org_ds/{real_name}/{name}.jpg")
                shutil.copy(f"/combined_ds/labels/{label}", f"/org_ds/labels/{name}.txt")

                # if augment > 0:
                #     augment_image(curr_name)
                    
                x+=1
            else:
                print(f"{name} is copyrighted, cannot be used")
def get_train_val_test_splits(include_backgrounds=False):

    ftrain, fval, ftest = np.array([]), np.array([]), np.array([])
    # stratify splitting of data
    print("getting splits")
    for name in global_vars.names:
        allFileNames = os.listdir(f"datasets/org_ds/{name}")
        np.random.seed(0)
        np.random.shuffle(allFileNames)

        train, val, test = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8), int(len(allFileNames)*0.9)])

        ftrain = np.concatenate((ftrain, train))
        fval = np.concatenate((fval, val))
        ftest = np.concatenate((ftest, test))

        print(name, len(train), len(val), len(test))
    
    

    if include_backgrounds:
        allFileNames = os.listdir(f"datasets/org_ds/backgrounds")
        np.random.seed(0)
        np.random.shuffle(allFileNames)

        train, val, test = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8), int(len(allFileNames)*0.9)])

        ftrain = np.concatenate((ftrain, train))
        fval = np.concatenate((fval, val))
        ftest = np.concatenate((ftest, test))

    print("final lengths after stratified split: ", len(ftrain), len(fval), len(ftest))
    
    return ftrain, fval, ftest

def reorganize_to_final(ftrain, fval, ftest):
    splits = ['train', 'valid', 'test']
    
    # clear existing final ds
    if os.path.exists(f"datasets/{global_vars.send_to}"):
        shutil.rmtree(f"datasets/{global_vars.send_to}")
    os.mkdir(f"datasets/{global_vars.send_to}")
    
    for split in splits:
        os.mkdir(f"datasets/{global_vars.send_to}")
        os.mkdir(f"datasets/{global_vars.send_to}")

        os.mkdir(f"datasets/{global_vars.send_to}/{split}/images")
        os.mkdir(f"datasets/{global_vars.send_to}/{split}/labels")
        
        if split == 'train':
            curr_split = ftrain
        elif split == 'valid':
            curr_split = fval
        elif split == 'test':
            curr_split = ftest
        
        for img in curr_split:
            classname = None
            for name in global_vars.names:
                if name.lower() in file.lower():
                    classname = name
                    print(classname)
                    break
            
            img_name, ext = os.path.splitext(img)

            if os.path.exists(f"datasets/org_ds/labels/{img_name}.txt"):
                shutil.copy(f"datasets/org_ds/labels/{img_name}.txt", f"datasets/{global_vars.send_to}/{split}/labels/{img_name}.txt")
                shutil.copy(f"datasets/org_ds/{classname}/{file}", f"datasets/{global_vars.send_to}/{split}/images/{file}")
            elif os.path.exists(f"datasets/org_ds/backgrounds/{file}"):
                shutil.copy(f"datasets/org_ds/backgrounds/{file}", f"datasets/{global_vars.send_to}/{split}/images/{file}")
            


    # add data.yaml file
    with open(f"datasets/{global_vars.send_to}/data.yaml", "w") as file:
        file.write(global_vars.datayaml)


    
    
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


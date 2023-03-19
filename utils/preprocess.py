import os
import cv2
from PIL import Image
import shutil
import pandas as pd
import numpy as np

def generate_CLAHE(image_path, output_path, grayscale = False):
    if not grayscale:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (512, 512))
        lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
        lab_planes = list(cv2.split(lab))
        
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(32,32))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(tuple(lab_planes))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (512, 512))
        clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(32,32))
        image = clahe.apply(image)
    
    cv2.imwrite(output_path, image)

def preprocess(useCLAHE = False, grayscale = False, base = 'final_ds'):
    # images are already resized to 512x512
    # apply CLAHE to images
    for split in os.listdir(base):
        if split == 'train' or split == 'test' or split == 'valid':
            for img_path in os.listdir(f'{base}/{split}/images'):
                
                image = cv2.imread(f'{base}/{split}/images/{img_path}')
                image = cv2.resize(image, (512, 512))

                cv2.imwrite(f'{base}/{split}/images/{img_path}', image)
                
                if useCLAHE:
                    generate_CLAHE(f'{base}/{split}/images/{img_path}', f'{base}/{split}/images/{img_path}', grayscale)

                else:
                    cv2.imwrite(f'{base}/{split}/images/{img_path}', image)

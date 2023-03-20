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
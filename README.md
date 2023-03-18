## This repo intends to make it easy to process images for datasets. It aims to allow for comprehensive evaluation of dataset quality and to provide a simple way to generate datasets for training and testing (stratified).


While investigating existing solutions, such as using websites like roboflow, stratified splitting mechanisms (equal distributions across classes), evaluating & deleting certain images, and more were extremely difficult or impossible. 

The point of this repository is to make it easy to process datasets and check their health. The intended use of this repository is **AFTER YOUR DATA IS ANNOTATED AND EXPORTED FROM ROBOFLOW**. 

It aims to have many built in methods that roboflow lacks, including...

1. Proper stratified splitting within each class
2. Greater control over preprocessing and augmentation techniques
3. Automatic scanning and resolution of potentially critical dataset issues, including duplicate image scanning


Notes:

1. This only supports **roboflow-formated datasets**, as roboflow is probably the most popular website for generating datasets. I might add support for other formats later.

Directions:
1. Install dependencies - `pipenv install -r requirements.txt`. Make sure you are using pipenv. If you don't have pipenv, install it with `pip install pipenv`. Afterwards, run `pipenv shell` to enter the virtual environment.
2. Import a dataset (ideally, annotated and already in labeled format - right now only YOLOv5 format is supported)
3. Visit one of the jupyter notebooks and read the instructions there.
4. Run certain cells of the notebook.
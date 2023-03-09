## This repo intends to make it easy to process images for datasets. It aims to allow for comprehensive evaluation of dataset quality and to provide a simple way to generate datasets for training and testing (stratified).


While investigating existing solutions, such as using websites like roboflow, splitting mechanisms and evaluating & deleting certain images was not possible.

Directions:
1. Install dependencies - `pipenv install -r requirements.txt`. Make sure you are using pipenv. If you don't have pipenv, install it with `pip install pipenv`.
2. Import a dataset (ideally, annotated and already in labeled format - right now only YOLOv5 format is supported)
3. Visit one of the jupyter notebooks and read the instructions there.
4. Run certain cells of the notebook.


Notes:
1. Right now, this only runs from jupyter notebooks. I'll probably add python scripts later
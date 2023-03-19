import os
import argparse
import utils.restructure
import utils.global_vars
import utils.misc
import shutil


class Dataset():
    _args = {}
    def __init__(self, args):
        self._args = args
    def get_output_path(self) -> str:
        """
        Where the dataset gets exported when all processing is finished
        """
        return self._args.output
    def get_input_path(self) -> str:
        """
        Initial dataset path
        """
        return self._args.input
    def get_exclude(self) -> list:
        """
        List of classes to exclude from the dataset
        """
        return self._args.exclude.split(",") if self._args.exclude else []
    def get_args(self) -> dict:
        """
        Raw args
        """
        return self._args
    def get_split_values(self) -> dict:
        """
        train_split, valid_split, test_split
        """
        return { "train_split": self._args.train_split, "valid_split": self._args.valid_split, "test_split": self._args.test_split }
    def get_preprocessing(self) -> dict:
        """
        resize_to, contrast, grayscale
        """
        return { "resize_to": self._args.resize_to, "contrast": self._args.contrast, "grayscale": self._args.grayscale }
    def get_detection_type(self) -> str:
        """
        object, segmentation, classification
        """
        return self._args.detection_type
    def get_dataset_format(self) -> str:
        """
        yolo, coco
        """
        return self._args.format
    def get_names(self) -> list:
        """
        List of classes in the dataset
        """
        return self.names
    def get_datayaml(self) -> str:
        """
        data.yaml file
        """
        return self.datayaml
import os
import argparse
import utils.restructure_obj
import utils.global_vars
import utils.misc
import shutil


class Dataset():
    _args = {}
    
    def __init__(self, args):
        self._args = args

        # clean the dataset and check for incorrect labels
        utils.restructure_obj.clean_roboflow_dataset()
        utils.restructure_obj.check_for_incorrect_labels()

        # get names and data.yaml
        utils.misc.get_names_and_yaml()

        if not os.path.exists(f'datasets/{args.input}_copy'):
            # make a copy
            print("making a copy of the current dataset so we don't lose it!")
            shutil.copytree(f'datasets/{args.input}', f'datasets/{args.input}_copy')

    
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
    
    
    def restructure(self) -> None:
        """
        Restructures the dataset
        """
        if self.get_detection_type() == 'obj':
            self.split_dataset_obj()
        
        elif self.get_detection_type() == 'seg':
            self.split_dataset_seg()
        
        elif self.get_detection_type() == 'cls':
            self.split_dataset_cls()
    
    def get_backgrounds(self) -> int:
        """
        maximum ratio of backgrounds to total images in the dataset
        """
        return self.backgrounds
    
    def get_augment(self) -> bool:
        """
        Whether to augment the dataset.
        Reminder to ONLY use if your model does not support online augmentation.
        """
        return self.augment
    
    def get_augment_amount(self) -> int:
        """
        Offline augmentation. Amount to augment the dataset by (multiplier). Images will not be duplicated. 
        ONLY use if your model does not support online augmentation.
        """
        return self.augment_amount
    
    def split_dataset_obj(self) -> None:
        utils.restructure_obj.move_to_combined()

        print("organizing to global_vars.names")

        utils.restructure_obj.organize_to_names(self.get_backgrounds(), augment=augmentImages)
        
        ftrain, fval, ftest = get_train_val_test_splits(include_backgrounds = (backgrounds > 0))
        # check_freqs(ftrain, fval, ftest)
        print(fval)
        print(ftest)
        
        reorganize_to_final(ftrain, fval, ftest, base = send_to, data=data)

        print("preprocessing")
        preprocess(useCLAHE = useCLAHE, grayscale = grayscale, base = send_to)

        # send to zip
        shutil.make_archive(path + f"/{send_to}", 'zip', path + f"/{send_to}")



    
        
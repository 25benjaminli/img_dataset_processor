import os
import argparse
import utils.restructure_obj
import utils.misc
import utils.preprocess
import shutil


class Dataset():
    """
    Takes in command line arguments and provides a suite of tools 
    for manipulating and processing the dataset.
    """
    _args = {}
    
    def __init__(self, args: dict):
        self._args = args

        # clean the dataset and check for incorrect labels
        utils.restructure_obj.clean_roboflow_dataset()
        utils.restructure_obj.check_for_incorrect_labels()

        # get names and data.yaml
        utils.misc.get_names_and_yaml(self)
        self.check_args_validity()

        if not os.path.exists(f'datasets/{args.input}_copy'):
            # make a copy
            print("making a copy of the current dataset so we don't lose it!")
            shutil.copytree(f'datasets/{args.input}', f'datasets/{args.input}_copy')
        

    def check_args_validity(self) -> None:
        assert os.path.exists(f'datasets/{self._args.input}'), f"dataset \"{self._args.input}\" does not exist"
        assert self._args.train_split + self._args.valid_split + self._args.test_split == 1, "train, valid, and test splits must add up to 1"
        assert self._args.backgrounds >= 0, "backgrounds must be greater than 0"
        assert self._args.b_delimiter != "", "background delimiter cannot be empty"
        assert self._args.augment_by >= 0, "augment by must be greater than 0"
        
        return True
    
    def check_stats_of_final(self) -> None:
        """
        Checks stats of final dataset
        """

        utils.misc.check_freqs(self.get_args().output)
    
    def check_stats_of_initial(self) -> None:
        """
        Checks stats of initial dataset
        """
        utils.misc.check_freqs(self.get_args().input)
    
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
        output, input
        train_split, valid_split, test_split
        resize_to, contrast, grayscale
        detection_type, format, datayaml
        backgrounds, b_delimiter, exclude
        augment_by, augment_options, synth_aug
        """

        return self._args
    
    def get_backgrounds(self) -> int:
        """
        maximum ratio of backgrounds to total images in the dataset
        """
        return self._args.backgrounds
    
    def get_augment_by(self) -> int:
        """
        Offline augmentation. Amount to augment the dataset by (multiplier). Images will not be duplicated. 
        ONLY use if your model does not support online augmentation.
        """
        return self._args.augment_by
    
    def get_augment_options(self) -> dict:
        """
        Augmentation options
        """
        return self._args.augment_options
    
    
    def restructure(self) -> None:
        """
        Performs the following operations:
        1. Moves all images & labels to a single folder
        2. Organizes images & labels to their respective folders based on classes. Integrates a specified # of background images
        3. Performs preprocessing
        4. May perform offline augmentation
        5. Splits the dataset into train, valid, and test folders. Stratified by class
        6. Exports the dataset to the output folder

        """
        if self.get_detection_type() == 'obj':
            self.split_dataset_obj()
        
        elif self.get_detection_type() == 'seg':
            self.split_dataset_seg()
        
        elif self.get_detection_type() == 'cls':
            self.split_dataset_cls()
    
    def split_dataset_obj(self) -> None:
        utils.restructure_obj.move_to_combined()

        print("organizing to global_vars.names")

        utils.restructure_obj.organize_to_names()
        
        ftrain, fval, ftest = utils.restructure_obj.get_train_val_test_splits(include_backgrounds = (self.get_backgrounds() > 0))
        
        utils.restructure_obj.reorganize_to_final(ftrain, fval, ftest)

        print("preprocessing")
        utils.preprocess.preprocess()
import os
import argparse
import utils.restructure_obj
import utils.misc
import utils.preprocess
import shutil
import cv2
import numpy as np
from os.path import normpath, basename


class Dataset():
    """
    Takes in command line arguments and provides a suite of tools 
    for manipulating and processing the dataset.
    """

    _args = {}
    names = []
    datayaml = {}
    
    def __init__(self, args: dict):
        self._args = args

        # clean the dataset and check for incorrect labels
        print("cleaning roboflow dataset...")
        self.clean_roboflow_dataset()

        print("removing bgs... - need to store later")
        self.remove_existing_backgrounds()
        
        print("checking for incorrect labels...")
        self.check_for_incorrect_labels()


        # get names and data.yaml
        self.names, self.datayaml = self.get_names_and_yaml()
        
        self.check_args_validity()

        if not os.path.exists(f'datasets/{args.input}_copy'):
            # make a copy
            print("making a copy of the current dataset so we don't lose it!")
            shutil.copytree(f'datasets/{args.input}', f'datasets/{args.input}_copy')

    def remove_all_supp(self) -> None:
        shutil.rmtree('datasets/combined_ds')
        shutil.rmtree('datasets/org_ds')
    
    def get_names_and_yaml(self) -> None:
        """
        Read data.yaml file and retrieve label names and file contents
        """
        base_name = basename(normpath(self._args.input))
        with open(f'datasets/{self._args.input}/data.yaml', 'r') as f:
            data = f.read()

        # remove everything after "roboflow"
        data = data.split('roboflow')[0] # this will be the same...

        # get all names from data.yaml
        names = eval(data.split('names: ')[1])

        # rewrite train, val, test to go to final_ds/{split}/images instead of ../{split}/images
        data = data.replace('../train/images', f'{base_name}/train/images').replace('../valid/images', f'{base_name}/valid/images').replace('../test/images', f'{base_name}/test/images')

        return list(names), data

    def clean_roboflow_dataset(self) -> None:
        for split in os.listdir():
            if split == 'test' or split == 'train' or split == 'valid':
                print(split)
                for image in os.listdir(f'datasets/{self._args.input}/{split}/images'):
                    # get index of "_jpg"
                    image_name, ext = os.path.splitext(image)
                    new_image = image[:image.rfind('_')] + ext
                    # replace current image with new image
                    # get image path
                    new_image_name  = image[:image.rfind('_')]

                    os.rename(f'datasets/{self._args.input}/{split}/images/{image}', f'datasets/{self._args.input}/{split}/images/{new_image}')
                    os.rename(f'datasets/{self._args.input}/{split}/labels/{image_name}.txt', f'datasets/{self._args.input}/{split}/labels/{new_image_name}.txt')

                    found = False
                    for comp_name in self.names:
                        if comp_name.lower() in new_image.lower():
                            found = True
                            break
                    
                    if not found:
                        # retrieve label from corresponding txt file
                        if os.path.exists(f'datasets/{self._args.input}/{split}/labels/{new_image_name}.txt'):   
                            with open(f'datasets/{self._args.input}/{split}/labels/{new_image_name}.txt', 'r') as f:
                                label = f.read()
                                if len(label) > 0:
                                    label = int(label.split(' ')[0])
                                    # copy image to corresponding folder
                                    os.rename(f'datasets/{self._args.input}/{split}/images/{new_image}', f'datasets/{self._args.input}/{split}/images/{self.names[label]}-{new_image}')
                                    # copy label to corresponding folder
                                    os.rename(f'datasets/{self._args.input}/{split}/labels/{new_image_name}.txt', f'datasets/{self._args.input}/{split}/labels/{self.names[label]}-{new_image_name}.txt')

    def check_for_incorrect_labels(self) -> None:
        incorrects = []
        for split in os.listdir(f'datasets/{self._args.input}'):
            if split == 'train' or split == 'valid' or split == 'test':
                for image in os.listdir(f'datasets/{self._args.input}/{split}/images'):
                    matches = False
                    for name in self.names:
                        if name.lower() in image.lower():
                            if not os.path.exists(f'datasets/{self._args.input}/{split}/labels/{image[:-4]}.txt'):
                                continue
                            with open(f'datasets/{self._args.input}/{split}/labels/{image[:-4]}.txt', 'r') as f:
                                data = f.read()
                                if len(data) == 0:
                                    continue
                                data = int(data.split(' ')[0])

                                if self.names[data] == name:
                                    matches = True
                                    break
                    if not matches:
                        incorrects.append(image)
        
        return incorrects
    
    def remove_existing_backgrounds(self) -> None:
        """
        remove all background images from {self._args.input}
        self._args.input, global_vars.args.b_delimiter
        
        """
        for split in os.listdir(f'datasets/{self._args.input}'):
            if split == 'train' or split == 'valid' or split == 'test':
                for filename in os.listdir(f'datasets/{self._args.input}/{split}/images'):
                    first_part, ext = os.path.splitext(filename)
                    if self._args.b_delimiter in filename:
                        os.remove(f'datasets/{self._args.input}/{split}/images/{filename}')
                        try:
                            os.remove(f'datasets/{self._args.input}/{split}/labels/{first_part}.txt')
                        except:
                            continue

        print(len(os.listdir(f'datasets/{self._args.input}/train/images')) + len(os.listdir(f'datasets/{self._args.input}/valid/images')) + len(os.listdir(f'datasets/{self._args.input}/test/images')))
    
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

        utils.misc.check_freqs(self._args.output)
    
    def check_stats_of_initial(self) -> None:
        """
        Checks stats of initial dataset
        """
        utils.misc.check_freqs(self._args.input)
    
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
        resize_to, contrast, grayscale,
        detection_type, format, datayaml,
        backgrounds, b_delimiter, exclude,
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
    
    def preprocess(self) -> None:
        """
        performs different preprocessing techniques with specified args
        """

        base = f'datasets/{self.get_output_path()}'
        for split in os.listdir(f'{base}'):
            if split == 'train' or split == 'test' or split == 'valid':
                for img_path in os.listdir(f'{base}/{split}/images'):
                    image = cv2.imread(f'{base}/{split}/images/{img_path}')
                    w, h = [int(x) for x in self._args.resize_to.split("x")]
                    image = cv2.resize(image, (w, h))
                    
                    cv2.imwrite(f'{base}/{split}/images/{img_path}', image)
                    
                    if self._args.contrast == 'CLAHE':
                        utils.preprocess.generate_CLAHE(f'{base}/{split}/images/{img_path}', f'{base}/{split}/images/{img_path}', self._args.grayscale)

                    else:
                        cv2.imwrite(f'{base}/{split}/images/{img_path}', image)

    def move_to_combined(self) -> None:
        """
        Moves all images & labels to a single folder
        """
        print("moving images and labels to a single folder")
        
        if os.path.exists("datasets/combined_ds"):
            shutil.rmtree("datasets/combined_ds")
        os.mkdir("datasets/combined_ds")
        os.mkdir("datasets/combined_ds/images")
        os.mkdir("datasets/combined_ds/labels")

        for split in os.listdir(f"datasets/{self.get_input_path()}"):
            if split == "train" or split == "valid" or split == "test":
                for image in os.listdir(f"datasets/{self.get_input_path()}/{split}/images"):
                    img_name, ext = os.path.splitext(image)
                    if len(self.get_exclude()) == 0:
                        shutil.copy(f"datasets/{self.get_input_path()}/{split}/images/{image}", "datasets/combined_ds/images")
                        shutil.copy(f"datasets/{self.get_input_path()}/{split}/labels/{img_name}.txt", "datasets/combined_ds/labels")
                    else:
                        for name in self.get_exclude():
                            if name.lower() in image.lower():
                                shutil.copy(f"datasets/{self.get_input_path()}/{split}/images/{image}", "datasets/combined_ds/images")
                                shutil.copy(f"datasets/{self.get_input_path()}/{split}/labels/{img_name}.txt", "datasets/combined_ds/labels")
                                break
    
    def organize_to_names(self) -> None:
        names_with_freqs = [0 for i in range(len(self.names))]
        x=0
        utils.restructure_obj.rm_and_make("datasets/org_ds")
        for name in self.names:
            os.makedirs(f"datasets/org_ds/{name}")
        
        os.makedirs(f"datasets/org_ds/labels")

        if self.get_backgrounds() > 0:
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
                real_name = self.names[numval]
                
                # move image and label to folder
                names_with_freqs[numval] += 1
                
                name, extension = os.path.splitext(label)
                # assume that the file is already in jpg form

                if not name in self.get_exclude():
                    shutil.copy(f"datasets/combined_ds/images/{name}.jpg", f"datasets/org_ds/{real_name}/{name}.jpg")
                    shutil.copy(f"datasets/combined_ds/labels/{label}", f"datasets/org_ds/labels/{name}.txt")
                        
                    x+=1

    def get_train_val_test_splits(self) -> None:
        ftrain, fval, ftest = np.array([]), np.array([]), np.array([])
        # stratify splitting of data
        print("getting splits")
        for name in self.names:
            allFileNames = os.listdir(f"datasets/org_ds/{name}")
            np.random.seed(0)
            np.random.shuffle(allFileNames)

            train, val, test = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8), int(len(allFileNames)*0.9)])

            ftrain = np.concatenate((ftrain, train))
            fval = np.concatenate((fval, val))
            ftest = np.concatenate((ftest, test))

            print(name, len(train), len(val), len(test))
        
        

        if self.get_backgrounds() > 0:
            allFileNames = os.listdir(f"datasets/org_ds/backgrounds")
            np.random.seed(0)
            np.random.shuffle(allFileNames)

            train, val, test = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8), int(len(allFileNames)*0.9)])

            ftrain = np.concatenate((ftrain, train))
            fval = np.concatenate((fval, val))
            ftest = np.concatenate((ftest, test))

        print("final lengths after stratified split: ", len(ftrain), len(fval), len(ftest))
        
        return ftrain, fval, ftest
    
    def reorganize_to_final(self, ftrain, fval, ftest):
        splits = ['train', 'valid', 'test']
        
        # clear existing final ds
        if os.path.exists(f"datasets/{self.get_output_path()}"):
            shutil.rmtree(f"datasets/{self.get_output_path()}")
        os.mkdir(f"datasets/{self.get_output_path()}")
        
        for split in splits:

            os.mkdir(f"datasets/{self.get_output_path()}/{split}")
            os.mkdir(f"datasets/{self.get_output_path()}/{split}/images")
            os.mkdir(f"datasets/{self.get_output_path()}/{split}/labels")
            
            if split == 'train':
                curr_split = ftrain
            elif split == 'valid':
                curr_split = fval
            elif split == 'test':
                curr_split = ftest
            
            for img in curr_split:
                classname = None
                for name in self.names:
                    if name.lower() in img.lower():
                        classname = name
                        break
                
                img_name, ext = os.path.splitext(img)

                if os.path.exists(f"datasets/org_ds/labels/{img_name}.txt"):
                    shutil.copy(f"datasets/org_ds/labels/{img_name}.txt", f"datasets/{self.get_output_path()}/{split}/labels/{img_name}.txt")
                    shutil.copy(f"datasets/org_ds/{classname}/{img}", f"datasets/{self.get_output_path()}/{split}/images/{img}")
                elif os.path.exists(f"datasets/org_ds/backgrounds/{img}"):
                    shutil.copy(f"datasets/org_ds/backgrounds/{img}", f"datasets/{self.get_output_path()}/{split}/images/{img}")
                


        # add data.yaml file
        with open(f"datasets/{self.get_output_path()}/data.yaml", "w") as file:
            file.write(self.datayaml)


    def split_dataset_obj(self) -> None:
        self.remove_all_supp()
        
        self.move_to_combined()
        
        print("organizing to names")
        self.organize_to_names()
        
        ftrain, fval, ftest = self.get_train_val_test_splits()
        
        self.reorganize_to_final(ftrain, fval, ftest)
        
        print("preprocessing")
        self.preprocess()
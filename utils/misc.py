import os
import shutil
from utils import global_vars
from os.path import normpath, basename

def reset_dataset_to_orig():
    shutil.rmtree(f'{global_vars.ds_path}')
    shutil.copytree(f'{global_vars.ds_path}_copy', f'{global_vars.ds_path}')

def get_names_and_yaml():
    # doing stuff
    # read data.yaml from {old_base}
    base_name = basename(normpath(global_vars.ds_path))
    with open(f'{global_vars.ds_path}/data.yaml', 'r') as f:
        data = f.read()

    # remove everything after "roboflow"
    data = data.split('roboflow')[0] # this will be the same...

    # get all names from data.yaml
    names = eval(data.split('names: ')[1])

    # rewrite train, val, test to go to final_ds/{split}/images instead of ../{split}/images
    data = data.replace('../train/images', f'{base_name}/train/images').replace('../valid/images', f'{base_name}/valid/images').replace('../test/images', f'{base_name}/test/images')
    

    return list(names), data


def check_freqs():
    splits = [os.listdir(f'{global_vars.ds_path}/train/images'), os.listdir(f'{global_vars.ds_path}/valid/images'), os.listdir(f'{global_vars.ds_path}/test/images')]
    print("printing frequency information")
    trainlen, vallen, testlen = len(os.listdir(f'{global_vars.ds_path}/train/images')), len(os.listdir(f'{global_vars.ds_path}/valid/images')), len(os.listdir(f'{global_vars.ds_path}/test/images'))
    print(trainlen, vallen, testlen)
    print("total dataset size: ", trainlen + vallen + testlen)

    print("train", trainlen, "valid", vallen, "test", testlen)

    splitarr = ['train', 'valid', 'test']
    # dictionary storing percentage frequencies across splits
    freqs = {}
    for name in global_vars.names:
        freqs[name] = [0,0,0]
    
    freqs["background"] = [0,0,0]

    
    for i in range(len(splits)):
        for name in global_vars.names:
            for file in splits[i]:
                if name.lower() in file.lower():
                    freqs[name][i] += 1

    
    print("background frequencies")
    # check background frequencies
    for i in range(len(splits)):
        for file in splits[i]:
            if "background" in file:
                freqs["background"][i] += 1
    
    print(freqs)
    
    for name in global_vars.names:
        if sum(freqs[name]) == 0:
            continue
        for i in range(len(freqs[name])):
            print(name, splitarr[i], freqs[name][i]/sum(freqs[name]), freqs[name][i])
        
        # print total number of images
        print(f"total number of {name}: ", sum(freqs[name]))

    

    for i in range(len(freqs["background"])):
        if sum(freqs["background"]) == 0:
            continue
        print("background", splitarr[i], freqs["background"][i]/sum(freqs["background"]), freqs["background"][i])

    print("total background images", sum(freqs["background"]))
    # print proportion of background images
    print("proportion of background images: ", sum(freqs["background"])/(trainlen+vallen+testlen))
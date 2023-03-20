import os
import argparse
import utils.misc
import utils.restructure_obj
import shutil
import utils
# get command line arguments
from dataset import Dataset


parser = argparse.ArgumentParser()

parser.add_argument('--input', type=str, help='path to dataset - within the datasets folder', required=True)


args = parser.parse_args()

# check if input dataset exists
if not os.path.exists(f'datasets/{args.input}'):
    print(f"dataset \"{args.input}\" does not exist")
    exit()

if not os.path.exists(f'datasets/{args.input}_copy'):
    # make a copy
    print("making a copy of the current dataset so we don't lose it!")
    shutil.copytree(f'datasets/{args.input}', f'datasets/{args.input}_copy')

ds = Dataset(args.input)

ds.check_stats_of_initial()
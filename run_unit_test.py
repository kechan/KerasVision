import argparse
import os, shutil, glob
from subprocess import check_call
import sys

from utils import Params

from data.augmentation.CustomImageDataGenerator import * 

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='CoinSee',
                    help="Top Directory for the entire project")



def run_unit_test(model_dir, data_dir):
     
    cmd = "{python} train.py --model_dir {model_dir} --data_dir {data_dir}".format(python=PYTHON,
            model_dir=model_dir, data_dir=data_dir)

    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parser.parse_args()
    project_dir = args.project_dir

    home_dir = os.getenv("HOME")
    project_dir = os.path.join(home_dir, "Documents", project_dir)

    experiment_dir = os.path.join(project_dir, "experiments") 
    top_data_dir = os.path.join(project_dir, "data")

    # Unit Test 1: hdf5 and using .fit(...) with logistic regression

    #test_dirs = ["unit_test1", "unit_test2", "unit_test3]
    test_dirs = []
    #data_folders = ["224x224_cropped_hdf5", "64x64_cropped_hdf5", "64x64_cropped_hdf5"]
    data_folders = []

    for test_dir, data_folder in zip(test_dirs, data_folders):
    
        model_dir = os.path.join(experiment_dir, test_dir)
        data_dir = os.path.join(top_data_dir, data_folder)
        run_unit_test(model_dir, data_dir)

        # remove the weights directories
        for dirs in glob.glob(os.path.join(model_dir, "weights_*")):
            shutil.rmtree(dirs)

    




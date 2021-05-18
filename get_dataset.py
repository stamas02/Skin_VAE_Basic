import os.path

from src import web_helper
import zipfile

import argparse
from os import listdir
from os.path import isfile, join

DOWNLOAD_URL = "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
DEFALUT_FOLDER_NAME = "ISBI2016_ISIC_Part1_Test_Data"

def parseargs():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='Downloads the ISIC 2016 Challenge training dataset')
    parser.add_argument('--output', '-o', type=str, help="path to the directory where you wanna store the dataset")

    args = parser.parse_args()
    return args

def download_dataset(output):
    output_file = os.path.join(output, "data.zip")

    web_helper.download_url(DOWNLOAD_URL, output_file)
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output)

    os.remove(output_file)

    source_dir = os.path.join(output, DEFALUT_FOLDER_NAME)
    destination_dir = os.path.join(output, "unknown")
    os.rename(source_dir, destination_dir)


if __name__ == "__main__":
    args = parseargs()
    download_dataset(**args.__dict__)
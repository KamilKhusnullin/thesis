import os
import subprocess
import sys
import re
from pathlib import Path

sys.path.append('D:\\thesis')
import image_to_latex.data.utils as utils

METADATA = {
    "im2latex_formulas.norm.lst": "https://zenodo.org/record/56198/files/im2latex_formulas.lst?download=1",
    "im2latex_validate_filter.lst": "https://zenodo.org/record/56198/files/im2latex_validate.lst?download=1",
    "im2latex_train_filter.lst": "https://zenodo.org/record/56198/files/im2latex_train.lst?download=1",
    "im2latex_test_filter.lst": "https://zenodo.org/record/56198/files/im2latex_test.lst?download=1",
    "formula_images.tar.gz": "https://zenodo.org/record/56198/files/formula_images.tar.gz?download=1",
}
PROJECT_DIRNAME = Path(__file__).resolve().parents[1]
DATA_DIRNAME = PROJECT_DIRNAME / "data"
RAW_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images"
PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"


def main():
    DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
    cur_dir = os.getcwd()
    os.chdir(DATA_DIRNAME)

    # Download images and grouth truth files
    for filename, url in METADATA.items():
        if not Path(filename).is_file():
            utils.download_url(url, filename)

    # Unzip
    if not RAW_IMAGES_DIRNAME.exists():
        RAW_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        utils.extract_tar_file("formula_images.tar.gz")

    # Extract regions of interest
    if not PROCESSED_IMAGES_DIRNAME.exists():
        PROCESSED_IMAGES_DIRNAME.mkdir(parents=True, exist_ok=True)
        print("Cropping images...")
        for image_filename in RAW_IMAGES_DIRNAME.glob("*.png"):
            cropped_image = utils.crop(image_filename, padding=8)
            if not cropped_image:
                continue
            cropped_image.save(PROCESSED_IMAGES_DIRNAME / image_filename.name)

    # Clean the ground truth file
    print('Cleaning data...')
    fr = open('../data/im2latex_formulas.norm.lst', 'r')
    fw = open('../data/im2latex_formulas.norm.new.lst', 'w')

    str = fr.read()
    str = str.replace('\left(', '(')
    str = str.replace('\\right)', ')')
    str = str.replace('\left[', '[')
    str = str.replace('\\right]', ']')
    str = str.replace('\left{', '{')
    str = str.replace('\\right}', '}')
    str = re.sub(r'\\vspace.?\{.*?\}', '', str)
    str = re.sub(r'\\hspace.?\{.*?\}', '', str)
    fw.write(str)

    fw.close()
    fr.close()

    # Build vocabulary
    if not VOCAB_FILE.is_file():
        print("Building vocabulary...")
        all_formulas = utils.get_all_formulas('im2latex_formulas.norm.new.lst')
        _, train_formulas = utils.get_split(all_formulas, "im2latex_train_filter.lst")
        tokenizer = utils.Tokenizer()
        tokenizer.train(train_formulas)
        tokenizer.save(VOCAB_FILE)
    os.chdir(cur_dir)


if __name__ == "__main__":
    main()

import argparse
import glob
from itk.support.types import PixelTypes
import skimage.transform
import sklearn
import monai
import itk
import shutil
import numpy as np
import util


from monai.data import partition_dataset
from pathlib import Path

# Any and all data goes here
DATA_DIR = Path("data")

# Original untouched data
RAW_DATA_DIR = DATA_DIR / "raw_data"

# Data separated into train, val, and test
EXTRACT_DIR = DATA_DIR / "extracted_data"

# Original data as given was separated into 3 directories
RAW_PROJ_DIR = RAW_DATA_DIR / "stage2-10K_500_proj"
RAW_REC_DIR = RAW_DATA_DIR / "stage2-10K_500_rec"
RAW_SOURCE_DIR = RAW_DATA_DIR / "stage2-10K-500_source"

# Synthetic dataset for use before actual data comes in
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
SYNTHETIC_SINO_DIR = SYNTHETIC_DATA_DIR / "sinograms"
SYNTHETIC_INCOMPLETE_SINO_DIR = SYNTHETIC_DATA_DIR / "incomplete_sinograms"

# Train directories
TRAIN_DATA_DIR = EXTRACT_DIR / "train"
TRAIN_X_DIR = TRAIN_DATA_DIR / "x"
TRAIN_Y_DIR = TRAIN_DATA_DIR / "y"

# Validation directories
VAL_DATA_DIR = EXTRACT_DIR / "val"
VAL_X_DIR = VAL_DATA_DIR / "x"
VAL_Y_DIR = VAL_DATA_DIR / "y"

# Test directories
TEST_DATA_DIR = EXTRACT_DIR / "test"
TEST_X_DIR = TEST_DATA_DIR / "x"
TEST_Y_DIR = TEST_DATA_DIR / "y"

SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_SINO_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_INCOMPLETE_SINO_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_X_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_Y_DIR.mkdir(parents=True, exist_ok=True)
VAL_X_DIR.mkdir(parents=True, exist_ok=True)
VAL_Y_DIR.mkdir(parents=True, exist_ok=True)
TEST_X_DIR.mkdir(parents=True, exist_ok=True)
TEST_Y_DIR.mkdir(parents=True, exist_ok=True)

# ITK stuff
PixelType = itk.F
SourceDataDim = 3
ExtractedDataDim = 2

SourceImageType = itk.Image[PixelType, SourceDataDim]
ImageType = itk.Image[PixelType, ExtractedDataDim]

NETWORK_INPUT_SIZE = (256, 256)


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    """

    my_parser = argparse.ArgumentParser()
    sub_parsers = my_parser.add_subparsers(dest="sub_command")

    sub_parsers.add_parser("generate_synthetic_data")
    sub_parsers.add_parser("extract", help="Extract data")
    sub_parsers.add_parser("extract_synthetic_data")

    return my_parser


def _extract_datalist(dl, xoutdir, youtdir):
    for d in dl:
        xfname = Path(d["x"]).name
        yfname = Path(d["y"]).name

        x = itk.imread(d["x"])
        y = itk.imread(d["y"])

        itk.imwrite(x, str(xoutdir / xfname))
        itk.imwrite(y, str(youtdir / yfname))


# TODO: Make this modular on axis?
def remove_random_cols_of_image(sino: itk.Image, frac_to_remove=0.1):
    sino_arr = itk.array_from_image(sino)
    num_rows, num_cols = sino_arr.shape

    num_samples_to_remove = int(frac_to_remove * num_cols)
    cols_to_remove = np.random.choice(
        range(num_cols), size=num_samples_to_remove, replace=False
    )

    sino_arr[:, cols_to_remove] = np.zeros((num_rows, 1))

    return itk.image_from_array(sino_arr)


def convert_to_sinogram(im: itk.Image, theta=None):
    im_size = im.GetLargestPossibleRegion().GetSize()
    if len(im_size) == 3:
        if im_size[2] != 1:
            raise ValueError(
                "im argument must either be a 2d image or a 3d image with length 1 in final dimension"
            )

        im = util.extract_slice(im, 0, PixelType=PixelType)
        im_size = im.GetLargestPossibleRegion().GetSize()

    ncol, nrow = im_size
    if theta is None:
        theta = np.linspace(0, 180, max(ncol, nrow))

    sino_arr = skimage.transform.radon(im, theta=theta, circle=False)
    return itk.image_from_array(sino_arr)


def generate_synthetic_data(
    source_dir=RAW_SOURCE_DIR,
    complete_sino_dir=SYNTHETIC_SINO_DIR,
    incomplete_sino_dir=SYNTHETIC_INCOMPLETE_SINO_DIR,
):
    """
    Takes a directory with reconstructed images as input, then generates
    matched pairs of complete sinograms and incomplete sinograms. Writes them
    out to `complete_sino_dir` and `incomplete_sino_dir` respectively.
    """
    if isinstance(complete_sino_dir, str):
        complete_sino_dir = Path(complete_sino_dir)

    if isinstance(incomplete_sino_dir, str):
        incomplete_sino_dir = Path(incomplete_sino_dir)

    for p in [complete_sino_dir, incomplete_sino_dir]:
        if p.exists():
            shutil.rmtree(str(p))
            p.mkdir(parents=True)

    paths = sorted(glob.glob(str(source_dir / "*.dcm")))
    for i, path in enumerate(paths):
        if i % 20 == 0:
            print("On image {} of {}".format(i, len(paths)))

        source_im = itk.imread(path, pixel_type=PixelType)
        complete_sino = convert_to_sinogram(source_im)
        incomplete_sino = remove_random_cols_of_image(complete_sino)

        # Now write to disk
        stem = Path(path).stem

        itk.imwrite(complete_sino, str(complete_sino_dir / (stem + ".mha")))
        itk.imwrite(incomplete_sino, str(incomplete_sino_dir / (stem + ".mha")))


# Convert all of the images into incomplete sinograms
# extract data with correct params
def extract_data(
    sino_x_dir,
    sino_y_dir,
    train_split=0.6,
    val_split=0.2,
    test_split=0.2,
):
    """
    Note that if the splits dont add up to 1, the ratio is taken
    """

    all_fnames = [fp.name for fp in sino_x_dir.glob("*.*") if fp.is_file()]

    datalist = [
        {
            "x": str(sino_x_dir / fn),
            "y": str(sino_y_dir / fn),
        }
        for fn in all_fnames
    ]
    train_part, val_part, test_part = partition_dataset(
        datalist, [train_split, val_split, test_split]
    )

    # Clear out the directories before filling them again.
    for p in [
        TRAIN_X_DIR,
        TRAIN_Y_DIR,
        VAL_X_DIR,
        VAL_Y_DIR,
        TEST_X_DIR,
        TEST_Y_DIR,
    ]:
        if p.exists():
            shutil.rmtree(str(p))
            p.mkdir(parents=True)

    _extract_datalist(train_part, TRAIN_X_DIR, TRAIN_Y_DIR)
    _extract_datalist(val_part, VAL_X_DIR, VAL_Y_DIR)
    _extract_datalist(test_part, TEST_X_DIR, TEST_Y_DIR)


def main():
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    if args.sub_command == "generate_synthetic_data":
        generate_synthetic_data()

    # elif args.sub_command == "extract":
    #     extract_data()

    elif args.sub_command == "extract_synthetic_data":
        extract_data(
            SYNTHETIC_SINO_DIR, SYNTHETIC_INCOMPLETE_SINO_DIR
        )


if __name__ == "__main__":
    main()

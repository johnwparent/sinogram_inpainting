import argparse
import pathlib as path
import glob
import sklearn
import monai
import itk

from monai.data import partition_dataset

# Any and all data goes here
DATA_DIR = path.Path("data")

# Original untouched data
RAW_DATA_DIR = DATA_DIR / "raw_data"

# Data separated into train, val, and test
EXTRACT_DIR = DATA_DIR / "extracted_data"

# Original data as given was separated into 3 directories
RAW_PROJ_DIR = RAW_DATA_DIR / "stage2-10K_500_proj"
RAW_REC_DIR = RAW_DATA_DIR / "stage2-10K_500_proj"
RAW_SOURCE_DIR = RAW_DATA_DIR / "stage2-10K-500_source"

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


TRAIN_X_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_Y_DIR.mkdir(parents=True, exist_ok=True)
VAL_X_DIR.mkdir(parents=True, exist_ok=True)
VAL_Y_DIR.mkdir(parents=True, exist_ok=True)
TEST_X_DIR.mkdir(parents=True, exist_ok=True)
TEST_Y_DIR.mkdir(parents=True, exist_ok=True)


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    """

    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract ",
    )

    return my_parser



def _extract_datalist(dl, xoutdir, youtdir):
    for d in dl:
        xfname = path.Path(d["x"]).name
        yfname = path.Path(d["y"]).name

        x = itk.imread(d["x"])
        y = itk.imread(d["y"])

        itk.imwrite(x, str(xoutdir / xfname))
        itk.imwrite(y, str(youtdir / yfname))


def extract_data(xdir=RAW_REC_DIR, ydir=RAW_SOURCE_DIR, train_split=0.6, val_split=0.2, test_split=0.2):
    # Note that if the splits dont add up to 1, the ratio is taken

    # Assumes that the filenames in both above directories match.
    # TODO: Make sure all of the names match
    xfilepaths = sorted(glob.glob(str(xdir / "*.dcm")))
    yfilepaths = sorted(glob.glob(str(ydir / "*.dcm")))

    print(xfilepaths[:5])
    print(yfilepaths[:5])

    datalist = [{"x": xfp, "y": yfp} for xfp, yfp in zip(xfilepaths, yfilepaths)]
    train_dl, val_dl, test_dl = partition_dataset(datalist, [train_split, val_split, test_split])

    _extract_datalist(train_dl, TRAIN_X_DIR, TRAIN_Y_DIR)
    _extract_datalist(val_dl, VAL_X_DIR, VAL_Y_DIR)
    _extract_datalist(test_dl, TEST_X_DIR, TEST_Y_DIR)



def main():
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    if args.extract:
        extract_data()

if __name__ == "__main__":
    main()

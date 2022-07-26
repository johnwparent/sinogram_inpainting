# Sources:
# https://github.com/astra-toolbox/astra-toolbox/blob/3cdfbe01f4f0dc1eb2df5224f10dd53d7804a642/samples/python/s004_cpu_reconstruction.py


import argparse
import glob
import os
from skimage.exposure.exposure import rescale_intensity
import skimage.transform
import itk
import shutil
import numpy as np
import util

import monai
import torch
import itk
import numpy as np
import matplotlib.pyplot as plt
import shutil


from mpl_toolkits.axes_grid1 import ImageGrid
from monai.data import partition_dataset
from pathlib import Path

from monai.data import ArrayDataset, DataLoader
from dl_with_unet import *
from skimage import exposure

from monai.transforms import ScaleIntensity

# Any and all data goes here
DATA_DIR = Path("data")

# Original untouched data
RAW_DATA_DIR = DATA_DIR / "raw_data"

# Data separated into train, val, and test
EXTRACT_DIR = DATA_DIR / "extracted_data"

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

# ITK stuff
PixelType = itk.F
SourceDataDim = 3
ExtractedDataDim = 2

SourceImageType = itk.Image[PixelType, SourceDataDim]
ImageType = itk.Image[PixelType, ExtractedDataDim]

# ITK stuff
PixelType = itk.F
InputDataDim = 3
ExtractedDataDim = 2

RawDataImageType = itk.Image[PixelType, InputDataDim]
ImageType = itk.Image[PixelType, ExtractedDataDim]


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    """

    my_parser = argparse.ArgumentParser()
    sub_parsers = my_parser.add_subparsers(dest="sub_command")

    sub_parser_extract = sub_parsers.add_parser("extract", help="Extract data")
    sub_parser_extract.add_argument(
        "--sino_x_dir",
        action="store",
        type=Path,
        help="Directory to the incomplete sinogram data"
    )
    sub_parser_extract.add_argument(
        "--sino_y_dir",
        action="store",
        type=Path,
        help="Directory to the y sinogram data without missing portions"
    )
    sub_parser_extract.add_argument(
        "--no_shuffle",
        action="store_true",
        help="Dont shuffle the filenames before assigning to train, test, val.",
    )
    sub_parsers.add_parser("train", help="Train a fresh model")
    sub_parsers.add_parser("test", help="Test a trained model")

    sub_parser_predict_and_show = sub_parsers.add_parser("predict_and_show")
    # TODO: Do mutual exclusion properly here
    # group = sub_parser_predict_and_show.add_mutually_exclusive_group(required=True)
    sub_parser_predict_and_show.add_argument("--random_image", action="store_true")
    sub_parser_predict_and_show.add_argument("--image_path", action="store")
    sub_parser_predict_and_show.add_argument("--gt_path", action="store")

    sub_parser_predict_and_write = sub_parsers.add_parser("predict_and_write")
    sub_parser_predict_and_write.add_argument(
        "--file_patterns",
        nargs="+",
        action="store",
    )
    sub_parser_predict_and_write.add_argument(
        "--save_path",
        type=Path,
        action="store",
    )
    sub_parser_predict_and_write.add_argument(
        "--keep_file_structure",
        action="store_true",
    )

    return my_parser


def _extract_datalist(dl, xoutdir, youtdir):
    for d in dl:
        xfname = Path(d["x"]).name
        yfname = Path(d["y"]).name

        x = itk.imread(d["x"], pixel_type=PixelType)
        y = itk.imread(d["y"], pixel_type=PixelType)

        itk.imwrite(x, str(xoutdir / xfname))
        itk.imwrite(y, str(youtdir / yfname))


def extract_data(
    sino_x_dir,
    sino_y_dir,
    train_split=0.6,
    val_split=0.2,
    test_split=0.2,
    random_shuffle=True
):
    """
    Note that if the splits dont add up to 1, the ratio is taken
    """

    all_fnames = np.array([fp.name for fp in sino_x_dir.glob("*.*") if fp.is_file()])
    if random_shuffle:
        np.random.shuffle(all_fnames)


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


def get_data_loader(
    xdir, ydir, xtransforms=None, ytransforms=None, batch_size=BATCH_SIZE
):
    xs = sorted(glob.glob(str(xdir / "*")))
    ys = sorted(glob.glob(str(ydir / "*")))

    # Should have the same filenames in both
    assert [Path(p).name for p in xs] == [Path(p).name for p in ys]

    dataset = ArrayDataset(xs, xtransforms, ys, ytransforms)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=torch.cuda.is_available()
    )


def _display_images_in_grid(images, grid_shape, titles=None):
    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=grid_shape, axes_pad=0.3)

    for i, (ax, im) in enumerate(zip(grid, images)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        if titles is not None:
            title = titles[i]
        ax.set_title(title)

    plt.show()


def _convert_to_array_if_itk_image(image):
    if isinstance(image, itk.Image):
        return itk.array_from_image(image)


# TODO:
# In doing so, I found out that ensuretype doesnt actually cast to a tensor when itk image is passed. Check out monai tests for ensuretype
# TODO: Display these correctly with metrics printed.
def display_image_truth_and_prediction(
    sino_input,
    sino_gt,
    sino_pred,
    eval_metric=monai.metrics.MSEMetric(),
    back_projector=None,
):
    rescale_intensity = ScaleIntensity()
    sino_input = rescale_intensity(_convert_to_array_if_itk_image(sino_input))
    sino_gt = rescale_intensity(_convert_to_array_if_itk_image(sino_gt))
    sino_pred = rescale_intensity(_convert_to_array_if_itk_image(sino_pred))

    images_to_display = [sino_input, sino_gt, sino_pred]
    titles = ["Input Sinogram", "GT Sinogram", "Predicted Sinogram"]
    grid_shape = (1, 3)

    metric_value_sinos = eval_metric(torch.Tensor(sino_pred).unsqueeze(0), torch.Tensor(sino_gt).unsqueeze(0)).item()
    print("Metric prediction and ground truth sinos:", metric_value_sinos)
    if back_projector is not None:
        rec_input = back_projector(sino_input)
        images_to_display.append(rec_input)
        titles.append("Input Reconstructed")

        rec_gt = back_projector(sino_gt)
        images_to_display.append(rec_gt)
        titles.append("GT Reconstructed")

        rec_pred = back_projector(sino_pred)
        images_to_display.append(rec_pred)
        titles.append("Prediction Reconstructed")

        grid_shape = (2, 3)

    _display_images_in_grid(images_to_display, grid_shape, titles)

# TODO: Make this modular so gt is not required
def predict_image_and_show(
    model,
    image,
    device,
    gt,
    eval_metric=monai.metrics.MSEMetric(),
    back_projector=None,
):

    prediction = run_inference(model, image, device)
    # Display
    display_image_truth_and_prediction(
        image, gt, prediction, eval_metric, back_projector
    )


def predict_and_write_to_disk(model, filepaths, save_path_root, device, keep_file_structure=True):
    # TODO: licensing here
    commonpath = None
    if keep_file_structure and len(filepaths) > 1:
        commonpath = os.path.commonpath(filepaths)

    for fp in filepaths:
        fp = Path(fp)
        im = itk.imread(str(fp))
        inpainted = run_inference(model, im, device)
        # TODO: licensing here
        save_path = save_path_root
        if commonpath is not None:
            parent_dir = fp.parents[0]
            save_path = save_path_root / parent_dir.relative_to(commonpath)

        save_path /= fp.name
        print("Writing inpainted sinogram for", fp.name, "to", save_path)
        itk.imwrite(inpainted, str(save_path))


def get_random_image_and_gt(image_dir, gt_dir):
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)

    if isinstance(gt_dir, str):
        gt_dir = Path(gt_dir)

    fpaths = list(image_dir.glob("*.*"))
    random_im_path = np.random.choice(fpaths, size=1)[0]
    random_im_path = Path(random_im_path)
    image_fname = random_im_path.name

    # Assuming image and gt fnames are the same, just in different dirs
    gt_path = str(gt_dir / image_fname)
    return itk.imread(str(random_im_path)), itk.imread(gt_path)

# TODO: licensing
def get_filepaths_matching_pattern(pattern, recursive=False):
    filepaths = []
    for p in pattern:
        filepaths.extend(glob.glob(p, recursive=recursive))

    return set(filepaths)


def main():
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    if args.sub_command == "extract":
        extract_data(args.sino_x_dir, args.sino_y_dir, random_shuffle=not args.no_shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    # if args.sub_command == "predict_and_show":
    #     model = load_model()
    #     image, gt = None, None
    #     back_projector = None
    #     if args.random_image:
    #         image, gt = get_random_image_and_gt(TEST_X_DIR, TEST_Y_DIR)

    #     elif args.image_path and args.gt_path:
    #         image = itk.imread(args.image_path)
    #         gt = itk.imread(args.gt_path)

    #     else:
    #         raise RuntimeError(
    #             "Please specify either --random_image or --image_path and --gt_path"
    #         )

    #     predict_image_and_show(model, image, device, gt, back_projector=back_projector)

    if args.sub_command == "train":
        model = get_model()
        train_loader = get_data_loader(
            TRAIN_X_DIR,
            TRAIN_Y_DIR,
            xtransforms=train_x_transforms,
            ytransforms=train_y_transforms,
        )
        val_loader = get_data_loader(
            VAL_X_DIR,
            VAL_Y_DIR,
            xtransforms=val_x_transforms,
            ytransforms=val_y_transforms,
        )
        val_losses = train_model(model, train_loader, val_loader, device)
        print(val_losses)

    if args.sub_command == "test":
        model = load_model()
        test_loader = get_data_loader(
            TEST_X_DIR,
            TEST_Y_DIR,
            xtransforms=test_x_transforms,
            ytransforms=test_y_transforms,
        )

        test_model(model, test_loader, device)

    if args.sub_command == "predict_and_write":
        model = load_model()
        files = get_filepaths_matching_pattern(args.file_patterns, recursive=True)
        if not args.save_path.exists():
            args.save_path.mkdir()
        predict_and_write_to_disk(model, files, args.save_path, device, args.keep_file_structure)


if __name__ == "__main__":
    main()

# Sources:
# https://github.com/astra-toolbox/astra-toolbox/blob/3cdfbe01f4f0dc1eb2df5224f10dd53d7804a642/samples/python/s004_cpu_reconstruction.py


import argparse
import glob
from monai.transforms.utility.array import TorchVision
from skimage.exposure.exposure import rescale_intensity
import skimage.transform
import itk
import shutil
import numpy as np
from torch.serialization import save
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
# from dl_with_unet import *
from pcon_unet.train import *
from skimage import exposure

from monai.transforms import ScaleIntensity

# Any and all data goes here
DATA_DIR = Path('data')

# Original untouched data
RAW_DATA_DIR = DATA_DIR / 'raw_data'

# Data separated into train, val, and test
EXTRACT_DIR = DATA_DIR / 'extracted_data'

# Original data as given was separated into 3 directories
RAW_PROJ_DIR = RAW_DATA_DIR / 'stage2-10K_500_proj'
RAW_REC_DIR = RAW_DATA_DIR / 'stage2-10K_500_rec'
RAW_SOURCE_DIR = RAW_DATA_DIR / 'stage2-10K-500_source'

# Synthetic dataset for use before actual data comes in
SYNTHETIC_DATA_DIR = DATA_DIR / 'synthetic'
SYNTHETIC_SINO_DIR = SYNTHETIC_DATA_DIR / 'sinograms'
SYNTHETIC_INCOMPLETE_SINO_DIR = SYNTHETIC_DATA_DIR / 'incomplete_sinograms'
SYNTHETIC_MASK_DIR = SYNTHETIC_DATA_DIR / 'masks'

# Train directories
TRAIN_DATA_DIR = EXTRACT_DIR / 'train'
TRAIN_X_DIR = TRAIN_DATA_DIR / 'x'
TRAIN_Y_DIR = TRAIN_DATA_DIR / 'y'
TRAIN_MASK_DIR = TRAIN_DATA_DIR / 'z'

# Validation directories
VAL_DATA_DIR = EXTRACT_DIR / 'val'
VAL_X_DIR = VAL_DATA_DIR / 'x'
VAL_Y_DIR = VAL_DATA_DIR / 'y'
VAL_MASK_DIR = VAL_DATA_DIR / 'z'

# Test directories
TEST_DATA_DIR = EXTRACT_DIR / 'test'
TEST_X_DIR = TEST_DATA_DIR / 'x'
TEST_Y_DIR = TEST_DATA_DIR / 'y'
TEST_MASK_DIR = TEST_DATA_DIR / 'z'

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


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    """

    my_parser = argparse.ArgumentParser()
    sub_parsers = my_parser.add_subparsers(dest='sub_command')

    sub_parsers.add_parser('generate_synthetic_data')
    sub_parsers.add_parser('extract', help='Extract data')
    sub_parsers.add_parser('extract_synthetic_data')
    sub_parsers.add_parser('train', help='Train a fresh model')
    sub_parsers.add_parser('test', help='Test a trained model')

    sub_parser_predict_and_show = sub_parsers.add_parser('predict_and_show')
    # TODO: Do mutual exclusion properly here
    # group = sub_parser_predict_and_show.add_mutually_exclusive_group(required=True)
    sub_parser_predict_and_show.add_argument(
        '--show_iradon',
        action='store_true',
        help='Run each of the images through iradon',
    )
    sub_parser_predict_and_show.add_argument('--random_image', action='store_true')
    sub_parser_predict_and_show.add_argument('--image_path', action='store')
    sub_parser_predict_and_show.add_argument('--gt_path', action='store')

    return my_parser


def _extract_datalist(dl, xoutdir, youtdir, zoutdir):
    for d in dl:
        xfname = Path(d['x']).name
        yfname = Path(d['y']).name
        maskname = Path(d['z']).name

        x = itk.imread(d['x'], pixel_type=PixelType)
        y = itk.imread(d['y'], pixel_type=PixelType)
        mask = itk.imread(d['z'], pixel_type=PixelType)

        itk.imwrite(x, str(xoutdir / xfname))
        itk.imwrite(y, str(youtdir / yfname))
        itk.imwrite(mask, str(zoutdir / maskname))

def to_image(arr):
    return itk.image_from_array(arr)

# TODO: Make this modular on axis?
def remove_random_cols_of_image(sino: itk.Image, frac_to_remove=0.1):
    sino_arr = itk.array_from_image(sino)
    num_rows, num_cols = sino_arr.shape

    num_samples_to_remove = int(frac_to_remove * num_cols)
    cols_to_remove = np.random.choice(
        range(num_cols), size=num_samples_to_remove, replace=False
    )
    mask = np.ones((num_rows, num_cols))
    mask[:, cols_to_remove] = np.zeros((num_rows,1))
    sino_arr[:, cols_to_remove] = np.zeros((num_rows, 1))

    return to_image(sino_arr), to_image(mask)


def extract_data(
    sino_x_dir,
    sino_y_dir,
    sino_mask_dir,
    train_split=0.6,
    val_split=0.2,
    test_split=0.2,
):
    """
    Note that if the splits dont add up to 1, the ratio is taken
    """

    all_fnames = [fp.name for fp in sino_x_dir.glob('*.*') if fp.is_file()]

    datalist = [
        {
            'x': str(sino_x_dir / fn),
            'y': str(sino_y_dir / fn),
            'z': str(sino_mask_dir / fn)
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

    _extract_datalist(train_part, TRAIN_X_DIR, TRAIN_Y_DIR, TRAIN_MASK_DIR)
    _extract_datalist(val_part, VAL_X_DIR, VAL_Y_DIR, VAL_MASK_DIR)
    _extract_datalist(test_part, TEST_X_DIR, TEST_Y_DIR, TEST_MASK_DIR)


def forward_project_radon(im: itk.Image, theta=None):
    im_size = im.GetLargestPossibleRegion().GetSize()
    if len(im_size) == 3:
        if im_size[2] != 1:
            raise ValueError(
                'im argument must either be a 2d image or a 3d image with length 1 in final dimension'
            )

        im = util.extract_slice(im, 0, PixelType=PixelType)
        im_size = im.GetLargestPossibleRegion().GetSize()

    ncol, nrow = im_size
    if theta is None:
        theta = np.linspace(0, 180, max(ncol, nrow))

    sino_arr = skimage.transform.radon(im, theta=theta, circle=False)
    return itk.image_from_array(sino_arr)


def back_project_iradon(im, theta=None):
    """
    Back projects using iradon. Note that the reconstructed image does not have
    meaningful spacing, direction, etc. and should be used for displaying ONLY.
    """

    if isinstance(im, itk.Image):
        im = itk.array_from_image(im)

    im = exposure.rescale_intensity(im)

    rec_im_arr = skimage.transform.iradon(im, theta, circle=False)
    return itk.image_from_array(rec_im_arr)


def generate_synthetic_data(
    source_dir=RAW_SOURCE_DIR,
    complete_sino_dir=SYNTHETIC_SINO_DIR,
    incomplete_sino_dir=SYNTHETIC_INCOMPLETE_SINO_DIR,
    mask_dir=SYNTHETIC_MASK_DIR
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

    if isinstance(mask_dir, str):
        mask_dir = Path(mask_dir)

    for p in [complete_sino_dir, incomplete_sino_dir, mask_dir]:
        if p.exists():
            shutil.rmtree(str(p))
            p.mkdir(parents=True)

    paths = sorted(glob.glob(str(source_dir / '*.dcm')))
    for i, path in enumerate(paths):
        if i % 20 == 0:
            print('On image {} of {}'.format(i, len(paths)))

        source_im = itk.imread(path, pixel_type=PixelType)
        complete_sino = forward_project_radon(source_im)
        incomplete_sino, mask = remove_random_cols_of_image(complete_sino)
        # Now write to disk
        stem = Path(path).stem

        itk.imwrite(complete_sino, str(complete_sino_dir / (stem + '.mha')))
        itk.imwrite(incomplete_sino, str(incomplete_sino_dir / (stem + '.mha')))
        itk.imwrite(mask, str(mask_dir / (stem + '.mha')))


def get_data_loader(
    xdir, ydir, zdir, xtransforms=None, ytransforms=None, ztransforms=None, batch_size=BATCH_SIZE
):
    xs = sorted(glob.glob(str(xdir / '*')))
    ys = sorted(glob.glob(str(ydir / '*')))
    zs = sorted(glob.glob(str(zdir / '*')))

    # Should have the same filenames in both
    assert [Path(p).name for p in xs] == [Path(p).name for p in ys] == [Path(p).name for p in zs]

    # We abuse the monai 'label' argument here to append the masks alongside
    # the gt and source images
    dataset = ArrayDataset(xs, xtransforms, ys, ytransforms, zs, ztransforms)
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


def my_normalize(arr):
    arr -= np.min(arr)
    return arr / np.max(arr)


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
    titles = ['Input Sinogram', 'GT Sinogram', 'Predicted Sinogram']
    grid_shape = (1, 3)

    metric_value_sinos = eval_metric(torch.Tensor(sino_pred).unsqueeze(0), torch.Tensor(sino_gt).unsqueeze(0)).item()
    print('Metric prediction and ground truth sinos:', metric_value_sinos)
    if back_projector is not None:
        rec_input = back_projector(sino_input)
        images_to_display.append(rec_input)
        titles.append('Input Reconstructed')

        rec_gt = back_projector(sino_gt)
        images_to_display.append(rec_gt)
        titles.append('GT Reconstructed')

        rec_pred = back_projector(sino_pred)
        images_to_display.append(rec_pred)
        titles.append('Prediction Reconstructed')

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
    # display_image_truth_and_prediction(
    #     image, gt, prediction, eval_metric, back_projector
    # )
    save_image(prediction, 'tst1.jpg')


def get_random_image_and_gt(image_dir, gt_dir):
    if isinstance(image_dir, str):
        image_dir = Path(image_dir)

    if isinstance(gt_dir, str):
        gt_dir = Path(gt_dir)

    fpaths = list(image_dir.glob('*.*'))
    random_im_path = np.random.choice(fpaths, size=1)[0]
    random_im_path = Path(random_im_path)
    image_fname = random_im_path.name

    # Assuming image and gt fnames are the same, just in different dirs
    gt_path = str(gt_dir / image_fname)
    return itk.imread(str(random_im_path)), itk.imread(gt_path)


def main():
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    if args.sub_command == 'generate_synthetic_data':
        generate_synthetic_data()

    # elif args.sub_command == 'extract':
    #     extract_data()

    elif args.sub_command == 'extract_synthetic_data':
        extract_data(SYNTHETIC_INCOMPLETE_SINO_DIR, SYNTHETIC_SINO_DIR, SYNTHETIC_MASK_DIR)

    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache()

    if args.sub_command == 'predict_and_show':
        model = load_model()
        image, gt = None, None
        back_projector = None
        if args.random_image:
            image, gt = get_random_image_and_gt(TRAIN_X_DIR, TRAIN_Y_DIR)

        elif args.image_path and args.gt_path:
            image = itk.imread(args.image_path)
            gt = itk.imread(args.gt_path)

        else:
            raise RuntimeError(
                'Please specify either --random_image or --image_path and --gt_path'
            )

        if args.show_iradon:
            back_projector = back_project_iradon
        val_loader = get_data_loader(
            VAL_X_DIR,
            VAL_Y_DIR,
            xtransforms=val_x_transforms,
            ytransforms=val_y_transforms,
        )
        predict_image_and_show(model, val_loader, device, gt, back_projector=back_projector)

    if args.sub_command == 'train':
        model = get_model()
        train_loader = get_data_loader(
            TRAIN_X_DIR,
            TRAIN_Y_DIR,
            TRAIN_MASK_DIR,
            xtransforms=train_x_transforms,
            ytransforms=train_y_transforms,
            ztransforms=train_mask_transforms
        )
        val_loader = get_data_loader(
            VAL_X_DIR,
            VAL_Y_DIR,
            VAL_MASK_DIR,
            xtransforms=val_x_transforms,
            ytransforms=val_y_transforms,
            ztransforms=val_mask_tranforms
        )
        val_losses = train_model(model, train_loader, val_loader, device)
        print(val_losses)

    if args.sub_command == 'test':
        model = load_model()
        test_loader = get_data_loader(
            TEST_X_DIR,
            TEST_Y_DIR,
            xtransforms=test_x_transforms,
            ytransforms=test_y_transforms,
        )

        test_model(model, test_loader, device)


if __name__ == '__main__':
    main()

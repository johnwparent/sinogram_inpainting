# Sources:
# https://github.com/astra-toolbox/astra-toolbox/blob/3cdfbe01f4f0dc1eb2df5224f10dd53d7804a642/samples/python/s004_cpu_reconstruction.py


import argparse
import glob
import skimage.transform
import itk
import shutil
import numpy as np
import util

import monai
import torch
import itk
# import astra
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import shutil


from ignite.engine import Events
from monai.data import partition_dataset
from pathlib import Path
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import ValidationHandler, CheckpointSaver, MeanAbsoluteError
from torch import optim, nn
from monai.data import ArrayDataset, DataLoader
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    EnsureType,
    Resize,
    LoadImage,
    RandFlip,
    RandZoom,
    ScaleIntensity,
    SqueezeDim,
    Transform,
    KeepLargestConnectedComponent,
    SqueezeDim,
)


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
# Some other constants

# Default number of epochs
NUM_EPOCHS = 1

BATCH_SIZE = 16

BEST_MODEL_NAME = "best_model.pt"
MODEL_SAVE_DIR = Path("./models")

# ITK stuff
PixelType = itk.F
InputDataDim = 3
ExtractedDataDim = 2

RawDataImageType = itk.Image[PixelType, InputDataDim]
ImageType = itk.Image[PixelType, ExtractedDataDim]

NETWORK_INPUT_SIZE = (256, 256)


# Define our transforms
train_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        SqueezeDim(2),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # # RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="nearest"), # TODO: Make sure we dont center zoom at cut out the top. Uncomment
        # ScaleIntensity(),
        # RandFlip(prob=0.5, spatial_axis=1),
    ]
)
train_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        SqueezeDim(2),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE)
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # # RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="nearest"), # TODO: Make sure we dont center zoom at cut out the top. Uncomment
        # RandFlip(prob=0.5, spatial_axis=1),
    ]
)
val_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        SqueezeDim(2),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE)
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # ScaleIntensity(),
        # EnsureType(),
    ]
)

val_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        SqueezeDim(2),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE)
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # EnsureType(),
        # AsDiscrete(to_onehot=True, n_classes=3),
    ]
)


def _construct_parser():
    """
    Constructs the ArgumentParser object with the appropriate options

    """

    my_parser = argparse.ArgumentParser()
    sub_parsers = my_parser.add_subparsers(dest="sub_command")

    sub_parsers.add_parser("generate_synthetic_data")
    sub_parsers.add_parser("extract", help="Extract data")
    sub_parsers.add_parser("extract_synthetic_data")
    sub_parsers.add_parser("train", help="Train a fresh model")


    return my_parser


def _extract_datalist(dl, xoutdir, youtdir):
    for d in dl:
        xfname = Path(d["x"]).name
        yfname = Path(d["y"]).name

        x = itk.imread(d["x"], pixel_type=PixelType)
        y = itk.imread(d["y"], pixel_type=PixelType)

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


# TODO: Weird that we have the "classes" keyword below.
# This is only meant to be a proof of concept.
def get_model():
    return smp.Unet(
        encoder_weights="imagenet", in_channels=1, classes=1, activation=None
    )


def get_data_loader(
    xdir, ydir, xtransforms=None, ytransforms=None, batch_size=BATCH_SIZE
):
    xs = sorted(glob.glob(str(xdir / "*")))
    ys = sorted(glob.glob(str(ydir / "*")))

    # Should have the same filenames in both
    assert xs == ys

    dataset = ArrayDataset(xs, xtransforms, ys, ytransforms)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=torch.cuda.is_available()
    )


def _display_image_truth_and_prediction(image=None, truth=None, prediction=None):
    num_ims_to_display = 3 - ((image is None) + (truth is None) + (prediction is None))
    curr_image_num = 1

    fig = plt.figure(figsize=(10, 10))

    if image is not None:
        # Display the image
        sp1 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp1.set_title("Image")
        plt.imshow(image)

        curr_image_num += 1

    if truth is not None:
        # Display the truth
        sp2 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp2.set_title("Ground Truth")
        plt.imshow(truth, cmap="gray", vmin=0, vmax=2)

        curr_image_num += 1

    if prediction is not None:
        # Display the corresponding mask patch
        sp3 = fig.add_subplot(1, num_ims_to_display, curr_image_num)
        sp3.set_title("Prediction")
        plt.imshow(prediction, cmap="gray", vmin=0, vmax=2)

        curr_image_num += 1

    plt.show()


# def predict_image_and_show(
#     model,
#     image_full_path,
#     device,
#     gt_full_path=None,
#     eval_metric=monai.metrics.MSEMetric(),
# ):

#     image = itk.imread(image_full_path)
#     gt = None if gt_full_path is None else itk.imread(gt_full_path)

#     # TODO: three return values? Get rid of last one? Doing lots of converting
#     prediction = model(some_transform(image))

#     # Unsqueeze for batch. Only need to do this for eval_metric
#     # TODO: transformations for prediction
#     predicted_mask_eval = output_to_3d_mask_transformsed(raw_output).unsqueeze(0)
#     if gt_full_path is not None:
#         mask_eval = eval_mask_transforms(gt_full_path).unsqueeze(0)
#         # TODO: move this to label
#         print("Dice scores:", eval_metric(predicted_mask_eval, mask_eval))
#     print("predicted_onsd:", predicted_onsd)

#     # Display
#     _display_image_truth_and_prediction(image, mask, predicted_mask)



def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=NUM_EPOCHS,
    loss_function=nn.MSELoss(),
):

    model.to(device)
    # TODO: Keep track of the cumulative loss.
    optimizer = optim.Adam(model.parameters())
    metric_values = []
    iter_losses = []
    batch_sizes = []
    epoch_loss_values = []

    steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
    if len(train_loader.dataset) % train_loader.batch_size != 0:
        steps_per_epoch += 1

    def trans_batch_val(x):
        preds = []
        labels = []
        for d in x:
            preds.append((d["pred"]))
            labels.append(d["label"])
        return preds, labels

    def prepare_batch_train(batchdata, device, non_blocking):
        imgs, masks = batchdata
        return imgs.to(device), masks.to(device)

    def prepare_batch_val(batchdata, device, non_blocking):
        imgs, masks = batchdata
        return imgs.to(device), masks.to(device)

    key_val_metric = {"MSE": MeanAbsoluteError(output_transform=trans_batch_val)}
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_saver = CheckpointSaver(
        save_dir=MODEL_SAVE_DIR,
        save_dict={"model": model},
        save_key_metric=True,
        key_metric_filename=BEST_MODEL_NAME,
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        key_val_metric=key_val_metric,
        prepare_batch=prepare_batch_val,
        val_handlers=[checkpoint_saver],
    )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=num_epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        train_handlers=[ValidationHandler(1, evaluator)],
        prepare_batch=prepare_batch_train,
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def _end_iter(engine):
        loss = np.average([o["loss"] for o in engine.state.output])
        iter_losses.append(loss)
        epoch = engine.state.epoch
        epoch_len = engine.state.max_epochs
        step = (engine.state.iteration % steps_per_epoch) + 1
        batch_len = len(engine.state.batch[0])
        batch_sizes.append(batch_len)

        if step % 5 == 0:
            print(
                f"epoch {epoch}/{epoch_len}, step {step}/{steps_per_epoch}, training_loss = {loss:.4f}"
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        overall_average_loss = np.average(iter_losses, weights=batch_sizes)
        epoch_loss_values.append(overall_average_loss)

        # clear the contents of iter_losses and batch_sizes for the next epoch
        iter_losses.clear()
        batch_sizes.clear()

        # fetch and report the validation metrics
        mse = evaluator.state.metrics["MSE"]
        metric_values.append(mse)
        print(f"evaluation for epoch {engine.state.epoch},  MSE = {mse:.4f}")

    trainer.run()

    return metric_values


def main():
    my_parser = _construct_parser()
    args = my_parser.parse_args()

    if args.sub_command == "generate_synthetic_data":
        generate_synthetic_data()

    # elif args.sub_command == "extract":
    #     extract_data()

    elif args.sub_command == "extract_synthetic_data":
        extract_data(
            SYNTHETIC_INCOMPLETE_SINO_DIR, SYNTHETIC_SINO_DIR
        )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    if args.sub_command == "train":
        model = get_model()
        train_loader = get_data_loader(
            TRAIN_X_DIR,
            TRAIN_X_DIR,
            xtransforms=train_x_transforms,
            ytransforms=train_y_transforms,
        )
        val_loader = get_data_loader(
            TRAIN_X_DIR,
            TRAIN_X_DIR,
            xtransforms=val_x_transforms,
            ytransforms=val_y_transforms,
        )
        val_losses = train_model(model, train_loader, val_loader, device)
        print(val_losses)


if __name__ == "__main__":
    main()

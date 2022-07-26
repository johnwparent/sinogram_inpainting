import glob
import itk
import torch

import numpy as np

from pathlib import Path
from itk import RTK as rtk

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

itk_image_type = rtk.Image[itk.F, 1]

def define_geometry():
    # Defines the RTK geometry object
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    numberOfProjections = 195
    firstAngle = 0.
    angularArc = 360.
    sid = 470
    sdd = 690
    for x in range(numberOfProjections):
        angle = firstAngle + x * angularArc / numberOfProjections
        geometry.AddProjection(sid, sdd, angle)


def _data_loader(xdir, ydir, batch_size):

    train_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        ScaleIntensity(),
        RecComposeTransform(),
    ]
    )

    train_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        ScaleIntensity(),
        RecGTComposeTransform(),
    ]
    )
    xs = sorted(glob.glob(str(xdir / "*")))
    ys = sorted(glob.glob(str(ydir / "*")))

    # Should have the same filenames in both
    assert [Path(p).name for p in xs] == [Path(p).name for p in ys]

    dataset = ArrayDataset(xs, train_x_transforms, ys, train_y_transforms)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=torch.cuda.is_available()
    )


def get_train_data_loader(batch_size):
    return _data_loader(TRAIN_X_DIR,
            TRAIN_Y_DIR, batch_size)

def get_val_data_loader(batch_size):
    return _data_loader(VAL_X_DIR, VAL_Y_DIR, batch_size)

def back_projection_filter(image):
    proj = itk.GetImageViewFromArray(image)
    vol = rtk.ConstantImageSource[itk_image_type].New()
    vol.SetConstant(0.)
    bp = rtk.BackProjectionImageFilter[itk_image_type, itk_image_type].New()
    bp.SetGeometry(define_geometry())
    bp.SetInput(vol.GetOutput())
    bp.SetInput(1, proj.GetOutput())
    bp.Update()
    return itk.GetArrayFromImage(bp.GetOutput())


def fdk_back_projection_filter(image):
    proj = itk.GetImageViewFromArray(image)
    vol = rtk.ConstantImageSource[itk_image_type].New()
    vol.SetConstant(0.)
    bp = rtk.FDKBackProjectionImageFilter[itk_image_type, itk_image_type].New()
    bp.SetGeometry(define_geometry())
    bp.SetInput(vol.GetOutput())
    bp.SetInput(1, proj.GetOutput())
    bp.Update()
    return itk.GetArrayFromImage(bp.GetOutput())


def filtered_back_projection_filter(image):
    proj = itk.GetImageViewFromArray(image)
    vol = rtk.ConstantImageSource[itk_image_type].New()
    vol.SetConstant(0.)
    bp = rtk.BackProjectionImageFilter[itk_image_type, itk_image_type].New()
    bp.SetGeometry(define_geometry())
    bp.SetInput(vol.GetOutput())
    bp.SetInput(1, proj.GetOutput())
    bp.GetRampFilter().SetTruncationCorrection(0.0)
    bp.SetHannCutFrequency(0.0)
    bp.Update()
    return itk.GetArrayFromImage(bp.GetOutput())


class RecComposeTransform(Transform):
    """
    Transform that consumes single channel projections image and
    composes into multichannel volume image array
    """
    def __call__(self, data):
        tbp = back_projection_filter(data)
        fdk_bp = fdk_back_projection_filter(data)
        fbp = filtered_back_projection_filter(data)
        return np.concatenate((tbp, fdk_bp, fbp), axis=2)


class RecGTComposeTransform(Transform):
    def __call__(self, data):
        return fdk_back_projection_filter(data)

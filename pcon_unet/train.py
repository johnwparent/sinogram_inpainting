import torch
import itk
import numpy as np
import os
import sys
import segmentation_models_pytorch as smp
from tqdm import tqdm

from . import model, util, loss

from pathlib import Path
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
from ignite.engine import Events
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    ValidationHandler,
    CheckpointSaver,
    MeanAbsoluteError,
    MeanSquaredError,
)
from torch import optim, nn
from torchvision import models
from torchvision.utils import make_grid
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..')))
from partialconv.models.loss import VGG16PartialLoss

__imp = 0
if not __imp:
    vgg16 = models.vgg16(pretrained=True)
    torch.save(vgg16.state_dict(), "vgg.pth")
    __imp = 1


NETWORK_INPUT_SIZE = (256, 256)
NUM_EPOCHS = 50

# This is part of the API, main.py relies on a default number of epochs
BATCH_SIZE = 16

BEST_MODEL_NAME = "best_model.pt"
MODEL_SAVE_DIR = Path("./models")
BEST_MODEL_PATH = MODEL_SAVE_DIR / BEST_MODEL_NAME


# All of the transforms are part of the API, except itk_image_to_model_input.
# Need to know which transforms to apply when training, testing, etc.
train_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # # RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="nearest"), # TODO: Make sure we dont center zoom at cut out the top. Uncomment
        ScaleIntensity(),
        RandFlip(prob=0.5, spatial_axis=1),
    ]
)
train_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # # RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.3, mode="nearest"), # TODO: Make sure we dont center zoom at cut out the top. Uncomment
        ScaleIntensity(),
        RandFlip(prob=0.5, spatial_axis=1),
    ]
)
val_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        ScaleIntensity(),
        # EnsureType(),
    ]
)

val_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        ScaleIntensity(),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # EnsureType(),
        # AsDiscrete(to_onehot=True, n_classes=3),
    ]
)

test_x_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        ScaleIntensity(),
        # EnsureType(),
    ]
)

test_y_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureType(),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE),
        ScaleIntensity(),
        # Resize(NETWORK_INPUT_SHAPE, mode="nearest"),
        # EnsureType(),
        # AsDiscrete(to_onehot=True, n_classes=3),
    ]
)



itk_image_to_model_input = Compose(
    [
        EnsureType(data_type="tensor"),
        AddChannel(),
        Resize(NETWORK_INPUT_SIZE, mode="nearest"),
        ScaleIntensity(),
        AddChannel(),
        EnsureType(),
    ]
)

def evaluate(model, dataset, device, filename):
    image, gt = next(iter(dataset))
    mask = gt - image
    with torch.no_grad():
        output, _ = model(image.to(device), mask.to(device))
    output = output.to(torch.device('cpu'))
    output_comp = mask * image + (1 - mask) * output
    save_image(output, filename)
# All of the below functions are part of the api, except ones that start with
# an underscore


def load_model(path=BEST_MODEL_PATH):
    '''
    Get a model trained on our data from disk
    '''
    model = get_model()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    load_ckpt(path, [('model', model)], [('optimizer', optimizer)])
    return model


def load_ckpt(ckpt_name, models, optimizers=None):
    ckpt_dict = torch.load(ckpt_name)
    for prefix, model in models:
        assert isinstance(model, nn.Module)
        model.load_state_dict(ckpt_dict[prefix], strict=False)
    if optimizers is not None:
        for prefix, optimizer in optimizers:
            optimizer.load_state_dict(ckpt_dict[prefix])
    return ckpt_dict['n_iter']


# TODO: Weird that we have the "classes" keyword below.
# This is only meant to be a proof of concept.
def get_model():
    '''
    Get the untrained model
    '''
    return model.UNetPconv()

def get_state_dict_on_cpu(obj):
    cpu_device = torch.device('cpu')
    state_dict = obj.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(cpu_device)
    return state_dict

def save_ckpt(ckpt_name, models, optimizers, n_iter):
    ckpt_dict = {'n_iter': n_iter}
    for prefix, model in models:
        ckpt_dict[prefix] = get_state_dict_on_cpu(model)

    for prefix, optimizer in optimizers:
        ckpt_dict[prefix] = optimizer.state_dict()
    torch.save(ckpt_dict, ckpt_name)

def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor([0.229, 0.224, 0.225]) + torch.Tensor([0.485, 0.456, 0.406])
    x = x.transpose(1, 3)
    return x

def run_inference(model, input_image: itk.Image, device):
    '''
    Accepts a model, itk image, and device. Runs the inference and returns
    the result as a 2d itk image
    '''

    model.eval()
    model.to(device)

    return evaluate(model, input_image, device, "test.jpg")

def _transform_batch(x):
    preds = []
    labels = []
    for d in x:
        preds.append((d["pred"]))
        labels.append(d["label"])
    return preds, labels

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=NUM_EPOCHS,
    loss_function=VGG16PartialLoss(vgg_path="vgg.pth"),
):
    '''
    Should train model and save best model to a known path.
    Returns the validation losses
    '''
    loss_function = nn.L1Loss(size_average=True)
    model =  smp.Unet(
        encoder_weights="imagenet", in_channels=1, classes=1, activation=None
    )
    train_gen = iter(train_loader)
    model.to(device)
    model.train()
    # TODO: Keep track of the cumulative loss.
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    loss = 0.0
    for i in tqdm(range(num_epochs)):
        try:
            image, gt = [x.to(device) for x in next(train_gen)]
        except StopIteration:
            train_gen = iter(train_loader)
            image, gt = [x.to(device) for x in next(train_gen)]
        mask = gt - image
        output, _ = model(image, mask)
    #     loss_dict = loss_function(image, mask, output, gt)
    #     LAMBDA_DICT = {
    # 'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
    #     loss = 0.0
    #     for key, coef in LAMBDA_DICT.items():
    #         value = coef * loss_dict[key]
    #         loss += value
        loss = loss_function(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0 or (i + 1) == num_epochs:
            try:
                save_ckpt('{}/ckpt/{}.pt'.format(MODEL_SAVE_DIR, i + 1),
                        [('model', model)], [('optimizer', optimizer)], i + 1)
            except:
                save_ckpt('{}.pt'.format(i + 1),
                        [('model', model)], [('optimizer', optimizer)], i + 1)

        if (i + 1) % 10 == 0:
            try:
                print("Loss: {}: {}".format(loss.item(), input.size(0)))
                model.eval()
                evaluate(model, val_loader, device,
                        'images/test_{:d}.jpg'.format(i + 1))
            except:
                pass


def test_model(model, test_loader, device):
    '''
    Tests the model on the data in test_loader and prints the MAE,
    or some other testing metric.
    '''
    model.to(device)
    def prepare_batch_test(batchdata, device, non_blocking):
        imgs, labels = batchdata
        return imgs.to(device), labels.to(device)

    key_val_metric = {"MAE": MeanAbsoluteError(output_transform=_transform_batch)}

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=model,
        key_val_metric=key_val_metric,
        prepare_batch=prepare_batch_test,
    )

    evaluator.run()

    print("Testing:", evaluator.state.metrics["MAE"])

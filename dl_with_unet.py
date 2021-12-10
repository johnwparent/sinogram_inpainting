import segmentation_models_pytorch as smp
import torch
import itk
import util
import numpy as np

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

NETWORK_INPUT_SIZE = (256, 256)
NUM_EPOCHS = 200

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



# All of the below functions are part of the api, except ones that start with
# an underscore


def load_model(path=BEST_MODEL_PATH):
    '''
    Get a model trained on our data from disk
    '''
    model = get_model()
    model.load_state_dict(torch.load(str(path)))
    return model


# TODO: Weird that we have the "classes" keyword below.
# This is only meant to be a proof of concept.
def get_model():
    '''
    Get the untrained model
    '''
    return smp.Unet(
        encoder_weights="imagenet", in_channels=1, classes=1, activation=None
    )


def run_inference(model, input_image: itk.Image, device):
    '''
    Accepts a model, itk image, and device. Runs the inference and returns
    the result as a 2d itk image
    '''

    model.eval()
    model.to(device)

    # Generate predictions
    with torch.no_grad():
        prediction = (
            model(itk_image_to_model_input(input_image).to(device)).cpu().squeeze(0)
        )

    w, h = itk.size(input_image)
    transform = Compose(
        [
            AddChannel(),
            Resize((h, w), mode="nearest"),
        ]
    )

    prediction = transform(prediction.squeeze(0)).squeeze(0)
    prediction_image = util.image_from_array(prediction, reference_image=input_image)

    return prediction_image

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
    loss_function=nn.L1Loss(),
):
    '''
    Should train model and save best model to a known path.
    Returns the validation losses
    '''
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


    def prepare_batch(batchdata, device, non_blocking):
        imgs, masks = batchdata
        return imgs.to(device), masks.to(device)

    key_val_metric = {"MAE": MeanAbsoluteError(output_transform=_transform_batch)}
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_saver = CheckpointSaver(
        save_dir=MODEL_SAVE_DIR,
        save_dict={"model": model},
        save_key_metric=True,
        key_metric_negative_sign=True,
        key_metric_filename=BEST_MODEL_NAME,
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        key_val_metric=key_val_metric,
        prepare_batch=prepare_batch,
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
        prepare_batch=prepare_batch,
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
        mse = evaluator.state.metrics["MAE"]
        metric_values.append(mse)
        print(f"evaluation for epoch {engine.state.epoch},  MAE = {mse:.4f}")

    trainer.run()

    return metric_values


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

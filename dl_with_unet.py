import segmentation_models_pytorch as smp
import torch
import itk
import util
import numpy as np
import os

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
NUM_EPOCHS = 10

# This is part of the API, main.py relies on a default number of epochs
BATCH_SIZE = 8

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

    return prediction

def _transform_batch(x):
    preds = []
    labels = []
    for d in x:
        preds.append((d["pred"]))
        labels.append(d["label"])
    return preds, labels

ORIGINAL_IMAGE_SHAPE = (195, 1848)

# TODO: Make more idiomatic
def get_hardcoded_mask(original_image_shape=ORIGINAL_IMAGE_SHAPE, reshape=True):
    # Note, this is hardcoded based on where the images are zero in
    # the data we've received
    # Columns from 597-636 and 1213-1252 get set to 1
    cg1 = slice(597, 636)
    cg2 = slice(1213, 1252)

    # The following rows also need to be set to 1
    rg1 = slice(1, 10)
    rg2 = slice(66, 75)
    rg3 = slice(121, 130)
    rg4 = slice(131, 140)
    rg5 = slice(186, 195)

    ret = np.zeros(original_image_shape)
    # Setting the necessary rows to 1
    ret[..., cg1] = 1
    ret[..., cg2] = 1

    # Setting the necessary columns to 1
    ret[..., rg1, :] = 1
    ret[..., rg2, :] = 1
    ret[..., rg3, :] = 1
    ret[..., rg4, :] = 1
    ret[..., rg5, :] = 1

    if reshape:
        t = Compose(
            [
                AddChannel(),
                Resize(NETWORK_INPUT_SIZE),
                EnsureType(),
            ]
        )
    else:
        t = Compose(
            [
                AddChannel(),
                EnsureType()
            ]
        )
    return t(ret)


def app_mask(mask, *args):
    """
    Apply derived mask to provided args
    """
    if not args:
        raise RuntimeError("Incorrect Usgage, requires tensors to mask")

    return (x * mask for x in args) if len(args) > 1 else args[0] * mask


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
    model.train()
    lowest = 1E50
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    # Re-use the same mask over and over.
    mask = get_hardcoded_mask()
    def create_mask(mask, sz):
        mask = np.repeat(mask[np.newaxis, :, :, :], sz, axis=0)
        mask = mask.to(device)
        return mask

    for epoch in range(num_epochs):
        model.train()
        for i, imset in enumerate(train_loader):
            im, gt = imset
            im, gt = im.to(device), gt.to(device)
            out = model(im)
            gt, out = app_mask(create_mask(mask, gt.size()[0]), gt, out)
            loss = loss_function(out, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"[training] trained epoch: {epoch}")
        model.eval()

        total_loss = 0
        for imset in val_loader:
            im, gt = imset
            im, gt = im.to(device), gt.to(device)
            out = model(im)
            gt, out = app_mask(create_mask(mask, gt.size()[0]), gt, out)
            tot_loss = loss_function(out, gt)
            total_loss += float(tot_loss.detach().cpu())
        print(f"[validation] Total Loss val: {total_loss}")

        if total_loss < lowest:
            print("[EPOCH {}] Lowest loss {} found".format(epoch, total_loss))
            torch.save(model.state_dict(),  os.path.join(os.getcwd(), 'models', 'ckpt', '{}.pt'.format(epoch)))
            lowest = total_loss
        else:
            print("[EPOCH {}] loss is {}".format(epoch, total_loss))


# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     device,
#     num_epochs=NUM_EPOCHS,
#     loss_function=nn.L1Loss(),
# ):
#     '''
#     Should train model and save best model to a known path.
#     Returns the validation losses
#     '''
#     model.to(device)
#     # TODO: Keep track of the cumulative loss.
#     optimizer = optim.Adam(model.parameters())
#     metric_values = []
#     iter_losses = []
#     batch_sizes = []
#     epoch_loss_values = []

#     steps_per_epoch = len(train_loader.dataset) // train_loader.batch_size
#     if len(train_loader.dataset) % train_loader.batch_size != 0:
#         steps_per_epoch += 1


#     def prepare_batch(batchdata, device, non_blocking):
#         imgs, masks = batchdata
#         return imgs.to(device), masks.to(device)

#     key_val_metric = {"MAE": MeanAbsoluteError(output_transform=_transform_batch)}
#     MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
#     checkpoint_saver = CheckpointSaver(
#         save_dir=MODEL_SAVE_DIR,
#         save_dict={"model": model},
#         save_key_metric=True,
#         key_metric_negative_sign=True,
#         key_metric_filename=BEST_MODEL_NAME,
#     )

#     evaluator = SupervisedEvaluator(
#         device=device,
#         val_data_loader=val_loader,
#         network=model,
#         key_val_metric=key_val_metric,
#         prepare_batch=prepare_batch,
#         val_handlers=[checkpoint_saver],
#     )

#     trainer = SupervisedTrainer(
#         device=device,
#         max_epochs=num_epochs,
#         train_data_loader=train_loader,
#         network=model,
#         optimizer=optimizer,
#         loss_function=loss_function,
#         train_handlers=[ValidationHandler(1, evaluator)],
#         prepare_batch=prepare_batch,
#     )

#     @trainer.on(Events.ITERATION_COMPLETED)
#     def _end_iter(engine):
#         loss = np.average([o["loss"] for o in engine.state.output])
#         iter_losses.append(loss)
#         epoch = engine.state.epoch
#         epoch_len = engine.state.max_epochs
#         step = (engine.state.iteration % steps_per_epoch) + 1
#         batch_len = len(engine.state.batch[0])
#         batch_sizes.append(batch_len)

#         if step % 5 == 0:
#             print(
#                 f"epoch {epoch}/{epoch_len}, step {step}/{steps_per_epoch}, training_loss = {loss:.4f}"
#             )

#     @trainer.on(Events.EPOCH_COMPLETED)
#     def run_validation(engine):
#         overall_average_loss = np.average(iter_losses, weights=batch_sizes)
#         epoch_loss_values.append(overall_average_loss)

#         # clear the contents of iter_losses and batch_sizes for the next epoch
#         iter_losses.clear()
#         batch_sizes.clear()

#         # fetch and report the validation metrics
#         mae = evaluator.state.metrics["MAE"]
#         metric_values.append(mae)
#         print(f"evaluation for epoch {engine.state.epoch},  MAE = {mae:.4f}")

#     trainer.run()

#     return metric_values


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

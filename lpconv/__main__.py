import os
import glob
import math
import sys
import numpy as np

from PIL import Image

from network.unet import gram_matrix, nvidia_unet

import keras
import tensorflow as tf

from keras.models import Sequential, Model

from typing import Tuple

testIs = None

data_size = 200


def upscale_to_nearest(val, mul):
    return mul * math.ceil(val / mul)

input_shape = (upscale_to_nearest(1848, 256),
               upscale_to_nearest(195, 256))

def load_data(dir, Is):
    ims = glob.glob(dir)[:data_size]
    for d in ims:
        i = np.array(Image.open(d))
        i = np.repeat(i[..., np.newaxis], 3, -1)
        Is.append(tf.image.resize(np.array(i), (input_shape)))

test_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/test/*/*.tiff')
test_Is = []
load_data(test_data_dir, test_Is)
test_Is = np.array(test_Is)

val_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/val/*/*.tiff')
val_Is = []
load_data(val_data_dir, val_Is)
val_Is = np.array(val_Is)

model = nvidia_unet(input_shape)
loss = []

# Load loss model
lossModel = keras.applications.vgg16.VGG16(weights = None, include_top=False, input_shape=(input_shape[0], input_shape[1], 3))


for layer in lossModel.layers:
    layer.trainable=False


selectedlayers = [3, 6, 10]

layers_loss_model = Model(lossModel.inputs, [lossModel.layers[idx].output for idx in selectedlayers] +
                                            [gram_matrix(lossModel.layers[idx].output) for idx in selectedlayers])

lossModel.layers[1].output

validation_perceptual_Is = layers_loss_model.predict(test_Is) + [test_Is]

loss_model_outputs = layers_loss_model(model.output)


full_model = Model(model.input, loss_model_outputs + model.outputs)
full_model.compile(keras.optimizers.Adam(), ['mean_absolute_error'] * 7, loss_weights=[.05, .05, .05, 120, 120, 120, 9])


def generate_mask(input_size: Tuple[int, int]):
    mask = np.full((input_size[0], input_size[1], 3), 1, np.float32)
    cg1 = slice(0,20)
    cg2 = slice(596, 636)
    cg3 = slice(1212, 1252)
    cg4 = slice(1829, 1848)

    rg1 = slice(0, 10)
    rg2 = slice(55, 75)
    rg3 = slice(120, 140)
    rg4 = slice(185, 195)

    mask[:, cg1, :] = 0
    mask[:, cg2, :] = 0
    mask[:, cg3, :] = 0
    mask[:, cg4, :] = 0
    mask[rg1, :, :] = 0
    mask[rg2, :, :] = 0
    mask[rg3, :, :] = 0
    mask[rg4, :, :] = 0
    return np.array([mask] * data_size)

Is_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/train/*/*.tiff')
import sys
import gc
def trainer():
    gc.collect()
    mask = generate_mask(input_shape)

    Is = []
    load_data(Is_data_dir, Is)
    Is = np.array(Is)

    sample = Is
    try:
        perceptual_Is = layers_loss_model.predict(sample) + [sample]

        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   )
                   )

    finally:
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        gc.collect()

    sample = Is
    try:

        perceptual_Is = layers_loss_model.predict(sample) + [sample]
        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   validation_data=([test_Is, mask[:data_size]],
                                                     validation_perceptual_Is))
                   )

    finally:
        del perceptual_Is
        gc.collect()
        model.save("best_inpainter")

import gc
perceptual_Is = []
for _ in range(30):
    trainer()
model.save("best_inpainter")


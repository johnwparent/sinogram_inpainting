import os
import sys
import numpy as np

from network import unet, gram_matrix

import keras

from keras.models import Sequential, Model

from typing import Tuple

testIs = None

data_size = 2000


model = unet.nvidia_unet(depth=1)
loss = []

lossModel = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(256, 256, 1))

for layer in lossModel.layers:
    layer.trainable=False

selectedlayers = [3, 6, 10]

layers_loss_model = Model(lossModel.inputs, [lossModel.layers[idx].output for idx in selectedlayers] +
                                            [gram_matrix(lossModel.layers[idx].output) for idx in selectedlayers])

lossModel.layers[1].output

#validation_perceptual_Is = layers_loss_model.predict(test_Is * 256 - 127) + [test_Is]
validation_perceptual_Is = layers_loss_model.predict(test_Is) + [test_Is]

#loss_model_outputs = Lambda(lambda x: x * 256 - 127)(model.output)
loss_model_outputs = layers_loss_model(model.output)

full_model = Model(model.input, loss_model_outputs + model.outputs)

full_model.compile(keras.optimizers.Adam(), ['mean_absolute_error'] * 7, loss_weights=[.05, .05, .05, 120, 120, 120, 9])

def generate_mask(input_size: Tuple[int, int]):
    mask = np.full((input_size[0], input_size[1], 1), 255, np.float32)
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
    mask/=255
    return np.array([mask]* data_size)

import sys
import gc
def trainer():
    gc.collect()
    mask = generate_mask()


    sample = Is
    print("init")
    try:
        perceptual_Is = layers_loss_model.predict(sample) + [sample]

        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   )
                   )

    finally:
        print("del_start")
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        gc.collect()
        print("del end")


    sample = Is
    print("init")
    try:

        perceptual_Is = layers_loss_model.predict(sample) + [sample]

        loss.append(full_model.fit([sample, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   validation_data=([test_Is, mask[:(data_size/2)]],
                                                     validation_perceptual_Is))
                   )

    finally:
        print("del_start")
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        gc.collect()
        print("del end")
        model.save("best_inpainter")

import gc
perceptual_Is = []
for _ in range(30):
    trainer()
model.save("best_inpainter")


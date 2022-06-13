import os
import gc
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


data_slice = int(sys.argv[2])


def upscale_to_nearest(val, mul):
    return mul * math.ceil(val / mul)


input_shape = (upscale_to_nearest(195, 256),
               upscale_to_nearest(1848, 256)
            )


test_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/test/*/*.tiff')


class ProgressiveLoader(object):
    def __init__(self, data_root, slice=10):
        self.im_root = data_root
        self.set_slice = slice
        self.data_set = glob.glob(self.im_root)
        self.current_pos = 0

    def load_next_set(self):
        ims = []
        old_pos = self.current_pos
        self.current_pos += self.set_slice
        for d in self.data_set[old_pos:self.current_pos]:
            i = np.array(Image.open(d))
            i = np.repeat(i[..., np.newaxis], 3, -1)
            ims.append(tf.image.pad_to_bounding_box(np.array(i), 0, 0, input_shape[0], input_shape[1]))
        return np.array(ims)


test_Is = ProgressiveLoader(test_data_dir, slice=data_slice)
test_Is_ims = test_Is.load_next_set()

val_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/val/*/*.tiff')
# val_Is = []
# load_data(val_data_dir, val_Is)
# val_Is = np.array(val_Is)

model = nvidia_unet(input_shape)
loss = []

# Load loss model
lossModel = keras.applications.vgg16.VGG16(include_top=False, input_shape=(input_shape[0], input_shape[1], 3))

for layer in lossModel.layers:
    layer.trainable=False

selectedlayers = [3, 6, 10]
layers_loss_model = Model(lossModel.inputs, [lossModel.layers[idx].output for idx in selectedlayers] +
                                            [gram_matrix(lossModel.layers[idx].output) for idx in selectedlayers])
lossModel.layers[1].output

validation_perceptual_Is = layers_loss_model.predict(test_Is_ims) + [test_Is_ims]
loss_model_outputs = layers_loss_model(model.output)

# cleanup what we can
del test_Is_ims
gc.collect()


full_model = Model(model.input, loss_model_outputs + model.outputs)
full_model.compile(keras.optimizers.Adam(), ['mean_absolute_error'] * 7, loss_weights=[.0, .00, .00, 0, 0, 0, 1])


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
    return np.array([mask] * data_slice)

Is_data_dir = os.path.join(os.path.dirname(__file__), '../data/extracted_data/train/*/*.tiff')
Is_set = ProgressiveLoader(Is_data_dir, slice=data_slice)

def trainer():
    gc.collect()
    mask = generate_mask(input_shape)
    Is = Is_set.load_next_set()
    sample1 = Is
    sample2 = Is
    del Is
    gc.collect()
    try:
        perceptual_Is = layers_loss_model.predict(sample1) + [sample1]

        loss.append(full_model.fit([sample1, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   )
                   )
    finally:
        print(sys.getrefcount(perceptual_Is[0]))
        del perceptual_Is
        del sample1
        gc.collect()
    try:
        perceptual_Is = layers_loss_model.predict(sample2) + [sample2]
        loss.append(full_model.fit([sample2, mask], perceptual_Is, epochs=1,
                                   batch_size=8,
                                   validation_data=([test_Is.load_next_set(), mask],
                                                     validation_perceptual_Is))
                   )
    finally:
        del perceptual_Is
        del sample2
        gc.collect()
        model.save("best_inpainter")

import gc
perceptual_Is = []
if sys.argv[1] == 'train':
    for _ in range(500):
        trainer()
    model.save("best_inpainter")
elif sys.argv[1] == 'resume':
    model = tf.keras.models.load_model("best_inpainter")
    for _ in range(500):
        trainer()
    model.save("best_inpainting")
else:
    model = tf.keras.models.load_model("best_inpainter")
    model.compile()
    eval_im = ProgressiveLoader(val_data_dir, slice=1)
    ret = model.predict([eval_im.load_next_set(), generate_mask(input_shape)])
    import pdb; pdb.set_trace()

import keras

import keras.backend as K

from keras.models import Model
from keras.layers import Reshape
from keras.layers import Lambda


def gram_matrix(input_):
    reshape_layer = Reshape((int(input_.shape[1] * input_.shape[2]), int(input_.shape[3])))
    flatten = reshape_layer(input_)
    K_n = 1 / int(input_.shape[1] * input_.shape[2]* input_.shape[3])
    flatten_T = Lambda(lambda a: K_n * K.permute_dimensions(a, (0, 2, 1)))(flatten)
    output = keras.layers.dot([flatten, flatten_T], (1, 2))
    return output


def vggperceptual(model, dim, depth, val_sample):
    lossModel = keras.applications.vgg16.VGG16(include_top=False,
                                               weights='imagenet',
                                               input_shape=(dim, dim, depth))

    for layer in lossModel.layers:
        layer.trainable=False

    selectedlayers = [3, 6, 10]

    layers_loss_model = Model(lossModel.inputs,
                              [lossModel.layers[idx].output
                                for idx in selectedlayers] +
                              [gram_matrix(lossModel.layers[idx].output)
                                for idx in selectedlayers])

    lossModel.layers[1].output

    #validation_perceptual_Is = layers_loss_model.predict(test_Is * 256 - 127) + [test_Is]
    validation_perceptual_ls = layers_loss_model.predict(val_sample) + [val_sample]

    #loss_model_outputs = Lambda(lambda x: x * 256 - 127)(model.output)
    loss_model_outputs = layers_loss_model(model.output)
    return validation_perceptual_ls, loss_model_outputs

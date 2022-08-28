
# from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
# from lib.utils import get_backend
# from ._base import KerasModel, ModelBase

if True:# get_backend() == "amd":
    from keras.layers import Dense, Flatten, Reshape, Input
    from keras.models import load_model, Model as KModel
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import Dense, Flatten, Reshape, Input  # noqa pylint:disable=import-error,no-name-in-module
from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
import sys

from threading import Lock
from time import sleep
from typing import cast, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

# import cv2
import numpy as np
# from matplotlib import pyplot as plt, rcParams

from lib.image import read_image_meta
from lib.keypress import KBHit
from lib.multithreading import MultiThread
from lib.utils import (deprecation_warning, get_dpi, get_folder, get_image_paths,
                       FaceswapError, _image_extensions)
# from plugins.plugin_loader import PluginLoader
from importlib import import_module

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

class Model():
    """ Original Faceswap Model.

    This is the original faceswap model and acts as a template for plugin development.

    All plugins must define the following attribute override after calling the parent's
    :func:`__init__` method:

        * :attr:`input_shape` (`tuple` or `list`): a tuple of ints defining the shape of the \
        faces that the model takes as input. If the input size is the same for both sides, this \
        can be a single 3 dimensional tuple. If the inputs have different sizes for "A" and "B" \
        this should be a list of 2 3 dimensional shape tuples, 1 for each side.

    Any additional attributes used exclusively by this model should be defined here, but make sure
    that you are not accidentally overriding any existing
    :class:`~plugins.train.model._base.ModelBase` attributes.

    Parameters
    ----------
    args: varies
        The default command line arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    kwargs: varies
        The default keyword arguments passed in from :class:`~scripts.train.Train` or
        :class:`~scripts.train.Convert`
    """
    # @property
    # def config(self) -> dict:
    #     """ dict: The configuration dictionary for current plugin, as set by the user's
    #     configuration settings. """
    #     global _CONFIG  # pylint: disable=global-statement
    #     if not _CONFIG:
    #         model_name = self._config_section
    #         log/ger.debug("Loading config for: %s", model_name)
    #         _CONFIG = Config(model_name, configfile=self._configfile).config_dict
    #     return _CONFIG


    def __init__(self, *args, **kwargs):
        self.name = "modeliuks"
        super().__init__(*args, **kwargs)
        self.input_shape = (64, 64, 3)
        self.low_mem = True # self.config.get("lowmem", False)
        self.learn_mask = None #self.config["learn_mask"]
        self.encoder_dim = 512 if self.low_mem else 1024

    def build_model(self, inputs):
        """ Create the model's structure.

        This function is automatically called immediately after :func:`__init__` has been called if
        a new model is being created. It is ignored if an existing model is being loaded from disk
        as the model structure will be defined in the saved model file.

        The model's final structure is defined here.

        For the original model, An encoder instance is defined, then the same instance is
        referenced twice, one for each input "A" and "B" so that the same model is used for
        both inputs.

        2 Decoders are then defined (one for each side) with the encoder instances passed in as
        input to the corresponding decoders.

        It is important to note that any models and sub-models should not call
        :class:`keras.models.Model` directly, but rather call
        :class:`plugins.train.model._base.KerasModel`. This acts as a wrapper for Keras' Model
        class, but handles some minor differences which need to be handled between Nvidia and AMD
        backends.

        The final output of the model should always call :class:`lib.model.nn_blocks.Conv2DOutput`
        so that the correct data type is set for the final activation, to support Mixed Precision
        Training. Failure to do so is likely to lead to issues when Mixed Precision is enabled.

        Parameters
        ----------
        inputs: list
            A list of input tensors for the model. This will be a list of 2 tensors of
            shape :attr:`input_shape`, the first for side "a", the second for side "b".

        Returns
        -------
        :class:`keras.models.Model`
            The output of this function must be a keras model generated from
            :class:`plugins.train.model._base.KerasModel`. See Keras documentation for the correct
            structure, but note that parameter :attr:`name` is a required rather than an optional
            argument in Faceswap. You should assign this to the attribute ``self.name`` that is
            automatically generated from the plugin's filename.
        """
        input_a = inputs[0]
        input_b = inputs[1]

        encoder = self.encoder()
        encoder_a = [encoder(input_a)]
        encoder_b = [encoder(input_b)]

        outputs = [self.decoder("a")(encoder_a), self.decoder("b")(encoder_b)]

        if True: #get_backend() == "amd":
            # #logger.debug("Flattening inputs (%s) and outputs (%s) for AMD", inputs, outputs)
            import numpy as np
            inputs = np.array(inputs).flatten().tolist()
            outputs = np.array(outputs).flatten().tolist()
            # #logger.debug("Flattened inputs (%s) and outputs (%s)", inputs, outputs)

        autoencoder = KModel(inputs, outputs, name=self.model_name)

        return autoencoder

    def encoder(self):
        """ The original Faceswap Encoder Network.

        The encoder for the original model has it's weights shared between both the "A" and "B"
        side of the model, so only one instance is created :func:`build_model`. However this same
        instance is then used twice (once for A and once for B) meaning that the weights get
        shared.

        Returns
        -------
        :class:`keras.models.Model`
            The Keras encoder model, for sharing between inputs from both sides.
        """
        input_ = Input(shape=self.input_shape)
        var_x = input_
        var_x = Conv2DBlock(128, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(256, activation="leakyrelu")(var_x)
        var_x = Conv2DBlock(512, activation="leakyrelu")(var_x)
        if not self.low_mem:
            var_x = Conv2DBlock(1024, activation="leakyrelu")(var_x)
        var_x = Dense(self.encoder_dim)(Flatten()(var_x))
        var_x = Dense(4 * 4 * 1024)(var_x)
        var_x = Reshape((4, 4, 1024))(var_x)
        var_x = UpscaleBlock(512, activation="leakyrelu")(var_x)
        return KModel(input_, var_x, name="encoder")

    def decoder(self, side):
        """ The original Faceswap Decoder Network.

        The decoders for the original model have separate weights for each side "A" and "B", so two
        instances are created in :func:`build_model`, one for each side.

        Parameters
        ----------
        side: str
            Either `"a` or `"b"`. This is used for naming the decoder model.

        Returns
        -------
        :class:`keras.models.Model`
            The Keras decoder model. This will be called twice, once for each side.
        """
        input_ = Input(shape=(8, 8, 512))
        var_x = input_
        var_x = UpscaleBlock(256, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(128, activation="leakyrelu")(var_x)
        var_x = UpscaleBlock(64, activation="leakyrelu")(var_x)
        var_x = Conv2DOutput(3, 5, name=f"face_out_{side}")(var_x)
        outputs = [var_x]

        if self.learn_mask:
            var_y = input_
            var_y = UpscaleBlock(256, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(128, activation="leakyrelu")(var_y)
            var_y = UpscaleBlock(64, activation="leakyrelu")(var_y)
            var_y = Conv2DOutput(1, 5, name=f"mask_out_{side}")(var_y)
            outputs.append(var_y)
        return KModel(input_, outputs=outputs, name=f"decoder_{side}")

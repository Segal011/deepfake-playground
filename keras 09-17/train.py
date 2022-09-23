
from threading import Lock
from time import sleep
from typing import cast, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from importlib import import_module

# import cv2
import numpy as np
# from matplotlib import pyplot as plt, rcParams
import sys
from helpers.model import import_model
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal


# from lib.image import read_image_meta
# from lib.keypress import KBHit
# from lib.multithreading import MultiThread
# from lib.utils import  get_folder
# from plugins.plugin_loader import PluginLoader
from helpers.image import get_image_paths, read_image_meta
from helpers.model import import_model

# from lib.image import read_image_meta



if True:# get_backend() == "amd":
    from keras.layers import Dense, Flatten, Reshape, Input
    from keras.models import load_model, Model as KModel
else:
    # Ignore linting errors from Tensorflow's thoroughly broken import system
    from tensorflow.keras.layers import Dense, Flatten, Reshape, Input  # noqa pylint:disable=import-error,no-name-in-module
# from lib.model.nn_blocks import Conv2DOutput, Conv2DBlock, UpscaleBlock
import sys
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class Train():

    def __init__(self, args) -> None:
        self._args = args
        self._images = self._get_images()
        self._stop = False
        self._save_now: bool = False

        # self._preview = Preview()
            
    def train(self) -> None:
        """ The training process to be run inside a thread. """
        try:
            # sleep(0.5)  # Let preview instructions flush out to #logger
            logger.debug("Commencing Training")
            # #logger.info("Loading data, this may take a while...")
            # model = self._load_model()
            model = self._load_model(r"C:\Users\37060\Desktop\smth\model")
            # model = load_model(r"C:\Users\37060\Desktop\smth\model\lightweight.h5")
            # model = load_model(r"C:\Users\37060\Documents\GitHub\deepfake-playground\keras\train.hdf5")
            trainer = self._load_trainer(model)            
            self._run_training_cycle(model, trainer)
        except KeyboardInterrupt:
            try:
                logger.debug("Keyboard Interrupt Caught. Saving Weights and exiting")
                model.save(is_exit=True)
                trainer.clear_tensorboard()
            except KeyboardInterrupt:
                pass
                logger.info("Saving model weights has been cancelled!")
            sys.exit(0)
        except Exception as err:
            logger.error(err.__str__)
            raise err

    def _load_model(self, filepath=None) -> KModel:
        """ Load the model requested for training.

        Returns
        -------
        :file:`plugins.train.model` plugin
            The requested model plugin
        """
        #logger.debug("Loading Model")
        # model_dir = get_folder(self._args.model_dir)
        # model = PluginLoader.get_model(self._args.trainer)

        model = import_model( "train.model", self._args.trainer, False)(
            filepath,
            self._args,
            predict=False)
        model.build()
        # logger.debug("Loaded Model")
        return model

        # name = self._args.trainer.replace("-", "_")
        # attr = "train.model"
        # ttl = attr.split(".")[-1].title()
        # # if not disable_logging:
        # #     logger.info("Loading %s from %s plugin...", ttl, name.title())
        # attr = "model" if attr == "Trainer" else attr.lower()
        # mod = ".".join(("plugins", attr, name))
        # mod = "model"
        # # print(mod)
        # module = import_module(mod)
        # # print(module)
        # # print(ttl)
        # # print(attr)
        # model = getattr(module, "model")
        # print(model)
        # print(model_dir)
        # print(self._args)
        # model = model(
        #     model_dir,
        #     self._args, predict=False)

        # model.build()
        # #logger.debug("Loaded Model")
        # return model

    def _load_trainer(self, model):
        """ Load the trainer requested for training.

        Parameters
        ----------
        model: :file:`plugins.train.model` plugin
            The requested model plugin

        Returns
        -------
        :file:`plugins.train.trainer` plugin
            The requested model trainer plugin
        """
        #logger.debug("Loading Trainer")
        # base = PluginLoader.get_trainer(model.trainer)
        # trainer = base(model,
        #                               self._images,
        #                               self._args.batch_size,
        #                               self._args.configfile)
        #logger.debug("Loaded Trainer")

        name = self._args.trainer
        attr = "train.trainer"
        ttl = attr.split(".")[-1].title()
        # if not disable_logging:
        #     logger.info("Loading %s from %s plugin...", ttl, name.title())
        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        # print("RERE")
        module = import_module(mod)
        # print("RERE")
        trainer = getattr(module, ttl)(
            model,
            self._images,
            self._args.batch_size,
            self._args.configfile
        )
        # print(module)
        # print(trainer)
        return trainer

    def _run_training_cycle(self, model, trainer) -> None:
        """ Perform the training cycle.

        Handles the background training, updating previews/time-lapse on each save interval,
        and saving the model.

        Parameters
        ----------
        model: :file:`plugins.train.model` plugin
            The requested model plugin
        trainer: :file:`plugins.train.trainer` plugin
            The requested model trainer plugin
        """
        logger.debug("Running Training Cycle")
        # if self._args.write_image or self._args.redirect_gui or self._args.preview:
        #     display_func: Optional[Callable] = self._show
        # else:
        #     display_func = None
        for iteration in range(1, self._args.iterations + 1):
            logger.info("Training iteration: %s", iteration)  # type:ignore
            save_iteration = iteration % self._args.save_interval == 0 or iteration == 1

            # if self._preview.should_toggle_mask():
            #     trainer.toggle_mask()
            #     self._preview.request_refresh()

                # if self._preview.should_refresh():
                #     viewer = display_func
                # else:
                # viewer = None
            viewer = None
            timelapse = {} #self._timelapse if save_iteration else {}
            trainer.train_one_step(viewer, timelapse)

            if viewer is not None and not save_iteration:
                # Spammy but required by GUI to know to update window
                print("\n")
                #logger.info("[Preview Updated]")

            if self._stop:
                logger.debug("Stop received. Terminating")
                break

            if save_iteration or self._save_now:
                logger.debug("Saving (save_iterations: %s, save_now: %s) Iteration: "
                              "(iteration: %s)", save_iteration, self._save_now, iteration)
                model.save(is_exit=False)
                self._save_now = False
                # self._preview.request_refresh()

        logger.info("Training cycle complete")
        model.save(is_exit=True)
        trainer.clear_tensorboard()
        self._stop = True
    
    def _get_images(self) -> Dict[Literal["a", "b"], List[str]]:
        """ Check the image folders exist and contains valid extracted faces. Obtain image paths.

        Returns
        -------
        dict
            The image paths for each side. The key is the side, the value is the list of paths
            for that side.
        """
        # logger.debug("Getting image paths")
        images = {}
        import os
        for side in ("a", "b"):
            side = cast(Literal["a", "b"], side)
            image_dir = getattr(self._args, f"input_{side}")
            if not os.path.isdir(image_dir):
                # logger.error("Error: '%s' does not exist", image_dir)
                sys.exit(1)

            images[side] = get_image_paths(image_dir, ".png")
            if not images[side]:
                # logger.error("Error: '%s' contains no images", image_dir)
                sys.exit(1)
            # Validate the first image is a detected face
            test_image = next(img for img in images[side])

            meta = read_image_meta(test_image)
            # logger.debug("Test file: (filename: %s, metadata: %s)", test_image, meta)
            if "itxt" not in meta or "alignments" not in meta["itxt"]:
                # logger.error("The input folder '%s' contains images that are not extracted faces.",
                #              image_dir)
                # logger.error("You can only train a model on faces generated from Faceswap's "
                #              "extract process. Please check your sources and try again.")
                sys.exit(1)

        #     logger.info("Model %s Directory: '%s' (%s images)",
        #                 side.upper(), image_dir, len(images[side]))
        # logger.debug("Got image paths: %s", [(key, str(len(val)) + " images")
        #                                      for key, val in images.items()])
        # self._validate_image_counts(images)
        return images


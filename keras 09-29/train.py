
from typing import cast, Dict, List
from importlib import import_module

import sys
if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

from helpers.image import get_image_paths, read_image_meta
from helpers.model import import_model

from keras.models import load_model, Model as KModel

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
            logger.debug("Commencing Training")
            model = self._load_model(self._args.model_dir)
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
        model = import_model("train.model", self._args.trainer, False)(
            filepath,
            self._args,
            predict=False)
        model.build()
        return model

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

        name = self._args.trainer
        attr = "train.trainer"
        ttl = attr.split(".")[-1].title()

        attr = "model" if attr == "Trainer" else attr.lower()
        mod = ".".join(("plugins", attr, name))
        module = import_module(mod)
        trainer = getattr(module, ttl)(
            model,
            self._images,
            self._args.batch_size,
            self._args.configfile
        )
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

            viewer = None
            timelapse = {} #self._timelapse if save_iteration else {}
            trainer.train_one_step(viewer, timelapse)

            if viewer is not None and not save_iteration:
                # Spammy but required by GUI to know to update window
                print("\n")

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
                sys.exit(1)

            images[side] = get_image_paths(image_dir, ".png")
            if not images[side]:
                sys.exit(1)
            test_image = next(img for img in images[side])

            meta = read_image_meta(test_image)
            if "itxt" not in meta or "alignments" not in meta["itxt"]:

                sys.exit(1)
        return images


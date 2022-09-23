#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import logging
import numpy as np


from plugins.convert.writer.opencv import Writer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DiskIO():
    """ Disk Input/Output for the converter process.

    Background threads to:
        * Load images from disk and get the detected faces
        * Save images back to disk

    Parameters
    ----------
    alignments: :class:`lib.alignmnents.Alignments`
        The alignments for the input video
    images: :class:`lib.image.ImagesLoader`
        The input images
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """

    def __init__(self, alignments, images, arguments):
        logger.debug("Initializing %s: (alignments: %s, images: %s, arguments: %s)",
                     self.__class__.__name__, alignments, images, arguments)
        self._args = arguments
        logger.debug("Initialized %s", self.__class__.__name__)


    @property
    def pre_encode(self):
        logger.info("Pre-encoding image")  # type:ignore
              
        args = [self._args.output_dir]
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        writer =    Writer(*args, configfile=configfile)
        dummy = np.zeros((20, 20, 3), dtype="uint8")
        test = writer.pre_encode(dummy)
        retval = None if test is None else writer.pre_encode
        logger.debug("Writer pre_encode function: %s", retval)
        print("retval", retval)
        return retval

  
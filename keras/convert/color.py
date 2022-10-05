#!/usr/bin/env python3
""" Average colour adjustment color matching adjustment plugin for faceswap.py converter """

#!/usr/bin/env python3
""" Parent class for color Adjustments for faceswap.py converter """

import logging
import numpy as np

from plugins.convert._config import Config

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def get_config(plugin_name, configfile=None):
    """ Return the config for the requested model """
    return Config(plugin_name, configfile=configfile).config_dict


class Color():
    """ Parent class for adjustments """
    def __init__(self, configfile=None, config=None):
        logger.debug("Initializing %s: (configfile: %s, config: %s)",
                     self.__class__.__name__, configfile, config)
        self.config = self.set_config(configfile, config)
        logger.debug("config: %s", self.config)
        logger.debug("Initialized %s", self.__class__.__name__)

    def set_config(self, configfile, config):
        """ Set the config to either global config or passed in config """
        section = ".".join(self.__module__.split(".")[-2:])
        if config is None:
            retval = get_config(section, configfile)
        else:
            config.section = section
            retval = config.config_dict
            config.section = None
        logger.debug("Config: %s", retval)
        return retval

    def run(self, old_face, new_face, raw_mask):
        """ Perform selected adjustment on face """
        logger.info("Performing color adjustment")
        # Remove Mask for processing
        reinsert_mask = False
        if new_face.shape[2] == 4:
            reinsert_mask = True
            final_mask = new_face[:, :, -1]
            new_face = new_face[:, :, :3]
        new_face = self.process(old_face, new_face, raw_mask)
        new_face = np.clip(new_face, 0.0, 1.0)
        if reinsert_mask and new_face.shape[2] != 4:
            # Reinsert Mask
            new_face = np.concatenate((new_face, np.expand_dims(final_mask, axis=-1)), -1)
        logger.info("Performed color adjustment")
        return new_face
    """ Adjust the mean of the color channels to be the same for the swap and old frame """

    # TODO play with different color processors. Here is color.py
    @staticmethod
    def process(old_face, new_face, raw_mask):
        for _ in [0, 1]:
            diff = old_face - new_face
            avg_diff = np.sum(diff * raw_mask, axis=(0, 1))
            adjustment = avg_diff / np.sum(raw_mask, axis=(0, 1))
            new_face += adjustment
        return new_face

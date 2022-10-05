#!/usr/bin/env python3
""" Plugin to blend the edges of the face between the swap and the original face. """


import logging
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np

from convert.align import BlurMask, DetectedFace

logger = logging.getLogger(__name__)


class Mask():  # pylint:disable=too-few-public-methods
    """ Manipulations to perform to the mask that is to be applied to the output of the Faceswap
    model.

    Parameters
    ----------
    mask_type: str
        The mask type to use for this plugin
    output_size: int
        The size of the output from the Faceswap model.
    coverage_ratio: float
        The coverage ratio that the Faceswap model was trained at.
    configfile: str, Optional
        Optional location of custom configuration ``ini`` file. If ``None`` then use the default
        config location. Default: ``None``
    config: :class:`lib.config.FaceswapConfig`, Optional
        Optional pre-loaded :class:`lib.config.FaceswapConfig`. If passed, then this will be used
        over any configuration on disk. If ``None`` then it is ignored. Default: ``None``

    """
    def __init__(self,
                 output_size: int,
                 coverage_ratio: float) -> None:

        self._config = {'type': 'normalized', 'kernel_size': 3, 'passes': 4}
        logger.debug("config: %s", self._config)

        self._coverage_ratio = coverage_ratio
        self._output_size =output_size

    def run(self,
            detected_face: DetectedFace) -> Tuple[np.ndarray, np.ndarray]:
        """ Obtain the requested mask type and perform any defined mask manipulations.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        sub_crop_offset: :class:`numpy.ndarray`, optional
            The (x, y) offset to crop the mask from the center point.
        centering: [`"legacy"`, `"face"`, `"head"`]
            The centering to obtain the mask for
        predicted_mask: :class:`numpy.ndarray`, optional
            The predicted mask as output from the Faceswap Model, if the model was trained
            with a mask, otherwise ``None``. Default: ``None``.

        Returns
        -------
        mask: :class:`numpy.ndarray`
            The mask with all requested manipulations applied
        raw_mask: :class:`numpy.ndarray`
            The mask with no erosion/dilation applied
        """

        box = np.zeros((self._output_size, self._output_size, 1), dtype="float32")
        edge = (self._output_size // 32) + 1
        box[edge:-edge, edge:-edge] = 1.0

        box = BlurMask("gaussian", box, 6, is_ratio=True).blurred

        mask = self._get_stored_mask(detected_face, box)
        raw_mask = mask.copy()

        out = np.minimum(mask, box)

        logger.info(  # type: ignore
            "mask shape: %s, raw_mask shape: %s", mask.shape, raw_mask.shape)
        return out, raw_mask

    def _get_stored_mask(self, detected_face: DetectedFace, box: np.ndarray) -> np.ndarray:
        """ get the requested stored mask from the detected face object.

        Parameters
        ----------
        detected_face: :class:`lib.align.DetectedFace`
            The DetectedFace object as returned from :class:`scripts.convert.Predictor`.
        centering: [`"legacy"`, `"face"`, `"head"`]
            The centering to obtain the mask for
        sub_crop_offset: :class:`numpy.ndarray`
            The (x, y) offset to crop the mask from the center point. Set to `None` if the mask
            does not need to be offset for alternative centering

        Returns
        -------
        :class:`numpy.ndarray`
            The mask sized to Faceswap model output with any requested blurring applied.
        """
        mask = detected_face.mask

        mask = self._crop_to_coverage(mask)
        mask_size = mask.shape[0]
        face_size = box.shape[0]
        if mask_size != face_size:
            interp = cv2.INTER_CUBIC if mask_size < face_size else cv2.INTER_AREA
            mask = cv2.resize(mask,
                              box.shape[:2],
                              interpolation=interp)[..., None].astype("float32")
        else:
            mask = np.float32(mask)
        return mask

    def _crop_to_coverage(self, mask: np.ndarray) -> np.ndarray:
        """ Crop the mask to the correct dimensions based on coverage ratio.

        Parameters
        ----------
        mask: :class:`numpy.ndarray`
            The original mask to be cropped

        Returns
        -------
        :class:`numpy.ndarray`
            The cropped mask
        """
        if self._coverage_ratio == 1.0:
            return mask
        mask_size = mask.shape[0]
        padding = round((mask_size * (1 - self._coverage_ratio)) / 2)
        mask_slice = slice(padding, mask_size - padding)
        mask = mask[mask_slice, mask_slice, :]
        logger.info("mask_size: %s, coverage: %s, padding: %s, final shape: %s",  # type: ignore
                     mask_size, self._coverage_ratio, padding, mask.shape)
        return mask

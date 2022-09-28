#!/usr/bin/env python3
""" Converter for Faceswap """

import logging
import cv2
import numpy as np

from convert.convert.mask.mask_blend import Mask
from convert.convert.color.avg_color import Color

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Converter():
    def __init__(self, output_size, coverage_ratio, centering,
                 arguments, configfile=None):
        self._output_size = output_size
        self._coverage_ratio = coverage_ratio
        self._centering = centering
        self._args = arguments
        self._configfile = configfile

        self._scale = arguments.output_scale / 100
        self._adjustments = dict(mask=None, color=None, seamless=None, sharpening=None)

# TODO pazaist su colot ir mask, kurie aprasyti argumentuose
        self._adjustments["mask"] = Mask(
            self._args.mask_type,
            self._output_size,
            self._coverage_ratio,
            configfile=self._configfile
        )

        self._adjustments["color"] = Color(configfile=self._configfile)

    def process(self, item):
        """ Main convert process.

        Takes items from the in queue, runs the relevant adjustments, patches faces to final frame
        and outputs patched frame to the out queue.

        Parameters
        ----------
        in_queue: :class:`queue.Queue`
            The output from :class:`scripts.convert.Predictor`. Contains detected faces from the
            Faceswap model as well as the frame to be patched.
        out_queue: :class:`queue.Queue`
            The queue to place patched frames into for writing by one of Faceswap's
            :mod:`plugins.convert.writer` plugins.
        """
        logger.debug("Starting convert process.")
        images = {}
        logger.info("Patch queue got: '%s'", item["filename"])
        try:
            image = self._patch_image(item)
        except Exception as err:  # pylint: disable=broad-except
            # Log error and output original frame
            logger.error("Failed to convert image: '%s'. Reason: %s",  item["filename"], str(err))
            image = item["image"]
            logger.warning("Convert error traceback:", exc_info=True)

        logger.info("Out queue put: %s", item["filename"])
        images[item["filename"]] = image
        
        logger.debug("Completed convert process")
        return images

    def _patch_image(self, predicted):
        """ Patch a swapped face onto a frame.

        Run selected adjustments and swap the faces in a frame.

        Parameters
        ----------
        predicted: dict
            The output from :class:`scripts.convert.Predictor`.

        Returns
        -------
        :class: `numpy.ndarray` or pre-encoded image output
            The final frame ready for writing by a :mod:`plugins.convert.writer` plugin.
            Frame is either an array, or the pre-encoded output from the writer's pre-encode
            function (if it has one)

        """
        logger.info("Patching image: '%s'", predicted["filename"])
        frame_size = (predicted["image"].shape[1], predicted["image"].shape[0])
        new_image, background = self._get_new_image(predicted, frame_size)
        patched_face = self._post_warp_adjustments(background, new_image)
        patched_face = self._scale_image(patched_face)
        patched_face *= 255.0
        patched_face = np.rint(patched_face,
                               out=np.empty(patched_face.shape, dtype="uint8"),
                               casting='unsafe')
        patched_face = [cv2.imencode('.png',  # pylint: disable=no-member
                                    patched_face,
                                    (cv2.IMWRITE_PNG_COMPRESSION, 3)
                                    )[1]]

        logger.info("Patched image: '%s'", predicted["filename"])
        return patched_face

    def _get_new_image(self, predicted, frame_size):
        """ Get the new face from the predictor and apply pre-warp manipulations.

        Applies any requested adjustments to the raw output of the Faceswap model
        before transforming the image into the target frame.

        Parameters
        ----------
        predicted: dict
            The output from :class:`scripts.convert.Predictor`.
        frame_size: tuple
            The (`width`, `height`) of the final frame in pixels

        Returns
        -------
        placeholder:  :class: `numpy.ndarray`
            The original frame with the swapped faces patched onto it
        background:  :class: `numpy.ndarray`
            The original frame
        """
        logger.info("Getting: (filename: '%s', faces: %s)",
                     predicted["filename"], len(predicted["swapped_faces"]))

        placeholder = np.zeros((frame_size[1], frame_size[0], 4), dtype="float32")
        background = predicted["image"] / np.array(255.0, dtype="float32")
        placeholder[:, :, :3] = background

        for new_face, detected_face, reference_face in zip(predicted["swapped_faces"],
                                                           predicted["detected_faces"],
                                                           predicted["reference_faces"]):
            predicted_mask = new_face[:, :, -1] if new_face.shape[2] == 4 else None
            new_face = new_face[:, :, :3]
            interpolator = reference_face.interpolators[1]

            new_face = self._pre_warp_adjustments(new_face,
                                                  detected_face,
                                                  reference_face,
                                                  predicted_mask)

            # Warp face with the mask
            cv2.warpAffine(new_face,
                           reference_face.adjusted_matrix,
                           frame_size,
                           placeholder,
                           flags=cv2.WARP_INVERSE_MAP | interpolator,
                           borderMode=cv2.BORDER_TRANSPARENT)

        logger.info("Got filename: '%s'. (placeholders: %s)",
                     predicted["filename"], placeholder.shape)

        return placeholder, background

    def _pre_warp_adjustments(self, new_face, detected_face, reference_face, predicted_mask):
        """ Run any requested adjustments that can be performed on the raw output from the Faceswap
        model.

        Any adjustments that can be performed before warping the face into the final frame are
        performed here.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model.
        detected_face: :class:`~lib.align.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        reference_face: :class:`~lib.align.AlignedFace`
            The aligned face object sized to the model output of the original face for reference
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask

        Returns
        -------
        :class:`numpy.ndarray`
            The face output from the Faceswap Model with any requested pre-warp adjustments
            performed.
        """
        logger.info("new_face shape: %s, predicted_mask shape: %s", new_face.shape,
                     predicted_mask.shape if predicted_mask is not None else None)
        old_face = reference_face.face[..., :3] / 255.0
        new_face, raw_mask = self._get_image_mask(new_face,
                                                  detected_face,
                                                  predicted_mask,
                                                  reference_face)
        if self._adjustments["color"] is not None:
            new_face = self._adjustments["color"].run(old_face, new_face, raw_mask)
        if self._adjustments["seamless"] is not None:
            new_face = self._adjustments["seamless"].run(old_face, new_face, raw_mask)
        logger.info("returning: new_face shape %s", new_face.shape)
        return new_face

    def _get_image_mask(self, new_face, detected_face, predicted_mask, reference_face):
        """ Return any selected image mask

        Places the requested mask into the new face's Alpha channel.

        Parameters
        ----------
        new_face: :class:`numpy.ndarray`
            The swapped face received from the faceswap model.
        detected_face: :class:`~lib.DetectedFace`
            The detected_face object as defined in :class:`scripts.convert.Predictor`
        predicted_mask: :class:`numpy.ndarray` or ``None``
            The predicted mask output from the Faceswap model. ``None`` if the model
            did not learn a mask
        reference_face: :class:`~lib.align.AlignedFace`
            The aligned face object sized to the model output of the original face for reference

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped face with the requested mask added to the Alpha channel
        """
        logger.info("Getting mask. Image shape: %s", new_face.shape)
        if self._args.mask_type not in ("none", "predicted"):
            mask_centering = detected_face.mask[self._args.mask_type].stored_centering
        else:
            mask_centering = "face"  # Unused but requires a valid value
        crop_offset = (reference_face.pose.offset[self._centering] -
                       reference_face.pose.offset[mask_centering])
        mask, raw_mask = self._adjustments["mask"].run(detected_face, crop_offset, self._centering,
                                                       predicted_mask=predicted_mask)
        logger.info("Adding mask to alpha channel")
        new_face = np.concatenate((new_face, mask), -1)
        logger.info("Got mask. Image shape: %s", new_face.shape)
        return new_face, raw_mask

    def _post_warp_adjustments(self, background, new_image):
        """ Perform any requested adjustments to the swapped faces after they have been transformed
        into the final frame.

        Parameters
        ----------
        background: :class:`numpy.ndarray`
            The original frame
        new_image: :class:`numpy.ndarray`
            A blank frame of original frame size with the faces warped onto it

        Returns
        -------
        :class:`numpy.ndarray`
            The final merged and swapped frame with any requested post-warp adjustments applied
        """
        if self._adjustments["sharpening"] is not None:
            new_image = self._adjustments["sharpening"].run(new_image)

     
        foreground, mask = np.split(new_image,  # pylint:disable=unbalanced-tuple-unpacking
                                    (3, ),
                                    axis=-1)
        foreground *= mask
        background *= (1.0 - mask)
        background += foreground
        frame = background
        np.clip(frame, 0.0, 1.0, out=frame)
        return frame

    def _scale_image(self, frame):
        """ Scale the final image if requested.

        If output scale has been requested in command line arguments, scale the output
        otherwise return the final frame.

        Parameters
        ----------
        frame: :class:`numpy.ndarray`
            The final frame with faces swapped

        Returns
        -------
        :class:`numpy.ndarray`
            The final frame scaled by the requested scaling factor
        """
        if self._scale == 1:
            return frame
        # logger.info("source frame: %s", frame.shape)
        # interp = cv2.INTER_CUBIC if self._scale > 1 else cv2.INTER_AREA
        # dims = (round((frame.shape[1] / 2 * self._scale) * 2),
        #         round((frame.shape[0] / 2 * self._scale) * 2))
        # frame = cv2.resize(frame, dims, interpolation=interp)
        # logger.info("resized frame: %s", frame.shape)
        # np.clip(frame, 0.0, 1.0, out=frame)
        # return frame

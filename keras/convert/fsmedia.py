#!/usr/bin/env python3
""" Helper functions for :mod:`~scripts.extract` and :mod:`~scripts.convert`.

Holds the classes for the 2 main Faceswap 'media' objects: Images and Alignments.

Holds optional pre/post processing functions for convert and extract.
"""

import logging
import os
import cv2


from convert.align import Alignments as AlignmentsBase


logger = logging.getLogger(__name__)  # pylint:disable=invalid-name

def read_image(filename):
    retval = cv2.imread(filename)
    if retval is None:
        raise ValueError("Image is None")
    logger.info("Loaded image: '%s'.", filename)
    return retval



def finalize(images_found, num_faces_detected, verify_output):
    """ Output summary statistics at the end of the extract or convert processes.

    Parameters
    ----------
    images_found: int
        The number of images/frames that were processed
    num_faces_detected: int
        The number of faces that have been detected
    verify_output: bool
        ``True`` if multiple faces were detected in frames otherwise ``False``.
     """
    logger.info("-------------------------")
    logger.info("Images found:        %s", images_found)
    logger.info("Faces detected:      %s", num_faces_detected)
    logger.info("-------------------------")

    if verify_output:
        logger.info("Note:")
        logger.info("Multiple faces were detected in one or more pictures.")
        logger.info("Double check your results.")
        logger.info("-------------------------")

    logger.info("Process Succesfully Completed. Shutting Down...")


class Alignments(AlignmentsBase):
    """ Override :class:`lib.align.Alignments` to add custom loading based on command
    line arguments.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The command line arguments that were passed to Faceswap
    is_extract: bool
        ``True`` if the process calling this class is extraction otherwise ``False``
    input_is_video: bool, optional
        ``True`` if the input to the process is a video, ``False`` if it is a folder of images.
        Default: False
    """
    def __init__(self, arguments, is_extract, input_is_video=False):
        logger.debug("Initializing %s: (is_extract: %s, input_is_video: %s)",
                     self.__class__.__name__, is_extract, input_is_video)
        self._args = arguments
        self._is_extract = is_extract
        folder, filename = self._set_folder_filename(input_is_video)
        super().__init__(folder, filename=filename)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _set_folder_filename(self, input_is_video):
        """ Return the folder and the filename for the alignments file.

        If the input is a video, the alignments file will be stored in the same folder
        as the video, with filename `<videoname>_alignments`.

        If the input is a folder of images, the alignments file will be stored in folder with
        the images and just be called 'alignments'

        Parameters
        ----------
        input_is_video: bool, optional
            ``True`` if the input to the process is a video, ``False`` if it is a folder of images.

        Returns
        -------
        folder: str
            The folder where the alignments file will be stored
        filename: str
            The filename of the alignments file
        """
        if self._args.alignments_path:
            logger.debug("Alignments File provided: '%s'", self._args.alignments_path)
            folder, filename = os.path.split(str(self._args.alignments_path))
        else:
            logger.debug("Alignments from Input Folder: '%s'", self._args.input_dir)
            folder = str(self._args.input_dir)
            filename = "alignments"
        logger.debug("Setting Alignments: (folder: '%s' filename: '%s')", folder, filename)
        return folder, filename

    def _load(self):
        """ Override the parent :func:`~lib.align.Alignments._load` to handle skip existing
        frames and faces on extract.

        If skip existing has been selected, existing alignments are loaded and returned to the
        calling script.

        Returns
        -------
        dict
            Any alignments that have already been extracted if skip existing has been selected
            otherwise an empty dictionary
        """
        data = dict()
        if not self._is_extract:
            if not self.have_alignments_file:
                return data
            data = super()._load()
            return data

        skip_existing = hasattr(self._args, 'skip_existing') and self._args.skip_existing
        skip_faces = hasattr(self._args, 'skip_faces') and self._args.skip_faces

        if not skip_existing and not skip_faces:
            logger.debug("No skipping selected. Returning empty dictionary")
            return data

        if not self.have_alignments_file and (skip_existing or skip_faces):
            logger.warning("Skip Existing/Skip Faces selected, but no alignments file found!")
            return data

        data = super()._load()

        if skip_faces:
            # Remove items from alignments that have no faces so they will
            # be re-detected
            del_keys = [key for key, val in data.items() if not val["faces"]]
            logger.debug("Frames with no faces selected for redetection: %s", len(del_keys))
            for key in del_keys:
                if key in data:
                    logger.info("Selected for redetection: '%s'", key)
                    del data[key]
        return data



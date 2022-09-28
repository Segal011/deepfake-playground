#!/usr/bin/env python3
""" Helper functions for :mod:`~scripts.extract` and :mod:`~scripts.convert`.

Holds the classes for the 2 main Faceswap 'media' objects: Images and Alignments.

Holds optional pre/post processing functions for convert and extract.
"""

import logging
import os
import pickle
import zlib

import cv2



from convert.utils import FaceswapError

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


class Alignments():
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
        self._file = self._get_location(folder, filename)
        self._data = self._load()
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
        if not self.have_alignments_file:
            return data
        logger.debug("Loading alignments")
        if not self.have_alignments_file:
            raise FaceswapError("Error: Alignments file not found at "
                                "{}".format(self._file))

        logger.info("Reading alignments from: '%s'", self._file)
        try:
            with open(self._file, "rb") as s_file:
                data = s_file.read()
                logger.debug("stored data type: %s", type(data))
                data = zlib.decompress(data)
                data = pickle.loads(data)
                data = data.get("__data__", data)
        except IOError as err:
            msg = f"Error reading from '{self._file}': {err.strerror}"
            raise FaceswapError(msg) from err

        logger.debug("Loaded alignments")
        return data

    @property
    def file(self):
        """ str: The full path to the currently loaded alignments file. """
        return self._file

    @property
    def data(self):
        """ dict: The loaded alignments :attr:`file` in dictionary form. """
        return self._data

    @property
    def have_alignments_file(self):
        """ bool: ``True`` if an alignments file exists at location :attr:`file` otherwise
        ``False``. """
        retval = os.path.exists(self._file)
        logger.info(retval)
        return retval

    @property
    def mask_summary(self):
        """ dict: The mask type names stored in the alignments :attr:`data` as key with the number
        of faces which possess the mask type as value. """
        masks = dict()
        for val in self._data.values():
            for face in val["faces"]:
                if face.get("mask", None) is None:
                    masks["none"] = masks.get("none", 0) + 1
                for key in face.get("mask", dict()):
                    masks[key] = masks.get(key, 0) + 1
        return masks

    def _get_location(self, folder, filename):
        """ Obtains the location of an alignments file.

        If a legacy alignments file is provided/discovered, then the alignments file will be
        updated to the custom ``.fsa`` format and saved.

        Parameters
        ----------
        folder: str
            The folder that the alignments file is located in
        filename: str
            The filename of the alignments file

        Returns
        -------
        str
            The full path to the alignments file
        """
        noext_name, _ = os.path.splitext(filename)
        return os.path.join(str(folder), "{}.{}".format(noext_name, 'fsa'))

    def save(self):
        """ Write the contents of :attr:`data` and :attr:`_meta` to a serialized ``.fsa`` file at
        the location :attr:`file`. """
        logger.debug("Saving alignments")
        logger.info("Writing alignments to: '%s'", self._file)
        data = dict(__data__=self._data)

        # filename = self._check_extension(filename)
        try:
            with open(self._file, "wb") as s_file:
                data = pickle.dump(data)  # pylint: disable=protected-access
                data = zlib.compress(data)
                s_file.write(data)

        except IOError as err:
            msg = f"Error writing to '{self._file}': {err.strerror}"
            raise FaceswapError(msg) from err

        logger.debug("Saved alignments")

    def frame_exists(self, frame_name):
        """ Check whether a given frame_name exists within the alignments :attr:`data`.

        Parameters
        ----------
        frame_name: str
            The frame name to check. This should be the base name of the frame, not the full path

        Returns
        -------
        bool
            ``True`` if the given frame_name exists within the alignments :attr:`data`
            otherwise ``False``
        """
        retval = frame_name in self._data.keys()
        logger.info("'%s': %s", frame_name, retval)
        return retval

    def mask_is_valid(self, mask_type):
        """ Ensure the given ``mask_type`` is valid for the alignments :attr:`data`.

        Every face in the alignments :attr:`data` must have the given mask type to successfully
        pass the test.

        Parameters
        ----------
        mask_type: str
            The mask type to check against the current alignments :attr:`data`

        Returns
        -------
        bool:
            ``True`` if all faces in the current alignments possess the given ``mask_type``
            otherwise ``False``
        """
        retval = any([(face.get("mask", None) is not None and
                       face["mask"].get(mask_type, None) is not None)
                      for val in self._data.values()
                      for face in val["faces"]])
        logger.debug(retval)
        return retval

    # << DATA >> #

    def get_faces_in_frame(self, frame_name):
        """ Obtain the faces from :attr:`data` associated with a given frame_name.

        Parameters
        ----------
        frame_name: str
            The frame name to return faces for. This should be the base name of the frame, not the
            full path

        Returns
        -------
        list
            The list of face dictionaries that appear within the requested frame_name
        """
        logger.info("Getting faces for frame_name: '%s'", frame_name)
        return self._data.get(frame_name, dict()).get("faces", [])

#!/usr/bin/env python3
""" Alignments file functions for reading, writing and manipulating the data stored in a
serialized alignments file. """

import logging
import os
import pickle
import zlib


from convert.utils import FaceswapError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# VERSION TRACKING
# 1.0 - Never really existed. Basically any alignments file prior to version 2.0
# 2.0 - Implementation of full head extract. Any alignments version below this will have used
#       legacy extract
# 2.1 - Alignments data to extracted face PNG header. SHA1 hashes of faces no longer calculated
#       or stored in alignments file
# 2.2 - Add support for differently centered masks (i.e. not all masks stored as face centering)

class Alignments():
    """ The alignments file is a custom serialized ``.fsa`` file that holds information for each
    frame for a video or series of images.

    Specifically, it holds a list of faces that appear in each frame. Each face contains
    information detailing their detected bounding box location within the frame, the 68 point
    facial landmarks and any masks that have been extracted.

    Additionally it can also hold video meta information (timestamp and whether a frame is a
    key frame.)

    Parameters
    ----------
    folder: str
        The folder that contains the alignments ``.fsa`` file
    filename: str, optional
        The filename of the ``.fsa`` alignments file. If not provided then the given folder will be
        checked for a default alignments file filename. Default: "alignments"
    """
    def __init__(self, folder, filename="alignments"):
        logger.debug("Initializing %s: (folder: '%s', filename: '%s')",
                     self.__class__.__name__, folder, filename)
        self._file = self._get_location(folder, filename)
        self._data = self._load()

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


    def _load(self):
        """ Load the alignments data from the serialized alignments :attr:`file`.

        Populates :attr:`_meta` with the alignment file's meta information as well as returning
        the serialized data.

        Returns
        -------
        dict:
            The loaded alignments data
        """
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

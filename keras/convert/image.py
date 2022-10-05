#!/usr/bin python3
""" Utilities for working with images and videos """

import logging
import os
import cv2

from convert.align import  DetectedFace

logger = logging.getLogger(__name__)  # pylint:disable=invalid-name

def read_image(filename):
    retval = cv2.imread(filename)
    if retval is None:
        raise ValueError("Image is None")
    logger.info("Loaded image: '%s'.", filename)
    return retval



class ImagesLoader():
    """ Perform image loading from a folder of images or a video.

    Images will be loaded and returned in the order that they appear in the folder, or in the video
    to ensure deterministic ordering. Loading occurs in a background thread, caching 8 images at a
    time so that other processes do not need to wait on disk reads.

    See also :class:`ImageIO` for additional attributes.

    Parameters
    ----------
    path: str or list
        The path to load images from. This can be a folder which contains images a video file or a
        list of image files.
    queue_size: int, optional
        The amount of images to hold in the internal buffer. Default: 8.
    fast_count: bool, optional
        When loading from video, the video needs to be parsed frame by frame to get an accurate
        count. This can be done quite quickly without guaranteed accuracy, or slower with
        guaranteed accuracy. Set to ``True`` to count quickly, or ``False`` to count slower
        but accurately. Default: ``True``.
    skip_list: list, optional
        Optional list of frame/image indices to not load. Any indices provided here will be skipped
        when executing the :func:`load` function from the given location. Default: ``None``
    count: int, optional
        If the number of images that the loader will encounter is already known, it can be passed
        in here to skip the image counting step, which can save time at launch. Set to ``None`` if
        the count is not already known. Default: ``None``

    Examples
    --------
    Loading from a video file:

    >>> loader = ImagesLoader('/path/to/video.mp4')
    >>> for filename, image in loader.load():
    >>>     <do processing>
    """

    def __init__(self, path, alignments):

        # self.files = {}
        self.loaded_images = []
        self._location = path
        self._alignments = alignments
        self._count = 1
        self._file_list = []
        self._get_and_check_filelist() #TODO only for training and extracting

        self._read_images()
    def _get_and_check_filelist(self):
        "Checks directory if exsits and creates new if not"

        if not isinstance(self._location, str):
            logger.error("Location should be a string", self._location)

        if not os.path.exists(self._location):
            logger.debug("Creating folder: '%s'", self._location)
            os.makedirs(self._location, exist_ok=True)

        image_extensions = [  # pylint:disable=invalid-name
            ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]
        dir_scanned = sorted(os.scandir(self._location), key=lambda x: x.name)
        logger.debug("Scanned Folder contains %s files", len(dir_scanned))
        logger.info("Scanned Folder Contents: %s", dir_scanned)

        for chkfile in dir_scanned:
            if any(chkfile.name.lower().endswith(ext) for ext in image_extensions):
                logger.info("Adding '%s' to image list", chkfile.path)  # type:ignore
                self._file_list.append(chkfile.path)

        logger.debug("Returning %s images", len(self._file_list))

    @property
    def file_list(self):
        return self._file_list

    def _read_images(self):
        """ The load thread."""
        logger.info("Read images from file_list")
        for filename in self.file_list:
            image = read_image(filename)
            if image is None:
                logger.warning("Frame not loaded: '%s'", filename[0])
                continue
            if not image.any() and image.ndim not in (2, 3):
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue

            item = self._get_detected_faces(filename, image)
            self.loaded_images.append(item)

    def _get_detected_faces(self, filename, image):
        """ Return the detected faces for the given image.

        If we have an alignments file, then the detected faces are created from that file. If
        we're running On-The-Fly then they will be extracted from the extractor.

        Parameters
        ----------
        filename: str
            The filename to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.align.DetectedFace` objects
        """
        logger.info("Getting faces for: '%s'", filename)
        detected_faces = list()

        if False:  # not self._alignments.frame_exists(frame_name): TODO test aligments reee
            face = DetectedFace()
            face.from_image(image)
            detected_faces.append(face)
        else:
            faces = self._alignments.data.get(os.path.basename(filename), dict()).get("faces", [])

            for rawface in faces:
                face = DetectedFace()
                face.from_alignment(rawface, image=image)
                face.mask = face.mask["extended"].stored_mask
                detected_faces.append(face)
        return dict(filename=filename, image=image, detected_faces=detected_faces)


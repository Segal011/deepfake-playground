from ast import literal_eval
import logging
import os
import cv2

import struct
from typing import Optional, List
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_image_extensions = [  # pylint:disable=invalid-name
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"]


def get_folder(path: str, make_folder: bool = True) -> str:
    """ Return a path to a folder, creating it if it doesn't exist

    Parameters
    ----------
    path: str
        The path to the folder to obtain
    make_folder: bool, optional
        ``True`` if the folder should be created if it does not already exist, ``False`` if the
        folder should not be created

    Returns
    -------
    str or `None`
        The path to the requested folder. If `make_folder` is set to ``False`` and the requested
        path does not exist, then ``None`` is returned
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    logger.debug("Requested path: '%s'", path)
    if not make_folder and not os.path.isdir(path):
        logger.debug("%s does not exist", path)
        return ""
    os.makedirs(path, exist_ok=True)
    logger.debug("Returning: '%s'", path)
    return path


def get_image_paths(directory: str, extension: Optional[str] = None) -> List[str]:
    """ Obtain a list of full paths that reside within a folder.

    Parameters
    ----------
    directory: str
        The folder that contains the images to be returned
    extension: str
        The specific image extensions that should be returned

    Returns
    -------
    list
        The list of full paths to the images contained within the given folder
    """
    logger = logging.getLogger(__name__)  # pylint:disable=invalid-name
    image_extensions = _image_extensions if extension is None else [extension]
    dir_contents = []

    if not os.path.exists(directory):
        logger.debug("Creating folder: '%s'", directory)
        directory = get_folder(directory)

    dir_scanned = sorted(os.scandir(directory), key=lambda x: x.name)
    logger.debug("Scanned Folder contains %s files", len(dir_scanned))
    logger.info("Scanned Folder Contents: %s", dir_scanned)  # type:ignore

    for chkfile in dir_scanned:
        if any(chkfile.name.lower().endswith(ext) for ext in image_extensions):
            logger.info("Adding '%s' to image list", chkfile.path)  # type:ignore
            dir_contents.append(chkfile.path)

    logger.debug("Returning %s images", len(dir_contents))
    return dir_contents


def read_image_meta(filename):
    """ Read the Faceswap metadata stored in an extracted face's exif header.

    Parameters
    ----------
    filename: str
        Full path to the image to be retrieve the meta information for.

    Returns
    -------
    dict
        The output dictionary will contain the `width` and `height` of the png image as well as any
        `itxt` information.
    Example
    -------
    >>> image_file = "/path/to/image.png"
    >>> metadata = read_image_meta(image_file)
    >>> width = metadata["width]
    >>> height = metadata["height"]
    >>> faceswap_info = metadata["itxt"]
    """
    retval = dict()
    if os.path.splitext(filename)[-1].lower() != ".png":
        # Get the dimensions directly from the image for non-pngs
        logger.info("Non png found. Loading file for dimensions: '%s'", filename)
        img = cv2.imread(filename)
        retval["height"], retval["width"] = img.shape[:2]
        return retval
    with open(filename, "rb") as infile:
        try:
            chunk = infile.read(8)
        except PermissionError:
            raise PermissionError(f"PermissionError while reading: {filename}")

        if chunk != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Invalid header found in png: {filename}")

        while True:
            chunk = infile.read(8)
            length, field = struct.unpack(">I4s", chunk)
            logger.info("Read chunk: (chunk: %s, length: %s, field: %s", chunk, length, field)
            if not chunk or field == b"IDAT":
                break
            if field == b"IHDR":
                # Get dimensions
                chunk = infile.read(8)
                retval["width"], retval["height"] = struct.unpack(">II", chunk)
                length -= 8
            elif field == b"iTXt":
                keyword, value = infile.read(length).split(b"\0", 1)
                if keyword == b"faceswap":
                    retval["itxt"] = literal_eval(value[4:].decode("utf-8"))
                    break
                else:
                    logger.info("Skipping iTXt chunk: '%s'", keyword.decode("latin-1", "ignore"))
                    length = 0  # Reset marker for next chunk
            infile.seek(length + 4, 1)
    logger.info("filename: %s, metadata: %s", filename, retval)
    return retval


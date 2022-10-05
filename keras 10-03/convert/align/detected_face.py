#!/usr/bin python3
""" Face and landmarks detection for faceswap.py """
import logging
from zlib import compress, decompress
import dlib
import cv2
import numpy as np

from convert.dlib_face_detector import DlibDetectedFace

# from convert.image import read_image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def read_image(filename):
    retval = cv2.imread(filename)
    if retval is None:
        raise ValueError("Image is None")
    logger.info("Loaded image: '%s'.", filename)
    return retval

def parse_parts(landmarks):
    """ Extended face hull mask """
    # mid points between the side of face and eye point
    ml_pnt = (landmarks[36] + landmarks[0]) // 2
    mr_pnt = (landmarks[16] + landmarks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (landmarks[36] + ml_pnt) // 2
    qr_pnt = (landmarks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, landmarks[36], landmarks[37], landmarks[38], landmarks[39]))
    bot_r = np.array((landmarks[42], landmarks[43], landmarks[44], landmarks[45], qr_pnt))

    # Eyebrow arrays
    top_l = landmarks[17:22]
    top_r = landmarks[22:27]

    # Adjust eyebrow arrays
    landmarks[17:22] = top_l + ((top_l - bot_l) // 2)
    landmarks[22:27] = top_r + ((top_r - bot_r) // 2)

    r_jaw = (landmarks[0:9], landmarks[17:18])
    l_jaw = (landmarks[8:17], landmarks[26:27])
    r_cheek = (landmarks[17:20], landmarks[8:9])
    l_cheek = (landmarks[24:27], landmarks[8:9])
    nose_ridge = (landmarks[19:25], landmarks[8:9],)
    r_eye = (landmarks[17:22],
             landmarks[27:28],
             landmarks[31:36],
             landmarks[8:9])
    l_eye = (landmarks[22:27],
             landmarks[27:28],
             landmarks[31:36],
             landmarks[8:9])
    nose = (landmarks[27:31], landmarks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]
    return parts

class DetectedFace():
    def __init__(self, image=None, x=None, w=None, y=None, h=None, landmarks_xy=None, mask=None,
                 filename=None):
        logger.info("Initializing %s: (image: %s, x: %s, w: %s, y: %s, h:%s, landmarks_xy: %s, "
                     "mask: %s, filename: %s)",
                     self.__class__.__name__,
                     image.shape if image is not None and image.any() else image,
                     x, w, y, h, landmarks_xy,
                     {k: v.shape for k, v in mask} if mask is not None else mask,
                     filename)
        self.detector = dlib.get_frontal_face_detector() # TODO pabandyt ir cnn_face_detection_model
        self.predictor = dlib.shape_predictor(
            r"C:\Users\37060\Documents\GitHub\magistras\deepfake-playground\shape_predictor_68_face_landmarks.dat")


        self.image = image
        self.x = x  # pylint:disable=invalid-name
        self.w = w  # pylint:disable=invalid-name
        self.y = y  # pylint:disable=invalid-name
        self.h = h  # pylint:disable=invalid-name

        self.shape = None
        self.landmarks_xy = landmarks_xy
        self.mask = dict() if mask is None else mask

        self.aligned = None
        logger.info("Initialized %s", self.__class__.__name__)

    @property
    def left(self):
        """int: Left point (in pixels) of face detection bounding box within the parent image """
        return self.shape.left()

    @property
    def top(self):
        """int: Top point (in pixels) of face detection bounding box within the parent image """
        return self.shape.top()

    @property
    def right(self):
        """int: Right point (in pixels) of face detection bounding box within the parent image """
        return self.shape.right()

    @property
    def bottom(self):
        """int: Bottom point (in pixels) of face detection bounding box within the parent image """
        return self.shape.bottom()


    def detect_from_image(self, image):
        # image = self.tensor_or_path_to_ndarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_faces = self.detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        found_faces = []
        for face in detected_faces:
            shape = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), face)
            detected_face = dict(
                shape=[face.left(), face.top(), face.right(), face.bottom()],
                landmarks=[(shape.part(n).x, shape.part(n).y) for n in range(0, 68)]
            )

            massk = np.zeros((1, 128, 128, 1), dtype="float32")

            parts = parse_parts(np.array(detected_face['landmarks']))
            for item in parts:
                item = np.rint(np.concatenate(item)).astype("int32")
                hull = cv2.convexHull(item)
                cv2.fillConvexPoly(massk, hull, 1.0, lineType=cv2.LINE_AA)

            detected_face['mask'] = massk
            found_faces.append(detected_face)

        return found_faces[0]
    def from_alignment(self, alignment, image=None, with_thumb=False):
        """ Set the attributes of this class from an alignments file and optionally load the face
        into the ``image`` attribute.

        Parameters
        ----------
        alignment: dict
            A dictionary entry for a face from an alignments file containing the keys
            ``x``, ``w``, ``y``, ``h``, ``landmarks_xy``.
            Optionally the key ``thumb`` will be provided. This is for use in the manual tool and
            contains the compressed jpg thumbnail of the face to be allocated to :attr:`thumbnail.
            Optionally the key ``mask`` will be provided, but legacy alignments will not have
            this key.
        image: numpy.ndarray, optional
            If an image is passed in, then the ``image`` attribute will
            be set to the cropped face based on the passed in bounding box co-ordinates
        with_thumb: bool, optional
            Whether to load the jpg thumbnail into the detected face object, if provided.
            Default: ``False``
        """

        logger.info("Creating from alignment: (alignment: %s, has_image: %s)",
                     alignment, bool(image is not None))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        detected_faces = self.detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        found_faces = []
        for face in detected_faces:
            self.shape = face
            points = self.predictor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), face)
            detected_face = dict(
                shape=face,
                landmarks=[(points.part(n).x, points.part(n).y) for n in range(0, 68)]
            )
            self.landmarks = [(points.part(n).x, points.part(n).y) for n in range(0, 68)]

        # self.x = alignment["x"]
        # self.w = alignment["w"]
        # self.y = alignment["y"]
        # self.h = alignment["h"]
        self.landmarks_xy= alignment["landmarks_xy"]
        if not isinstance( self.landmarks_xy, np.ndarray):
            self.landmarks_xy= np.array( self.landmarks_xy , dtype="float32")
        # self.landmarks_xy = landmarks.copy()

        faceeee = DlibDetectedFace().detect_from_image(image)
        # Manual tool and legacy alignments will not have a mask
        self.aligned = None

        if alignment.get("mask", None) is not None:
            self.mask = dict()
            for name, mask_dict in alignment["mask"].items():
                self.mask[name] = Mask()
                self.mask[name].from_dict(mask_dict)
        if image is not None and image.any():
            self.image = image[self.shape.top(): self.shape.bottom(), self.shape.left(): self.shape.right()]
        # logger.info("Created from alignment: (x: %s, w: %s, y: %s. h: %s, "
        #              "landmarks: %s, mask: %s)",
        #              self.x, self.w, self.y, self.h, self.landmarks_xy, self.mask)


class Mask():
    """ Face Mask information and convenience methods

    Holds a Faceswap mask as generated from :mod:`plugins.extract.mask` and the information
    required to transform it to its original frame.

    Holds convenience methods to handle the warping, storing and retrieval of the mask.

    Parameters
    ----------
    storage_size: int, optional
        The size (in pixels) that the mask should be stored at. Default: 128.
    storage_centering, str (optional):
        The centering to store the mask at. One of `"legacy"`, `"face"`, `"head"`.
        Default: `"face"`

    Attributes
    ----------
    stored_size: int
        The size, in pixels, of the stored mask across its height and width.
    stored_centering: str
        The centering that the mask is stored at. One of `"legacy"`, `"face"`, `"head"`
    """
    def __init__(self, storage_size=128, storage_centering="face"):
        logger.info("Initializing: %s (storage_size: %s, storage_centering: %s)",
                     self.__class__.__name__, storage_size, storage_centering)
        self.stored_size = storage_size
        self.stored_centering = storage_centering

        self._mask = None
        self._affine_matrix = None
        self._interpolator = None

        self._blur = dict()
        self._blur_kernel = 0
        self._threshold = 0.0
        self._sub_crop = dict(size=None, slice_in=[], slice_out=[])
        self.set_blur_and_threshold()
        logger.info("Initialized: %s", self.__class__.__name__)

    @property
    def mask(self):
        """ numpy.ndarray: The mask at the size of :attr:`stored_size` with any requested blurring,
        threshold amount and centering applied."""
        mask = self.stored_mask
        if self._threshold != 0.0 or self._blur["kernel"] != 0:
            mask = mask.copy()
        if self._threshold != 0.0:
            mask[mask < self._threshold] = 0.0
            mask[mask > 255.0 - self._threshold] = 255.0
        # if self._blur["kernel"] != 0:
        #     mask = BlurMask(self._blur["type"],
        #                     mask,
        #                     self._blur["kernel"],
        #                     passes=self._blur["passes"]).blurred
        if self._sub_crop["size"]:  # Crop the mask to the given centering
            out = np.zeros((self._sub_crop["size"], self._sub_crop["size"], 1), dtype=mask.dtype)
            slice_in, slice_out = self._sub_crop["slice_in"], self._sub_crop["slice_out"]
            out[slice_out[0], slice_out[1], :] = mask[slice_in[0], slice_in[1], :]
            mask = out
        logger.info("mask shape: %s", mask.shape)
        return mask

    @property
    def stored_mask(self):
        """ :class:`numpy.ndarray`: The mask at the size of :attr:`stored_size` as it is stored
        (i.e. with no blurring/centering applied). """
        dims = (self.stored_size, self.stored_size, 1)
        mask = np.frombuffer(decompress(self._mask), dtype="uint8").reshape(dims)
        logger.info("stored mask shape: %s", mask.shape)
        return mask

    @property
    def original_roi(self):
        """ :class: `numpy.ndarray`: The original region of interest of the mask in the
        source frame. """
        points = np.array([[0, 0],
                           [0, self.stored_size - 1],
                           [self.stored_size - 1, self.stored_size - 1],
                           [self.stored_size - 1, 0]], np.int32).reshape((-1, 1, 2))
        matrix = cv2.invertAffineTransform(self._affine_matrix)
        roi = cv2.transform(points, matrix).reshape((4, 2))
        logger.info("Returning: %s", roi)
        return roi

    @property
    def affine_matrix(self):
        """ :class: `numpy.ndarray`: The affine matrix to transpose the mask to a full frame. """
        return self._affine_matrix

    @property
    def interpolator(self):
        """ int: The cv2 interpolator required to transpose the mask to a full frame. """
        return self._interpolator

    def get_full_frame_mask(self, width, height):
        """ Return the stored mask in a full size frame of the given dimensions

        Parameters
        ----------
        width: int
            The width of the original frame that the mask was extracted from
        height: int
            The height of the original frame that the mask was extracted from

        Returns
        -------
        numpy.ndarray: The mask affined to the original full frame of the given dimensions
        """
        frame = np.zeros((width, height, 1), dtype="uint8")
        mask = cv2.warpAffine(self.mask,
                              self._affine_matrix,
                              (width, height),
                              frame,
                              flags=cv2.WARP_INVERSE_MAP | self._interpolator,
                              borderMode=cv2.BORDER_CONSTANT)
        logger.info("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s",
                     mask.shape, mask.dtype, mask.min(), mask.max())
        return mask

    def add(self, mask, affine_matrix, interpolator):
        """ Add a Faceswap mask to this :class:`Mask`.

        The mask should be the original output from  :mod:`plugins.extract.mask`

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        affine_matrix: numpy.ndarray
            The transformation matrix required to transform the mask to the original frame.
        interpolator, int:
            The CV2 interpolator required to transform this mask to it's original frame
        """
        logger.info("mask shape: %s, mask dtype: %s, mask min: %s, mask max: %s, "
                     "affine_matrix: %s, interpolator: %s)", mask.shape, mask.dtype, mask.min(),
                     affine_matrix, mask.max(), interpolator)
        self._affine_matrix = self._adjust_affine_matrix(mask.shape[0], affine_matrix)
        self._interpolator = interpolator
        self.replace_mask(mask)

    def replace_mask(self, mask):
        """ Replace the existing :attr:`_mask` with the given mask.

        Parameters
        ----------
        mask: numpy.ndarray
            The mask that is to be added as output from :mod:`plugins.extract.mask`.
            It should be in the range 0.0 - 1.0 ideally with a ``dtype`` of ``float32``
        """
        mask = (cv2.resize(mask,
                           (self.stored_size, self.stored_size),
                           interpolation=cv2.INTER_AREA) * 255.0).astype("uint8")
        self._mask = compress(mask)

    def set_blur_and_threshold(self,
                               blur_kernel=0, blur_type="gaussian", blur_passes=1, threshold=0):
        """ Set the internal blur kernel and threshold amount for returned masks

        Parameters
        ----------
        blur_kernel: int, optional
            The kernel size, in pixels to apply gaussian blurring to the mask. Set to 0 for no
            blurring. Should be odd, if an even number is passed in (outside of 0) then it is
            rounded up to the next odd number. Default: 0
        blur_type: ["gaussian", "normalized"], optional
            The blur type to use. ``gaussian`` or ``normalized`` box filter. Default: ``gaussian``
        blur_passes: int, optional
            The number of passed to perform when blurring. Default: 1
        threshold: int, optional
            The threshold amount to minimize/maximize mask values to 0 and 100. Percentage value.
            Default: 0
        """
        logger.info("blur_kernel: %s, threshold: %s", blur_kernel, threshold)
        if blur_type is not None:
            blur_kernel += 0 if blur_kernel == 0 or blur_kernel % 2 == 1 else 1
            self._blur["kernel"] = blur_kernel
            self._blur["type"] = blur_type
            self._blur["passes"] = blur_passes
        self._threshold = (threshold / 100.0) * 255.0

    def _adjust_affine_matrix(self, mask_size, affine_matrix):
        """ Adjust the affine matrix for the mask's storage size

        Parameters
        ----------
        mask_size: int
            The original size of the mask.
        affine_matrix: numpy.ndarray
            The affine matrix to transform the mask at original size to the parent frame.

        Returns
        -------
        affine_matrix: numpy,ndarray
            The affine matrix adjusted for the mask at its stored dimensions.
        """
        zoom = self.stored_size / mask_size
        zoom_mat = np.array([[zoom, 0, 0.], [0, zoom, 0.]])
        adjust_mat = np.dot(zoom_mat, np.concatenate((affine_matrix, np.array([[0., 0., 1.]]))))
        logger.info("storage_size: %s, mask_size: %s, zoom: %s, original matrix: %s, "
                     "adjusted_matrix: %s", self.stored_size, mask_size, zoom, affine_matrix.shape,
                     adjust_mat.shape)
        return adjust_mat

    def to_dict(self):
        """ Convert the mask to a dictionary for saving to an alignments file

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        retval = dict()
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            retval[key] = getattr(self, self._attr_name(key))
        logger.info({k: v if k != "mask" else type(v) for k, v in retval.items()})
        return retval

    def to_png_meta(self):
        """ Convert the mask to a dictionary supported by png itxt headers.

        Returns
        -------
        dict:
            The :class:`Mask` for saving to an alignments file. Contains the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        retval = dict()
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            val = getattr(self, self._attr_name(key))
            if isinstance(val, np.ndarray):
                retval[key] = val.tolist()
            else:
                retval[key] = val
        logger.info({k: v if k != "mask" else type(v) for k, v in retval.items()})
        return retval

    def from_dict(self, mask_dict):
        """ Populates the :class:`Mask` from a dictionary loaded from an alignments file.

        Parameters
        ----------
        mask_dict: dict
            A dictionary stored in an alignments file containing the keys ``mask``,
            ``affine_matrix``, ``interpolator``, ``stored_size``, ``stored_centering``
        """
        for key in ("mask", "affine_matrix", "interpolator", "stored_size", "stored_centering"):
            val = mask_dict.get(key)
            val = "face" if key == "stored_centering" and val is None else val
            if key == "affine_matrix" and not isinstance(val, np.ndarray):
                val = np.array(val, dtype="float64")
            setattr(self, self._attr_name(key), val)
            logger.info("%s - %s", key, val if key != "mask" else type(val))

    @staticmethod
    def _attr_name(dict_key):
        """ The :class:`Mask` attribute name for the given dictionary key

        Parameters
        ----------
        dict_key: str
            The key name from an alignments dictionary

        Returns
        -------
        attribute_name: str
            The attribute name for the given key for :class:`Mask`
        """
        retval = "_{}".format(dict_key) if not dict_key.startswith("stored") else dict_key
        logger.info("dict_key: %s, attribute_name: %s", dict_key, retval)
        return retval


class BlurMask():  # pylint:disable=too-few-public-methods
    """ Factory class to return the correct blur object for requested blur type.

    Works for square images only. Currently supports Gaussian and Normalized Box Filters.

    Parameters
    ----------
    blur_type: ["gaussian", "normalized"]
        The type of blur to use
    mask: :class:`numpy.ndarray`
        The mask to apply the blur to
    kernel: int or float
        Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
    is_ratio: bool, optional
        Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
        actual kernel size will be calculated from the given ratio and the mask size. If
        ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.
        Default: ``False``
    passes: int, optional
        The number of passes to perform when blurring. Default: ``1``

    Example
    -------
    >>> print(mask.shape)
    (128, 128, 1)
    >>> new_mask = BlurMask("gaussian", mask, 3, is_ratio=False, passes=1).blurred
    >>> print(new_mask.shape)
    (128, 128, 1)
    """
    def __init__(self, blur_type, mask, kernel, is_ratio=False, passes=1):
        logger.info("Initializing %s: (blur_type: '%s', mask_shape: %s, kernel: %s, "
                     "is_ratio: %s, passes: %s)", self.__class__.__name__, blur_type, mask.shape,
                     kernel, is_ratio, passes)
        self._blur_type = blur_type.lower()
        self._mask = mask
        self._passes = passes
        kernel_size = self._get_kernel_size(kernel, is_ratio)
        self._kernel_size = self._get_kernel_tuple(kernel_size)
        logger.info("Initialized %s", self.__class__.__name__)

    @property
    def blurred(self):
        """ :class:`numpy.ndarray`: The final mask with blurring applied. """
        func = self._func_mapping[self._blur_type]
        kwargs = self._get_kwargs()
        blurred = self._mask
        for i in range(self._passes):
            ksize = int(kwargs["ksize"][0])
            logger.info("Pass: %s, kernel_size: %s", i + 1, (ksize, ksize))
            blurred = func(blurred, **kwargs)
            ksize = int(round(ksize * self._multipass_factor))
            kwargs["ksize"] = self._get_kernel_tuple(ksize)
        blurred = blurred[..., None]
        logger.info("Returning blurred mask. Shape: %s", blurred.shape)
        return blurred

    @property
    def _multipass_factor(self):
        """ For multiple passes the kernel must be scaled down. This value is
            different for box filter and gaussian """
        factor = dict(gaussian=0.8, normalized=0.5)
        return factor[self._blur_type]

    @property
    def _sigma(self):
        """ int: The Sigma for Gaussian Blur. Returns 0 to force calculation from kernel size. """
        return 0

    @property
    def _func_mapping(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function name. """
        return dict(gaussian=cv2.GaussianBlur,  # pylint: disable = no-member
                    normalized=cv2.blur)  # pylint: disable = no-member

    @property
    def _kwarg_requirements(self):
        """ dict: :attr:`_blur_type` mapped to cv2 Function required keyword arguments. """
        return dict(gaussian=["ksize", "sigmaX"],
                    normalized=["ksize"])

    @property
    def _kwarg_mapping(self):
        """ dict: cv2 function keyword arguments mapped to their parameters. """
        return dict(ksize=self._kernel_size,
                    sigmaX=self._sigma)

    def _get_kernel_size(self, kernel, is_ratio):
        """ Set the kernel size to absolute value.

        If :attr:`is_ratio` is ``True`` then the kernel size is calculated from the given ratio and
        the :attr:`_mask` size, otherwise the given kernel size is just returned.

        Parameters
        ----------
        kernel: int or float
            Either the kernel size (in pixels) or the size of the kernel as a ratio of mask size
        is_ratio: bool, optional
            Whether the given :attr:`kernel` parameter is a ratio or not. If ``True`` then the
            actual kernel size will be calculated from the given ratio and the mask size. If
            ``False`` then the kernel size will be set directly from the :attr:`kernel` parameter.

        Returns
        -------
        int
            The size (in pixels) of the blur kernel
        """
        if not is_ratio:
            return kernel

        mask_diameter = np.sqrt(np.sum(self._mask))
        radius = round(max(1., mask_diameter * kernel / 100.))
        kernel_size = int(radius * 2 + 1)
        logger.info("kernel_size: %s", kernel_size)
        return kernel_size

    @staticmethod
    def _get_kernel_tuple(kernel_size):
        """ Make sure kernel_size is odd and return it as a tuple.

        Parameters
        ----------
        kernel_size: int
            The size in pixels of the blur kernel

        Returns
        -------
        tuple
            The kernel size as a tuple of ('int', 'int')
        """
        kernel_size += 1 if kernel_size % 2 == 0 else 0
        retval = (kernel_size, kernel_size)
        logger.info(retval)
        return retval

    def _get_kwargs(self):
        """ dict: the valid keyword arguments for the requested :attr:`_blur_type` """
        retval = {kword: self._kwarg_mapping[kword]
                  for kword in self._kwarg_requirements[self._blur_type]}
        logger.info("BlurMask kwargs: %s", retval)
        return retval

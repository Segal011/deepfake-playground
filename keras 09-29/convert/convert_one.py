#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import json
import logging
import os
from importlib import import_module

import numpy as np

from convert.converter import Converter
from convert.fsmedia import Alignments, finalize
from convert.align import AlignedFace
from convert.image import ImagesLoader
from convert.utils import FaceswapError, get_backend, get_folder

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Convert():  # pylint:disable=too-few-public-methods
    """ The Faceswap Face Conversion Process.

    The conversion process is responsible for swapping the faces on source frames with the output
    from a trained model.

    It leverages a series of user selected post-processing plugins, executed from
    :class:`lib.convert.Converter`.

    The convert process is self contained and should not be referenced by any other scripts, so it
    contains no public properties.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments to be passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, arguments):
        logger.debug("Initializing %s: (args: %s)", self.__class__.__name__, arguments)
        self._args = arguments

        # self._patch_threads = None
        self._alignments = Alignments(self._args, False, False) #self._images.is_video)

        self._images = ImagesLoader(self._args.input_dir, self._alignments)
        self._predictor = Predict(self._images.loaded_images, arguments)
        self._validate()
        get_folder(self._args.output_dir)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None

        self._converter = Converter(self._predictor.output_size,
                                    self._predictor.coverage_ratio,
                                    self._predictor.centering,
                                    arguments,
                                    configfile=configfile)

        logger.debug("Initialized %s", self.__class__.__name__)


    def _validate(self):
        """ Validate the Command Line Options.

        Ensure that certain cli selections are valid and won't result in an error. Checks:
            * If frames have been passed in with video output, ensure user supplies reference
            video.
            * If "on-the-fly" and a Neural Network mask is selected, warn and switch to 'extended'
            * If a mask-type is selected, ensure it exists in the alignments file.
            * If a predicted mask-type is selected, ensure model has been trained with a mask
            otherwise attempt to select first available masks, otherwise raise error.

        Raises
        ------
        FaceswapError
            If an invalid selection has been found.

        """

        if (self._args.on_the_fly and
                self._args.mask_type not in ("none", "extended", "components")):
            logger.warning("You have selected an incompatible mask type ('%s') for On-The-Fly "
                           "conversion. Switching to 'extended'", self._args.mask_type)
            self._args.mask_type = "extended"

        if (not self._args.on_the_fly and
                self._args.mask_type not in ("none", "predicted") and
                not self._alignments.mask_is_valid(self._args.mask_type)):
            raise FaceswapError()

        if self._args.mask_type == "predicted" and not self._predictor.has_predicted_mask:
            available_masks = [k for k, v in self._alignments.mask_summary.items()
                               if k != "none" and v == self._alignments.faces_count]
            if not available_masks:
                msg = ("Predicted Mask selected, but the model was not trained with a mask and no "
                       "masks are stored in the Alignments File.\nYou should generate the "
                       "required masks with the Mask Tool or set the Mask Type to `none`.")
                raise FaceswapError(msg)
            mask_type = available_masks[0]
            logger.warning("Predicted Mask selected, but the model was not trained with a "
                           "mask. Selecting first available mask: '%s'", mask_type)
            self._args.mask_type = mask_type

    def process(self):
        """ The entry point for triggering the Conversion Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.debug("Starting Conversion")
        try:
            self._convert_images()
            finalize(1,
                     self._predictor.faces_count,
                     self._predictor.verify_output)
            logger.debug("Completed Conversion")
        except MemoryError as err:
            msg = ("Faceswap ran out of RAM running convert. Conversion is very system RAM "
                   "heavy, so this can happen in certain circumstances when you have a lot of "
                   "cpus but not enough RAM to support them all."
                   "\nYou should lower the number of processes in use by either setting the "
                   "'singleprocess' flag (-sp) or lowering the number of parallel jobs (-j).")
            raise FaceswapError(msg) from err

    def _convert_images(self):
        """ Start the multi-threaded patching process, monitor all threads for errors and join on
        completion. """
        logger.debug("Converting images")
        item = self._predictor.predicted_face
        images = self._converter.process(item)
        for name, image in images.items():
            fname = name.replace(self._args.input_dir, self._args.output_dir)

            logger.info("Outputting: (filename: '%s'", name)  # type:ignore
            try:
                with open(fname, "wb") as outfile:
                    outfile.write(image[0])
            except Exception as err:  # pylint: disable=broad-except
                logger.error("Failed to save image '%s'. Original Error: %s", fname, err)
        logger.debug("Converted images")


class Predict():
    """ Obtains the output from the Faceswap model.

    Parameters
    ----------
    in_queue: :class:`queue.Queue`
        The queue that contains images and detected faces for feeding the model
    queue_size: int
        The maximum size of the input queue
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """
    def __init__(self, input_images, arguments):
        logger.debug("Initializing %s: (args: %s)",
                     self.__class__.__name__, arguments)
        self._args = arguments
        # self._serializer = get_serializer("json")
        self._faces_count = 0
        self._verify_output = False

        self._model = self._load_model()
        self._sizes = self._get_io_sizes()
        self._coverage_ratio = self._model.coverage_ratio
        self._centering = self._model.config["centering"]

        # self._thread = self._launch_predictor()

        self.predicted_face = self._predict_face(input_images)
        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def thread(self):
        """ :class:`~lib.multithreading.MultiThread`: The thread that is running the prediction
        function from the Faceswap model. """
        return self._thread

    # @property
    # def in_queue(self):
    #     """ :class:`queue.Queue`: The input queue to the predictor. """
    #     return self._in_queue

    # @property
    # def out_queue(self):
    #     """ :class:`queue.Queue`: The output queue from the predictor. """
    #     return self._out_queue

    @property
    def faces_count(self):
        """ int: The total number of faces seen by the Predictor. """
        return self._faces_count

    @property
    def verify_output(self):
        """ bool: ``True`` if multiple faces have been found in frames, otherwise ``False``. """
        return self._verify_output

    @property
    def coverage_ratio(self):
        """ float: The coverage ratio that the model was trained at. """
        return self._coverage_ratio

    @property
    def centering(self):
        """ str: The centering that the model was trained on (`"face"` or `"legacy"`) """
        return self._centering

    @property
    def has_predicted_mask(self):
        """ bool: ``True`` if the model was trained to learn a mask, otherwise ``False``. """
        return bool(self._model.config.get("learn_mask", False))

    @property
    def output_size(self):
        """ int: The size in pixels of the Faceswap model output. """
        return self._sizes["output"]

    def _get_io_sizes(self):
        """ Obtain the input size and output size of the model.

        Returns
        -------
        dict
            input_size in pixels and output_size in pixels
        """
        input_shape = self._model.model.input_shape
        input_shape = [input_shape] if not isinstance(input_shape, list) else input_shape
        output_shape = self._model.model.output_shape
        output_shape = [output_shape] if not isinstance(output_shape, list) else output_shape
        retval = dict(input=input_shape[0][1], output=output_shape[-1][1])
        logger.debug(retval)
        return retval

    def _load_model(self):
        """ Load the Faceswap model.

        Returns
        -------
        :mod:`plugins.train.model` plugin
            The trained model in the specified model folder
        """
        logger.debug("Loading Model")
        model_dir = get_folder(self._args.model_dir, make_folder=False)
        if not model_dir:
            raise FaceswapError(f"{self._args.model_dir} does not exist.")

        model = self._args.model_class(model_dir, self._args, predict=True)

        model.build()
        logger.debug("Loaded Model")
        return model

    def _get_model_name(self, model_dir):
        """ Return the name of the Faceswap model used.

        If a "trainer" option has been selected in the command line arguments, use that value,
        otherwise retrieve the name of the model from the model's state file.

        Parameters
        ----------
        model_dir: str
            The folder that contains the trained Faceswap model

        Returns
        -------
        str
            The name of the Faceswap model being used.

        """
        if hasattr(self._args, "trainer") and self._args.trainer:
            logger.debug("Trainer name provided: '%s'", self._args.trainer)
            return self._args.trainer

        statefile = [fname for fname in os.listdir(str(model_dir))
                     if fname.endswith("_state.json")]
        if len(statefile) != 1:
            raise FaceswapError("There should be 1 state file in your model folder. {} were "
                                "found. Specify a trainer with the '-t', '--trainer' "
                                "option.".format(len(statefile)))
        statefile = os.path.join(str(model_dir), statefile[0])

        # state = self._serializer.load(statefile)
        try:
            with open(statefile, "rb") as s_file:
                data = s_file.read()
                logger.debug("stored data type: %s", type(data))
                state = json.loads(data.decode("utf-8"))

        except IOError as err:
            msg = f"Error reading from '{statefile}': {err.strerror}"
            raise FaceswapError(msg) from err
        trainer = state.get("name", None)

        if not trainer:
            raise FaceswapError("Trainer name could not be read from state file. "
                                "Specify a trainer with the '-t', '--trainer' option.")
        logger.debug("Trainer from state file: '%s'", trainer)
        return trainer

    # def _launch_predictor(self):
    #     """ Launch the prediction process in a background thread.

    #     Starts the prediction thread and returns the thread.

    #     Returns
    #     -------
    #     :class:`~lib.multithreading.MultiThread`
    #         The started Faceswap model prediction thread.
    #     """
    #     thread = MultiThread(self._predict_faces, thread_count=1)
    #     thread.start()
    #     return thread


    def _predict_face(self, item):
        """ Run Prediction on the Faceswap model in a background thread.

        Reads from the :attr:`self._in_queue`, prepares images for prediction
        then puts the predictions back to the :attr:`self.out_queue`
        """
        faces_seen = 0
        consecutive_no_faces = 0
        batch = list()

        item = item[0]
        logger.info("Got from queue: '%s'", item["filename"])
        faces_count = len(item["detected_faces"])

        # Safety measure. If a large stream of frames appear that do not have faces,
        # these will stack up into RAM. Keep a count of consecutive frames with no faces.
        # If self._batchsize number of frames appear, force the current batch through
        # to clear RAM.
        consecutive_no_faces = consecutive_no_faces + 1 if faces_count == 0 else 0
        self._faces_count += faces_count
        if faces_count > 1:
            self._verify_output = True
            logger.info("Found more than one face in an image! '%s'",
                            os.path.basename(item["filename"]))

        self.load_aligned(item)

        faces_seen += faces_count

        logger.info("Batching to predictor. Frames: %s, Faces: %s",
                        len(batch), faces_seen)

        if faces_seen != 0:
            feed_faces = self._compile_feed_faces(item["feed_faces"])
            predicted = self._predict(feed_faces)
        else:
            predicted = list()

        if len(item["detected_faces"]) == 0:
            item["swapped_faces"] = np.array(list())
        else:
            item["swapped_faces"] = predicted[0:len(item["detected_faces"])]


        logger.debug("Load queue complete")
        return item

    def load_aligned(self, item):
        """ Load the model's feed faces and the reference output faces.

        For each detected face in the incoming item, load the feed face and reference face
        images, correctly sized for input and output respectively.

        Parameters
        ----------
        item: dict
            The incoming image, list of :class:`~lib.align.DetectedFace` objects and list of
            :class:`~lib.align.AlignedFace` objects for the feed face(s) and list of
            :class:`~lib.align.AlignedFace` objects for the reference face(s)

        """
        logger.info("Loading aligned faces: '%s'", item["filename"])
        feed_faces = []
        reference_faces = []
        for detected_face in item["detected_faces"]:
            feed_face = AlignedFace(detected_face.landmarks_xy,
                                    image=item["image"],
                                    centering=self._centering,
                                    size=self._sizes["input"],
                                    coverage_ratio=self._coverage_ratio,
                                    dtype="float32")
            if self._sizes["input"] == self._sizes["output"]:
                reference_faces.append(feed_face)
            else:
                reference_faces.append(AlignedFace(detected_face.landmarks_xy,
                                                   image=item["image"],
                                                   centering=self._centering,
                                                   size=self._sizes["output"],
                                                   coverage_ratio=self._coverage_ratio,
                                                   dtype="float32"))
            feed_faces.append(feed_face)
        item["feed_faces"] = feed_faces
        item["reference_faces"] = reference_faces
        logger.info("Loaded aligned faces: '%s'", item["filename"])

    @staticmethod
    def _compile_feed_faces(feed_faces):
        """ Compile a batch of faces for feeding into the Predictor.

        Parameters
        ----------
        feed_faces: list
            List of :class:`~lib.align.AlignedFace` objects sized for feeding into the model

        Returns
        -------
        :class:`numpy.ndarray`
            A batch of faces ready for feeding into the Faceswap model.
        """
        logger.info("Compiling feed face. Batchsize: %s", len(feed_faces))
        retval = np.stack([feed_face.face[..., :3] for feed_face in feed_faces]) / 255.0
        logger.info("Compiled Feed faces. Shape: %s", retval.shape)
        return retval

    def _predict(self, feed_faces):
        """ Run the Faceswap models' prediction function.

        Parameters
        ----------
        feed_faces: :class:`numpy.ndarray`
            The batch to be fed into the model
        batch_size: int, optional
            Used for plaidml only. Indicates to the model what batch size is being processed.
            Default: ``None``

        Returns
        -------
        :class:`numpy.ndarray`
            The swapped faces for the given batch
        """
        logger.info("Predicting: Batchsize: %s", len(feed_faces))

        if self._model.color_order.lower() == "rgb":
            feed_faces = feed_faces[..., ::-1]

        feed = [feed_faces]
        logger.info("Input shape(s): %s", [item.shape for item in feed])

        predicted = self._model.model.predict(feed, verbose=0)
        predicted = predicted if isinstance(predicted, list) else [predicted]

        if self._model.color_order.lower() == "rgb":
            predicted[0] = predicted[0][..., ::-1]

        logger.info("Output shape(s): %s", [predict.shape for predict in predicted])

        # Only take last output(s)
        if predicted[-1].shape[-1] == 1:  # Merge mask to alpha channel
            predicted = np.concatenate(predicted[-2:], axis=-1).astype("float32")
        else:
            predicted = predicted[-1].astype("float32")

        logger.info("Final shape: %s", predicted.shape)
        return predicted



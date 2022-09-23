#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import logging
import os
import sys

import numpy as np
from tqdm import tqdm

from convert.converter import Converter
from convert.fsmedia import Alignments, finalize
from convert.disk_io import DiskIO
from convert.serializer import get_serializer
from convert.align import AlignedFace, update_legacy_png_header
from convert.gpu_stats import GPUStats
from convert.image import read_image_meta_batch, ImagesLoader
from convert.utils import FaceswapError, get_backend, get_folder, get_image_paths
from helpers.model import import_model

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
        self._images = ImagesLoader(self._args.input_dir, fast_count=True)
        self._alignments = Alignments(self._args, False, False) #self._images.is_video)
        if self._alignments.version == 1.0:
            logger.error("The alignments file format has been updated since the given alignments "
                         "file was generated. You need to update the file to proceed.")
            logger.error("To do this run the 'Alignments Tool' > 'Extract' Job.")
            sys.exit(1)

        self._opts = OptionalActions(self._args, self._images.file_list, self._alignments)
        print("self._images.file_list", self._images)
        # self._add_queues()
        self._disk_io = DiskIO(self._alignments, self._images, arguments)
        self._predictor = Predict(self._disk_io.loaded_images, self._queue_size, arguments)
        self._validate()
        get_folder(self._args.output_dir)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None

        self._converter = Converter(self._predictor.output_size,
                                    self._predictor.coverage_ratio,
                                    self._predictor.centering,
                                    self._disk_io.draw_transparent,
                                    self._disk_io.pre_encode,
                                    arguments,
                                    configfile=configfile)

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def _queue_size(self):
        """ int: Size of the converter queues. 16 for single process otherwise 32 """
        if self._args.singleprocess:
            retval = 16
        else:
            retval = 32
        logger.debug(retval)
        return retval

    # @property
    # def _pool_processes(self):
    #     """ int: The number of threads to run in parallel. Based on user options and number of
    #     available processors. """
    #     if self._args.singleprocess:
    #         retval = 1
    #     elif self._args.jobs > 0:
    #         retval = min(self._args.jobs, total_cpus(), self._images.count)
    #     else:
    #         retval = min(total_cpus(), self._images.count)
    #     retval = 1 if retval == 0 else retval
    #     logger.debug(retval)
    #     return retval

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
        if (self._args.writer == "ffmpeg" and
                not False and #self._images.is_video and
                self._args.reference_video is None):
            raise FaceswapError("Output as video selected, but using frames as input. You must "
                                "provide a reference video ('-ref', '--reference-video').")

        if (self._args.on_the_fly and
                self._args.mask_type not in ("none", "extended", "components")):
            logger.warning("You have selected an incompatible mask type ('%s') for On-The-Fly "
                           "conversion. Switching to 'extended'", self._args.mask_type)
            self._args.mask_type = "extended"

        if (not self._args.on_the_fly and
                self._args.mask_type not in ("none", "predicted") and
                not self._alignments.mask_is_valid(self._args.mask_type)):
            msg = ("You have selected the Mask Type `{}` but at least one face does not have this "
                   "mask stored in the Alignments File.\nYou should generate the required masks "
                   "with the Mask Tool or set the Mask Type option to an existing Mask Type.\nA "
                   "summary of existing masks is as follows:\nTotal faces: {}, Masks: "
                   "{}".format(self._args.mask_type, self._alignments.faces_count,
                               self._alignments.mask_summary))
            raise FaceswapError(msg)

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

    # def _add_queues(self):
    #     """ Add the queues for in, patch and out. """
    #     logger.debug("Adding queues. Queue size: %s", self._queue_size)
    #     for qname in ("convert_in", "convert_out", "patch"):
    #         queue_manager.add_queue(qname, self._queue_size)

    def process(self):
        """ The entry point for triggering the Conversion Process.

        Should only be called from  :class:`lib.cli.launcher.ScriptExecutor`
        """
        logger.debug("Starting Conversion")
        # queue_manager.debug_monitor(5)
        try:
            self._convert_images()
            # self._disk_io.save_thread.join()
            # queue_manager.terminate_queues()

            finalize(self._images.count,
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
        # save_queue = queue_manager.get_queue("convert_out")
        # patch_queue = queue_manager.get_queue("patch")

        item = self._predictor.predicted_face
        images = self._converter.process(item)
        print("images", images)
        # print("rerererer", image)
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
    def __init__(self, input_images, queue_size, arguments):
        logger.debug("Initializing %s: (args: %s, queue_size: %s)",
                     self.__class__.__name__, arguments, queue_size)
        self._args = arguments
        self._serializer = get_serializer("json")
        self._faces_count = 0
        self._verify_output = False

        self._model = self._load_model()
        self._batchsize = self._get_batchsize(queue_size)
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
        trainer = self._get_model_name(model_dir)
        model = import_model("train.model", trainer, False)(model_dir, self._args, predict=True)
        model.build()
        logger.debug("Loaded Model")
        return model

    def _get_batchsize(self, queue_size):
        """ Get the batch size for feeding the model.

        Sets the batch size to 1 if inference is being run on CPU, otherwise the minimum of the
        input queue size and the model's `convert_batchsize` configuration option.

        Parameters
        ----------
        queue_size: int
            The queue size that is feeding the predictor

        Returns
        -------
        int
            The batch size that the model is to be fed at.
        """
        logger.debug("Getting batchsize")
        is_cpu = GPUStats().device_count == 0
        batchsize = 1 if is_cpu else self._model.config["convert_batchsize"]
        batchsize = min(queue_size, batchsize)
        logger.debug("Batchsize: %s", batchsize)
        logger.debug("Got batchsize: %s", batchsize)
        return batchsize

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

        state = self._serializer.load(statefile)
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
        is_amd = get_backend() == "amd"
    
        item = item[0]
        if item != "EOF":
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
            batch.append(item)

        if item != "EOF" and (faces_seen < self._batchsize and
                                consecutive_no_faces < self._batchsize):
            logger.info("Continuing. Current batchsize: %s, consecutive_no_faces: %s",
                            faces_seen, consecutive_no_faces)
            

        if batch:
            logger.info("Batching to predictor. Frames: %s, Faces: %s",
                            len(batch), faces_seen)
            feed_batch = [feed_face for item in batch
                            for feed_face in item["feed_faces"]]
            if faces_seen != 0:
                feed_faces = self._compile_feed_faces(feed_batch)
                batch_size = None
                if is_amd and feed_faces.shape[0] != self._batchsize:
                    logger.info("Fallback to BS=1")
                    batch_size = 1
                predicted = self._predict(feed_faces, batch_size)
            else:
                predicted = list()

            final_batch = self._queue_out_frames(batch, predicted)

            consecutive_no_faces = 0
            faces_seen = 0
            if item == "EOF":
                logger.debug("EOF Received")
                
        # logger.debug("Putting EOF")
        # self.out_queue.put("EOF")
        logger.debug("Load queue complete")
        return final_batch

    # def _predict_faces(self):
    #     """ Run Prediction on the Faceswap model in a background thread.

    #     Reads from the :attr:`self._in_queue`, prepares images for prediction
    #     then puts the predictions back to the :attr:`self.out_queue`
    #     """
    #     faces_seen = 0
    #     consecutive_no_faces = 0
    #     batch = list()
    #     is_amd = get_backend() == "amd"
    #     batch_list = []
    #     while True:
    #         item = self._in_queue.get()
    #         if item != "EOF":
    #             logger.info("Got from queue: '%s'", item["filename"])
    #             faces_count = len(item["detected_faces"])

    #             # Safety measure. If a large stream of frames appear that do not have faces,
    #             # these will stack up into RAM. Keep a count of consecutive frames with no faces.
    #             # If self._batchsize number of frames appear, force the current batch through
    #             # to clear RAM.
    #             consecutive_no_faces = consecutive_no_faces + 1 if faces_count == 0 else 0
    #             self._faces_count += faces_count
    #             if faces_count > 1:
    #                 self._verify_output = True
    #                 logger.info("Found more than one face in an image! '%s'",
    #                                os.path.basename(item["filename"]))

    #             self.load_aligned(item)

    #             faces_seen += faces_count
    #             batch.append(item)

    #         if item != "EOF" and (faces_seen < self._batchsize and
    #                               consecutive_no_faces < self._batchsize):
    #             logger.info("Continuing. Current batchsize: %s, consecutive_no_faces: %s",
    #                          faces_seen, consecutive_no_faces)
    #             continue

    #         if batch:
    #             logger.info("Batching to predictor. Frames: %s, Faces: %s",
    #                          len(batch), faces_seen)
    #             feed_batch = [feed_face for item in batch
    #                           for feed_face in item["feed_faces"]]
    #             if faces_seen != 0:
    #                 feed_faces = self._compile_feed_faces(feed_batch)
    #                 batch_size = None
    #                 if is_amd and feed_faces.shape[0] != self._batchsize:
    #                     logger.info("Fallback to BS=1")
    #                     batch_size = 1
    #                 predicted = self._predict(feed_faces, batch_size)
    #             else:
    #                 predicted = list()

    #             batch_list.append(self._queue_out_frames(batch, predicted))

    #         consecutive_no_faces = 0
    #         faces_seen = 0
    #         batch = list()
    #         if item == "EOF":
    #             logger.debug("EOF Received")
    #             break
    #     logger.debug("Putting EOF")
    #     self._out_queue.put("EOF")
    #     logger.debug("Load queue complete")

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

    def _predict(self, feed_faces, batch_size=None):
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

        predicted = self._model.model.predict(feed, verbose=0,batch_size=batch_size)
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

    def _queue_out_frames(self, batch, swapped_faces):
        """ Compile the batch back to original frames and put to the Out Queue.

        For batching, faces are split away from their frames. This compiles all detected faces
        back to their parent frame before putting each frame to the out queue in batches.

        Parameters
        ----------
        batch: dict
            The batch that was used as the input for the model predict function
        swapped_faces: :class:`numpy.ndarray`
            The predictions returned from the model's predict function
        """
        logger.info("Queueing out batch. Batchsize: %s", len(batch))
        pointer = 0
        for item in batch:
            num_faces = len(item["detected_faces"])
            if num_faces == 0:
                item["swapped_faces"] = np.array(list())
            else:
                item["swapped_faces"] = swapped_faces[pointer:pointer + num_faces]

            logger.info("Putting to queue. ('%s', detected_faces: %s, reference_faces: %s, "
                         "swapped_faces: %s)", item["filename"], len(item["detected_faces"]),
                         len(item["reference_faces"]), item["swapped_faces"].shape[0])
            pointer += num_faces
        # self._out_queue.put(batch)
        logger.info("Queued out batch. Batchsize: %s", len(batch))
        return batch


class OptionalActions():  # pylint:disable=too-few-public-methods
    """ Process specific optional actions for Convert.

    Currently only handles skip faces. This class should probably be (re)moved.

    Parameters
    ----------
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    input_images: list
        List of input image files
    alignments: :class:`lib.align.Alignments`
        The alignments file for this conversion
    """

    def __init__(self, arguments, input_images, alignments):
        logger.debug("Initializing %s", self.__class__.__name__)
        self._args = arguments
        self._input_images = input_images
        self._alignments = alignments

        self._remove_skipped_faces()
        logger.debug("Initialized %s", self.__class__.__name__)

    # SKIP FACES #
    def _remove_skipped_faces(self):
        """ If the user has specified an input aligned directory, remove any non-matching faces
        from the alignments file. """
        logger.debug("Filtering Faces")
        accept_dict = self._get_face_metadata()
        if not accept_dict:
            logger.debug("No aligned face data. Not skipping any faces")
            return
        pre_face_count = self._alignments.faces_count
        self._alignments.filter_faces(accept_dict, filter_out=False)
        logger.info("Faces filtered out: %s", pre_face_count - self._alignments.faces_count)

    def _get_face_metadata(self):
        """ Check for the existence of an aligned directory for identifying which faces in the
        target frames should be swapped. If it exists, scan the folder for face's metadata

        Returns
        -------
        dict
            Dictionary of source frame names with a list of associated face indices to be skipped
        """
        retval = dict()
        input_aligned_dir = self._args.input_aligned_dir

        if input_aligned_dir is None:
            logger.info("Aligned directory not specified. All faces listed in the "
                           "alignments file will be converted")
            return retval
        if not os.path.isdir(input_aligned_dir):
            logger.warning("Aligned directory not found. All faces listed in the "
                           "alignments file will be converted")
            return retval

        log_once = False
        filelist = get_image_paths(input_aligned_dir)
        for fullpath, metadata in tqdm(read_image_meta_batch(filelist),
                                       total=len(filelist),
                                       desc="Reading Face Data",
                                       leave=False):
            if "itxt" not in metadata or "source" not in metadata["itxt"]:
                # UPDATE LEGACY FACES FROM ALIGNMENTS FILE
                if not log_once:
                    logger.warning("Legacy faces discovered in '%s'. These faces will be updated",
                                   input_aligned_dir)
                    log_once = True
                data = update_legacy_png_header(fullpath, self._alignments)
                if not data:
                    raise FaceswapError(
                        "Some of the faces being passed in from '{}' could not be matched to the "
                        "alignments file '{}'\nPlease double check your sources and try "
                        "again.".format(input_aligned_dir, self._alignments.file))
                meta = data["source"]
            else:
                meta = metadata["itxt"]["source"]
            retval.setdefault(meta["source_filename"], list()).append(meta["face_index"])

        if not retval:
            raise FaceswapError("Aligned directory is empty, no faces will be converted!")
        if len(retval) <= len(self._input_images) / 3:
            logger.warning("Aligned directory contains far fewer images than the input "
                           "directory, are you sure this is the right folder?")
        return retval

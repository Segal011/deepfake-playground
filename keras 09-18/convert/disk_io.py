#!/usr/bin python3
""" Main entry point to the convert process of FaceSwap """

import logging
import re
import os
import sys
from threading import Event
from time import sleep

import cv2
import numpy as np
from tqdm import tqdm

from fsmedia import  PostProcess
from lib.align import  DetectedFace
from lib.multithreading import MultiThread
from lib.queue_manager import queue_manager
from lib.utils import FaceswapError
from plugins.extract.pipeline import Extractor, ExtractMedia
from plugins.plugin_loader import PluginLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DiskIO():
    """ Disk Input/Output for the converter process.

    Background threads to:
        * Load images from disk and get the detected faces
        * Save images back to disk

    Parameters
    ----------
    alignments: :class:`lib.alignmnents.Alignments`
        The alignments for the input video
    images: :class:`lib.image.ImagesLoader`
        The input images
    arguments: :class:`argparse.Namespace`
        The arguments that were passed to the convert process as generated from Faceswap's command
        line arguments
    """

    def __init__(self, alignments, images, arguments):
        logger.debug("Initializing %s: (alignments: %s, images: %s, arguments: %s)",
                     self.__class__.__name__, alignments, images, arguments)
        self._alignments = alignments
        self._images = images
        self.loaded_images = []
        self._args = arguments
        self._pre_process = PostProcess(arguments)
        self._completion_event = Event()

        # For frame skipping
        self._imageidxre = re.compile(r"(\d+)(?!.*\d\.)(?=\.\w+$)")
        self._frame_ranges = self._get_frame_ranges()
        self._writer = self._get_writer()

        # Extractor for on the fly detection
        self._extractor = self._load_extractor()

        self._queues = dict(load=None, save=None)
        self._threads = dict(oad=None, save=None)
        # self._init_threads()
        self._load()

        logger.debug("Initialized %s", self.__class__.__name__)

    @property
    def completion_event(self):
        """ :class:`event.Event`: Event is set when the DiskIO Save task is complete """
        return self._completion_event

    @property
    def draw_transparent(self):
        """ bool: ``True`` if the selected writer's Draw_transparent configuration item is set
        otherwise ``False`` """
        return self._writer.config.get("draw_transparent", False)

    @property
    def pre_encode(self):
        """ python function: Selected writer's pre-encode function, if it has one,
        otherwise ``None`` """
        dummy = np.zeros((20, 20, 3), dtype="uint8")
        test = self._writer.pre_encode(dummy)
        retval = None if test is None else self._writer.pre_encode
        logger.debug("Writer pre_encode function: %s", retval)
        return retval

    # @property
    # def save_thread(self):
    #     """ :class:`lib.multithreading.MultiThread`: The thread that is running the image writing
    #     operation. """
    #     return self._threads["save"]

    # @property
    # def load_thread(self):
    #     """ :class:`lib.multithreading.MultiThread`: The thread that is running the image loading
    #     operation. """
    #     return self._threads["load"]

    # @property
    # def load_queue(self):
    #     """ :class:`queue.Queue()`: The queue that images and detected faces are loaded into. """
    #     return self._queues["load"]

    @property
    def _total_count(self):
        """ int: The total number of frames to be converted """
        if self._frame_ranges and not self._args.keep_unchanged:
            retval = sum([fr[1] - fr[0] + 1 for fr in self._frame_ranges])
        else:
            retval = self._images.count
        logger.debug(retval)
        return retval

    # Initialization
    def _get_writer(self):
        """ Load the selected writer plugin.

        Returns
        -------
        :mod:`plugins.convert.writer` plugin
            The requested writer plugin
        """
        args = [self._args.output_dir]
        if self._args.writer in ("ffmpeg", "gif"):
            args.extend([self._total_count, self._frame_ranges])
        if self._args.writer == "ffmpeg":
            if self._images.is_video:
                args.append(self._args.input_dir)
            else:
                args.append(self._args.reference_video)
        logger.debug("Writer args: %s", args)
        configfile = self._args.configfile if hasattr(self._args, "configfile") else None
        return PluginLoader.get_converter("writer", self._args.writer)(*args,
                                                                       configfile=configfile)

    def _get_frame_ranges(self):
        """ Obtain the frame ranges that are to be converted.

        If frame ranges have been specified, then split the command line formatted arguments into
        ranges that can be used.

        Returns
        list or ``None``
            A list of  frames to be processed, or ``None`` if the command line argument was not
            used
        """
        if not self._args.frame_ranges:
            logger.debug("No frame range set")
            return None

        minframe, maxframe = None, None
        if self._images.is_video:
            minframe, maxframe = 1, self._images.count
        else:
            indices = [int(self._imageidxre.findall(os.path.basename(filename))[0])
                       for filename in self._images.file_list]
            if indices:
                minframe, maxframe = min(indices), max(indices)
        logger.debug("minframe: %s, maxframe: %s", minframe, maxframe)

        if minframe is None or maxframe is None:
            raise FaceswapError("Frame Ranges specified, but could not determine frame numbering "
                                "from filenames")

        retval = list()
        for rng in self._args.frame_ranges:
            if "-" not in rng:
                raise FaceswapError("Frame Ranges not specified in the correct format")
            start, end = rng.split("-")
            retval.append((max(int(start), minframe), min(int(end), maxframe)))
        logger.debug("frame ranges: %s", retval)
        return retval

    def _load_extractor(self):
        """ Load the CV2-DNN Face Extractor Chain.

        For On-The-Fly conversion we use a CPU based extractor to avoid stacking the GPU.
        Results are poor.

        Returns
        -------
        :class:`plugins.extract.Pipeline.Extractor`
            The face extraction chain to be used for on-the-fly conversion
        """
        if not self._alignments.have_alignments_file and not self._args.on_the_fly:
            logger.error("No alignments file found. Please provide an alignments file for your "
                         "destination video (recommended) or enable on-the-fly conversion (not "
                         "recommended).")
            sys.exit(1)
        if self._alignments.have_alignments_file:
            if self._args.on_the_fly:
                logger.info("On-The-Fly conversion selected, but an alignments file was found. "
                            "Using pre-existing alignments file: '%s'", self._alignments.file)
            else:
                logger.debug("Alignments file found: '%s'", self._alignments.file)
            return None

        logger.debug("Loading extractor")
        logger.warning("On-The-Fly conversion selected. This will use the inferior cv2-dnn for "
                       "extraction and will produce poor results.")
        logger.warning("It is recommended to generate an alignments file for your destination "
                       "video with Extract first for superior results.")
        extractor = Extractor(detector="cv2-dnn",
                              aligner="cv2-dnn",
                              masker=self._args.mask_type,
                              multiprocess=True,
                              rotate_images=None,
                              min_size=20)
        extractor.launch()
        logger.debug("Loaded extractor")
        return extractor

    def _init_threads(self):
        """ Initialize queues and threads.

        Creates the load and save queues and the load and save threads. Starts the threads.
        """
        logger.debug("Initializing DiskIO Threads")
        for task in ("load", "save"):
            self._add_queue(task)
            self._start_thread(task)
        logger.debug("Initialized DiskIO Threads")

    def _add_queue(self, task):
        """ Add the queue to queue_manager and to :attr:`self._queues` for the given task.

        Parameters
        ----------
        task: {"load", "save"}
            The task that the queue is to be added for
        """
        logger.debug("Adding queue for task: '%s'", task)
        if task == "load":
            q_name = "convert_in"
        elif task == "save":
            q_name = "convert_out"
        else:
            q_name = task
        self._queues[task] = queue_manager.get_queue(q_name)
        logger.debug("Added queue for task: '%s'", task)

    def _start_thread(self, task):
        """ Create the thread for the given task, add it it :attr:`self._threads` and start it.

        Parameters
        ----------
        task: {"load", "save"}
            The task that the thread is to be created for
        """
        logger.debug("Starting thread: '%s'", task)
        args = self._completion_event if task == "save" else None
        func = getattr(self, "_{}".format(task))
        io_thread = MultiThread(func, args, thread_count=1)
        io_thread.start()
        self._threads[task] = io_thread
        logger.debug("Started thread: '%s'", task)

    # Loading tasks
    def _load(self):  # pylint: disable=unused-argument
        """ Load frames from disk.

        In a background thread:
            * Loads frames from disk.
            * Discards or passes through cli selected skipped frames
            * Pairs the frame with its :class:`~lib.align.DetectedFace` objects
            * Performs any pre-processing actions
            * Puts the frame and detected faces to the load queue
        """
        logger.debug("Load Images: Start")
        idx = 0
        for filename, image in self._images.load():
            idx += 1
            # if self._queues["load"].shutdown.is_set():
            #     logger.debug("Load Queue: Stop signal received. Terminating")
            #     break
            if image is None or (not image.any() and image.ndim not in (2, 3)):
                # All black frames will return not numpy.any() so check dims too
                logger.warning("Unable to open image. Skipping: '%s'", filename)
                continue
            if self._check_skipframe(filename):
                if self._args.keep_unchanged:
                    logger.info("Saving unchanged frame: %s", filename)
                    out_file = os.path.join(self._args.output_dir, os.path.basename(filename))
                    # self._queues["save"].put((out_file, image))
                else:
                    logger.info("Discarding frame: '%s'", filename)
                continue

            detected_faces = self._get_detected_faces(filename, image)
            item = dict(filename=filename, image=image, detected_faces=detected_faces)
            self.loaded_images.append(item)
            # self._pre_process.do_actions(item) 
            # self._queues["load"].put(item)

        # logger.debug("Putting EOF")
        # self._queues["load"].put("EOF")
        logger.debug("Load Images: Complete")

    def _check_skipframe(self, filename):
        """ Check whether a frame is to be skipped.

        Parameters
        ----------
        filename: str
            The filename of the frame to check

        Returns
        -------
        bool
            ``True`` if the frame is to be skipped otherwise ``False``
        """
        if not self._frame_ranges:
            return None
        indices = self._imageidxre.findall(filename)
        if not indices:
            logger.warning("Could not determine frame number. Frame will be converted: '%s'",
                           filename)
            return False
        idx = int(indices[0]) if indices else None
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self._frame_ranges))
        logger.info("idx: %s, skipframe: %s", idx, skipframe)
        return skipframe

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
        if not self._extractor:
            detected_faces = self._alignments_faces(os.path.basename(filename), image)
        else:
            detected_faces = self._detect_faces(filename, image)
        logger.info("Got %s faces for: '%s'", len(detected_faces), filename)
        return detected_faces

    def _alignments_faces(self, frame_name, image):
        """ Return detected faces from an alignments file.

        Parameters
        ----------
        frame_name: str
            The name of the frame to return the detected faces for
        image: :class:`numpy.ndarray`
            The frame that the detected faces exist in

        Returns
        -------
        list
            List of :class:`lib.align.DetectedFace` objects
        """
        if not self._check_alignments(frame_name):
            return list()

        faces = self._alignments.get_faces_in_frame(frame_name)
        detected_faces = list()

        for rawface in faces:
            face = DetectedFace()
            face.from_alignment(rawface, image=image)
            detected_faces.append(face)
        return detected_faces

    def _check_alignments(self, frame_name):
        """ Ensure that we have alignments for the current frame.

        If we have no alignments for this image, skip it and output a message.

        Parameters
        ----------
        frame_name: str
            The name of the frame to check that we have alignments for

        Returns
        -------
        bool
            ``True`` if we have alignments for this face, otherwise ``False``
        """
        have_alignments = self._alignments.frame_exists(frame_name)
        if not have_alignments:
            tqdm.write("No alignment found for {}, "
                       "skipping".format(frame_name))
        return have_alignments

    def _detect_faces(self, filename, image):
        """ Extract the face from a frame for On-The-Fly conversion.

        Pulls detected faces out of the Extraction pipeline.

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
        self._extractor.input_queue.put(ExtractMedia(filename, image))
        faces = next(self._extractor.detected_faces())
        return faces.detected_faces

    # Saving tasks
    def _save(self, completion_event):
        """ Save the converted images.

        Puts the selected writer into a background thread and feeds it from the output of the
        patch queue.

        Parameters
        ----------
        completion_event: :class:`event.Event`
            An even that this process triggers when it has finished saving
        """
        logger.debug("Save Images: Start")
        write_preview = self._args.redirect_gui and self._writer.is_stream
        preview_image = os.path.join(self._writer.output_folder, ".gui_preview.jpg")
        logger.debug("Write preview for gui: %s", write_preview)
        for idx in tqdm(range(self._total_count), desc="Converting", file=sys.stdout):
            if self._queues["save"].shutdown.is_set():
                logger.debug("Save Queue: Stop signal received. Terminating")
                break
            item = self._queues["save"].get()
            if item == "EOF":
                logger.debug("EOF Received")
                break
            filename, image = item
            # Write out preview image for the GUI every 10 frames if writing to stream
            if write_preview and idx % 10 == 0 and not os.path.exists(preview_image):
                logger.debug("Writing GUI Preview image: '%s'", preview_image)
                cv2.imwrite(preview_image, image)
            self._writer.write(filename, image)
        self._writer.close()
        completion_event.set()
        logger.debug("Save Faces: Complete")

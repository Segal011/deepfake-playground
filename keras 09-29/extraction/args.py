#!/usr/bin/env python3
""" The Command Line Argument options for faceswap.py """

# pylint:disable=too-many-lines
import argparse
import gettext
import logging
import re
import textwrap
from typing import Any, Dict, List, Optional

from convert.utils import get_backend

from plugins.plugin_loader import PluginLoader

from convert.cli.actions import (DirFullPaths, DirOrFileFullPaths, FileFullPaths, FilesFullPaths, MultiOption,
                      Radio, SaveFileFullPaths, Slider)

from convert.cli.args import ExtractConvertArgs


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
_GPUS = []

# # LOCALES
_LANG = gettext.translation("lib.cli.args", localedir="locales", fallback=True)
_ = _LANG.gettext


class SmartFormatter(argparse.HelpFormatter):
    """ Extends the class :class:`argparse.HelpFormatter` to allow custom formatting in help text.

    Adapted from: https://stackoverflow.com/questions/3853722

    Notes
    -----
    Prefix help text with "R|" to override default formatting and use explicitly defined formatting
    within the help text.
    Prefixing a new line within the help text with "L|" will turn that line into a list item in
    both the cli help text and the GUI.
    """
    def __init__(self,
                 prog: str,
                 indent_increment: int = 2,
                 max_help_position: int = 24,
                 width: Optional[int] = None) -> None:
        super().__init__(prog, indent_increment, max_help_position, width)
        self._whitespace_matcher_limited = re.compile(r'[ \r\f\v]+', re.ASCII)

    def _split_lines(self, text: str, width: int) -> List[str]:
        """ Split the given text by the given display width.

        If the text is not prefixed with "R|" then the standard
        :func:`argparse.HelpFormatter._split_lines` function is used, otherwise raw
        formatting is processed,

        Parameters
        ----------
        text: str
            The help text that is to be formatted for display
        width: int
            The display width, in characters, for the help text

        Returns
        -------
        list
            A list of split strings
        """
        if text.startswith("R|"):
            text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
            output = []
            for txt in text.splitlines():
                indent = ""
                if txt.startswith("L|"):
                    indent = "    "
                    txt = f"  - {txt[2:]}"
                output.extend(textwrap.wrap(txt, width, subsequent_indent=indent))
            return output
        return argparse.HelpFormatter._split_lines(self,  # pylint: disable=protected-access
                                                   text,
                                                   width)


class FaceSwapArgs():
    """ Faceswap argument parser functions that are universal to all commands.

    This is the parent class to all subsequent argparsers which holds global arguments that pertain
    to all commands.

    Process the incoming command line arguments, validates then launches the relevant faceswap
    script with the given arguments.

    Parameters
    ----------
    subparser: :class:`argparse._SubParsersAction`
        The subparser for the given command
    command: str
        The faceswap command that is to be executed
    description: str, optional
        The description for the given command. Default: "default"
    """
    def __init__(self,
                 subparser: argparse._SubParsersAction,
                 command: str,
                 description: str = "default") -> None:
        self.global_arguments = self._get_global_arguments()
        self.info = self.get_info()
        self.argument_list = self.get_argument_list()
        self.optional_arguments = self.get_optional_arguments()
        self._process_suppressions()
        if not subparser:
            return
        self.parser = self._create_parser(subparser, command, description)
        self._add_arguments()


    @staticmethod
    def get_info() -> str:
        """ Returns the information text for the current command.

        This function should be overridden with the actual command help text for each
        commands' parser.

        Returns
        -------
        str
            The information text for this command.
        """
        return ""

    @staticmethod
    def get_argument_list() -> List[Dict[str, Any]]:
        """ Returns the argument list for the current command.

        The argument list should be a list of dictionaries pertaining to each option for a command.
        This function should be overridden with the actual argument list for each command's
        argument list.

        See existing parsers for examples.

        Returns
        -------
        list
            The list of command line options for the given command
        """
        argument_list: List[Dict[str, Any]] = []
        return argument_list

    @staticmethod
    def get_optional_arguments() -> List[Dict[str, Any]]:
        """ Returns the optional argument list for the current command.

        The optional arguments list is not always required, but is used when there are shared
        options between multiple commands (e.g. convert and extract). Only override if required.

        Returns
        -------
        list
            The list of optional command line options for the given command
        """
        argument_list: List[Dict[str, Any]] = []
        return argument_list

    @staticmethod
    def _get_global_arguments() -> List[Dict[str, Any]]:
        """ Returns the global Arguments list that are required for ALL commands in Faceswap.

        This method should NOT be overridden.

        Returns
        -------
        list
            The list of global command line options for all Faceswap commands.
        """
        global_args: List[Dict[str, Any]] = []
        if _GPUS:
            global_args.append(dict(
                opts=("-X", "--exclude-gpus"),
                dest="exclude_gpus",
                action=MultiOption,
                type=str.lower,
                nargs="+",
                choices=[str(idx) for idx in range(len(_GPUS))],
                group=_("Global Options"),
                help=_("R|Exclude GPUs from use by Faceswap. Select the number(s) which "
                       "correspond to any GPU(s) that you do not wish to be made available to "
                       "Faceswap. Selecting all GPUs here will force Faceswap into CPU mode."
                       "\nL|{}").format(" \nL|".join(_GPUS))))
        global_args.append(dict(
            opts=("-C", "--configfile"),
            action=FileFullPaths,
            filetypes="ini",
            type=str,
            group=_("Global Options"),
            help=_("Optionally overide the saved config with the path to a custom config file.")))
        global_args.append(dict(
            opts=("-L", "--loglevel"),
            type=str.upper,
            dest="loglevel",
            default="INFO",
            choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
            group=_("Global Options"),
            help=_("Log level. Stick with INFO or VERBOSE unless you need to file an error "
                   "report. Be careful with TRACE as it will generate a lot of data")))
        global_args.append(dict(
            opts=("-LF", "--logfile"),
            action=SaveFileFullPaths,
            filetypes='log',
            type=str,
            dest="logfile",
            default=None,
            group=_("Global Options"),
            help=_("Path to store the logfile. Leave blank to store in the faceswap folder")))
        # These are hidden arguments to indicate that the GUI/Colab is being used
        global_args.append(dict(
            opts=("-gui", "--gui"),
            action="store_true",
            dest="redirect_gui",
            default=False,
            help=argparse.SUPPRESS))
        global_args.append(dict(
            opts=("-colab", "--colab"),
            action="store_true",
            dest="colab",
            default=False,
            help=argparse.SUPPRESS))
        return global_args

    @staticmethod
    def _create_parser(subparser: argparse._SubParsersAction,
                       command: str,
                       description: str) -> argparse.ArgumentParser:
        """ Create the parser for the selected command.

        Parameters
        ----------
        subparser: :class:`argparse._SubParsersAction`
            The subparser for the given command
        command: str
            The faceswap command that is to be executed
        description: str
            The description for the given command


        Returns
        -------
        :class:`~lib.cli.args.FullHelpArgumentParser`
            The parser for the given command
        """
        parser = subparser.add_parser(command,
                                      help=description,
                                      description=description,
                                      epilog="Questions and feedback: https://faceswap.dev/forum",
                                      formatter_class=SmartFormatter)
        return parser

    def _add_arguments(self) -> None:
        """ Parse the list of dictionaries containing the command line arguments and convert to
        argparse parser arguments. """
        options = self.global_arguments + self.argument_list + self.optional_arguments
        for option in options:
            args = option["opts"]
            kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
            self.parser.add_argument(*args, **kwargs)

    def _process_suppressions(self) -> None:
        """ Certain options are only available for certain backends.

        Suppresses command line options that are not available for the running backend.
        """
        fs_backend = get_backend()
        for opt_list in [self.global_arguments, self.argument_list, self.optional_arguments]:
            for opts in opt_list:
                if opts.get("backend", None) is None:
                    continue
                opt_backend = opts.pop("backend")
                if isinstance(opt_backend, (list, tuple)):
                    opt_backend = [backend.lower() for backend in opt_backend]
                else:
                    opt_backend = [opt_backend.lower()]
                if fs_backend not in opt_backend:
                    opts["help"] = argparse.SUPPRESS

class ExtractArgs(ExtractConvertArgs):
    """ Creates the command line arguments for extraction.

    This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
    for both Extract and Convert should be placed.

    Commands explicit to Extract should be added in :func:`get_optional_arguments`
    """

    @staticmethod
    def get_info() -> str:
        """ The information text for the Extract command.

        Returns
        -------
        str
            The information text for the Extract command.
        """
        return _("Extract faces from image or video sources.\n"
                 "Extraction plugins can be configured in the 'Settings' Menu")

    @staticmethod
    def get_optional_arguments() -> List[Dict[str, Any]]:
        """ Returns the argument list unique to the Extract command.

        Returns
        -------
        list
            The list of optional command line options for the Extract command
        """
        if get_backend() == "cpu":
            default_detector = default_aligner = "cv2-dnn"
        else:
            default_detector = "s3fd"
            default_aligner = "fan"

        argument_list: List[Dict[str, Any]] = []
        argument_list.append(dict(
            opts=("-D", "--detector"),
            action=Radio,
            type=str.lower,
            default=default_detector,
            choices=PluginLoader.get_available_extractors("detect"),
            group=_("Plugins"),
            help=_("R|Detector to use. Some of these have configurable settings in "
                   "'/config/extract.ini' or 'Settings > Configure Extract 'Plugins':"
                   "\nL|cv2-dnn: A CPU only extractor which is the least reliable and least "
                   "resource intensive. Use this if not using a GPU and time is important."
                   "\nL|mtcnn: Good detector. Fast on CPU, faster on GPU. Uses fewer resources "
                   "than other GPU detectors but can often return more false positives."
                   "\nL|s3fd: Best detector. Slow on CPU, faster on GPU. Can detect more faces "
                   "and fewer false positives than other GPU detectors, but is a lot more "
                   "resource intensive.")))
        argument_list.append(dict(
            opts=("-A", "--aligner"),
            action=Radio,
            type=str.lower,
            default=default_aligner,
            choices=PluginLoader.get_available_extractors("align"),
            group=_("Plugins"),
            help=_("R|Aligner to use."
                   "\nL|cv2-dnn: A CPU only landmark detector. Faster, less resource intensive, "
                   "but less accurate. Only use this if not using a GPU and time is important."
                   "\nL|fan: Best aligner. Fast on GPU, slow on CPU.")))
        argument_list.append(dict(
            opts=("-M", "--masker"),
            action=MultiOption,
            type=str.lower,
            nargs="+",
            default='bisenet-fp', #plugins.extract.mask.bisenet_fp.Mask,
            choices=[mask for mask in PluginLoader.get_available_extractors("mask")
                     if mask not in ("components", "extended")],
            group=_("Plugins"),
            help=_("R|Additional Masker(s) to use. The masks generated here will all take up GPU "
                   "RAM. You can select none, one or multiple masks, but the extraction may take "
                   "longer the more you select. NB: The Extended and Components (landmark based) "
                   "masks are automatically generated on extraction."
                   "\nL|bisenet-fp: Relatively lightweight NN based mask that provides more "
                   "refined control over the area to be masked including full head masking "
                   "(configurable in mask settings)."
                   "\nL|custom: A dummy mask that fills the mask area with all 1s or 0s "
                   "(configurable in settings). This is only required if you intend to manually "
                   "edit the custom masks yourself in the manual tool. This mask does not use the "
                   "GPU so will not use any additional VRAM."
                   "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
                   "faces clear of obstructions. Profile faces and obstructions may result in "
                   "sub-par performance."
                   "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
                   "frontal faces. The mask model has been specifically trained to recognize "
                   "some facial obstructions (hands and eyeglasses). Profile faces may result in "
                   "sub-par performance."
                   "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
                   "faces. The mask model has been trained by community members and will need "
                   "testing for further description. Profile faces may result in sub-par "
                   "performance."
                   "\nThe auto generated masks are as follows:"
                   "\nL|components: Mask designed to provide facial segmentation based on the "
                   "positioning of landmark locations. A convex hull is constructed around the "
                   "exterior of the landmarks to create a mask."
                   "\nL|extended: Mask designed to provide facial segmentation based on the "
                   "positioning of landmark locations. A convex hull is constructed around the "
                   "exterior of the landmarks and the mask is extended upwards onto the "
                   "forehead."
                   "\n(eg: `-M unet-dfl vgg-clear`, `--masker vgg-obstructed`)")))
        argument_list.append(dict(
            opts=("-nm", "--normalization"),
            action=Radio,
            type=str.lower,
            dest="normalization",
            default="hist",
            choices=["none", "clahe", "hist", "mean"],
            group=_("Plugins"),
            help=_("R|Performing normalization can help the aligner better align faces with "
                   "difficult lighting conditions at an extraction speed cost. Different methods "
                   "will yield different results on different sets. NB: This does not impact the "
                   "output face, just the input to the aligner."
                   "\nL|none: Don't perform normalization on the face."
                   "\nL|clahe: Perform Contrast Limited Adaptive Histogram Equalization on the "
                   "face."
                   "\nL|hist: Equalize the histograms on the RGB channels."
                   "\nL|mean: Normalize the face colors to the mean.")))
        argument_list.append(dict(
            opts=("-rf", "--re-feed"),
            action=Slider,
            min_max=(0, 10),
            rounding=1,
            type=int,
            dest="re_feed",
            default=8,
            group=_("Plugins"),
            help=_("The number of times to re-feed the detected face into the aligner. Each time "
                   "the face is re-fed into the aligner the bounding box is adjusted by a small "
                   "amount. The final landmarks are then averaged from each iteration. Helps to "
                   "remove 'micro-jitter' but at the cost of slower extraction speed. The more "
                   "times the face is re-fed into the aligner, the less micro-jitter should occur "
                   "but the longer extraction will take.")))
        argument_list.append(dict(
            opts=("-r", "--rotate-images"),
            type=str,
            dest="rotate_images",
            default=None,
            group=_("Plugins"),
            help=_("If a face isn't found, rotate the images to try to find a face. Can find "
                   "more faces at the cost of extraction speed. Pass in a single number to use "
                   "increments of that size up to 360, or pass in a list of numbers to enumerate "
                   "exactly what angles to check.")))
        argument_list.append(dict(
            opts=("-min", "--min-size"),
            action=Slider,
            min_max=(0, 1080),
            rounding=20,
            type=int,
            dest="min_size",
            default=0,
            group=_("Face Processing"),
            help=_("Filters out faces detected below this size. Length, in pixels across the "
                   "diagonal of the bounding box. Set to 0 for off")))
        argument_list.append(dict(
            opts=("-sz", "--size"),
            action=Slider,
            min_max=(256, 1024),
            rounding=64,
            type=int,
            default=512,
            group=_("output"),
            help=_("The output size of extracted faces. Make sure that the model you intend to "
                   "train supports your required size. This will only need to be changed for "
                   "hi-res models.")))
        argument_list.append(dict(
            opts=("-een", "--extract-every-n"),
            action=Slider,
            min_max=(1, 100),
            rounding=1,
            type=int,
            dest="extract_every_n",
            default=1,
            group=_("output"),
            help=_("Extract every 'nth' frame. This option will skip frames when extracting "
                   "faces. For example a value of 1 will extract faces from every frame, a value "
                   "of 10 will extract faces from every 10th frame.")))

        argument_list.append(dict(
            opts=("-dl", "--debug-landmarks"),
            action="store_true",
            dest="debug_landmarks",
            default=False,
            group=_("output"),
            help=_("Draw landmarks on the ouput faces for debugging purposes.")))

        argument_list.append(dict(
            opts=("-s", "--skip-existing"),
            action="store_true",
            dest="skip_existing",
            default=False,
            group=_("settings"),
            help=_("Skips frames that have already been extracted and exist in the alignments "
                   "file")))
        argument_list.append(dict(
            opts=("-sf", "--skip-existing-faces"),
            action="store_true",
            dest="skip_faces",
            default=False,
            group=_("settings"),
            help=_("Skip frames that already have detected faces in the alignments file")))
        argument_list.append(dict(
            opts=("-ssf", "--skip-saving-faces"),
            action="store_true",
            dest="skip_saving_faces",
            default=False,
            group=_("settings"),
            help=_("Skip saving the detected faces to disk. Just create an alignments file")))
        return argument_list

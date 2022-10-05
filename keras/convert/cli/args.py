# #!/usr/bin/env python3
# """ The Command Line Argument options for faceswap.py """
#
# # pylint:disable=too-many-lines
# import argparse
# from email.policy import default
# import gettext
# import logging
# import re
# import sys
# import textwrap
# from typing import Any, Dict, List, NoReturn, Optional
#
# from convert.utils import get_backend
#
# from plugins.plugin_loader import PluginLoader
#
# from .actions import (DirFullPaths, DirOrFileFullPaths, FileFullPaths, FilesFullPaths, MultiOption,
#                       Radio, SaveFileFullPaths, Slider)
# from ..model._base import ModelBase
# from ..model.lightweight import Model
#
# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# _GPUS = []
#
# # # LOCALES
# _LANG = gettext.translation("lib.cli.args", localedir="locales", fallback=True)
# _ = _LANG.gettext
#
#
# class FullHelpArgumentParser(argparse.ArgumentParser):
#     """ Extends :class:`argparse.ArgumentParser` to output full help on bad arguments. """
#     def error(self, message: str) -> NoReturn:
#         self.print_help(sys.stderr)
#         self.exit(2, f"{self.prog}: error: {message}\n")
#
#
# class SmartFormatter(argparse.HelpFormatter):
#     """ Extends the class :class:`argparse.HelpFormatter` to allow custom formatting in help text.
#
#     Adapted from: https://stackoverflow.com/questions/3853722
#
#     Notes
#     -----
#     Prefix help text with "R|" to override default formatting and use explicitly defined formatting
#     within the help text.
#     Prefixing a new line within the help text with "L|" will turn that line into a list item in
#     both the cli help text and the GUI.
#     """
#     def __init__(self,
#                  prog: str,
#                  indent_increment: int = 2,
#                  max_help_position: int = 24,
#                  width: Optional[int] = None) -> None:
#         super().__init__(prog, indent_increment, max_help_position, width)
#         self._whitespace_matcher_limited = re.compile(r'[ \r\f\v]+', re.ASCII)
#
#     def _split_lines(self, text: str, width: int) -> List[str]:
#         """ Split the given text by the given display width.
#
#         If the text is not prefixed with "R|" then the standard
#         :func:`argparse.HelpFormatter._split_lines` function is used, otherwise raw
#         formatting is processed,
#
#         Parameters
#         ----------
#         text: str
#             The help text that is to be formatted for display
#         width: int
#             The display width, in characters, for the help text
#
#         Returns
#         -------
#         list
#             A list of split strings
#         """
#         if text.startswith("R|"):
#             text = self._whitespace_matcher_limited.sub(' ', text).strip()[2:]
#             output = []
#             for txt in text.splitlines():
#                 indent = ""
#                 if txt.startswith("L|"):
#                     indent = "    "
#                     txt = f"  - {txt[2:]}"
#                 output.extend(textwrap.wrap(txt, width, subsequent_indent=indent))
#             return output
#         return argparse.HelpFormatter._split_lines(self,  # pylint: disable=protected-access
#                                                    text,
#                                                    width)
#
#
# class FaceSwapArgs():
#     """ Faceswap argument parser functions that are universal to all commands.
#
#     This is the parent class to all subsequent argparsers which holds global arguments that pertain
#     to all commands.
#
#     Process the incoming command line arguments, validates then launches the relevant faceswap
#     script with the given arguments.
#
#     Parameters
#     ----------
#     subparser: :class:`argparse._SubParsersAction`
#         The subparser for the given command
#     command: str
#         The faceswap command that is to be executed
#     description: str, optional
#         The description for the given command. Default: "default"
#     """
#     def __init__(self,
#                  subparser: argparse._SubParsersAction,
#                  command: str,
#                  description: str = "default") -> None:
#         self.global_arguments = self._get_global_arguments()
#         self.info = self.get_info()
#         self.argument_list = self.get_argument_list()
#         self.optional_arguments = self.get_optional_arguments()
#         self._process_suppressions()
#         if not subparser:
#             return
#         self.parser = self._create_parser(subparser, command, description)
#         self._add_arguments()
#
#
#     @staticmethod
#     def get_info() -> str:
#         """ Returns the information text for the current command.
#
#         This function should be overridden with the actual command help text for each
#         commands' parser.
#
#         Returns
#         -------
#         str
#             The information text for this command.
#         """
#         return ""
#
#     @staticmethod
#     def get_argument_list() -> List[Dict[str, Any]]:
#         """ Returns the argument list for the current command.
#
#         The argument list should be a list of dictionaries pertaining to each option for a command.
#         This function should be overridden with the actual argument list for each command's
#         argument list.
#
#         See existing parsers for examples.
#
#         Returns
#         -------
#         list
#             The list of command line options for the given command
#         """
#         argument_list: List[Dict[str, Any]] = []
#         return argument_list
#
#     @staticmethod
#     def get_optional_arguments() -> List[Dict[str, Any]]:
#         """ Returns the optional argument list for the current command.
#
#         The optional arguments list is not always required, but is used when there are shared
#         options between multiple commands (e.g. convert and extract). Only override if required.
#
#         Returns
#         -------
#         list
#             The list of optional command line options for the given command
#         """
#         argument_list: List[Dict[str, Any]] = []
#         return argument_list
#
#     @staticmethod
#     def _get_global_arguments() -> List[Dict[str, Any]]:
#         """ Returns the global Arguments list that are required for ALL commands in Faceswap.
#
#         This method should NOT be overridden.
#
#         Returns
#         -------
#         list
#             The list of global command line options for all Faceswap commands.
#         """
#         global_args: List[Dict[str, Any]] = []
#         if _GPUS:
#             global_args.append(dict(
#                 opts=("-X", "--exclude-gpus"),
#                 dest="exclude_gpus",
#                 action=MultiOption,
#                 type=str.lower,
#                 nargs="+",
#                 choices=[str(idx) for idx in range(len(_GPUS))],
#                 group=_("Global Options"),
#                 help=_("R|Exclude GPUs from use by Faceswap. Select the number(s) which "
#                        "correspond to any GPU(s) that you do not wish to be made available to "
#                        "Faceswap. Selecting all GPUs here will force Faceswap into CPU mode."
#                        "\nL|{}").format(" \nL|".join(_GPUS))))
#         global_args.append(dict(
#             opts=("-C", "--configfile"),
#             action=FileFullPaths,
#             filetypes="ini",
#             type=str,
#             group=_("Global Options"),
#             help=_("Optionally overide the saved config with the path to a custom config file.")))
#         global_args.append(dict(
#             opts=("-L", "--loglevel"),
#             type=str.upper,
#             dest="loglevel",
#             default="INFO",
#             choices=("INFO", "VERBOSE", "DEBUG", "TRACE"),
#             group=_("Global Options"),
#             help=_("Log level. Stick with INFO or VERBOSE unless you need to file an error "
#                    "report. Be careful with TRACE as it will generate a lot of data")))
#         global_args.append(dict(
#             opts=("-LF", "--logfile"),
#             action=SaveFileFullPaths,
#             filetypes='log',
#             type=str,
#             dest="logfile",
#             default=None,
#             group=_("Global Options"),
#             help=_("Path to store the logfile. Leave blank to store in the faceswap folder")))
#         # These are hidden arguments to indicate that the GUI/Colab is being used
#         global_args.append(dict(
#             opts=("-gui", "--gui"),
#             action="store_true",
#             dest="redirect_gui",
#             default=False,
#             help=argparse.SUPPRESS))
#         global_args.append(dict(
#             opts=("-colab", "--colab"),
#             action="store_true",
#             dest="colab",
#             default=False,
#             help=argparse.SUPPRESS))
#         return global_args
#
#     @staticmethod
#     def _create_parser(subparser: argparse._SubParsersAction,
#                        command: str,
#                        description: str) -> argparse.ArgumentParser:
#         """ Create the parser for the selected command.
#
#         Parameters
#         ----------
#         subparser: :class:`argparse._SubParsersAction`
#             The subparser for the given command
#         command: str
#             The faceswap command that is to be executed
#         description: str
#             The description for the given command
#
#
#         Returns
#         -------
#         :class:`~lib.cli.args.FullHelpArgumentParser`
#             The parser for the given command
#         """
#         parser = subparser.add_parser(command,
#                                       help=description,
#                                       description=description,
#                                       epilog="Questions and feedback: https://faceswap.dev/forum",
#                                       formatter_class=SmartFormatter)
#         return parser
#
#     def _add_arguments(self) -> None:
#         """ Parse the list of dictionaries containing the command line arguments and convert to
#         argparse parser arguments. """
#         options = self.global_arguments + self.argument_list + self.optional_arguments
#         for option in options:
#             args = option["opts"]
#             kwargs = {key: option[key] for key in option.keys() if key not in ("opts", "group")}
#             self.parser.add_argument(*args, **kwargs)
#
#     def _process_suppressions(self) -> None:
#         """ Certain options are only available for certain backends.
#
#         Suppresses command line options that are not available for the running backend.
#         """
#         fs_backend = get_backend()
#         for opt_list in [self.global_arguments, self.argument_list, self.optional_arguments]:
#             for opts in opt_list:
#                 if opts.get("backend", None) is None:
#                     continue
#                 opt_backend = opts.pop("backend")
#                 if isinstance(opt_backend, (list, tuple)):
#                     opt_backend = [backend.lower() for backend in opt_backend]
#                 else:
#                     opt_backend = [opt_backend.lower()]
#                 if fs_backend not in opt_backend:
#                     opts["help"] = argparse.SUPPRESS
#
#
# class ExtractConvertArgs(FaceSwapArgs):
#     """ Parent class to capture arguments that will be used in both extract and convert processes.
#
#     Extract and Convert share a fair amount of arguments, so arguments that can be used in both of
#     these processes should be placed here.
#
#     No further processing is done in this class (this is handled by the children), this just
#     captures the shared arguments.
#     """
#
#     @staticmethod
#     def get_argument_list() -> List[Dict[str, Any]]:
#         """ Returns the argument list for shared Extract and Convert arguments.
#
#         Returns
#         -------
#         list
#             The list of command line options for the given Extract and Convert
#         """
#         argument_list: List[Dict[str, Any]] = []
#         argument_list.append(dict(
#             opts=("-i", "--input-dir"),
#             action=DirOrFileFullPaths,
#             filetypes="video",
#             dest="input_dir",
#             required=True,
#             group=_("Data"),
#             help=_("Input directory or video. Either a directory containing the image files you "
#                    "wish to process or path to a video file. NB: This should be the source video/"
#                    "frames NOT the source faces.")))
#         argument_list.append(dict(
#             opts=("-o", "--output-dir"),
#             action=DirFullPaths,
#             dest="output_dir",
#             required=True,
#             group=_("Data"),
#             help=_("Output directory. This is where the converted files will be saved.")))
#         argument_list.append(dict(
#             opts=("-al", "--alignments"),
#             action=FileFullPaths,
#             filetypes="alignments",
#             type=str,
#             dest="alignments_path",
#             group=_("Data"),
#             help=_("Optional path to an alignments file. Leave blank if the alignments file is "
#                    "at the default location.")))
#         return argument_list
#
# class ConvertArgs(ExtractConvertArgs):
#     """ Creates the command line arguments for conversion.
#
#     This class inherits base options from :class:`ExtractConvertArgs` where arguments that are used
#     for both Extract and Convert should be placed.
#
#     Commands explicit to Convert should be added in :func:`get_optional_arguments`
#     """
#
#     @staticmethod
#     def get_info() -> str:
#         """ The information text for the Convert command.
#
#         Returns
#         -------
#         str
#             The information text for the Convert command.
#         """
#         return _("Swap the original faces in a source video/images to your final faces.\n"
#                  "Conversion plugins can be configured in the 'Settings' Menu")
#
#     @staticmethod
#     def get_optional_arguments() -> List[Dict[str, Any]]:
#         """ Returns the argument list unique to the Convert command.
#
#         Returns
#         -------
#         list
#             The list of optional command line options for the Convert command
#         """
#
#         argument_list: List[Dict[str, Any]] = []
#
#         argument_list.append(dict(
#             opts=("-m", "--model-dir"),
#             action=DirFullPaths,
#             dest="model_dir",
#             required=True,
#             group=_("Data"),
#             help=_("Model directory. The directory containing the trained model you wish to use "
#                    "for conversion.")))
#
#         argument_list.append(dict(
#             opts=("-mc", "--model_class"),
#             type=ModelBase,
#             dest="model_class",
#             default=Model,
#             group=_("Data"),
#             help=_("Model directory. The directory containing the trained model you wish to use "
#                    "for conversion.")))
#         argument_list.append(dict(
#             opts=("-c", "--color-adjustment"),
#             action=Radio,
#             type=str.lower,
#             dest="color_adjustment",
#             default="avg-color",
#             choices=PluginLoader.get_available_convert_plugins("color", True),
#             group=_("Plugins"),
#             help=_("R|Performs color adjustment to the swapped face. Some of these options have "
#                    "configurable settings in '/config/convert.ini' or 'Settings > Configure "
#                    "Convert Plugins':"
#                    "\nL|avg-color: Adjust the mean of each color channel in the swapped "
#                    "reconstruction to equal the mean of the masked area in the original image."
#                    "\nL|color-transfer: Transfers the color distribution from the source to the "
#                    "target image using the mean and standard deviations of the L*a*b* "
#                    "color space."
#                    "\nL|manual-balance: Manually adjust the balance of the image in a variety of "
#                    "color spaces. Best used with the Preview tool to set correct values."
#                    "\nL|match-hist: Adjust the histogram of each color channel in the swapped "
#                    "reconstruction to equal the histogram of the masked area in the original "
#                    "image."
#                    "\nL|seamless-clone: Use cv2's seamless clone function to remove extreme "
#                    "gradients at the mask seam by smoothing colors. Generally does not give "
#                    "very satisfactory results."
#                    "\nL|none: Don't perform color adjustment.")))
#         argument_list.append(dict(
#             opts=("-M", "--mask-type"),
#             action=Radio,
#             type=str.lower,
#             dest="mask_type",
#             default="extended",
#             choices=PluginLoader.get_available_extractors("mask",
#                                                           add_none=True,
#                                                           extend_plugin=True) + ["predicted"],
#             group=_("Plugins"),
#             help=_("R|Masker to use. NB: The mask you require must exist within the alignments "
#                    "file. You can add additional masks with the Mask Tool."
#                    "\nL|none: Don't use a mask."
#                    "\nL|bisenet-fp_face: Relatively lightweight NN based mask that provides more "
#                    "refined control over the area to be masked (configurable in mask settings). "
#                    "Use this version of bisenet-fp if your model is trained with 'face' or "
#                    "'legacy' centering."
#                    "\nL|bisenet-fp_head: Relatively lightweight NN based mask that provides more "
#                    "refined control over the area to be masked (configurable in mask settings). "
#                    "Use this version of bisenet-fp if your model is trained with 'head' centering."
#                    "\nL|custom_face: Custom user created, face centered mask."
#                    "\nL|custom_head: Custom user created, head centered mask."
#                    "\nL|components: Mask designed to provide facial segmentation based on the "
#                    "positioning of landmark locations. A convex hull is constructed around the "
#                    "exterior of the landmarks to create a mask."
#                    "\nL|extended: Mask designed to provide facial segmentation based on the "
#                    "positioning of landmark locations. A convex hull is constructed around the "
#                    "exterior of the landmarks and the mask is extended upwards onto the forehead."
#                    "\nL|vgg-clear: Mask designed to provide smart segmentation of mostly frontal "
#                    "faces clear of obstructions. Profile faces and obstructions may result in "
#                    "sub-par performance."
#                    "\nL|vgg-obstructed: Mask designed to provide smart segmentation of mostly "
#                    "frontal faces. The mask model has been specifically trained to recognize "
#                    "some facial obstructions (hands and eyeglasses). Profile faces may result in "
#                    "sub-par performance."
#                    "\nL|unet-dfl: Mask designed to provide smart segmentation of mostly frontal "
#                    "faces. The mask model has been trained by community members and will need "
#                    "testing for further description. Profile faces may result in sub-par "
#                    "performance."
#                    "\nL|predicted: If the 'Learn Mask' option was enabled during training, this "
#                    "will use the mask that was created by the trained model.")))
#         argument_list.append(dict(
#             opts=("-w", "--writer"),
#             action=Radio,
#             type=str,
#             default="opencv",
#             choices=PluginLoader.get_available_convert_plugins("writer", False),
#             group=_("Plugins"),
#             help=_("R|The plugin to use to output the converted images. The writers are "
#                    "configurable in '/config/convert.ini' or 'Settings > Configure Convert "
#                    "Plugins:'"
#                    "\nL|ffmpeg: [video] Writes out the convert straight to video. When the input "
#                    "is a series of images then the '-ref' (--reference-video) parameter must be "
#                    "set."
#                    "\nL|gif: [animated image] Create an animated gif."
#                    "\nL|opencv: [images] The fastest image writer, but less options and formats "
#                    "than other plugins."
#                    "\nL|pillow: [images] Slower than opencv, but has more options and supports "
#                    "more formats.")))
#         argument_list.append(dict(
#             opts=("-osc", "--output-scale"),
#             action=Slider,
#             min_max=(25, 400),
#             rounding=1,
#             type=int,
#             dest="output_scale",
#             default=100,
#             group=_("Frame Processing"),
#             help=_("Scale the final output frames by this amount. 100%% will output the frames "
#                    "at source dimensions. 50%% at half size 200%% at double size")))
#         argument_list.append(dict(
#             opts=("-fr", "--frame-ranges"),
#             type=str,
#             nargs="+",
#             group=_("Frame Processing"),
#             help=_("Frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use "
#                    "--frame-ranges 10-50 90-100. Frames falling outside of the selected range "
#                    "will be discarded unless '-k' (--keep-unchanged) is selected. NB: If you are "
#                    "converting from images, then the filenames must end with the frame-number!")))
#         argument_list.append(dict(
#             opts=("-a", "--input-aligned-dir"),
#             action=DirFullPaths,
#             dest="input_aligned_dir",
#             default=None,
#             group=_("Face Processing"),
#             help=_("If you have not cleansed your alignments file, then you can filter out faces "
#                    "by defining a folder here that contains the faces extracted from your input "
#                    "files/video. If this folder is defined, then only faces that exist within "
#                    "your alignments file and also exist within the specified folder will be "
#                    "converted. Leaving this blank will convert all faces that exist within the "
#                    "alignments file.")))
#         argument_list.append(dict(
#             opts=("-n", "--nfilter"),
#             action=FilesFullPaths,
#             filetypes="image",
#             dest="nfilter",
#             default=None,
#             nargs="+",
#             group=_("Face Processing"),
#             help=_("Optionally filter out people who you do not wish to process by passing in an "
#                    "image of that person. Should be a front portrait with a single person in the "
#                    "image. Multiple images can be added space separated. NB: Using face filter "
#                    "will significantly decrease extraction speed and its accuracy cannot be "
#                    "guaranteed.")))
#         argument_list.append(dict(
#             opts=("-f", "--filter"),
#             action=FilesFullPaths,
#             filetypes="image",
#             dest="filter",
#             default=None,
#             nargs="+",
#             group=_("Face Processing"),
#             help=_("Optionally select people you wish to process by passing in an image of that "
#                    "person. Should be a front portrait with a single person in the image. "
#                    "Multiple images can be added space separated. NB: Using face filter will "
#                    "significantly decrease extraction speed and its accuracy cannot be "
#                    "guaranteed.")))
#         argument_list.append(dict(
#             opts=("-l", "--ref_threshold"),
#             action=Slider,
#             min_max=(0.01, 0.99),
#             rounding=2,
#             type=float,
#             dest="ref_threshold",
#             default=0.4,
#             group=_("Face Processing"),
#             help=_("For use with the optional nfilter/filter files. Threshold for positive face "
#                    "recognition. Lower values are stricter. NB: Using face filter will "
#                    "significantly decrease extraction speed and its accuracy cannot be "
#                    "guaranteed.")))
#         argument_list.append(dict(
#             opts=("-j", "--jobs"),
#             action=Slider,
#             min_max=(0, 40),
#             rounding=1,
#             type=int,
#             dest="jobs",
#             default=0,
#             group=_("settings"),
#             help=_("The maximum number of parallel processes for performing conversion. "
#                    "Converting images is system RAM heavy so it is possible to run out of memory "
#                    "if you have a lot of processes and not enough RAM to accommodate them all. "
#                    "Setting this to 0 will use the maximum available. No matter what you set "
#                    "this to, it will never attempt to use more processes than are available on "
#                    "your system. If singleprocess is enabled this setting will be ignored.")))
#         argument_list.append(dict(
#             opts=("-t", "--trainer"),
#             type=str.lower,
#             choices=PluginLoader.get_available_models(),
#             group=_("settings"),
#             help=_("[LEGACY] This only needs to be selected if a legacy model is being loaded or "
#                    "if there are multiple models in the model folder")))
#         argument_list.append(dict(
#             opts=("-otf", "--on-the-fly"),
#             action="store_true",
#             dest="on_the_fly",
#             default=True,
#             group=_("settings"),
#             help=_("Enable On-The-Fly Conversion. NOT recommended. You should generate a clean "
#                    "alignments file for your destination video. However, if you wish you can "
#                    "generate the alignments on-the-fly by enabling this option. This will use "
#                    "an inferior extraction pipeline and will lead to substandard results. If an "
#                    "alignments file is found, this option will be ignored.")))
#         argument_list.append(dict(
#             opts=("-k", "--keep-unchanged"),
#             action="store_true",
#             dest="keep_unchanged",
#             default=False,
#             group=_("Frame Processing"),
#             help=_("When used with --frame-ranges outputs the unchanged frames that are not "
#                    "processed instead of discarding them.")))
#         argument_list.append(dict(
#             opts=("-s", "--swap-model"),
#             action="store_true",
#             dest="swap_model",
#             default=False,
#             group=_("settings"),
#             help=_("Swap the model. Instead converting from of A -> B, converts B -> A")))
#         argument_list.append(dict(
#             opts=("-sp", "--singleprocess"),
#             action="store_true",
#             default=False,
#             group=_("settings"),
#             help=_("Disable multiprocessing. Slower but less resource intensive.")))
#         return argument_list

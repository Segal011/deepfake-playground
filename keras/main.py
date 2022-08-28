from train import Train

from lib.cli import args as cli_args
from lib.config import generate_configs
from lib.utils import get_backend
import sys
_PARSER = cli_args.FullHelpArgumentParser()

def _bad_args(*args) -> None:  # pylint:disable=unused-argument
    """ Print help to console when bad arguments are provided. """
    print(cli_args)
    _PARSER.print_help()
    sys.exit(0)

if __name__ == '__main__':


    subparser = _PARSER.add_subparsers()
    # cli_args.ExtractArgs(subparser, "extract", _("Extract the faces from pictures or a video"))
    arguments= cli_args.TrainArgs(subparser, "train", ("Train a model for the two faces A and B"))
    # cli_args.ConvertArgs(subparser,
    #                      "convert",
    #                      _("Convert source pictures or video to a new one with the face swapped"))
    # cli_args.GuiArgs(subparser, "gui", _("Launch the Faceswap Graphical User Interface"))
    # _PARSER.set_defaults(func=_bad_args)
    # arguments = _PARSER.parse_args()
    # arguments.func(arguments)
    arguments = arguments.parser.parse_args()
    # print(arguments)


    Train(arguments).train()
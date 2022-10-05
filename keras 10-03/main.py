from argparse import ArgumentParser
from train import Train
from convert.convert_one import Convert
# from lib.cli import args as cli_args

from convert.cli import args as cli_args
# from lib.config import generate_configs
# from lib.utils import get_backend
import sys

_PARSER = cli_args.FullHelpArgumentParser()


def _bad_args(*args) -> None:  # pylint:disable=unused-argument
    """ Print help to console when bad arguments are provided. """
    print(cli_args)
    _PARSER.print_help()
    sys.exit(0)


if __name__ == '__main__':
    # # Training
    #     subparser = _PARSER.add_subparsers()
    #     arguments = cli_args.TrainArgs(subparser, "train", ("Train a model for the two faces A and B"))
    #     arguments = arguments.parser.parse_args()
    #     Train(arguments).train()

    # Converting
    subparser = _PARSER.add_subparsers()
    arguments = cli_args.ConvertArgs(subparser,
                                     "convert",
                                     ("Convert source pictures or video to a new one with the face swapped"))
    arguments = arguments.parser.parse_args()
    Convert(arguments).process()

    # import json
#
#     a_file = open("data.json", "w")
#
#     json.dumps(arguments.__dict__, a_file)
#
#     a_file.close()
#     a_file = open("data.json", "r")
#
#     output = a_file.read()
#     print(output)
#     # Convert(arguments).process()
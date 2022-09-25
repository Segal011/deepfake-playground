from importlib import import_module
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def import_model(attr, name, disable_logging):
    """ Import the plugin's module

    Parameters
    ----------
    name: str
        The name of the requested converter plugin
    disable_logging: bool
        Whether to disable the INFO log message that the plugin is being imported.

    Returns
    -------
    :class:`plugin` object:
        A plugin
    """

    name = name.replace("-", "_")
    ttl = attr.split(".")[-1].title()
    if not disable_logging:
        logger.info("Loading %s from %s plugin...", ttl, name.title())
    attr = "model" if attr == "Trainer" else attr.lower()
    mod = ".".join(("plugins", attr, name))
    module = import_module(mod)
    return getattr(module, ttl)
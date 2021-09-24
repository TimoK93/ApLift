import yaml
import os
from attrdict import AttrDict
import argparse


def load_config(file):
    """
    Loads an attribute dict from yaml.

    You can access them via:
        value1 = the_dict.key1
        value2 = the_dict[key2]

    """
    assert os.path.exists(file), "Config file %s not existing!" % file
    os.environ["CONFIG_FILE"] = file
    with open(file, "r") as f:
        config = yaml.load(f)
    attr_dict = AttrDict(config)
    return attr_dict


def main_function(func):
    """ This is a function decorator which loads configs via argparse and passes them to the wrapped function """
    parser = argparse.ArgumentParser(description='Wrapper for main function to load configs')
    parser.add_argument('config', type=str, help='config file')
    args = parser.parse_args()

    # ## Load config ## #
    print("Load config %s\n" % args.config)
    assert os.path.exists(args.config)
    config = load_config(args.config)
    print("=== configs ===")

    def print_dict(cfg, indent=0):
        for key, value in cfg.items():
            if hasattr(value, "items"):
                print("%s%s:" % ("".ljust(indent * 4, " "), key))
                print_dict(value, indent+1)
            else:
                print("%s%s: %s" % ("".ljust(indent*4, " "), key, value))

    print_dict(config, 0)
    print("===============\n")

    # ## Run function # ##
    def func_wrapper():
        # Wrap config
        cfg = {key: eval("config.%s" % key, {"config": config}) for key in config.keys()}
        result = func(**cfg)

        if type(result) == dict:
            print("\n=== metrics ===")
            print_dict(result, 0)
            print("===============\n")

        return result

    return func_wrapper


''' Testing and development '''
if __name__ == "__main__":
    config_file = "./config/default.yaml"
    config = load_config(config_file)


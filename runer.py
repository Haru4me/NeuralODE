import os
import json


if __name__ == '__main__':

    with open('exp_config.json') as f:
        config = json.load(f)

    for name in config.keys():
        command = f"python3 src/experiment.py -n {name} "
        command += " ".join([f"-{key} {config[name][key]}" for key in config[name].keys()])
        os.system(command)

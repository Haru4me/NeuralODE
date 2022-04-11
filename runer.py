import os
import json
import argparse

parser = argparse.ArgumentParser(prog='Runner',
                                 description="""Скрипт запускает список экспериментов из JSON-файла""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-exps' ,'--experiments', nargs='+', type=int, help='Номера экспериментов')
opt = parser.parse_args()

if __name__ == '__main__':

    with open('exp_config.json') as f:
        config = json.load(f)

    if opt.exps:
        for name in list(config.keys())[opt.exps]:
            command = f"python3 src/experiment.py -n {name} "
            command += " ".join([f"-{key} {config[name][key]}" for key in config[name].keys()])
            os.system(command)
    else:
        for name in config.keys():
            command = f"python3 src/experiment.py -n {name} "
            command += " ".join([f"-{key} {config[name][key]}" for key in config[name].keys()])
            os.system(command)

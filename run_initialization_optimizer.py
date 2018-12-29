from learning.InitializationOptimizer import InitializationOptimizer
import sys
from Configuration import Configuration
from input.SmallNorb import SmallNorb
import json


def run(argv):
    config = Configuration.load(argv[1])
    #@TODO iteration count should be added to config, but currently run.py also doesn't get it from config
    # so both should be changed at the same time
    iteration_count = 3

    result_name = config.parse_arguments()
    sn = SmallNorb.from_cache()

    constants = InitializationOptimizer.optimize_standard_deviations(config.config["architecture"], sn, iteration_count, config["batch_size"])

    with open(argv[2], 'w') as f:
        json.dump(constants, f)


if __name__ == '__main__':

    run(sys.argv)


import sys

import tomli

from . import Pipeline

if __name__ == '__main__':
    Pipeline(config=tomli.load(open(sys.argv[1], 'rb')))()

import json
import argparse
import sys

class Configuration:

    Singleton = None

    @staticmethod
    def load(filePath='./defaults.json'):
        if Configuration.Singleton is None:
            Configuration.Singleton = Configuration(json.load(open(filePath, 'r')), filePath)

        return Configuration.Singleton

    def __init__(self, config, filePath=None):
        self.filePath = filePath

        self.config = config
        self.pathSeparator = '/'

        self.changesToDefault = {}

        self.config_name = 'default'
        self.config_name_arg_separator = '#'

    def parseArguments(self, args=None):
        args = args if args is not None else sys.argv[2:]

        if len(args) > 1:
            self.config_name = self.config_name_arg_separator.join([str(value) for value in args])

        pathToType = self.getPathToTypeDictionary()

        parser = argparse.ArgumentParser(add_help=False)

        # parser.add_argument('--help', '-h', action='store_true')

        for path in pathToType.keys():
            parser.add_argument('--' + path, type=pathToType[path])

        parsed = parser.parse_args(args)

        # must_help = parsed.help
        #
        # if must_help:
        #     print(open(self.filePath, 'r').read())
        #     exit()
        # else:
        #     del parsed.help

        self.apply_arguments(vars(parsed))

        return self.config_name

    def save(self, target_file_path=None):
        target_file_path = target_file_path if target_file_path is not None else self.filePath

        json.dump(self.config, open(target_file_path, 'w'))

    def getPathToTypeDictionary(self, config=None, pathOffset='', pathToTypeDictionary=None):
        config = config if config is not None else self.config
        pathToTypeDictionary = pathToTypeDictionary if pathToTypeDictionary is not None else {}

        if not isinstance(config, dict):
            pathToTypeDictionary[pathOffset] = type(config)
            return

        for key in config.keys():
            if len(pathOffset) > 0:
                pathOffset = pathOffset + self.pathSeparator

            self.getPathToTypeDictionary(config[key], pathOffset + key, pathToTypeDictionary)

        return pathToTypeDictionary

    def setPath(self, path, value):

        cursor = self.config
        changesCursor = self.changesToDefault

        pathNodes = path.split(self.pathSeparator)

        for i in range(len(pathNodes) - 1):
            nextNode = pathNodes[i]

            if nextNode not in changesCursor:
                changesCursor[nextNode] = {}

            cursor = cursor[nextNode]
            changesCursor = changesCursor[nextNode]

        cursor[pathNodes[-1]] = value
        changesCursor[pathNodes[-1]] = value

    def apply_arguments(self, parsed):

        for path in parsed.keys():
            if parsed[path] is not None:
                self.setPath(path, parsed[path])

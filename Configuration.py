import json
import argparse
import sys

class Configuration:

    Singleton = None

    @staticmethod
    def get():
        if Configuration.Singleton is None:
            Configuration.Singleton = Configuration()

        return Configuration.Singleton

    def __init__(self, filePath='./defaults.json'):
        self.filePath = filePath
        self.config = json.load(open(self.filePath, 'r'))
        self.pathSeparator = '/'

        self.changesToDefault = {}

        self.config_name = 'default'
        self.config_name_arg_separator = '#'

    def parseArguments(self, args=None):
        args = args if args is not None else sys.argv

        if len(args) > 1:
            self.config_name = args[1:].join(self.config_name_arg_separator)

        pathToType = self.getPathToTypeDictionary()

        parser = argparse.ArgumentParser(add_help=False)

        parser.add_argument('--save', nargs=1, default='defaults.json')
        parser.add_argument('--help', '-h', action='store_true')

        for path in pathToType.keys():
            parser.add_argument('--' + path, pathToType[path])

        parsed = parser.parse_args(args)

        target_file_path = parsed.save[0]
        must_help = parsed.help

        if target_file_path:
            del parsed.save

        if must_help:
            print(open(self.filePath, 'r').read())
            exit()

        self.apply_arguments(parsed)

        if target_file_path:
            self.save(target_file_path)
            print("saved updated config")
            exit()

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
            self.getPathToTypeDictionary(config, pathOffset + self.pathSeparator + key, pathToTypeDictionary)

        return pathToTypeDictionary

    def setPath(self, path, value):

        cursor = self.config
        changesCursor = self.changesToDefault

        pathNodes = path.split(self.pathSeparator)

        for i in range(len(pathNodes) - 2):
            nextNode = pathNodes[i]

            if nextNode not in changesCursor:
                changesCursor[nextNode] = {}

            cursor = cursor[nextNode]
            changesCursor = changesCursor[nextNode]

        cursor[pathNodes[-2]] = pathNodes[-1]
        changesCursor[pathNodes[-2]] = pathNodes[-1]



    def apply_arguments(self, parsed):

        for path in parsed.keys():
            self.setPath(path, parsed[path])

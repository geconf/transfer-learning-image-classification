import tensorflow as tf
import importlib
import sys
import __init__ as booger
from common.logger import Logger

class Trainer():
    def __init__(self, ARCH, DATA, data_dir, log_dir):
        self.ARCH = ARCH
        self.DATA = DATA
        self.data_dir = data_dir
        self.log_dir = log_dir

        self.tb_logger = Logger(self.log_dir + "/tb")

        # dynamically import the parser specified in the DATA config file
        parserModuleName = "parserModule"
        parserSpec = importlib.util.spec_from_file_location(parserModuleName,
                                       booger.TRAIN_PATH + '/tasks/classification/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        parserModule = importlib.util.module_from_spec(parserSpec)
        sys.modules[parserModuleName] = parserModule
        parserSpec.loader.exec_module(parserModule)

        self.parser = parserModule.Parser(root=self.data_dir,
                                          labels=self.DATA["labels"],
                                          batch_size=self.ARCH["train"]["batch_size"])
        

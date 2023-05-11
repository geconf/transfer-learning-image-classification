import tensorflow as tf
from common.logger import Logger

class Trainer():
    def __init__(self, ARCH, DATA, data_dir, log_dir):
        self.ARCH = ARCH
        self.DATA = DATA
        self.data_dir = data_dir
        self.log_dir = log_dir

        self.tb_logger = Logger(self.log_dir + "/tb")
        

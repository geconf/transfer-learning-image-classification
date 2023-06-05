import tensorflow as tf

class Parser():
    def __init__(self,
                 root,              # directory for data
                 labels,            # labels in the data
                 batch_size):       # batch size for train and val
        super(Parser, self).__init__()

        self.root = root
        self.labels = labels
        self.batch_size = batch_size

        print("Hi Python this runs fine")
    
import tensorflow as tf
import tensorflow_hub as hub

class Classifier(tf.module):
    def __init__(self, ARCH, nclasses, path=None, path_append="", name=None):
        super().__init__(name=name)

        self.ARCH = ARCH

        self.feature_extractor = tf.keras.Sequential(
            [hub.KerasLayer(ARCH["model"]["feature_extractor"]["model_path"],
            trainable=False, 
            arguments=dict(batch_norm_momentum=ARCH["model"]["feature_extractor"]["batch_norm_momentum"]))])
        self.dense_1 = Dense(in_features=ARCH["model"]["feature_vector_size"], 
                             out_features=ARCH["model"]["hidden_layer_sizes"]["hidden_1"])
        
        self.dense_2 = Dense(in_features=ARCH["model"]["hidden_layer_sizes"]["hidden_1"], 
                             out_features=ARCH["model"]["hidden_layer_sizes"]["hidden_2"])
        
        self.dense_3 = Dense(in_features=ARCH["model"]["hidden_layer_sizes"]["hidden_2"], 
                             out_features=ARCH["model"]["hidden_layer_sizes"]["hidden_3"])
        
        self.head = Dense(in_features=ARCH["model"]["hidden_layer_sizes"]["hidden_3"], 
                          out_features=nclasses, 
                          activation="softmax")

    # Add @tf.function: decorator
    def __call__(self, x):
        encoding = self.feature_extractor(x)
        hidden_1 = self.dense_1(encoding)
        hidden_2 = self.dense_2(hidden_1)
        hidden_3 = self.dense_3(hidden_2)
        head = self.head(hidden_3)
        return head
    
class Dense(tf.module):
    def __init__(self, in_features, out_features, activation="relu", name=None):
        super().__init__(self, name=None)
        self.w = tf.Variable(
            tf.random.normal([in_features, out_features]), name = 'w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')
        self.activation = activation

    # Add @tf.function: decorator
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        if (self.activation == "softmax"):
            return tf.nn.softmax(y)
        elif (self.activation == "relu"):
            return tf.nn.relu(y)
        else:
            raise Exception("Invalid activation function for Dense layer, please use relu or softmax")
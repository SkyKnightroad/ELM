""" AbstractNetwork Class

This class defines an interface for trainable network for use in a
learning system. It defines the basic interactions that need to be
supported in order to provide learning and recall capabilities. 

"""

class AbstractNetwork(object):
    """ An instance of an ELM-based Neural Network """

    def __init__(self, InSize, HidSize, OutSize):
        """ Creates a new ELM network with the specified number of input,
            hidden and output neurons """
        self.InSize = int(InSize) # size of input layer
        self.HidSize = int(HidSize) # size of hidden layer
        self.OutSize = int(OutSize) # size of output layer

    def train(self, train_item, train_label):
        """ Trains the network with the given input and label """
        raise NotImplementedError()

    def recall(self, test_item):
        """ Returns the network output for the given inputs """
        raise NotImplementedError

    def save(self, filename):
        """ Saves the network state to the specified file. """
        raise NotImplementedError()

    def load(self, filename):
        """ Restores an ELM network from a file saved using the Save function
        """
        raise NotImplementedError()

    @property
    def NumberOfInSize(self):
	raise NotImplementedError()

    @property
    def NumberOfHidSize(self):
	raise NotImplementedError()

    @property
    def NumberOfOutSize(self):
        raise NotImplementedError()






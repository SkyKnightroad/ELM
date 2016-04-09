from os.path import exists
from numpy import zeros, bitwise_and, any, bitwise_or, arange, reshape
from random import randint
from MNISTDataset import MNISTDataset
import pickle


class MNISTCanvasImage(object):
    """ Generates image canvases with NIST digits randomly placed in a canvas"""
    _dataset = None
    _canvasSize = [800,600]
    _canvas = None
    _label = None

    def __init__(self, dataPath, canvasSize=[]):
        """ Creates a new NIST Canvas object """        
        if canvasSize:
            self._canvasSize = canvasSize
        self._dataset = MNISTDataset(dataPath)
        
        # Create a canvas and label canvas of the requested size
        self._canvas = zeros(self._canvasSize)
        self._label = zeros(self._canvasSize)
        self._labelvector = []

    def generateImage(self, numDigits):
        """ Generates a canvas and places digits at random locations """       
        for i in range(0, numDigits):
            placed = False
            counter = 0
            while not placed and counter < 100:            
                # Grab a random training image from the dataset
                idx = randint(0, self._dataset.numberOfTrainingItems)
                _labelsingle, img = self._dataset.getTrainingItem(idx)                
                img = img > 0       # Convert the MNIST Digit to binary
                img = reshape(img, (28,28))
                placed = self._tryPlaceImage(img, _labelsingle)
                
            if placed:
                self._labelvector.append(_labelsingle.argmax())                
        if not placed:
            raise Exception("Could not place all digits")
        return self._canvas, self._labelvector


    def _tryPlaceImage(self, img, label):
        """ Attempts to place the image randomly, otherwise returns false """
        img = img.astype('uint8')
        size = (self._dataset.itemDimension[0]) / 2
        # Pick a random location within the image to place the image
        x = randint(size, self._canvasSize[0] - size)
        y = randint(size, self._canvasSize[1] - size)
        #print "Attempting to place at (%d,%d)" % (x,y)     
        #xRange = arange(x - size, x + size)
        #yRange = arange(y - size, y + size)

        # Check for a collision with the existing canvas by exploiting the fact
        # that all images are binary at this stage        
        existing = self._canvas[x-size:x+size,y-size:y+size].astype('uint8')                
        overlap = (existing * img)
        if any(overlap):
            # Cannot place here. Return
            return False
        # Place the image in this location, and update the label accordingly
        self._canvas[x-size:x+size,y-size:y+size] = bitwise_or(existing, img)        
        labelUpdate = (label.argmax() + 1) * img
        self._label[x-size:x+size,y-size:y+size] += labelUpdate
        return True
        
        
    def save(self, Filename):
        
        data_dict = { 'Canvas':self._canvas, 'Label':self._label }
                    
        with open(Filename, 'wb') as f:
            pickle.dump(data_dict , f)
    
    def load(self, Filename):
        
        with open(Filename, 'rb') as f:
            data_dict = pickle.load(f)

        self._canvas = data_dict['Canvas']
        self._label = data_dict['Label']

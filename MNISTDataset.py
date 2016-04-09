from AbstractDataset import AbstractDataset
from os.path import exists, join
from struct import unpack
from numpy import zeros

class MNISTDataset(AbstractDataset):
    """ The dataset object providing access to the MNIST dataset """
    _basepath = None                # The path pointing to the MNIST dataset files
    _trainDataFilePath = None
    _trainLabelFilePath = None
    _testDataFilePath = None
    _testLabelFilePath = None
    _NumTesting = 0
    _NumTraining = 0

# ---------------------------------------------------------------------------------
# Constructors and destructors
# ---------------------------------------------------------------------------------

    def __init__(self, path):
        """ Initialises the MNIST dataset using the folder specified by path. """
        super(AbstractDataset, self).__init__();            # Call the base constructor
       
        # Generate the paths to the specified MNIST data files
        self._basepath = path
        self._trainDataFilePath = join(self._basepath, "train-images-idx3-ubyte")
        self._trainLabelFilePath = join(self._basepath, "train-labels-idx1-ubyte")
        self._testLabelFilePath = join(self._basepath, "t10k-labels-idx1-ubyte")
        self._testDataFilePath =  join(self._basepath, "t10k-images-idx3-ubyte")

        # Validate the path and check that the required files exist 
        if not exists(self._basepath):
            raise IOError("Path to MNIST data files not found. Given path was: %s" % (self._basepath))
        if not exists(self._trainDataFilePath) or not exists(self._trainLabelFilePath):
            raise IOError("MNIST training files not found on base path %s" % (self._basepath))
        if not exists(self._testDataFilePath) or not exists(self._testLabelFilePath):
            raise IOError("MNIST testing files not found on base path %s" % (self._basepath))

        # Open the data files for quick access
        self.trainDataFile = open(self._trainDataFilePath, "rb")
        self.trainLabelFile = open(self._trainLabelFilePath, "rb")
        self.testDataFile = open(self._testDataFilePath, "rb")
        self.testLabelFile = open(self._testLabelFilePath, "rb")

        # Check that the magic numbers for the specified files are correct
        if (self.__ExtractMagicByte(self.trainDataFile) != 2051):
            raise Exception("Magic number in Training Data file is incorrect ( was %d expected %d)" % (self.__ExtractMagicByte(self.trainDataFile), 2051))
        if (self.__ExtractMagicByte(self.trainLabelFile) != 2049):
            raise Exception("Magic number in Training Data file is incorrect ( was %d expected %d)" % (self.__ExtractMagicByte(self.trainLabelFile), 2049))
        if (self.__ExtractMagicByte(self.testDataFile) != 2051):
            raise Exception("Magic number in Training Data file is incorrect ( was %d expected %d)" % (self.__ExtractMagicByte(self.testDataFile), 2051))
        if (self.__ExtractMagicByte(self.testLabelFile) != 2049):
            raise Exception("Magic number in Training Data file is incorrect ( was %d expected %d)" % (self.__ExtractMagicByte(self.testLabelFile), 2049))
        
        # Extract the total counts from each file
        self._NumTraining = self.__GetNumberOfItemsInFile(self.trainDataFile)
        self._NumTesting = self.__GetNumberOfItemsInFile(self.testDataFile)


    def __del__(self):
        """ Destructor closes any open dataset files """
        self.trainDataFile.close()
        self.trainLabelFile.close()
        self.testDataFile.close()
        self.testLabelFile.close()
   

# ---------------------------------------------------------------------------------
# Properties    
# ---------------------------------------------------------------------------------
    @property
    def numberOfTrainingItems(self):
        """ Returns the number of training items in the dataset """
        return self._NumTraining

    @property
    def numberOfTestingItems(self):
        """ Returns the number of testing items in the dataset """
        return self._NumTesting

    @property
    def itemDimension(self):
        """ Returns a tuple containing the dimensions of each training/testing item in the dataset """
        return ([28,28])

    @property
    def labelDimension(self):
        """ Returns a tuple containing the dimensions of the label for each training/testing item in the dataset """
        return ([10,1])

# ---------------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------------

    def getTrainingItem(self, i):
        """ Returns the ith training item. Throws an exception if i exceeds the available number of training samples """
        if ( i >= self.numberOfTrainingItems) or (i < 0):
            raise IndexError("Index of requested training image exceeds total images. Requested %d from %d" % (i, self.numberOfTrainingItems))
        # Retrieve and return the pixel and label
        pixels = self.__GetImage(self.trainDataFile, i)
        label = self.__GetLabel(self.trainLabelFile, i)
        return label, pixels

    def getTestingItem(self, i):
        """ Returns the ith testing item. Throws an exception if i exceeds the available number of testing samples """
        if ( i >= self.numberOfTestingItems) or (i < 0):
            raise IndexError("Index of requested training image exceeds total images. Requested %d from %d" % (i, self.numberOfTrainingItems))
        # Retrieve and return the pixel and label
        pixels = self.__GetImage(self.testDataFile, i)
        label = self.__GetLabel(self.testLabelFile, i)
        return label, pixels

# ---------------------------------------------------------------------------------
# Internal Functions
# ---------------------------------------------------------------------------------

    def __ExtractMagicByte(self, filePointer):
        """ Reads and returns the magic number from the beginning of the dataset file """
        filePointer.seek(0)       
        return unpack(">i", filePointer.read(4))[0]

    def __GetNumberOfItemsInFile(self, filePointer):
        """ Reads and returns the number of items (labels / images) in the specified file """
        filePointer.seek(4)
        return unpack(">i", filePointer.read(4))[0]

    def __GetImage(self, filePointer, index):
        """ Returns a tuple containing a numpy array for the specified image """
        if (index < 0):
            raise IndexError("Image index cannot be negative")
        # Attempt to seek to the appropriate spot in the file
        filePointer.seek(8 + (index * (28*28)))
        # Read in the rows and columnns
        rows = unpack(">i", filePointer.read(4))[0]
        cols = unpack(">i", filePointer.read(4))[0]
        # Read in the actual pixel values (unsigned bytes)
        pixels = zeros([28*28])
        for i in range(0, 28*28):
            pixels[i] = unpack("B", filePointer.read(1))[0]
        return pixels

    def __GetLabel(self, filePointer, index):
        """ Returns an integer containing the label size for the specified index """
        if (index < 0):
            raise IndexError("Label index cannot be negative")
        # Attempt to seek to the appropriate spot in the file
        filePointer.seek(8 + index)
        label = unpack("B", filePointer.read(1))[0]
        # Convert the label itself into a vector
        labelVector = zeros(10)
        labelVector[label] = 1
        return labelVector  



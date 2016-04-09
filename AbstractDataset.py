"""

The intention is that the dataset object will manage the splits between
training and testing internally.

For datasets where there is no defined split, the percentage reserved for
testing should be passed as a parameter and the returned count of training
and testing items (i.e. GetNumberOfTrainingItems and GetNumberOfTestingItems)
should return the results calculated by the split.

The randomisation of the ordering should also be controlled using the
parameters passed during initialization. Different randomisation
strategies (i.e. uniform / gaussian ) can be specified as well as
sequential ordering. Randomisation seeds should also be passed (and
stored) to ensure repeatability and consistency.

In the case of leave-one-out-crossvalidation testing (LOUCV), the order
in which the items are returned must remain consistent between each trial.
When operating in this mode, the testing set will always comprise of a
single item representing the item under test.  The LOUCV implementation
can be implemented using a seperate class that consumes a dataset object
and only makes use of the training component (internally tracking the item
to leave out).

This means that we need to have an iterativeTrainerRunner and a
crossvalidationTestRunner.
"""
# pylint: disable=C0103
class AbstractDataset(object):
    """An abstract dataset divided into a training and testing set"""

    def __init__(self):
        """ Initializes the dataset. The dataset object should be ready
            for use once this function completes
        """
        pass

    @property
    def itemDimension(self):
        """ Returns a tuple containing the dimensions of each training
            and testing item in the dataset
        """
        raise NotImplementedError()

    @property
    def labelDimension(self):
        """ Returns a tuple containing the dimensions of the label for
            each training/testing item in the dataset
        """
        raise NotImplementedError()

    @property
    def numberOfTrainingItems(self):
        """ Returns an integer containing the number of training
            samples in this dataset
        """
        raise NotImplementedError()

    @property
    def numberOfTestingItems(self):
        """ Returns an integer containing the number of testing samples in this dataset """
        raise NotImplementedError()

    def getTrainingItem(self, i):
        """ Returns the ith training item.  Throws an exception if i exceeds
            the available number of training samples """
        raise NotImplementedError()

    def getTestingItem(self, i):
        """ Returns the ith testing item.  Throws an exception if i exceeds
            the available number of testing samples """
        raise NotImplementedError()

    def saveStateToFile(self, path):
        """ Saves the state of the dataset to a file
            This should preserve the chosen randomised index orders
        """
        raise NotImplementedError

    def loadStateFromFile(self, path):
        """ Loads the state from the specified file """
        raise NotImplementedError
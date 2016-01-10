__author__ = 'Alok'


class AbstractDataSet:
    """
    An abstract base class that data grabbers can use to fetch data in chunks or blocks
    e.g:
        1.  A mongodb implementation of this class would fetch training/validation/testing data from a mongo collection
        2.  A local file reader implementation of this class would fetch data from a local file
    """
    def get(self, limit, *args, **kwargs):
        """
        Work-horse method for the AbstractDataSet base class
        Implementation of this method would returns a list of data for un-labelled data or a pair of lists containing
        Both input data and labels
        :param limit: Number of items to be fetched
        :param skip: Number of items to be skipped
        :return: Already mentioned
        """
        raise NotImplementedError

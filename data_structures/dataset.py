import os



from netflow.data_structures.util import parse_meta_file



class NetflowDataset(object):
    def __init__(self, filename):
        """
        Constructor for the generic dataset class. This class serves as a wrapepr
        for the other datasets and lists required functions/variables.
        """
        self.filename = filename 
        self.attributes = parse_meta_file(self.filename)
        # all datasets need to have a unique prefix 
        self.prefix = self.attributes['prefix'][0]

        # create a temporary directory based on unique prefix
        self.temp_directory = 'temp/datasets/{}'.format(self.prefix)
        if not os.path.exists(self.temp_directory):
            os.makedirs(self.temp_directory, exist_ok = True)
        
        # create a results directory based on unique prefix 
        self.results_directory = 'results/{}'.format(self.prefix)
        if not os.path.exists(self.results_directory):
            os.makedirs(self.results_directory, exist_ok = True)

        # create a figures directory based on unique prefix 
        self.figures_directory = '{}/figures'.format(self.results_directory)
        if not os.path.exists(self.figures_directory):
            os.makedirs(self.figures_directory, exist_ok = True)

    def read_processed_data(self):
        """
        Placeholder function that reads and returns data.
        """
        assert ('Not implemented for this dataset.')

    def create_train_test_splits(self, nsplits = 50):
        """
        Placeholder function that creates various data splits. 

        @param nsplits: the number of splits
        """
        assert ('Not implemented for this dataset.')
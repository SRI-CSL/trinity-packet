from netflow.context_pcap.fnn import ContextFNNBatchNormDropout
from netflow.context_pcap.cnn import ContextCNNBatchNormDropout
from netflow.context_pcap.dataset import ContextPCAPTorchDataset, HEADER_LENGTH
from netflow.context_pcap.transformer import ContextTransformer
from netflow.data_structures.experiment import NetflowExperiment 
from netflow.data_structures.network_model import NetworkModel



class ContextPCAPExperiment(NetflowExperiment):   
    def __init__(self, filename, split_index, gpu = 4):
        """
        Constructor for the experiment class. This class serves as a wrapper
        for various file locations and attributes associated with a PCAP 
        network experiment.

        @param filename: location of the meta file that contains relevant info
        @param split_index: which train/test split to read
        @param gpu: the gpu to run the experiment on (default = 4)
        """
        super(ContextPCAPExperiment, self).__init__(filename, split_index, gpu)

        # flag to include context in this experiment
        if 'header context' in self.attributes: self.context = True
        else: self.context = False

        # set the header length and payload length
        if self.context: self.header_length = HEADER_LENGTH
        else: self.header_length = 0

        self.payload_length = int(self.attributes['payload'][0])

        # temporary fail safe for change in input 
        assert (self.payload_length == 1500)

    def compile_model(self):
        """
        Compile the neural network model for this experiment. 
        """
        # do not compile the model if already compiled since that changes the layer names
        if hasattr(self, 'model'): return 

        # create a new model here to decrease load up time for 
        if self.network_architecture == 'ContextCNNBatchNormDropout':
            self.model = NetworkModel(
                self.header_length,
                self.payload_length, 
                self.output_shape, 
                ContextCNNBatchNormDropout, 
                ContextPCAPTorchDataset,
                gpu = self.gpu,
            )
        if self.network_architecture == 'ContextFNNBatchNormDropout':
            self.model = NetworkModel(
                self.header_length,
                self.payload_length,
                self.output_shape, 
                ContextFNNBatchNormDropout, 
                ContextPCAPTorchDataset,
                gpu = self.gpu,
            )
        if self.network_architecture == 'ContextTransformer':
            self.model = NetworkModel(
                self.header_length,
                self.payload_length,
                self.output_shape, 
                ContextTransformer, 
                ContextPCAPTorchDataset,
                gpu = self.gpu,
            )